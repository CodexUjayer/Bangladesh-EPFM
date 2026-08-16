"""ETAS residual diagnostics.

After fitting ETAS, the transformed residual process (using the integrated
intensity) should be approximately a standard Poisson process if the model
is correctly specified. We check:

  1. Temporal residuals: inter-event times in transformed time should be
     ~Exponential(1).
  2. Spatial residuals: per-cell residual counts (observed - expected).
  3. Magnitude residuals: observed vs GR-predicted magnitude distribution.
  4. Remaining clustering: check if transformed times still show clustering.
  5. Residual rate by time: rolling rate in transformed time.

If strong residual clustering remains, we IDENTIFY WHERE it occurs (which
time periods, which spatial regions) rather than declaring success.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .model import ETASModel
from .omori import omori_integral_over_window


@dataclass
class ResidualDiagnostics:
    """ETAS residual diagnostics."""

    n_events: int
    # Temporal
    transformed_times: list
    transformed_inter_event_times: list
    mean_transformed_iet: float            # should be ~1 if model is correct
    ks_stat_temporal: float                # KS vs Exp(1)
    # Spatial
    spatial_residuals: list                # (cell_id, observed, expected, residual)
    spatial_chi2: float
    spatial_df: int
    # Magnitude
    magnitude_residuals: dict              # bin -> (observed, expected)
    # Rolling rate in transformed time
    rolling_rate: list
    # Diagnostics
    remaining_clustering: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_events": self.n_events,
            "mean_transformed_iet": self.mean_transformed_iet,
            "ks_stat_temporal": self.ks_stat_temporal,
            "spatial_chi2": self.spatial_chi2,
            "spatial_df": self.spatial_df,
            "remaining_clustering": self.remaining_clustering,
            "notes": "; ".join(self.notes),
            "spatial_residuals_summary": [
                {"cell_id": r[0], "observed": r[1], "expected": round(r[2], 2),
                 "residual": round(r[3], 2)}
                for r in self.spatial_residuals[:20]
            ],
        }


def compute_residuals(
    model: ETASModel,
    history_times_days,
    history_lats,
    history_lons,
    history_mags,
    grid_lats=None,
    grid_lons=None,
    cell_area_km2: float = 10000.0,
) -> ResidualDiagnostics:
    """Compute ETAS residual diagnostics.

    The transformed time is τ(t) = ∫_0^t Λ(s) ds where Λ is the TOTAL
    integrated intensity (over the region). Under a correct model, the
    event times in τ-space are a standard Poisson process.
    """
    times = np.asarray(history_times_days, dtype=float)
    lats = np.asarray(history_lats, dtype=float)
    lons = np.asarray(history_lons, dtype=float)
    mags = np.asarray(history_mags, dtype=float)
    n = len(times)
    if n < 10:
        return ResidualDiagnostics(
            n_events=n, transformed_times=[], transformed_inter_event_times=[],
            mean_transformed_iet=float("nan"), ks_stat_temporal=float("nan"),
            spatial_residuals=[], spatial_chi2=float("nan"), spatial_df=0,
            magnitude_residuals={}, rolling_rate=[],
            remaining_clustering=False, notes=["Insufficient events for diagnostics."],
        )

    p = model.params
    mu_per_day = p.mu_total_per_year / 365.25

    # Transformed time: cumulative integrated intensity at each event
    # τ(t_j) = ∫_0^{t_j} Λ(s) ds = μ·t_j + Σ_i K·exp(α(M_i-Mc))·∫_0^{t_j} g(s-t_i) ds
    #        = μ·t_j + Σ_{i<j} K·exp(α(M_i-Mc))·[1 - (c/(t_j-t_i+c))^{p-1}]
    transformed = np.zeros(n)
    for j in range(n):
        tj = times[j]
        tau_j = mu_per_day * tj
        if j > 0:
            # contributions from events i < j
            ti = times[:j]
            mi = mags[:j]
            delta = tj - ti
            delta = np.maximum(delta, 0.0)
            G = 1.0 - (p.c_days / (delta + p.c_days)) ** (p.p - 1.0)
            prod = p.K * np.exp(p.alpha * (mi - p.Mc))
            tau_j += float(np.sum(prod * G))
        transformed[j] = tau_j

    # Inter-event times in transformed time
    iet = np.diff(transformed)
    iet = np.maximum(iet, 1e-12)
    mean_iet = float(np.mean(iet))

    # KS statistic vs Exponential(1)
    # F_exp(x) = 1 - exp(-x); sort iet and compare
    iet_sorted = np.sort(iet)
    ecdf = np.arange(1, len(iet_sorted) + 1) / len(iet_sorted)
    cdf_exp = 1.0 - np.exp(-iet_sorted)
    ks_stat = float(np.max(np.abs(ecdf - cdf_exp)))

    # Spatial residuals: assign events to grid cells, compare observed vs expected
    spatial_residuals = []
    chi2 = 0.0
    df = 0
    if grid_lats is not None and grid_lons is not None:
        grid_lats = np.asarray(grid_lats)
        grid_lons = np.asarray(grid_lons)
        # Expected count per cell over the whole catalog period
        T_days = float(times[-1] - times[0])
        for i, la in enumerate(grid_lats):
            for j, lo in enumerate(grid_lons):
                # Background expected
                bg = model.background.at(la, lo)
                exp_bg = bg / 365.25 * T_days * cell_area_km2
                # Triggered expected (approx: use average productivity × cell area)
                # For simplicity, distribute the total triggered expected uniformly
                # weighted by the spatial kernel at the cell center from each event.
                exp_trig = 0.0
                if n > 0:
                    from .spatial_kernels import _equirect_km
                    r_km = np.array([_equirect_km(la, lo, rl, ro) for rl, ro in zip(lats, lons)])
                    s = p.sigma_km * np.exp(p.gamma * (mags - p.Mc))
                    s2 = s * s
                    f = (p.q - 1.0) / (np.pi * s2) * (1.0 + (r_km * r_km) / s2) ** (-(1.0 + p.q))
                    # Each event contributes prod * f * cell_area over its Omori window
                    G_total = 1.0  # approx: full Omori integral per event
                    prod = p.K * np.exp(p.alpha * (mags - p.Mc))
                    exp_trig = float(np.sum(prod * f * cell_area_km2 * G_total))
                exp_total = exp_bg + exp_trig
                # Observed
                # Find which events fall in this cell
                if i < len(grid_lats) - 1:
                    la_min, la_max = grid_lats[i], grid_lats[i + 1]
                else:
                    la_min, la_max = grid_lats[i], grid_lats[i] + 1.0
                if j < len(grid_lons) - 1:
                    lo_min, lo_max = grid_lons[j], grid_lons[j + 1]
                else:
                    lo_min, lo_max = grid_lons[j], grid_lons[j] + 1.0
                obs = int(np.sum((lats >= la_min) & (lats < la_max) &
                                 (lons >= lo_min) & (lons < lo_max)))
                resid = obs - exp_total
                if exp_total > 0:
                    chi2 += (obs - exp_total) ** 2 / max(exp_total, 1e-6)
                    df += 1
                spatial_residuals.append((f"cell_{i:02d}_{j:02d}", obs, exp_total, resid))

    # Magnitude residuals: observed vs GR-predicted
    b = model.fit_info.get("b_value", 1.0)
    mag_bins = np.arange(p.Mc, mags.max() + 0.2, 0.2)
    mag_res = {}
    for k in range(len(mag_bins) - 1):
        m_lo, m_hi = mag_bins[k], mag_bins[k + 1]
        obs = int(np.sum((mags >= m_lo) & (mags < m_hi)))
        # GR predicted count in [m_lo, m_hi) given total N
        pred_frac = 10 ** (-b * m_lo) - 10 ** (-b * m_hi)
        exp = n * pred_frac
        mag_res[f"{m_lo:.1f}-{m_hi:.1f}"] = (obs, round(exp, 1))

    # Rolling rate in transformed time (window of ~10 events)
    window = min(10, n // 3) if n >= 30 else n
    rolling = []
    if window > 0:
        for k in range(0, n - window + 1, max(window // 2, 1)):
            chunk = iet[k:k + window]
            rolling.append(float(np.mean(chunk)))

    # Remaining clustering: if rolling rate varies by > 2x, flag
    remaining = False
    if len(rolling) > 2:
        if max(rolling) > 2.0 * min(rolling) + 1e-6:
            remaining = True

    notes = []
    if mean_iet < 0.8 or mean_iet > 1.2:
        notes.append(
            f"Mean transformed IET = {mean_iet:.3f} (expected ~1 for correct model); "
            "the model may be mis-specified."
        )
    if ks_stat > 0.2:
        notes.append(
            f"KS statistic vs Exp(1) = {ks_stat:.3f} (>0.2); temporal residuals "
            "deviate from the Poisson assumption. Remaining temporal clustering likely."
        )
    if remaining:
        notes.append(
            "Rolling rate in transformed time varies by >2x; remaining non-Poisson "
            "structure. Identify where (see spatial residuals)."
        )
    if not notes:
        notes.append("Residuals broadly consistent with the model specification.")

    return ResidualDiagnostics(
        n_events=n,
        transformed_times=transformed.tolist(),
        transformed_inter_event_times=iet.tolist(),
        mean_transformed_iet=mean_iet,
        ks_stat_temporal=ks_stat,
        spatial_residuals=spatial_residuals,
        spatial_chi2=float(chi2),
        spatial_df=df,
        magnitude_residuals=mag_res,
        rolling_rate=rolling,
        remaining_clustering=remaining,
        notes=notes,
    )
