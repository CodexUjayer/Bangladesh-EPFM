"""ETAS forecasting: temporal and spatial probability forecasts.

Temporal forecast:
  For a forecast horizon Δt starting at t0, the expected number of events
  above threshold M_f is:

    E[N] = ∫_{t0}^{t0+Δt} λ(t) dt

  where λ(t) = μ + Σ_{i: t_i < t} K·exp(α(M_i-Mc))·g(t-t_i)·∫f dx dy
              = μ + Σ_i K·exp(α(M_i-Mc))·g(t-t_i)   (since ∫f = 1)

  P(N >= 1) = 1 - exp(-E[N])  under the Poisson approximation.

Spatial forecast:
  For each grid cell c, E[N_c] = ∫_{cell} ∫_{t0}^{t0+Δt} λ(x,y,t) dt dx dy.
  We approximate by evaluating λ at the cell center and multiplying by the
  cell area and horizon. P(N_c >= 1) = 1 - exp(-E[N_c]).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..baselines.uncertainty import probability_ci_from_rate_ci
from .background import BackgroundRate
from .estimation import prepare_catalog
from .model import ETASModel, ETASParams, conditional_intensity
from .omori import omori_integral_over_window


@dataclass
class ETASForecast:
    """An ETAS forecast for one horizon and threshold."""

    forecast_start_days: float
    horizon_days: float
    threshold: float
    expected_total_count: float       # E[N] over the whole region
    probability_at_least_one: float
    per_cell: list = field(default_factory=list)   # list of dicts

    def to_rows(self) -> list[dict]:
        rows = []
        for c in self.per_cell:
            rows.append({
                "forecast_start_days": round(self.forecast_start_days, 3),
                "horizon_days": round(self.horizon_days, 4),
                "threshold": self.threshold,
                "cell_id": c["cell_id"],
                "lat_center": round(c["lat_center"], 3),
                "lon_center": round(c["lon_center"], 3),
                "expected_count": round(c["expected_count"], 6),
                "probability_at_least_one": round(c["probability_at_least_one"], 6),
                "rate_per_year": round(c["rate_per_year"], 6),
            })
        return rows


def forecast_temporal(
    model: ETASModel,
    history_times_days,
    history_lats,
    history_lons,
    history_mags,
    forecast_start_days: float,
    horizon_days: float,
    threshold: float,
) -> tuple[float, float]:
    """Temporal ETAS forecast: expected count and P(N>=1) over the region.

    Returns (expected_count, P(N>=1)).

    The integrated intensity over [t0, t0+Δt] is:
      μ·Δt + Σ_i K·exp(α(M_i-Mc)) · ∫_{t0}^{t0+Δt} g(t-t_i) dt
    (the spatial integral is 1 since f is normalized; this gives the total
    regional expected count, not per-cell).
    """
    p = model.params
    # Background contribution (per year -> per day)
    mu_per_day = p.mu_total_per_year / 365.25
    expected = mu_per_day * horizon_days

    # Triggered contribution
    history_times = np.asarray(history_times_days, dtype=float)
    history_mags = np.asarray(history_mags, dtype=float)
    mask = (history_times < forecast_start_days) & (history_mags >= p.Mc - 0.05)
    if mask.any():
        ti = history_times[mask]
        mi = history_mags[mask]
        # For each past event, ∫_{t0}^{t0+Δt} g(t-t_i) dt = ∫_{t0-t_i}^{t0+Δt-t_i} g(τ) dτ
        for k in range(len(ti)):
            a = forecast_start_days - ti[k]
            b = forecast_start_days + horizon_days - ti[k]
            # Clamp a to >= 0 (g defined for τ >= 0)
            a_eff = max(a, 0.0)
            b_eff = max(b, 0.0)
            if b_eff <= a_eff:
                continue
            G = omori_integral_over_window(p.c_days, p.p, a_eff, b_eff)
            # BASE-10 productivity (Phase A correction)
            prod = p.K * math.pow(10.0, p.alpha * (mi[k] - p.Mc))
            expected += prod * G

    # Threshold adjustment: the above counts events above Mc. If threshold > Mc,
    # scale by the GR probability above threshold:
    # P(M >= threshold | M >= Mc) = 10^(-b (threshold - Mc))
    if threshold > p.Mc:
        # We don't have b stored on the model; use a default b from the catalog
        # (passed in via model.fit_info if available)
        b = model.fit_info.get("b_value", 1.0)
        scale = 10.0 ** (-b * (threshold - p.Mc))
        expected = expected * scale

    p_ge1 = 1.0 - math.exp(-expected)
    return float(expected), float(p_ge1)


def forecast_spatial(
    model: ETASModel,
    history_times_days,
    history_lats,
    history_lons,
    history_mags,
    forecast_start_days: float,
    horizon_days: float,
    threshold: float,
    grid_lats,
    grid_lons,
    cell_area_km2: float,
) -> ETASForecast:
    """Spatial ETAS forecast on a grid.

    For each cell, evaluates the conditional intensity at the cell center
    and multiplies by cell area and horizon to get the expected count, then
    P(N>=1) = 1 - exp(-E[N_cell]).
    """
    # Total expected count (for the regional P)
    expected_total, p_total = forecast_temporal(
        model, history_times_days, history_lats, history_lons, history_mags,
        forecast_start_days, horizon_days, threshold,
    )

    per_cell = []
    # For efficiency, compute the triggered contribution at each cell center
    # using a time-cutoff subset of the history.
    p = model.params
    history_times = np.asarray(history_times_days, dtype=float)
    history_lats = np.asarray(history_lats, dtype=float)
    history_lons = np.asarray(history_lons, dtype=float)
    history_mags = np.asarray(history_mags, dtype=float)
    tau_max = max(10.0 * p.c_days / max(p.p - 1.0, 0.01), 365.0)
    recent_mask = (history_times < forecast_start_days) & \
                  (history_times >= forecast_start_days - tau_max) & \
                  (history_mags >= p.Mc - 0.05)
    r_times = history_times[recent_mask] if recent_mask.any() else np.array([])
    r_lats = history_lats[recent_mask] if recent_mask.any() else np.array([])
    r_lons = history_lons[recent_mask] if recent_mask.any() else np.array([])
    r_mags = history_mags[recent_mask] if recent_mask.any() else np.array([])

    # Threshold scaling for non-Mc thresholds
    if threshold > p.Mc:
        b = model.fit_info.get("b_value", 1.0)
        scale = 10.0 ** (-b * (threshold - p.Mc))
    else:
        scale = 1.0

    for i, la in enumerate(grid_lats):
        for j, lo in enumerate(grid_lons):
            # Background rate density at cell center (per year per km²)
            bg = model.background.at(la, lo)
            bg_per_day_per_km2 = bg / 365.25
            expected_cell = bg_per_day_per_km2 * cell_area_km2 * horizon_days * scale

            # Triggered
            if len(r_times) > 0:
                tau = forecast_start_days - r_times
                tau = np.maximum(tau, 0.0)
                g = (p.p - 1.0) * (p.c_days ** (p.p - 1.0)) / (tau + p.c_days) ** p.p
                # BASE-10 productivity (Phase A correction)
                prod = p.K * np.power(10.0, p.alpha * (r_mags - p.Mc))
                # integrate g over [t0, t0+Δt] for each past event
                # (approx: g evaluated at the midpoint tau_mid = t0 - t_i + Δt/2)
                # Better: use the closed-form integral
                a_arr = np.maximum(forecast_start_days - r_times, 0.0)
                b_arr = np.maximum(forecast_start_days + horizon_days - r_times, 0.0)
                G = np.where(
                    b_arr > a_arr,
                    (p.c_days ** (p.p - 1.0)) * (a_arr ** (-(p.p - 1.0)) - b_arr ** (-(p.p - 1.0))),
                    0.0,
                )
                # spatial kernel at cell center
                from .spatial_kernels import _equirect_km, magnitude_scaled_length
                r_km = np.array([_equirect_km(la, lo, rl, ro) for rl, ro in zip(r_lats, r_lons)])
                s = p.sigma_km * np.exp(p.gamma * (r_mags - p.Mc))
                s2 = s * s
                f = (p.q - 1.0) / (np.pi * s2) * (1.0 + (r_km * r_km) / s2) ** (-(1.0 + p.q))
                # expected triggered count in this cell = Σ prod · G · f · area
                expected_cell += float(np.sum(prod * G * f * cell_area_km2)) * scale

            p_cell = 1.0 - math.exp(-max(expected_cell, 0.0))
            per_cell.append({
                "cell_id": f"cell_{i:02d}_{j:02d}",
                "lat_center": la,
                "lon_center": lo,
                "expected_count": float(expected_cell),
                "probability_at_least_one": float(p_cell),
                "rate_per_year": float(expected_cell / horizon_days * 365.25) if horizon_days > 0 else 0.0,
            })

    return ETASForecast(
        forecast_start_days=forecast_start_days,
        horizon_days=horizon_days,
        threshold=threshold,
        expected_total_count=float(expected_total),
        probability_at_least_one=float(p_total),
        per_cell=per_cell,
    )
