"""Stage 7B: Causal spatial-Poisson baseline + direct ML-vs-Spatial-Poisson comparison.

THE SCIENTIFICALLY DECISIVE TEST:
  "Does ML add predictive information beyond the historical spatial seismicity-
  rate model?"

Stage 7 showed ML beats UNIFORM Poisson, but ML-A (historical rate) already
captured most of the improvement — strongly suggesting the gain is spatial
heterogeneity, which Spatial Poisson also captures. Stage 7B compares ML
directly against a CAUSALLY-RECONSTRUCTED Spatial Poisson baseline.

CAUSAL SPATIAL RATE: For a forecast at time t, the spatial Poisson rate per
cell is computed using ONLY events before t. Two estimators tested:
  A. Expanding historical rate: λ_cell(t) = N_cell(<t) / exposure(<t)
  B. Recent-window rate: λ_cell(t) = N_cell([t-W, t)) / W  for W in {1,3,5,10} yr

Smoothing: raw vs neighboring-cell smoothed (pre-specified, not tuned).

BASE-RATE CHECK: sum(cell P) ≈ regional P; documented mutual-exclusivity
assumption (events assigned to exactly one cell).

BLOCK BOOTSTRAP: resample forecast ORIGINS (not individual cell rows) to
respect temporal dependence. 95% CIs on ΔBrier and Δlog-likelihood.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from ..ingestion.schema import CanonicalEvent
from .features import MLGridConfig


# ---------------------------------------------------------------------------
# Causal spatial-Poisson estimators
# ---------------------------------------------------------------------------


@dataclass
class SpatialPoissonConfig:
    """Configuration for the causal spatial-Poisson baseline."""

    method: str = "expanding"     # "expanding" or "recent_window"
    window_years: float = 5.0     # for recent_window
    smoothing: str = "raw"        # "raw" or "neighbor_smoothed"


def causal_spatial_rate(
    events: list[CanonicalEvent],
    origin_time: datetime,
    grid: MLGridConfig,
    threshold: float,
    catalog_start: datetime,
    method: str = "expanding",
    window_years: float = 5.0,
    smoothing: str = "raw",
) -> np.ndarray:
    """Compute causal spatial-Poisson rate per cell at forecast origin.

    Returns array of shape (n_cells,) with rate per year per cell.

    STRICTLY CAUSAL: only events before origin_time are used.

    Parameters
    ----------
    method : "expanding" (all history) or "recent_window" (last W years)
    window_years : window length for recent_window
    smoothing : "raw" or "neighbor_smoothed" (average of 8 neighbors + self)
    """
    # Filter history
    history = [e for e in events if e.origin_time_utc < origin_time]
    # Filter by threshold
    hist_above = [e for e in history
                  if (e.mw if e.mw is not None else e.original_magnitude) >= threshold]

    # Exposure time
    if method == "expanding":
        exposure_years = max((origin_time - catalog_start).total_seconds() / (365.25 * 86400), 1e-6)
        window_start = catalog_start
    elif method == "recent_window":
        window_start = origin_time - timedelta(days=window_years * 365.25)
        exposure_years = window_years
        hist_above = [e for e in hist_above if e.origin_time_utc >= window_start]
    else:
        raise ValueError(f"Unknown method: {method}")

    # Count events per cell
    n_cells = grid.n_cells
    counts = np.zeros(n_cells)
    for e in hist_above:
        i_lat, i_lon = grid.cell_of(e.latitude, e.longitude)
        counts[i_lat * grid.n_lon + i_lon] += 1

    # Rate per cell
    rates = counts / exposure_years

    # Smoothing
    if smoothing == "neighbor_smoothed":
        rates_2d = rates.reshape(grid.n_lat, grid.n_lon)
        smoothed = np.zeros_like(rates_2d)
        for i in range(grid.n_lat):
            for j in range(grid.n_lon):
                vals = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < grid.n_lat and 0 <= nj < grid.n_lon:
                            vals.append(rates_2d[ni, nj])
                smoothed[i, j] = float(np.mean(vals))
        rates = smoothed.flatten()

    return rates


def spatial_poisson_forecast(
    rates_per_year: np.ndarray,
    horizon_years: float,
) -> np.ndarray:
    """Convert cell rates to P(N>=1) per cell.

    P_cell = 1 - exp(-λ_cell * Δt)

    This is the cell-level probability of at least one event in that cell
    during the horizon, assuming a Poisson process in each cell. Cells are
    treated as independent (mutual-exclusivity holds approximately because
    events are assigned to exactly one cell).
    """
    return 1.0 - np.exp(-rates_per_year * horizon_years)


# ---------------------------------------------------------------------------
# Base-rate check
# ---------------------------------------------------------------------------


def base_rate_check(
    cell_probs: np.ndarray,
    observed_regional_rate: float,
    horizon_years: float,
) -> dict:
    """Verify that sum(cell probabilities) ≈ expected regional probability.

    Phase A correction: the previous version compared sum(cell P) to a
    SINGLE-ORIGIN binary outcome (0 or 1), which produced ratio explosions
    when no event occurred. This corrected version compares to the
    observed_regional_rate, which should be the MEAN regional rate across
    origins (not a single binary).

    Under the Poisson-independence assumption, the regional P(N>=1) is:
      P_regional = 1 - prod(1 - P_cell)  (probability of >=1 event anywhere)

    For small P_cell, sum(P_cell) ≈ P_regional.

    Parameters
    ----------
    cell_probs : array of per-cell P(N>=1) for ONE forecast origin
    observed_regional_rate : MEAN fraction of origins with >=1 event anywhere
        (NOT a single-origin binary). Computed by the caller across all origins.
    """
    p_regional_indep = 1.0 - float(np.prod(1.0 - cell_probs))
    p_sum = float(np.sum(cell_probs))
    expected_count = p_sum  # expected total count ≈ sum λ Δt
    return {
        "sum_cell_probs": p_sum,
        "regional_p_independent": p_regional_indep,
        "observed_regional_rate_mean": observed_regional_rate,
        "expected_total_count": expected_count,
        # Ratio should be ~1 if cell probs are correctly calibrated
        "sum_vs_observed_ratio": p_sum / max(observed_regional_rate, 1e-6),
        # Pass if sum is within 50% of observed mean (generous for small samples)
        "passes": abs(p_sum - observed_regional_rate) < 0.5 * max(observed_regional_rate, 0.01),
    }


def aggregate_base_rate_check(per_origin_checks: list) -> dict:
    """Aggregate base-rate checks across all forecast origins.

    Phase A correction: the per-origin check is informative but the
    AGGREGATE check is the meaningful one. We compare the MEAN of
    sum(cell P) across origins to the MEAN observed regional rate.

    This replaces the previous incorrect comparison of sum(cell P) to a
    single-origin binary.
    """
    if not per_origin_checks:
        return {"passes": False, "notes": "No origin checks to aggregate."}
    sum_probs = [c["sum_cell_probs"] for c in per_origin_checks]
    obs_rates = [c.get("observed_regional_rate_mean", 0.0) for c in per_origin_checks]
    # The observed regional rate is the same for all origins (it's the mean)
    obs_mean = float(np.mean(obs_rates)) if obs_rates else 0.0
    pred_mean = float(np.mean(sum_probs))
    pred_std = float(np.std(sum_probs))
    # Also compute the actual observed regional rate from the per-origin binaries
    # (the caller should set observed_regional_binary per origin)
    return {
        "mean_sum_cell_probs": pred_mean,
        "std_sum_cell_probs": pred_std,
        "observed_regional_rate_mean": obs_mean,
        "ratio_pred_to_observed": pred_mean / max(obs_mean, 1e-6),
        # Pass if predicted mean is within 50% of observed mean
        "passes": abs(pred_mean - obs_mean) < 0.5 * max(obs_mean, 0.01),
        "n_origins": len(per_origin_checks),
    }


# ---------------------------------------------------------------------------
# Block bootstrap (over forecast origins)
# ---------------------------------------------------------------------------


def block_bootstrap_delta(
    ml_preds_per_origin: list[np.ndarray],
    sp_preds_per_origin: list[np.ndarray],
    y_true_per_origin: list[np.ndarray],
    n_bootstrap: int = 500,
    seed: int = 42,
) -> dict:
    """Block bootstrap over forecast origins for ΔBrier and Δlog-likelihood.

    Resamples ORIGINS (not individual cell rows) to preserve temporal
    dependence. Returns 95% CIs.

    ΔBrier = Brier_SP - Brier_ML  (positive = ML better)
    Δloglik = loglik_ML - loglik_SP  (positive = ML better)
    """
    rng = np.random.default_rng(seed)
    n_origins = len(ml_preds_per_origin)
    if n_origins == 0:
        return {"delta_brier_ci": (float("nan"), float("nan")),
                "delta_loglik_ci": (float("nan"), float("nan"))}

    eps = 1e-12
    delta_brier_boot = []
    delta_loglik_boot = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_origins, size=n_origins)
        ml_arrs = [ml_preds_per_origin[i] for i in idx]
        sp_arrs = [sp_preds_per_origin[i] for i in idx]
        yt_arrs = [y_true_per_origin[i] for i in idx]
        ml_all = np.concatenate(ml_arrs)
        sp_all = np.concatenate(sp_arrs)
        yt_all = np.concatenate(yt_arrs)
        # ΔBrier = Brier_SP - Brier_ML
        b_ml = np.mean((ml_all - yt_all) ** 2)
        b_sp = np.mean((sp_all - yt_all) ** 2)
        delta_brier_boot.append(b_sp - b_ml)
        # Δloglik = loglik_ML - loglik_SP
        f_ml = np.clip(ml_all, eps, 1 - eps)
        f_sp = np.clip(sp_all, eps, 1 - eps)
        ll_ml = np.mean(yt_all * np.log(f_ml) + (1 - yt_all) * np.log(1 - f_ml))
        ll_sp = np.mean(yt_all * np.log(f_sp) + (1 - yt_all) * np.log(1 - f_sp))
        delta_loglik_boot.append(ll_ml - ll_sp)

    return {
        "delta_brier_mean": float(np.mean(delta_brier_boot)),
        "delta_brier_ci": (float(np.percentile(delta_brier_boot, 2.5)),
                           float(np.percentile(delta_brier_boot, 97.5))),
        "delta_loglik_mean": float(np.mean(delta_loglik_boot)),
        "delta_loglik_ci": (float(np.percentile(delta_loglik_boot, 2.5)),
                            float(np.percentile(delta_loglik_boot, 97.5))),
        "n_bootstrap": n_bootstrap,
        "n_origins": n_origins,
    }
