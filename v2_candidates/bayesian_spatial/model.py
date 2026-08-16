"""Bayesian Hierarchical Spatial Seismicity-Rate Model — FINAL_v2.0 CANDIDATE.

Model structure:
  N_i ~ Poisson(T * λ_i)
  log(λ_i) ~ Normal(μ, τ²)   (hierarchical prior)
  μ ~ Normal(log(mean_rate), 1.0)   (weakly informative)
  τ ~ HalfCauchy(0, 1.0)   (weakly informative, regularizes between-cell variation)

The hierarchical structure partially pools information across cells:
  - High-activity cells are shrunk slightly toward the regional mean
  - Low-activity cells are pulled up toward the regional mean
  - The degree of pooling is controlled by τ (data-driven)

This provides calibrated uncertainty intervals that account for:
  - Aleatory uncertainty (Poisson counting)
  - Epistemic uncertainty (parameter estimation, between-cell variation)

Inference: Conjugate Gamma-Poisson with empirical-Bayes hyperparameters
(computationally efficient for live operation; avoids MCMC convergence issues).
The hierarchical shrinkage is implemented via an empirical-Bayes Gamma prior
with shape and rate estimated from the cross-cell rate distribution.

For each cell i:
  Posterior: λ_i | N_i, T ~ Gamma(α + N_i, β + T)
  where α, β are estimated from the empirical distribution of cell rates:
    α = μ_rate² / σ_rate²    (shape)
    β = μ_rate / σ_rate²     (rate)
    μ_rate = mean(cell_rates)
    σ_rate² = var(cell_rates)

This is equivalent to a Gamma-Poisson (negative binomial) hierarchical model
with empirical-Bayes hyperparameter estimation, which is the standard
approach for spatial rate smoothing (Clayton & Kaldor 1987; Robbins 1955).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger("v2.bayesian")

# === FROZEN v1.0 PARAMETERS (for comparison only — DO NOT MODIFY) ===
V1_MC = 4.13
V1_B = 0.808
V1_GRID_SIZE = 1.0
BBOX = (20.0, 28.0, 88.0, 96.0)
N_LAT = 8
N_LON = 8
N_CELLS = 64


@dataclass
class BayesianSpatialConfig:
    """Configuration for the Bayesian hierarchical spatial model."""
    model_name: str = "bayesian_hierarchical_spatial"
    model_version: str = "v2.0_CANDIDATE_BAYESIAN_SPATIAL"
    # Prior specifications (weakly informative)
    prior_type: str = "empirical_bayes"  # or "fixed"
    # Fixed prior parameters (used when prior_type="fixed" or for sensitivity)
    fixed_alpha: float = 1.0    # Gamma shape
    fixed_beta: float = 0.1     # Gamma rate
    # Grid
    cell_size_deg: float = 1.0
    # Mc (frozen from v1.0 for fair comparison)
    mc: float = 4.13
    b_value: float = 0.808
    # Number of posterior samples for predictive distributions
    n_posterior_samples: int = 1000
    random_seed: int = 42


@dataclass
class CellPosterior:
    """Posterior distribution for one cell."""
    cell_id: str
    i_lat: int
    i_lon: int
    lat_center: float
    lon_center: float
    n_observed: int
    exposure_years: float
    # Posterior parameters (Gamma)
    alpha: float   # shape
    beta: float    # rate
    # Posterior summary
    rate_mean: float          # E[λ] = α/β
    rate_median: float        # median of Gamma
    rate_lower: float         # 2.5th percentile
    rate_upper: float         # 97.5th percentile
    # Derived probability
    prob_mean: float          # E[P(≥1)] = 1 - E[exp(-λΔt)]
    prob_median: float
    prob_lower: float         # 2.5th percentile of P
    prob_upper: float         # 97.5th percentile of P


def fit_bayesian_hierarchical(
    events: list,
    threshold: float,
    catalog_start: datetime,
    forecast_origin: datetime,
    config: BayesianSpatialConfig,
) -> list[CellPosterior]:
    """Fit the Bayesian hierarchical model causally (only events before origin).

    Parameters
    ----------
    events : list of CanonicalEvent
    threshold : magnitude threshold
    catalog_start : datetime (for exposure calculation)
    forecast_origin : datetime (only events before this are used)
    config : BayesianSpatialConfig
    """
    rng = np.random.default_rng(config.random_seed)

    # Filter to events before forecast origin and above threshold
    history = [e for e in events if e.origin_time_utc < forecast_origin]
    above = [e for e in history
             if (e.mw if e.mw is not None else e.original_magnitude) >= threshold]

    exposure_years = max(
        (forecast_origin - catalog_start).total_seconds() / (365.25 * 86400), 1e-6
    )

    # Count events per cell
    counts = np.zeros(N_CELLS, dtype=int)
    for e in above:
        i_lat = min(int((e.latitude - BBOX[0]) / config.cell_size_deg), N_LAT - 1)
        i_lon = min(int((e.longitude - BBOX[2]) / config.cell_size_deg), N_LON - 1)
        i_lat = max(i_lat, 0)
        i_lon = max(i_lon, 0)
        counts[i_lat * N_LON + i_lon] += 1

    # Cell rates (MLE)
    rates_mle = counts / exposure_years

    # Estimate hyperparameters (empirical Bayes)
    if config.prior_type == "empirical_bayes":
        # Only use cells with at least some activity for hyperparameter estimation
        active_rates = rates_mle[rates_mle > 0]
        if len(active_rates) >= 5:
            mu_rate = float(np.mean(active_rates))
            var_rate = float(np.var(active_rates, ddof=1))
            if var_rate > 0 and mu_rate > 0:
                alpha_prior = mu_rate**2 / var_rate
                beta_prior = mu_rate / var_rate
            else:
                alpha_prior = config.fixed_alpha
                beta_prior = config.fixed_beta
        else:
            # Too few active cells; use weakly informative fixed prior
            alpha_prior = config.fixed_alpha
            beta_prior = config.fixed_beta
    else:
        alpha_prior = config.fixed_alpha
        beta_prior = config.fixed_beta

    # Compute posterior for each cell
    # Posterior: λ_i | N_i, T ~ Gamma(α + N_i, β + T)
    cells = []
    lats_grid = [BBOX[0] + (i + 0.5) * config.cell_size_deg for i in range(N_LAT)]
    lons_grid = [BBOX[2] + (j + 0.5) * config.cell_size_deg for j in range(N_LON)]

    for idx in range(N_CELLS):
        i_lat = idx // N_LON
        i_lon = idx % N_LON
        n = int(counts[idx])

        # Posterior parameters
        alpha_post = alpha_prior + n
        beta_post = beta_prior + exposure_years

        # Posterior samples for probability distribution
        samples = rng.gamma(shape=alpha_post, scale=1.0/beta_post,
                            size=config.n_posterior_samples)

        # Rate summary
        rate_mean = alpha_post / beta_post
        rate_median = float(np.median(samples))
        rate_lower = float(np.percentile(samples, 2.5))
        rate_upper = float(np.percentile(samples, 97.5))

        cells.append(CellPosterior(
            cell_id=f"cell_{i_lat:02d}_{i_lon:02d}",
            i_lat=i_lat, i_lon=i_lon,
            lat_center=lats_grid[i_lat], lon_center=lons_grid[i_lon],
            n_observed=n, exposure_years=exposure_years,
            alpha=alpha_post, beta=beta_post,
            rate_mean=rate_mean,
            rate_median=rate_median,
            rate_lower=rate_lower,
            rate_upper=rate_upper,
            prob_mean=0, prob_median=0, prob_lower=0, prob_upper=0,  # filled below
        ))

    return cells, alpha_prior, beta_prior, exposure_years


def compute_probabilities(
    cells: list[CellPosterior],
    horizon_years: float,
    config: BayesianSpatialConfig,
) -> None:
    """Compute posterior predictive probabilities for each cell.

    P(≥1 event in horizon) = 1 - exp(-λ * Δt)

    For the Bayesian model, we compute:
    - prob_mean: E[P] = E[1 - exp(-λΔt)] = 1 - E[exp(-λΔt)]
      For Gamma(α,β): E[exp(-λΔt)] = (β/(β+Δt))^α
      So prob_mean = 1 - (β/(β+Δt))^α
    - prob_median: median of the P distribution (via sampling)
    - prob_lower/upper: 2.5th/97.5th percentiles (via sampling)
    """
    rng = np.random.default_rng(config.random_seed + 1)

    for cell in cells:
        # Analytic posterior predictive mean
        # P(≥1) = 1 - E[exp(-λΔt)] = 1 - (β/(β+Δt))^α
        ratio = cell.beta / (cell.beta + horizon_years)
        cell.prob_mean = 1.0 - ratio ** cell.alpha

        # Sample-based quantiles
        samples = rng.gamma(shape=cell.alpha, scale=1.0/cell.beta,
                            size=config.n_posterior_samples)
        probs_samples = 1.0 - np.exp(-samples * horizon_years)
        cell.prob_median = float(np.median(probs_samples))
        cell.prob_lower = float(np.percentile(probs_samples, 2.5))
        cell.prob_upper = float(np.percentile(probs_samples, 97.5))

        # Ensure bounds [0, 1]
        cell.prob_mean = max(0.0, min(1.0, cell.prob_mean))
        cell.prob_median = max(0.0, min(1.0, cell.prob_median))
        cell.prob_lower = max(0.0, min(1.0, cell.prob_lower))
        cell.prob_upper = max(0.0, min(1.0, cell.prob_upper))


def generate_forecast(
    cells: list[CellPosterior],
    threshold: float,
    horizon: str,
    horizon_years: float,
    alpha_prior: float,
    beta_prior: float,
    config: BayesianSpatialConfig,
) -> dict:
    """Generate a complete forecast record."""
    # Regional summary
    total_rate_mean = sum(c.rate_mean for c in cells)
    total_rate_lower = sum(c.rate_lower for c in cells)
    total_rate_upper = sum(c.rate_upper for c in cells)
    p_regional = 1.0 - math.exp(-total_rate_mean * horizon_years)
    p_regional_lo = 1.0 - math.exp(-total_rate_upper * horizon_years)  # upper rate → lower P
    p_regional_hi = 1.0 - math.exp(-total_rate_lower * horizon_years)  # lower rate → upper P

    return {
        "model_version": config.model_version,
        "threshold": threshold,
        "horizon": horizon,
        "horizon_years": horizon_years,
        "prior_type": config.prior_type,
        "prior_alpha": round(alpha_prior, 6),
        "prior_beta": round(beta_prior, 6),
        "regional_rate_mean": round(total_rate_mean, 4),
        "regional_rate_lower": round(total_rate_lower, 4),
        "regional_rate_upper": round(total_rate_upper, 4),
        "regional_probability": round(p_regional, 6),
        "regional_probability_lower": round(max(p_regional_lo, 0), 6),
        "regional_probability_upper": round(min(p_regional_hi, 1), 6),
        "n_cells": len(cells),
        "cells": [
            {
                "cell_id": c.cell_id,
                "lat_center": c.lat_center,
                "lon_center": c.lon_center,
                "n_observed": c.n_observed,
                "rate_mean": round(c.rate_mean, 6),
                "rate_median": round(c.rate_median, 6),
                "rate_lower": round(c.rate_lower, 6),
                "rate_upper": round(c.rate_upper, 6),
                "prob_mean": round(c.prob_mean, 6),
                "prob_median": round(c.prob_median, 6),
                "prob_lower": round(c.prob_lower, 6),
                "prob_upper": round(c.prob_upper, 6),
                "alpha": round(c.alpha, 4),
                "beta": round(c.beta, 4),
            }
            for c in cells
        ],
    }


def evaluate_forecast(
    v2_forecast: dict,
    v1_forecast: dict,
    y_true: np.ndarray,
) -> dict:
    """Evaluate v2 vs v1 on the same observed outcomes.

    Parameters
    ----------
    v2_forecast : dict with cells containing prob_mean, prob_lower, prob_upper
    v1_forecast : dict with cells containing probability, probability_lower, probability_upper
    y_true : array of binary outcomes per cell
    """
    eps = 1e-12

    # Extract probabilities
    v2_probs = np.array([c["prob_mean"] for c in v2_forecast["cells"]])
    v2_lo = np.array([c["prob_lower"] for c in v2_forecast["cells"]])
    v2_hi = np.array([c["prob_upper"] for c in v2_forecast["cells"]])

    v1_probs = np.array([c["probability"] for c in v1_forecast["cells"]])
    v1_lo = np.array([c["probability_lower"] for c in v1_forecast["cells"]])
    v1_hi = np.array([c["probability_upper"] for c in v1_forecast["cells"]])

    # Brier scores
    brier_v2 = float(np.mean((v2_probs - y_true) ** 2))
    brier_v1 = float(np.mean((v1_probs - y_true) ** 2))

    # Log-likelihoods
    f_v2 = np.clip(v2_probs, eps, 1 - eps)
    f_v1 = np.clip(v1_probs, eps, 1 - eps)
    ll_v2 = float(np.mean(y_true * np.log(f_v2) + (1 - y_true) * np.log(1 - f_v2)))
    ll_v1 = float(np.mean(y_true * np.log(f_v1) + (1 - y_true) * np.log(1 - f_v1)))

    # ECE (7-bin reliability)
    bins = np.linspace(0, 1, 8)
    def compute_ece(probs):
        ece = 0.0
        for i in range(len(bins) - 1):
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
            if mask.sum() > 0:
                mean_pred = float(probs[mask].mean())
                obs_freq = float(y_true[mask].mean())
                ece += abs(mean_pred - obs_freq) * mask.sum() / len(probs)
        return ece

    ece_v2 = compute_ece(v2_probs)
    ece_v1 = compute_ece(v1_probs)

    # Sharpness
    sharp_v2 = float(np.std(v2_probs))
    sharp_v1 = float(np.std(v1_probs))

    # Uncertainty interval coverage
    # For cells where events occurred (y=1), check if P_lower ≤ 1 ≤ P_upper
    # For cells where no event (y=0), check if 0 is within [P_lower, P_upper]
    # More meaningful: check if the observed rate falls within the rate CI
    # For probability: check if observed binary outcome is "consistent" with the CI
    # Standard approach: check if y_true is within [prob_lower, prob_upper]
    coverage_v2 = float(np.mean((y_true >= v2_lo) & (y_true <= v2_hi)))
    coverage_v1 = float(np.mean((y_true >= v1_lo) & (y_true <= v1_hi)))

    # Mean interval width
    width_v2 = float(np.mean(v2_hi - v2_lo))
    width_v1 = float(np.mean(v1_hi - v1_lo))

    # Reliability bins
    def reliability_bins(probs):
        result = []
        for i in range(len(bins) - 1):
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
            if mask.sum() > 0:
                result.append({
                    "bin": f"{bins[i]:.2f}-{bins[i+1]:.2f}",
                    "n": int(mask.sum()),
                    "mean_pred": round(float(probs[mask].mean()), 4),
                    "obs_freq": round(float(y_true[mask].mean()), 4),
                })
            else:
                result.append({"bin": f"{bins[i]:.2f}-{bins[i+1]:.2f}", "n": 0,
                               "mean_pred": None, "obs_freq": None})
        return result

    return {
        "brier_v2": round(brier_v2, 6),
        "brier_v1": round(brier_v1, 6),
        "delta_brier": round(brier_v1 - brier_v2, 6),  # positive = v2 better
        "log_lik_v2": round(ll_v2, 6),
        "log_lik_v1": round(ll_v1, 6),
        "delta_log_lik": round(ll_v2 - ll_v1, 6),
        "ece_v2": round(ece_v2, 6),
        "ece_v1": round(ece_v1, 6),
        "delta_ece": round(ece_v1 - ece_v2, 6),  # positive = v2 better
        "sharpness_v2": round(sharp_v2, 6),
        "sharpness_v1": round(sharp_v1, 6),
        "coverage_v2": round(coverage_v2, 4),
        "coverage_v1": round(coverage_v1, 4),
        "interval_width_v2": round(width_v2, 6),
        "interval_width_v1": round(width_v1, 6),
        "n_positive": int(y_true.sum()),
        "n_cells": len(y_true),
        "reliability_v2": reliability_bins(v2_probs),
        "reliability_v1": reliability_bins(v1_probs),
    }


def block_bootstrap_delta(
    v2_probs_per_origin: list,
    v1_probs_per_origin: list,
    y_true_per_origin: list,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> dict:
    """Block bootstrap CI for ΔBrier (v1 - v2) and Δlog-lik (v2 - v1)."""
    rng = np.random.default_rng(seed)
    n = len(v2_probs_per_origin)
    if n == 0:
        return {}

    deltas_brier = []
    deltas_ll = []
    eps = 1e-12
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        v2_all = np.concatenate([v2_probs_per_origin[i] for i in idx])
        v1_all = np.concatenate([v1_probs_per_origin[i] for i in idx])
        yt_all = np.concatenate([y_true_per_origin[i] for i in idx])

        b_v2 = np.mean((v2_all - yt_all) ** 2)
        b_v1 = np.mean((v1_all - yt_all) ** 2)
        deltas_brier.append(b_v1 - b_v2)

        f_v2 = np.clip(v2_all, eps, 1 - eps)
        f_v1 = np.clip(v1_all, eps, 1 - eps)
        ll_v2 = np.mean(yt_all * np.log(f_v2) + (1 - yt_all) * np.log(1 - f_v2))
        ll_v1 = np.mean(yt_all * np.log(f_v1) + (1 - yt_all) * np.log(1 - f_v1))
        deltas_ll.append(ll_v2 - ll_v1)

    return {
        "delta_brier_mean": round(float(np.mean(deltas_brier)), 6),
        "delta_brier_ci": [round(float(np.percentile(deltas_brier, 2.5)), 6),
                           round(float(np.percentile(deltas_brier, 97.5)), 6)],
        "delta_log_lik_mean": round(float(np.mean(deltas_ll)), 6),
        "delta_log_lik_ci": [round(float(np.percentile(deltas_ll, 2.5)), 6),
                             round(float(np.percentile(deltas_ll, 97.5)), 6)],
        "n_bootstrap": n_bootstrap,
        "n_origins": n,
    }


def posterior_predictive_check(
    cells: list[CellPosterior],
    observed_counts: np.ndarray,
    exposure_years: float,
    config: BayesianSpatialConfig,
) -> dict:
    """Posterior predictive check: can the model reproduce observed catalog statistics?"""
    rng = np.random.default_rng(config.random_seed + 100)

    # Simulate from posterior predictive
    n_sims = 1000
    sim_counts = np.zeros((n_sims, N_CELLS), dtype=int)
    for s in range(n_sims):
        for idx, cell in enumerate(cells):
            rate_sample = rng.gamma(shape=cell.alpha, scale=1.0/cell.beta)
            sim_counts[s, idx] = rng.poisson(rate_sample * exposure_years)

    # Compare statistics
    obs_total = int(observed_counts.sum())
    sim_totals = sim_counts.sum(axis=1)

    obs_occupied = int(np.sum(observed_counts > 0))
    sim_occupied = np.sum(sim_counts > 0, axis=1)

    obs_max = int(observed_counts.max())
    sim_max = sim_counts.max(axis=1)

    obs_gini = _gini(observed_counts.astype(float))
    sim_ginis = np.array([_gini(sim_counts[s].astype(float)) for s in range(n_sims)])

    return {
        "observed_total": obs_total,
        "sim_total_mean": round(float(np.mean(sim_totals)), 1),
        "sim_total_ci": [int(np.percentile(sim_totals, 2.5)), int(np.percentile(sim_totals, 97.5))],
        "observed_occupied_cells": obs_occupied,
        "sim_occupied_mean": round(float(np.mean(sim_occupied)), 1),
        "sim_occupied_ci": [int(np.percentile(sim_occupied, 2.5)), int(np.percentile(sim_occupied, 97.5))],
        "observed_max_count": obs_max,
        "sim_max_mean": round(float(np.mean(sim_max)), 1),
        "sim_max_ci": [int(np.percentile(sim_max, 2.5)), int(np.percentile(sim_max, 97.5))],
        "observed_gini": round(obs_gini, 4),
        "sim_gini_mean": round(float(np.mean(sim_ginis)), 4),
        "sim_gini_ci": [round(float(np.percentile(sim_ginis, 2.5)), 4),
                        round(float(np.percentile(sim_ginis, 97.5)), 4)],
    }


def _gini(x: np.ndarray) -> float:
    """Gini coefficient."""
    x = x[x >= 0]
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    x_sorted = np.sort(x)
    n = len(x_sorted)
    cumsum = np.cumsum(x_sorted)
    return (2 * np.sum(np.arange(1, n + 1) * x_sorted) / (n * cumsum[-1])) - (n + 1) / n
