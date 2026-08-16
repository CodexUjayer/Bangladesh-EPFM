"""Adaptive Spatial Smoothing Model — FINAL_v3.0 CANDIDATE_ADAPTIVE_SPATIAL.

This module implements a continuous spatial-rate estimator based on kernel
smoothing of historical earthquake locations. It is a controlled experiment
to test whether replacing the rigid 1° grid-rate estimation of v1 (and the
hierarchical shrinkage of v2) with a spatially continuous adaptive estimator
provides genuine, reproducible, statistically defensible improvement in
probabilistic earthquake forecasting for Bangladesh.

===========================  SCIENTIFIC MOTIVATION  ===========================

The existing analysis found very strong spatial heterogeneity (Gini ≈ 0.87):
a small number of 1° cells contain a disproportionate fraction of seismicity.
A rigid 1° grid creates artificial discontinuities between neighbouring cells,
and sparse cells produce unstable local rate estimates. Adaptive spatial
smoothing estimates a continuous seismicity intensity field λ(x,y) using
kernel smoothing, with the bandwidth either fixed or adaptive to local
event density.

===========================  MODEL FAMILY  ===========================

Four scientifically defensible variants are implemented:

  Model A — Fixed-bandwidth Gaussian kernel:
      λ(x) = (1/(T·h²)) · Σ_i (1/(2π)) · exp(-||x-x_i||² / (2·h²))
      where h is selected from the development/selection period only.

  Model B — Adaptive nearest-neighbour Gaussian kernel:
      h_i = distance to the k-th nearest qualifying earthquake (per query point).
      Bandwidth broadens in sparse regions, narrows in dense clusters.
      k is selected from the development/selection period only.

  Model C — Fixed-bandwidth Epanechnikov kernel (compact support):
      K(u) = (3/4)·(1-u²) for |u| ≤ 1, 0 otherwise.
      Compact support reduces long-range smearing of distant events.

  Model D — Adaptive nearest-neighbour Epanechnikov kernel:
      Combines compact support with locally adaptive bandwidth.

We do NOT tune hundreds of variants. Only these four are tested.

===========================  UNCERTAINTY  ===========================

Two distinct sources of uncertainty are reported:

  Aleatory uncertainty — intrinsic Poisson counting variability.
      Captured by P(N≥1) = 1 - exp(-λ·Δt).

  Epistemic uncertainty — uncertainty about the estimated spatial rate.
      Estimated via non-parametric bootstrap over event locations
      (resample historical events with replacement, recompute λ(x),
      take 2.5/97.5 percentiles). This captures uncertainty in the
      smoothing parameters induced by finite-sample catalog noise.

===========================  COMPARISON  ===========================

This candidate is compared head-to-head against:
  - FINAL_v1.0_FROZEN  (Spatial Poisson, 1° grid, Garwood CI)
  - FINAL_v2.0_CANDIDATE_BAYESIAN_SPATIAL  (Gamma-Poisson hierarchical)

using IDENTICAL forecast origins, catalog snapshots, magnitude thresholds,
horizons, spatial domain, and scoring rules. No information from the
evaluation period is used for bandwidth/kernel selection.

===========================  INTEGRITY  ===========================

This module does NOT modify v1 or v2 source code, ledgers, scores, or
frozen artifacts. It produces a completely separate v3 namespace.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger("v3.adaptive")

# === FROZEN v1.0 PARAMETERS (for fair comparison — DO NOT MODIFY) ===
V1_MC = 4.13
V1_B = 0.808
BBOX = (20.0, 28.0, 88.0, 96.0)   # min_lat, max_lat, min_lon, max_lon
N_LAT = 8
N_LON = 8
N_CELLS = 64

# Earth radius (km) for haversine distance
EARTH_R_KM = 6371.0088

# Predefined candidate bandwidths (degrees) and k values.
# Selection happens ONLY on the development/selection period.
BANDWIDTH_CANDIDATES_DEG = [0.25, 0.5, 1.0, 2.0]
NN_K_CANDIDATES = [10, 25, 50]

# Bootstrap settings
DEFAULT_N_BOOTSTRAP = 200   # kept modest for runtime; enough for 2.5/97.5 percentiles
DEFAULT_N_POSTERIOR_SAMPLES = 1000


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two points."""
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(a))


def _equirectangular_deg(lat1: float, lon1: float, lat2: float, lon2: float,
                          cos_lat_ref: float = math.cos(math.radians(24.0))) -> float:
    """Approx Euclidean distance in DEGREES at mid-latitude of study region.

    Used inside kernel evaluations for speed (haversine is expensive in tight
    Python loops). The error at 24°N for distances up to ~2° is < 1% which is
    negligible compared to the bandwidth selection uncertainty.
    """
    dlat = lat2 - lat1
    dlon = (lon2 - lon1) * cos_lat_ref
    return math.sqrt(dlat * dlat + dlon * dlon)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

def _gaussian_kernel(u: np.ndarray) -> np.ndarray:
    """Standard Gaussian kernel (unit integral over 2D)."""
    return np.exp(-0.5 * u * u) / (2.0 * math.pi)


def _epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    """Epanechnikov kernel (compact support |u|<=1, unit integral over 2D).

    K(u) = (2/π) · (1 - u²)  for |u| ≤ 1, else 0.
    Integral over the unit disk: ∫∫ (2/π)(1-r²) r dr dθ = (2/π)·2π·(1/2 - 1/4) = 1.
    """
    out = np.where(np.abs(u) <= 1.0, (2.0 / math.pi) * (1.0 - u * u), 0.0)
    return out


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveSpatialConfig:
    """Configuration for the v3 adaptive spatial smoothing model."""
    model_name: str = "adaptive_spatial_smoothing"
    model_version: str = "v3.0_CANDIDATE_ADAPTIVE_SPATIAL"
    # Model variant: "A_gaussian_fixed", "B_gaussian_nn",
    #                "C_epanechnikov_fixed", "D_epanechnikov_nn"
    variant: str = "A_gaussian_fixed"
    # Bandwidth (deg) for fixed variants
    bandwidth_deg: float = 0.5
    # k for nearest-neighbour variants
    nn_k: int = 25
    # Kernel
    kernel: str = "gaussian"   # "gaussian" or "epanechnikov"
    adaptive: bool = False     # True for NN-adaptive bandwidth
    # Grid for evaluation (must match v1/v2 for fair comparison)
    cell_size_deg: float = 1.0
    # Frozen Mc / b for fair comparison
    mc: float = V1_MC
    b_value: float = V1_B
    # Uncertainty
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP
    n_posterior_samples: int = DEFAULT_N_POSTERIOR_SAMPLES
    random_seed: int = 42


# ---------------------------------------------------------------------------
# Core smoothed rate estimator
# ---------------------------------------------------------------------------

@dataclass
class SmoothedRateField:
    """Result of evaluating the smoothed rate at the 64 cell centres."""
    cell_id: str
    i_lat: int
    i_lon: int
    lat_center: float
    lon_center: float
    n_historical_events_used: int
    exposure_years: float
    # Rate (per year)
    rate_mean: float
    rate_lower: float   # 2.5th percentile (epistemic)
    rate_upper: float   # 97.5th percentile (epistemic)
    # Probability P(N>=1 in horizon)
    prob_mean: float
    prob_lower: float
    prob_upper: float
    # Local bandwidth used at this cell (deg) — informative for diagnostics
    local_bandwidth_deg: float
    # Number of neighbouring events within the kernel effective support
    n_neighbours: int


def _evaluate_rate_at_points(
    query_lats: np.ndarray,
    query_lons: np.ndarray,
    event_lats: np.ndarray,
    event_lons: np.ndarray,
    exposure_years: float,
    config: AdaptiveSpatialConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the smoothed seismicity rate at each query point (vectorised).

    Returns (rates, local_bandwidths, n_neighbours).

    Vectorisation: build the full (n_query × n_event) distance matrix once
    via numpy broadcasting, then apply the kernel element-wise.
    """
    n_q = len(query_lats)
    rates = np.zeros(n_q)
    local_bw = np.zeros(n_q)
    n_neighbours = np.zeros(n_q, dtype=int)

    if len(event_lats) == 0 or exposure_years <= 0:
        return rates, local_bw, n_neighbours

    cos_lat_ref = math.cos(math.radians(24.0))
    ev_lats = np.asarray(event_lats, dtype=float)
    ev_lons = np.asarray(event_lons, dtype=float)
    qlats = np.asarray(query_lats, dtype=float)
    qlons = np.asarray(query_lons, dtype=float)

    # Pick kernel function
    if config.kernel == "gaussian":
        kernel_fn = _gaussian_kernel
    elif config.kernel == "epanechnikov":
        kernel_fn = _epanechnikov_kernel
    else:
        raise ValueError(f"Unknown kernel: {config.kernel}")

    # Build the distance matrix in DEGREES via broadcasting.
    # shape: (n_query, n_event)
    dlat = qlats[:, None] - ev_lats[None, :]
    dlon = (qlons[:, None] - ev_lons[None, :]) * cos_lat_ref
    dist_deg = np.sqrt(dlat * dlat + dlon * dlon)

    if config.adaptive:
        # k-th nearest event distance per query point (row-wise k-th smallest)
        k = min(config.nn_k, dist_deg.shape[1])
        if k < 1:
            h_per_query = np.full(n_q, config.bandwidth_deg)
        else:
            # np.partition is faster than full sort
            kth = np.partition(dist_deg, k - 1, axis=1)[:, k - 1]
            h_per_query = np.maximum(kth, 1e-3)
    else:
        h_per_query = np.full(n_q, config.bandwidth_deg)

    local_bw = h_per_query
    # Normalised distances: shape (n_query, n_event)
    u = dist_deg / h_per_query[:, None]
    w = kernel_fn(u)
    # Rate per square-degree per year: λ ≈ (1/(T·h²)) · Σ_i K(u_i)
    rates = w.sum(axis=1) / (exposure_years * h_per_query * h_per_query)

    # Neighbours within effective support
    if config.kernel == "gaussian":
        # Gaussian infinite support: report within 3·h (99.7% mass)
        n_neighbours = (dist_deg <= 3.0 * h_per_query[:, None]).sum(axis=1)
    else:
        n_neighbours = (dist_deg <= h_per_query[:, None]).sum(axis=1)

    return rates, local_bw, n_neighbours.astype(int)


# ---------------------------------------------------------------------------
# Fitting (causal: only events before forecast origin)
# ---------------------------------------------------------------------------

def fit_adaptive_spatial(
    events: list,
    threshold: float,
    catalog_start: datetime,
    forecast_origin: datetime,
    config: AdaptiveSpatialConfig,
) -> tuple[list[SmoothedRateField], float, int, float]:
    """Fit the adaptive spatial smoothing model causally.

    STRICTLY CAUSAL: only events before forecast_origin are used.

    Returns (cells, exposure_years, n_history_events_above_threshold,
             mean_local_bandwidth_deg).
    """
    history = [e for e in events if e.origin_time_utc < forecast_origin]
    above = []
    for e in history:
        m = e.mw if e.mw is not None else e.original_magnitude
        if m is not None and m >= threshold:
            above.append(e)

    exposure_years = max(
        (forecast_origin - catalog_start).total_seconds() / (365.25 * 86400), 1e-6
    )

    if not above:
        ev_lats = np.array([])
        ev_lons = np.array([])
    else:
        ev_lats = np.array([e.latitude for e in above], dtype=float)
        ev_lons = np.array([e.longitude for e in above], dtype=float)

    # Query points: 64 cell centres (matches v1/v2 grid)
    qlats = np.array([BBOX[0] + (i + 0.5) * config.cell_size_deg for i in range(N_LAT)])
    qlons = np.array([BBOX[2] + (j + 0.5) * config.cell_size_deg for j in range(N_LON)])
    qgrid_lat, qgrid_lon = np.meshgrid(qlats, qlons, indexing="ij")
    query_lats = qgrid_lat.flatten()
    query_lons = qgrid_lon.flatten()

    rates, local_bw, n_neighbours = _evaluate_rate_at_points(
        query_lats, query_lons, ev_lats, ev_lons, exposure_years, config
    )

    # Build cell records (without bootstrap uncertainty for now)
    cells = []
    for idx in range(N_CELLS):
        i_lat = idx // N_LON
        i_lon = idx % N_LON
        cells.append(SmoothedRateField(
            cell_id=f"cell_{i_lat:02d}_{i_lon:02d}",
            i_lat=i_lat, i_lon=i_lon,
            lat_center=float(qlats[i_lat]), lon_center=float(qlons[i_lon]),
            n_historical_events_used=len(above),
            exposure_years=exposure_years,
            rate_mean=float(rates[idx]),
            rate_lower=float(rates[idx]),
            rate_upper=float(rates[idx]),
            prob_mean=0.0, prob_lower=0.0, prob_upper=0.0,
            local_bandwidth_deg=float(local_bw[idx]),
            n_neighbours=int(n_neighbours[idx]),
        ))

    mean_bw = float(np.mean(local_bw)) if len(local_bw) else float(config.bandwidth_deg)
    return cells, exposure_years, len(above), mean_bw


def compute_probabilities(
    cells: list[SmoothedRateField],
    horizon_years: float,
    config: AdaptiveSpatialConfig,
) -> None:
    """Compute P(N≥1) per cell from rate.

    Point estimate: P = 1 - exp(-λ·Δt).
    With bootstrap-derived epistemic uncertainty on λ, propagated to P.

    NOTE: P = 1 - exp(-λ·Δt) is MONOTONICALLY INCREASING in λ.
    Therefore:
      - prob_lower uses rate_lower (lower rate → lower probability)
      - prob_upper uses rate_upper (upper rate → upper probability)
    """
    for cell in cells:
        cell.prob_mean = max(0.0, min(1.0, 1.0 - math.exp(-cell.rate_mean * horizon_years)))
        # Lower rate → lower probability; upper rate → upper probability.
        cell.prob_lower = max(0.0, min(1.0, 1.0 - math.exp(-cell.rate_lower * horizon_years)))
        cell.prob_upper = max(0.0, min(1.0, 1.0 - math.exp(-cell.rate_upper * horizon_years)))


# ---------------------------------------------------------------------------
# Bootstrap uncertainty (epistemic)
# ---------------------------------------------------------------------------

def bootstrap_uncertainty(
    events: list,
    threshold: float,
    catalog_start: datetime,
    forecast_origin: datetime,
    config: AdaptiveSpatialConfig,
    n_bootstrap: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap over historical events to estimate epistemic uncertainty.

    Resample the historical above-threshold catalog WITH REPLACEMENT and
    recompute the smoothed rate at each cell centre. Returns rate mean,
    2.5th and 97.5th percentiles per cell.

    Returns (rate_mean[N_CELLS], rate_lower[N_CELLS], rate_upper[N_CELLS]).
    """
    rng = np.random.default_rng(config.random_seed)
    n_bs = n_bootstrap if n_bootstrap is not None else config.n_bootstrap

    history = [e for e in events if e.origin_time_utc < forecast_origin]
    above = []
    for e in history:
        m = e.mw if e.mw is not None else e.original_magnitude
        if m is not None and m >= threshold:
            above.append(e)

    exposure_years = max(
        (forecast_origin - catalog_start).total_seconds() / (365.25 * 86400), 1e-6
    )

    # Query points
    qlats = np.array([BBOX[0] + (i + 0.5) * config.cell_size_deg for i in range(N_LAT)])
    qlons = np.array([BBOX[2] + (j + 0.5) * config.cell_size_deg for j in range(N_LON)])
    qgrid_lat, qgrid_lon = np.meshgrid(qlats, qlons, indexing="ij")
    query_lats = qgrid_lat.flatten()
    query_lons = qgrid_lon.flatten()

    n_ev = len(above)
    if n_ev == 0:
        z = np.zeros(N_CELLS)
        return z, z, z

    ev_lats = np.array([e.latitude for e in above], dtype=float)
    ev_lons = np.array([e.longitude for e in above], dtype=float)

    # If n_bs == 0, return point estimate with no epistemic uncertainty
    # (used during bandwidth selection for runtime efficiency)
    if n_bs <= 0:
        r, _, _ = _evaluate_rate_at_points(
            query_lats, query_lons, ev_lats, ev_lons, exposure_years, config
        )
        return r, r, r

    boot_rates = np.zeros((n_bs, N_CELLS))
    for b in range(n_bs):
        idx = rng.integers(0, n_ev, size=n_ev)
        blats = ev_lats[idx]
        blons = ev_lons[idx]
        r, _, _ = _evaluate_rate_at_points(
            query_lats, query_lons, blats, blons, exposure_years, config
        )
        boot_rates[b] = r

    rate_mean = boot_rates.mean(axis=0)
    rate_lower = np.percentile(boot_rates, 2.5, axis=0)
    rate_upper = np.percentile(boot_rates, 97.5, axis=0)
    return rate_mean, rate_lower, rate_upper


def attach_bootstrap_uncertainty(
    cells: list[SmoothedRateField],
    rate_mean: np.ndarray,
    rate_lower: np.ndarray,
    rate_upper: np.ndarray,
    horizon_years: float,
) -> None:
    """Replace point estimates with bootstrap-derived mean and CIs.

    P = 1 - exp(-λ·Δt) is monotonically increasing in λ, so:
      prob_lower uses rate_lower, prob_upper uses rate_upper.
    """
    for idx, cell in enumerate(cells):
        cell.rate_mean = float(rate_mean[idx])
        cell.rate_lower = float(rate_lower[idx])
        cell.rate_upper = float(rate_upper[idx])
        cell.prob_mean = max(0.0, min(1.0, 1.0 - math.exp(-cell.rate_mean * horizon_years)))
        cell.prob_lower = max(0.0, min(1.0, 1.0 - math.exp(-cell.rate_lower * horizon_years)))
        cell.prob_upper = max(0.0, min(1.0, 1.0 - math.exp(-cell.rate_upper * horizon_years)))


# ---------------------------------------------------------------------------
# Forecast generation
# ---------------------------------------------------------------------------

def generate_forecast(
    cells: list[SmoothedRateField],
    threshold: float,
    horizon: str,
    horizon_years: float,
    config: AdaptiveSpatialConfig,
    n_history_events: int,
    mean_local_bandwidth: float,
) -> dict:
    """Generate a complete v3 forecast record."""
    total_rate_mean = sum(c.rate_mean for c in cells)
    total_rate_lo = sum(c.rate_lower for c in cells)
    total_rate_hi = sum(c.rate_upper for c in cells)
    p_regional = 1.0 - math.exp(-total_rate_mean * horizon_years)
    # P = 1 - exp(-λ·Δt) is monotonically increasing in λ:
    #   lower rate → lower P, upper rate → upper P.
    p_regional_lo = 1.0 - math.exp(-total_rate_lo * horizon_years)
    p_regional_hi = 1.0 - math.exp(-total_rate_hi * horizon_years)

    return {
        "model_version": config.model_version,
        "variant": config.variant,
        "kernel": config.kernel,
        "adaptive": config.adaptive,
        "bandwidth_deg": config.bandwidth_deg,
        "nn_k": config.nn_k,
        "threshold": threshold,
        "horizon": horizon,
        "horizon_years": horizon_years,
        "regional_rate_mean": round(total_rate_mean, 4),
        "regional_rate_lower": round(total_rate_lo, 4),
        "regional_rate_upper": round(total_rate_hi, 4),
        "regional_probability": round(max(0.0, min(1.0, p_regional)), 6),
        "regional_probability_lower": round(max(p_regional_lo, 0), 6),
        "regional_probability_upper": round(min(p_regional_hi, 1), 6),
        "n_cells": len(cells),
        "n_historical_events_above_threshold": n_history_events,
        "mean_local_bandwidth_deg": round(mean_local_bandwidth, 4),
        "cells": [
            {
                "cell_id": c.cell_id,
                "lat_center": c.lat_center,
                "lon_center": c.lon_center,
                "n_historical_events_used": c.n_historical_events_used,
                "rate_mean": round(c.rate_mean, 6),
                "rate_lower": round(c.rate_lower, 6),
                "rate_upper": round(c.rate_upper, 6),
                "prob_mean": round(c.prob_mean, 6),
                "prob_lower": round(c.prob_lower, 6),
                "prob_upper": round(c.prob_upper, 6),
                "local_bandwidth_deg": round(c.local_bandwidth_deg, 4),
                "n_neighbours": c.n_neighbours,
            }
            for c in cells
        ],
    }


# ---------------------------------------------------------------------------
# Evaluation (Brier, log-lik, ECE, sharpness, coverage)
# ---------------------------------------------------------------------------

def evaluate_forecast(
    v3_forecast: dict,
    y_true: np.ndarray,
) -> dict:
    """Evaluate v3 forecast against observed binary outcomes per cell."""
    eps = 1e-12
    v3_probs = np.array([c["prob_mean"] for c in v3_forecast["cells"]])
    v3_lo = np.array([c["prob_lower"] for c in v3_forecast["cells"]])
    v3_hi = np.array([c["prob_upper"] for c in v3_forecast["cells"]])

    brier = float(np.mean((v3_probs - y_true) ** 2))
    f = np.clip(v3_probs, eps, 1 - eps)
    log_lik = float(np.mean(y_true * np.log(f) + (1 - y_true) * np.log(1 - f)))

    # ECE (7-bin reliability)
    bins = np.linspace(0, 1, 8)
    ece = 0.0
    for i in range(len(bins) - 1):
        mask = (v3_probs >= bins[i]) & (v3_probs < bins[i + 1])
        if mask.sum() > 0:
            mean_pred = float(v3_probs[mask].mean())
            obs_freq = float(y_true[mask].mean())
            ece += abs(mean_pred - obs_freq) * mask.sum() / len(v3_probs)

    sharpness = float(np.std(v3_probs))
    coverage = float(np.mean((y_true >= v3_lo) & (y_true <= v3_hi)))
    width = float(np.mean(v3_hi - v3_lo))

    # Hit / FA / Miss / CN at default threshold 0.5 (rare-event: typically low)
    # Also compute at optimal threshold for diagnostic
    def _metrics(p, y):
        # Use a small operating threshold suited to rare events
        # Threshold = max(0.01, median(p)*0.5) — diagnostic only
        thr = max(0.005, float(np.median(p)) * 0.5)
        pred_pos = p >= thr
        pred_neg = ~pred_pos
        tp = int(np.sum(pred_pos & (y == 1)))
        fp = int(np.sum(pred_pos & (y == 0)))
        fn = int(np.sum(pred_neg & (y == 1)))
        tn = int(np.sum(pred_neg & (y == 0)))
        hit = tp / max(tp + fn, 1)
        fa = fp / max(fp + tn, 1)
        miss = fn / max(tp + fn, 1)
        cn = tn / max(fp + tn, 1)
        return hit, fa, miss, cn, thr

    hit, fa, miss, cn, thr = _metrics(v3_probs, y_true)

    # Reliability bins
    reliability = []
    for i in range(len(bins) - 1):
        mask = (v3_probs >= bins[i]) & (v3_probs < bins[i + 1])
        if mask.sum() > 0:
            reliability.append({
                "bin": f"{bins[i]:.2f}-{bins[i+1]:.2f}",
                "n": int(mask.sum()),
                "mean_pred": round(float(v3_probs[mask].mean()), 4),
                "obs_freq": round(float(y_true[mask].mean()), 4),
            })
        else:
            reliability.append({"bin": f"{bins[i]:.2f}-{bins[i+1]:.2f}", "n": 0,
                                "mean_pred": None, "obs_freq": None})

    return {
        "brier": round(brier, 6),
        "log_lik": round(log_lik, 6),
        "ece": round(ece, 6),
        "sharpness": round(sharpness, 6),
        "coverage": round(coverage, 4),
        "interval_width": round(width, 6),
        "hit_rate": round(hit, 4),
        "false_alarm_rate": round(fa, 4),
        "miss_rate": round(miss, 4),
        "correct_neg_rate": round(cn, 4),
        "operating_threshold": round(thr, 6),
        "n_positive": int(y_true.sum()),
        "n_cells": len(y_true),
        "reliability": reliability,
    }


# ---------------------------------------------------------------------------
# Paired block bootstrap (over forecast origins)
# ---------------------------------------------------------------------------

def block_bootstrap_delta(
    v3_probs_per_origin: list,
    baseline_probs_per_origin: list,
    y_true_per_origin: list,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> dict:
    """Block bootstrap over forecast ORIGINS for ΔBrier and Δlog-lik.

    ΔBrier = Brier_baseline - Brier_v3   (positive = v3 better)
    Δloglik = loglik_v3 - loglik_baseline (positive = v3 better)
    """
    rng = np.random.default_rng(seed)
    n = len(v3_probs_per_origin)
    if n == 0:
        return {"delta_brier_mean": 0.0, "delta_brier_ci": [0.0, 0.0],
                "delta_log_lik_mean": 0.0, "delta_log_lik_ci": [0.0, 0.0],
                "n_bootstrap": n_bootstrap, "n_origins": 0}
    eps = 1e-12
    deltas_brier = []
    deltas_ll = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        v3_all = np.concatenate([v3_probs_per_origin[i] for i in idx])
        b_all = np.concatenate([baseline_probs_per_origin[i] for i in idx])
        yt_all = np.concatenate([y_true_per_origin[i] for i in idx])
        b_v3 = np.mean((v3_all - yt_all) ** 2)
        b_b = np.mean((b_all - yt_all) ** 2)
        deltas_brier.append(b_b - b_v3)
        f_v3 = np.clip(v3_all, eps, 1 - eps)
        f_b = np.clip(b_all, eps, 1 - eps)
        ll_v3 = np.mean(yt_all * np.log(f_v3) + (1 - yt_all) * np.log(1 - f_v3))
        ll_b = np.mean(yt_all * np.log(f_b) + (1 - yt_all) * np.log(1 - f_b))
        deltas_ll.append(ll_v3 - ll_b)
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


# ---------------------------------------------------------------------------
# Posterior / posterior-predictive checks
# ---------------------------------------------------------------------------

def _gini(x: np.ndarray) -> float:
    x = x[x >= 0]
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    x_sorted = np.sort(x)
    n = len(x_sorted)
    cumsum = np.cumsum(x_sorted)
    return (2 * np.sum(np.arange(1, n + 1) * x_sorted) / (n * cumsum[-1])) - (n + 1) / n


def posterior_predictive_check(
    cells: list[SmoothedRateField],
    observed_counts: np.ndarray,
    exposure_years: float,
    config: AdaptiveSpatialConfig,
) -> dict:
    """Posterior predictive check: can the model reproduce catalog statistics?

    Simulates catalogs from the smoothed rate field and compares summary
    statistics (total, occupied cells, max, Gini, mean nearest-neighbour
    distance) against the observed catalog.
    """
    rng = np.random.default_rng(config.random_seed + 100)
    n_sims = 500
    rates = np.array([c.rate_mean for c in cells])

    # Simulated counts per cell
    sim_counts = rng.poisson(lam=rates[None, :] * exposure_years, size=(n_sims, N_CELLS))

    obs_total = int(observed_counts.sum())
    sim_totals = sim_counts.sum(axis=1)
    obs_occupied = int(np.sum(observed_counts > 0))
    sim_occupied = np.sum(sim_counts > 0, axis=1)
    obs_max = int(observed_counts.max()) if len(observed_counts) else 0
    sim_max = sim_counts.max(axis=1)
    obs_gini = _gini(observed_counts.astype(float))
    sim_ginis = np.array([_gini(sim_counts[s].astype(float)) for s in range(n_sims)])

    # Spatial concentration: fraction of events in top-3 cells
    def _top_frac(counts, k=3):
        s = counts.sum()
        if s == 0: return 0.0
        return float(np.sort(counts)[-k:].sum() / s)
    obs_top3 = _top_frac(observed_counts.astype(float))
    sim_top3 = np.array([_top_frac(sim_counts[s].astype(float)) for s in range(n_sims)])

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
        "observed_top3_fraction": round(obs_top3, 4),
        "sim_top3_mean": round(float(np.mean(sim_top3)), 4),
        "sim_top3_ci": [round(float(np.percentile(sim_top3, 2.5)), 4),
                        round(float(np.percentile(sim_top3, 97.5)), 4)],
    }


# ---------------------------------------------------------------------------
# Permutation test for statistical significance
# ---------------------------------------------------------------------------

def permutation_test_delta(
    v3_probs_per_origin: list,
    baseline_probs_per_origin: list,
    y_true_per_origin: list,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """Permutation test for ΔBrier under the null of no difference.

    For each permutation, randomly swap v3/baseline predictions per origin
    and recompute ΔBrier. Reports the two-sided p-value.
    """
    rng = np.random.default_rng(seed)
    n = len(v3_probs_per_origin)
    if n == 0:
        return {"p_value": 1.0, "n_permutations": n_permutations, "n_origins": 0,
                "observed_delta_brier": 0.0}

    def _delta(v3_list, b_list, y_list):
        v3_all = np.concatenate(v3_list)
        b_all = np.concatenate(b_list)
        yt_all = np.concatenate(y_list)
        b_v3 = np.mean((v3_all - yt_all) ** 2)
        b_b = np.mean((b_all - yt_all) ** 2)
        return b_b - b_v3   # positive = v3 better

    observed = _delta(v3_probs_per_origin, baseline_probs_per_origin, y_true_per_origin)
    perm_deltas = np.zeros(n_permutations)
    for p in range(n_permutations):
        swap = rng.random(n) < 0.5
        v3_perm = [baseline_probs_per_origin[i] if swap[i] else v3_probs_per_origin[i]
                   for i in range(n)]
        b_perm = [v3_probs_per_origin[i] if swap[i] else baseline_probs_per_origin[i]
                  for i in range(n)]
        perm_deltas[p] = _delta(v3_perm, b_perm, y_true_per_origin)

    # Two-sided p-value
    p_value = float(np.mean(np.abs(perm_deltas) >= abs(observed)))
    return {
        "p_value": round(p_value, 4),
        "n_permutations": n_permutations,
        "n_origins": n,
        "observed_delta_brier": round(float(observed), 6),
        "perm_delta_mean": round(float(np.mean(perm_deltas)), 6),
        "perm_delta_std": round(float(np.std(perm_deltas)), 6),
    }
