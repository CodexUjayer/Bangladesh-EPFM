"""Stage 7B backtest: ML vs Spatial Poisson on identical origins.

For every forecast origin:
  1. Compute causal features (ML).
  2. Compute causal spatial-Poisson rate (expanding-window, raw).
  3. Fit ML models on prior origins' feature matrices.
  4. Generate both ML and Spatial-Poisson forecasts for the current origin.
  5. Score both against the same observed outcomes.

IDENTICAL: catalog, grid, origins, training cutoff, horizons, thresholds,
completeness, observed outcomes. No model receives information unavailable
to the other.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from ..baselines.poisson import HORIZON_YEARS
from ..ingestion.schema import CanonicalEvent
from .evaluation import EvalMetrics, evaluate_model
from .features import (
    ALL_FEATURE_NAMES,
    FEATURE_GROUPS,
    FeatureMatrix,
    MLGridConfig,
    compute_features_at_origin,
    features_for_group,
)
from .models import (
    fit_gradient_boosting,
    fit_logistic_l2,
)
from .spatial_poisson import (
    SpatialPoissonConfig,
    base_rate_check,
    block_bootstrap_delta,
    causal_spatial_rate,
    spatial_poisson_forecast,
)

logger = logging.getLogger("stage7b")


@dataclass
class Stage7BOriginResult:
    """Results from one forecast origin."""

    origin_time: datetime
    horizon: str
    threshold: float
    # Per-cell arrays
    y_true: np.ndarray
    ml_preds: dict          # model_key -> y_pred array
    spatial_poisson_pred: np.ndarray
    uniform_poisson_pred: np.ndarray
    # Base-rate check
    base_rate_check: dict


@dataclass
class Stage7BConfig:
    """Configuration for the Stage 7B backtest."""

    horizon: str = "7d"
    threshold: float = 4.5
    mc: float = 4.5
    grid: MLGridConfig = field(default_factory=MLGridConfig)
    origin_start_year: int = 1995
    origin_end_year: int = 2024
    origin_step_years: int = 3
    feature_sets: list = field(default_factory=lambda: ["ML-A", "ML-F"])
    models: list = field(default_factory=lambda: ["gb", "logistic_l2"])
    spatial_method: str = "expanding"
    spatial_smoothing: str = "raw"
    random_seed: int = 42


def run_stage7b_backtest(
    events: list[CanonicalEvent],
    catalog_start: datetime,
    config: Stage7BConfig,
) -> list[Stage7BOriginResult]:
    """Run the Stage 7B backtest: ML vs Spatial Poisson on identical origins."""
    hy = HORIZON_YEARS[config.horizon]
    horizon_days = hy * 365.25
    cell_area_km2 = config.grid.cell_size_deg * 110.574 * \
                    config.grid.cell_size_deg * 111.32 * math.cos(math.radians(24.0))

    # Build feature matrices for ALL origins
    all_origins_fm: list[FeatureMatrix] = []
    for year in range(config.origin_start_year, config.origin_end_year, config.origin_step_years):
        t0 = datetime(year, 1, 1, tzinfo=timezone.utc)
        logger.warning("  computing features for origin %d", year)
        fm = compute_features_at_origin(
            events, origin_time=t0, horizon=config.horizon, threshold=config.threshold,
            grid=config.grid, catalog_start=catalog_start,
            horizon_days=horizon_days, cell_area_km2=cell_area_km2,
        )
        all_origins_fm.append(fm)

    # For each origin: compute spatial Poisson, fit ML on prior origins, predict
    all_results: list[Stage7BOriginResult] = []
    # First pass: compute the observed regional rate (mean across ALL origins)
    # This is needed for the corrected base-rate check (Phase A).
    all_origins_y = [fm.y for fm in all_origins_fm[1:]]  # skip first (no training)
    if all_origins_y:
        observed_regional_rate_mean = float(np.mean([
            1.0 if np.any(y > 0) else 0.0 for y in all_origins_y
        ]))
    else:
        observed_regional_rate_mean = 0.0

    for i, fm in enumerate(all_origins_fm):
        if i == 0:
            continue
        # Spatial Poisson (causal)
        sp_rates = causal_spatial_rate(
            events, origin_time=fm.origin_time, grid=config.grid,
            threshold=config.threshold, catalog_start=catalog_start,
            method=config.spatial_method, smoothing=config.spatial_smoothing,
        )
        sp_pred = spatial_poisson_forecast(sp_rates, hy)
        # Uniform Poisson (expanding regional rate)
        p_uniform = 1.0 - math.exp(-fm.poisson_rate_per_year * hy)
        uniform_pred = np.full(len(fm.y), p_uniform)

        # Base-rate check (Phase A corrected: uses MEAN observed regional rate,
        # not single-origin binary)
        brc = base_rate_check(sp_pred, observed_regional_rate_mean, hy)

        # ML: train on prior origins, predict on current
        train_fms = all_origins_fm[:i]
        X_train_full = np.vstack([f.X for f in train_fms])
        y_train_full = np.concatenate([f.y for f in train_fms])

        ml_preds = {}
        for fs_name in config.feature_sets:
            feat_idx = [ALL_FEATURE_NAMES.index(fn) for fn in features_for_group(fs_name)]
            X_train_fs = X_train_full[:, feat_idx]
            X_test_fs = fm.X[:, feat_idx]
            for model_name in config.models:
                key = f"{model_name}|{fs_name}"
                try:
                    if len(np.unique(y_train_full)) < 2:
                        base_rate = float(np.mean(y_train_full)) if len(y_train_full) > 0 else 0.0
                        y_pred = np.full(len(fm.y), base_rate)
                    elif model_name == "logistic_l2":
                        y_pred, _, _ = fit_logistic_l2(X_train_fs, y_train_full, X_test_fs)
                    elif model_name == "gb":
                        y_pred, _, _ = fit_gradient_boosting(X_train_fs, y_train_full, X_test_fs)
                    else:
                        continue
                    ml_preds[key] = y_pred
                except Exception as e:
                    logger.warning("  model %s failed at origin %d: %s",
                                   key, fm.origin_time.year, str(e)[:80])
                    ml_preds[key] = np.full(len(fm.y), float("nan"))

        all_results.append(Stage7BOriginResult(
            origin_time=fm.origin_time, horizon=config.horizon, threshold=config.threshold,
            y_true=fm.y.astype(float), ml_preds=ml_preds,
            spatial_poisson_pred=sp_pred, uniform_poisson_pred=uniform_pred,
            base_rate_check=brc,
        ))

    return all_results


def aggregate_stage7b(all_results: list[Stage7BOriginResult]) -> dict:
    """Aggregate Stage 7B results: ML vs Spatial Poisson vs Uniform Poisson.

    Returns dict: model_key -> EvalMetrics, plus bootstrap CIs for the
    Spatial-Poisson-vs-ML comparison.
    """
    if not all_results:
        return {}

    # Concatenate all origins
    y_true_all = np.concatenate([r.y_true for r in all_results])
    sp_all = np.concatenate([r.spatial_poisson_pred for r in all_results])
    uniform_all = np.concatenate([r.uniform_poisson_pred for r in all_results])

    results = {}
    # Spatial Poisson baseline
    results["spatial_poisson"] = evaluate_model(
        "spatial_poisson", sp_all, y_true_all, sp_all)
    # Uniform Poisson (for reference)
    results["uniform_poisson"] = evaluate_model(
        "uniform_poisson", uniform_all, y_true_all, sp_all)
    # ML models
    all_ml_keys = set()
    for r in all_results:
        all_ml_keys.update(r.ml_preds.keys())

    bootstrap_results = {}
    for key in sorted(all_ml_keys):
        ml_preds_per_origin = []
        sp_preds_per_origin = []
        y_true_per_origin = []
        for r in all_results:
            if key in r.ml_preds and not np.all(np.isnan(r.ml_preds[key])):
                ml_preds_per_origin.append(r.ml_preds[key])
                sp_preds_per_origin.append(r.spatial_poisson_pred)
                y_true_per_origin.append(r.y_true)
        if not ml_preds_per_origin:
            continue
        ml_all = np.concatenate(ml_preds_per_origin)
        yt_all = np.concatenate(y_true_per_origin)
        sp_sub = np.concatenate(sp_preds_per_origin)
        mask = ~np.isnan(ml_all)
        results[key] = evaluate_model(key, ml_all[mask], yt_all[mask], sp_sub[mask])
        # Block bootstrap Δ vs spatial Poisson
        bootstrap_results[key] = block_bootstrap_delta(
            ml_preds_per_origin, sp_preds_per_origin, y_true_per_origin,
            n_bootstrap=500, seed=42,
        )

    return {"evaluations": results, "bootstrap": bootstrap_results}
