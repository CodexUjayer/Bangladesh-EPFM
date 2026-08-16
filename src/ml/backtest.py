"""Chronological ML backtest with spatiotemporal leakage control.

ABSOLUTE NO-LEAKAGE: every feature at origin t uses only events before t.
SPATIOTEMPORAL: all cells from one forecast origin stay in the same temporal
split. The model never sees neighboring future cells from the same timestamp
during training.

Structure:
  - Forecast origins are placed at fixed intervals (e.g., yearly).
  - For each origin: compute features (causal), generate Poisson baseline,
    fit ML models on ALL data before the origin, predict on the origin's cells.
  - Aggregate predictions across origins for evaluation.
  - Block bootstrap over origins (not individual rows) for CIs.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from ..baselines.poisson import HORIZON_YEARS
from ..ingestion.schema import CanonicalEvent
from .features import (
    ALL_FEATURE_NAMES,
    FEATURE_GROUPS,
    FeatureMatrix,
    MLGridConfig,
    compute_features_at_origin,
    features_for_group,
)
from .models import (
    ModelResult,
    fit_calibrated_gradient_boosting,
    fit_gradient_boosting,
    fit_logistic_elasticnet,
    fit_logistic_l2,
    fit_random_forest,
)
from .evaluation import EvalMetrics, evaluate_model


@dataclass
class OriginPredictions:
    """Predictions from all models at one forecast origin."""

    origin_time: datetime
    horizon: str
    threshold: float
    cell_ids: list
    y_true: np.ndarray
    poisson_pred: np.ndarray
    model_preds: dict   # model_name -> y_pred array


@dataclass
class BacktestConfig:
    """Configuration for the chronological ML backtest."""

    horizon: str = "7d"
    threshold: float = 5.0
    mc: float = 4.5
    grid: MLGridConfig = field(default_factory=MLGridConfig)
    # Forecast origins: yearly from origin_start_year to origin_end_year
    origin_start_year: int = 1995
    origin_end_year: int = 2024
    origin_step_years: int = 1
    # Feature sets to test (ablation)
    feature_sets: list = field(default_factory=lambda: ["ML-A", "ML-B", "ML-C", "ML-D", "ML-E", "ML-F"])
    # Models to test
    models: list = field(default_factory=lambda: ["logistic_l2", "rf", "gb", "calibrated_gb"])
    random_seed: int = 42


def run_chronological_backtest(
    events: list[CanonicalEvent],
    catalog_start: datetime,
    config: BacktestConfig,
) -> list[OriginPredictions]:
    """Run the chronological ML backtest.

    For each forecast origin:
      1. Compute features (strictly causal).
      2. Build training set from ALL prior origins' feature matrices (each
         prior origin contributes its cells as training rows).
      3. Fit each model on the training set.
      4. Predict on the current origin's cells.
      5. Record predictions.

    SPATIOTEMPORAL LEAKAGE CONTROL: training rows come from PRIOR origins
    only; the current origin's cells are test-only. No cell from the current
    timestamp appears in training.
    """
    hy = HORIZON_YEARS[config.horizon]
    horizon_days = hy * 365.25
    cell_area_km2 = config.grid.cell_size_deg * 110.574 * \
                    config.grid.cell_size_deg * 111.32 * math.cos(math.radians(24.0))

    # Build feature matrices for ALL origins first (so training can use prior origins)
    all_origins_fm: list[FeatureMatrix] = []
    for year in range(config.origin_start_year, config.origin_end_year, config.origin_step_years):
        t0 = datetime(year, 1, 1, tzinfo=timezone.utc)
        import logging
        logging.getLogger("stage7").warning("  computing features for origin %d", year)
        fm = compute_features_at_origin(
            events, origin_time=t0, horizon=config.horizon, threshold=config.threshold,
            grid=config.grid, catalog_start=catalog_start,
            horizon_days=horizon_days, cell_area_km2=cell_area_km2,
        )
        all_origins_fm.append(fm)

    # For each origin, train on all PRIOR origins, predict on current
    all_preds: list[OriginPredictions] = []
    for i, fm in enumerate(all_origins_fm):
        if i == 0:
            continue  # no training data
        # Training: all prior origins' feature matrices
        train_fms = all_origins_fm[:i]
        # Stack training rows (all cells from all prior origins)
        X_train_full = np.vstack([f.X for f in train_fms])
        y_train_full = np.concatenate([f.y for f in train_fms])
        # Filter to non-zero-variance features
        # (keep all features; models handle selection)

        # Poisson baseline: expanding-window rate at THIS origin
        p_poisson = 1.0 - math.exp(-fm.poisson_rate_per_year * hy)
        poisson_pred = np.full(len(fm.y), p_poisson)

        # Fit each model on each feature set
        model_preds = {}
        for fs_name in config.feature_sets:
            feat_idx = [ALL_FEATURE_NAMES.index(fn) for fn in features_for_group(fs_name)]
            X_train_fs = X_train_full[:, feat_idx]
            X_test_fs = fm.X[:, feat_idx]
            for model_name in config.models:
                key = f"{model_name}|{fs_name}"
                try:
                    t_fit = time.time()
                    # Handle single-class training (too few positives): predict base rate
                    if len(np.unique(y_train_full)) < 2:
                        base_rate = float(np.mean(y_train_full)) if len(y_train_full) > 0 else 0.0
                        y_pred = np.full(len(fm.y), base_rate)
                    elif model_name == "logistic_l2":
                        y_pred, _, _ = fit_logistic_l2(X_train_fs, y_train_full, X_test_fs)
                    elif model_name == "logistic_elasticnet":
                        y_pred, _, _ = fit_logistic_elasticnet(X_train_fs, y_train_full, X_test_fs)
                    elif model_name == "rf":
                        y_pred, _, _ = fit_random_forest(X_train_fs, y_train_full, X_test_fs)
                    elif model_name == "gb":
                        y_pred, _, _ = fit_gradient_boosting(X_train_fs, y_train_full, X_test_fs)
                    elif model_name == "calibrated_gb":
                        y_pred, _, _ = fit_calibrated_gradient_boosting(X_train_fs, y_train_full, X_test_fs)
                    else:
                        continue
                    model_preds[key] = y_pred
                except Exception as e:
                    import logging
                    logging.getLogger("stage7").warning(
                        "  model %s failed at origin %s: %s",
                        key, fm.origin_time.year, str(e)[:100])
                    model_preds[key] = np.full(len(fm.y), float("nan"))

        all_preds.append(OriginPredictions(
            origin_time=fm.origin_time, horizon=config.horizon, threshold=config.threshold,
            cell_ids=fm.cell_ids, y_true=fm.y.astype(float),
            poisson_pred=poisson_pred, model_preds=model_preds,
        ))

    return all_preds


def aggregate_evaluations(all_preds: list[OriginPredictions]) -> dict:
    """Aggregate predictions across origins and evaluate each model.

    Returns dict: model_key -> EvalMetrics.
    """
    if not all_preds:
        return {}
    # Concatenate all origins' predictions
    y_true_all = np.concatenate([op.y_true for op in all_preds])
    poisson_all = np.concatenate([op.poisson_pred for op in all_preds])
    results = {}
    # Collect all model keys
    all_keys = set()
    for op in all_preds:
        all_keys.update(op.model_preds.keys())
    for key in sorted(all_keys):
        preds = []
        y_true_subset = []
        poisson_subset = []
        for op in all_preds:
            if key in op.model_preds:
                p = op.model_preds[key]
                if not np.all(np.isnan(p)):
                    preds.append(p)
                    y_true_subset.append(op.y_true)
                    poisson_subset.append(op.poisson_pred)
        if not preds:
            continue
        y_pred_all = np.concatenate(preds)
        y_true_sub = np.concatenate(y_true_subset)
        poisson_sub = np.concatenate(poisson_subset)
        # Skip any remaining NaN rows
        mask = ~np.isnan(y_pred_all)
        if mask.sum() == 0:
            continue
        results[key] = evaluate_model(
            model_name=key, y_pred=y_pred_all[mask], y_true=y_true_sub[mask],
            poisson_pred=poisson_sub[mask],
        )
    return results
