"""B2: Spatial holdout — does ML generalize or memorize?

Tests whether ML models learn transferable relationships or simply memorize
historically active cells. Uses geographic block holdout (not random cell
split, because neighboring cells are spatially dependent).

Design: hold out 2 spatial blocks (e.g., NE and SW quadrants) during training;
evaluate on held-out blocks. Compare ML vs Spatial Poisson.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from ..baselines.poisson import HORIZON_YEARS
from ..ingestion.schema import CanonicalEvent
from ..ml.features import MLGridConfig, compute_features_at_origin
from ..ml.models import fit_gradient_boosting, fit_logistic_l2
from ..ml.spatial_poisson import causal_spatial_rate, spatial_poisson_forecast

logger = logging.getLogger("phase_b.b2")


def run_spatial_holdout(
    events: list[CanonicalEvent],
    catalog_start: datetime,
    horizon: str = "7d",
    threshold: float = 4.5,
    grid: Optional[MLGridConfig] = None,
    origin_start_year: int = 1998,
    origin_end_year: int = 2024,
    origin_step_years: int = 3,
) -> dict:
    """Run the spatial holdout test.

    Splits the 8×8 grid into 4 quadrants (NW, NE, SW, SE). For each fold,
    holds out one quadrant (16 cells) and trains on the other 48 cells.
    Evaluates ML and SP on the held-out 16 cells.

    This tests whether ML generalizes to spatial regions it never saw in
    training, or whether it merely memorizes which cells are historically
    active.
    """
    if grid is None:
        grid = MLGridConfig()

    hy = HORIZON_YEARS[horizon]
    cell_area_km2 = grid.cell_size_deg * 110.574 * grid.cell_size_deg * 111.32 * math.cos(math.radians(24.0))

    # Define 4 quadrants
    n_lat_half = grid.n_lat // 2  # 4
    n_lon_half = grid.n_lon // 2  # 4
    quadrants = {
        "NW": (0, n_lat_half, 0, n_lon_half),
        "NE": (0, n_lat_half, n_lon_half, grid.n_lon),
        "SW": (n_lat_half, grid.n_lat, 0, n_lon_half),
        "SE": (n_lat_half, grid.n_lat, n_lon_half, grid.n_lon),
    }

    # Build feature matrices for all origins
    all_fms = []
    for year in range(origin_start_year, origin_end_year, origin_step_years):
        t0 = datetime(year, 1, 1, tzinfo=timezone.utc)
        fm = compute_features_at_origin(
            events, origin_time=t0, horizon=horizon, threshold=threshold,
            grid=grid, catalog_start=catalog_start,
            horizon_days=hy * 365.25, cell_area_km2=cell_area_km2,
        )
        all_fms.append(fm)

    results = {"quadrants": {}, "config": {"horizon": horizon, "threshold": threshold}}

    for qname, (la_lo, la_hi, lo_lo, lo_hi) in quadrants.items():
        logger.warning("B2: spatial holdout quadrant %s", qname)
        # Held-out cell indices
        held_out_idx = []
        train_idx = []
        for i_lat in range(grid.n_lat):
            for i_lon in range(grid.n_lon):
                idx = i_lat * grid.n_lon + i_lon
                if la_lo <= i_lat < la_hi and lo_lo <= i_lon < lo_hi:
                    held_out_idx.append(idx)
                else:
                    train_idx.append(idx)

        # For each origin: train ML on train_idx cells, predict held_out_idx
        from ..ml.features import ALL_FEATURE_NAMES, features_for_group
        feat_idx = [ALL_FEATURE_NAMES.index(fn) for fn in features_for_group("ML-F")]

        ml_preds_gb = []
        ml_preds_log = []
        sp_preds = []
        y_true_held = []

        for i, fm in enumerate(all_fms):
            if i == 0:
                continue
            # Training: prior origins' train_idx cells
            train_fms = all_fms[:i]
            X_train = np.vstack([f.X[train_idx, :] for f in train_fms])[:, feat_idx]
            y_train = np.concatenate([f.y[train_idx] for f in train_fms])
            # Test: current origin's held_out_idx cells
            X_test = fm.X[held_out_idx, :][:, feat_idx]
            y_test = fm.y[held_out_idx]

            # SP on held-out cells (causal)
            sp_rates = causal_spatial_rate(
                events, origin_time=fm.origin_time, grid=grid, threshold=threshold,
                catalog_start=catalog_start, method="expanding", smoothing="raw",
            )
            sp_pred_held = spatial_poisson_forecast(sp_rates[held_out_idx], hy)

            # ML
            try:
                if len(np.unique(y_train)) < 2:
                    base_rate = float(np.mean(y_train)) if len(y_train) > 0 else 0.0
                    p_gb = np.full(len(y_test), base_rate)
                    p_log = np.full(len(y_test), base_rate)
                else:
                    p_gb, _, _ = fit_gradient_boosting(X_train, y_train, X_test)
                    p_log, _, _ = fit_logistic_l2(X_train, y_train, X_test)
            except Exception:
                p_gb = np.full(len(y_test), float("nan"))
                p_log = np.full(len(y_test), float("nan"))

            ml_preds_gb.append(p_gb)
            ml_preds_log.append(p_log)
            sp_preds.append(sp_pred_held)
            y_true_held.append(y_test.astype(float))

        # Evaluate
        from ..ml.evaluation import evaluate_model
        y_all = np.concatenate(y_true_held)
        sp_all = np.concatenate(sp_preds)
        gb_all = np.concatenate(ml_preds_gb)
        log_all = np.concatenate(ml_preds_log)

        mask_gb = ~np.isnan(gb_all)
        mask_log = ~np.isnan(log_all)

        q_result = {
            "n_held_out_cells": len(held_out_idx),
            "n_train_cells": len(train_idx),
            "n_origins": len(y_true_held),
            "n_positive": int(y_all.sum()),
            "spatial_poisson": evaluate_model("sp", sp_all, y_all, sp_all).to_dict(),
            "gb_ml_f": evaluate_model("gb", gb_all[mask_gb], y_all[mask_gb], sp_all[mask_gb]).to_dict() if mask_gb.any() else None,
            "logistic_ml_f": evaluate_model("log", log_all[mask_log], y_all[mask_log], sp_all[mask_log]).to_dict() if mask_log.any() else None,
        }
        results["quadrants"][qname] = q_result

    return results
