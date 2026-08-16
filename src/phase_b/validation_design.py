"""B7: Validation design analysis.

Tests: more frequent origins, expanding vs rolling window, different start dates.
Creates clear distinction between model development / selection / evaluation data.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from ..baselines.poisson import HORIZON_YEARS
from ..ingestion.schema import CanonicalEvent
from ..ml.features import MLGridConfig, compute_features_at_origin
from ..ml.spatial_poisson import causal_spatial_rate, spatial_poisson_forecast
from ..ml.evaluation import evaluate_model

logger = logging.getLogger("phase_b.b7")


def run_validation_design_analysis(
    events: list[CanonicalEvent],
    catalog_start: datetime,
    horizon: str = "7d",
    threshold: float = 4.5,
    grid: Optional[MLGridConfig] = None,
) -> dict:
    """Analyze validation design: origin frequency, window type, data splits."""
    if grid is None:
        grid = MLGridConfig()

    hy = HORIZON_YEARS[horizon]
    cell_area_km2 = grid.cell_size_deg * 110.574 * grid.cell_size_deg * 111.32 * math.cos(math.radians(24.0))

    results = {
        "config": {"horizon": horizon, "threshold": threshold},
        "data_splits": _define_data_splits(events, catalog_start),
        "origin_frequency": {},
        "window_comparison": {},
    }

    # Test different origin frequencies
    for step in [1, 2, 3]:
        logger.warning("B7: origin frequency every %d years", step)
        origins = list(range(1995, 2024, step))
        brier = _evaluate_with_origins(events, catalog_start, origins, horizon, threshold, grid, hy, cell_area_km2,
                                        method="expanding")
        results["origin_frequency"][f"every_{step}yr"] = {
            "n_origins": len(origins),
            "brier_sp": brier,
        }

    # Expanding vs rolling window
    for method in ["expanding", "rolling_10yr"]:
        logger.warning("B7: window method %s", method)
        origins = list(range(1995, 2024, 2))
        brier = _evaluate_with_origins(events, catalog_start, origins, horizon, threshold, grid, hy, cell_area_km2,
                                        method=method)
        results["window_comparison"][method] = {"n_origins": len(origins), "brier_sp": brier}

    return results


def _define_data_splits(events, catalog_start):
    """Define the development / selection / evaluation split."""
    t_min = min(e.origin_time_utc for e in events)
    t_max = max(e.origin_time_utc for e in events)
    span_years = (t_max - t_min).total_seconds() / (365.25 * 86400)
    # Split: 50% development, 25% selection, 25% evaluation
    dev_end = t_min + timedelta(days=span_years * 0.5 * 365.25)
    sel_end = t_min + timedelta(days=span_years * 0.75 * 365.25)
    return {
        "development": {"start": t_min.isoformat(), "end": dev_end.isoformat(),
                         "purpose": "model development and initial training"},
        "selection": {"start": dev_end.isoformat(), "end": sel_end.isoformat(),
                       "purpose": "model selection and hyperparameter tuning"},
        "evaluation": {"start": sel_end.isoformat(), "end": t_max.isoformat(),
                        "purpose": "FINAL untouched evaluation only"},
        "span_years": span_years,
        "note": "The current system used all 9 origins (1995-2022) as both selection "
                "and evaluation. A proper split would reserve the last 25% as "
                "untouched evaluation. With 52 years: dev=1973-1999, sel=1999-2012, "
                "eval=2012-2024.",
    }


def _evaluate_with_origins(events, catalog_start, origin_years, horizon, threshold, grid, hy, cell_area_km2, method):
    """Evaluate spatial Poisson with given origin list and window method."""
    sp_preds = []
    y_trues = []
    for year in origin_years:
        t0 = datetime(year, 1, 1, tzinfo=timezone.utc)
        fm = compute_features_at_origin(
            events, origin_time=t0, horizon=horizon, threshold=threshold,
            grid=grid, catalog_start=catalog_start,
            horizon_days=hy * 365.25, cell_area_km2=cell_area_km2,
        )
        if method == "rolling_10yr":
            # Rolling 10-year window
            from datetime import timedelta
            window_start = t0 - timedelta(days=10 * 365.25)
            window_events = [e for e in events if window_start <= e.origin_time_utc < t0]
            sp_rates = causal_spatial_rate(
                window_events, origin_time=t0, grid=grid, threshold=threshold,
                catalog_start=window_start, method="expanding", smoothing="raw",
            )
        else:
            sp_rates = causal_spatial_rate(
                events, origin_time=t0, grid=grid, threshold=threshold,
                catalog_start=catalog_start, method="expanding", smoothing="raw",
            )
        sp_pred = spatial_poisson_forecast(sp_rates, hy)
        sp_preds.append(sp_pred)
        y_trues.append(fm.y.astype(float))

    y_all = np.concatenate(y_trues)
    sp_all = np.concatenate(sp_preds)
    m = evaluate_model("sp", sp_all, y_all, sp_all)
    return m.brier
