"""B3: Depth-stratified analysis.

Tests whether separate depth regimes improve forecasting. Stage 5 showed:
  shallow <25 km: CV_IET ≈ 2.48 (strong clustering)
  intermediate 25-70 km: CV_IET ≈ 1.28
  deep ≥70 km: CV_IET ≈ 1.20

Compares:
  A. pooled model (all depths)
  B. shallow / intermediate / deep separate models
  C. shallow vs non-shallow
  D. continuous depth feature (ML only)

Each model compared against Spatial Poisson on the same depth-stratified test set.
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

logger = logging.getLogger("phase_b.b3")

DEPTH_GROUPS = {
    "shallow": (0.0, 25.0),
    "intermediate": (25.0, 70.0),
    "deep": (70.0, 800.0),
}


def run_depth_stratified_analysis(
    events: list[CanonicalEvent],
    catalog_start: datetime,
    horizon: str = "7d",
    threshold: float = 4.5,
    grid: Optional[MLGridConfig] = None,
    origin_start_year: int = 1998,
    origin_end_year: int = 2024,
    origin_step_years: int = 3,
) -> dict:
    """Run depth-stratified forecasting analysis.

    For each forecast origin, splits future events by depth and evaluates
    whether depth-stratified spatial Poisson beats pooled spatial Poisson.
    """
    if grid is None:
        grid = MLGridConfig()

    hy = HORIZON_YEARS[horizon]
    horizon_td = timedelta(days=hy * 365.25)
    cell_area_km2 = grid.cell_size_deg * 110.574 * grid.cell_size_deg * 111.32 * math.cos(math.radians(24.0))

    # Build feature matrices
    all_fms = []
    for year in range(origin_start_year, origin_end_year, origin_step_years):
        t0 = datetime(year, 1, 1, tzinfo=timezone.utc)
        fm = compute_features_at_origin(
            events, origin_time=t0, horizon=horizon, threshold=threshold,
            grid=grid, catalog_start=catalog_start,
            horizon_days=hy * 365.25, cell_area_km2=cell_area_km2,
        )
        all_fms.append(fm)

    results = {
        "config": {"horizon": horizon, "threshold": threshold},
        "depth_groups": DEPTH_GROUPS,
        "pooled": _evaluate_pooled(events, all_fms, catalog_start, grid, threshold, hy),
        "stratified": {},
    }

    # For each depth group, compute spatial Poisson using only events of that depth
    for dname, (d_min, d_max) in DEPTH_GROUPS.items():
        logger.warning("B3: depth group %s (%.0f-%.0f km)", dname, d_min, d_max)
        depth_events = [e for e in events if d_min <= e.depth_km < d_max]
        if len(depth_events) < 50:
            results["stratified"][dname] = {"n_events": len(depth_events), "skipped": True}
            continue
        results["stratified"][dname] = _evaluate_depth_group(
            depth_events, events, all_fms, catalog_start, grid, threshold, hy, dname,
        )

    return results


def _evaluate_pooled(events, all_fms, catalog_start, grid, threshold, hy):
    """Evaluate pooled spatial Poisson (all depths) — the baseline."""
    sp_preds = []
    y_trues = []
    for i, fm in enumerate(all_fms):
        if i == 0:
            continue
        sp_rates = causal_spatial_rate(
            events, origin_time=fm.origin_time, grid=grid, threshold=threshold,
            catalog_start=catalog_start, method="expanding", smoothing="raw",
        )
        sp_pred = spatial_poisson_forecast(sp_rates, hy)
        sp_preds.append(sp_pred)
        y_trues.append(fm.y.astype(float))

    y_all = np.concatenate(y_trues)
    sp_all = np.concatenate(sp_preds)
    m = evaluate_model("sp_pooled", sp_all, y_all, sp_all)
    return {"n_events": len(events), "n_origins": len(y_trues),
            "n_positive": int(y_all.sum()), "brier": m.brier, "ece": m.expected_calibration_error,
            "eval": m.to_dict()}


def _evaluate_depth_group(depth_events, all_events, all_fms, catalog_start, grid, threshold, hy, dname):
    """Evaluate depth-stratified spatial Poisson for one depth group.

    The spatial rate is estimated from depth-group events only. The test set
    is all cells, but only depth-group events in the forecast window count as
    positives.
    """
    sp_preds = []
    y_trues = []
    for i, fm in enumerate(all_fms):
        if i == 0:
            continue
        # Spatial rate from depth-group events only (causal)
        sp_rates = causal_spatial_rate(
            depth_events, origin_time=fm.origin_time, grid=grid, threshold=threshold,
            catalog_start=catalog_start, method="expanding", smoothing="raw",
        )
        sp_pred = spatial_poisson_forecast(sp_rates, hy)
        sp_preds.append(sp_pred)
        # y_true: only depth-group events in the forecast window count
        from datetime import timedelta
        t0 = fm.origin_time
        t1 = t0 + timedelta(days=hy * 365.25)
        d_min, d_max = DEPTH_GROUPS[dname]
        future_depth_events = [e for e in all_events
                               if t0 <= e.origin_time_utc < t1
                               and d_min <= e.depth_km < d_max
                               and (e.mw if e.mw is not None else e.original_magnitude) >= threshold]
        y = np.zeros(len(fm.y))
        for e in future_depth_events:
            i_lat, i_lon = grid.cell_of(e.latitude, e.longitude)
            y[i_lat * grid.n_lon + i_lon] = 1
        y_trues.append(y.astype(float))

    y_all = np.concatenate(y_trues)
    sp_all = np.concatenate(sp_preds)
    m = evaluate_model(f"sp_{dname}", sp_all, y_all, sp_all)
    return {
        "n_events": len(depth_events),
        "n_origins": len(y_trues),
        "n_positive": int(y_all.sum()),
        "brier": m.brier,
        "ece": m.expected_calibration_error,
        "eval": m.to_dict(),
    }
