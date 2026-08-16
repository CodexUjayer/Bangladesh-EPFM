"""Stage 4 runner: temporal Poisson + GR + spatial + large-events + backtest.

Uses the M>=2.5-query USGS catalog (floor M3.2) as the working source
catalog, with M>=4.5 as the conservative working modeling threshold for
model fitting. Mc uncertainty (M3.5-4.5 working range) is propagated
throughout via sensitivity scenarios.

Produces:
  outputs/stage4_report.md
  outputs/stage4_baseline_results.csv
  outputs/stage4_probability_maps/
  outputs/stage4_backtest/
  outputs/stage4_model_metadata.json
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.baselines import (
    assess_large_events,
    estimate_temporal_poisson,
    fit_gr_multiple_thresholds,
    forecast_spatial,
    generate_stage4_report,
    run_chronological_backtest,
    save_stage4_artifacts,
)
from src.baselines.report import build_spatial_grid  # noqa
from src.baselines.spatial import GridConfig, build_spatial_grid as _build_grid
from src.ingestion import build_canonical_events, read_usgs_csv

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("stage4")


def main() -> int:
    root = Path(__file__).resolve().parent
    catalog_file = root / "data/raw/usgs/usgs_bangladesh_1973_2025_m25.csv"
    if not catalog_file.exists():
        logger.error("Working catalog not found: %s", catalog_file)
        return 1

    # ---- Load + canonicalize ----
    obs = read_usgs_csv(catalog_file)
    events = build_canonical_events(obs, time_window_s=60.0, spatial_window_km=50.0)
    logger.warning("Loaded %d canonical events from %s", len(events), catalog_file)

    times = [e.origin_time_utc for e in events]
    t_min, t_max = min(times), max(times)
    exposure_years = (t_max - t_min).total_seconds() / (365.25 * 86400)

    working_threshold = 4.5
    mc_scenarios = [4.0, 4.5, 5.0]

    catalog_metadata = {
        "catalog_file": str(catalog_file),
        "catalog_version": "usgs_bangladesh_1973_2025_m25 (USGS ComCat M>=2.5 query; floor M3.2)",
        "n_events_total": len(events),
        "time_range": f"{t_min.isoformat()} -> {t_max.isoformat()}",
        "exposure_years": exposure_years,
        "working_threshold": working_threshold,
        "mc_scenarios": mc_scenarios,
        "geographic_region": "lat [20,28] x lon [88,96]",
    }

    # ---- 1. Temporal Poisson for multiple thresholds ----
    logger.warning("Step 1: temporal Poisson baselines...")
    poisson_thresholds = [4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
    poisson_results = []
    for th in poisson_thresholds:
        r = estimate_temporal_poisson(events, threshold=th,
                                      exposure_years=exposure_years)
        poisson_results.append(r)

    # ---- 2. Gutenberg-Richter under multiple Mc scenarios ----
    logger.warning("Step 2: Gutenberg-Richter (MLE) under Mc scenarios %s...", mc_scenarios)
    gr_results = fit_gr_multiple_thresholds(events, thresholds=mc_scenarios,
                                            exposure_years=exposure_years)

    # ---- 3. Mc sensitivity is captured by the GR scenarios above ----
    # (reported in Section 4 of the report)

    # ---- 4. Spatial baseline ----
    logger.warning("Step 4: spatial baseline (1.0 deg grid)...")
    grid_config = GridConfig(cell_size_deg=1.0, min_lat=20.0, max_lat=28.0,
                             min_lon=88.0, max_lon=96.0, min_events_for_stable_rate=5)
    spatial_grid = _build_grid(events, threshold=working_threshold, config=grid_config,
                               exposure_years=exposure_years,
                               additional_thresholds=[4.5, 5.0, 5.5, 6.0])

    # ---- 5. Spatial + magnitude forecast ----
    logger.warning("Step 5: spatial + magnitude forecast...")
    spatial_forecast = forecast_spatial(
        events,
        thresholds=[4.5, 5.0, 5.5, 6.0],
        horizons=["24h", "7d", "30d", "90d", "1y"],
        grid_config=grid_config,
        exposure_years=exposure_years,
    )

    # ---- 6. Large-event limitation ----
    logger.warning("Step 6: large-event assessment (M>=6.5, M>=7.0)...")
    large_event_assessments = []
    for th in [6.5, 7.0]:
        a = assess_large_events(events, threshold=th, exposure_years=exposure_years)
        large_event_assessments.append(a)

    # ---- 7. Backtesting ----
    logger.warning("Step 7: chronological backtesting...")
    backtest_configs = [
        (4.5, "7d"), (4.5, "30d"), (4.5, "90d"),
        (5.0, "7d"), (5.0, "30d"), (5.0, "90d"),
        (5.5, "30d"), (6.0, "90d"),
    ]
    backtest_results = []
    for th, h in backtest_configs:
        bt = run_chronological_backtest(
            events, threshold=th, horizon=h,
            origin_start_year=1995, origin_end_year=2024, origin_step_years=1,
            catalog_start_time=t_min,
        )
        backtest_results.append(bt)

    # ---- 8. Generate report ----
    logger.warning("Generating Stage 4 report...")
    report_md = generate_stage4_report(
        events=events,
        working_threshold=working_threshold,
        mc_scenarios=mc_scenarios,
        poisson_results=poisson_results,
        gr_results=gr_results,
        spatial_grid=spatial_grid,
        spatial_forecast=spatial_forecast,
        large_event_assessments=large_event_assessments,
        backtest_results=backtest_results,
        catalog_metadata=catalog_metadata,
    )

    save_stage4_artifacts(
        events=events,
        report_md=report_md,
        poisson_results=poisson_results,
        gr_results=gr_results,
        spatial_grid=spatial_grid,
        spatial_forecast=spatial_forecast,
        large_event_assessments=large_event_assessments,
        backtest_results=backtest_results,
        catalog_metadata=catalog_metadata,
        output_dir=root / "outputs",
    )
    logger.warning("Stage 4 complete. See outputs/stage4_report.md")
    print("\n" + "=" * 70)
    print(report_md[:3500])
    print("...[truncated; see outputs/stage4_report.md for full report]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
