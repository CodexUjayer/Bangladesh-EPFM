"""Stage 5 runner: ETAS fitting, branching ratio, residuals, event-conditioned backtest.

Uses the M>=2.5-query USGS catalog with M>=4.5 as the conservative working
modeling threshold. Runs ETAS under three Mc sensitivity scenarios (4.0, 4.5,
5.0) — these are NOT validated completeness thresholds.

Produces:
  outputs/stage5_report.md
  outputs/stage5_etas_parameters.csv
  outputs/stage5_etas_forecasts.csv
  outputs/stage5_backtest/
  outputs/stage5_probability_maps/
  outputs/stage5_residual_diagnostics/
  outputs/stage5_model_metadata.json
"""

from __future__ import annotations

import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.baselines.gutenberg_richter import fit_gutenberg_richter
from src.etas import (
    branching_plausibility,
    compute_branching_ratio,
    compute_residuals,
    event_conditioned_backtest,
    fit_etas_mle,
    forecast_spatial,
    generate_stage5_report,
    save_stage5_artifacts,
)
from src.etas.estimation import prepare_catalog
from src.etas.model import ETASModel
from src.ingestion import build_canonical_events, read_usgs_csv

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("stage5")


def main() -> int:
    root = Path(__file__).resolve().parent
    catalog_file = root / "data/raw/usgs/usgs_bangladesh_1973_2025_m25.csv"
    if not catalog_file.exists():
        logger.error("Working catalog not found: %s", catalog_file)
        return 1

    obs = read_usgs_csv(catalog_file)
    events = build_canonical_events(obs, time_window_s=60.0, spatial_window_km=50.0)
    logger.warning("Loaded %d canonical events", len(events))

    times_dt = [e.origin_time_utc for e in events]
    t_min, t_max = min(times_dt), max(times_dt)
    exposure_years = (t_max - t_min).total_seconds() / (365.25 * 86400)

    mc_scenarios = [4.0, 4.5, 5.0]
    working_threshold = 4.5
    bbox = (20.0, 28.0, 88.0, 96.0)

    catalog_metadata = {
        "catalog_file": str(catalog_file),
        "catalog_version": "usgs_bangladesh_1973_2025_m25",
        "n_events_total": len(events),
        "exposure_years": exposure_years,
        "mc_scenarios": mc_scenarios,
        "working_threshold": working_threshold,
        "geographic_region": "lat [20,28] x lon [88,96]",
    }

    # ---- 1. Fit ETAS under each Mc scenario ----
    logger.warning("Step 1: fitting ETAS under Mc scenarios %s...", mc_scenarios)
    etas_fits = []
    for mc in mc_scenarios:
        logger.warning("  Fitting ETAS at Mc=%s...", mc)
        fit = fit_etas_mle(events, Mc=mc, bbox=bbox,
                           background_kind="kde", spatial_kernel="powerlaw")
        # Attach a b-value to fit_info for forecast scaling
        gr = fit_gutenberg_richter(events, mc=mc)
        fit.fit_info = {"b_value": gr.b_mle if not math.isnan(gr.b_mle) else 1.0}
        etas_fits.append(fit)

    # ---- 2. Branching ratio for each fit ----
    logger.warning("Step 2: branching ratio...")
    branching_results = []
    for fit in etas_fits:
        cat = prepare_catalog(events, Mc=fit.Mc, bbox=bbox)
        gr = fit_gutenberg_richter(events, mc=fit.Mc)
        br = compute_branching_ratio(
            K=fit.params.K, alpha=fit.params.alpha, Mc=fit.Mc,
            mags=cat["mags"], b_value=gr.b_mle if not math.isnan(gr.b_mle) else 1.0,
        )
        br_dict = br.to_dict()
        br_dict["Mc"] = fit.Mc
        branching_results.append(br_dict)

    # ---- 3. Residual diagnostics for each fit ----
    logger.warning("Step 3: residual diagnostics...")
    residual_diagnostics = []
    grid_lats = np.arange(20.5, 28.0, 1.0)
    grid_lons = np.arange(88.5, 96.0, 1.0)
    for fit in etas_fits:
        if math.isnan(fit.params.K):
            residual_diagnostics.append({"Mc": fit.Mc, "n_events": 0,
                                          "notes": "Insufficient data for fit."})
            continue
        cat = prepare_catalog(events, Mc=fit.Mc, bbox=bbox)
        model = ETASModel(params=fit.params, background=fit.background,
                          bbox=bbox, fit_info=fit.fit_info)
        rd = compute_residuals(model, cat["times_days"], cat["lats"],
                               cat["lons"], cat["mags"],
                               grid_lats=grid_lats, grid_lons=grid_lons,
                               cell_area_km2=12000.0)
        rd_dict = rd.to_dict()
        rd_dict["Mc"] = fit.Mc
        residual_diagnostics.append(rd_dict)

    # ---- 4. Event-conditioned backtest ----
    logger.warning("Step 4: event-conditioned backtest (this is the key result)...")
    backtest_results = event_conditioned_backtest(
        events,
        thresholds=[4.5, 5.0],
        horizons=["7d", "30d"],
        Mc=4.5,
        mainshock_threshold=5.0,
    )

    # ---- 5. Spatial forecasts (using the Mc=4.5 fit) ----
    logger.warning("Step 5: spatial forecasts...")
    spatial_forecasts = []
    fit_45 = next((f for f in etas_fits if f.Mc == 4.5), etas_fits[0])
    if not math.isnan(fit_45.params.K):
        cat = prepare_catalog(events, Mc=4.5, bbox=bbox)
        model = ETASModel(params=fit_45.params, background=fit_45.background,
                          bbox=bbox, fit_info=fit_45.fit_info)
        for threshold in [4.5, 5.0]:
            for horizon_name, h_days in [("7d", 7.0 / 365.25), ("30d", 30.0 / 365.25)]:
                sf = forecast_spatial(
                    model, cat["times_days"], cat["lats"], cat["lons"], cat["mags"],
                    forecast_start_days=cat["t_end_days"],
                    horizon_days=h_days, threshold=threshold,
                    grid_lats=grid_lats, grid_lons=grid_lons,
                    cell_area_km2=12000.0,
                )
                spatial_forecasts.append(sf)

    # ---- 6. Generate report ----
    logger.warning("Generating Stage 5 report...")
    report_md = generate_stage5_report(
        events=events,
        etas_fits=etas_fits,
        branching_results=branching_results,
        residual_diagnostics=residual_diagnostics,
        backtest_results=backtest_results,
        spatial_forecasts=spatial_forecasts,
        catalog_metadata=catalog_metadata,
    )

    save_stage5_artifacts(
        events=events,
        report_md=report_md,
        etas_fits=etas_fits,
        branching_results=branching_results,
        residual_diagnostics=residual_diagnostics,
        backtest_results=backtest_results,
        spatial_forecasts=spatial_forecasts,
        catalog_metadata=catalog_metadata,
        output_dir=root / "outputs",
    )
    logger.warning("Stage 5 complete. See outputs/stage5_report.md")
    print("\n" + "=" * 70)
    print(report_md[:4000])
    print("...[truncated; see outputs/stage5_report.md for full report]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
