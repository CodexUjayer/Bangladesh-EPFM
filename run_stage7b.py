"""Stage 7B runner: ML vs Spatial Poisson on identical origins.

Tests whether ML adds predictive information beyond the historical spatial
seismicity-rate model. The decisive experiment.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ingestion import build_canonical_events, read_usgs_csv
from src.ml.features import MLGridConfig
from src.ml.stage7b_backtest import Stage7BConfig, aggregate_stage7b, run_stage7b_backtest
from src.ml.stage7b_report import generate_stage7b_report, save_stage7b_artifacts

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("stage7b")


def main() -> int:
    root = Path(__file__).resolve().parent
    catalog_file = root / "data/raw/usgs/usgs_bangladesh_1973_2025_m25.csv"
    if not catalog_file.exists():
        logger.error("Working catalog not found: %s", catalog_file)
        return 1

    obs = read_usgs_csv(catalog_file)
    events = build_canonical_events(obs, time_window_s=60.0, spatial_window_km=50.0)
    logger.warning("Loaded %d canonical events", len(events))

    t_min = min(e.origin_time_utc for e in events)

    catalog_metadata = {
        "catalog_file": str(catalog_file),
        "catalog_version": "usgs_bangladesh_1973_2025_m25",
        "n_events_total": len(events),
        "grid": "1.0 deg, 64 cells",
    }

    # Two configs: 7d M>=4.5 and 30d M>=5.0 (same as Stage 7)
    config_matrix = [
        ("7d", 4.5),
        ("30d", 5.0),
    ]
    all_configs_results = {}
    all_base_rate_checks = []
    experiment_manifest = []

    for horizon, threshold in config_matrix:
        logger.warning("Stage 7B: horizon=%s, threshold=M>=%s", horizon, threshold)
        config = Stage7BConfig(
            horizon=horizon, threshold=threshold, mc=4.5,
            grid=MLGridConfig(),
            origin_start_year=1995, origin_end_year=2024, origin_step_years=3,
            feature_sets=["ML-A", "ML-F"],
            models=["gb", "logistic_l2"],
            spatial_method="expanding", spatial_smoothing="raw",
            random_seed=42,
        )
        results = run_stage7b_backtest(events, t_min, config)
        logger.warning("  %d forecast origins evaluated", len(results))
        # Collect base-rate checks
        for r in results:
            brc = r.base_rate_check
            brc["origin"] = r.origin_time.year
            all_base_rate_checks.append(brc)
        # Aggregate
        agg = aggregate_stage7b(results)
        all_configs_results[(horizon, threshold)] = agg
        # Manifest
        evals = agg.get("evaluations", {})
        for key, m in evals.items():
            model_name = key.split("|")[0] if "|" in key else key
            fs = key.split("|")[1] if "|" in key else "none"
            experiment_manifest.append({
                "stage": "7B",
                "dataset_version": "usgs_bangladesh_1973_2025_m25",
                "catalog_version": "usgs M>=2.5 query, floor M3.2",
                "mc_scenario": 4.5,
                "forecast_horizon": horizon,
                "magnitude_threshold": threshold,
                "model": model_name,
                "feature_set": fs,
                "random_seed": 42,
                "brier": m.brier,
                "brier_spatial_poisson": evals.get("spatial_poisson", m).brier if "spatial_poisson" in evals else None,
                "delta_brier_vs_sp": (evals["spatial_poisson"].brier - m.brier) if "spatial_poisson" in evals and key != "spatial_poisson" and key != "uniform_poisson" else None,
                "information_gain_vs_sp": (m.log_likelihood - evals["spatial_poisson"].log_likelihood) if "spatial_poisson" in evals and key != "spatial_poisson" and key != "uniform_poisson" else None,
            })

    # Generate report
    logger.warning("Generating Stage 7B report...")
    report_md = generate_stage7b_report(
        all_configs_results=all_configs_results,
        base_rate_checks=all_base_rate_checks,
        catalog_metadata=catalog_metadata,
        experiment_manifest=experiment_manifest,
    )
    save_stage7b_artifacts(
        all_configs_results=all_configs_results,
        base_rate_checks=all_base_rate_checks,
        catalog_metadata=catalog_metadata,
        experiment_manifest=experiment_manifest,
        report_md=report_md,
        output_dir=root / "outputs",
    )
    logger.warning("Stage 7B complete. See outputs/stage7b_report.md")
    print("\n" + "=" * 70)
    print(report_md[:4500])
    print("...[truncated; see outputs/stage7b_report.md for full report]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
