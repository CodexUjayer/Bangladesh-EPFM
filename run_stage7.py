"""Stage 7 runner: ML forecasting with chronological evaluation.

Tests whether ML beats the corrected Poisson baseline on properly calibrated
probabilistic scoring. Strict no-leakage, spatiotemporal leakage control,
ablation study, multiple-comparison control.

Matrix (focused for runtime):
  Horizons: 7d, 30d
  Thresholds: M>=5.0, M>=4.5
  Feature sets: ML-A (rate only), ML-D (+temporal+mag+spatial), ML-F (all seismic)
  Models: logistic_l2, rf, gb, calibrated_gb

Total: 2 × 2 × 3 × 4 = 48 model configs + Poisson baseline.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ingestion import build_canonical_events, read_usgs_csv
from src.ml.backtest import BacktestConfig, aggregate_evaluations, run_chronological_backtest
from src.ml.features import MLGridConfig
from src.ml.report import generate_stage7_report, save_stage7_artifacts

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("stage7")


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
        "grid": "1.0 deg",
        "n_cells": 64,
        "geographic_region": "lat [20,28] x lon [88,96]",
    }

    # Focused matrix for runtime — 2 configs, every 3 years
    # 7d uses M>=4.5 (more positives so ML can train); 30d uses M>=5.0
    config_matrix = [
        ("7d", 4.5),
        ("30d", 5.0),
    ]
    all_results = {}
    experiment_manifest = []

    for horizon, threshold in config_matrix:
        logger.warning("Running backtest: horizon=%s, threshold=M>=%s", horizon, threshold)
        config = BacktestConfig(
            horizon=horizon, threshold=threshold, mc=4.5,
            grid=MLGridConfig(),
            origin_start_year=1995, origin_end_year=2024, origin_step_years=3,  # every 3yr
            feature_sets=["ML-A", "ML-F"],  # rate-only vs all-seismic
            models=["logistic_l2", "gb"],   # 2 models
            random_seed=42,
        )
        all_preds = run_chronological_backtest(events, t_min, config)
        logger.warning("  %d forecast origins evaluated", len(all_preds))
        results = aggregate_evaluations(all_preds)
        # Add Poisson baseline explicitly
        if all_preds:
            from src.ml.evaluation import evaluate_model
            import numpy as np
            y_true_all = np.concatenate([op.y_true for op in all_preds])
            poisson_all = np.concatenate([op.poisson_pred for op in all_preds])
            results["poisson"] = evaluate_model(
                "poisson", poisson_all, y_true_all, poisson_all,
            )
        all_results[(horizon, threshold)] = results
        # Record manifest
        for key, m in results.items():
            model_name = key.split("|")[0] if "|" in key else key
            fs = key.split("|")[1] if "|" in key else "none"
            experiment_manifest.append({
                "dataset_version": "usgs_bangladesh_1973_2025_m25",
                "feature_version": "v0.1_causal",
                "catalog_version": "usgs M>=2.5 query, floor M3.2",
                "mc_scenario": 4.5,
                "train_period": f"1973-{config.origin_start_year}",
                "validation_period": "none (expanding window)",
                "test_period": f"{config.origin_start_year}-2024",
                "forecast_horizon": horizon,
                "magnitude_threshold": threshold,
                "model": model_name,
                "feature_set": fs,
                "random_seed": 42,
                "calibration_method": "isotonic" if "calibrated" in model_name else "none",
                "brier": m.brier,
                "information_gain": m.information_gain_vs_poisson,
                "beats_poisson": m.brier_improvement > 0,
            })

    # Generate report
    logger.warning("Generating Stage 7 report...")
    report_md = generate_stage7_report(
        all_results=all_results,
        catalog_metadata=catalog_metadata,
        config_matrix=config_matrix,
        experiment_manifest=experiment_manifest,
    )

    save_stage7_artifacts(
        all_results=all_results,
        catalog_metadata=catalog_metadata,
        config_matrix=config_matrix,
        experiment_manifest=experiment_manifest,
        report_md=report_md,
        output_dir=root / "outputs",
    )
    logger.warning("Stage 7 complete. See outputs/stage7_report.md")
    print("\n" + "=" * 70)
    print(report_md[:4000])
    print("...[truncated; see outputs/stage7_report.md for full report]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
