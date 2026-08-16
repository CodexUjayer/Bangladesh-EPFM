"""Phase B runner: complete the missing validation experiments."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ingestion import build_canonical_events, read_usgs_csv
from src.phase_b import (
    run_depth_stratified_analysis,
    run_etas_vs_sp_comparison,
    run_mc_sensitivity,
    run_multiple_comparison_control,
    run_power_analysis,
    run_spatial_holdout,
    run_uncertainty_propagation,
    run_validation_design_analysis,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("phase_b")


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

    results = {}

    # B1: ETAS vs Spatial Poisson direct comparison
    logger.warning("=== B1: ETAS vs Spatial Poisson ===")
    results["b1_etas_vs_sp"] = run_etas_vs_sp_comparison(
        events, t_min, horizons=["7d", "30d"], thresholds=[4.5, 5.0],
        origin_start_year=1998, origin_end_year=2024, origin_step_years=2,
    )

    # B8: Multiple comparison (depends on B1)
    logger.warning("=== B8: Multiple comparison control ===")
    results["b8_multiple_comparison"] = run_multiple_comparison_control(results["b1_etas_vs_sp"])

    # B2: Spatial holdout
    logger.warning("=== B2: Spatial holdout ===")
    results["b2_spatial_holdout"] = run_spatial_holdout(
        events, t_min, horizon="7d", threshold=4.5,
        origin_start_year=1998, origin_end_year=2024, origin_step_years=3,
    )

    # B3: Depth-stratified
    logger.warning("=== B3: Depth-stratified analysis ===")
    results["b3_depth_stratified"] = run_depth_stratified_analysis(
        events, t_min, horizon="7d", threshold=4.5,
        origin_start_year=1998, origin_end_year=2024, origin_step_years=3,
    )

    # B4: Uncertainty propagation
    logger.warning("=== B4: Uncertainty propagation ===")
    results["b4_uncertainty"] = run_uncertainty_propagation(events, t_min)

    # B5: Power analysis
    logger.warning("=== B5: Power analysis ===")
    results["b5_power"] = run_power_analysis()

    # B6: Mc sensitivity
    logger.warning("=== B6: Mc sensitivity ===")
    results["b6_mc_sensitivity"] = run_mc_sensitivity(events, t_min)

    # B7: Validation design
    logger.warning("=== B7: Validation design ===")
    results["b7_validation_design"] = run_validation_design_analysis(events, t_min)

    # Save all results
    out = root / "outputs"
    out.mkdir(exist_ok=True)
    # Save as JSON (convert numpy types)
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, datetime):
            return o.isoformat()
        return str(o)

    # B1 results need special handling (contains arrays + tuple keys)
    b1_save = {}
    for (h, th), res in results["b1_etas_vs_sp"].items():
        evals = {k: v.to_dict() for k, v in res["evaluations"].items()}
        b1_save[f"{h}_{th}"] = {
            "evaluations": evals,
            "bootstrap": res["bootstrap"],
            "permutation": res["permutation"],
        }
    (out / "phase_b_b1_etas_vs_sp.json").write_text(
        json.dumps(b1_save, indent=2, default=_default), encoding="utf-8"
    )
    (out / "phase_b_b2_spatial_holdout.json").write_text(
        json.dumps(results["b2_spatial_holdout"], indent=2, default=_default), encoding="utf-8"
    )
    (out / "phase_b_b3_depth.json").write_text(
        json.dumps(results["b3_depth_stratified"], indent=2, default=_default), encoding="utf-8"
    )
    (out / "phase_b_b4_uncertainty.json").write_text(
        json.dumps(results["b4_uncertainty"], indent=2, default=_default), encoding="utf-8"
    )
    (out / "phase_b_b5_power.json").write_text(
        json.dumps({f"{k[0]}_{k[1]}": v for k, v in results["b5_power"].items()},
                   indent=2, default=_default), encoding="utf-8"
    )
    (out / "phase_b_b6_mc_sensitivity.json").write_text(
        json.dumps(results["b6_mc_sensitivity"], indent=2, default=_default), encoding="utf-8"
    )
    (out / "phase_b_b7_validation_design.json").write_text(
        json.dumps(results["b7_validation_design"], indent=2, default=_default), encoding="utf-8"
    )
    (out / "phase_b_b8_multiple_comparison.json").write_text(
        json.dumps(results["b8_multiple_comparison"], indent=2, default=_default), encoding="utf-8"
    )

    logger.warning("Phase B complete. Generating report...")

    # Generate the Phase B report
    from src.phase_b.report import generate_phase_b_report
    report_md = generate_phase_b_report(results)
    (out / "PHASE_B_REPORT.md").write_text(report_md, encoding="utf-8")
    logger.warning("Phase B report saved to outputs/PHASE_B_REPORT.md")
    print("\n" + "=" * 70)
    print(report_md[:5000])
    print("...[truncated; see outputs/PHASE_B_REPORT.md for full report]")
    return 0


if __name__ == "__main__":
    import numpy as np  # needed for _default
    raise SystemExit(main())
