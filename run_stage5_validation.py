"""Stage 5 validation runner: rebuilt backtest + sensitivity + depth + diagnostics.

Resolves the two methodological issues from the initial Stage 5:
  1. Event-conditioned backtest rebuilt with mutually-exclusive windows.
  2. Externally-informed ETAS properly labeled, sensitivity-tested, validated
     against multiple published priors.

Also adds:
  - Depth-dependence analysis
  - Direct Omori-decay diagnostic (non-parametric R(Δt))
  - Spatial aftershock diagnostic
  - Corrected scientific conclusion (exact user wording)
  - Stage-6-gate question answers
"""

from __future__ import annotations

import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.etas import (
    DEFAULT_EXTERNAL_PARAMS,
    analyze_depth_dependence,
    build_conditioned_origins,
    compute_omori_diagnostic,
    compute_spatial_diagnostic,
    generate_stage5_validation_report,
    run_full_conditioned_backtest,
    run_sensitivity_analysis,
    save_stage5_validation_artifacts,
)
from src.etas.event_conditioned import build_conditioned_origins as build_origins
from src.etas.model import ETASParams
from src.etas.sensitivity import build_param_set
from src.ingestion import build_canonical_events, read_usgs_csv

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("stage5_validation")


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

    catalog_metadata = {
        "catalog_file": str(catalog_file),
        "catalog_version": "usgs_bangladesh_1973_2025_m25",
        "n_events_total": len(events),
        "exposure_years": exposure_years,
        "working_threshold": 4.5,
        "geographic_region": "lat [20,28] x lon [88,96]",
    }

    # ---- 1. Rebuilt event-conditioned backtest ----
    logger.warning("Step 1: rebuilt event-conditioned backtest (mutually-exclusive windows)...")
    forced_params = ETASParams(
        mu_total_per_year=10.0,
        K=DEFAULT_EXTERNAL_PARAMS["K"],
        alpha=DEFAULT_EXTERNAL_PARAMS["alpha"],
        c_days=DEFAULT_EXTERNAL_PARAMS["c_days"],
        p=DEFAULT_EXTERNAL_PARAMS["p"],
        sigma_km=DEFAULT_EXTERNAL_PARAMS["sigma_km"],
        gamma=DEFAULT_EXTERNAL_PARAMS["gamma"],
        q=DEFAULT_EXTERNAL_PARAMS["q"],
        Mc=4.5, spatial_kernel="powerlaw",
        fixed_parameters=DEFAULT_EXTERNAL_PARAMS,
    )
    # Reduced matrix for runtime; the key comparison is M>=5.0 (where Stage 5
    # found ETAS helps) at 7d and 30d, with mainshock definition M>=5.0.
    conditioned_results = run_full_conditioned_backtest(
        events,
        thresholds=[5.0],
        horizons=["7d", "30d"],
        mainshock_definitions=[5.0],
        Mc=4.5,
        forced_params=forced_params,
        catalog_start=t_min,
    )
    logger.warning("  %d conditioned backtest result rows", len(conditioned_results))

    # ---- 2. Sensitivity analysis on M>=5.0 30d post-mainshock (the config where ETAS helps) ----
    logger.warning("Step 2: externally-informed ETAS sensitivity analysis (30d, where ETAS helps)...")
    # Build origins for the M>=5.0 / 30d / M>=5.0 mainshock config
    sens_origins = build_origins(
        events, horizon="30d", threshold=5.0,
        mainshock_definition=5.0, catalog_start=t_min,
    )
    # Use all post-mainshock origins (0-24h, 1-7d, 8-30d, 31-90d) and SAMPLE for runtime
    sens_origins = [o for o in sens_origins if o.is_post_mainshock]
    import numpy as np
    rng = np.random.default_rng(42)
    if len(sens_origins) > 150:
        idx = rng.choice(len(sens_origins), 150, replace=False)
        sens_origins = [sens_origins[i] for i in idx]
    logger.warning("  sensitivity sweep on %d sampled origins (30d horizon)", len(sens_origins))
    # Need to score them first (with Poisson) so observations are set
    from src.etas.event_conditioned import score_origins
    score_origins(events, sens_origins, threshold=5.0, horizon="30d", Mc=4.5,
                  forced_params=forced_params, catalog_start=t_min)
    # Compute Poisson rate from training data
    train_events = [e for e in events if e.origin_time_utc < min(o.origin_time for o in sens_origins)]
    train_above = [e for e in train_events
                   if (e.mw if e.mw is not None else e.original_magnitude) >= 5.0]
    train_span = (min(o.origin_time for o in sens_origins) - t_min).total_seconds() / (365.25 * 86400)
    poisson_rate = len(train_above) / max(train_span, 1e-6)
    oat_results, published_results, sens_summary = run_sensitivity_analysis(
        events, sens_origins, threshold=5.0, horizon="30d",
        poisson_rate=poisson_rate, Mc=4.5, catalog_start=t_min,
    )
    logger.warning("  %d OAT + %d published-prior results; %d/%d beat Poisson",
                    len(oat_results), len(published_results),
                    sens_summary.n_beat_poisson, sens_summary.n_param_sets)

    # ---- 3. Depth dependence ----
    logger.warning("Step 3: depth-dependence analysis...")
    depth_results = analyze_depth_dependence(events, Mc=4.5)

    # ---- 4. Omori-decay diagnostic ----
    logger.warning("Step 4: direct Omori-decay diagnostic...")
    omori_results = []
    for ms_thr in [5.0, 6.0]:
        od = compute_omori_diagnostic(events, mainshock_threshold=ms_thr,
                                      target_threshold=4.5)
        omori_results.append(od)

    # ---- 5. Spatial aftershock diagnostic ----
    logger.warning("Step 5: spatial aftershock diagnostic...")
    spatial_results = []
    for ms_thr in [5.0, 6.0]:
        sd = compute_spatial_diagnostic(events, mainshock_threshold=ms_thr,
                                        target_threshold=4.5)
        spatial_results.append(sd)

    # ---- 6. Generate report ----
    logger.warning("Generating Stage 5 validation report...")
    report_md = generate_stage5_validation_report(
        events=events,
        conditioned_results=conditioned_results,
        oat_results=oat_results,
        published_prior_results=published_results,
        sensitivity_summary=sens_summary,
        depth_results=depth_results,
        omori_results=omori_results,
        spatial_results=spatial_results,
        catalog_metadata=catalog_metadata,
    )

    save_stage5_validation_artifacts(
        events=events,
        report_md=report_md,
        conditioned_results=conditioned_results,
        oat_results=oat_results,
        published_prior_results=published_results,
        sensitivity_summary=sens_summary,
        depth_results=depth_results,
        omori_results=omori_results,
        spatial_results=spatial_results,
        catalog_metadata=catalog_metadata,
        output_dir=root / "outputs",
    )
    logger.warning("Stage 5 validation complete. See outputs/stage5_validation_report.md")
    print("\n" + "=" * 70)
    print(report_md[:4000])
    print("...[truncated; see outputs/stage5_validation_report.md for full report]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
