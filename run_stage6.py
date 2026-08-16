"""Stage 6 runner: Coulomb data audit + prototype + (data-limited) forecast.

DATA-INTEGRITY RULE: If validated receiver-fault geometry is unavailable,
real Coulomb forecasting is DISABLED. A mathematical prototype is implemented
and unit-tested with synthetic geometry only.

Produces:
  outputs/stage6_report.md
  outputs/stage6_data_audit.csv / .json
  outputs/stage6_coulomb_parameters.csv
  outputs/stage6_forecasts.csv
  outputs/stage6_backtest/ (empty if data-limited)
  outputs/stage6_stress_maps/ (empty if data-limited)
  outputs/stage6_residual_diagnostics/ (empty if data-limited)
  outputs/stage6_model_metadata.json
"""

from __future__ import annotations

import csv
import io
import logging
import sys
import urllib.request
import json
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.coulomb import (
    CouplingFormulation,
    CouplingParams,
    ElasticParams,
    audit_coulomb_data,
    build_receiver_grid,
    build_source_earthquakes,
    document_formulation,
    forecast_coulomb_modulated_poisson,
)
from src.coulomb.report import generate_stage6_report, save_stage6_artifacts
from src.ingestion import build_canonical_events, read_usgs_csv

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("stage6")


def count_usgs_focal_mechanisms(rows: list[dict], max_check: int = 30) -> int:
    """Count how many of the largest M>=5.5 events have USGS focal-mechanism
    products. Uses the per-event detail API (rate-limited)."""
    m55 = sorted([r for r in rows if float(r["mag"]) >= 5.5],
                 key=lambda r: -float(r["mag"]))[:max_check]
    n_fm = 0
    for r in m55:
        eid = r["id"]
        url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?eventid={eid}&format=geojson"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research/0.1"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                d = json.loads(resp.read())
            prods = d.get("properties", {}).get("products", {})
            if "focal-mechanism" in prods or "moment-tensor" in prods:
                n_fm += 1
        except Exception:
            pass
    return n_fm


def run_unit_tests() -> str:
    """Run the Coulomb prototype unit tests and capture output."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "tests/test_coulomb_prototype.py"],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent),
    )
    return result.stdout


def main() -> int:
    root = Path(__file__).resolve().parent
    catalog_file = root / "data/raw/usgs/usgs_bangladesh_1973_2025_m25.csv"
    if not catalog_file.exists():
        logger.error("Working catalog not found: %s", catalog_file)
        return 1

    obs = read_usgs_csv(catalog_file)
    events = build_canonical_events(obs, time_window_s=60.0, spatial_window_km=50.0)
    logger.warning("Loaded %d canonical events", len(events))

    # ---- 1. Data audit ----
    logger.warning("Step 1: Coulomb data audit...")
    # Count USGS focal mechanisms (rate-limited; sample top 30 M>=5.5)
    rows = list(csv.DictReader(open(catalog_file)))
    n_fm = count_usgs_focal_mechanisms(rows, max_check=30)
    logger.warning("  USGS focal-mechanism products available for %d/30 largest M>=5.5 events", n_fm)

    audit = audit_coulomb_data(
        gcmt_dir=root / "data/raw/gcmt",
        gem_gafd_cache=root / "data/external/gem_gafd.geojson",
        usgs_focal_mechanism_count=n_fm,
        bbox=(20.0, 28.0, 88.0, 96.0),
    )
    logger.warning("  Real Coulomb forecasting ENABLED: %s", audit.real_forecasting_enabled)
    if audit.blocking_gaps:
        for gap in audit.blocking_gaps:
            logger.warning("  BLOCKING GAP: %s", gap)

    # ---- 2. Unit tests (synthetic geometry) ----
    logger.warning("Step 2: running Coulomb prototype unit tests (synthetic geometry)...")
    unit_test_output = run_unit_tests()
    logger.warning("  Unit tests completed.")

    # ---- 3. Elastic + coupling parameters ----
    elastic = ElasticParams(
        shear_modulus_GPa=30.0, poissons_ratio=0.25,
        effective_friction=0.4, skempton_coefficient=0.5,
    )
    coupling = CouplingParams(
        formulation=CouplingFormulation.RATE_AND_STATE,
        A_sigma_bar_MPa=1.0,
    )

    # ---- 4. Forecasts (data-limited if disabled) ----
    logger.warning("Step 3: Coulomb forecasts (data-limited=%s)...",
                    not audit.real_forecasting_enabled)
    forecasts = []
    # If enabled, we would build sources from USGS FMs and receivers from validated fault data.
    # Since disabled, we produce a single data-limited forecast placeholder.
    if not audit.real_forecasting_enabled:
        from datetime import timedelta
        forecasts.append(type("CF", (), {
            "forecast_start": datetime.now(timezone.utc),
            "horizon": "7d", "threshold": 5.0,
            "enabled": False, "per_cell": [],
            "expected_total_count": float("nan"),
            "probability_at_least_one": float("nan"),
            "notes": ["DATA-LIMITED: real Coulomb forecasting disabled by data audit."],
        })())
    else:
        # (Would build real sources/receivers here if data were available)
        pass

    # ---- 5. Stress diagnostics (only if enabled) ----
    stress_diagnostics = []
    # Not computed in data-limited mode

    # ---- 6. Backtest summary ----
    backtest_summary = {
        "enabled": audit.real_forecasting_enabled,
        "n_origins": 0,
        "brier_coulomb": float("nan"),
        "brier_poisson": float("nan"),
        "information_gain": float("nan"),
        "verdict": "DISABLED — real Coulomb forecasting is data-limited.",
    }

    catalog_metadata = {
        "catalog_file": str(catalog_file),
        "catalog_version": "usgs_bangladesh_1973_2025_m25",
        "n_events_total": len(events),
        "geographic_region": "lat [20,28] x lon [88,96]",
    }

    # ---- 7. Generate report ----
    logger.warning("Generating Stage 6 report...")
    report_md = generate_stage6_report(
        audit=audit,
        coupling_params=coupling,
        forecasts=forecasts,
        stress_diagnostics=stress_diagnostics,
        backtest_summary=backtest_summary,
        catalog_metadata=catalog_metadata,
        unit_test_results=unit_test_output,
    )

    save_stage6_artifacts(
        audit=audit,
        coupling_params=coupling,
        forecasts=forecasts,
        stress_diagnostics=stress_diagnostics,
        backtest_summary=backtest_summary,
        catalog_metadata=catalog_metadata,
        report_md=report_md,
        elastic_params=elastic,
        output_dir=root / "outputs",
    )
    logger.warning("Stage 6 complete. See outputs/stage6_report.md")
    print("\n" + "=" * 70)
    print(report_md[:3500])
    print("...[truncated; see outputs/stage6_report.md for full report]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
