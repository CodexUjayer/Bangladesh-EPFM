"""Stage 3 runner: ingest local files -> canonical events -> completeness ->
declustering -> empirical report.

This script is the single entry point for Stage 3. It reads LOCAL files
from data/raw/{usgs,gcmt,isc-gem}/ and produces outputs/stage3_report.md
plus outputs/stage3_catalog.{json,csv}.

If a source's local file is absent, it is reported as missingness in the
report (no fabrication, no live API fallback unless explicitly enabled).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make the package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ingestion import (
    build_canonical_events,
    read_gcmt_ndk,
    read_iscgem_csv,
    read_usgs_csv,
    read_usgs_geojson,
)
from src.ingestion.provenance import step_acquired_local_file
from src.preprocessing.report import generate_stage3_report, save_stage3_artifacts

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("stage3")


# Study-region subregions (must match configs/study_region.yaml).
SPATIAL_SUBREGIONS = [
    ("shillong_plateau", (24.5, 26.5, 89.5, 92.5)),
    ("indo_burman_fold_belt", (21.0, 26.0, 92.0, 95.5)),
    ("arakan_megathrust", (18.0, 23.0, 92.0, 95.0)),
    ("bangladesh_platform", (22.0, 26.0, 88.0, 92.0)),
    ("chittagong_tripura_fold_belt", (21.0, 24.0, 91.0, 92.5)),
    ("surrounding_himalaya", (26.0, 28.0, 88.0, 93.0)),
]


def main() -> int:
    root = Path(__file__).resolve().parent
    raw = root / "data" / "raw"
    out = root / "outputs"

    files_loaded: dict[str, str] = {}
    files_missing: list[str] = []
    observations_by_source: dict[str, list] = {}

    # ---- USGS (CSV primary, GeoJSON fallback) ----
    usgs_csv = raw / "usgs" / "usgs_bangladesh_1973_2025_m4.csv"
    usgs_geojson = raw / "usgs" / "usgs_bangladesh_1973_2025_m4.geojson"
    if usgs_csv.exists():
        obs = read_usgs_csv(usgs_csv)
        observations_by_source["usgs"] = obs
        files_loaded["usgs"] = str(usgs_csv)
        logger.info("USGS: %d observations from %s", len(obs), usgs_csv)
    elif usgs_geojson.exists():
        obs = read_usgs_geojson(usgs_geojson)
        observations_by_source["usgs"] = obs
        files_loaded["usgs"] = str(usgs_geojson)
        logger.info("USGS: %d observations from %s", len(obs), usgs_geojson)
    else:
        files_missing.append("usgs")
        logger.warning("USGS: no local CSV/GeoJSON file found in data/raw/usgs/")

    # ---- GCMT NDK ----
    gcmt_ndk = raw / "gcmt" / "gcmt_bangladesh.ndk"
    gcmt_dir = raw / "gcmt"
    ndk_files = sorted(gcmt_dir.glob("*.ndk")) if gcmt_dir.exists() else []
    if ndk_files:
        obs = []
        for f in ndk_files:
            obs.extend(read_gcmt_ndk(f))
        observations_by_source["gcmt"] = obs
        files_loaded["gcmt"] = ", ".join(str(f) for f in ndk_files)
        logger.info("GCMT: %d observations from %d NDK files", len(obs), len(ndk_files))
    else:
        files_missing.append("gcmt")
        logger.warning("GCMT: no local .ndk file found in data/raw/gcmt/")

    # ---- ISC-GEM CSV ----
    iscgem_dir = raw / "isc-gem"
    iscgem_files = sorted(iscgem_dir.glob("*.csv")) if iscgem_dir.exists() else []
    if iscgem_files:
        obs = []
        for f in iscgem_files:
            obs.extend(read_iscgem_csv(f))
        observations_by_source["isc-gem"] = obs
        files_loaded["isc-gem"] = ", ".join(str(f) for f in iscgem_files)
        logger.info("ISC-GEM: %d observations from %d CSV files", len(obs), len(iscgem_files))
    else:
        files_missing.append("isc-gem")
        logger.warning("ISC-GEM: no local CSV found in data/raw/isc-gem/")

    # ---- Merge + canonical matching ----
    all_obs = []
    for src, obs_list in observations_by_source.items():
        all_obs.extend(obs_list)
    if not all_obs:
        logger.error("No observations ingested. Supply local files under data/raw/.")
        return 1

    logger.info("Total observations: %d. Building canonical events...", len(all_obs))
    events = build_canonical_events(
        all_obs, time_window_s=60.0, spatial_window_km=50.0
    )
    logger.info("Canonical events: %d", len(events))

    # Attach acquisition provenance step to each event's trail.
    for src, obs_list in observations_by_source.items():
        path = files_loaded.get(src, "?")
        step = step_acquired_local_file(src, path, len(obs_list))
        for ev in events:
            # only add to events that have an observation from this source
            if any(o.source_catalog == src for o in ev.observations):
                ev.provenance.steps.insert(0, step)

    # ---- Generate report ----
    report_md = generate_stage3_report(
        events=events,
        observations_by_source=observations_by_source,
        files_loaded=files_loaded,
        files_missing=files_missing,
        spatial_subregions=SPATIAL_SUBREGIONS,
    )
    save_stage3_artifacts(events, report_md, out)
    logger.info("Stage 3 report saved to %s/stage3_report.md", out)
    logger.info("Stage 3 catalog saved to %s/stage3_catalog.{json,csv}", out)
    print("\n" + "=" * 70)
    print(report_md[:2500])
    print("...[truncated; see outputs/stage3_report.md for full report]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
