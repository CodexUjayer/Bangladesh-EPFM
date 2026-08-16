"""ISC catalog reader — parses ISC FDSN text format into SourceObservation records.

ISC FDSN text format (pipe-delimited):
  #EventID|Time|Latitude|Longitude|Depth/km|Author|Catalog|Contributor|ContributorID|
  MagType|Magnitude|MagAuthor|EventLocationName|EventType
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from ..ingestion.schema import SourceObservation, QualityFlag, EventType

logger = logging.getLogger(__name__)


def read_isc_text(path: str | Path) -> list[SourceObservation]:
    """Read an ISC FDSN text-format catalog into SourceObservation records."""
    path = Path(path)
    observations: list[SourceObservation] = []
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 12:
                continue
            try:
                event_id = parts[0]
                time_str = parts[1]
                lat = float(parts[2])
                lon = float(parts[3])
                depth = float(parts[4]) if parts[4] else 0.0
                mag_type = parts[9].strip().lower() if parts[9] else ""
                mag_val = float(parts[10]) if parts[10] else float("nan")

                t_str = time_str
                if t_str.endswith("Z"):
                    t_str = t_str[:-1] + "+00:00"
                elif "+" not in t_str[10:]:
                    t_str = t_str + "+00:00"
                origin_time = datetime.fromisoformat(t_str)
                if origin_time.tzinfo is None:
                    origin_time = origin_time.replace(tzinfo=timezone.utc)

                quality = QualityFlag.REVIEWED
                if "tmp" in mag_type or "auto" in parts[5].lower():
                    quality = QualityFlag.AUTOMATIC

                obs = SourceObservation(
                    source_catalog="isc",
                    native_event_id=f"isc:{event_id}",
                    origin_time_utc=origin_time,
                    latitude=lat, longitude=lon, depth_km=depth,
                    original_magnitude=mag_val,
                    original_magnitude_type=mag_type,
                    quality_flag=quality,
                    event_type=EventType.EARTHQUAKE,
                    acquired_at_utc=datetime.now(timezone.utc),
                    acquisition_method="isc_fdsn_api",
                )
                observations.append(obs)
            except (ValueError, IndexError) as exc:
                logger.warning("Skipping ISC line: %s", str(exc)[:80])

    logger.info("Read %d ISC observations from %s", len(observations), path)
    return observations


def read_isc_allmags(path: str | Path) -> dict[str, list[SourceObservation]]:
    """Read ISC includeallmagnitudes catalog, grouping by EventID.

    Returns dict: event_id -> list of SourceObservation (one per magnitude).
    """
    obs_list = read_isc_text(path)
    grouped: dict[str, list[SourceObservation]] = {}
    for obs in obs_list:
        eid = obs.native_event_id
        grouped.setdefault(eid, []).append(obs)
    logger.info("Grouped %d ISC observations into %d unique events", len(obs_list), len(grouped))
    return grouped
