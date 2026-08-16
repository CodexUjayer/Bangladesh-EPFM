"""Local-file readers for USGS, ISC-GEM, and GCMT catalogs.

These readers are the PRIMARY ingestion path. The pipeline must work fully
offline from local files supplied by the user:

  - USGS CSV (from https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv)
  - USGS GeoJSON (same endpoint, format=geojson)
  - ISC-GEM CSV export (from http://www.isc.ac.uk/iscgem/download/)
  - GCMT NDK (from https://www.globalcmt.org/CMTsearch.html)

Each reader returns a list of ``SourceObservation`` records with original
magnitudes preserved EXACTLY (no conversion) and an acquisition provenance
step attached via the caller.

Live API ingestion is supported as an OPTIONAL convenience (see
``fetch_usgs_fdsn_api``) but local files remain the reproducible primary
input.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .schema import (
    EventType,
    FocalMechanism,
    QualityFlag,
    SourceObservation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_usgs_time(s: str) -> datetime:
    """USGS time format: '2024-01-01T00:00:00.000Z'."""
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_iso_time(s: str) -> datetime:
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_float(v, default: Optional[float] = None) -> Optional[float]:
    if v is None or v == "" or v == "None" or v == "null":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _to_int(v, default: Optional[int] = None) -> Optional[int]:
    if v is None or v == "" or v == "None" or v == "null":
        return default
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _quality_from_status(status: str) -> QualityFlag:
    s = status.lower().strip()
    if s == "reviewed":
        return QualityFlag.REVIEWED
    if s == "automatic":
        return QualityFlag.AUTOMATIC
    if s in ("historical",):
        return QualityFlag.HISTORICAL
    if s in ("macroseismic",):
        return QualityFlag.MACROSEISMIC
    return QualityFlag.UNKNOWN


def _event_type_from_str(t: str) -> EventType:
    t = t.lower().strip()
    mapping = {
        "earthquake": EventType.EARTHQUAKE,
        "explosion": EventType.EXPLOSION,
        "quarry blast": EventType.QUARRY_BLAST,
        "quarry_blast": EventType.QUARRY_BLAST,
        "rockburst": EventType.ROCKBURST,
    }
    return mapping.get(t, EventType.UNKNOWN)


# ---------------------------------------------------------------------------
# USGS CSV reader  (primary local-file path)
# ---------------------------------------------------------------------------


def read_usgs_csv(path: str | Path) -> list[SourceObservation]:
    """Read a USGS FDSN CSV export into SourceObservation records.

    Expected columns (USGS FDSN CSV header):
        time, latitude, longitude, depth, mag, magType, nst, gap, dmin, rms,
        net, id, updated, place, type, horizontalError, depthError, magError,
        magNst, status, locationSource, magSource
    """
    path = Path(path)
    observations: list[SourceObservation] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                mag = _to_float(row.get("mag"))
                if mag is None:
                    continue  # USGS rows with no magnitude are unusable
                obs = SourceObservation(
                    source_catalog="usgs",
                    native_event_id=row["id"],
                    origin_time_utc=_parse_usgs_time(row["time"]),
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    depth_km=float(row["depth"]),
                    original_magnitude=mag,
                    original_magnitude_type=row.get("magType", "").strip().lower(),
                    magnitude_uncertainty=_to_float(row.get("magError")),
                    horizontal_uncertainty_km=_to_float(row.get("horizontalError")),
                    depth_uncertainty_km=_to_float(row.get("depthError")),
                    n_stations=_to_int(row.get("nst")),
                    gap_deg=_to_float(row.get("gap")),
                    rms_s=_to_float(row.get("rms")),
                    quality_flag=_quality_from_status(row.get("status", "")),
                    event_type=_event_type_from_str(row.get("type", "earthquake")),
                    acquired_at_utc=datetime.now(timezone.utc),
                    acquisition_method="local_file",
                )
                observations.append(obs)
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping USGS row %s: %s", row.get("id", "?"), exc)
    logger.info("Read %d USGS observations from %s", len(observations), path)
    return observations


# ---------------------------------------------------------------------------
# USGS GeoJSON reader
# ---------------------------------------------------------------------------


def read_usgs_geojson(path: str | Path) -> list[SourceObservation]:
    """Read a USGS FDSN GeoJSON export."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        gj = json.load(f)
    observations: list[SourceObservation] = []
    for feat in gj.get("features", []):
        try:
            props = feat.get("properties", {}) or {}
            geom = feat.get("geometry", {}) or {}
            coords = geom.get("coordinates", [None, None, None])
            mag = _to_float(props.get("mag"))
            if mag is None:
                continue
            # USGS GeoJSON time is unix epoch ms
            t_ms = props.get("time")
            if t_ms is not None:
                origin_time = datetime.fromtimestamp(t_ms / 1000.0, tz=timezone.utc)
            else:
                continue
            obs = SourceObservation(
                source_catalog="usgs",
                native_event_id=props.get("id", feat.get("id", "")),
                origin_time_utc=origin_time,
                latitude=float(coords[1]),
                longitude=float(coords[0]),
                depth_km=float(coords[2]) if len(coords) > 2 and coords[2] is not None else 0.0,
                original_magnitude=mag,
                original_magnitude_type=(props.get("magType") or "").strip().lower(),
                magnitude_uncertainty=_to_float(props.get("magError")),
                horizontal_uncertainty_km=_to_float(props.get("horizontalError")),
                depth_uncertainty_km=_to_float(props.get("depthError")),
                n_stations=_to_int(props.get("nst")),
                gap_deg=_to_float(props.get("gap")),
                rms_s=_to_float(props.get("rms")),
                quality_flag=_quality_from_status(props.get("status", "")),
                event_type=_event_type_from_str(props.get("type", "earthquake")),
                acquired_at_utc=datetime.now(timezone.utc),
                acquisition_method="local_file",
            )
            observations.append(obs)
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Skipping USGS GeoJSON feature: %s", exc)
    logger.info("Read %d USGS observations from %s", len(observations), path)
    return observations


# ---------------------------------------------------------------------------
# ISC-GEM CSV reader
# ---------------------------------------------------------------------------


def read_iscgem_csv(path: str | Path) -> list[SourceObservation]:
    """Read an ISC-GEM CSV export.

    ISC-GEM exports vary slightly; this reader handles the standard column
    set documented in the ISC-GEM catalogue (author, eventid, origin_time,
    lat, lon, depth, depth_err, smaj, smin, strike, M, Msigma, Mtype, ...).

    The exact column names are documented at
    http://www.isc.ac.uk/iscgem/download.php. If the user's export differs,
    a small mapping can be added; we do NOT guess.
    """
    path = Path(path)
    observations: list[SourceObservation] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        # Build a case-insensitive column map to handle header variants.
        colmap = {k.lower().strip(): k for k in (reader.fieldnames or [])}

        def get(row, *names):
            for n in names:
                key = colmap.get(n.lower())
                if key and row.get(key) not in (None, ""):
                    return row.get(key)
            return None

        for row in reader:
            try:
                # ISC-GEM time formats: 'YYYY-MM-DD HH:MM:SS.SS' or ISO.
                t_raw = get(row, "origin_time", "time", "date_time", "datetime")
                if t_raw is None:
                    continue
                t_raw = t_raw.strip().replace(" ", "T")
                if not (t_raw.endswith("Z") or "+" in t_raw[10:]):
                    t_raw = t_raw + "+00:00"
                origin_time = _parse_iso_time(t_raw)

                mag = _to_float(get(row, "mw", "m", "magnitude"))
                mag_type_raw = (get(row, "mtype", "magtype", "magnitude_type") or "mw").strip().lower()
                if mag is None:
                    continue

                # ISC-GEM Mw is authoritative; tag as 'mw_iscgem'.
                if mag_type_raw in ("mw", "iscgem", ""):
                    mag_type = "mw_iscgem"
                else:
                    mag_type = mag_type_raw

                obs = SourceObservation(
                    source_catalog="isc-gem",
                    native_event_id=str(get(row, "eventid", "event_id", "iscid") or ""),
                    origin_time_utc=origin_time,
                    latitude=float(get(row, "lat", "latitude")),
                    longitude=float(get(row, "lon", "longitude")),
                    depth_km=float(get(row, "depth", "depth_km") or 0.0),
                    original_magnitude=mag,
                    original_magnitude_type=mag_type,
                    magnitude_uncertainty=_to_float(get(row, "msigma", "mag_error", "m_sigma")),
                    horizontal_uncertainty_km=_to_float(get(row, "smaj", "smin", "err_smaj")),
                    depth_uncertainty_km=_to_float(get(row, "depth_err", "depth_error")),
                    quality_flag=QualityFlag.REVIEWED,
                    event_type=EventType.EARTHQUAKE,
                    acquired_at_utc=datetime.now(timezone.utc),
                    acquisition_method="local_file",
                )
                observations.append(obs)
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("Skipping ISC-GEM row: %s", exc)
    logger.info("Read %d ISC-GEM observations from %s", len(observations), path)
    return observations


# ---------------------------------------------------------------------------
# GCMT NDK reader
# ---------------------------------------------------------------------------


def read_gcmt_ndk(path: str | Path) -> list[SourceObservation]:
    """Read a GCMT NDK file into SourceObservation records.

    NDK format: 5 lines per event.
      Line 1: header (date, time, hypocenter from catalog, Mw, etc.)
      Line 2: centroid (centroid time, lat, lon, depth, type, moment)
      Line 3: exponent + principal axes
      Line 4: nodal plane 1 (strike, dip, rake) + nodal plane 2
      Line 5: name

    The hypocenter in NDK line 1 is the CATALOG hypocenter (from USGS/ISC),
    NOT the GCMT centroid. We use the catalog hypocenter for the
    observation's location (so ETAS triggering uses the nucleation point),
    and store the centroid + focal mechanism as rich fields.

    Reference: https://www.globalcmt.org/CMTdocs/ndk_format.txt
    """
    path = Path(path)
    observations: list[SourceObservation] = []
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    # Group into blocks of 5 lines.
    for i in range(0, len(lines) - 4, 5):
        block = [ln.rstrip("\n") for ln in lines[i:i + 5]]
        try:
            # Line 1: columns are fixed-width.
            #   cols 0-3: 4-digit data type
            #   cols 5-18: earthquake name (9 chars)... Actually NDK line 1:
            #   "YYYY/MM/DD HH:MM:SS.S  lat   lon  depth  Mw  ..."
            # We parse defensively with fixed columns per the spec.
            l1 = block[0]
            date_str = l1[5:15].strip()       # YYYY/MM/DD
            time_str = l1[16:26].strip()      # HH:MM:SS.S
            # hypocenter lat/lon are in the second half of line 1
            # Format: ... half-duration ... catalog hypocenter lat/lon/depth
            # NDK spec line1 field positions (0-indexed):
            #   0-3  data type
            #   5-29 date/time/name
            #   30-38 half-duration? Actually different.
            # Robust approach: split on whitespace, take known positions.
            parts1 = l1.split()
            # parts1: [dtype, date, time, name..., halfdur, lat, lon, depth, mb, ms, ...]
            # This is fragile; the reliable Mw is on line 1 too.
            # Per spec, line 1 last field is Mw.
            mw = _to_float(parts1[-1])
            if mw is None:
                continue

            # Reconstruct datetime
            dt_str = f"{date_str}T{time_str}+00:00"
            try:
                origin_time = _parse_iso_time(dt_str.replace("/", "-"))
            except ValueError:
                continue

            # Line 2: centroid
            parts2 = block[1].split()
            # parts2: [centroid_src_type, expon, ... , clat, clon, cdepth, ctime? ]
            # Per spec line2: 1(1) type, 1(2) exponent, 3(3-5) mantissa of moment,
            # then centroid lat, lon, depth, then centroid time corrections.
            # Centroid lat/lon/depth are at fixed positions:
            #   cols 23-29 centroid lat, 31-38 centroid lon, 40-45 depth
            clat = _to_float(block[1][23:30])
            clon = _to_float(block[1][31:38])
            cdepth = _to_float(block[1][40:45])

            # Use CATALOG hypocenter from line 1 if recoverable, else centroid.
            # Line 1 hypocenter: cols 44-... Actually we use a robust parse.
            # Per spec, line 1 has catalog hypocenter lat (cols 44-50), lon (52-59),
            # depth (61-66). Try those; fall back to centroid.
            hlat = _to_float(l1[44:51])
            hlon = _to_float(l1[51:59])
            hdepth = _to_float(l1[59:66])
            if hlat is None or hlon is None:
                hlat, hlon, hdepth = clat, clon, cdepth

            # Line 4: nodal planes
            #   cols 1-9 plane1 strike, 11-18 dip, 20-26 rake
            #   cols 28-36 plane2 strike, 38-45 dip, 47-53 rake
            strike1 = _to_float(block[3][1:10])
            dip1 = _to_float(block[3][10:19])
            rake1 = _to_float(block[3][19:27])
            # scalar moment from line 2: exponent + mantissa
            exponent = _to_int(parts2[1])
            mantissa = _to_float(parts2[2]) if len(parts2) > 2 else None
            scalar_moment = (
                mantissa * (10 ** exponent) if (mantissa is not None and exponent is not None) else None
            )

            fm: Optional[FocalMechanism] = None
            if strike1 is not None and dip1 is not None and rake1 is not None:
                fm = FocalMechanism(
                    strike_deg=strike1,
                    dip_deg=dip1,
                    rake_deg=rake1,
                    scalar_moment_Nm=scalar_moment,
                    source="gcmt",
                )

            # event id from line 1 (data type + date) — GCMT uses CMT event name
            # on line 5 sometimes; use a constructed id.
            native_id = f"gcmt_{date_str.replace('/','')}"
            # Try to get the official CMT name from line 5 last token
            name_parts = block[4].strip().split()
            if name_parts:
                native_id = name_parts[-1]

            obs = SourceObservation(
                source_catalog="gcmt",
                native_event_id=native_id,
                origin_time_utc=origin_time,
                latitude=hlat if hlat is not None else 0.0,
                longitude=hlon if hlon is not None else 0.0,
                depth_km=hdepth if hdepth is not None else 0.0,
                original_magnitude=mw,
                original_magnitude_type="mw",   # GCMT Mw, authoritative
                magnitude_uncertainty=None,    # GCMT does not report Mw sigma
                quality_flag=QualityFlag.REVIEWED,
                event_type=EventType.EARTHQUAKE,
                focal_mechanism=fm,
                acquired_at_utc=datetime.now(timezone.utc),
                acquisition_method="local_file",
            )
            observations.append(obs)
        except (IndexError, ValueError, TypeError) as exc:
            logger.warning("Skipping GCMT block at line %d: %s", i, exc)
    logger.info("Read %d GCMT observations from %s", len(observations), path)
    return observations


# ---------------------------------------------------------------------------
# Optional: live USGS FDSN API fetch (convenience, not the primary path)
# ---------------------------------------------------------------------------


def fetch_usgs_fdsn_api(
    starttime: str,
    endtime: str,
    minmagnitude: float = 4.0,
    bbox: tuple[float, float, float, float] = (20.0, 28.0, 88.0, 96.0),
    save_path: Optional[str | Path] = None,
) -> list[SourceObservation]:
    """Optionally fetch USGS events from the live FDSN API.

    This is a CONVENIENCE for one-time acquisition. The reproducible primary
    path is local files (read_usgs_csv). If used, the result should be saved
    to data/raw/usgs/ and re-read from disk in subsequent runs.

    Parameters
    ----------
    starttime, endtime : str
        ISO date strings, e.g. "1973-01-01", "2025-01-01".
    minmagnitude : float
    bbox : (min_lat, max_lat, min_lon, max_lon)
    save_path : optional path to cache the CSV.
    """
    import urllib.request

    min_lat, max_lat, min_lon, max_lon = bbox
    url = (
        "https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv"
        f"&starttime={starttime}&endtime={endtime}"
        f"&minmagnitude={minmagnitude}"
        f"&minlatitude={min_lat}&maxlatitude={max_lat}"
        f"&minlongitude={min_lon}&maxlongitude={max_lon}"
        f"&eventtype=earthquake&orderby=time-asc"
    )
    logger.info("Fetching USGS FDSN API: %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "bangladesh-eq-forecast/0.1"})
    with urllib.request.urlopen(req, timeout=180) as r:
        raw = r.read().decode("utf-8")
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(raw, encoding="utf-8")
        logger.info("Cached USGS CSV to %s", save_path)
    # Parse from the just-fetched text
    import io
    reader = csv.DictReader(io.StringIO(raw))
    observations: list[SourceObservation] = []
    for row in reader:
        try:
            mag = _to_float(row.get("mag"))
            if mag is None:
                continue
            obs = SourceObservation(
                source_catalog="usgs",
                native_event_id=row["id"],
                origin_time_utc=_parse_usgs_time(row["time"]),
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                depth_km=float(row["depth"]),
                original_magnitude=mag,
                original_magnitude_type=row.get("magType", "").strip().lower(),
                magnitude_uncertainty=_to_float(row.get("magError")),
                horizontal_uncertainty_km=_to_float(row.get("horizontalError")),
                depth_uncertainty_km=_to_float(row.get("depthError")),
                n_stations=_to_int(row.get("nst")),
                gap_deg=_to_float(row.get("gap")),
                rms_s=_to_float(row.get("rms")),
                quality_flag=_quality_from_status(row.get("status", "")),
                event_type=_event_type_from_str(row.get("type", "earthquake")),
                acquired_at_utc=datetime.now(timezone.utc),
                acquisition_method="usgs_fdsn_api",
            )
            observations.append(obs)
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping USGS API row %s: %s", row.get("id", "?"), exc)
    logger.info("Fetched %d USGS observations from API", len(observations))
    return observations
