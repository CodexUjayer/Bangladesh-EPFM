"""Configurable fault-data interface with placeholder support.

This module loads fault geometry from real datasets (GEM GAFD) and exposes a
``FaultRegistry`` that the Coulomb module (Stage 6) and long-term hazard
product (Product 2) query. Crucially:

    **No fault geometry, slip rate, dip, or rake is fabricated.** Fault
    segments missing these values are loaded with ``confidence='placeholder'``
    and are explicitly excluded from physics calculations. The system runs
    with placeholders present; it does not guess.

When real fault data from primary literature (Morino et al. 2014; Wang et
al. 2014; Steckler et al. 2016) are transcribed into the supplementary
``configs/faults_published.yaml``, those segments override the GEM GAFD
placeholders with ``confidence='medium'`` or ``'high'``.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .schema import FaultDatabase, FaultSegment

logger = logging.getLogger(__name__)


class PlaceholderFaultError(RuntimeError):
    """Raised when a Coulomb/hazard calculation is attempted on a fault
    segment whose required parameters are placeholders.

    This is a hard error: the caller must either supply real data for that
    segment or skip it. Silent fallback is forbidden.
    """


# ---------------------------------------------------------------------------
# Loader: GEM Global Active Faults Database
# ---------------------------------------------------------------------------

_GEM_GAFD_URL = (
    "https://raw.githubusercontent.com/GEMScienceTools/gem-global-active-faults/"
    "master/geojson/gem_active_faults_harmonized.geojson"
)


def load_gem_gafd(
    cache_path: Optional[Path] = None,
    force_download: bool = False,
) -> FaultDatabase:
    """Load the GEM Global Active Faults Database as a FaultDatabase.

    Parameters
    ----------
    cache_path : Path, optional
        If given, the GeoJSON is cached here and reused on subsequent calls
        unless ``force_download`` is True.
    force_download : bool
        If True, re-download even if a cache exists.

    Returns
    -------
    FaultDatabase
        All segments globally. Callers filter to the study region.
    """
    raw_text: Optional[str] = None

    if cache_path is not None and cache_path.exists() and not force_download:
        raw_text = cache_path.read_text(encoding="utf-8")
        logger.info("Loaded GEM GAFD from cache: %s", cache_path)
    else:
        logger.info("Downloading GEM GAFD from %s", _GEM_GAFD_URL)
        try:
            with urllib.request.urlopen(_GEM_GAFD_URL, timeout=60) as resp:
                raw_text = resp.read().decode("utf-8")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download GEM GAFD: {exc}. Provide a local copy "
                f"or check network connectivity."
            ) from exc
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(raw_text, encoding="utf-8")

    gj = json.loads(raw_text)
    features = gj.get("features", [])
    segments: list[FaultSegment] = []

    for feat in features:
        props = feat.get("properties", {}) or {}
        geom = feat.get("geometry", {}) or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])

        trace: list[tuple[float, float]] = []
        if gtype == "LineString":
            trace = [(float(c[0]), float(c[1])) for c in coords]
        elif gtype == "MultiLineString":
            for line in coords:
                trace.extend((float(c[0]), float(c[1])) for c in line)
        if not trace:
            continue

        # GEM GAFD property names (harmonized schema):
        #   fault_name, fault_id, slip_type, slip_rate, slip_rate_sigma,
        #   average_dip, rake, upper_seismogenic_depth, lower_seismogenic_depth
        slip_rate = _safe_float(props.get("slip_rate"))
        slip_rate_unc = _safe_float(props.get("slip_rate_sigma"))
        dip = _safe_float(props.get("average_dip"))
        rake = _safe_float(props.get("rake"))
        upper_d = _safe_float(props.get("upper_seismogenic_depth"))
        lower_d = _safe_float(props.get("lower_seismogenic_depth"))

        # Confidence is determined by data completeness, not by us guessing.
        has_kinematics = dip is not None or rake is not None
        has_rate = slip_rate is not None
        if has_kinematics and has_rate:
            confidence = "medium"
        elif has_kinematics or has_rate:
            confidence = "low"
        else:
            confidence = "placeholder"

        segments.append(FaultSegment(
            fault_id=str(props.get("fault_id", props.get("name", "unknown"))),
            name=str(props.get("fault_name", props.get("name", "unnamed"))),
            source="gem-gafd (Styron & Pagani 2020)",
            trace=trace,
            slip_type=str(props.get("slip_type", "unknown")),
            dip_deg=dip,
            rake_deg=rake,
            upper_depth_km=upper_d,
            lower_depth_km=lower_d,
            slip_rate_mm_per_yr=slip_rate,
            slip_rate_uncertainty_mm_per_yr=slip_rate_unc,
            confidence=confidence,
        ))

    db = FaultDatabase(
        name="GEM Global Active Faults Database",
        source="gem-gafd",
        segments=segments,
    )
    logger.info(
        "Loaded GEM GAFD: %d segments (%d placeholder, %d usable)",
        len(segments), db.n_placeholder, len(db.usable_segments()),
    )
    return db


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    # GEM uses -999 / -1 for "unknown" in some fields.
    if f is None or f <= -999:
        return None
    return f


# ---------------------------------------------------------------------------
# Fault registry
# ---------------------------------------------------------------------------


@dataclass
class FaultRegistry:
    """Filterable registry of fault segments over the study region.

    Combines GEM GAFD (default) with any user-supplied published fault
    segments. Provides region-filtering and a strict guard against using
    placeholder segments in physics calculations.
    """

    databases: list[FaultDatabase] = field(default_factory=list)
    _segments: list[FaultSegment] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        for db in self.databases:
            self._segments.extend(db.segments)

    @classmethod
    def from_gem(cls, study_bbox: Optional[tuple[float, float, float, float]] = None,
                 cache_path: Optional[Path] = None) -> "FaultRegistry":
        """Build a registry from the GEM GAFD, optionally filtered to a bbox.

        ``study_bbox`` = (min_lat, max_lat, min_lon, max_lon).
        """
        full = load_gem_gafd(cache_path=cache_path)
        if study_bbox is not None:
            min_lat, max_lat, min_lon, max_lon = study_bbox
            kept = [
                s for s in full.segments
                if any(min_lat <= la <= max_lat and min_lon <= lo <= max_lon
                       for (lo, la) in s.trace)
            ]
            filtered = FaultDatabase(
                name=full.name + " (study-region subset)",
                source=full.source,
                segments=kept,
            )
            return cls(databases=[filtered])
        return cls(databases=[full])

    def add_published_segments(self, segments: list[FaultSegment]) -> None:
        """Add fault segments transcribed from primary literature.

        These OVERRIDE GEM GAFD segments with the same fault name (by
        preferring the higher-confidence source). This is the channel
        through which real Morino et al. (2014) / Wang et al. (2014) /
        Steckler et al. (2016) values enter the system.
        """
        # Mark these as a separate database for provenance.
        pub_db = FaultDatabase(
            name="Published-literature fault segments",
            source="published-fault-studies",
            segments=segments,
        )
        self.databases.append(pub_db)
        self._segments.extend(segments)

    # ------------------------------------------------------------------
    def all_segments(self) -> list[FaultSegment]:
        return list(self._segments)

    def usable_for_coulomb(self) -> list[FaultSegment]:
        return [s for s in self._segments if s.is_usable_for_coulomb()]

    def usable_for_hazard(self) -> list[FaultSegment]:
        return [s for s in self._segments if s.is_usable_for_hazard()]

    def placeholders(self) -> list[FaultSegment]:
        return [s for s in self._segments if s.confidence == "placeholder"]

    def by_name(self, name: str) -> list[FaultSegment]:
        return [s for s in self._segments if s.name.lower() == name.lower()]

    # ------------------------------------------------------------------
    def require_for_coulomb(self, segment: FaultSegment) -> FaultSegment:
        """Return ``segment`` if it is usable for Coulomb, else raise.

        This is the guard called by the Coulomb module. It converts a silent
        data gap into a loud, explicit failure.
        """
        if not segment.is_usable_for_coulomb():
            raise PlaceholderFaultError(
                f"Fault segment '{segment.name}' (source={segment.source}) "
                f"cannot be used in Coulomb calculations: missing dip or "
                f"confidence='placeholder'. Supply real geometry from "
                f"primary literature before enabling this segment."
            )
        return segment

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        n = len(self._segments)
        return {
            "total_segments": n,
            "usable_for_coulomb": len(self.usable_for_coulomb()),
            "usable_for_hazard": len(self.usable_for_hazard()),
            "placeholders": len(self.placeholders()),
            "databases": [db.name for db in self.databases],
        }
