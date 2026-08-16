"""Quality control and catalog statistics.

Stage 3 scope: empirically determine catalog overlap, duplicate rate,
usable temporal coverage, spatial coverage, magnitude distributions,
magnitude-type distributions. No ETAS / ML / Coulomb here.

All statistics are computed from the ACTUAL ingested catalog. No number
is fabricated; missingness is reported explicitly.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from ..ingestion.schema import CanonicalEvent, QualityFlag

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Duplicate detection (within-source, before canonical matching)
# ---------------------------------------------------------------------------


def find_within_source_duplicates(
    observations,
    time_window_s: float = 30.0,
    spatial_window_km: float = 10.0,
) -> list[tuple]:
    """Find likely duplicates within a single source's observations.

    Returns a list of (obs_id_a, obs_id_b, dt_s, dist_km) tuples.
    Uses a tighter window than cross-source matching because within a
    single catalog, the same event should not appear twice unless there is
    a duplicate entry.
    """
    from ..ingestion.canonical import _haversine_km, _time_diff_s

    obs_sorted = sorted(observations, key=lambda o: o.origin_time_utc)
    dups = []
    n = len(obs_sorted)
    for i in range(n):
        for j in range(i + 1, n):
            dt = _time_diff_s(obs_sorted[i].origin_time_utc,
                              obs_sorted[j].origin_time_utc)
            if dt > time_window_s:
                break
            dist = _haversine_km(
                obs_sorted[i].latitude, obs_sorted[i].longitude,
                obs_sorted[j].latitude, obs_sorted[j].longitude,
            )
            if dist <= spatial_window_km:
                dups.append((
                    obs_sorted[i].observation_id,
                    obs_sorted[j].observation_id,
                    dt, dist,
                ))
    return dups


# ---------------------------------------------------------------------------
# Catalog statistics
# ---------------------------------------------------------------------------


@dataclass
class CatalogStats:
    """Empirical statistics of a canonical catalog. All numbers come from
    the actual ingested data; no fabrication."""

    n_events: int
    n_observations: int
    n_sources: int
    source_counts: dict   # source_catalog -> n_observations
    multi_source_events: int   # events with >=2 observations
    duplicate_rate_within_source: float   # fraction
    temporal_range_utc: tuple[datetime, datetime]
    temporal_span_years: float
    n_distinct_years: int
    events_per_year: dict   # year -> count
    spatial_bbox: tuple[float, float, float, float]
    lat_range: tuple[float, float]
    lon_range: tuple[float, float]
    depth_range_km: tuple[float, float]
    depth_mean: float
    depth_median: float
    magnitude_original_range: tuple[float, float]
    magnitude_type_counts: dict   # original_magnitude_type -> count
    mw_available_count: int
    mw_missing_count: int
    mw_missing_reasons: Counter
    quality_flag_counts: dict
    review_status_fraction: float
    # binned distributions
    magnitude_histogram: dict   # bin_center -> count (0.1 bins)
    yearly_counts: dict

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, tuple) and v and isinstance(v[0], datetime):
                d[k] = [x.isoformat() for x in v]
            elif isinstance(v, Counter):
                d[k] = dict(v)
            elif isinstance(v, datetime):
                d[k] = v.isoformat()
            else:
                d[k] = v
        return d


def compute_catalog_stats(
    events: list[CanonicalEvent],
    observations=None,
    within_source_duplicates: Optional[int] = None,
) -> CatalogStats:
    """Compute empirical statistics from a list of CanonicalEvent.

    ``observations`` and ``within_source_duplicates`` are passed when
    available so the duplicate rate can be reported; if None, the
    within-source duplicate rate is reported as not-computed (None).
    """
    n_events = len(events)
    n_obs = sum(len(e.observations) for e in events)
    source_counts = Counter()
    for e in events:
        for o in e.observations:
            source_counts[o.source_catalog] += 1
    multi_source = sum(1 for e in events if e.n_sources >= 2)

    times = [e.origin_time_utc for e in events]
    t_min, t_max = min(times), max(times)
    span_years = (t_max - t_min).total_seconds() / (365.25 * 86400)

    years = [t.year for t in times]
    yearly = Counter(years)
    n_distinct_years = len(yearly)

    lats = [e.latitude for e in events]
    lons = [e.longitude for e in events]
    depths = [e.depth_km for e in events]
    mags = [e.original_magnitude for e in events]
    magtypes = Counter(e.original_magnitude_type for e in events)

    n_mw = sum(1 for e in events if e.mw is not None)
    n_missing = n_events - n_mw
    missing_reasons: Counter = Counter()
    for e in events:
        if e.mw is None:
            # find the mw_left_missing step
            for s in e.provenance.steps:
                if s.action == "mw_left_missing":
                    missing_reasons[s.parameters.get("reason", "unspecified")] += 1
                    break

    qflags = Counter()
    reviewed = 0
    for e in events:
        for o in e.observations:
            qflags[o.quality_flag.value] += 1
            if o.quality_flag == QualityFlag.REVIEWED:
                reviewed += 1
    review_frac = reviewed / n_obs if n_obs else 0.0

    # magnitude histogram (0.1-unit bins)
    mag_hist = Counter()
    for m in mags:
        b = round(math.floor(m * 10) / 10.0, 1)
        mag_hist[b] += 1

    dup_rate = (
        within_source_duplicates / n_obs
        if (within_source_duplicates is not None and n_obs) else None
    )

    depths_arr = np.array(depths)
    return CatalogStats(
        n_events=n_events,
        n_observations=n_obs,
        n_sources=len(source_counts),
        source_counts=dict(source_counts),
        multi_source_events=multi_source,
        duplicate_rate_within_source=dup_rate if dup_rate is not None else None,
        temporal_range_utc=(t_min, t_max),
        temporal_span_years=span_years,
        n_distinct_years=n_distinct_years,
        events_per_year=dict(yearly),
        spatial_bbox=(min(lats), max(lats), min(lons), max(lons)),
        lat_range=(min(lats), max(lats)),
        lon_range=(min(lons), max(lons)),
        depth_range_km=(min(depths), max(depths)),
        depth_mean=float(np.mean(depths_arr)),
        depth_median=float(np.median(depths_arr)),
        magnitude_original_range=(min(mags), max(mags)),
        magnitude_type_counts=dict(magtypes),
        mw_available_count=n_mw,
        mw_missing_count=n_missing,
        mw_missing_reasons=missing_reasons,
        quality_flag_counts=dict(qflags),
        review_status_fraction=review_frac,
        magnitude_histogram=dict(mag_hist),
        yearly_counts=dict(yearly),
    )


# ---------------------------------------------------------------------------
# Catalog overlap (cross-source) — only meaningful with >=2 sources
# ---------------------------------------------------------------------------


@dataclass
class OverlapStats:
    """Cross-catalog overlap statistics."""

    n_sources: int
    n_observations_total: int
    n_canonical_events: int
    n_multi_source_events: int
    overlap_fraction: float   # multi_source / canonical_events
    mean_observations_per_event: float
    per_source_event_counts: dict   # source -> n observations
    pairwise_overlap: dict   # "src1|src2" -> n shared events
    note: str = ""

    def to_dict(self) -> dict:
        return self.__dict__


def compute_overlap_stats(events: list[CanonicalEvent]) -> OverlapStats:
    """Compute cross-source overlap. With a single source, overlap is 0."""
    n_events = len(events)
    n_obs = sum(len(e.observations) for e in events)
    sources = sorted({o.source_catalog for e in events for o in e.observations})
    multi = sum(1 for e in events if e.n_sources >= 2)
    per_source = Counter()
    for e in events:
        for o in e.observations:
            per_source[o.source_catalog] += 1

    # pairwise overlap: for each pair of sources, count events where both appear
    pairwise: dict[str, int] = {}
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            s1, s2 = sources[i], sources[j]
            cnt = 0
            for e in events:
                cats = {o.source_catalog for o in e.observations}
                if s1 in cats and s2 in cats:
                    cnt += 1
            pairwise[f"{s1}|{s2}"] = cnt

    note = (
        "Single-source catalog: cross-source overlap not applicable."
        if len(sources) < 2
        else ""
    )
    return OverlapStats(
        n_sources=len(sources),
        n_observations_total=n_obs,
        n_canonical_events=n_events,
        n_multi_source_events=multi,
        overlap_fraction=(multi / n_events) if n_events else 0.0,
        mean_observations_per_event=(n_obs / n_events) if n_events else 0.0,
        per_source_event_counts=dict(per_source),
        pairwise_overlap=pairwise,
        note=note,
    )
