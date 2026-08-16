"""Declustereding: Gardner-Knopoff and Reasenberg methods.

Both methods are configurable. Outputs attach ``cluster_id`` and
``is_mainshock`` to each CanonicalEvent, plus a provenance step.

Gardner-Knopoff (1974): for each event, define a space-time window based
on magnitude; any event within the window of a LARGER (or equal and
earlier) event is classified as an aftershock (or foreshock if it
precedes). Mainshocks are the largest event of each cluster.

Reasenberg (1985): links events into clusters using an interaction zone
(radius proportional to the rupture size) and a time window that looks
both forward and backward. More permissive than GK; tends to keep more
independent events.

The rupture-length / window-size relations used here are the standard
Gardner & Knopoff (1974) / Knopoff (2000) empirical relations. They are
GLOBAL relations; no Bangladesh-specific adjustment is made (none is
published). This is documented as a limitation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..ingestion.canonical import _haversine_km
from ..ingestion.provenance import step_declustered
from ..ingestion.schema import CanonicalEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Empirical window relations (Gardner & Knopoff 1974; Knopoff 2000)
# ---------------------------------------------------------------------------


def _gk_distance_km(magnitude: float) -> float:
    """Gardner-Knopoff spatial window radius in km."""
    # Knopoff (2000) relation: r = 0.1238 * M + 0.983 (in deg) -> convert.
    # Gardner-Knopoff original: log10(r/km) = 0.439*M - 1.037 (for M>=6.5 a
    # different branch). We use the Knopoff-2000 single relation (degrees)
    # converted to km at ~mid-latitude, which is the common modern choice.
    r_deg = 0.1238 * magnitude + 0.983
    return r_deg * 111.0   # approximate km per degree


def _gk_time_days(magnitude: float) -> float:
    """Gardner-Knopoff temporal window in days."""
    # Knopoff (2000): log10(T/days) = 0.5*M - 0.052 (M < 6.5)
    #                       log10(T/days) = 0.5*M - 0.052 also used for all M
    # Original Gardner-Knopoff has a piecewise form. We use the Knopoff-2000
    # single relation; documented as global, no Bangladesh adjustment.
    return 10.0 ** (0.5 * magnitude - 0.052)


def _rupture_radius_km(magnitude: float) -> float:
    """Rough source radius (km) for Reasenberg, from Wells & Coppersmith
    (1994) surface-rupture-length relation (a global empirical relation)."""
    # log10(L) = -3.22 + 0.69*M  (all slip types, subsurface length)
    if magnitude < 4.0:
        return 1.0
    L = 10.0 ** (-3.22 + 0.69 * magnitude)   # km
    return max(L, 1.0)


# ---------------------------------------------------------------------------
# Gardner-Knopoff
# ---------------------------------------------------------------------------


@dataclass
class DeclusterResult:
    method: str
    n_total: int
    n_mainshocks: int
    n_aftershocks: int
    n_foreshocks: int
    n_clusters: int
    parameters: dict


def gardner_knopoff(
    events: list[CanonicalEvent],
    magnitude_field: str = "mw",   # "mw" or "original"
    min_magnitude: Optional[float] = None,
) -> DeclusterResult:
    """Gardner-Knopoff (1974) window declustering.

    Classifies each event as mainshock / aftershock / foreshock. Events are
    processed in order of decreasing magnitude; the largest unassigned event
    starts a new cluster and defines the window; all events within the
    space-time window (regardless of time direction) join that cluster and
    are flagged as aftershocks/foreshocks.

    Parameters
    ----------
    magnitude_field : 'mw' uses the derived Mw when available (falls back to
        original for events with missing Mw); 'original' uses the original
        magnitude. The choice is recorded in parameters.
    min_magnitude : optional floor; events below are flagged but not used
        as mainshock seeds.
    """
    def mag_of(e: CanonicalEvent) -> float:
        if magnitude_field == "mw" and e.mw is not None:
            return e.mw
        return e.original_magnitude

    # Sort by descending magnitude, then ascending time (deterministic).
    indexed = list(enumerate(events))
    indexed.sort(key=lambda iv: (-mag_of(iv[1]), iv[1].origin_time_utc))
    assigned = [False] * len(events)
    cluster_id = [-1] * len(events)
    is_main = [False] * len(events)
    role = [""] * len(events)   # "main" | "aftershock" | "foreshock"

    cluster_n = 0
    for idx, ev in indexed:
        if assigned[idx]:
            continue
        m = mag_of(ev)
        if min_magnitude is not None and m < min_magnitude:
            continue
        r_km = _gk_distance_km(m)
        t_days = _gk_time_days(m)
        cluster_id[idx] = cluster_n
        is_main[idx] = True
        role[idx] = "main"
        assigned[idx] = True
        # Assign neighbors within window
        for j, ev2 in enumerate(events):
            if assigned[j]:
                continue
            dt_days = (ev2.origin_time_utc - ev.origin_time_utc).total_seconds() / 86400.0
            if abs(dt_days) > t_days:
                continue
            dist = _haversine_km(
                ev.latitude, ev.longitude, ev2.latitude, ev2.longitude
            )
            if dist <= r_km:
                cluster_id[j] = cluster_n
                assigned[j] = True
                if ev2.origin_time_utc < ev.origin_time_utc:
                    role[j] = "foreshock"
                else:
                    role[j] = "aftershock"
        cluster_n += 1

    # Any unassigned events (below min_magnitude) become singletons.
    for j in range(len(events)):
        if not assigned[j]:
            cluster_id[j] = cluster_n
            is_main[j] = True
            role[j] = "main"
            cluster_n += 1

    # Write back onto events + provenance.
    for j, ev in enumerate(events):
        ev.cluster_id = cluster_id[j]
        ev.is_mainshock = bool(is_main[j])
        ev.declustering_method = "gardner_knopoff"
        ev.provenance.add(step_declustered(
            method="gardner_knopoff",
            n_mainshocks=sum(is_main),
            n_aftershocks=sum(1 for r in role if r != "main"),
        ))

    n_main = sum(1 for r in role if r == "main")
    n_aft = sum(1 for r in role if r == "aftershock")
    n_fore = sum(1 for r in role if r == "foreshock")
    return DeclusterResult(
        method="gardner_knopoff",
        n_total=len(events),
        n_mainshocks=n_main,
        n_aftershocks=n_aft,
        n_foreshocks=n_fore,
        n_clusters=cluster_n,
        parameters={
            "magnitude_field": magnitude_field,
            "min_magnitude": min_magnitude,
            "window_relation": "Knopoff (2000); global, no Bangladesh adjustment",
        },
    )


# ---------------------------------------------------------------------------
# Reasenberg (1985) — simplified cluster linking
# ---------------------------------------------------------------------------


def reasenberg(
    events: list[CanonicalEvent],
    magnitude_field: str = "mw",
    lookahead_days: float = 30.0,
    min_magnitude: Optional[float] = None,
) -> DeclusterResult:
    """Reasenberg (1985) cluster linking (simplified).

    Forward-linking: process events in time order. For each event not yet
    in a cluster, open a new cluster and look forward up to ``lookahead_days``
    for events within the interaction zone (sum of source radii). Extend
    the cluster's active time window when new members are added. The first
    (earliest) event in a cluster is the mainshock only if it is the
    largest; otherwise the largest event in the cluster is the mainshock.
    """
    def mag_of(e: CanonicalEvent) -> float:
        if magnitude_field == "mw" and e.mw is not None:
            return e.mw
        return e.original_magnitude

    order = sorted(range(len(events)), key=lambda i: events[i].origin_time_utc)
    cluster_id = [-1] * len(events)
    is_main = [False] * len(events)
    role = [""] * len(events)
    cluster_n = 0

    for i in order:
        if cluster_id[i] != -1:
            continue
        # open a new cluster
        members = [i]
        cluster_id[i] = cluster_n
        # interaction radius of the seed
        seed = events[i]
        seed_r = _rupture_radius_km(mag_of(seed))
        # extend forward
        t_end = seed.origin_time_utc
        from datetime import timedelta
        changed = True
        while changed:
            changed = False
            for j in order:
                if cluster_id[j] != -1:
                    continue
                ev2 = events[j]
                if (ev2.origin_time_utc - t_end).total_seconds() / 86400.0 > lookahead_days:
                    continue
                # check interaction with ANY current member
                for k in members:
                    ek = events[k]
                    rsum = seed_r + _rupture_radius_km(mag_of(ek))
                    # use the most recent member for the time-distance check
                    dt = abs((ev2.origin_time_utc - ek.origin_time_utc).total_seconds()) / 86400.0
                    if dt > lookahead_days:
                        continue
                    dist = _haversine_km(ek.latitude, ek.longitude,
                                         ev2.latitude, ev2.longitude)
                    if dist <= rsum * 10.0:   # Reasenberg uses ~10x source radius
                        cluster_id[j] = cluster_n
                        members.append(j)
                        t_end = max(t_end, ev2.origin_time_utc)
                        changed = True
                        break
        # mainshock = largest in cluster
        mags = [mag_of(events[k]) for k in members]
        main_local = members[int(np.argmax(mags))]
        for k in members:
            if k == main_local:
                is_main[k] = True
                role[k] = "main"
            elif events[k].origin_time_utc < events[main_local].origin_time_utc:
                role[k] = "foreshock"
            else:
                role[k] = "aftershock"
        cluster_n += 1

    for j, ev in enumerate(events):
        ev.cluster_id = cluster_id[j]
        ev.is_mainshock = bool(is_main[j])
        ev.declustering_method = "reasenberg"
        ev.provenance.add(step_declustered(
            method="reasenberg",
            n_mainshocks=sum(is_main),
            n_aftershocks=sum(1 for r in role if r != "main"),
        ))

    n_main = sum(1 for r in role if r == "main")
    n_aft = sum(1 for r in role if r == "aftershock")
    n_fore = sum(1 for r in role if r == "foreshock")
    return DeclusterResult(
        method="reasenberg",
        n_total=len(events),
        n_mainshocks=n_main,
        n_aftershocks=n_aft,
        n_foreshocks=n_fore,
        n_clusters=cluster_n,
        parameters={
            "magnitude_field": magnitude_field,
            "lookahead_days": lookahead_days,
            "min_magnitude": min_magnitude,
            "rupture_relation": "Wells & Coppersmith (1994); global",
        },
    )
