"""Provenance helpers for constructing processing trails.

Every CanonicalEvent carries a ProvenanceTrail. This module provides
factory helpers so that each processing stage records a consistent step.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .schema import ProvenanceStep, ProvenanceTrail


def _now() -> datetime:
    return datetime.now(timezone.utc)


def step_acquired_local_file(
    source_catalog: str,
    file_path: str,
    n_events: int,
) -> ProvenanceStep:
    return ProvenanceStep(
        action="acquired_local_file",
        timestamp_utc=_now(),
        inputs={"source_catalog": source_catalog, "file_path": file_path},
        outputs={"n_observations": n_events},
        parameters={"acquisition_method": "local_file"},
        notes=f"Read {n_events} observations from {file_path}.",
    )


def step_acquired_api(
    source_catalog: str,
    endpoint: str,
    n_events: int,
) -> ProvenanceStep:
    return ProvenanceStep(
        action="acquired_api",
        timestamp_utc=_now(),
        inputs={"source_catalog": source_catalog, "endpoint": endpoint},
        outputs={"n_observations": n_events},
        parameters={"acquisition_method": "api"},
        notes=f"Fetched {n_events} observations from {endpoint}.",
    )


def step_canonical_matched(
    n_observations: int,
    n_canonical_events: int,
    time_window_s: float,
    spatial_window_km: float,
) -> ProvenanceStep:
    return ProvenanceStep(
        action="canonical_matched",
        timestamp_utc=_now(),
        inputs={"n_observations": n_observations},
        outputs={"n_canonical_events": n_canonical_events},
        parameters={
            "time_window_s": time_window_s,
            "spatial_window_km": spatial_window_km,
        },
        notes=(
            f"Matched {n_observations} observations into {n_canonical_events} "
            f"canonical events (time window +/- {time_window_s}s, "
            f"spatial window {spatial_window_km} km)."
        ),
    )


def step_origin_selected(
    canonical_id: str,
    chosen_catalog: str,
    rule: str,
) -> ProvenanceStep:
    return ProvenanceStep(
        action="origin_selected",
        timestamp_utc=_now(),
        inputs={"canonical_id": canonical_id},
        outputs={"chosen_catalog": chosen_catalog},
        parameters={"selection_rule": rule},
        notes=f"Origin chosen from {chosen_catalog} (rule: {rule}).",
    )


def step_magnitude_selected(
    canonical_id: str,
    chosen_catalog: str,
    original_type: str,
    mw_status: str,
    rule: str,
) -> ProvenanceStep:
    return ProvenanceStep(
        action="magnitude_selected",
        timestamp_utc=_now(),
        inputs={"canonical_id": canonical_id},
        outputs={
            "chosen_catalog": chosen_catalog,
            "original_type": original_type,
            "mw_status": mw_status,
        },
        parameters={"selection_rule": rule},
        notes=(
            f"Magnitude chosen from {chosen_catalog}; original type "
            f"{original_type}; Mw status: {mw_status}."
        ),
    )


def step_mw_derived(
    canonical_id: str,
    method: str,
    source: str,
    status: str,
) -> ProvenanceStep:
    return ProvenanceStep(
        action="mw_derived",
        timestamp_utc=_now(),
        inputs={"canonical_id": canonical_id},
        outputs={"method": method, "source": source, "status": status},
        parameters={},
        notes=f"Derived Mw via {method} ({source}); status {status}.",
    )


def step_mw_missing(
    canonical_id: str,
    reason: str,
) -> ProvenanceStep:
    return ProvenanceStep(
        action="mw_left_missing",
        timestamp_utc=_now(),
        inputs={"canonical_id": canonical_id},
        outputs={"mw": None},
        parameters={"reason": reason},
        notes=f"Mw left missing: {reason}",
    )


def step_completeness_filtered(
    method: str,
    mc: float,
    n_above: int,
    n_below: int,
) -> ProvenanceStep:
    return ProvenanceStep(
        action="completeness_filtered",
        timestamp_utc=_now(),
        inputs={},
        outputs={"n_above_mc": n_above, "n_below_mc": n_below},
        parameters={"method": method, "mc": mc},
        notes=f"Mc={mc:.2f} ({method}); {n_above} above, {n_below} below.",
    )


def step_declustered(
    method: str,
    n_mainshocks: int,
    n_aftershocks: int,
) -> ProvenanceStep:
    return ProvenanceStep(
        action="declustered",
        timestamp_utc=_now(),
        inputs={},
        outputs={"n_mainshocks": n_mainshocks, "n_aftershocks": n_aftershocks},
        parameters={"method": method},
        notes=f"Declustering ({method}): {n_mainshocks} mainshocks, "
              f"{n_aftershocks} aftershocks/foreshocks.",
    )


def new_trail() -> ProvenanceTrail:
    return ProvenanceTrail()
