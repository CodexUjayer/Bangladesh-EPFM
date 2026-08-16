"""Standardized schema: source observations, canonical events, provenance.

This module is the single source of truth for how earthquake data are
represented. It implements the user's Stage-2 corrections:

1. ORIGINAL MAGNITUDES ARE PRESERVED EXACTLY.
   Every catalog reports its own magnitude; we never overwrite it. The
   homogenized Mw is a SEPARATE derived field that is populated ONLY when a
   scientifically justified, validated conversion relation exists for the
   relevant magnitude range and (where applicable) region. When no
   validated conversion exists, ``DerivedMw`` is None — we leave Mw missing
   rather than inventing one.

2. MULTI-SOURCE OBSERVATIONS PER EVENT.
   A ``CanonicalEvent`` holds a list of ``SourceObservation`` records — one
   per catalog that reported the event. Canonical origin/magnitude are
   chosen by EXPLICIT quality rules (see canonical.py), never by a
   hard-coded "source A before year X, source B after" rule.

3. FULL PROVENANCE.
   Every derived field carries a ``ProvenanceTrail`` so the chain
   final_event -> source observations -> original catalog -> original
   magnitude -> transformation -> filtering -> declustering classification
   is fully traceable.

Design principles
-----------------
- Missing data are represented as ``None`` and surfaced, never silently
  imputed.
- No fabricated magnitudes, fault parameters, or catalog counts.
- The schema is pure-Python (no heavy deps) so it can be imported anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ProductClass(str, Enum):
    """The two scientifically distinct forecasting products."""

    SHORT_TERM_FORECAST = "short_term_forecast"
    LONG_TERM_HAZARD = "long_term_hazard"


class DataSourceClass(str, Enum):
    """Dataset availability classification (A/B/C/D)."""

    REQUIRED = "A_required"
    HIGHLY_USEFUL = "B_highly_useful"
    OPTIONAL = "C_optional"
    UNAVAILABLE = "D_unavailable_placeholder"


class QualityFlag(str, Enum):
    """Quality tier of an event origin, as reported by the source."""

    REVIEWED = "reviewed"
    AUTOMATIC = "automatic"
    HISTORICAL = "historical"
    MACROSEISMIC = "macroseismic"
    UNKNOWN = "unknown"


class EventType(str, Enum):
    EARTHQUAKE = "earthquake"
    EXPLOSION = "explosion"
    QUARRY_BLAST = "quarry_blast"
    ROCKBURST = "rockburst"
    UNKNOWN = "unknown"


class ConversionStatus(str, Enum):
    """Outcome of a magnitude-conversion attempt.

    The key change from the previous version: there is no 'LENIENT' fallback
    that invents a value. Mw is either derived from a validated relation
    (``CONVERTED``), retained as authoritative because the input was already
    Mw-family (``AUTHORITATIVE_MW``), or left MISSING with a documented
    reason (``NO_VALIDATED_RELATION`` / ``OUT_OF_RANGE`` / ``UNKNOWN_TYPE``).
    """

    AUTHORITATIVE_MW = "authoritative_mw"        # input was already Mw-family
    CONVERTED = "converted"                      # validated relation applied
    NO_VALIDATED_RELATION = "no_validated_relation"  # e.g. ML, MD -> no Mw
    OUT_OF_RANGE = "out_of_range"                # relation exists, value outside range
    UNKNOWN_TYPE = "unknown_type"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


@dataclass
class ProvenanceStep:
    """A single step in the processing history of an event or field.

    Examples of actions: 'acquired_from_usgs_csv', 'magnitude_converted',
    'completeness_filtered', 'declustered_mainshock', 'declustered_aftershock'.
    """

    action: str
    timestamp_utc: datetime
    inputs: dict = field(default_factory=dict)   # what was read
    outputs: dict = field(default_factory=dict)  # what was produced
    parameters: dict = field(default_factory=dict)  # config used
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "inputs": self.inputs,
            "outputs": self.outputs,
            "parameters": self.parameters,
            "notes": self.notes,
        }


@dataclass
class ProvenanceTrail:
    """Ordered list of provenance steps. Every CanonicalEvent carries one."""

    steps: list[ProvenanceStep] = field(default_factory=list)

    def add(self, step: ProvenanceStep) -> None:
        self.steps.append(step)

    def to_list(self) -> list[dict]:
        return [s.to_dict() for s in self.steps]

    def summary(self) -> str:
        return " -> ".join(s.action for s in self.steps)


# ---------------------------------------------------------------------------
# Focal mechanism
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FocalMechanism:
    """Double-couple focal mechanism (strike/dip/rake) plus moment tensor.

    Only populated when a real CMT / moment-tensor solution exists. Never
    fabricated.
    """

    strike_deg: float
    dip_deg: float
    rake_deg: float
    scalar_moment_Nm: Optional[float] = None
    mrr: Optional[float] = None
    mtt: Optional[float] = None
    mpp: Optional[float] = None
    mrt: Optional[float] = None
    mrp: Optional[float] = None
    mtp: Optional[float] = None
    source: Optional[str] = None  # "gcmt", "usgs-wphase", etc.

    def validate(self) -> None:
        if not (0.0 <= self.strike_deg <= 360.0):
            raise ValueError(f"strike out of range: {self.strike_deg}")
        if not (0.0 <= self.dip_deg <= 90.0):
            raise ValueError(f"dip out of range: {self.dip_deg}")
        if not (-180.0 <= self.rake_deg <= 180.0):
            raise ValueError(f"rake out of range: {self.rake_deg}")


# ---------------------------------------------------------------------------
# Source observation  (one catalog's raw observation of one event)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceObservation:
    """A single catalog's raw observation of an earthquake.

    ORIGINAL VALUES ARE PRESERVED EXACTLY. No magnitude conversion is
    applied at this level. ``original_magnitude`` and
    ``original_magnitude_type`` are exactly what the source reported.

    A CanonicalEvent holds one or more of these (one per reporting catalog).
    """

    # Identity / provenance
    source_catalog: str            # "usgs", "isc-gem", "gcmt", "isc", "bmd", "historical"
    native_event_id: str           # source-native id

    # Origin (exactly as reported)
    origin_time_utc: datetime      # timezone-aware UTC
    latitude: float
    longitude: float
    depth_km: float

    # Magnitude — ORIGINAL, never overwritten
    original_magnitude: float
    original_magnitude_type: str   # e.g. "mb", "mww", "ml", "ms"

    # Quality / uncertainty (None = not reported by source)
    magnitude_uncertainty: Optional[float] = None
    horizontal_uncertainty_km: Optional[float] = None
    depth_uncertainty_km: Optional[float] = None
    location_uncertainty_km: Optional[float] = None
    n_stations: Optional[int] = None
    n_phases: Optional[int] = None
    gap_deg: Optional[float] = None
    rms_s: Optional[float] = None
    quality_flag: QualityFlag = QualityFlag.UNKNOWN
    event_type: EventType = EventType.EARTHQUAKE

    # Optional rich fields (None = not available from this source)
    focal_mechanism: Optional[FocalMechanism] = None

    # When this observation was acquired
    acquired_at_utc: Optional[datetime] = None
    acquisition_method: str = "local_file"   # "local_file" | "usgs_fdsn_api" | "manual"

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not (-90.0 <= self.latitude <= 90.0):
            raise ValueError(f"latitude out of range: {self.latitude}")
        if not (-180.0 <= self.longitude <= 180.0):
            raise ValueError(f"longitude out of range: {self.longitude}")
        if not (0.0 <= self.depth_km <= 800.0):
            raise ValueError(f"depth out of range: {self.depth_km}")
        if not (-2.0 <= self.original_magnitude <= 10.5):
            raise ValueError(
                f"original_magnitude out of range: {self.original_magnitude}"
            )
        if self.origin_time_utc.tzinfo is None:
            raise ValueError("origin_time_utc must be timezone-aware (UTC)")
        if self.focal_mechanism is not None:
            self.focal_mechanism.validate()

    @property
    def observation_id(self) -> str:
        return f"{self.source_catalog}:{self.native_event_id}"

    @property
    def is_mw_family(self) -> bool:
        """True if the original magnitude is already a moment-magnitude variant."""
        return self.original_magnitude_type.lower().strip() in _MW_FAMILY


# Magnitude types that are ALREADY moment magnitude (or a direct estimate of
# it). These need no conversion; the value is retained as authoritative Mw.
_MW_FAMILY: set[str] = {
    "mw", "mww", "mwr", "mwb", "mwc", "mwp",
    "mw_iscgem", "iscgem", "gcmt", "mwc_iscgem",
}


# ---------------------------------------------------------------------------
# Derived Mw  (separate field, only when a validated conversion exists)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DerivedMw:
    """A derived moment magnitude, produced ONLY when a validated conversion
    relation applies to the relevant magnitude range and region.

    If no validated relation exists, ``DerivedMw`` is None on the
    CanonicalEvent — we leave Mw missing rather than inventing one.

    Provenance is mandatory: which relation, which source magnitude, what
    uncertainty, and the validity range used.
    """

    mw: float
    status: ConversionStatus
    conversion_method: str          # e.g. "scordilis2006_mb_to_mw" or "authoritative_mw_family"
    conversion_source: str          # citation, e.g. "Scordilis (2006)"
    conversion_uncertainty: Optional[float]  # combined sigma in Mw units
    input_magnitude: float
    input_magnitude_type: str
    input_source_catalog: str       # which observation the Mw was derived from
    validity_range: Optional[tuple[float, float]] = None  # (min, max) of relation
    notes: str = ""

    @property
    def is_authoritative(self) -> bool:
        return self.status == ConversionStatus.AUTHORITATIVE_MW

    @property
    def was_converted(self) -> bool:
        return self.status == ConversionStatus.CONVERTED


# ---------------------------------------------------------------------------
# Chosen canonical fields (with provenance of the choice)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChosenOrigin:
    """The canonical hypocenter chosen from among the source observations."""

    origin_time_utc: datetime
    latitude: float
    longitude: float
    depth_km: float
    chosen_from_catalog: str        # which source provided this origin
    chosen_from_observation_id: str
    selection_rule: str             # human-readable rule that picked it
    horizontal_uncertainty_km: Optional[float] = None
    depth_uncertainty_km: Optional[float] = None


@dataclass(frozen=True)
class ChosenMagnitude:
    """The canonical magnitude chosen for the event.

    This holds EITHER:
      - the original magnitude of a chosen observation (when no Mw derivation
        is possible, ``original_only=True`` and ``derived_mw=None``), OR
      - a derived Mw (when a validated conversion exists, ``derived_mw`` is
        populated and ``original_only=False``), OR
      - an authoritative Mw (when the chosen observation's original magnitude
        is already Mw-family; ``derived_mw`` populated with status
        AUTHORITATIVE_MW).
    """

    original_magnitude: float           # the chosen observation's original mag
    original_magnitude_type: str        # its original type
    chosen_from_catalog: str            # which source provided it
    chosen_from_observation_id: str
    selection_rule: str
    derived_mw: Optional[DerivedMw]     # None when no validated Mw available
    original_only: bool                 # True when Mw is missing on purpose


# ---------------------------------------------------------------------------
# Canonical event
# ---------------------------------------------------------------------------


@dataclass
class CanonicalEvent:
    """A matched group of source observations representing one earthquake.

    Holds:
      - all source observations (preserved exactly),
      - the chosen canonical origin and magnitude (with selection rules),
      - a derived Mw (separate, only when validated),
      - a full provenance trail.

    Downstream stages (completeness, declustering, ETAS) operate on this.
    """

    canonical_id: str                       # internal id, e.g. "ev_000123"
    observations: list[SourceObservation]   # one per reporting catalog
    chosen_origin: ChosenOrigin
    chosen_magnitude: ChosenMagnitude
    provenance: ProvenanceTrail = field(default_factory=ProvenanceTrail)

    # Pipeline bookkeeping (filled by later stages; None until then)
    completeness_magnitude: Optional[float] = None
    above_completeness: Optional[bool] = None
    cluster_id: Optional[int] = None
    is_mainshock: Optional[bool] = None
    declustering_method: Optional[str] = None

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def n_sources(self) -> int:
        return len(self.observations)

    @property
    def source_catalogs(self) -> list[str]:
        return sorted({o.source_catalog for o in self.observations})

    @property
    def mw(self) -> Optional[float]:
        """The best-estimate Mw, or None when no validated Mw is available.

        Returns the derived/authoritative Mw if present; otherwise None
        (NOT the original magnitude — mixing scales silently is forbidden).
        Callers that can tolerate a non-Mw magnitude must read
        ``chosen_magnitude.original_magnitude`` explicitly.
        """
        if self.chosen_magnitude.derived_mw is not None:
            return self.chosen_magnitude.derived_mw.mw
        return None

    @property
    def original_magnitude(self) -> float:
        return self.chosen_magnitude.original_magnitude

    @property
    def original_magnitude_type(self) -> str:
        return self.chosen_magnitude.original_magnitude_type

    @property
    def origin_time_utc(self) -> datetime:
        return self.chosen_origin.origin_time_utc

    @property
    def latitude(self) -> float:
        return self.chosen_origin.latitude

    @property
    def longitude(self) -> float:
        return self.chosen_origin.longitude

    @property
    def depth_km(self) -> float:
        return self.chosen_origin.depth_km

    @property
    def magnitude_uncertainty(self) -> Optional[float]:
        """Uncertainty on the best-estimate magnitude.

        If Mw was derived, returns the conversion uncertainty. If the
        original is authoritative Mw, returns the source-reported uncertainty.
        If Mw is missing, returns the source-reported uncertainty on the
        original magnitude (so downstream code can see that it is a
        non-Mw value with this uncertainty).
        """
        if self.chosen_magnitude.derived_mw is not None:
            return self.chosen_magnitude.derived_mw.conversion_uncertainty
        # Mw missing — return original's reported uncertainty
        for o in self.observations:
            if o.observation_id == self.chosen_magnitude.chosen_from_observation_id:
                return o.magnitude_uncertainty
        return None

    @property
    def focal_mechanism(self) -> Optional[FocalMechanism]:
        """Best available focal mechanism across observations (GCMT preferred)."""
        # Prefer GCMT, then USGS moment-tensor, then any.
        order = ["gcmt", "usgs", "isc-gem", "isc"]
        for src in order:
            for o in self.observations:
                if o.source_catalog == src and o.focal_mechanism is not None:
                    return o.focal_mechanism
        for o in self.observations:
            if o.focal_mechanism is not None:
                return o.focal_mechanism
        return None

    # ------------------------------------------------------------------
    def to_row(self) -> dict:
        """Flatten to a dict for downstream tabular processing / export.

        Provenance is preserved as a list of step dicts. Every original
        observation is preserved under ``observations``.
        """
        return {
            "canonical_id": self.canonical_id,
            "origin_time_utc": self.origin_time_utc.isoformat(),
            "latitude": self.latitude,
            "longitude": self.longitude,
            "depth_km": self.depth_km,
            # ORIGINAL magnitude (always present, never overwritten)
            "original_magnitude": self.original_magnitude,
            "original_magnitude_type": self.original_magnitude_type,
            "original_magnitude_source_catalog": self.chosen_magnitude.chosen_from_catalog,
            # DERIVED Mw (separate, may be None)
            "mw": self.mw,
            "mw_conversion_method": (
                self.chosen_magnitude.derived_mw.conversion_method
                if self.chosen_magnitude.derived_mw else None
            ),
            "mw_conversion_source": (
                self.chosen_magnitude.derived_mw.conversion_source
                if self.chosen_magnitude.derived_mw else None
            ),
            "mw_conversion_uncertainty": (
                self.chosen_magnitude.derived_mw.conversion_uncertainty
                if self.chosen_magnitude.derived_mw else None
            ),
            "mw_status": (
                self.chosen_magnitude.derived_mw.status.value
                if self.chosen_magnitude.derived_mw else "missing"
            ),
            "magnitude_uncertainty": self.magnitude_uncertainty,
            # Origin provenance
            "origin_source_catalog": self.chosen_origin.chosen_from_catalog,
            "origin_selection_rule": self.chosen_origin.selection_rule,
            "magnitude_selection_rule": self.chosen_magnitude.selection_rule,
            # Multi-source
            "n_sources": self.n_sources,
            "source_catalogs": ",".join(self.source_catalogs),
            # Uncertainties
            "horizontal_uncertainty_km": self.chosen_origin.horizontal_uncertainty_km,
            "depth_uncertainty_km": self.chosen_origin.depth_uncertainty_km,
            # Pipeline bookkeeping
            "completeness_magnitude": self.completeness_magnitude,
            "above_completeness": self.above_completeness,
            "cluster_id": self.cluster_id,
            "is_mainshock": self.is_mainshock,
            "declustering_method": self.declustering_method,
            # Full provenance trail (list of step dicts)
            "provenance": self.provenance.to_list(),
            # All original observations preserved
            "observations": [
                {
                    "observation_id": o.observation_id,
                    "source_catalog": o.source_catalog,
                    "native_event_id": o.native_event_id,
                    "origin_time_utc": o.origin_time_utc.isoformat(),
                    "latitude": o.latitude,
                    "longitude": o.longitude,
                    "depth_km": o.depth_km,
                    "original_magnitude": o.original_magnitude,
                    "original_magnitude_type": o.original_magnitude_type,
                    "magnitude_uncertainty": o.magnitude_uncertainty,
                    "quality_flag": o.quality_flag.value,
                    "has_focal_mechanism": o.focal_mechanism is not None,
                    "acquisition_method": o.acquisition_method,
                }
                for o in self.observations
            ],
        }

    def validate(self) -> None:
        if not self.observations:
            raise ValueError("CanonicalEvent must have >=1 observation")
        if self.chosen_origin.chosen_from_observation_id not in {
            o.observation_id for o in self.observations
        }:
            raise ValueError("chosen_origin references an unknown observation")
        if self.chosen_magnitude.chosen_from_observation_id not in {
            o.observation_id for o in self.observations
        }:
            raise ValueError("chosen_magnitude references an unknown observation")


# ---------------------------------------------------------------------------
# Fault data interface  (unchanged from before)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaultSegment:
    """A single fault segment with geometry and (optionally) kinematics."""

    fault_id: str
    name: str
    source: str
    trace: list[tuple[float, float]]
    slip_type: str
    dip_deg: Optional[float] = None
    rake_deg: Optional[float] = None
    upper_depth_km: Optional[float] = None
    lower_depth_km: Optional[float] = None
    slip_rate_mm_per_yr: Optional[float] = None
    slip_rate_uncertainty_mm_per_yr: Optional[float] = None
    recurrence_yr: Optional[float] = None
    max_magnitude: Optional[float] = None
    confidence: str = "placeholder"

    def is_usable_for_coulomb(self) -> bool:
        return self.dip_deg is not None and self.confidence != "placeholder"

    def is_usable_for_hazard(self) -> bool:
        return (
            self.confidence != "placeholder"
            and (self.slip_rate_mm_per_yr is not None or self.recurrence_yr is not None)
        )


@dataclass
class FaultDatabase:
    """Collection of fault segments with provenance."""

    name: str
    source: str
    segments: list[FaultSegment]
    n_placeholder: int = 0

    def __post_init__(self) -> None:
        self.n_placeholder = sum(
            1 for s in self.segments if s.confidence == "placeholder"
        )

    def usable_segments(self) -> list[FaultSegment]:
        return [s for s in self.segments if s.confidence != "placeholder"]

    def placeholder_segments(self) -> list[FaultSegment]:
        return [s for s in self.segments if s.confidence == "placeholder"]


# ---------------------------------------------------------------------------
# Catalog metadata
# ---------------------------------------------------------------------------


@dataclass
class CatalogMetadata:
    """Provenance metadata for a processed catalog."""

    name: str
    sources: list[str]
    time_range_utc: tuple[datetime, datetime]
    bbox: tuple[float, float, float, float]
    n_events: int
    magnitude_range: tuple[float, float]
    magnitude_homogenization: str
    declustering_method: Optional[str] = None
    completeness_method: Optional[str] = None
    mc_summary: Optional[float] = None
    created_at_utc: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    schema_version: str = "0.2.0"
    notes: str = ""
