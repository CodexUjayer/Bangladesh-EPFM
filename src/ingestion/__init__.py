"""Ingestion layer: schema, source registry, magnitude conversion, fault
interface, local-file readers, canonical selection, and provenance.

Stage 2 (corrected) + Stage 3 ingestion.
"""

from .schema import (
    CanonicalEvent,
    CatalogMetadata,
    ChosenMagnitude,
    ChosenOrigin,
    ConversionStatus,
    DataSourceClass,
    DerivedMw,
    EventType,
    FaultDatabase,
    FaultSegment,
    FocalMechanism,
    ProductClass,
    ProvenanceStep,
    ProvenanceTrail,
    QualityFlag,
    SourceObservation,
)
from .sources import DataSourceRegistry, DataSourceSpec
from .magnitude import (
    derive_mw,
    explain_no_mw,
    is_mw_family,
    list_available_relations,
)
from .faults import FaultRegistry, PlaceholderFaultError, load_gem_gafd
from .local import (
    fetch_usgs_fdsn_api,
    read_gcmt_ndk,
    read_iscgem_csv,
    read_usgs_csv,
    read_usgs_geojson,
)
from .canonical import (
    build_canonical_events,
    match_observations,
    select_canonical_magnitude,
    select_canonical_origin,
)
from . import provenance as provenance_steps

__all__ = [
    "CanonicalEvent",
    "CatalogMetadata",
    "ChosenMagnitude",
    "ChosenOrigin",
    "ConversionStatus",
    "DataSourceClass",
    "DerivedMw",
    "EventType",
    "FaultDatabase",
    "FaultSegment",
    "FocalMechanism",
    "ProductClass",
    "ProvenanceStep",
    "ProvenanceTrail",
    "QualityFlag",
    "SourceObservation",
    "DataSourceRegistry",
    "DataSourceSpec",
    "derive_mw",
    "explain_no_mw",
    "is_mw_family",
    "list_available_relations",
    "FaultRegistry",
    "PlaceholderFaultError",
    "load_gem_gafd",
    "fetch_usgs_fdsn_api",
    "read_gcmt_ndk",
    "read_iscgem_csv",
    "read_usgs_csv",
    "read_usgs_geojson",
    "build_canonical_events",
    "match_observations",
    "select_canonical_magnitude",
    "select_canonical_origin",
    "provenance_steps",
]
