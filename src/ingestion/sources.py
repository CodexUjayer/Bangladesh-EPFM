"""Data source registry with real, documented endpoints.

This module documents where each dataset comes from and how it should be
harmonized. After the Stage-2 corrections:

  - There is NO hard-coded 'ISC-GEM pre-2000, USGS post-2000' priority.
    Canonical origin/magnitude selection is quality-based and lives in
    ``canonical.py``. This registry only documents the sources and their
    access methods.
  - Local files are the PRIMARY ingestion path (see ``local.py``). Live
    APIs are OPTIONAL convenience.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .schema import DataSourceClass, ProductClass


@dataclass(frozen=True)
class DataSourceSpec:
    """Specification of a single data source."""

    source_id: str
    name: str
    provides: str
    magnitude_type: str
    temporal_coverage: str
    geographic_coverage: str
    limitations: str
    harmonization: str
    classification: DataSourceClass
    products: list[ProductClass]
    endpoint: str
    is_api: bool
    access_notes: str
    citation: Optional[str] = None
    native_id_prefix: str = ""
    # Local-file format this source ships as (for the local reader).
    local_format: Optional[str] = None  # "usgs_csv" | "usgs_geojson" | "iscgem_csv" | "gcmt_ndk"


class DataSourceRegistry:
    """Registry of all data sources used by the system."""

    def __init__(self) -> None:
        self._sources: dict[str, DataSourceSpec] = {}
        self._register_defaults()

    def register(self, spec: DataSourceSpec) -> None:
        self._sources[spec.source_id] = spec

    def get(self, source_id: str) -> DataSourceSpec:
        if source_id not in self._sources:
            raise KeyError(f"unknown data source: {source_id}")
        return self._sources[source_id]

    def all_sources(self) -> list[DataSourceSpec]:
        return list(self._sources.values())

    def by_classification(self, cls: DataSourceClass) -> list[DataSourceSpec]:
        return [s for s in self._sources.values() if s.classification == cls]

    def by_product(self, product: ProductClass) -> list[DataSourceSpec]:
        return [s for s in self._sources.values() if product in s.products]

    def _register_defaults(self) -> None:
        self.register(DataSourceSpec(
            source_id="usgs",
            name="USGS ComCat (FDSN web service / CSV / GeoJSON)",
            provides="Origin time, hypocenter, magnitude, magnitude type, location "
                     "error, review status, event type; moment-tensor products for "
                     "larger events (via the per-event detail API).",
            magnitude_type="Mixed: mww, mwr, mwb, mwc, mb, ml, md, ms. "
                           "Mw-family (mww/mwr/mwb/mwc) is authoritative Mw.",
            temporal_coverage="1973-present globally (M>=4.5); Bangladesh region "
                              "completeness improves from ~1990.",
            geographic_coverage="Global.",
            limitations="Pre-1990 sparse in region; depth poorly constrained; "
                        "magnitude type varies; automatic solutions for small events.",
            harmonization="Original magnitude preserved exactly. Mw derived only "
                          "when a validated relation applies (see magnitude.py). "
                          "Canonical selection is quality-based (see canonical.py); "
                          "no hard-coded source priority.",
            classification=DataSourceClass.REQUIRED,
            products=[ProductClass.SHORT_TERM_FORECAST, ProductClass.LONG_TERM_HAZARD],
            endpoint="https://earthquake.usgs.gov/fdsnws/event/1/query",
            is_api=True,
            access_notes="Public, no registration. CSV/GeoJSON output. PRIMARY "
                         "ingestion is from a locally saved CSV/GeoJSON file "
                         "(local.read_usgs_csv / read_usgs_geojson); the API is "
                         "an optional one-time acquisition convenience.",
            citation="USGS Earthquake Hazards Program.",
            native_id_prefix="usgs",
            local_format="usgs_csv",
        ))

        self.register(DataSourceSpec(
            source_id="isc-gem",
            name="ISC-GEM Global Instrumental Earthquake Catalogue",
            provides="Homogenized Mw 1904-present, original magnitudes preserved, "
                     "location, depth, uncertainty.",
            magnitude_type="Mw (homogenized). Original types preserved.",
            temporal_coverage="1904-present. Threshold ~M5.5 historic, ~M4 recent.",
            geographic_coverage="Global.",
            limitations="Higher threshold early; ~2-3 year update lag; not real-time.",
            harmonization="ISC-GEM Mw is authoritative (tagged 'mw_iscgem'). "
                          "Loaded as local CSV (local.read_iscgem_csv).",
            classification=DataSourceClass.REQUIRED,
            products=[ProductClass.SHORT_TERM_FORECAST, ProductClass.LONG_TERM_HAZARD],
            endpoint="http://www.isc.ac.uk/iscgem/download/",
            is_api=False,
            access_notes="Free for research, requires registration. CSV download. "
                         "PRIMARY ingestion: local CSV supplied by user.",
            citation="Storchak et al. (2013, 2015, 2021).",
            native_id_prefix="iscgem",
            local_format="iscgem_csv",
        ))

        self.register(DataSourceSpec(
            source_id="gcmt",
            name="Global Centroid Moment Tensor (GCMT) Project",
            provides="Centroid Mw, focal mechanism (strike/dip/rake), full moment "
                     "tensor. PRIMARY role: provide Mw + focal mechanism for M>=5.5 "
                     "events. (Hypocenter for ETAS comes from the best-supported "
                     "hypocenter source, NOT from the GCMT centroid.)",
            magnitude_type="Mw (centroid, authoritative for M>=5.5).",
            temporal_coverage="1976-present (M>=5.5).",
            geographic_coverage="Global.",
            limitations="Centroid != hypocenter; M5.5+ threshold; months of lag.",
            harmonization="GCMT Mw retained as authoritative 'mw'. Focal mechanism "
                          "attached to the matched canonical event. Loaded as local "
                          "NDK (local.read_gcmt_ndk).",
            classification=DataSourceClass.REQUIRED,
            products=[ProductClass.SHORT_TERM_FORECAST, ProductClass.LONG_TERM_HAZARD],
            endpoint="https://www.globalcmt.org/CMTsearch.html",
            is_api=False,
            access_notes="Public, .ndk text. PRIMARY ingestion: local NDK supplied "
                         "by user.",
            citation="Ekstrom, Nettles & Dziewonski (2012).",
            native_id_prefix="gcmt",
            local_format="gcmt_ndk",
        ))

        self.register(DataSourceSpec(
            source_id="isc",
            name="ISC Bulletin (full)",
            provides="All reported magnitudes, hypocenters, phase picks.",
            magnitude_type="All types, multi-agency.",
            temporal_coverage="1904-present.",
            geographic_coverage="Global.",
            limitations="Highly duplicative; requires dedup.",
            harmonization="Cross-check source; multiple magnitudes retained.",
            classification=DataSourceClass.HIGHLY_USEFUL,
            products=[ProductClass.SHORT_TERM_FORECAST, ProductClass.LONG_TERM_HAZARD],
            endpoint="http://www.isc.ac.uk/iscbulletin/search/",
            is_api=False,
            access_notes="Public web search; bulk extract.",
            citation="ISC, On-line Bulletin.",
            native_id_prefix="isc",
        ))

        self.register(DataSourceSpec(
            source_id="bmd",
            name="Bangladesh Meteorological Department bulletins",
            provides="Local Bangladesh reports.",
            magnitude_type="Variable.",
            temporal_coverage="Modern era.",
            geographic_coverage="Bangladesh + immediate surroundings.",
            limitations="Not publicly downloadable [R].",
            harmonization="If obtained: merged by matching; treated as ML unless "
                          "documented; NO Bangladesh ML->Mw relation exists.",
            classification=DataSourceClass.UNAVAILABLE,
            products=[ProductClass.SHORT_TERM_FORECAST, ProductClass.LONG_TERM_HAZARD],
            endpoint="Formal request to BMD.",
            is_api=False,
            access_notes="Status: NOT OBTAINED.",
            citation="BMD.",
            native_id_prefix="bmd",
        ))

        self.register(DataSourceSpec(
            source_id="historical",
            name="Published Bangladesh historical compilations",
            provides="Pre-instrumental locations/magnitudes/intensities.",
            magnitude_type="Mw (estimated) or MI; large uncertainty.",
            temporal_coverage="~810 BC per Alam & Dominey-Howes (2016).",
            geographic_coverage="Bangladesh + Bay of Bengal.",
            limitations="+/-0.5 Mw, +/-tens of km; not for ETAS calibration.",
            harmonization="Retained as-is with wide uncertainty; Product 2 only.",
            classification=DataSourceClass.REQUIRED,
            products=[ProductClass.LONG_TERM_HAZARD],
            endpoint="Published literature; manual transcription.",
            is_api=False,
            access_notes="Alam & Dominey-Howes (2016); Morino et al. (2014); "
                         "Steckler et al. (2016).",
            citation="Various.",
            native_id_prefix="hist",
        ))

        self.register(DataSourceSpec(
            source_id="gem-gafd",
            name="GEM Global Active Faults Database",
            provides="Digitized active fault traces; slip rate/type/dip where avail.",
            magnitude_type="N/A.",
            temporal_coverage="N/A.",
            geographic_coverage="Global; Bangladesh incomplete.",
            limitations="Slip rates often missing; simplified geometry.",
            harmonization="FaultDatabase; placeholders where data missing.",
            classification=DataSourceClass.HIGHLY_USEFUL,
            products=[ProductClass.SHORT_TERM_FORECAST, ProductClass.LONG_TERM_HAZARD],
            endpoint="https://github.com/GEMScienceTools/gem-global-active-faults",
            is_api=True,
            access_notes="Public, CC-BY-NC-SA 4.0. GeoJSON.",
            citation="Styron & Pagani (2020).",
            native_id_prefix="gafd",
        ))

        self.register(DataSourceSpec(
            source_id="published-fault-studies",
            name="Primary-literature fault geometry and slip rates",
            provides="Fault-specific geometry, dip, slip rate, recurrence.",
            magnitude_type="N/A.",
            temporal_coverage="N/A.",
            geographic_coverage="Per-fault.",
            limitations="Manual transcription per fault.",
            harmonization="Authoritative fault source.",
            classification=DataSourceClass.REQUIRED,
            products=[ProductClass.LONG_TERM_HAZARD],
            endpoint="Published literature.",
            is_api=False,
            access_notes="Morino 2014; Wang 2014; Steckler 2016; Bilham various.",
            citation="Various.",
            native_id_prefix="lit",
        ))

        self.register(DataSourceSpec(
            source_id="gnss-strain",
            name="GNSS / GPS strain rates (published)",
            provides="Station velocities, strain-rate field.",
            magnitude_type="N/A.",
            temporal_coverage="Campaign + continuous.",
            geographic_coverage="Bangladesh + surroundings.",
            limitations="Sparse; raw RINEX may need collaboration.",
            harmonization="Optional ML feature + moment budget.",
            classification=DataSourceClass.OPTIONAL,
            products=[ProductClass.LONG_TERM_HAZARD],
            endpoint="Published literature.",
            is_api=False,
            access_notes="Steckler et al. (2016).",
            citation="Steckler et al. (2016).",
            native_id_prefix="gnss",
        ))

        self.register(DataSourceSpec(
            source_id="plate-motion",
            name="GSRM v2",
            provides="Regional strain-rate, plate boundaries.",
            magnitude_type="N/A.",
            temporal_coverage="N/A.",
            geographic_coverage="Global.",
            limitations="Low resolution.",
            harmonization="Coarse prior.",
            classification=DataSourceClass.OPTIONAL,
            products=[ProductClass.LONG_TERM_HAZARD],
            endpoint="http://gsrm.unavco.org/",
            is_api=True,
            access_notes="Public.",
            citation="Kreemer et al. (2014).",
            native_id_prefix="gsrm",
        ))

        self.register(DataSourceSpec(
            source_id="usgs-global-tl",
            name="USGS catalogs for transfer-learning pretraining regions",
            provides="USGS FDSN queries for Japan, Taiwan, California, etc.",
            magnitude_type="As per USGS.",
            temporal_coverage="1973-present.",
            geographic_coverage="Per-region bboxes.",
            limitations="Domain similarity must be verified (Stage 8).",
            harmonization="Same as Bangladesh USGS query.",
            classification=DataSourceClass.REQUIRED,
            products=[ProductClass.SHORT_TERM_FORECAST],
            endpoint="https://earthquake.usgs.gov/fdsnws/event/1/query",
            is_api=True,
            access_notes="Same API; bboxes in configs/data_sources.yaml.",
            citation="USGS.",
            native_id_prefix="usgs",
        ))

        self.register(DataSourceSpec(
            source_id="magnitude-relations",
            name="Published magnitude conversion relations",
            provides="Scordilis (2006) global mb/MS->Mw coefficients.",
            magnitude_type="N/A.",
            temporal_coverage="N/A.",
            geographic_coverage="Global.",
            limitations="Sigma ~0.3-0.4; no Bangladesh-specific relations exist.",
            harmonization="magnitude.derive_mw; Mw left missing when no relation.",
            classification=DataSourceClass.REQUIRED,
            products=[ProductClass.SHORT_TERM_FORECAST, ProductClass.LONG_TERM_HAZARD],
            endpoint="Published literature.",
            is_api=False,
            access_notes="Scordilis (2006). No uncited regional relations.",
            citation="Scordilis (2006).",
            native_id_prefix="magrel",
        ))

        self.register(DataSourceSpec(
            source_id="boundaries",
            name="Natural Earth boundaries",
            provides="Country/coast vectors for plotting.",
            magnitude_type="N/A.",
            temporal_coverage="N/A.",
            geographic_coverage="Global.",
            limitations="N/A.",
            harmonization="Visualization only.",
            classification=DataSourceClass.OPTIONAL,
            products=[ProductClass.SHORT_TERM_FORECAST, ProductClass.LONG_TERM_HAZARD],
            endpoint="https://www.naturalearthdata.com/",
            is_api=True,
            access_notes="Public domain.",
            citation="Natural Earth.",
            native_id_prefix="ne",
        ))
