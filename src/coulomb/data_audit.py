"""Coulomb data audit: classify every physical-data field as A/B/C/D.

USER REQUIREMENT (Stage 6): Before implementing real Coulomb forecasting,
determine exactly what validated data are available. For every field:

  A = directly observed/validated
  B = literature-derived
  C = engineering assumption
  D = unavailable

Do NOT silently convert C/D into A/B.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..ingestion.schema import FaultSegment
from ..ingestion.faults import FaultRegistry, load_gem_gafd


@dataclass
class FieldAudit:
    """Audit result for one physical-data field."""

    field_name: str
    classification: str    # "A", "B", "C", or "D"
    source: str            # provenance
    value_summary: str     # what value(s) are available, or "unavailable"
    notes: str = ""


@dataclass
class CoulombDataAudit:
    """Full data audit for Coulomb forecasting."""

    fields: list[FieldAudit] = field(default_factory=list)
    gcmt_available: bool = False
    gem_gafd_segments_in_region: int = 0
    gem_gafd_with_dip: int = 0
    gem_gafd_with_slip_rate: int = 0
    usgs_focal_mechanisms_available: int = 0   # count of events with FM products
    real_forecasting_enabled: bool = False
    blocking_gaps: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "fields": [
                {"field": f.field_name, "class": f.classification,
                 "source": f.source, "value_summary": f.value_summary, "notes": f.notes}
                for f in self.fields
            ],
            "gcmt_available": self.gcmt_available,
            "gem_gafd_segments_in_region": self.gem_gafd_segments_in_region,
            "gem_gafd_with_dip": self.gem_gafd_with_dip,
            "gem_gafd_with_slip_rate": self.gem_gafd_with_slip_rate,
            "usgs_focal_mechanisms_available": self.usgs_focal_mechanisms_available,
            "real_forecasting_enabled": self.real_forecasting_enabled,
            "blocking_gaps": self.blocking_gaps,
            "notes": self.notes,
        }


def audit_coulomb_data(
    gcmt_dir: Optional[Path] = None,
    gem_gafd_cache: Optional[Path] = None,
    usgs_focal_mechanism_count: int = 0,
    bbox: tuple[float, float, float, float] = (20.0, 28.0, 88.0, 96.0),
) -> CoulombDataAudit:
    """Run the full Coulomb data audit.

    Parameters
    ----------
    gcmt_dir : directory containing GCMT .ndk files (None = not supplied)
    gem_gafd_cache : path to cache the GEM GAFD GeoJSON
    usgs_focal_mechanism_count : number of USGS events with focal-mechanism
        products (determined externally by querying the USGS detail API)
    bbox : study region (min_lat, max_lat, min_lon, max_lon)
    """
    audit = CoulombDataAudit()

    # --- GCMT ---
    gcmt_files = []
    if gcmt_dir is not None and gcmt_dir.exists():
        gcmt_files = list(gcmt_dir.glob("*.ndk"))
    audit.gcmt_available = len(gcmt_files) > 0
    audit.fields.append(FieldAudit(
        field_name="focal_mechanisms (GCMT)",
        classification="A" if audit.gcmt_available else "D",
        source="GCMT NDK (local files)" if audit.gcmt_available else "not supplied",
        value_summary=f"{len(gcmt_files)} NDK files" if audit.gcmt_available else "0 files",
        notes="GCMT is the gold-standard global CMT catalog. Required for source focal mechanisms."
    ))

    # --- USGS focal mechanisms (from detail API products) ---
    audit.usgs_focal_mechanisms_available = usgs_focal_mechanism_count
    audit.fields.append(FieldAudit(
        field_name="focal_mechanisms (USGS)",
        classification="A" if usgs_focal_mechanism_count > 0 else "D",
        source="USGS ComCat focal-mechanism product (per-event detail API)",
        value_summary=f"{usgs_focal_mechanism_count} events with focal-mechanism products" if usgs_focal_mechanism_count > 0 else "0",
        notes="USGS focal-mechanism products provide strike/dip/rake for moderate-large events."
    ))

    # --- GEM GAFD ---
    try:
        db = load_gem_gafd(cache_path=gem_gafd_cache)
        min_lat, max_lat, min_lon, max_lon = bbox
        bd_faults = [s for s in db.segments
                     if any(min_lat <= la <= max_lat and min_lon <= lo <= max_lon
                            for (lo, la) in s.trace)]
        audit.gem_gafd_segments_in_region = len(bd_faults)
        audit.gem_gafd_with_dip = sum(1 for s in bd_faults if s.dip_deg is not None)
        audit.gem_gafd_with_slip_rate = sum(1 for s in bd_faults if s.slip_rate_mm_per_yr is not None)
        audit.fields.append(FieldAudit(
            field_name="fault_geometry (GEM GAFD traces)",
            classification="A" if len(bd_faults) > 0 else "D",
            source="GEM Global Active Faults Database (Styron & Pagani 2020)",
            value_summary=f"{len(bd_faults)} fault traces in region",
            notes="Surface-trace geometry (lon/lat vertices)."
        ))
        audit.fields.append(FieldAudit(
            field_name="fault_dip (GEM GAFD)",
            classification="A" if audit.gem_gafd_with_dip > 0 else "D",
            source="GEM GAFD",
            value_summary=f"{audit.gem_gafd_with_dip}/{len(bd_faults)} segments have dip",
            notes="Dip is REQUIRED for receiver-fault ΔCFS. GEM GAFD is geometry-only for Bangladesh."
        ))
        audit.fields.append(FieldAudit(
            field_name="fault_rake (GEM GAFD)",
            classification="A" if sum(1 for s in bd_faults if s.rake_deg is not None) > 0 else "D",
            source="GEM GAFD",
            value_summary=f"{sum(1 for s in bd_faults if s.rake_deg is not None)}/{len(bd_faults)} have rake",
            notes="Rake (slip direction) is REQUIRED for receiver-fault ΔCFS."
        ))
        audit.fields.append(FieldAudit(
            field_name="fault_slip_rate (GEM GAFD)",
            classification="A" if audit.gem_gafd_with_slip_rate > 0 else "D",
            source="GEM GAFD",
            value_summary=f"{audit.gem_gafd_with_slip_rate}/{len(bd_faults)} have slip rate",
            notes="Slip rate needed for long-term hazard, not directly for Coulomb ΔCFS."
        ))
    except Exception as e:
        audit.notes.append(f"GEM GAFD load failed: {e}")
        audit.fields.append(FieldAudit(
            field_name="fault_geometry (GEM GAFD)",
            classification="D", source="GEM GAFD",
            value_summary="load failed", notes=str(e)[:80]
        ))

    # --- Published Bangladesh fault studies (Morino, Wang, Steckler) ---
    audit.fields.append(FieldAudit(
        field_name="fault_geometry (published literature)",
        classification="D",
        source="Morino et al. 2014; Wang et al. 2014; Steckler et al. 2016",
        value_summary="NOT TRANSCRIBED — requires manual literature acquisition",
        notes="Primary-literature fault geometry would override GEM GAFD placeholders. Not currently loaded."
    ))

    # --- Elastic parameters ---
    audit.fields.append(FieldAudit(
        field_name="elastic_params (shear modulus, Poisson's ratio)",
        classification="C",
        source="engineering assumption (standard crustal values)",
        value_summary="μ=30 GPa, ν=0.25 (Okada 1992 defaults)",
        notes="No Bangladesh-specific elastic model available. Standard crustal values used; sensitivity tested."
    ))
    audit.fields.append(FieldAudit(
        field_name="effective_friction (μ')",
        classification="C",
        source="engineering assumption",
        value_summary="μ'=0.4 (King et al. 1994 typical; range 0.2-0.8 tested)",
        notes="No Bangladesh-specific friction data. Sensitivity analysis required."
    ))
    audit.fields.append(FieldAudit(
        field_name="skempton_coefficient (B)",
        classification="C",
        source="engineering assumption",
        value_summary="B=0.5 (typical crystalline crust; range 0.5-1.0)",
        notes="Pore-pressure coupling. No Bangladesh data."
    ))

    # --- Regional stress orientation ---
    audit.fields.append(FieldAudit(
        field_name="regional_stress_orientation",
        classification="D",
        source="not available",
        value_summary="No Bangladesh stress map in the World Stress Map database for this region",
        notes="If available, could be used to define 'optimal' receiver faults. Currently unavailable."
    ))

    # --- Receiver-fault orientations ---
    audit.fields.append(FieldAudit(
        field_name="receiver_fault_orientations",
        classification="D",
        source="not available",
        value_summary="No validated receiver-fault dataset for Bangladesh",
        notes="BLOCKING: receiver-fault geometry is required for ΔCFS. Without it, only stress-tensor components can be computed."
    ))

    # --- Coseismic slip distributions ---
    audit.fields.append(FieldAudit(
        field_name="coseismic_slip_distributions",
        classification="D",
        source="not available",
        value_summary="No finite-fault slip models for Bangladesh events in the catalog",
        notes="For M>7 events, finite-source would improve accuracy; point-source used as approximation."
    ))

    # --- Determine if real forecasting can be enabled ---
    # Real Coulomb forecasting requires BOTH:
    #   (a) source focal mechanisms (GCMT or USGS focal-mechanism products), AND
    #   (b) receiver-fault geometry (validated dip/rake) OR regional stress.
    source_fm_available = audit.gcmt_available or (usgs_focal_mechanism_count > 0)
    receiver_geometry_available = (
        audit.gem_gafd_with_dip > 0
        or any(f.field_name == "fault_geometry (published literature)" and f.classification in ("A", "B")
               for f in audit.fields)
    )

    if source_fm_available and receiver_geometry_available:
        audit.real_forecasting_enabled = True
        audit.notes.append("Real Coulomb forecasting ENABLED: source focal mechanisms and receiver geometry both available.")
    else:
        audit.real_forecasting_enabled = False
        if not source_fm_available:
            audit.blocking_gaps.append("source focal mechanisms (GCMT NDK not supplied; USGS FM products = 0)")
        if not receiver_geometry_available:
            audit.blocking_gaps.append(
                "receiver-fault geometry (GEM GAFD has traces but NO dip/rake; "
                "published literature not transcribed; regional stress unavailable)"
            )
        audit.notes.append(
            "Real Coulomb forecasting DISABLED due to data gaps. "
            "Mathematical prototype and unit tests are implemented; "
            "stress-tensor components can be computed from USGS source focal "
            "mechanisms on ASSUMED receiver faults (Class C) for diagnostic "
            "purposes only, NOT for validated Bangladesh forecasts."
        )

    return audit


def save_data_audit(audit: CoulombDataAudit, path: Path) -> None:
    """Save the data audit as CSV + JSON."""
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    # CSV
    with path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["field_name", "classification", "source", "value_summary", "notes"])
        for fld in audit.fields:
            w.writerow([fld.field_name, fld.classification, fld.source, fld.value_summary, fld.notes])
    # JSON (full audit)
    with path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(audit.to_dict(), f, indent=2, default=str)
