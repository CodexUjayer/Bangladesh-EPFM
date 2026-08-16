"""Magnitude conversion — derived Mw only when a validated relation applies.

USER CORRECTION (Stage 2 fix):

    Do NOT automatically apply Scordilis (2006), or any other generic
    conversion, to all mb/ms observations. Keep every original magnitude
    exactly as reported. Create a separate derived Mw field ONLY when a
    scientifically justified conversion relationship is available for the
    relevant magnitude range and region. For every conversion, record
    conversion_method, conversion_source, conversion_uncertainty. If no
    validated conversion is available, leave Mw as missing rather than
    inventing one.

This module therefore exposes a single function ``derive_mw(...)`` that
returns a ``DerivedMw`` (with full conversion provenance) when a validated
relation applies, or ``None`` when it does not. It NEVER silently invents a
value. There is no 'LENIENT' policy that inflates uncertainty as a stand-in
for a conversion — that was removed.

What counts as 'validated':
  - Mw-family magnitude types (mw, mww, mwr, mwb, mwc, mwp): AUTHORITATIVE.
    The value is retained as Mw; no regression applied.
  - mb -> Mw: Scordilis (2006) global relation, valid 3.5 <= mb <= 6.2,
    sigma = 0.41. This is a peer-reviewed GLOBAL relation. It is applied
    only when 3.5 <= mb <= 6.2. Outside that range, Mw is left MISSING.
  - MS -> Mw: Scordilis (2006), two segments (3.0-6.1 and 5.8-8.0).
  - ML, MD, mb_lg, MH, MN: NO validated global -> Mw relation exists.
    Bangladesh-specific relations DO NOT EXIST in the literature and are
    NOT invented. Mw is left MISSING.

This is especially important because magnitude conversion directly affects
Mc, Gutenberg-Richter b-value, ETAS productivity, and large-event
probability estimates. By keeping Mw missing for un-convertible types, we
ensure those events are handled honestly downstream (e.g. excluded from
Mw-based b-value estimation, or used only with the original type and a
documented caveat).

References
----------
Scordilis, E.M. (2006). Empirical global relations converting MS and mb to
    moment magnitude. J. Seismology 10, 225-236.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from .schema import ConversionStatus, DerivedMw

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Published relations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PublishedRelation:
    """A single peer-reviewed magnitude-conversion relation."""

    name: str
    input_type: str
    output_type: str
    slope: float
    intercept: float
    sigma: float
    valid_min: float
    valid_max: float
    citation: str

    def applies(self, m: float) -> bool:
        return self.valid_min <= m <= self.valid_max

    def convert(self, m: float) -> Optional[float]:
        if not self.applies(m):
            return None
        return self.slope * m + self.intercept


# Global relations from Scordilis (2006). Coefficients and sigma taken
# directly from the publication.
_PUBLISHED_RELATIONS: list[PublishedRelation] = [
    PublishedRelation(
        name="scordilis2006_mb_to_mw",
        input_type="mb",
        output_type="mw",
        slope=0.85,
        intercept=1.03,
        sigma=0.41,
        valid_min=3.5,
        valid_max=6.2,
        citation="Scordilis (2006), J. Seismology 10, 225-236",
    ),
    PublishedRelation(
        name="scordilis2006_ms_to_mw_low",
        input_type="ms",
        output_type="mw",
        slope=0.67,
        intercept=2.07,
        sigma=0.37,
        valid_min=3.0,
        valid_max=6.1,
        citation="Scordilis (2006), J. Seismology 10, 225-236",
    ),
    PublishedRelation(
        name="scordilis2006_ms_to_mw_high",
        input_type="ms",
        output_type="mw",
        slope=1.10,
        intercept=-0.57,
        sigma=0.28,
        valid_min=5.8,
        valid_max=8.0,
        citation="Scordilis (2006), J. Seismology 10, 225-236",
    ),
]


# Magnitude types that are ALREADY moment magnitude — retained as
# authoritative, no regression applied.
_MW_FAMILY: set[str] = {
    "mw", "mww", "mwr", "mwb", "mwc", "mwp",
    "mw_iscgem", "iscgem", "gcmt", "mwc_iscgem",
}

# Magnitude types for which a published global relation exists.
_CONVERTIBLE: set[str] = {"mb", "ms"}

# Magnitude types with NO validated global -> Mw relation.
_UNCONVERTIBLE: set[str] = {"ml", "md", "mb_lg", "mh", "mn"}


# ---------------------------------------------------------------------------
# Derive Mw (the only public entry point)
# ---------------------------------------------------------------------------


def derive_mw(
    original_magnitude: float,
    original_magnitude_type: str,
    source_catalog: str,
    source_uncertainty: Optional[float] = None,
    native_observation_id: str = "",
) -> Optional[DerivedMw]:
    """Derive a moment magnitude from an original magnitude.

    Returns a ``DerivedMw`` with full conversion provenance when a validated
    relation applies, or ``None`` when no validated conversion exists
    (leaving Mw MISSING rather than inventing one).

    Parameters
    ----------
    original_magnitude : float
        The magnitude value exactly as reported by the source.
    original_magnitude_type : str
        The magnitude type code (e.g. "mb", "mww", "ml").
    source_catalog : str
        Which catalog reported this magnitude (for provenance).
    source_uncertainty : float, optional
        Uncertainty reported by the source, in magnitude units.
    native_observation_id : str
        The observation id, recorded in the DerivedMw for traceability.

    Returns
    -------
    DerivedMw or None
        None means: no validated conversion available for this magnitude
        type / range; Mw is left missing. The reason is LOGGED but not
        stored on the event (callers can inspect via ``explain_no_mw``).
    """
    mtype = original_magnitude_type.lower().strip()

    # Case 1: already Mw-family -> AUTHORITATIVE. Retain as-is.
    if mtype in _MW_FAMILY:
        return DerivedMw(
            mw=original_magnitude,
            status=ConversionStatus.AUTHORITATIVE_MW,
            conversion_method="authoritative_mw_family",
            conversion_source=f"{source_catalog}:{mtype} (already moment magnitude)",
            conversion_uncertainty=source_uncertainty,
            input_magnitude=original_magnitude,
            input_magnitude_type=mtype,
            input_source_catalog=source_catalog,
            validity_range=None,
            notes=f"Original {mtype} from {source_catalog} retained as authoritative Mw.",
        )

    # Case 2: a published relation exists. Apply ONLY if in valid range.
    if mtype in _CONVERTIBLE:
        relations = [r for r in _PUBLISHED_RELATIONS if r.input_type == mtype]
        relations.sort(key=lambda r: r.valid_min)
        for rel in relations:
            if rel.applies(original_magnitude):
                mw = rel.convert(original_magnitude)
                unc = _combine_quadrature(source_uncertainty, rel.sigma)
                return DerivedMw(
                    mw=mw,
                    status=ConversionStatus.CONVERTED,
                    conversion_method=rel.name,
                    conversion_source=rel.citation,
                    conversion_uncertainty=unc,
                    input_magnitude=original_magnitude,
                    input_magnitude_type=mtype,
                    input_source_catalog=source_catalog,
                    validity_range=(rel.valid_min, rel.valid_max),
                    notes=f"Converted {mtype}={original_magnitude} from "
                          f"{source_catalog} using {rel.name}.",
                )
        # Relation exists but magnitude outside validity range -> Mw MISSING.
        logger.info(
            "Mw left missing: %s=%g from %s is outside the validity range of "
            "published %s->Mw relations (Scordilis 2006).",
            mtype, original_magnitude, source_catalog, mtype,
        )
        return None

    # Case 3: no validated relation (ML, MD, ...) -> Mw MISSING.
    if mtype in _UNCONVERTIBLE:
        logger.info(
            "Mw left missing: %s=%g from %s. No validated global %s->Mw "
            "relation exists; no Bangladesh-specific relation is published. "
            "Original magnitude retained; Mw is missing.",
            mtype, original_magnitude, source_catalog, mtype,
        )
        return None

    # Case 4: unrecognized type -> Mw MISSING.
    logger.info(
        "Mw left missing: unrecognized magnitude type '%s' (value %g from %s).",
        mtype, original_magnitude, source_catalog,
    )
    return None


def explain_no_mw(
    original_magnitude: float,
    original_magnitude_type: str,
) -> str:
    """Return a human-readable reason why Mw is missing for a given input.

    Useful for catalog audit reports.
    """
    mtype = original_magnitude_type.lower().strip()
    if mtype in _MW_FAMILY:
        return "Mw is available (authoritative)."
    if mtype in _CONVERTIBLE:
        rels = [r for r in _PUBLISHED_RELATIONS if r.input_type == mtype]
        for rel in rels:
            if rel.applies(original_magnitude):
                return "Mw is available (converted)."
        return (
            f"{mtype}={original_magnitude} is outside the validity range of "
            f"published {mtype}->Mw relations ({rels[0].valid_min}-"
            f"{rels[-1].valid_max}); Mw left missing."
        )
    if mtype in _UNCONVERTIBLE:
        return (
            f"No validated global {mtype}->Mw relation exists; no "
            f"Bangladesh-specific relation is published. Mw left missing."
        )
    return f"Unrecognized magnitude type '{mtype}'; Mw left missing."


def list_available_relations() -> list[PublishedRelation]:
    """Return all published relations currently implemented."""
    return list(_PUBLISHED_RELATIONS)


def is_mw_family(magnitude_type: str) -> bool:
    return magnitude_type.lower().strip() in _MW_FAMILY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _combine_quadrature(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None and b is None:
        return None
    if a is None:
        return float(b)
    if b is None:
        return float(a)
    return math.sqrt(a * a + b * b)
