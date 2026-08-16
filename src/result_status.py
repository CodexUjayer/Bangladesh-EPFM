"""Formal RESULT STATUS system for the Bangladesh earthquake forecasting project.

Every scientific result has one of:
  VALIDATED     — robustly established under proper chronological evaluation
  PRELIMINARY   — established but with caveats; needs confirmation
  SENSITIVITY   — result of a sensitivity analysis, not a primary finding
  DIAGNOSTIC    — non-parametric diagnostic, not a forecast model
  DATA-LIMITED  — cannot be fully established due to missing data
  SUPERSEDED    — replaced by a corrected result; preserved for reproducibility
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class ResultStatus(str, Enum):
    VALIDATED = "VALIDATED"
    PRELIMINARY = "PRELIMINARY"
    SENSITIVITY = "SENSITIVITY"
    DIAGNOSTIC = "DIAGNOSTIC"
    DATA_LIMITED = "DATA-LIMITED"
    SUPERSEDED = "SUPERSEDED"


@dataclass
class ResultEntry:
    """One scientific result with its status and provenance."""
    stage: str
    result_id: str
    description: str
    status: ResultStatus
    evidence: str
    caveats: str
    supersedes: Optional[str] = None  # result_id this supersedes (if any)
    superseded_by: Optional[str] = None  # result_id that supersedes this (if any)


# The complete status manifest for Stages 1-7B (Phase A corrected)
RESULT_MANIFEST = [
    # Stage 3
    ResultEntry("Stage 3", "S3-CATALOG", "USGS catalog: 2,293 events M>=2.5 (floor M3.2), 1973-2024",
                ResultStatus.VALIDATED, "Actual downloaded USGS CSV", "USGS-only; BMD/ISC not available"),
    ResultEntry("Stage 3", "S3-MC", "Mc = working range M3.5-4.5 (not validated below M3.5)",
                ResultStatus.DATA_LIMITED, "MAXC/GFT/EMR/Stepp on M>=2.5 catalog",
                "USGS floor M3.2 prevents Mc validation below ~M3.5; BMD/ISC needed"),
    ResultEntry("Stage 3", "S3-DECLUSTER", "Gardner-Knopoff: 202 mainshocks; Reasenberg: 1603 mainshocks",
                ResultStatus.VALIDATED, "Both methods implemented and run", "Global window relations; no Bangladesh-specific adjustment"),

    # Stage 4
    ResultEntry("Stage 4", "S4-POISSON", "Temporal Poisson rates for M>=4.5-7.0",
                ResultStatus.VALIDATED, "Exact Poisson CIs; expanding-window backtest",
                "Mc uncertainty propagates into rate uncertainty"),
    ResultEntry("Stage 4", "S4-GR", "Gutenberg-Richter b-value: 0.95 (Mc=4.5 working)",
                ResultStatus.SENSITIVITY, "MLE Aki-Utsu; b=0.49 (Mc=4.0 biased), b=1.43 (Mc=5.0)",
                "Mc-sensitive; b at Mc=4.0 is biased by truncation"),
    ResultEntry("Stage 4", "S4-SPATIAL", "Spatial Poisson: 64 cells, per-cell rates",
                ResultStatus.VALIDATED, "1-degree grid; Jeffreys CIs for low-stat cells",
                "Coarse grid; 20/64 cells low-statistics"),

    # Stage 5
    ResultEntry("Stage 5", "S5-ETAS-K0", "Locally fitted ETAS: K->0 (no triggering detected in-sample)",
                ResultStatus.PRELIMINARY, "MLE with base-10 corrected formulation; multi-start optimizer",
                "Phase A re-run needed; base-10 bug was confounding previous result"),
    ResultEntry("Stage 5", "S5-OMORI", "Omori diagnostic: R(Δt)=22x (M>=5), 377x (M>=6) at short lags",
                ResultStatus.DIAGNOSTIC, "Non-parametric rate-ratio over log time bins",
                "Catalog DOES exhibit post-mainshock temporal clustering"),
    ResultEntry("Stage 5", "S5-SPATIAL-DIAG", "Spatial diagnostic: 4x concentration at <50km for M>=6",
                ResultStatus.DIAGNOSTIC, "Post-mainshock distance distribution vs background",
                "M>=5 shows only 1.87x (weak)"),
    ResultEntry("Stage 5", "S5-DEPTH", "Depth analysis: shallow CV_IET=2.48, deep CV_IET=1.20",
                ResultStatus.DIAGNOSTIC, "Per-depth inter-event-time CV",
                "All depth groups show K->0 in per-depth ETAS fits"),
    ResultEntry("Stage 5", "S5-ETAS-VS-POISSON", "ETAS does not beat uniform Poisson (corrected)",
                ResultStatus.PRELIMINARY, "Chronological backtest with per-origin Poisson rate",
                "NOT tested vs spatial Poisson; base-10 bug was confounding"),

    # Stage 6
    ResultEntry("Stage 6", "S6-COULOMB", "Coulomb forecasting DISABLED",
                ResultStatus.DATA_LIMITED, "Data audit: 0/42 GEM faults have dip/rake; no GCMT",
                "Mathematical prototype validated with synthetic geometry; no Bangladesh forecast"),

    # Stage 7
    ResultEntry("Stage 7", "S7-ML-VS-UNIFORM", "ML beats uniform Poisson (8/8 configs)",
                ResultStatus.SUPERSEDED, "Stage 7 report",
                "Artifact of comparing against uniform Poisson; superseded by Stage 7B",
                superseded_by="S7B-ML-VS-SPATIAL"),
    ResultEntry("Stage 7B", "S7B-ML-VS-SPATIAL", "ML does NOT beat Spatial Poisson (0/8, CIs exclude zero)",
                ResultStatus.VALIDATED, "Block bootstrap over 9 origins; 500 resamples",
                "Limited to 2 models (GB, logistic) x 4 configs x 9 origins; spatial holdout not yet implemented"),

    # Phase A corrections
    ResultEntry("Phase A", "PA-ETAS-BASE10", "ETAS productivity corrected to base-10",
                ResultStatus.VALIDATED, "10^{alpha(M-Mc)} per research report",
                "Previous exp(alpha*(M-Mc)) was base-e, factor ln(10) mismatch"),
    ResultEntry("Phase A", "PA-ETAS-BG", "ETAS background uses Gardner-Knopoff declustering",
                ResultStatus.VALIDATED, "Stage 3 GK declustered mainshocks for KDE background",
                "Previous Mc+0.5 proxy was not declustering"),
    ResultEntry("Phase A", "PA-BASE-RATE", "Stage 7B base-rate check corrected",
                ResultStatus.VALIDATED, "Mean sum(cell P) vs mean observed regional rate",
                "Previous single-origin binary comparison was metrically wrong"),
    ResultEntry("Phase A", "PA-S7-SUPERSEDED", "Stage 7 marked SUPERSEDED",
                ResultStatus.SUPERSEDED, "Stage 7 report header updated",
                "Preserved for reproducibility; not a valid test of ML skill"),
]


def get_status(result_id: str) -> Optional[ResultEntry]:
    for r in RESULT_MANIFEST:
        if r.result_id == result_id:
            return r
    return None


def all_results() -> list[ResultEntry]:
    return list(RESULT_MANIFEST)
