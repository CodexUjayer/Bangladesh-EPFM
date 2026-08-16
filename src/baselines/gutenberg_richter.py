"""Gutenberg-Richter model fitting by maximum likelihood.

Fits log10 N(M >= m) = a - b*m using the Aki (1965) / Utsu (1965) MLE
estimator (NOT a visual line fit), with Shi-Bolt (1982) uncertainty and
bootstrap confidence intervals.

Reports:
  - b, a, standard errors, 95% CIs
  - sample size, fitting threshold (Mc), magnitude range used
  - sensitivity to Mc assumptions (fit under M>=4.0, 4.5, 5.0)

CRITICAL: We do NOT pick the threshold that gives the most attractive
b-value. We report all three scenarios as a sensitivity analysis, with the
Stage 3 caveat that Mc is a working range (M3.5-4.5), not a validated
threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..ingestion.schema import CanonicalEvent
from .uncertainty import bootstrap_bvalue_ci


@dataclass
class GRResult:
    """Gutenberg-Richter fit result for one threshold."""

    threshold: float                       # Mc used for the fit
    bin_width: float
    b_mle: float
    b_sigma_shibolt: float
    b_ci_lower: float                      # bootstrap 95%
    b_ci_upper: float
    a_value: float                         # log10 N at Mc
    a_sigma: float
    n_events_used: int
    magnitude_range: tuple[float, float]   # (min above Mc, max observed)
    mean_magnitude_above_mc: float
    scale_label: str
    notes: list[str] = field(default_factory=list)

    def to_row(self) -> dict:
        return {
            "model": "gutenberg_richter_mle",
            "threshold_Mc": self.threshold,
            "b_mle": round(self.b_mle, 4),
            "b_sigma_shibolt": round(self.b_sigma_shibolt, 4),
            "b_ci95_lower": round(self.b_ci_lower, 4),
            "b_ci95_upper": round(self.b_ci_upper, 4),
            "a_value_at_Mc": round(self.a_value, 4),
            "a_sigma": round(self.a_sigma, 4),
            "n_events_used": self.n_events_used,
            "magnitude_range": f"{self.magnitude_range[0]:.2f}-{self.magnitude_range[1]:.2f}",
            "scale_label": self.scale_label,
            "notes": "; ".join(self.notes),
        }

    def n_predicted_above(self, m: float) -> float:
        """GR-predicted number of events above magnitude m."""
        return 10.0 ** (self.a_value - self.b_mle * m)

    def rate_predicted_above(self, m: float, exposure_years: float) -> float:
        """GR-predicted annual rate above magnitude m."""
        if exposure_years <= 0:
            return float("nan")
        return self.n_predicted_above(m) / exposure_years


def fit_gutenberg_richter(
    events: list[CanonicalEvent],
    mc: float,
    bin_width: float = 0.1,
    exposure_years: Optional[float] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> GRResult:
    """Fit Gutenberg-Richter by MLE at a given Mc.

    Uses Aki-Utsu MLE:
        b = log10(e) / (<M> - (Mc - dM/2))
    Shi-Bolt (1982) sigma:
        sigma_b = 2.30 * b^2 * sigma_M / sqrt(N-1)
    Bootstrap 95% CI on b (nonparametric, fixed seed).
    a-value: log10(N) + b*Mc, with delta-method sigma.
    """
    # Magnitude selection: prefer Mw, else original (documented in scale_label).
    mws = [e.mw for e in events if e.mw is not None]
    if len(mws) >= 0.5 * len(events):
        mags = np.array(mws)
        scale_label = "Mw (derived/authoritative; events with missing Mw excluded)"
    else:
        mags = np.array([e.original_magnitude for e in events])
        scale_label = "original_magnitude (MIXED types; see Stage 3 report)"

    above = mags[mags >= mc - bin_width / 2]
    n = len(above)

    notes = []
    if n < 20:
        notes.append(f"Small sample (N={n}); b-value unreliable (need >=20).")

    if n < 20:
        # Return a degenerate result with NaNs; the report will flag it.
        return GRResult(
            threshold=mc, bin_width=bin_width,
            b_mle=float("nan"), b_sigma_shibolt=float("nan"),
            b_ci_lower=float("nan"), b_ci_upper=float("nan"),
            a_value=float("nan"), a_sigma=float("nan"),
            n_events_used=n,
            magnitude_range=(float(np.min(above)) if n > 0 else float("nan"),
                             float(np.max(above)) if n > 0 else float("nan")),
            mean_magnitude_above_mc=float(np.mean(above)) if n > 0 else float("nan"),
            scale_label=scale_label, notes=notes,
        )

    mean_m = float(np.mean(above))
    denom = mean_m - (mc - bin_width / 2)
    if denom <= 0:
        notes.append("Non-positive denominator in MLE; b not estimable.")
        b = float("nan")
        sigma_b = float("nan")
    else:
        b = math.log10(math.e) / denom
        std_m = float(np.std(above, ddof=1))
        sigma_b = 2.30 * b * b * std_m / math.sqrt(n - 1)

    # a-value at Mc: log10(N) + b*Mc  (so that 10^a = N at Mc)
    a = math.log10(n) + b * mc
    # delta-method sigma on a: sigma_a^2 = (1/(n ln10^2)) + (mc^2)(sigma_b^2)
    sigma_a = math.sqrt(1.0 / (n * (math.log(10) ** 2)) + (mc ** 2) * (sigma_b ** 2))

    # Bootstrap CI on b
    b_ci_lo, b_ci_hi = bootstrap_bvalue_ci(
        mags, mc, bin_width=bin_width,
        n_bootstrap=n_bootstrap, seed=seed,
    )

    mag_range = (float(np.min(above)), float(np.max(above)))

    return GRResult(
        threshold=mc, bin_width=bin_width,
        b_mle=float(b), b_sigma_shibolt=float(sigma_b),
        b_ci_lower=float(b_ci_lo), b_ci_upper=float(b_ci_hi),
        a_value=float(a), a_sigma=float(sigma_a),
        n_events_used=n,
        magnitude_range=mag_range,
        mean_magnitude_above_mc=mean_m,
        scale_label=scale_label, notes=notes,
    )


def fit_gr_multiple_thresholds(
    events: list[CanonicalEvent],
    thresholds: list[float] = None,
    bin_width: float = 0.1,
    exposure_years: Optional[float] = None,
) -> list[GRResult]:
    """Fit GR under multiple working Mc thresholds.

    Default thresholds [4.0, 4.5, 5.0] per the Stage 4 spec. We do NOT
    pick the 'best' one — all are reported as a sensitivity analysis.
    """
    if thresholds is None:
        thresholds = [4.0, 4.5, 5.0]
    results = []
    for mc in thresholds:
        r = fit_gutenberg_richter(events, mc, bin_width=bin_width,
                                  exposure_years=exposure_years)
        # Add a sensitivity note
        r.notes.append(
            f"Sensitivity scenario: Mc={mc}. Stage 3 established Mc is a "
            f"working range (M3.5-4.5), NOT a validated threshold."
        )
        results.append(r)
    return results
