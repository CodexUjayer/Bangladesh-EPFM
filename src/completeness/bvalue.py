"""Gutenberg-Richter b-value estimation.

Implements:
  - Aki (1965) / Utsu (1965) maximum-likelihood b-value:
        b = log10(e) / ( <M> - (Mc - dM/2) )
    where dM is the binning width. This is the standard MLE for binned
    magnitudes.
  - Shi & Bolt (1982) uncertainty:
        sigma_b = 2.30 * b^2 * sigma_M  / sqrt(N+1)
    where sigma_M is the sample standard deviation of the magnitudes
    above Mc. (Some references use b/sqrt(N); we report Shi-Bolt as the
    primary and the simpler b/sqrt(N) as a cross-check.)
  - A least-squares (linear regression on log10 cumulative frequency) b
    is also reported for comparison, but the MLE is the recommended
    estimator because it is unbiased for binned data.

The b-value is estimated on the SAME magnitude scale used for Mc
(see completeness.select_magnitude_series). The scale is documented in
the output so the report can carry the caveat.

IMPORTANT: b-value is sensitive to Mc, magnitude scale, and declustering.
This module reports b for the full catalog AND for the declustered
catalog when both are passed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BValueEstimate:
    method: str
    b: float
    sigma_b: float
    a: Optional[float]          # GR a-value (log10 N at Mc), if computed
    mc: float
    n_events_used: int
    scale_label: str
    sigma_b_simple: Optional[float] = None   # b/sqrt(N) cross-check
    details: dict = field(default_factory=dict)
    warning: str = ""

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "b": self.b,
            "sigma_b": self.sigma_b,
            "a": self.a,
            "mc": self.mc,
            "n_events_used": self.n_events_used,
            "scale_label": self.scale_label,
            "sigma_b_simple": self.sigma_b_simple,
            "details": self.details,
            "warning": self.warning,
        }


def bvalue_mle(
    magnitudes: np.ndarray,
    mc: float,
    bin_width: float = 0.1,
    scale_label: str = "",
) -> BValueEstimate:
    """Aki-Utsu MLE b-value with Shi-Bolt uncertainty."""
    m = np.asarray(magnitudes)
    above = m[m >= mc - bin_width / 2]
    n = len(above)
    if n < 20:
        return BValueEstimate(
            method="MLE_Aki-Utsu",
            b=float("nan"), sigma_b=float("nan"), a=None,
            mc=mc, n_events_used=n, scale_label=scale_label,
            warning=f"only {n} events >= Mc; b-value unreliable (need >=20).",
        )
    mean_m = float(np.mean(above))
    denom = mean_m - (mc - bin_width / 2)
    if denom <= 0:
        return BValueEstimate(
            method="MLE_Aki-Utsu",
            b=float("nan"), sigma_b=float("nan"), a=None,
            mc=mc, n_events_used=n, scale_label=scale_label,
            warning="non-positive denominator in MLE.",
        )
    b = math.log10(math.e) / denom
    # Shi & Bolt (1982)
    std_m = float(np.std(above, ddof=1)) if n > 1 else 0.0
    sigma_b = 2.30 * b * b * std_m / math.sqrt(n - 1) if n > 1 else float("nan")
    sigma_simple = b / math.sqrt(n)
    # a-value: log10(N) at Mc
    a = math.log10(n) + b * mc
    return BValueEstimate(
        method="MLE_Aki-Utsu",
        b=float(b),
        sigma_b=float(sigma_b),
        a=float(a),
        mc=float(mc),
        n_events_used=n,
        scale_label=scale_label,
        sigma_b_simple=float(sigma_simple),
        details={"mean_M_above_Mc": mean_m, "std_M_above_Mc": std_m, "bin_width": bin_width},
    )


def bvalue_ls(
    magnitudes: np.ndarray,
    mc: float,
    bin_width: float = 0.1,
    scale_label: str = "",
) -> BValueEstimate:
    """Least-squares b-value (linear regression on log10 cumulative N).

    Reported for comparison only; MLE is preferred.
    """
    m = np.asarray(magnitudes)
    above = m[m >= mc - bin_width / 2]
    n = len(above)
    if n < 20:
        return BValueEstimate(
            method="LS_log10_cumulative",
            b=float("nan"), sigma_b=float("nan"), a=None,
            mc=mc, n_events_used=n, scale_label=scale_label,
            warning="insufficient events.",
        )
    # cumulative counts per bin
    m_min = math.floor(mc / bin_width) * bin_width
    m_max = math.ceil(above.max() / bin_width) * bin_width
    bins = np.arange(m_min, m_max + bin_width, bin_width)
    counts, edges = np.histogram(above, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cum = np.cumsum(counts[::-1])[::-1].astype(float)
    mask = cum > 0
    x = centers[mask]
    y = np.log10(cum[mask])
    if len(x) < 3:
        return BValueEstimate(
            method="LS_log10_cumulative",
            b=float("nan"), sigma_b=float("nan"), a=None,
            mc=mc, n_events_used=n, scale_label=scale_label,
            warning="insufficient bins for regression.",
        )
    # fit y = a - b*(x - mc)
    A = np.vstack([np.ones_like(x), -(x - mc)]).T
    coef, residuals, *_ = np.linalg.lstsq(A, y, rcond=None)
    a_fit, b_fit = coef
    sigma_b = float(np.sqrt(residuals[0] / (len(x) - 2))) if len(residuals) else float("nan")
    return BValueEstimate(
        method="LS_log10_cumulative",
        b=float(b_fit),
        sigma_b=float(sigma_b),
        a=float(a_fit + b_fit * mc),   # a at Mc
        mc=float(mc),
        n_events_used=n,
        scale_label=scale_label,
        details={"n_bins_fit": int(len(x))},
        warning="LS b-value is biased for binned data; MLE preferred.",
    )


def estimate_bvalue(
    magnitudes: np.ndarray,
    mc: float,
    bin_width: float = 0.1,
    scale_label: str = "",
) -> dict:
    """Return both MLE and LS b-value estimates."""
    return {
        "mle": bvalue_mle(magnitudes, mc, bin_width, scale_label),
        "ls": bvalue_ls(magnitudes, mc, bin_width, scale_label),
    }
