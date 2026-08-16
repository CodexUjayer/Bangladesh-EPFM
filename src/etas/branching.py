"""Branching ratio computation and plausibility check.

The branching ratio n is the expected number of direct aftershocks per
event, averaged over the magnitude distribution:

    n = ∫ K · exp[α(M - Mc)] · pdf(M | M >= Mc) dM
      = K · E[exp(α(M - Mc))]

Under a truncated (above Mc) exponential GR magnitude distribution with
parameter β = b · ln(10), this expectation has a closed form:

    E[exp(α(M - Mc))] = β / (β - α)    for α < β

If α >= β, the expectation diverges (n → ∞), which corresponds to an
explosive (non-stationary) Hawkes process — physically implausible for a
stationary catalog. This is a critical diagnostic.

We compute n both:
  (1) analytically using the GR β (when α < β), and
  (2) empirically by averaging K·exp(α(M_i - Mc)) over the catalog.

The two should agree closely. Disagreement suggests the magnitude
distribution deviates from GR, or that α is poorly identified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BranchingResult:
    """Branching ratio assessment."""

    n_analytic: float          # using GR β (None if α >= β -> explosive)
    n_empirical: float         # averaging over catalog
    b_value: float
    beta: float                # b * ln(10)
    alpha: float
    explosive: bool            # True if α >= β (non-stationary)
    plausible: bool            # False if n > 1 (supercritical) or explosive
    notes: list[str]

    def to_dict(self) -> dict:
        return {
            "n_analytic": self.n_analytic if not math.isnan(self.n_analytic) else None,
            "n_empirical": self.n_empirical,
            "b_value": self.b_value,
            "beta": self.beta,
            "alpha": self.alpha,
            "explosive": self.explosive,
            "plausible": self.plausible,
            "notes": "; ".join(self.notes),
        }


def compute_branching_ratio(
    K: float,
    alpha: float,
    Mc: float,
    mags: np.ndarray,
    b_value: float,
) -> BranchingResult:
    """Compute branching ratio n.

    Parameters
    ----------
    K : productivity amplitude
    alpha : magnitude-scaling exponent
    Mc : completeness threshold
    mags : catalog magnitudes above Mc
    b_value : GR b-value (for β = b·ln10)
    """
    beta = b_value * math.log(10.0)
    notes = []
    explosive = False
    n_analytic = float("nan")

    if alpha >= beta:
        explosive = True
        notes.append(
            f"EXPLOSIVE: α={alpha:.3f} >= β={beta:.3f} (b={b_value:.3f}). "
            "Branching ratio diverges; the Hawkes process is non-stationary. "
            "This indicates either α is overestimated, b is underestimated, "
            "or the catalog is too short to constrain the magnitude tail."
        )
    else:
        n_analytic = K * beta / (beta - alpha)

    # Empirical: average K·10^{α(M_i - Mc)} over catalog (BASE-10, Phase A)
    mags = np.asarray(mags, dtype=float)
    mags = mags[mags >= Mc - 0.05]
    if len(mags) > 0:
        n_empirical = float(K * np.mean(np.power(10.0, alpha * (mags - Mc))))
    else:
        n_empirical = float("nan")

    plausible = (not explosive) and (n_analytic < 1.0) and (n_empirical < 1.0)
    if not explosive:
        if n_analytic >= 1.0:
            notes.append(
                f"n_analytic={n_analytic:.3f} >= 1 (supercritical). "
                "Process is explosive in expectation; treat forecasts with caution."
            )
        elif 0.5 <= n_analytic <= 0.95:
            notes.append(
                f"n_analytic={n_analytic:.3f} (typical tectonic regime; 0.5-0.95 is common)."
            )
        elif n_analytic < 0.3:
            notes.append(
                f"n_analytic={n_analytic:.3f} (low; mostly background-driven)."
            )
    if not math.isnan(n_empirical) and not math.isnan(n_analytic):
        if abs(n_empirical - n_analytic) > 0.2 * max(n_analytic, 1e-6):
            notes.append(
                f"Empirical n={n_empirical:.3f} differs from analytic n={n_analytic:.3f}; "
                "magnitude distribution may deviate from GR, or α is poorly identified."
            )
    return BranchingResult(
        n_analytic=n_analytic,
        n_empirical=n_empirical,
        b_value=b_value, beta=beta, alpha=alpha,
        explosive=explosive, plausible=plausible, notes=notes,
    )


def branching_plausibility(n: float) -> str:
    """Return a qualitative plausibility label for a branching ratio."""
    if math.isnan(n):
        return "indeterminate"
    if n >= 1.0:
        return "supercritical (explosive; implausible for stationary catalog)"
    if 0.5 <= n < 1.0:
        return "high but plausible (typical tectonic)"
    if 0.2 <= n < 0.5:
        return "moderate (plausible)"
    if 0.0 <= n < 0.2:
        return "low (background-dominated)"
    return "invalid"
