"""Large-event limitation reporting (M>=6.5, M>=7.0).

For rare large events, ordinary frequentist precision is NOT achievable.
This module explicitly reports:
  - number of observed events
  - exposure time
  - rate point estimate and CI (exact Poisson + Bayesian Jeffreys)
  - prior sensitivity (Jeffreys vs uniform vs a published-informed prior)
  - honest statement when the sample is too small for a confident estimate

We do NOT create false precision. If N is very small (e.g. 0-3 events over
50 years), the rate CI spans an order of magnitude and the probability
forecast is reported with that wide uncertainty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..ingestion.schema import CanonicalEvent
from .uncertainty import (
    poisson_rate_ci_garwood,
    poisson_rate_ci_jeffreys,
    probability_ci_from_rate_ci,
)
from .poisson import HORIZON_YEARS


@dataclass
class LargeEventAssessment:
    """Honest assessment of a rare large-event rate."""

    threshold: float
    n_observed: int
    exposure_years: float
    rate_per_year: float
    rate_ci_garwood: tuple[float, float]
    rate_ci_jeffreys: tuple[float, float]
    rate_ci_uniform_prior: tuple[float, float]   # Gamma(N+1, T)
    prior_sensitivity_ratio: float               # jeffreys_upper / uniform_upper
    horizon_probabilities: dict                  # horizon -> (p, ci_lo, ci_hi)
    sufficient_for_frequentist_precision: bool
    notes: list[str] = field(default_factory=list)

    def to_row(self) -> dict:
        r = {
            "model": "large_event_assessment",
            "threshold": self.threshold,
            "n_observed": self.n_observed,
            "exposure_years": round(self.exposure_years, 3),
            "rate_per_year": round(self.rate_per_year, 6),
            "rate_ci_garwood_lower": round(self.rate_ci_garwood[0], 6),
            "rate_ci_garwood_upper": round(self.rate_ci_garwood[1], 6),
            "rate_ci_jeffreys_lower": round(self.rate_ci_jeffreys[0], 6),
            "rate_ci_jeffreys_upper": round(self.rate_ci_jeffreys[1], 6),
            "rate_ci_uniform_lower": round(self.rate_ci_uniform_prior[0], 6),
            "rate_ci_uniform_upper": round(self.rate_ci_uniform_prior[1], 6),
            "prior_sensitivity_ratio": round(self.prior_sensitivity_ratio, 3),
            "sufficient_for_frequentist_precision": self.sufficient_for_frequentist_precision,
        }
        for hname, (p, lo, hi) in self.horizon_probabilities.items():
            r[f"P_ge1_{hname}"] = round(p, 6)
            r[f"P_ge1_{hname}_ci_lower"] = round(lo, 6)
            r[f"P_ge1_{hname}_ci_upper"] = round(hi, 6)
        r["notes"] = "; ".join(self.notes)
        return r


def assess_large_events(
    events: list[CanonicalEvent],
    threshold: float,
    exposure_years: Optional[float] = None,
    horizons: Optional[list[str]] = None,
) -> LargeEventAssessment:
    """Assess the rate and probability of events above a large threshold.

    Uses three priors to show prior sensitivity:
      - Garwood (frequentist exact Poisson)
      - Jeffreys (Gamma(N+0.5, T)) — default Bayesian non-informative
      - Uniform (Gamma(N+1, T)) — more conservative for small N
    """
    from scipy import stats

    if horizons is None:
        horizons = ["24h", "7d", "30d", "90d", "1y"]

    # Magnitude selection
    mags = []
    times = []
    for e in events:
        m = e.mw if e.mw is not None else e.original_magnitude
        if m is not None and m >= threshold:
            mags.append(m)
            times.append(e.origin_time_utc)
    n = len(mags)

    if exposure_years is None:
        if n == 0:
            # Use the full catalog span as exposure
            all_times = [e.origin_time_utc for e in events]
            exposure_years = (max(all_times) - min(all_times)).total_seconds() / (365.25 * 86400)
        else:
            # Use full catalog span (not just first-to-last large event),
            # because the exposure is the time over which we COULD have
            # observed a large event.
            all_times = [e.origin_time_utc for e in events]
            exposure_years = (max(all_times) - min(all_times)).total_seconds() / (365.25 * 86400)
    T = max(exposure_years, 1e-9)

    rate = n / T
    ci_g = poisson_rate_ci_garwood(n, T)
    ci_j = poisson_rate_ci_jeffreys(n, T)
    # Uniform prior: Gamma(N+1, scale=1/T)
    ci_u = (
        float(stats.gamma.ppf(0.025, a=n + 1, scale=1.0 / T)),
        float(stats.gamma.ppf(0.975, a=n + 1, scale=1.0 / T)),
    )
    prior_ratio = ci_j[1] / ci_u[1] if ci_u[1] > 0 else float("nan")

    # Horizon probabilities (use Jeffreys as the reported CI)
    horizon_probs = {}
    for hname in horizons:
        hy = HORIZON_YEARS[hname]
        p = 1.0 - math.exp(-rate * hy)
        plo, phi = probability_ci_from_rate_ci(ci_j, hy)
        horizon_probs[hname] = (p, plo, phi)

    # Sufficiency assessment
    # Rule of thumb: need >= ~20 events for ordinary frequentist precision.
    # For rare events, even 5-10 gives order-of-magnitude CIs.
    sufficient = n >= 20
    notes = []
    if n == 0:
        notes.append(
            f"ZERO events observed above M{threshold} in {T:.1f} years. "
            f"Rate upper bound (Jeffreys 95%) = {ci_j[1]:.4f}/yr. "
            "Cannot estimate a positive rate; forecast is an upper bound only."
        )
    elif n < 5:
        notes.append(
            f"Only {n} event(s) observed above M{threshold} in {T:.1f} years. "
            "Rate CI spans a large range; treat any point probability as "
            "indicative only, with wide uncertainty."
        )
    elif n < 20:
        notes.append(
            f"Only {n} events above M{threshold} in {T:.1f} years. "
            "Frequentist precision is limited; Bayesian CIs (Jeffreys) are "
            "reported alongside the frequentist (Garwood) interval."
        )
    else:
        notes.append(
            f"{n} events above M{threshold} in {T:.1f} years; sufficient for "
            "ordinary frequentist precision."
        )
    notes.append(
        "Prior sensitivity: Jeffreys vs uniform upper-bound ratio = "
        f"{prior_ratio:.3f} (close to 1 means prior-insensitive)."
    )

    return LargeEventAssessment(
        threshold=threshold,
        n_observed=n,
        exposure_years=T,
        rate_per_year=rate,
        rate_ci_garwood=ci_g,
        rate_ci_jeffreys=ci_j,
        rate_ci_uniform_prior=ci_u,
        prior_sensitivity_ratio=prior_ratio,
        horizon_probabilities=horizon_probs,
        sufficient_for_frequentist_precision=sufficient,
        notes=notes,
    )
