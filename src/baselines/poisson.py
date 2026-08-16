"""Temporal Poisson baseline.

Estimates the earthquake occurrence rate lambda = N / T for each magnitude
threshold and computes P(N >= 1 | horizon) = 1 - exp(-lambda * horizon).

CRITICAL DISTINCTIONS (enforced in the data structures):
  - lambda (rate per year) is NOT the same as expected count
  - expected count = lambda * horizon is NOT a probability
  - P(N >= 1 | horizon) = 1 - exp(-lambda * horizon) is a probability
  - all are conditional on the observed catalog and the working Mc

Uncertainty:
  - lambda CI: exact Poisson (Garwood 1936) by default; Jeffreys Bayesian
    for small N (used by assess_large_events).
  - P CI: propagate the lambda CI through 1 - exp(-lambda * horizon).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..ingestion.schema import CanonicalEvent
from .uncertainty import (
    poisson_rate_ci_garwood,
    probability_ci_from_rate_ci,
)

# Standard forecast horizons in years.
HORIZON_YEARS = {
    "24h":  1.0 / 365.25,
    "7d":   7.0 / 365.25,
    "30d":  30.0 / 365.25,
    "90d":  90.0 / 365.25,
    "1y":   1.0,
}


@dataclass(frozen=True)
class PoissonRateEstimate:
    """Poisson rate estimate for one (threshold, catalog) combination."""

    threshold: float
    n_observed: int
    exposure_years: float
    rate_per_year: float                # lambda_hat = N / T
    rate_ci_lower: float
    rate_ci_upper: float
    ci_method: str = "garwood_exact"
    rate_uncertainty_per_year: Optional[float] = None  # half-width of CI

    def expected_count(self, horizon_years: float) -> float:
        """Expected number of events over the horizon (NOT a probability)."""
        return self.rate_per_year * horizon_years

    def probability_at_least_one(self, horizon_years: float) -> float:
        """P(N >= 1 | horizon) under a Poisson process with this rate."""
        return 1.0 - math.exp(-self.rate_per_year * horizon_years)

    def probability_ci(self, horizon_years: float) -> tuple[float, float]:
        """95% CI on P(N >= 1 | horizon), propagated from the rate CI."""
        return probability_ci_from_rate_ci(
            (self.rate_ci_lower, self.rate_ci_upper), horizon_years
        )


@dataclass
class TemporalPoissonResult:
    """Full temporal Poisson baseline for one threshold across all horizons."""

    threshold: float
    rate: PoissonRateEstimate
    horizon_results: dict = field(default_factory=dict)  # horizon_name -> dict
    notes: list[str] = field(default_factory=list)

    def to_row(self) -> dict:
        r = {
            "model": "temporal_poisson",
            "threshold": self.threshold,
            "n_observed": self.rate.n_observed,
            "exposure_years": round(self.rate.exposure_years, 3),
            "rate_per_year": round(self.rate.rate_per_year, 6),
            "rate_ci_lower": round(self.rate.rate_ci_lower, 6),
            "rate_ci_upper": round(self.rate.rate_ci_upper, 6),
        }
        for hname, hy in HORIZON_YEARS.items():
            r[f"expected_count_{hname}"] = round(self.rate.expected_count(hy), 6)
            r[f"P_ge1_{hname}"] = round(self.rate.probability_at_least_one(hy), 6)
            plo, phi = self.rate.probability_ci(hy)
            r[f"P_ge1_{hname}_ci_lower"] = round(plo, 6)
            r[f"P_ge1_{hname}_ci_upper"] = round(phi, 6)
        r["notes"] = "; ".join(self.notes)
        return r


def estimate_temporal_poisson(
    events: list[CanonicalEvent],
    threshold: float,
    exposure_years: Optional[float] = None,
    start_time=None,
    end_time=None,
    ci_method: str = "garwood",
) -> TemporalPoissonResult:
    """Estimate the temporal Poisson rate for events above ``threshold``.

    Parameters
    ----------
    events : list of CanonicalEvent
        The catalog (filtered to Mw where available, else original magnitude).
    threshold : float
        Magnitude threshold (e.g. 4.5).
    exposure_years : float, optional
        If given, use this as T. Otherwise compute from the catalog's time
        span (last - first event). For a proper baseline we recommend
        passing an explicit exposure based on the catalog's instrumentation
        history; using last-first underestimates T slightly.
    start_time, end_time : datetime, optional
        If given, only count events in [start_time, end_time).
    ci_method : 'garwood' or 'jeffreys'
    """
    # Filter by time and magnitude
    if start_time is not None or end_time is not None:
        sel = []
        for e in events:
            t = e.origin_time_utc
            if start_time is not None and t < start_time:
                continue
            if end_time is not None and t >= end_time:
                continue
            sel.append(e)
    else:
        sel = list(events)

    # Magnitude selection: prefer Mw, else original (documented).
    mags = []
    for e in sel:
        m = e.mw if e.mw is not None else e.original_magnitude
        if m is not None and m >= threshold:
            mags.append(m)
    n = len(mags)

    # Exposure time
    if exposure_years is None:
        if len(sel) == 0:
            exposure_years = 0.0
        else:
            times = [e.origin_time_utc for e in sel]
            exposure_years = (max(times) - min(times)).total_seconds() / (365.25 * 86400)
            # Add a small correction: exposure is from first to last event
            # plus the average inter-event gap. For a baseline we use last-first
            # and note this slightly underestimates T.
    T = max(exposure_years, 1e-9)

    rate_hat = n / T
    if ci_method == "jeffreys":
        from .uncertainty import poisson_rate_ci_jeffreys
        ci_lo, ci_hi = poisson_rate_ci_jeffreys(n, T)
        method = "jeffreys"
    else:
        ci_lo, ci_hi = poisson_rate_ci_garwood(n, T)
        method = "garwood_exact"

    rate_est = PoissonRateEstimate(
        threshold=threshold,
        n_observed=n,
        exposure_years=T,
        rate_per_year=rate_hat,
        rate_ci_lower=ci_lo,
        rate_ci_upper=ci_hi,
        ci_method=method,
        rate_uncertainty_per_year=(ci_hi - ci_lo) / 2.0 if n > 0 else None,
    )

    # Per-horizon results
    horizon_results = {}
    notes = []
    if n < 20:
        notes.append(
            f"Small sample (N={n}); rate and probability CIs are wide. "
            "Treat point estimates as indicative only."
        )
    if exposure_years is None:
        notes.append(
            "Exposure T computed from catalog span (last-first event); "
            "may slightly underestimate true T."
        )
    for hname, hy in HORIZON_YEARS.items():
        horizon_results[hname] = {
            "horizon_years": hy,
            "expected_count": rate_est.expected_count(hy),
            "P_ge1": rate_est.probability_at_least_one(hy),
            "P_ge1_ci": rate_est.probability_ci(hy),
        }

    return TemporalPoissonResult(
        threshold=threshold, rate=rate_est,
        horizon_results=horizon_results, notes=notes,
    )


def probability_at_least_one(rate_per_year: float, horizon_years: float) -> float:
    """P(N >= 1 | horizon) = 1 - exp(-lambda * delta_t). Pure function."""
    return 1.0 - math.exp(-rate_per_year * horizon_years)


def expected_count(rate_per_year: float, horizon_years: float) -> float:
    """Expected number of events over the horizon. NOT a probability."""
    return rate_per_year * horizon_years
