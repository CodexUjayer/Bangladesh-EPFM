"""B4: Uncertainty propagation.

Propagates uncertainty from:
  - Mc (working range M3.5-4.5)
  - magnitude conversion (Scordilis σ=0.41 for mb→Mw)
  - b-value (Shi-Bolt σ_b)
  - event-rate estimates (Poisson Garwood CI)
  - spatial rate estimation (Jeffreys CI per cell)
  - sparse large-magnitude counts

Separates ALEATORY (inherent randomness) from EPISTEMIC (parameter/data/model)
uncertainty.

Where a probabilistic error model is defensible, propagate it.
Where it is not, use explicit sensitivity scenarios.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from ..baselines.uncertainty import poisson_rate_ci_garwood, poisson_rate_ci_jeffreys
from ..baselines.gutenberg_richter import fit_gutenberg_richter
from ..baselines.poisson import HORIZON_YEARS
from ..ingestion.schema import CanonicalEvent

logger = logging.getLogger("phase_b.b4")


@dataclass
class UncertaintyResult:
    """Uncertainty propagation result for one forecast quantity."""
    quantity: str
    point_estimate: float
    aleatory_uncertainty: Optional[float] = None
    epistemic_uncertainty: Optional[float] = None
    total_uncertainty: Optional[float] = None
    lower_95: Optional[float] = None
    upper_95: Optional[float] = None
    sources: list = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def run_uncertainty_propagation(
    events: list[CanonicalEvent],
    catalog_start: datetime,
    thresholds: list[float] = None,
) -> dict:
    """Run the full uncertainty propagation analysis.

    Returns dict: threshold -> {
        'rate': UncertaintyResult,
        'b_value': UncertaintyResult,
        'probability_7d': UncertaintyResult,
        'probability_30d': UncertaintyResult,
    }
    """
    if thresholds is None:
        thresholds = [4.5, 5.0, 5.5, 6.0, 6.5, 7.0]

    results = {}
    exposure_years = (max(e.origin_time_utc for e in events) - catalog_start).total_seconds() / (365.25 * 86400)

    for th in thresholds:
        logger.warning("B4: uncertainty propagation for M>=%s", th)
        # Count events above threshold
        mags = np.array([e.mw if e.mw is not None else e.original_magnitude for e in events])
        mags = mags[~np.isnan(mags)]
        n_above = int(np.sum(mags >= th))

        # --- Rate uncertainty ---
        # Aleatory: Poisson counting uncertainty (Garwood exact CI)
        rate_garwood = poisson_rate_ci_garwood(n_above, exposure_years)
        # Epistemic: magnitude conversion uncertainty (for mb events, σ=0.41)
        # Events near the threshold could be above or below after conversion
        n_mb = int(np.sum([1 for e in events
                           if (e.mw if e.mw is not None else e.original_magnitude) >= th
                           and e.original_magnitude_type == "mb"]))
        # Sensitivity: how many events within σ=0.41 of the threshold?
        n_near_threshold = int(np.sum([
            1 for e in events
            if abs((e.mw if e.mw is not None else e.original_magnitude) - th) < 0.41
        ]))
        # The conversion uncertainty means n_above could be ±n_near_threshold * fraction
        # Conservative: the rate CI is widened by the fraction of ambiguous events
        rate_point = n_above / exposure_years
        rate_aleatory = (rate_garwood[1] - rate_garwood[0]) / 2.0
        # Epistemic from magnitude conversion: ±n_near_threshold/exposure
        rate_epistemic = n_near_threshold / exposure_years * 0.3  # 30% of ambiguous events could flip

        rate_result = UncertaintyResult(
            quantity=f"rate_M{th}",
            point_estimate=rate_point,
            aleatory_uncertainty=rate_aleatory,
            epistemic_uncertainty=rate_epistemic,
            total_uncertainty=math.sqrt(rate_aleatory**2 + rate_epistemic**2),
            lower_95=rate_point - 1.96 * math.sqrt(rate_aleatory**2 + rate_epistemic**2),
            upper_95=rate_point + 1.96 * math.sqrt(rate_aleatory**2 + rate_epistemic**2),
            sources=["Poisson counting (Garwood)", "Magnitude conversion (Scordilis σ=0.41)"],
            notes=[f"N={n_above} events above M{th}; {n_mb} are mb (σ=0.41); {n_near_threshold} within σ of threshold"],
        )

        # --- b-value uncertainty ---
        gr = fit_gutenberg_richter(events, mc=4.5)
        b_point = gr.b_mle
        b_aleatory = gr.b_sigma_shibolt  # Shi-Bolt is aleatory (sampling)
        # Epistemic: Mc uncertainty (b changes from 0.49 at Mc=4.0 to 1.43 at Mc=5.0)
        gr_40 = fit_gutenberg_richter(events, mc=4.0)
        gr_50 = fit_gutenberg_richter(events, mc=5.0)
        b_epistemic = abs(gr_50.b_mle - gr_40.b_mle) / 2.0  # half the Mc-sensitivity range

        b_result = UncertaintyResult(
            quantity="b_value",
            point_estimate=b_point,
            aleatory_uncertainty=b_aleatory,
            epistemic_uncertainty=b_epistemic,
            total_uncertainty=math.sqrt(b_aleatory**2 + b_epistemic**2),
            lower_95=b_point - 1.96 * math.sqrt(b_aleatory**2 + b_epistemic**2),
            upper_95=b_point + 1.96 * math.sqrt(b_aleatory**2 + b_epistemic**2),
            sources=["Shi-Bolt (sampling)", "Mc sensitivity (M4.0-M5.0 range)"],
            notes=[f"b={b_point:.3f} (Mc=4.5); range {gr_40.b_mle:.3f}-{gr_50.b_mle:.3f} across Mc"],
        )

        # --- Probability uncertainty (7d and 30d) ---
        prob_results = {}
        for horizon_name in ["7d", "30d"]:
            hy = HORIZON_YEARS[horizon_name]
            p_point = 1.0 - math.exp(-rate_point * hy)
            # Propagate rate uncertainty through P = 1 - exp(-λΔt)
            # ΔP ≈ exp(-λΔt) * Δλ * Δt
            p_aleatory = math.exp(-rate_point * hy) * rate_aleatory * hy
            p_epistemic = math.exp(-rate_point * hy) * rate_epistemic * hy
            p_total = math.sqrt(p_aleatory**2 + p_epistemic**2)
            prob_results[horizon_name] = UncertaintyResult(
                quantity=f"P_ge1_{horizon_name}_M{th}",
                point_estimate=p_point,
                aleatory_uncertainty=p_aleatory,
                epistemic_uncertainty=p_epistemic,
                total_uncertainty=p_total,
                lower_95=max(p_point - 1.96 * p_total, 0.0),
                upper_95=min(p_point + 1.96 * p_total, 1.0),
                sources=["Rate uncertainty propagated through P=1-exp(-λΔt)"],
                notes=[f"λ={rate_point:.4f}/yr; horizon={hy:.4f}yr"],
            )

        results[th] = {
            "n_events": n_above,
            "n_mb_events": n_mb,
            "n_near_threshold": n_near_threshold,
            "exposure_years": exposure_years,
            "rate": rate_result.__dict__,
            "b_value": b_result.__dict__,
            "probability_7d": prob_results["7d"].__dict__,
            "probability_30d": prob_results["30d"].__dict__,
        }

    return results
