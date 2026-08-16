"""B5: Power / detectability analysis.

Estimates the minimum detectable effect size for Brier, log-score, and
information gain improvements, given the catalog size and validation design.

If the dataset is underpowered for a particular conclusion, that conclusion
is labeled "INSUFFICIENT POWER".
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..baselines.poisson import HORIZON_YEARS

logger = logging.getLogger("phase_b.b5")


@dataclass
class PowerResult:
    threshold: float
    horizon: str
    n_origins: int
    n_cells_per_origin: int
    n_total: int
    base_rate: float
    n_positive: int
    # Minimum detectable effect (MDE) at 80% power, α=0.05
    mde_brier: float          # minimum ΔBrier detectable
    mde_log_likelihood: float # minimum Δlog-lik detectable
    mde_information_gain: float
    sufficient_power: bool    # True if MDE < 0.01 for Brier
    notes: list[str] = field(default_factory=list)


def run_power_analysis(
    n_origins: int = 9,
    n_cells_per_origin: int = 64,
    base_rates: dict = None,
    horizons: list[str] = None,
    thresholds: list[float] = None,
) -> dict:
    """Estimate minimum detectable effect sizes.

    Uses the analytical approximation for paired Brier comparison:
      SE(ΔBrier) ≈ sqrt(2 * Brier * (1 - Brier) / N)
    where N = n_origins × n_cells_per_origin.

    For 80% power at α=0.05 (two-sided), MDE ≈ 2.8 × SE.

    For rare events (low base rate), the effective N is smaller because
    most cells are negative and carry little information.
    """
    if horizons is None:
        horizons = ["7d", "30d"]
    if thresholds is None:
        thresholds = [4.5, 5.0, 5.5, 6.0]
    if base_rates is None:
        # From Stage 7B actual results
        base_rates = {
            (4.5, "7d"): 0.012,
            (5.0, "7d"): 0.005,
            (5.5, "7d"): 0.001,
            (6.0, "7d"): 0.0003,
            (4.5, "30d"): 0.04,
            (5.0, "30d"): 0.01,
            (5.5, "30d"): 0.003,
            (6.0, "30d"): 0.001,
        }

    results = {}
    for th in thresholds:
        for h in horizons:
            br = base_rates.get((th, h), 0.01)
            n_total = n_origins * n_cells_per_origin
            n_positive = int(br * n_total)

            # Brier MDE: for paired comparison, SE(ΔBrier) ≈ sqrt(2*B*(1-B)/N)
            # where B ≈ base_rate (since predicting 0 gives Brier ≈ base_rate)
            brier_null = br  # Brier of "predict 0" model
            se_brier = math.sqrt(2 * brier_null * (1 - brier_null) / n_total)
            mde_brier = 2.8 * se_brier  # 80% power, α=0.05

            # Log-likelihood MDE: SE(Δloglik) ≈ sqrt(2/N) for Bernoulli
            se_ll = math.sqrt(2.0 / n_total)
            mde_ll = 2.8 * se_ll

            # Information gain MDE (same as log-lik per sample)
            mde_ig = mde_ll

            # Sufficient power: can detect ΔBrier >= 0.01?
            sufficient = mde_brier < 0.01

            notes = []
            if n_positive < 10:
                notes.append(f"INSUFFICIENT POWER: only {n_positive} positive test cases; "
                             "cannot reliably detect any improvement.")
            if n_positive < 30:
                notes.append(f"LOW POWER: {n_positive} positives; MDE={mde_brier:.4f} for Brier.")
            if not sufficient:
                notes.append(f"INSUFFICIENT POWER for Brier: MDE={mde_brier:.4f} > 0.01.")

            results[(th, h)] = PowerResult(
                threshold=th, horizon=h, n_origins=n_origins,
                n_cells_per_origin=n_cells_per_origin, n_total=n_total,
                base_rate=br, n_positive=n_positive,
                mde_brier=mde_brier, mde_log_likelihood=mde_ll,
                mde_information_gain=mde_ig,
                sufficient_power=sufficient,
                notes=notes,
            ).__dict__

    return results
