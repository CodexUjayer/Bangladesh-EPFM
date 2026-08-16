"""Stress-to-rate coupling: f(ΔCFS) formulations.

USER REQUIREMENT: Do not assume an arbitrary exponential stress relationship
without justification. Compare multiple physically defensible formulations if
the literature supports them.

Three formulations are implemented, each with documented physical basis:

1. RATE-AND-STATE (Dieterich 1994) — the physically grounded standard:
     f(ΔCFS) = R(t)/R0 = exp(ΔCFS / A·σ̄)
   where A is the rate-state direct-effect parameter and σ̄ is the effective
   normal stress. This is the most defensible because it derives from
   laboratory friction laws. Typical A·σ̄ ≈ 0.1-10 MPa (we use 1 MPa default;
   sensitivity tested).

2. EXPONENTIAL (Toda et al. 1998 empirical):
     f(ΔCFS) = exp(β · ΔCFS)
   where β ≈ 0.1-1.0 /MPa. This is an empirical simplification of the
   rate-and-state form. Less physically rigorous but widely used.

3. STEP / THRESHOLD (King et al. 1994 binary):
     f(ΔCFS) = 1 + α   if ΔCFS > 0
              = 1 - β  if ΔCFS < 0 (stress shadow)
   This is the crudest formulation: positive stress increases rate, negative
   decreases it. Used as a null/baseline stress-coupling.

All three are documented; the rate-and-state form is the default for real
analysis. Sensitivity analysis compares all three.

IMPORTANT: f(ΔCFS) multiplies the background rate λ₀(x). So:
   λ(x, t) = λ₀(x) · f(ΔCFS(x, t))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class CouplingFormulation(str, Enum):
    RATE_AND_STATE = "rate_and_state_dieterich1994"
    EXPONENTIAL = "exponential_toda1998"
    STEP = "step_king1994"


@dataclass(frozen=True)
class CouplingParams:
    """Parameters for the stress-to-rate coupling."""

    formulation: CouplingFormulation = CouplingFormulation.RATE_AND_STATE
    # Rate-and-state
    A_sigma_bar_MPa: float = 1.0     # A·σ̄ (rate-state direct effect × effective normal stress)
    # Exponential
    beta_per_MPa: float = 0.5        # empirical coefficient
    # Step
    step_alpha: float = 1.0          # rate multiplier for ΔCFS > 0
    step_beta: float = 0.5           # rate reduction for ΔCFS < 0
    # Clamp to prevent runaway
    max_multiplier: float = 100.0
    min_multiplier: float = 0.01


def stress_to_rate_factor(
    dcfs_Pa: np.ndarray,
    params: CouplingParams,
) -> np.ndarray:
    """Convert ΔCFS (Pa) to a rate multiplier f(ΔCFS).

    Parameters
    ----------
    dcfs_Pa : array of ΔCFS values in Pascals (positive = triggering)
    params : CouplingParams

    Returns
    -------
    array of rate multipliers (dimensionless; 1.0 = no change)
    """
    # Convert Pa to MPa for the rate-and-state / exponential forms
    dcfs_MPa = np.asarray(dcfs_Pa, dtype=float) / 1e6

    if params.formulation == CouplingFormulation.RATE_AND_STATE:
        # Dieterich (1994): R/R0 = exp(ΔCFS / A·σ̄)
        f = np.exp(dcfs_MPa / params.A_sigma_bar_MPa)
    elif params.formulation == CouplingFormulation.EXPONENTIAL:
        f = np.exp(params.beta_per_MPa * dcfs_MPa)
    elif params.formulation == CouplingFormulation.STEP:
        f = np.where(dcfs_MPa > 0, 1.0 + params.step_alpha,
                     1.0 - params.step_beta)
    else:
        raise ValueError(f"Unknown formulation: {params.formulation}")

    # Clamp to prevent runaway multipliers
    f = np.clip(f, params.min_multiplier, params.max_multiplier)
    return f


def document_formulation(params: CouplingParams) -> str:
    """Return a human-readable description of the chosen formulation."""
    if params.formulation == CouplingFormulation.RATE_AND_STATE:
        return (
            f"Rate-and-state (Dieterich 1994): f(ΔCFS) = exp(ΔCFS / A·σ̄), "
            f"A·σ̄ = {params.A_sigma_bar_MPa} MPa. "
            "This is the physically grounded standard, derived from laboratory "
            "friction laws. ΔCFS in Pa converted to MPa."
        )
    elif params.formulation == CouplingFormulation.EXPONENTIAL:
        return (
            f"Exponential (Toda et al. 1998): f(ΔCFS) = exp(β·ΔCFS), "
            f"β = {params.beta_per_MPa} /MPa. "
            "Empirical simplification of rate-and-state."
        )
    elif params.formulation == CouplingFormulation.STEP:
        return (
            f"Step (King et al. 1994): f(ΔCFS) = 1+α if ΔCFS>0, 1-β if ΔCFS<0, "
            f"α={params.step_alpha}, β={params.step_beta}. "
            "Crudest formulation; baseline stress-coupling."
        )
    return "Unknown formulation"
