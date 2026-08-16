"""Omori-Utsu temporal aftershock decay kernel.

g(τ; c, p) = (p - 1) · c^{p-1} / (τ + c)^p   for τ >= 0
            = 0                                for τ < 0

Normalized so that ∫_0^∞ g(τ) dτ = 1 (this is the standard ETAS convention;
the (p-1)·c^{p-1} prefactor achieves normalization).

For p <= 1 the integral diverges; we require p > 1 (standard in ETAS).
"""

from __future__ import annotations

import math


def omori_utsu_g(tau: float, c: float, p: float) -> float:
    """Omori-Utsu kernel value at lag tau (days). Returns 0 for tau < 0.

    g(τ) = (p-1) c^{p-1} / (τ + c)^p   for τ >= 0
    """
    if tau < 0:
        return 0.0
    if p <= 1.0:
        # Non-integrable; return unnormalized form (caller should ensure p>1)
        return 1.0 / (tau + c) ** p
    return (p - 1.0) * (c ** (p - 1.0)) / (tau + c) ** p


def omori_normalization(c: float, p: float) -> float:
    """The normalization constant (p-1)·c^{p-1} so that ∫_0^∞ g = 1.

    Returns 1 for the already-normalized form. Useful for documentation.
    """
    if p <= 1.0:
        return float("inf")
    return (p - 1.0) * (c ** (p - 1.0))


def omori_integral_over_window(c: float, p: float, t_start: float, t_end: float) -> float:
    """∫_{t_start}^{t_end} g(τ) dτ for the normalized Omori kernel.

    ∫ (p-1) c^{p-1} / (τ+c)^p dτ = -c^{p-1} (τ+c)^{-(p-1)} | from p-1 antiderivative
    Actually: d/dτ [ -(p-1) c^{p-1} / ((p-1)(τ+c)^{p-1}) ] = (p-1)c^{p-1}/(τ+c)^p
    => antiderivative = -c^{p-1} / (τ+c)^{p-1}
    => ∫_{a}^{b} g = c^{p-1} [ (a+c)^{-(p-1)} - (b+c)^{-(p-1)} ]
    """
    if p <= 1.0:
        return float("nan")
    a = max(t_start, 0.0) + c
    b = max(t_end, 0.0) + c
    if b <= a:
        return 0.0
    return (c ** (p - 1.0)) * (a ** (-(p - 1.0)) - b ** (-(p - 1.0)))
