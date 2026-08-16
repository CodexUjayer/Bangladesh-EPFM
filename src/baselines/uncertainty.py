"""Uncertainty quantification for Stage 4 baselines.

Implements:
  - poisson_rate_ci_garwood : exact Poisson (Garwood 1936) CI on the rate
  - poisson_rate_ci_jeffreys : Bayesian Jeffreys-prior credible interval
  - probability_ci_from_rate_ci : propagate a rate CI through 1-exp(-λΔt)
  - bootstrap_bvalue_ci : nonparametric bootstrap CI on the GR b-value

All intervals are 95% by default and are returned as (lower, upper).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy import stats


def poisson_rate_ci_garwood(
    n_obs: int,
    exposure: float,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Exact Poisson confidence interval on the rate (Garwood 1936).

    If N ~ Poisson(lambda * T) is observed, then:
      lower = 0.5 * chi2(2N, alpha/2) / T
      upper = 0.5 * chi2(2(N+1), 1-alpha/2) / T

    For N=0, lower=0 and upper = 0.5 * chi2(2, 1-alpha) / T.
    """
    if exposure <= 0:
        return (float("nan"), float("nan"))
    alpha = 1.0 - confidence
    if n_obs == 0:
        lower = 0.0
        upper = 0.5 * stats.chi2.ppf(1 - alpha, df=2) / exposure
    else:
        lower = 0.5 * stats.chi2.ppf(alpha / 2, df=2 * n_obs) / exposure
        upper = 0.5 * stats.chi2.ppf(1 - alpha / 2, df=2 * (n_obs + 1)) / exposure
    return (float(lower), float(upper))


def poisson_rate_ci_jeffreys(
    n_obs: int,
    exposure: float,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bayesian credible interval on the Poisson rate with Jeffreys prior.

    Posterior: lambda | N ~ Gamma(N + 0.5, scale=1/T).
    Jeffreys prior is non-informative and behaves better than Garwood for
    small N (it never has zero lower bound for N=0, reflecting genuine
    residual uncertainty).
    """
    if exposure <= 0:
        return (float("nan"), float("nan"))
    alpha = 1.0 - confidence
    shape = n_obs + 0.5
    scale = 1.0 / exposure
    lower = stats.gamma.ppf(alpha / 2, a=shape, scale=scale)
    upper = stats.gamma.ppf(1 - alpha / 2, a=shape, scale=scale)
    return (float(lower), float(upper))


def probability_ci_from_rate_ci(
    rate_ci: tuple[float, float],
    horizon_years: float,
) -> tuple[float, float]:
    """Propagate a rate confidence interval through P(N>=1) = 1 - exp(-λΔt).

    Because P is monotone increasing in λ, the rate-CI bounds map directly
    to the probability-CI bounds (lower rate -> lower probability).
    """
    lo, hi = rate_ci
    p_lo = 1.0 - math.exp(-lo * horizon_years)
    p_hi = 1.0 - math.exp(-hi * horizon_years)
    return (float(p_lo), float(p_hi))


def bootstrap_bvalue_ci(
    magnitudes: np.ndarray,
    mc: float,
    bin_width: float = 0.1,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Nonparametric bootstrap CI on the GR b-value (Aki-Utsu MLE).

    Resamples the magnitudes above Mc with replacement, recomputes b, and
    takes the percentile interval. Uses a fixed seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    m = np.asarray(magnitudes)
    above = m[m >= mc - bin_width / 2]
    n = len(above)
    if n < 20:
        return (float("nan"), float("nan"))
    boot_b = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(above, size=n, replace=True)
        mean_m = float(np.mean(sample))
        denom = mean_m - (mc - bin_width / 2)
        if denom <= 0:
            boot_b[i] = float("nan")
        else:
            boot_b[i] = math.log10(math.e) / denom
    boot_b = boot_b[~np.isnan(boot_b)]
    if len(boot_b) < 100:
        return (float("nan"), float("nan"))
    alpha = 1.0 - confidence
    lo = float(np.percentile(boot_b, 100 * alpha / 2))
    hi = float(np.percentile(boot_b, 100 * (1 - alpha / 2)))
    return (lo, hi)
