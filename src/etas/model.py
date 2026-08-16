"""ETAS model: conditional intensity, parameter container, evaluation.

The conditional intensity is (Phase A corrected — BASE-10 productivity):

    λ(x, y, t | H_t) = μ(x, y)
                     + Σ_{i: t_i < t}  K · 10^{α (M_i − M_c)}
                       · g(t − t_i; c, p)
                       · f(r_i(x,y); σ, γ, q)

with:
  g(τ) = (p-1) c^{p-1} / (τ + c)^p      (Omori-Utsu, normalized; τ in days)
  f(r; M_i) = (q-1)/(π s²) · [1 + (r/s)²]^(-(1+q))
            where s = σ · exp[γ (M_i - M_c)]  (magnitude-scaled length)

The productivity term K · 10^{α(M−Mc)} uses BASE-10 (per the research report
specification). The previous implementation used exp(α·(M−Mc)) (base-e),
which is inconsistent with the literature (Ogata 1998, Zhuang 2011). The
base-10 form means α is on the same scale as published values (typical
0.3-2.0), and the branching ratio n = K·β/(β−α) is derived for this base-10
form where β = b·ln(10).

Parameter bounds (used by MLE):
  μ_total  > 0   (events/year)
  K        > 0
  α        in [0.0, 3.0]   (typical ETAS: 0.3-2.0; α<0.5 = swarm-like)
  c        in [0.001, 1.0] (days; typical 0.01-0.1)
  p        in [1.01, 2.5]  (must be >1 for normalization; typical 1.0-1.2)
  σ_km     in [0.5, 200.0] (spatial scale; typical 1-50 km)
  γ        in [0.0, 2.0]   (magnitude-spatial scaling; typical 0.4-0.5)
  q        in [0.5, 3.0]   (spatial power-law exponent; typical 0.5-1.5)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .background import BackgroundRate
from .omori import omori_utsu_g
from .spatial_kernels import (
    _equirect_km,
    magnitude_scaled_length,
    power_law_spatial_kernel,
)


@dataclass
class ETASParams:
    """Container for ETAS parameters with explicit bounds."""

    mu_total_per_year: float
    K: float
    alpha: float
    c_days: float
    p: float
    sigma_km: float
    gamma: float
    q: float
    Mc: float                       # magnitude threshold used for fitting
    # Optional: spatial kernel type
    spatial_kernel: str = "powerlaw"   # "powerlaw" | "gaussian"
    # Optional: which parameters were fixed (not estimated) and why
    fixed_parameters: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "mu_total_per_year": self.mu_total_per_year,
            "K": self.K,
            "alpha": self.alpha,
            "c_days": self.c_days,
            "p": self.p,
            "sigma_km": self.sigma_km,
            "gamma": self.gamma,
            "q": self.q,
            "Mc": self.Mc,
            "spatial_kernel": self.spatial_kernel,
            "fixed_parameters": self.fixed_parameters,
        }


# Default parameter bounds for MLE.
PARAM_BOUNDS = {
    "mu_total_per_year": (1e-6, 1e4),
    "K":                 (1e-8, 1.0),
    "alpha":             (0.0, 3.0),
    "c_days":            (1e-3, 1.0),
    "p":                 (1.01, 2.5),
    "sigma_km":          (0.5, 200.0),
    "gamma":             (0.0, 2.0),
    "q":                 (0.5, 3.0),
}


@dataclass
class ETASModel:
    """A fitted ETAS model: parameters + background specification."""

    params: ETASParams
    background: BackgroundRate
    bbox: tuple[float, float, float, float]   # (min_lat, max_lat, min_lon, max_lon)
    fit_info: dict = field(default_factory=dict)


def conditional_intensity(
    model: ETASModel,
    t_query: float,                    # days since catalog start
    lat_query: float,
    lon_query: float,
    history_lats,
    history_lons,
    history_times_days,
    history_mags,
) -> float:
    """Evaluate λ(x, y, t | H_t) at (lat_query, lon_query, t_query).

    Parameters
    ----------
    t_query : float
        Time of the query, in days since the catalog start.
    history_* : arrays of past events (t_i < t_query).
    """
    p = model.params
    # Background contribution
    lam = model.background.at(lat_query, lon_query)  # per year per km²
    # Convert to per-day rate density: /365.25
    lam_per_day_per_km2 = lam / 365.25

    # Triggered contribution
    history_lats = np.asarray(history_lats, dtype=float)
    history_lons = np.asarray(history_lons, dtype=float)
    history_times = np.asarray(history_times_days, dtype=float)
    history_mags = np.asarray(history_mags, dtype=float)
    if len(history_times) == 0:
        return lam_per_day_per_km2

    # Only events before t_query
    mask = history_times < t_query
    if not mask.any():
        return lam_per_day_per_km2
    hlats = history_lats[mask]
    hlons = history_lons[mask]
    htimes = history_times[mask]
    hmags = history_mags[mask]

    # Time lags
    tau = t_query - htimes
    # Omori kernel
    g = np.array([omori_utsu_g(tau_i, p.c_days, p.p) for tau_i in tau])
    # Productivity — BASE-10 formulation per the research report:
    #   K · 10^{α(mj − m0)}
    # (previously used exp(α·(M−Mc)) which is base-e and inconsistent with
    #  the literature; corrected in Phase A)
    prod = p.K * np.power(10.0, p.alpha * (hmags - p.Mc))
    # Spatial kernel
    if p.spatial_kernel == "powerlaw":
        s_km = p.sigma_km * np.array([magnitude_scaled_length(m, p.Mc, p.gamma) for m in hmags])
        # distances
        r_km = np.array([_equirect_km(lat_query, lon_query, la, lo)
                         for la, lo in zip(hlats, hlons)])
        # avoid div-by-zero
        s2 = s_km * s_km
        f = (p.q - 1.0) / (np.pi * s2) * (1.0 + (r_km * r_km) / s2) ** (-(1.0 + p.q))
    else:  # gaussian
        s_km = p.sigma_km * np.array([magnitude_scaled_length(m, p.Mc, p.gamma) for m in hmags])
        r_km = np.array([_equirect_km(lat_query, lon_query, la, lo)
                         for la, lo in zip(hlats, hlons)])
        s2 = s_km * s_km
        f = 1.0 / (np.pi * s2) * np.exp(-(r_km * r_km) / s2)

    triggered = float(np.sum(prod * g * f))
    return lam_per_day_per_km2 + triggered
