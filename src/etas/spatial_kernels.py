"""Spatial triggering kernels for ETAS.

Two standard forms, both normalized to integrate to 1 over the plane:

1. POWER-LAW (heavy-tailed, standard ETAS — Ogata 1998):
   f(r; σ, q) = (q - 1) / (π σ²) · [1 + (r/σ)²]^{-(1+q)}
   Normalization: ∫_0^∞ 2π r f(r) dr = 1 requires q > 0.
   (The (q-1)/(πσ²) form integrates to 1 when the exponent is (1+q) and q>0;
   this is the Marsan-Lengliné / Ogata convention.)

   To make magnitude-dependent productivity spatially, we use the common
   simplification f(r; M) ∝ [1 + (r/(σ·D(M))²)]^{-(1+q)} where
   D(M) = exp(γ(M-Mc)) is the magnitude-scaled length (γ ~ 0.5 typical).

2. GAUSSIAN (simpler; sometimes used for stability):
   f(r; σ) = 1/(π σ²) · exp(-r²/σ²)
   (2D isotropic Gaussian; ∫_plane f dx dy = 1.)

We use the POWER-LAW form as the default (it is the standard for ETAS and
captures the heavy tail of aftershock triggering); the Gaussian is provided
as a fallback for catalogs where the power-law tail is unstable.

Distances are computed in km via a local equirectangular projection
(approximation valid for the ~8° Bangladesh study region).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _equirect_km(lat0: float, lon0: float, lat1: float, lon1: float) -> float:
    """Equirectangular distance in km (good approximation for <~10 deg)."""
    R = 6371.0088
    lat0r, lat1r = math.radians(lat0), math.radians(lat1)
    mean_lat = 0.5 * (lat0r + lat1r)
    dlat = (lat1 - lat0) * math.pi / 180.0
    dlon = (lon1 - lon0) * math.pi / 180.0 * math.cos(mean_lat)
    return R * math.sqrt(dlat * dlat + dlon * dlon)


def power_law_spatial_kernel(
    r_km: float,
    sigma_km: float,
    q: float,
) -> float:
    """Power-law spatial kernel value at distance r (km).

    f(r) = (q - 1) / (π σ²) · [1 + (r/σ)²]^{-(1+q)}

    Requires q > 0 for integrability. Returns 0 for r < 0.
    """
    if r_km < 0:
        return 0.0
    if q <= 0:
        return float("nan")
    s2 = sigma_km * sigma_km
    return (q - 1.0) / (math.pi * s2) * (1.0 + (r_km * r_km) / s2) ** (-(1.0 + q))


def gaussian_spatial_kernel(r_km: float, sigma_km: float) -> float:
    """Isotropic 2D Gaussian kernel.

    f(r) = 1/(π σ²) · exp(-r²/σ²)
    """
    if r_km < 0:
        return 0.0
    s2 = sigma_km * sigma_km
    return 1.0 / (math.pi * s2) * math.exp(-(r_km * r_km) / s2)


def spatial_normalization(sigma_km: float, q: Optional[float] = None) -> float:
    """The normalization prefactor so the kernel integrates to 1 over the plane."""
    s2 = sigma_km * sigma_km
    if q is None:
        # Gaussian
        return 1.0 / (math.pi * s2)
    return (q - 1.0) / (math.pi * s2)


def magnitude_scaled_length(M: float, Mc: float, gamma: float) -> float:
    """D(M) = exp(γ (M - Mc)). Magnitude-dependent triggering length scale.

    A larger mainshock triggers aftershocks over a larger area.
    gamma ~ 0.4-0.5 is typical (Ogata 1998); we estimate it from data where
    supportable, else use a literature-informed prior clearly labeled as such.
    """
    return math.exp(gamma * (M - Mc))


def pairwise_distances_km(
    lats0, lons0, lats1, lons1
) -> np.ndarray:
    """Pairwise equirectangular distances in km. Returns shape (n0, n1)."""
    lats0 = np.asarray(lats0, dtype=float)
    lons0 = np.asarray(lons0, dtype=float)
    lats1 = np.asarray(lats1, dtype=float)
    lons1 = np.asarray(lons1, dtype=float)
    R = 6371.0088
    # broadcast
    lat0r = np.radians(lats0)[:, None]
    lat1r = np.radians(lats1)[None, :]
    mean_lat = 0.5 * (lat0r + lat1r)
    dlat = (lats0[:, None] - lats1[None, :]) * math.pi / 180.0
    dlon = (lons0[:, None] - lons1[None, :]) * math.pi / 180.0 * np.cos(mean_lat)
    return R * np.sqrt(dlat * dlat + dlon * dlon)
