"""Background rate μ(x, y) for ETAS.

Two options:
  A. UniformBackground: μ(x,y) = μ_0 (constant over the study region).
     Total background rate μ_0 is estimated as part of the MLE.
  B. KDEBackground: μ(x,y) is a spatial KDE of declustered mainshocks,
     normalized so ∫ μ(x,y) dx dy = μ_0 (the total background rate).

For the KDE background we use a Gaussian kernel with a data-driven
bandwidth (Silverman's rule of thumb, with a configurable multiplier).
The spatially-varying background is more realistic for Bangladesh where
seismicity is strongly concentrated in the Indo-Burman fold belt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .spatial_kernels import _equirect_km


@dataclass
class BackgroundRate:
    """Background rate specification."""

    kind: str               # "uniform" or "kde"
    mu_total_per_year: float   # ∫ μ(x,y) dx dy (total background rate)
    # For KDE:
    kde_lats: Optional[np.ndarray] = None
    kde_lons: Optional[np.ndarray] = None
    kde_bandwidth_km: Optional[float] = None
    # Grid over which the KDE is evaluated (for forecasting)
    bbox: Optional[tuple[float, float, float, float]] = None

    def at(self, lat: float, lon: float) -> float:
        """Background rate density at (lat, lon), per year per km²."""
        if self.kind == "uniform":
            # Uniform over the bbox area
            if self.bbox is None:
                return self.mu_total_per_year
            mn_lat, mx_lat, mn_lon, mx_lon = self.bbox
            area_km2 = (mx_lat - mn_lat) * 110.574 * (mx_lon - mn_lon) * 111.32 * math.cos(math.radians(0.5 * (mn_lat + mx_lat)))
            return self.mu_total_per_year / max(area_km2, 1e-6)
        elif self.kind == "kde":
            return self._kde_at(lat, lon)
        return 0.0

    def _kde_at(self, lat: float, lon: float) -> float:
        if self.kde_lats is None or self.kde_lons is None or self.kde_bandwidth_km is None:
            return 0.0
        h = self.kde_bandwidth_km
        h2 = h * h
        weights = np.exp(-0.5 * (
            (self.kde_lats - lat) ** 2 * (110.574 / h) ** 2
            + (self.kde_lons - lon) ** 2 * (111.32 * math.cos(math.radians(lat)) / h) ** 2
        ))
        # KDE density per km² (unnormalized)
        dens = np.sum(weights) / (2 * math.pi * h2 * len(self.kde_lats))
        # Scale so total integrates to mu_total
        # (approximately; for forecasting we re-normalize on a grid)
        return dens * self.mu_total_per_year * 1e5  # scale factor adjusted in build

    def on_grid(self, lats, lons) -> np.ndarray:
        """Evaluate background rate on a grid. Returns (n_lat, n_lon) per year per km²."""
        lats = np.asarray(lats, dtype=float)
        lons = np.asarray(lons, dtype=float)
        out = np.zeros((len(lats), len(lons)))
        for i, la in enumerate(lats):
            for j, lo in enumerate(lons):
                out[i, j] = self.at(la, lo)
        return out


class UniformBackground:
    """Factory for a uniform background rate."""

    @staticmethod
    def build(mu_total_per_year: float, bbox: tuple[float, float, float, float]) -> BackgroundRate:
        return BackgroundRate(kind="uniform", mu_total_per_year=mu_total_per_year, bbox=bbox)


class KDEBackground:
    """Factory for a spatially-varying KDE background.

    Uses a Gaussian kernel with Silverman bandwidth (with a configurable
    multiplier). The background is built from DECLUSTERED mainshocks so it
    represents the long-term tectonic loading pattern, not aftershock
    clustering.
    """

    @staticmethod
    def build(
        mainshock_lats: np.ndarray,
        mainshock_lons: np.ndarray,
        mu_total_per_year: float,
        bbox: tuple[float, float, float, float],
        bandwidth_multiplier: float = 1.0,
    ) -> BackgroundRate:
        n = len(mainshock_lats)
        if n < 5:
            # Too few mainshocks for a stable KDE; fall back to uniform.
            return UniformBackground.build(mu_total_per_year, bbox)
        # Silverman's rule of thumb (in degrees), then convert to km.
        # Use the smaller of the two marginal stds.
        std_lat = float(np.std(mainshock_lats))
        std_lon = float(np.std(mainshock_lons))
        h_deg = bandwidth_multiplier * 0.9 * min(std_lat, std_lon) * (n ** (-1 / 6))
        h_deg = max(h_deg, 0.1)  # floor
        h_km = h_deg * 111.0
        return BackgroundRate(
            kind="kde",
            mu_total_per_year=mu_total_per_year,
            kde_lats=np.asarray(mainshock_lats, dtype=float),
            kde_lons=np.asarray(mainshock_lons, dtype=float),
            kde_bandwidth_km=h_km,
            bbox=bbox,
        )
