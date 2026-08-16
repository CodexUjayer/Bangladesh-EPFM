"""Spatial baseline: configurable grid, per-cell rates, low-statistics guards.

Divides the study region into a grid and estimates, per cell:
  - event rate lambda_cell = N_cell / T
  - magnitude distribution (mean, max, count above each threshold)
  - event density (per km^2)
  - P(N >= 1 | horizon) per cell

CRITICAL GUARDS AGAINST FALSE PRECISION:
  - Default grid is COARSE (1.0 deg) — finer is NOT automatically better.
  - Cells with very few events (N < min_events, default 5) are flagged as
    low-statistics; their rate CIs are wide and the cell is not used for
    high-resolution mapping without pooling.
  - Cells with zero events get a Jeffreys upper-bound rate (not zero),
    reflecting genuine residual uncertainty.
  - We do NOT produce misleading high-resolution maps from cells containing
    very few events; low-stat cells are visually distinguished.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..ingestion.schema import CanonicalEvent
from .uncertainty import poisson_rate_ci_jeffreys, poisson_rate_ci_garwood

# Approx km per degree at mid-latitude of the study region (~24 deg N).
KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON_AT_24N = 111.320 * math.cos(math.radians(24.0))


@dataclass(frozen=True)
class GridConfig:
    """Configuration for the spatial grid."""

    cell_size_deg: float = 1.0     # default coarse; finer is NOT auto-better
    min_lat: float = 20.0
    max_lat: float = 28.0
    min_lon: float = 88.0
    max_lon: float = 96.0
    min_events_for_stable_rate: int = 5  # below this, flag low-statistics


@dataclass
class GridCell:
    """One grid cell with its baseline statistics."""

    cell_id: str
    i_lat: int
    i_lon: int
    lat_center: float
    lon_center: float
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    area_km2: float
    n_events: int
    n_events_above_threshold: dict   # threshold -> count
    exposure_years: float
    rate_per_year: float             # lambda_cell = N / T
    rate_ci_lower: float
    rate_ci_upper: float
    rate_density_per_km2_per_year: float
    mean_magnitude: Optional[float]
    max_magnitude: Optional[float]
    low_statistics: bool             # True if N < min_events
    notes: list[str] = field(default_factory=list)

    def probability_at_least_one(self, horizon_years: float) -> float:
        return 1.0 - math.exp(-self.rate_per_year * horizon_years)

    def probability_ci(self, horizon_years: float) -> tuple[float, float]:
        from .uncertainty import probability_ci_from_rate_ci
        return probability_ci_from_rate_ci(
            (self.rate_ci_lower, self.rate_ci_upper), horizon_years
        )

    def to_row(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "lat_center": round(self.lat_center, 3),
            "lon_center": round(self.lon_center, 3),
            "area_km2": round(self.area_km2, 1),
            "n_events": self.n_events,
            "exposure_years": round(self.exposure_years, 3),
            "rate_per_year": round(self.rate_per_year, 6),
            "rate_ci_lower": round(self.rate_ci_lower, 6),
            "rate_ci_upper": round(self.rate_ci_upper, 6),
            "rate_density_per_km2_per_year": round(self.rate_density_per_km2_per_year, 9),
            "mean_magnitude": round(self.mean_magnitude, 3) if self.mean_magnitude is not None else None,
            "max_magnitude": round(self.max_magnitude, 3) if self.max_magnitude is not None else None,
            "low_statistics": self.low_statistics,
            "notes": "; ".join(self.notes),
        }


@dataclass
class SpatialGrid:
    """Collection of grid cells with full baseline statistics."""

    config: GridConfig
    cells: list[GridCell]
    n_events_total: int
    n_cells_with_events: int
    n_cells_low_statistics: int
    exposure_years: float
    threshold: float
    notes: list[str] = field(default_factory=list)


def build_spatial_grid(
    events: list[CanonicalEvent],
    threshold: float,
    config: GridConfig,
    exposure_years: Optional[float] = None,
    additional_thresholds: Optional[list[float]] = None,
) -> SpatialGrid:
    """Build the spatial grid with per-cell baseline statistics.

    Parameters
    ----------
    events : list of CanonicalEvent
    threshold : float
        Primary magnitude threshold for the baseline rate.
    config : GridConfig
    exposure_years : optional float; if None, computed from catalog span.
    additional_thresholds : optional list of thresholds for per-cell counts
        (used for multi-threshold reporting without re-running the grid).
    """
    # Exposure time
    if exposure_years is None:
        times = [e.origin_time_utc for e in events]
        exposure_years = (max(times) - min(times)).total_seconds() / (365.25 * 86400)
    T = max(exposure_years, 1e-9)

    # Grid dimensions
    n_lat = int(round((config.max_lat - config.min_lat) / config.cell_size_deg))
    n_lon = int(round((config.max_lon - config.min_lon) / config.cell_size_deg))

    # Bin events into cells, using Mw where available else original magnitude.
    # We track events above the PRIMARY threshold and above each additional threshold.
    all_thresholds = [threshold] + (additional_thresholds or [])
    # cell_index -> list of (magnitude, lat, lon)
    cell_events: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
    for e in events:
        m = e.mw if e.mw is not None else e.original_magnitude
        if m is None:
            continue
        if not (config.min_lat <= e.latitude <= config.max_lat and
                config.min_lon <= e.longitude <= config.max_lon):
            continue
        i_lat = min(int((e.latitude - config.min_lat) / config.cell_size_deg), n_lat - 1)
        i_lon = min(int((e.longitude - config.min_lon) / config.cell_size_deg), n_lon - 1)
        cell_events.setdefault((i_lat, i_lon), []).append((m, e.latitude, e.longitude))

    cells: list[GridCell] = []
    n_with = 0
    n_low = 0
    n_total = 0
    for i_lat in range(n_lat):
        for i_lon in range(n_lon):
            lat_min = config.min_lat + i_lat * config.cell_size_deg
            lat_max = lat_min + config.cell_size_deg
            lon_min = config.min_lon + i_lon * config.cell_size_deg
            lon_max = lon_min + config.cell_size_deg
            lat_c = 0.5 * (lat_min + lat_max)
            lon_c = 0.5 * (lon_min + lon_max)
            # Cell area (approx, using mid-latitude)
            dlat_km = config.cell_size_deg * KM_PER_DEG_LAT
            dlon_km = config.cell_size_deg * KM_PER_DEG_LON_AT_24N
            area_km2 = dlat_km * dlon_km

            evs = cell_events.get((i_lat, i_lon), [])
            mags_all = [m for (m, _, _) in evs]
            mags_above_primary = [m for m in mags_all if m >= threshold]
            n_above = len(mags_above_primary)
            n_total += n_above

            # Counts above each additional threshold
            n_above_thresholds = {}
            for th in all_thresholds:
                n_above_thresholds[th] = sum(1 for m in mags_all if m >= th)

            # Rate CI: use Jeffreys for robustness with small N (incl. N=0)
            ci_lo, ci_hi = poisson_rate_ci_jeffreys(n_above, T)
            rate = n_above / T
            rate_density = rate / area_km2 if area_km2 > 0 else float("nan")

            low_stat = n_above < config.min_events_for_stable_rate
            if n_above > 0:
                n_with += 1
            if low_stat:
                n_low += 1

            notes = []
            if low_stat:
                notes.append(
                    f"Low-statistics cell (N={n_above} < {config.min_events_for_stable_rate}); "
                    "rate CI is wide; do not interpret point rate as precise."
                )
            if n_above == 0:
                notes.append(
                    "Zero observed events above threshold; rate upper bound "
                    f"(Jeffreys 95%) = {ci_hi:.4f}/yr, not zero."
                )

            cells.append(GridCell(
                cell_id=f"cell_{i_lat:02d}_{i_lon:02d}",
                i_lat=i_lat, i_lon=i_lon,
                lat_center=lat_c, lon_center=lon_c,
                lat_min=lat_min, lat_max=lat_max,
                lon_min=lon_min, lon_max=lon_max,
                area_km2=area_km2,
                n_events=n_above,
                n_events_above_threshold=n_above_thresholds,
                exposure_years=T,
                rate_per_year=rate,
                rate_ci_lower=ci_lo,
                rate_ci_upper=ci_hi,
                rate_density_per_km2_per_year=rate_density,
                mean_magnitude=float(np.mean(mags_above_primary)) if mags_above_primary else None,
                max_magnitude=float(np.max(mags_above_primary)) if mags_above_primary else None,
                low_statistics=low_stat,
                notes=notes,
            ))

    grid = SpatialGrid(
        config=config,
        cells=cells,
        n_events_total=n_total,
        n_cells_with_events=n_with,
        n_cells_low_statistics=n_low,
        exposure_years=T,
        threshold=threshold,
        notes=[
            f"Grid: {n_lat}x{n_lon} = {n_lat*n_lon} cells at {config.cell_size_deg} deg.",
            f"Coarse grid chosen deliberately; finer resolution is NOT automatically better.",
            f"{n_low} of {n_lat*n_lon} cells flagged low-statistics (N < {config.min_events_for_stable_rate}).",
            f"Threshold: M>={threshold} (conservative working modeling threshold).",
            f"Exposure: {T:.2f} years (catalog span).",
        ],
    )
    return grid
