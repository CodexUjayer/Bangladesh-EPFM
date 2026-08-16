"""Spatial + magnitude forecast: per cell, per threshold, per horizon.

For every grid cell and magnitude threshold, computes:
  - expected rate lambda
  - expected count over horizon
  - P(N >= 1 | horizon) = 1 - exp(-lambda * horizon)
  - 95% uncertainty interval on P

All forecasts are conditional on the observed catalog and the working Mc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..ingestion.schema import CanonicalEvent
from .poisson import HORIZON_YEARS
from .spatial import GridConfig, SpatialGrid, build_spatial_grid
from .uncertainty import probability_ci_from_rate_ci


@dataclass
class CellForecast:
    """Forecast for one (cell, threshold, horizon) combination."""

    cell_id: str
    lat_center: float
    lon_center: float
    threshold: float
    horizon: str
    horizon_years: float
    n_events_above_threshold: int
    exposure_years: float
    expected_rate_per_year: float
    expected_count: float                 # lambda * horizon (NOT a probability)
    probability_at_least_one: float       # 1 - exp(-lambda * horizon)
    probability_ci_lower: float
    probability_ci_upper: float
    low_statistics: bool
    notes: list[str] = field(default_factory=list)

    def to_row(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "lat_center": round(self.lat_center, 3),
            "lon_center": round(self.lon_center, 3),
            "threshold": self.threshold,
            "horizon": self.horizon,
            "horizon_years": round(self.horizon_years, 6),
            "n_events_above_threshold": self.n_events_above_threshold,
            "exposure_years": round(self.exposure_years, 3),
            "expected_rate_per_year": round(self.expected_rate_per_year, 6),
            "expected_count": round(self.expected_count, 6),
            "probability_at_least_one": round(self.probability_at_least_one, 6),
            "probability_ci_lower": round(self.probability_ci_lower, 6),
            "probability_ci_upper": round(self.probability_ci_upper, 6),
            "low_statistics": self.low_statistics,
            "notes": "; ".join(self.notes),
        }


@dataclass
class SpatialForecast:
    """Collection of cell forecasts across thresholds and horizons."""

    grid_config: GridConfig
    forecasts: list[CellForecast]
    thresholds: list[float]
    horizons: list[str]
    n_cells: int

    def filter(self, threshold: float, horizon: str) -> list[CellForecast]:
        return [f for f in self.forecasts
                if f.threshold == threshold and f.horizon == horizon]


def forecast_spatial(
    events: list[CanonicalEvent],
    thresholds: list[float],
    horizons: list[str],
    grid_config: GridConfig,
    exposure_years: Optional[float] = None,
) -> SpatialForecast:
    """Build the full spatial + magnitude forecast table.

    For each threshold, builds a spatial grid and computes per-cell forecasts
    for each horizon.

    Parameters
    ----------
    thresholds : e.g. [4.5, 5.0, 5.5, 6.0]
    horizons : e.g. ['24h', '7d', '30d', '90d', '1y']
    """
    all_forecasts: list[CellForecast] = []
    n_cells = 0
    for th in thresholds:
        grid = build_spatial_grid(
            events, threshold=th, config=grid_config,
            exposure_years=exposure_years,
            additional_thresholds=thresholds,
        )
        n_cells = len(grid.cells)
        for cell in grid.cells:
            for hname in horizons:
                hy = HORIZON_YEARS[hname]
                p = 1.0 - math.exp(-cell.rate_per_year * hy)
                plo, phi = probability_ci_from_rate_ci(
                    (cell.rate_ci_lower, cell.rate_ci_upper), hy
                )
                notes = list(cell.notes)
                if cell.low_statistics:
                    notes.append(
                        "Low-statistics cell: forecast probability has a wide CI; "
                        "do not interpret point estimate as precise."
                    )
                all_forecasts.append(CellForecast(
                    cell_id=cell.cell_id,
                    lat_center=cell.lat_center,
                    lon_center=cell.lon_center,
                    threshold=th,
                    horizon=hname,
                    horizon_years=hy,
                    n_events_above_threshold=cell.n_events,
                    exposure_years=cell.exposure_years,
                    expected_rate_per_year=cell.rate_per_year,
                    expected_count=cell.rate_per_year * hy,
                    probability_at_least_one=p,
                    probability_ci_lower=plo,
                    probability_ci_upper=phi,
                    low_statistics=cell.low_statistics,
                    notes=notes,
                ))

    return SpatialForecast(
        grid_config=grid_config,
        forecasts=all_forecasts,
        thresholds=thresholds,
        horizons=horizons,
        n_cells=n_cells,
    )
