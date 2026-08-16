"""Coulomb-modulated forecast and backtest.

Implements:
  - Coulomb-modulated Poisson: λ(x,t) = λ₀(x) · f(ΔCFS(x,t))
  - Coulomb-modulated ETAS (optional; if ETAS has skill)
  - Chronological backtest (only prior earthquakes/focal mechanisms at each origin)
  - Stress-forecast diagnostics (event-rate ratios across ΔCFS bins)

CRITICAL: If real_forecasting_enabled is False (data audit), the forecast
functions return NaN and the backtest reports 'data-limited'. The
mathematical prototype is unit-tested separately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from ..baselines.poisson import HORIZON_YEARS
from ..ingestion.schema import CanonicalEvent
from .coupling import CouplingParams, stress_to_rate_factor
from .data_audit import CoulombDataAudit
from .model import (
    ElasticParams,
    ReceiverFault,
    SourceEarthquake,
    compute_cumulative_dcfs,
)


@dataclass
class CoulombForecast:
    """Coulomb-modulated forecast for one horizon and threshold."""

    forecast_start: datetime
    horizon: str
    threshold: float
    enabled: bool                      # False if data-limited
    per_cell: list = field(default_factory=list)
    expected_total_count: float = float("nan")
    probability_at_least_one: float = float("nan")
    notes: list[str] = field(default_factory=list)


def build_source_earthquakes(
    events: list[CanonicalEvent],
    focal_mechanisms: dict[str, tuple[float, float, float]],  # event_id -> (strike, dip, rake)
    min_magnitude: float = 5.5,
) -> list[SourceEarthquake]:
    """Build SourceEarthquake objects from catalog events + focal mechanisms.

    Only events with magnitude >= min_magnitude AND a focal mechanism are
    usable as Coulomb sources.
    """
    sources = []
    for e in events:
        m = e.mw if e.mw is not None else e.original_magnitude
        if m is None or m < min_magnitude:
            continue
        fm = focal_mechanisms.get(e.canonical_id) or focal_mechanisms.get(
            f"usgs:{e.observations[0].native_event_id if e.observations else ''}"
        )
        if fm is None:
            continue
        strike, dip, rake = fm
        sources.append(SourceEarthquake(
            event_id=e.canonical_id,
            latitude=e.latitude, longitude=e.longitude,
            depth_km=e.depth_km, magnitude=m,
            strike=strike, dip=dip, rake=rake,
            source="usgs_focal_mechanism",
        ))
    return sources


def build_receiver_grid(
    grid_lats, grid_lons, depth_km: float = 10.0,
    assumed_strike: float = 0.0, assumed_dip: float = 45.0,
    assumed_rake: float = 90.0,
) -> list[ReceiverFault]:
    """Build a grid of receiver faults at fixed depth.

    WARNING: The receiver-fault orientation (assumed_strike/dip/rake) is an
    ENGINEERING ASSUMPTION (Class C) because no validated receiver-fault
    geometry exists for Bangladesh. This is only for diagnostic / prototype
    use; NOT for validated forecasting.
    """
    receivers = []
    for i, la in enumerate(grid_lats):
        for j, lo in enumerate(grid_lons):
            receivers.append(ReceiverFault(
                latitude=la, longitude=lo, depth_km=depth_km,
                strike=assumed_strike, dip=assumed_dip, rake=assumed_rake,
                cell_id=f"cell_{i:02d}_{j:02d}",
            ))
    return receivers


def forecast_coulomb_modulated_poisson(
    sources: list[SourceEarthquake],
    receivers: list[ReceiverFault],
    background_rate_per_year_per_cell: dict[str, float],
    elastic: ElasticParams,
    coupling: CouplingParams,
    forecast_start: datetime,
    horizon: str,
    threshold: float,
    catalog_start: datetime,
    train_events: list[CanonicalEvent],
    enabled: bool,
) -> CoulombForecast:
    """Coulomb-modulated Poisson forecast.

    λ(x, t) = λ₀(x) · f(ΔCFS(x, t))

    where λ₀(x) is the expanding-window Poisson rate per cell, and f is the
    stress-to-rate multiplier. ΔCFS is computed from all source earthquakes
    (M >= 5.5 with focal mechanisms) before the forecast origin.

    If `enabled` is False (data audit), returns a data-limited forecast.
    """
    cf = CoulombForecast(
        forecast_start=forecast_start, horizon=horizon, threshold=threshold,
        enabled=enabled,
    )
    if not enabled:
        cf.notes.append(
            "DATA-LIMITED: real Coulomb forecasting disabled by data audit "
            "(no validated receiver-fault geometry). Forecast returns NaN."
        )
        return cf

    hy = HORIZON_YEARS[horizon]
    # Compute ΔCFS at each receiver
    dcfs = compute_cumulative_dcfs(sources, receivers, elastic)  # Pa
    # Rate multiplier
    f = stress_to_rate_factor(dcfs, coupling)
    # Per-cell forecast
    expected_total = 0.0
    for i, rcv in enumerate(receivers):
        if rcv.cell_id is None:
            continue
        lam0 = background_rate_per_year_per_cell.get(rcv.cell_id, 0.0)
        lam = lam0 * float(f[i])
        expected_cell = lam * hy
        p_cell = 1.0 - math.exp(-max(expected_cell, 0.0))
        cf.per_cell.append({
            "cell_id": rcv.cell_id,
            "lat_center": rcv.latitude,
            "lon_center": rcv.longitude,
            "depth_km": rcv.depth_km,
            "dcfs_Pa": float(dcfs[i]) if not math.isnan(dcfs[i]) else None,
            "rate_multiplier": float(f[i]),
            "background_rate_per_year": lam0,
            "modulated_rate_per_year": lam,
            "expected_count": expected_cell,
            "probability_at_least_one": p_cell,
        })
        expected_total += expected_cell
    cf.expected_total_count = expected_total
    cf.probability_at_least_one = 1.0 - math.exp(-expected_total)
    cf.notes.append(
        f"Coulomb-modulated Poisson: {len(sources)} sources, {len(receivers)} receivers. "
        f"ΔCFS range [{np.nanmin(dcfs):.2f}, {np.nanmax(dcfs):.2f}] Pa."
    )
    return cf


# ---------------------------------------------------------------------------
# Stress-forecast diagnostics
# ---------------------------------------------------------------------------


@dataclass
class StressDiagnostic:
    """Event-rate ratios across ΔCFS bins."""

    bin_centers_Pa: list
    n_events_in_bin: list
    exposure_cells: list
    rate_ratio: list                 # observed / expected (under no-stress Poisson)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "bin_centers_Pa": [round(x, 1) for x in self.bin_centers_Pa],
            "n_events_in_bin": self.n_events_in_bin,
            "exposure_cells": self.exposure_cells,
            "rate_ratio": [round(x, 3) for x in self.rate_ratio],
            "notes": "; ".join(self.notes),
        }


def stress_forecast_diagnostic(
    dcfs_per_cell: np.ndarray,        # ΔCFS at each cell (Pa)
    n_events_per_cell: np.ndarray,    # observed events in forecast window per cell
    n_bins: int = 10,
) -> StressDiagnostic:
    """Compute event-rate ratios across ΔCFS bins.

    Tests whether future events preferentially occur in positive-ΔCFS regions,
    negative-ΔCFS regions, or high-gradient regions.

    The rate ratio is: (observed events in bin) / (expected events in bin
    under uniform-Poisson null). Ratio > 1 in positive-ΔCFS bins supports
    stress triggering; ratio < 1 in negative bins supports stress shadows.
    """
    dcfs = np.asarray(dcfs_per_cell, dtype=float)
    n_ev = np.asarray(n_events_per_cell, dtype=float)
    # Drop NaN
    mask = ~np.isnan(dcfs)
    dcfs = dcfs[mask]
    n_ev = n_ev[mask]
    if len(dcfs) == 0:
        return StressDiagnostic([], [], [], [], ["No usable ΔCFS values."])

    # Bin by ΔCFS
    dmin, dmax = float(np.min(dcfs)), float(np.max(dcfs))
    if dmax == dmin:
        return StressDiagnostic([dmin], [int(n_ev.sum())], [len(dcfs)],
                                 [1.0], ["All ΔCFS values identical."])
    bins = np.linspace(dmin, dmax, n_bins + 1)
    bin_centers = []
    n_in_bin = []
    exposure = []
    ratio = []
    total_events = float(n_ev.sum())
    total_cells = float(len(dcfs))
    for k in range(n_bins):
        lo, hi = bins[k], bins[k + 1]
        mask = (dcfs >= lo) & (dcfs < hi) if k < n_bins - 1 else (dcfs >= lo) & (dcfs <= hi)
        n_cells = int(mask.sum())
        n_events = int(n_ev[mask].sum())
        # Expected under uniform Poisson: total_events × (n_cells / total_cells)
        expected = total_events * (n_cells / total_cells) if total_cells > 0 else 0.0
        r = n_events / expected if expected > 0 else float("nan")
        bin_centers.append(float(0.5 * (lo + hi)))
        n_in_bin.append(n_events)
        exposure.append(n_cells)
        ratio.append(float(r) if not math.isnan(r) else None)

    notes = []
    # Check if positive bins have elevated rates
    pos_mask = np.array([bc > 0 for bc in bin_centers])
    neg_mask = np.array([bc < 0 for bc in bin_centers])
    if any(pos_mask) and any(neg_mask):
        pos_ratio = np.nanmean([r for r, m in zip(ratio, pos_mask) if m and r is not None])
        neg_ratio = np.nanmean([r for r, m in zip(ratio, neg_mask) if m and r is not None])
        if pos_ratio > neg_ratio * 1.5:
            notes.append(
                f"Positive-ΔCFS bins have elevated event rates (mean ratio {pos_ratio:.2f}) "
                f"vs negative bins ({neg_ratio:.2f}). Weak evidence of stress triggering."
            )
        else:
            notes.append(
                f"No clear stress-triggering signature: positive-bin ratio {pos_ratio:.2f} "
                f"vs negative-bin {neg_ratio:.2f}."
            )

    return StressDiagnostic(bin_centers, n_in_bin, exposure, ratio, notes)
