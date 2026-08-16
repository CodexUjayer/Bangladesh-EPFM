"""Feature engineering for ML forecasting — strictly causal (no leakage).

Every feature at forecast origin t uses ONLY events with origin_time < t.
Features are computed per (cell, forecast_origin) for the configured grid.

Feature groups (for ablation):
  ML-A: historical rate only
  ML-B: historical + temporal
  ML-C: historical + temporal + magnitude
  ML-D: historical + temporal + magnitude + spatial
  ML-E: historical + temporal + magnitude + spatial + depth
  ML-F: all available seismic features (includes clustering)
  ML-G: ML-F + Coulomb (DISABLED; reserved for future validated data)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from ..ingestion.schema import CanonicalEvent


# ---------------------------------------------------------------------------
# Grid configuration (matches Stage 4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MLGridConfig:
    """Spatial grid for ML forecasting (matches Stage 4 baseline)."""

    cell_size_deg: float = 1.0
    min_lat: float = 20.0
    max_lat: float = 28.0
    min_lon: float = 88.0
    max_lon: float = 96.0

    @property
    def n_lat(self) -> int:
        return int(round((self.max_lat - self.min_lat) / self.cell_size_deg))

    @property
    def n_lon(self) -> int:
        return int(round((self.max_lon - self.min_lon) / self.cell_size_deg))

    @property
    def n_cells(self) -> int:
        return self.n_lat * self.n_lon

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        lats = np.array([self.min_lat + (i + 0.5) * self.cell_size_deg
                         for i in range(self.n_lat)])
        lons = np.array([self.min_lon + (j + 0.5) * self.cell_size_deg
                         for j in range(self.n_lon)])
        return lats, lons

    def cell_id(self, i_lat: int, i_lon: int) -> str:
        return f"cell_{i_lat:02d}_{i_lon:02d}"

    def cell_of(self, lat: float, lon: float) -> tuple[int, int]:
        i_lat = min(int((lat - self.min_lat) / self.cell_size_deg), self.n_lat - 1)
        i_lon = min(int((lon - self.min_lon) / self.cell_size_deg), self.n_lon - 1)
        return max(i_lat, 0), max(i_lon, 0)


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------


# Feature group membership (for ablation)
FEATURE_GROUPS = {
    "ML-A": ["hist_rate"],   # historical rate only
    "ML-B": ["hist_rate", "temporal"],
    "ML-C": ["hist_rate", "temporal", "magnitude"],
    "ML-D": ["hist_rate", "temporal", "magnitude", "spatial"],
    "ML-E": ["hist_rate", "temporal", "magnitude", "spatial", "depth"],
    "ML-F": ["hist_rate", "temporal", "magnitude", "spatial", "depth", "clustering"],
    "ML-G": ["hist_rate", "temporal", "magnitude", "spatial", "depth", "clustering", "coulomb"],
}

# All individual feature names (computed once; ablation selects subsets)
ALL_FEATURE_NAMES = [
    # ML-A: historical rate
    "hist_rate_all", "hist_rate_above_45", "hist_rate_above_50", "hist_rate_above_55",
    # ML-B: temporal
    "n_prev_1d", "n_prev_7d", "n_prev_30d", "n_prev_90d", "n_prev_365d",
    "time_since_last_event_days", "time_since_last_m5_days", "time_since_last_m6_days",
    "rolling_rate_30d", "rolling_rate_90d",
    # ML-C: magnitude
    "max_mag_recent_30d", "max_mag_recent_90d", "mean_mag_recent_90d",
    "mag_var_recent_90d", "mag_p90_recent_90d",
    "n_above_45_recent_30d", "n_above_50_recent_30d", "n_above_55_recent_90d",
    "rolling_b_value_365d",
    # ML-D: spatial
    "local_density_50km", "local_density_100km",
    "neighbor_activity_8cells", "dist_to_last_event_km",
    "dist_to_last_m5_km", "dist_to_last_m6_km",
    "spatial_concentration",
    # ML-E: depth
    "mean_depth_recent_90d", "depth_var_recent_90d",
    "n_shallow_recent_90d", "n_intermediate_recent_90d", "n_deep_recent_90d",
    "depth_weighted_density",
    # ML-F: clustering
    "rate_acceleration_30d_vs_90d", "omori_time_since_m5", "omori_time_since_m6",
    "post_m5_activity_7d", "post_m6_activity_7d",
    "n_mainshocks_90d",
    # ML-G: Coulomb (DISABLED — always NaN)
    "dcfs_cumulative_Pa",
]

# Map feature name -> group tag
FEATURE_TO_GROUP = {}
for g in ["hist_rate"]:
    pass
for fn in ["hist_rate_all", "hist_rate_above_45", "hist_rate_above_50", "hist_rate_above_55"]:
    FEATURE_TO_GROUP[fn] = "hist_rate"
for fn in ["n_prev_1d", "n_prev_7d", "n_prev_30d", "n_prev_90d", "n_prev_365d",
           "time_since_last_event_days", "time_since_last_m5_days", "time_since_last_m6_days",
           "rolling_rate_30d", "rolling_rate_90d"]:
    FEATURE_TO_GROUP[fn] = "temporal"
for fn in ["max_mag_recent_30d", "max_mag_recent_90d", "mean_mag_recent_90d",
           "mag_var_recent_90d", "mag_p90_recent_90d",
           "n_above_45_recent_30d", "n_above_50_recent_30d", "n_above_55_recent_90d",
           "rolling_b_value_365d"]:
    FEATURE_TO_GROUP[fn] = "magnitude"
for fn in ["local_density_50km", "local_density_100km", "neighbor_activity_8cells",
           "dist_to_last_event_km", "dist_to_last_m5_km", "dist_to_last_m6_km",
           "spatial_concentration"]:
    FEATURE_TO_GROUP[fn] = "spatial"
for fn in ["mean_depth_recent_90d", "depth_var_recent_90d",
           "n_shallow_recent_90d", "n_intermediate_recent_90d", "n_deep_recent_90d",
           "depth_weighted_density"]:
    FEATURE_TO_GROUP[fn] = "depth"
for fn in ["rate_acceleration_30d_vs_90d", "omori_time_since_m5", "omori_time_since_m6",
           "post_m5_activity_7d", "post_m6_activity_7d", "n_mainshocks_90d"]:
    FEATURE_TO_GROUP[fn] = "clustering"
FEATURE_TO_GROUP["dcfs_cumulative_Pa"] = "coulomb"


def features_for_group(group: str) -> list[str]:
    """Return the list of feature names included in an ablation group."""
    tags = FEATURE_GROUPS.get(group, [])
    return [fn for fn in ALL_FEATURE_NAMES if FEATURE_TO_GROUP.get(fn) in tags]


# ---------------------------------------------------------------------------
# Feature computation (causal — no leakage)
# ---------------------------------------------------------------------------


@dataclass
class FeatureMatrix:
    """One (forecast_origin × cell) feature matrix with labels."""

    origin_time: datetime
    horizon: str
    threshold: float
    grid: MLGridConfig
    feature_names: list[str]
    X: np.ndarray                  # shape (n_cells, n_features)
    cell_ids: list[str]
    cell_lats: np.ndarray
    cell_lons: np.ndarray
    y: np.ndarray                  # binary: >=1 event in cell during [origin, origin+horizon)
    n_events_in_cell: np.ndarray   # count of events in cell during horizon
    # For Poisson baseline comparison
    poisson_rate_per_year: float    # expanding-window Poisson rate at this origin
    exposure_years: float


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))


def compute_features_at_origin(
    events: list[CanonicalEvent],
    origin_time: datetime,
    horizon: str,
    threshold: float,
    grid: MLGridConfig,
    catalog_start: datetime,
    horizon_days: float,
    cell_area_km2: float,
) -> FeatureMatrix:
    """Compute the full feature matrix for one forecast origin.

    STRICTLY CAUSAL: only events with origin_time < origin_time are used.
    """
    # Partition events: history (< origin) vs future ([origin, origin+horizon))
    horizon_td = timedelta(days=horizon_days)
    history = [e for e in events if e.origin_time_utc < origin_time]
    future = [e for e in events if origin_time <= e.origin_time_utc < origin_time + horizon_td]

    lats_grid, lons_grid = grid.cell_centers()
    n_cells = grid.n_cells
    X = np.zeros((n_cells, len(ALL_FEATURE_NAMES)))
    cell_ids = []
    cell_lats = np.zeros(n_cells)
    cell_lons = np.zeros(n_cells)
    y = np.zeros(n_cells, dtype=int)
    n_events = np.zeros(n_cells, dtype=int)

    # Precompute: history events with magnitude
    hist_events = []
    for e in history:
        m = e.mw if e.mw is not None else e.original_magnitude
        if m is None:
            continue
        hist_events.append((e.origin_time_utc, e.latitude, e.longitude, e.depth_km, m))
    hist_events.sort(key=lambda x: x[0])

    # Last M>=5 and M>=6 events (global, not per-cell)
    last_m5 = None
    last_m6 = None
    for t, la, lo, d, m in reversed(hist_events):
        if m >= 6.0 and last_m6 is None:
            last_m6 = (t, la, lo, m)
        if m >= 5.0 and last_m5 is None:
            last_m5 = (t, la, lo, m)
        if last_m5 is not None and last_m6 is not None:
            break

    # Exposure years for Poisson baseline
    exposure_years = max((origin_time - catalog_start).total_seconds() / (365.25 * 86400), 1e-6)

    # Poisson rate (expanding window) — per-origin, correct
    n_above_thresh_hist = sum(1 for _, _, _, _, m in hist_events if m >= threshold)
    poisson_rate = n_above_thresh_hist / exposure_years

    for idx in range(n_cells):
        i_lat = idx // grid.n_lon
        i_lon = idx % grid.n_lon
        cell_lat = lats_grid[i_lat]
        cell_lon = lons_grid[i_lon]
        cell_ids.append(grid.cell_id(i_lat, i_lon))
        cell_lats[idx] = cell_lat
        cell_lons[idx] = cell_lon

        # Find events in this cell (history)
        cell_events = [(t, la, lo, d, m) for t, la, lo, d, m in hist_events
                       if (grid.min_lat + i_lat * grid.cell_size_deg <= la <
                           grid.min_lat + (i_lat + 1) * grid.cell_size_deg) and
                          (grid.min_lon + i_lon * grid.cell_size_deg <= lo <
                           grid.min_lon + (i_lon + 1) * grid.cell_size_deg)]

        # Future events in this cell
        future_cell = [e for e in future
                       if (grid.min_lat + i_lat * grid.cell_size_deg <= e.latitude <
                           grid.min_lat + (i_lat + 1) * grid.cell_size_deg) and
                          (grid.min_lon + i_lon * grid.cell_size_deg <= e.longitude <
                           grid.min_lon + (i_lon + 1) * grid.cell_size_deg)]
        m_future = [(e.mw if e.mw is not None else e.original_magnitude) for e in future_cell]
        m_future = [m for m in m_future if m is not None and m >= threshold]
        n_events[idx] = len(m_future)
        y[idx] = 1 if len(m_future) > 0 else 0

        # Compute features
        feats = _compute_cell_features(
            cell_events, cell_lat, cell_lon, origin_time, hist_events,
            last_m5, last_m6, exposure_years, cell_area_km2, grid, i_lat, i_lon,
        )
        X[idx, :] = feats

    return FeatureMatrix(
        origin_time=origin_time, horizon=horizon, threshold=threshold, grid=grid,
        feature_names=list(ALL_FEATURE_NAMES), X=X, cell_ids=cell_ids,
        cell_lats=cell_lats, cell_lons=cell_lons, y=y, n_events_in_cell=n_events,
        poisson_rate_per_year=poisson_rate, exposure_years=exposure_years,
    )


def _compute_cell_features(
    cell_events, cell_lat, cell_lon, origin_time, all_hist_events,
    last_m5, last_m6, exposure_years, cell_area_km2, grid, i_lat, i_lon,
) -> np.ndarray:
    """Compute all features for one cell. STRICTLY CAUSAL."""
    feats = np.zeros(len(ALL_FEATURE_NAMES))
    fname_idx = {n: i for i, n in enumerate(ALL_FEATURE_NAMES)}

    def setf(name, val):
        feats[fname_idx[name]] = val if val is not None and not (isinstance(val, float) and (math.isnan(val) or math.isinf(val))) else 0.0

    # Time helpers
    now = origin_time
    mags = [m for _, _, _, _, m in cell_events]
    times = [t for t, _, _, _, _ in cell_events]

    # --- ML-A: historical rate ---
    n_all = len(cell_events)
    setf("hist_rate_all", n_all / exposure_years)
    setf("hist_rate_above_45", sum(1 for m in mags if m >= 4.5) / exposure_years)
    setf("hist_rate_above_50", sum(1 for m in mags if m >= 5.0) / exposure_years)
    setf("hist_rate_above_55", sum(1 for m in mags if m >= 5.5) / exposure_years)

    # --- ML-B: temporal ---
    def count_in_window(days):
        t0 = now - timedelta(days=days)
        return sum(1 for t in times if t >= t0)
    setf("n_prev_1d", count_in_window(1))
    setf("n_prev_7d", count_in_window(7))
    setf("n_prev_30d", count_in_window(30))
    setf("n_prev_90d", count_in_window(90))
    setf("n_prev_365d", count_in_window(365))
    if times:
        setf("time_since_last_event_days", (now - max(times)).total_seconds() / 86400.0)
    else:
        setf("time_since_last_event_days", exposure_years * 365.25)
    if last_m5 is not None:
        setf("time_since_last_m5_days", (now - last_m5[0]).total_seconds() / 86400.0)
    else:
        setf("time_since_last_m5_days", exposure_years * 365.25)
    if last_m6 is not None:
        setf("time_since_last_m6_days", (now - last_m6[0]).total_seconds() / 86400.0)
    else:
        setf("time_since_last_m6_days", exposure_years * 365.25)
    setf("rolling_rate_30d", count_in_window(30) / 30.0)
    setf("rolling_rate_90d", count_in_window(90) / 90.0)

    # --- ML-C: magnitude ---
    recent_30 = [m for t, _, _, _, m in cell_events if (now - t).total_seconds() / 86400.0 <= 30]
    recent_90 = [m for t, _, _, _, m in cell_events if (now - t).total_seconds() / 86400.0 <= 90]
    setf("max_mag_recent_30d", max(recent_30) if recent_30 else 0.0)
    setf("max_mag_recent_90d", max(recent_90) if recent_90 else 0.0)
    setf("mean_mag_recent_90d", float(np.mean(recent_90)) if recent_90 else 0.0)
    setf("mag_var_recent_90d", float(np.var(recent_90)) if len(recent_90) > 1 else 0.0)
    setf("mag_p90_recent_90d", float(np.percentile(recent_90, 90)) if recent_90 else 0.0)
    setf("n_above_45_recent_30d", sum(1 for m in recent_30 if m >= 4.5))
    setf("n_above_50_recent_30d", sum(1 for m in recent_30 if m >= 5.0))
    setf("n_above_55_recent_90d", sum(1 for m in recent_90 if m >= 5.5))
    # Rolling b-value (Aki-Utsu MLE) on last 365d above Mc=4.5
    recent_365 = [m for t, _, _, _, m in cell_events
                  if (now - t).total_seconds() / 86400.0 <= 365 and m >= 4.45]
    if len(recent_365) >= 20:
        mean_m = float(np.mean(recent_365))
        denom = mean_m - 4.4
        if denom > 0:
            setf("rolling_b_value_365d", math.log10(math.e) / denom)

    # --- ML-D: spatial ---
    # Local density within 50km and 100km (over all history)
    n_50 = 0; n_100 = 0
    for _, la, lo, _, _ in all_hist_events:
        d = _haversine_km(cell_lat, cell_lon, la, lo)
        if d <= 50: n_50 += 1
        if d <= 100: n_100 += 1
    setf("local_density_50km", n_50 / exposure_years)
    setf("local_density_100km", n_100 / exposure_years)
    # Neighbor activity (8 surrounding cells) in last 30d
    t0 = now - timedelta(days=30)
    neighbor_n = 0
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0: continue
            ni, nj = i_lat + di, i_lon + dj
            if 0 <= ni < grid.n_lat and 0 <= nj < grid.n_lon:
                for t, la, lo, _, _ in all_hist_events:
                    if t < t0: continue
                    if (grid.min_lat + ni * grid.cell_size_deg <= la <
                        grid.min_lat + (ni + 1) * grid.cell_size_deg) and \
                       (grid.min_lon + nj * grid.cell_size_deg <= lo <
                        grid.min_lon + (nj + 1) * grid.cell_size_deg):
                        neighbor_n += 1
    setf("neighbor_activity_8cells", neighbor_n)
    # Distance to last event in cell
    if cell_events:
        last_t, last_la, last_lo, _, _ = max(cell_events, key=lambda x: x[0])
        setf("dist_to_last_event_km", _haversine_km(cell_lat, cell_lon, last_la, last_lo))
    else:
        setf("dist_to_last_event_km", 500.0)
    if last_m5 is not None:
        setf("dist_to_last_m5_km", _haversine_km(cell_lat, cell_lon, last_m5[1], last_m5[2]))
    else:
        setf("dist_to_last_m5_km", 500.0)
    if last_m6 is not None:
        setf("dist_to_last_m6_km", _haversine_km(cell_lat, cell_lon, last_m6[1], last_m6[2]))
    else:
        setf("dist_to_last_m6_km", 500.0)
    # Spatial concentration (fraction of last 30d events within 50km)
    n_30_global = sum(1 for t, _, _, _, _ in all_hist_events if t >= t0)
    setf("spatial_concentration", n_50 / max(n_30_global, 1) if n_30_global > 0 else 0.0)

    # --- ML-E: depth ---
    depths_90 = [d for t, _, _, d, _ in cell_events
                 if (now - t).total_seconds() / 86400.0 <= 90]
    setf("mean_depth_recent_90d", float(np.mean(depths_90)) if depths_90 else 0.0)
    setf("depth_var_recent_90d", float(np.var(depths_90)) if len(depths_90) > 1 else 0.0)
    setf("n_shallow_recent_90d", sum(1 for d in depths_90 if d < 25))
    setf("n_intermediate_recent_90d", sum(1 for d in depths_90 if 25 <= d < 70))
    setf("n_deep_recent_90d", sum(1 for d in depths_90 if d >= 70))
    # Depth-weighted density (closer events weighted more)
    dw_density = 0.0
    for _, la, lo, d, _ in cell_events:
        dist = _haversine_km(cell_lat, cell_lon, la, lo)
        if dist > 0:
            dw_density += 1.0 / (1.0 + dist / 50.0)
    setf("depth_weighted_density", dw_density / exposure_years)

    # --- ML-F: clustering ---
    rate_30 = count_in_window(30) / 30.0
    rate_90 = count_in_window(90) / 90.0
    setf("rate_acceleration_30d_vs_90d", (rate_30 - rate_90) / max(rate_90, 1e-6))
    # Omori-like: time since last M5/M6 (inverse — short time = high triggering)
    if last_m5 is not None:
        t5 = (now - last_m5[0]).total_seconds() / 86400.0
        setf("omori_time_since_m5", 1.0 / (t5 + 1.0))
    if last_m6 is not None:
        t6 = (now - last_m6[0]).total_seconds() / 86400.0
        setf("omori_time_since_m6", 1.0 / (t6 + 1.0))
    # Post-mainshock activity (events within 7d after last M5/M6)
    if last_m5 is not None:
        post_window = last_m5[0] + timedelta(days=7)
        post_n = sum(1 for t, _, _, _, _ in all_hist_events
                     if last_m5[0] <= t < post_window)
        setf("post_m5_activity_7d", post_n)
    if last_m6 is not None:
        post_window = last_m6[0] + timedelta(days=7)
        post_n = sum(1 for t, _, _, _, _ in all_hist_events
                     if last_m6[0] <= t < post_window)
        setf("post_m6_activity_7d", post_n)
    # Number of mainshocks (M>=5) in last 90d
    t0_90 = now - timedelta(days=90)
    setf("n_mainshocks_90d", sum(1 for t, _, _, _, m in all_hist_events
                                  if t >= t0_90 and m >= 5.0))

    # --- ML-G: Coulomb (DISABLED) ---
    setf("dcfs_cumulative_Pa", 0.0)  # always 0; Coulomb disabled per Stage 6

    return feats
