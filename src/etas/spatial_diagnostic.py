"""Spatial aftershock diagnostic.

USER CORRECTION (Stage 5 validation):

  Measure the spatial concentration of events around qualifying mainshocks.
  Compare:
    - distance distribution after mainshocks
    - background spatial distribution
    - depth distribution

  This helps determine whether the failure of standard ETAS is temporal,
  spatial, depth-related, or a combination.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..ingestion.schema import CanonicalEvent
from .spatial_kernels import _equirect_km


# Log-spaced distance bins (km): from 1 km to 500 km.
LOG_DIST_BINS_KM = np.logspace(0, math.log10(500), 20)


@dataclass
class SpatialDiagnosticResult:
    """Result of the spatial aftershock diagnostic."""

    mainshock_threshold: float
    target_threshold: float
    n_mainshocks: int
    n_target_events: int
    # Distance distribution (post-mainshock within 30d)
    dist_bin_centers_km: list
    n_post_events_in_dist_bin: list
    post_events_exposure: list       # mainshock-exposure in each bin
    post_event_density_per_km2: list # normalized
    # Background spatial distribution (pairwise distances among all target events)
    bg_dist_bin_centers_km: list
    bg_pairwise_density: list
    # Depth distribution
    mainshock_depths: list
    target_event_depths: list
    post_event_depths: list          # within 30d and 100km of mainshocks
    # Spatial concentration ratio
    spatial_concentration_ratio: float  # post / background density at <50km
    spatial_clustering_detected: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mainshock_threshold": self.mainshock_threshold,
            "target_threshold": self.target_threshold,
            "n_mainshocks": self.n_mainshocks,
            "n_target_events": self.n_target_events,
            "spatial_concentration_ratio": round(self.spatial_concentration_ratio, 3),
            "spatial_clustering_detected": self.spatial_clustering_detected,
            "notes": "; ".join(self.notes),
            "dist_bins": [
                {"bin_center_km": round(bc, 2),
                 "n_post_events": int(ne),
                 "density_per_km2": round(d, 8)}
                for bc, ne, d in zip(self.dist_bin_centers_km,
                                     self.n_post_events_in_dist_bin,
                                     self.post_event_density_per_km2)
            ],
            "depth_summary": {
                "mainshock_mean_depth": round(float(np.mean(self.mainshock_depths)), 2) if self.mainshock_depths else None,
                "target_mean_depth": round(float(np.mean(self.target_event_depths)), 2) if self.target_event_depths else None,
                "post_mean_depth": round(float(np.mean(self.post_event_depths)), 2) if self.post_event_depths else None,
            },
        }


def compute_spatial_diagnostic(
    events: list[CanonicalEvent],
    mainshock_threshold: float = 5.0,
    target_threshold: float = 4.5,
    post_window_days: float = 30.0,
    max_dist_km: float = 500.0,
    dist_bins: Optional[np.ndarray] = None,
) -> SpatialDiagnosticResult:
    """Compute the spatial aftershock concentration diagnostic.

    For each mainshock, counts target events within `post_window_days` and
    `max_dist_km`, in log-spaced distance bins. Compares against the
    background pairwise distance distribution.
    """
    if dist_bins is None:
        dist_bins = LOG_DIST_BINS_KM

    events_sorted = sorted(events, key=lambda e: e.origin_time_utc)
    mainshocks = [e for e in events_sorted
                  if (e.mw if e.mw is not None else e.original_magnitude) >= mainshock_threshold]
    target_events = [e for e in events_sorted
                     if (e.mw if e.mw is not None else e.original_magnitude) >= target_threshold]
    n_ms = len(mainshocks)
    n_te = len(target_events)

    n_bins = len(dist_bins) - 1
    bin_centers = [math.sqrt(dist_bins[k] * dist_bins[k + 1]) for k in range(n_bins)]
    n_post_in_bin = np.zeros(n_bins)
    post_exposure = np.zeros(n_bins)   # mainshock-area exposure in each bin

    # Post-mainshock distance distribution
    from datetime import timedelta
    post_event_depths = []
    for ms in mainshocks:
        ms_time = ms.origin_time_utc
        for te in target_events:
            if te.origin_time_utc <= ms_time:
                continue
            lag_days = (te.origin_time_utc - ms_time).total_seconds() / 86400.0
            if lag_days > post_window_days:
                continue
            dist = _equirect_km(ms.latitude, ms.longitude, te.latitude, te.longitude)
            if dist > max_dist_km:
                continue
            for k in range(n_bins):
                if dist_bins[k] <= dist < dist_bins[k + 1]:
                    n_post_in_bin[k] += 1
                    break
            post_event_depths.append(te.depth_km)
        # Exposure: each mainshock contributes annulus area × time in each bin
        for k in range(n_bins):
            r_in, r_out = dist_bins[k], dist_bins[k + 1]
            annulus_area = math.pi * (r_out ** 2 - r_in ** 2)
            # exposure = area × post_window (only if mainshock + window within catalog)
            post_exposure[k] += annulus_area * min(post_window_days, 30.0)

    post_density = np.where(post_exposure > 0, n_post_in_bin / post_exposure, 0.0)

    # Background pairwise distance distribution (sample if too large)
    bg_dist_centers = list(bin_centers)
    bg_density = np.zeros(n_bins)
    if n_te > 1:
        # Sample up to 2000 target events for pairwise computation
        rng = np.random.default_rng(42)
        if n_te > 2000:
            sample_idx = rng.choice(n_te, 2000, replace=False)
            sampled = [target_events[i] for i in sample_idx]
        else:
            sampled = target_events
        pair_count = 0
        for i in range(len(sampled)):
            for j in range(i + 1, len(sampled)):
                d = _equirect_km(sampled[i].latitude, sampled[i].longitude,
                                 sampled[j].latitude, sampled[j].longitude)
                if d > max_dist_km:
                    continue
                for k in range(n_bins):
                    if dist_bins[k] <= d < dist_bins[k + 1]:
                        bg_density[k] += 1
                        break
                pair_count += 1
        # Normalize background density by annulus area
        for k in range(n_bins):
            r_in, r_out = dist_bins[k], dist_bins[k + 1]
            annulus_area = math.pi * (r_out ** 2 - r_in ** 2)
            bg_density[k] = bg_density[k] / (annulus_area * max(pair_count, 1))

    # Spatial concentration ratio: post / background density at <50km
    near_mask = np.array([bc < 50.0 for bc in bin_centers])
    post_near = float(np.sum(post_density[near_mask])) if near_mask.any() else 0.0
    bg_near = float(np.sum(bg_density[near_mask])) if near_mask.any() else 0.0
    concentration_ratio = post_near / bg_near if bg_near > 0 else float("nan")

    # Spatial clustering detected if post density > 2× background at <50km
    spatial_clustering = (
        not math.isnan(concentration_ratio) and concentration_ratio > 2.0
    )

    # Depths
    mainshock_depths = [ms.depth_km for ms in mainshocks]
    target_depths = [te.depth_km for te in target_events]

    notes = []
    if n_ms < 5:
        notes.append(f"Only {n_ms} mainshocks; high variance.")
    if spatial_clustering:
        notes.append(
            f"Spatial clustering DETECTED: post-mainshock density at <50km is "
            f"{concentration_ratio:.2f}× background. Events DO concentrate spatially "
            f"after mainshocks; standard ETAS spatial kernel may be misspecified."
        )
    else:
        notes.append(
            f"No strong spatial clustering: post/background density ratio at <50km = "
            f"{concentration_ratio:.2f}. Spatial concentration is weak or absent."
        )
    # Depth comparison
    if mainshock_depths and post_event_depths:
        ms_mean_d = float(np.mean(mainshock_depths))
        post_mean_d = float(np.mean(post_event_depths))
        if abs(ms_mean_d - post_mean_d) > 15.0:
            notes.append(
                f"Depth mismatch: mainshocks mean depth {ms_mean_d:.1f}km vs post-events "
                f"{post_mean_d:.1f}km. Different depth regimes may be mixed."
            )

    return SpatialDiagnosticResult(
        mainshock_threshold=mainshock_threshold,
        target_threshold=target_threshold,
        n_mainshocks=n_ms,
        n_target_events=n_te,
        dist_bin_centers_km=bin_centers,
        n_post_events_in_dist_bin=n_post_in_bin.tolist(),
        post_events_exposure=post_exposure.tolist(),
        post_event_density_per_km2=post_density.tolist(),
        bg_dist_bin_centers_km=bg_dist_centers,
        bg_pairwise_density=bg_density.tolist(),
        mainshock_depths=mainshock_depths,
        target_event_depths=target_depths,
        post_event_depths=post_event_depths,
        spatial_concentration_ratio=concentration_ratio,
        spatial_clustering_detected=spatial_clustering,
        notes=notes,
    )
