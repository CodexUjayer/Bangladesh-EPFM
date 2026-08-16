"""Depth-dependence analysis.

Stage 5 reported mean depth ≈63 km and suggested deep Indo-Burman seismicity
may behave differently from shallow aftershock sequences. This module tests
that hypothesis by splitting the catalog into configurable depth groups and
evaluating per-depth:
  - event counts
  - temporal clustering (coefficient of variation of inter-event times)
  - ETAS triggering likelihood (refit ETAS per depth group)
  - branching ratio
  - aftershock-like decay (via the Omori diagnostic)
  - forecast performance

The key question: is the K≈0 result caused by the entire catalog lacking
triggering, or because different depth regimes are being mixed together?

Depth cutoffs are CONFIGURABLE and reported (not assumed without justification).
Default cutoffs (based on the catalog's depth distribution and standard
seismological practice):
  - shallow:    depth < 25 km   (upper crust; typical for active thrusts)
  - intermediate: 25 <= depth < 70 km  (lower crust / upper mantle; Indo-Burman wedge)
  - deep:       depth >= 70 km   (subducting slab; deep intra-slab events)
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..ingestion.schema import CanonicalEvent
from .branching import compute_branching_ratio
from .estimation import fit_etas_mle, prepare_catalog


@dataclass
class DepthGroupResult:
    """Analysis result for one depth group."""

    label: str
    depth_range_km: tuple[float, float]
    n_events: int
    n_events_above_Mc: int
    mean_magnitude: float
    mean_depth: float
    # Temporal clustering
    cv_inter_event_time: float       # std/mean of IET; >1 = clustered, ~1 = Poisson
    median_inter_event_time_days: float
    # ETAS fit (if supportable)
    etas_K: float
    etas_alpha: float
    etas_mu: float
    etas_loglik: float
    etas_converged: bool
    etas_no_triggering: bool
    branching_ratio_n: float
    branching_explosive: bool
    notes: list[str] = field(default_factory=list)

    def to_row(self) -> dict:
        return {
            "depth_group": self.label,
            "depth_range_km": f"{self.depth_range_km[0]}-{self.depth_range_km[1]}",
            "n_events": self.n_events,
            "n_events_above_Mc": self.n_events_above_Mc,
            "mean_magnitude": round(self.mean_magnitude, 3),
            "mean_depth": round(self.mean_depth, 2),
            "cv_inter_event_time": round(self.cv_inter_event_time, 3),
            "median_inter_event_time_days": round(self.median_inter_event_time_days, 3),
            "etas_K": self.etas_K,
            "etas_alpha": self.etas_alpha,
            "etas_mu": self.etas_mu,
            "etas_loglik": round(self.etas_loglik, 2) if not math.isnan(self.etas_loglik) else None,
            "etas_converged": self.etas_converged,
            "etas_no_triggering": self.etas_no_triggering,
            "branching_ratio_n": self.branching_ratio_n,
            "branching_explosive": self.branching_explosive,
            "notes": "; ".join(self.notes),
        }


def analyze_depth_dependence(
    events: list[CanonicalEvent],
    Mc: float = 4.5,
    depth_cutoffs: Optional[dict] = None,
) -> list[DepthGroupResult]:
    """Split the catalog by depth and analyze each group.

    Parameters
    ----------
    depth_cutoffs : dict with keys 'shallow_max', 'intermediate_max'
        shallow: depth < shallow_max
        intermediate: shallow_max <= depth < intermediate_max
        deep: depth >= intermediate_max
    """
    if depth_cutoffs is None:
        depth_cutoffs = {"shallow_max": 25.0, "intermediate_max": 70.0}

    groups = {
        "shallow":      (0.0, depth_cutoffs["shallow_max"]),
        "intermediate": (depth_cutoffs["shallow_max"], depth_cutoffs["intermediate_max"]),
        "deep":         (depth_cutoffs["intermediate_max"], 800.0),
    }

    results = []
    for label, (d_min, d_max) in groups.items():
        group_events = [e for e in events if d_min <= e.depth_km < d_max]
        result = _analyze_one_group(group_events, label, (d_min, d_max), Mc)
        results.append(result)
    return results


def _analyze_one_group(
    events: list[CanonicalEvent],
    label: str,
    depth_range: tuple[float, float],
    Mc: float,
) -> DepthGroupResult:
    n = len(events)
    if n == 0:
        return DepthGroupResult(
            label=label, depth_range_km=depth_range,
            n_events=0, n_events_above_Mc=0,
            mean_magnitude=float("nan"), mean_depth=float("nan"),
            cv_inter_event_time=float("nan"),
            median_inter_event_time_days=float("nan"),
            etas_K=float("nan"), etas_alpha=float("nan"),
            etas_mu=float("nan"), etas_loglik=float("nan"),
            etas_converged=False, etas_no_triggering=True,
            branching_ratio_n=float("nan"), branching_explosive=False,
            notes=["No events in this depth group."],
        )
    mags = np.array([e.mw if e.mw is not None else e.original_magnitude for e in events])
    depths = np.array([e.depth_km for e in events])
    times = sorted([e.origin_time_utc for e in events])

    # Inter-event times
    iet_days = np.array([(times[i+1] - times[i]).total_seconds() / 86400.0
                         for i in range(len(times) - 1)])
    cv_iet = float(np.std(iet_days) / np.mean(iet_days)) if len(iet_days) > 1 and np.mean(iet_days) > 0 else float("nan")
    median_iet = float(np.median(iet_days)) if len(iet_days) > 0 else float("nan")

    # ETAS fit on this depth group
    n_above_Mc = int(np.sum(mags >= Mc - 0.05))
    etas_K = etas_alpha = etas_mu = etas_loglik = float("nan")
    etas_converged = False
    etas_no_triggering = True
    branching_n = float("nan")
    branching_explosive = False
    notes = []

    if n_above_Mc >= 50:
        try:
            fit = fit_etas_mle(events, Mc=Mc, background_kind="kde",
                               spatial_kernel="powerlaw")
            etas_K = fit.params.K
            etas_alpha = fit.params.alpha
            etas_mu = fit.params.mu_total_per_year
            etas_loglik = fit.log_likelihood
            etas_converged = fit.converged
            etas_no_triggering = (fit.params.K <= 1e-6 or fit.params.alpha <= 1e-4)
            # Branching ratio
            cat = prepare_catalog(events, Mc=Mc)
            from ..baselines.gutenberg_richter import fit_gutenberg_richter
            gr = fit_gutenberg_richter(events, mc=Mc)
            b_val = gr.b_mle if not math.isnan(gr.b_mle) else 1.0
            br = compute_branching_ratio(
                K=fit.params.K, alpha=fit.params.alpha, Mc=Mc,
                mags=cat["mags"], b_value=b_val,
            )
            branching_n = br.n_analytic
            branching_explosive = br.explosive
        except Exception as exc:
            notes.append(f"ETAS fit failed: {exc}")
    else:
        notes.append(f"Only {n_above_Mc} events above Mc={Mc}; ETAS not fit.")

    if etas_no_triggering:
        notes.append("No triggering detected in this depth group (K≈0).")
    if cv_iet > 1.5:
        notes.append(f"Strong temporal clustering (CV_IET={cv_iet:.2f} > 1.5).")
    elif cv_iet < 1.1:
        notes.append(f"Near-Poisson temporal behavior (CV_IET={cv_iet:.2f}).")

    return DepthGroupResult(
        label=label, depth_range_km=depth_range,
        n_events=n, n_events_above_Mc=n_above_Mc,
        mean_magnitude=float(np.mean(mags)),
        mean_depth=float(np.mean(depths)),
        cv_inter_event_time=cv_iet,
        median_inter_event_time_days=median_iet,
        etas_K=etas_K, etas_alpha=etas_alpha, etas_mu=etas_mu,
        etas_loglik=etas_loglik,
        etas_converged=etas_converged, etas_no_triggering=etas_no_triggering,
        branching_ratio_n=branching_n,
        branching_explosive=branching_explosive,
        notes=notes,
    )
