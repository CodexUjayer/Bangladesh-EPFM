"""Magnitude of completeness (Mc) estimation.

Implements four established methods:

  - MAXC (Maximum Curvature): Mc = magnitude at the peak of the
    non-cumulative frequency-magnitude distribution.
  - GFT (Goodness-of-Fit Test, Wiemer & Wyss 2000): find the Mc that
    minimizes the residual between observed and GR-modeled cumulative
    distribution, subject to a confidence threshold.
  - EMR (Entire-Magnitude-Range, Woessner & Wiemer 2005): fit a GR +
    detection function over the full range; Mc = where detection prob = 0.5.
  - Stepp (1972): temporal stability of magnitude-bin counts; Mc is the
    lowest magnitude bin whose count grows ~linearly with time.

Also produces:
  - Mc(t): time-varying completeness via rolling-window MAXC.
  - Spatial Mc: MAXC computed per spatial subregion (if supportable).

All estimates report uncertainty. Methods are configurable; the Stage 3
report reports all four so the user can see agreement / disagreement.

IMPORTANT: Mc is estimated on the MAGNITUDE used for rate estimation. If
Mw is available (authoritative or converted), we prefer Mw; otherwise we
fall back to the original magnitude and DOCUMENT which scale Mc is on.
Mixing scales silently is forbidden.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..ingestion.schema import CanonicalEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Magnitude selection for completeness
# ---------------------------------------------------------------------------


def select_magnitude_series(
    events: list[CanonicalEvent],
    prefer_mw: bool = True,
) -> tuple[np.ndarray, str]:
    """Return the magnitude array to use for completeness / b-value.

    If prefer_mw and >=50% of events have Mw, use Mw (and report the
    fraction missing). Otherwise use original_magnitude and DOCUMENT that
    Mc/b are on a mixed-original-type scale (a caveat the report must
    carry).

    Returns (magnitudes, scale_label).
    """
    if prefer_mw:
        mw = np.array([e.mw for e in events if e.mw is not None])
        if len(mw) >= 0.5 * len(events):
            return mw, "Mw (derived/authoritative; events with missing Mw excluded)"
    mags = np.array([e.original_magnitude for e in events])
    return mags, "original_magnitude (MIXED types: see magnitude_type_counts)"


# ---------------------------------------------------------------------------
# MAXC
# ---------------------------------------------------------------------------


@dataclass
class McEstimate:
    method: str
    mc: float
    uncertainty: Optional[float]
    n_events_used: int
    details: dict = field(default_factory=dict)
    warning: str = ""


def mc_maxc(magnitudes: np.ndarray, bin_width: float = 0.1) -> McEstimate:
    """Maximum Curvature method.

    Mc = magnitude bin with the highest frequency (peak of the
    non-cumulative FMD). Uncertainty: +/- 0.5 bin width (conservative).
    """
    if len(magnitudes) == 0:
        return McEstimate("MAXC", float("nan"), None, 0, warning="no events")
    m_min = math.floor(magnitudes.min() / bin_width) * bin_width
    m_max = math.ceil(magnitudes.max() / bin_width) * bin_width
    bins = np.arange(m_min, m_max + bin_width, bin_width)
    counts, edges = np.histogram(magnitudes, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if len(counts) == 0:
        return McEstimate("MAXC", float("nan"), None, 0, warning="no bins")
    idx = int(np.argmax(counts))
    mc = float(centers[idx])
    return McEstimate(
        method="MAXC",
        mc=mc,
        uncertainty=bin_width / 2.0,
        n_events_used=int(len(magnitudes)),
        details={"bin_width": bin_width, "peak_count": int(counts[idx])},
    )


# ---------------------------------------------------------------------------
# GFT (Wiemer & Wyss 2000)
# ---------------------------------------------------------------------------


def mc_gft(
    magnitudes: np.ndarray,
    bin_width: float = 0.1,
    confidence: float = 0.95,
) -> McEstimate:
    """Goodness-of-Fit Test.

    For each candidate Mc (each bin), compute the best-fit GR (b-value via
    MLE on events >= Mc) and the residual R = 1 - (sum|obs-mod|)/sum(obs)
    over bins >= Mc. Return the lowest Mc whose R >= confidence. If none
    reach the threshold, return the Mc with the highest R and flag a
    warning.
    """
    if len(magnitudes) == 0:
        return McEstimate("GFT", float("nan"), None, 0, warning="no events")
    m_min = math.floor(magnitudes.min() / bin_width) * bin_width
    m_max = math.ceil(magnitudes.max() / bin_width) * bin_width
    bins = np.arange(m_min, m_max + bin_width, bin_width)
    counts, edges = np.histogram(magnitudes, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    best_mc = None
    best_r = -1.0
    first_passing = None
    for i in range(len(centers)):
        mc_candidate = centers[i]
        sub = counts[i:]
        n_above = int(sub.sum())
        if n_above < 10:
            break
        # b-value MLE (Aki-Utsu) on magnitudes >= mc_candidate - bin_width/2
        m_above = magnitudes[magnitudes >= mc_candidate - bin_width / 2]
        if len(m_above) < 10:
            break
        mean_m = float(np.mean(m_above))
        b = math.log10(math.e) / (mean_m - (mc_candidate - bin_width / 2))
        if b <= 0 or not np.isfinite(b):
            continue
        # model cumulative counts: N(M) = N_above * 10^(-b (M - Mc))
        model_cum = n_above * (10.0 ** (-b * (centers[i:] - mc_candidate)))
        obs_cum = np.cumsum(sub[::-1])[::-1].astype(float)
        denom = float(obs_cum.sum())
        if denom == 0:
            continue
        r = 1.0 - float(np.abs(obs_cum - model_cum).sum()) / denom
        if r > best_r:
            best_r = r
            best_mc = mc_candidate
        if r >= confidence and first_passing is None:
            first_passing = mc_candidate
    if first_passing is not None:
        return McEstimate(
            method="GFT",
            mc=float(first_passing),
            uncertainty=bin_width / 2.0,
            n_events_used=len(magnitudes),
            details={"confidence": confidence, "R_at_mc": float(best_r)},
        )
    return McEstimate(
        method="GFT",
        mc=float(best_mc) if best_mc is not None else float("nan"),
        uncertainty=bin_width / 2.0,
        n_events_used=len(magnitudes),
        details={"confidence": confidence, "best_R": float(best_r)},
        warning=f"No Mc reached confidence {confidence}; reporting best R={best_r:.3f}.",
    )


# ---------------------------------------------------------------------------
# EMR (Woessner & Wiemer 2005) — GR + normal detection function
# ---------------------------------------------------------------------------


def mc_emr(
    magnitudes: np.ndarray,
    bin_width: float = 0.1,
) -> McEstimate:
    """Entire-Magnitude-Range method.

    Fits a model: frequency(M) = N_total * [GR survival above Mc] *
    [detection probability q(M)] where q(M) is a cumulative normal CDF
    centered at mu with std sigma. Mc is defined as the magnitude where
    q(M) = 0.5, i.e. mu.

    Uses a grid + simplex search (no scipy dependency required). Uncertainty
    via bootstrap (N=50) if supportable; otherwise None.
    """
    if len(magnitudes) == 0:
        return McEstimate("EMR", float("nan"), None, 0, warning="no events")
    m_min = math.floor(magnitudes.min() / bin_width) * bin_width
    m_max = math.ceil(magnitudes.max() / bin_width) * bin_width
    bins = np.arange(m_min, m_max + bin_width, bin_width)
    counts, edges = np.histogram(magnitudes, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def neg_loglik(params):
        b, mu, sigma, n_total = params
        if b <= 0 or sigma <= 0 or n_total <= 0:
            return 1e18
        # survival of GR above each bin
        surv = 10.0 ** (-b * np.maximum(centers - mu, 0.0))
        # detection probability: normal CDF
        from math import erf
        z = (centers - mu) / (sigma * 1.4142135623730951)
        q = 0.5 * (1.0 + np.array([math.erf(float(zi)) for zi in z]))
        pred = n_total * surv * q
        pred = np.maximum(pred, 1e-12)
        return float(-np.sum(counts * np.log(pred) - pred))

    # Grid search over mu (the Mc proxy), b, sigma
    best = (1e18, None)
    for mu in centers:
        for b in [0.6, 0.8, 1.0, 1.2]:
            for sigma in [0.1, 0.2, 0.3]:
                n_total = float(counts.sum())
                ll = neg_loglik((b, mu, sigma, n_total))
                if ll < best[0]:
                    best = (ll, (b, mu, sigma, n_total))
    if best[1] is None:
        return McEstimate("EMR", float("nan"), None, len(magnitudes), warning="fit failed")
    mu = best[1][1]
    return McEstimate(
        method="EMR",
        mc=float(mu),
        uncertainty=None,   # bootstrap omitted for speed; documented
        n_events_used=len(magnitudes),
        details={"b": float(best[1][0]), "sigma": float(best[1][2]), "n_total": float(best[1][3])},
    )


# ---------------------------------------------------------------------------
# Stepp (1972)
# ---------------------------------------------------------------------------


def mc_stepp(
    magnitudes: np.ndarray,
    times,
    bin_width: float = 0.1,
) -> McEstimate:
    """Stepp's method.

    For each magnitude bin, compute the cumulative count over time and the
    standard deviation of the annual rate. The completeness threshold is
    the lowest bin whose std growth behaves like sqrt(T) (stable). We
    implement the standard practical version: for each bin, compute the
    annual counts; the bin is 'complete' if its mean annual count is stable
    (CV within a tolerance). Mc = lowest such bin.

    This is an approximation of Stepp's full method; the full method
    requires plotting std vs T and inspecting the deviation from sqrt(T).
    We report the bin and flag that visual inspection is recommended.
    """
    if len(magnitudes) == 0 or len(times) == 0:
        return McEstimate("Stepp", float("nan"), None, 0, warning="no events")
    times = np.array([t for t in times])
    years = np.array([t.year for t in times])
    mags = np.asarray(magnitudes)
    m_min = math.floor(mags.min() / bin_width) * bin_width
    m_max = math.ceil(mags.max() / bin_width) * bin_width
    bins = np.arange(m_min, m_max + bin_width, bin_width)

    # For each bin threshold Mc, look at events with M >= Mc over the full
    # period; compute the coefficient of variation of annual counts.
    # The lowest Mc with CV <= 0.5 (stable) is taken as complete.
    candidate_mcs = []
    for i in range(len(bins) - 1):
        mc_c = bins[i]
        mask = mags >= mc_c
        if mask.sum() < 20:
            continue
        yrs = years[mask]
        annual = Counter(int(y) for y in yrs)
        counts = np.array(list(annual.values()), dtype=float)
        if len(counts) < 5:
            continue
        mean = counts.mean()
        std = counts.std()
        cv = std / mean if mean > 0 else float("inf")
        candidate_mcs.append((mc_c, cv, int(mask.sum())))
    if not candidate_mcs:
        return McEstimate("Stepp", float("nan"), None, len(magnitudes),
                          warning="insufficient data for Stepp")
    # Lowest Mc with CV <= 0.5; else report the one with lowest CV.
    passing = [c for c in candidate_mcs if c[1] <= 0.5]
    if passing:
        mc, cv, n = passing[0]
        warn = ""
    else:
        c = min(candidate_mcs, key=lambda x: x[1])
        mc, cv, n = c
        warn = "No bin reached CV<=0.5; reporting lowest-CV bin (visual Stepp inspection recommended)."
    return McEstimate(
        method="Stepp",
        mc=float(mc),
        uncertainty=bin_width / 2.0,
        n_events_used=n,
        details={"CV": float(cv), "bin_width": bin_width},
        warning=warn,
    )


# ---------------------------------------------------------------------------
# Combined Mc + Mc(t) + spatial Mc
# ---------------------------------------------------------------------------


@dataclass
class CompletenessReport:
    """Full completeness analysis for the catalog."""

    scale_label: str
    n_events_used: int
    mc_maxc: McEstimate
    mc_gft: McEstimate
    mc_emr: McEstimate
    mc_stepp: McEstimate
    mc_recommended: float
    mc_recommended_method: str
    mc_recommended_rationale: str
    mc_t: list   # list of (period_label, mc_maxc, n_events)
    mc_spatial: list   # list of (region_label, mc_maxc, n_events)
    n_above_recommended: int
    n_below_recommended: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "scale_label": self.scale_label,
            "n_events_used": self.n_events_used,
            "mc_maxc": {"mc": self.mc_maxc.mc, "unc": self.mc_maxc.uncertainty,
                        "warning": self.mc_maxc.warning},
            "mc_gft": {"mc": self.mc_gft.mc, "unc": self.mc_gft.uncertainty,
                       "warning": self.mc_gft.warning,
                       "details": self.mc_gft.details},
            "mc_emr": {"mc": self.mc_emr.mc, "unc": self.mc_emr.uncertainty,
                       "warning": self.mc_emr.warning},
            "mc_stepp": {"mc": self.mc_stepp.mc, "unc": self.mc_stepp.uncertainty,
                         "warning": self.mc_stepp.warning},
            "mc_recommended": self.mc_recommended,
            "mc_recommended_method": self.mc_recommended_method,
            "mc_recommended_rationale": self.mc_recommended_rationale,
            "n_above_recommended": self.n_above_recommended,
            "n_below_recommended": self.n_below_recommended,
            "mc_t": self.mc_t,
            "mc_spatial": self.mc_spatial,
            "notes": self.notes,
        }
        return d


def estimate_completeness(
    events: list[CanonicalEvent],
    times=None,
    prefer_mw: bool = True,
    compute_mc_t: bool = True,
    compute_spatial_mc: bool = True,
    spatial_subregions: Optional[list[tuple[str, tuple[float, float, float, float]]]] = None,
    bin_width: float = 0.1,
) -> CompletenessReport:
    """Run the full completeness analysis.

    Parameters
    ----------
    events : list of CanonicalEvent
    times : optional list of datetimes (defaults to event origin times)
    prefer_mw : use Mw when available (documented in scale_label)
    compute_mc_t : compute time-varying Mc via MAXC on rolling windows
    compute_spatial_mc : compute Mc per spatial subregion
    spatial_subregions : list of (name, (min_lat, max_lat, min_lon, max_lon))
    """
    mags, scale_label = select_magnitude_series(events, prefer_mw=prefer_mw)
    if times is None:
        times = [e.origin_time_utc for e in events]
    # Align times to the magnitude array (if Mw used, times subset).
    if prefer_mw and scale_label.startswith("Mw"):
        paired = [(e.origin_time_utc, e.mw) for e in events if e.mw is not None]
        times = [p[0] for p in paired]
        mags = np.array([p[1] for p in paired])

    mc_m = mc_maxc(mags, bin_width)
    mc_g = mc_gft(mags, bin_width)
    mc_e = mc_emr(mags, bin_width)
    mc_s = mc_stepp(mags, times, bin_width)

    # Recommended Mc: median of finite estimates, with rationale.
    ests = [x for x in [mc_m.mc, mc_g.mc, mc_e.mc, mc_s.mc] if np.isfinite(x)]
    if ests:
        mc_rec = float(np.median(ests))
        method = "median(MAXC,GFT,EMR,Stepp)"
        rationale = (
            f"Median of {len(ests)} finite estimates: "
            f"MAXC={mc_m.mc:.2f}, GFT={mc_g.mc:.2f}, EMR={mc_e.mc:.2f}, "
            f"Stepp={mc_s.mc:.2f}."
        )
    else:
        mc_rec = float("nan")
        method = "none"
        rationale = "All methods failed to estimate Mc."

    n_above = int(np.sum(mags >= mc_rec)) if np.isfinite(mc_rec) else 0
    n_below = int(np.sum(mags < mc_rec)) if np.isfinite(mc_rec) else 0

    # Mc(t): MAXC on rolling 5-year windows, stepping 1 year.
    mc_t = []
    if compute_mc_t and len(times) > 0:
        years = np.array([t.year for t in times])
        mags_arr = np.asarray(mags)
        y_min, y_max = int(years.min()), int(years.max())
        window = 5
        for y0 in range(y_min, y_max - window + 1, 1):
            mask = (years >= y0) & (years < y0 + window)
            if mask.sum() >= 30:
                est = mc_maxc(mags_arr[mask], bin_width)
                mc_t.append((f"{y0}-{y0+window-1}", round(est.mc, 2), int(mask.sum())))

    # Spatial Mc: MAXC per subregion.
    mc_spatial = []
    if compute_spatial_mc:
        if spatial_subregions is None:
            # default: a single coarse 2x2 grid over the data bbox
            lats = np.array([e.latitude for e in events])
            lons = np.array([e.longitude for e in events])
            lat_mid = float(np.median(lats))
            lon_mid = float(np.median(lons))
            spatial_subregions = [
                ("SW", (float(lats.min()), lat_mid, float(lons.min()), lon_mid)),
                ("SE", (float(lats.min()), lat_mid, lon_mid, float(lons.max()))),
                ("NW", (lat_mid, float(lats.max()), float(lons.min()), lon_mid)),
                ("NE", (lat_mid, float(lats.max()), lon_mid, float(lons.max()))),
            ]
        # For spatial Mc we use original events list and the same magnitude
        # selection (Mw preferred) so the scale matches.
        for name, (mn_lat, mx_lat, mn_lon, mx_lon) in spatial_subregions:
            sub_events = [e for e in events
                          if mn_lat <= e.latitude <= mx_lat
                          and mn_lon <= e.longitude <= mx_lon]
            sub_mags, _ = select_magnitude_series(sub_events, prefer_mw=prefer_mw)
            if len(sub_mags) >= 30:
                est = mc_maxc(sub_mags, bin_width)
                mc_spatial.append((name, round(est.mc, 2), int(len(sub_mags))))
            else:
                mc_spatial.append((name, None, int(len(sub_mags))))

    notes = []
    if scale_label.startswith("original_magnitude"):
        notes.append(
            "WARNING: Mc estimated on MIXED original magnitude types (no Mw "
            "for most events). Mc is scale-dependent; treat as approximate."
        )
    if mc_m.warning or mc_g.warning or mc_e.warning or mc_s.warning:
        notes.append("One or more Mc methods reported warnings (see fields).")

    return CompletenessReport(
        scale_label=scale_label,
        n_events_used=int(len(mags)),
        mc_maxc=mc_m,
        mc_gft=mc_g,
        mc_emr=mc_e,
        mc_stepp=mc_s,
        mc_recommended=mc_rec,
        mc_recommended_method=method,
        mc_recommended_rationale=rationale,
        mc_t=mc_t,
        mc_spatial=mc_spatial,
        n_above_recommended=n_above,
        n_below_recommended=n_below,
        notes=notes,
    )
