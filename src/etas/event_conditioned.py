"""Event-conditioned backtest — rebuilt for proper mutual exclusivity.

USER CORRECTION (Stage 5 validation):

  The previous implementation produced 0 background origins because the
  90-day exclusion filter was too aggressive. This rebuild ensures:

    A. Post-mainshock forecast origins
    B. Background forecast origins
  are MUTUALLY EXCLUSIVE and both genuinely populated.

For each forecast origin, explicitly records:
  - forecast origin timestamp
  - most recent qualifying mainshock timestamp
  - mainshock magnitude
  - time since mainshock
  - whether the origin is classified post-mainshock
  - whether it is classified background
  - number of qualifying earthquakes in the preceding 1d / 7d / 30d / 90d
  - forecast horizon
  - observed event count
  - observed binary outcome

Mutual exclusivity rule (configurable):
  - An origin is "post_mainshock" if a qualifying mainshock occurred within
    the post-event window (e.g., 0-90d) BEFORE the origin.
  - An origin is "background" if NO qualifying mainshock occurred within the
    lookback window (e.g., 90d) before the origin.
  - The two sets are disjoint by construction.

Configurable mainshock definitions tested separately:
  - M >= 5.0
  - M >= 5.5
  - M >= 6.0

Configurable post-event windows (non-overlapping labels where possible):
  - 0-24h, 1-7d, 8-30d, 31-90d
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from ..baselines.backtest import (
    brier_score,
    log_likelihood_score,
    reliability_diagram,
    roc_auc,
)
from ..baselines.poisson import HORIZON_YEARS
from ..ingestion.schema import CanonicalEvent
from .estimation import fit_etas_mle, prepare_catalog
from .forecast import forecast_temporal
from .model import ETASModel, ETASParams
from .background import KDEBackground, UniformBackground


@dataclass
class ConditionedOrigin:
    """One forecast origin with full conditioning metadata."""

    origin_time: datetime
    horizon: str
    threshold: float
    # Conditioning metadata
    most_recent_mainshock_time: Optional[datetime]
    most_recent_mainshock_mag: Optional[float]
    time_since_mainshock_days: Optional[float]
    is_post_mainshock: bool
    is_background: bool
    post_event_window_label: str        # "0-24h", "1-7d", "8-30d", "31-90d", or "background"
    mainshock_definition: float          # the M threshold used (5.0, 5.5, 6.0)
    # Recent seismicity context
    n_events_preceding_1d: int
    n_events_preceding_7d: int
    n_events_preceding_30d: int
    n_events_preceding_90d: int
    n_mainshocks_preceding_90d: int
    # Forecasts (filled by scorer)
    forecast_probability_etas_mle: float = float("nan")
    forecast_probability_etas_forced: float = float("nan")
    forecast_probability_poisson: float = float("nan")
    # Observation
    n_observed_in_horizon: int = 0
    observed_binary: int = 0
    # Training context
    n_train_events: int = 0


@dataclass
class ConditionedBacktestResult:
    """Aggregated result for one (mainshock_def, threshold, horizon, window_label)."""

    mainshock_definition: float
    threshold: float
    horizon: str
    window_label: str
    n_origins: int
    n_positive: int
    base_rate: float
    mean_etas_mle_prob: float
    mean_etas_forced_prob: float
    mean_poisson_prob: float
    brier_etas_mle: float
    brier_etas_forced: float
    brier_poisson: float
    loglik_etas_mle: float
    loglik_etas_forced: float
    loglik_poisson: float
    ig_etas_mle_vs_poisson: float
    ig_etas_forced_vs_poisson: float
    roc_auc_etas_mle: Optional[float]
    roc_auc_etas_forced: Optional[float]
    roc_auc_poisson: Optional[float]
    calibration_error_etas_mle: float
    calibration_error_etas_forced: float
    calibration_error_poisson: float
    # Sharpness (mean forecast probability, std)
    sharpness_etas_mle: float
    sharpness_etas_forced: float
    sharpness_poisson: float
    origins: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_summary_row(self) -> dict:
        def _r(x):
            return round(x, 4) if not (isinstance(x, float) and math.isnan(x)) else None
        return {
            "mainshock_definition": self.mainshock_definition,
            "threshold": self.threshold,
            "horizon": self.horizon,
            "window_label": self.window_label,
            "n_origins": self.n_origins,
            "n_positive": self.n_positive,
            "base_rate": _r(self.base_rate),
            "mean_etas_mle_prob": _r(self.mean_etas_mle_prob),
            "mean_etas_forced_prob": _r(self.mean_etas_forced_prob),
            "mean_poisson_prob": _r(self.mean_poisson_prob),
            "brier_etas_mle": _r(self.brier_etas_mle),
            "brier_etas_forced": _r(self.brier_etas_forced),
            "brier_poisson": _r(self.brier_poisson),
            "ig_etas_mle_vs_poisson": _r(self.ig_etas_mle_vs_poisson),
            "ig_etas_forced_vs_poisson": _r(self.ig_etas_forced_vs_poisson),
            "roc_auc_etas_mle_sec": _r(self.roc_auc_etas_mle),
            "roc_auc_etas_forced_sec": _r(self.roc_auc_etas_forced),
            "roc_auc_poisson_sec": _r(self.roc_auc_poisson),
            "cal_error_etas_mle": _r(self.calibration_error_etas_mle),
            "cal_error_etas_forced": _r(self.calibration_error_etas_forced),
            "cal_error_poisson": _r(self.calibration_error_poisson),
            "sharpness_etas_mle": _r(self.sharpness_etas_mle),
            "sharpness_etas_forced": _r(self.sharpness_etas_forced),
            "sharpness_poisson": _r(self.sharpness_poisson),
            "notes": "; ".join(self.notes),
        }


# ---------------------------------------------------------------------------
# Origin construction
# ---------------------------------------------------------------------------


def build_conditioned_origins(
    events: list[CanonicalEvent],
    horizon: str,
    threshold: float,
    mainshock_definition: float,
    catalog_start: Optional[datetime] = None,
    catalog_end: Optional[datetime] = None,
    background_origin_step_days: int = 30,
) -> list[ConditionedOrigin]:
    """Build mutually-exclusive post-mainshock and background origins.

    Mutual exclusivity:
      - post_mainshock: a qualifying mainshock (M >= mainshock_definition)
        occurred within the post-event lookback (90d) before the origin.
      - background: NO qualifying mainshock within 90d before the origin.

    Post-mainshock origins are placed at fixed lags (1d, 7d, 30d, 90d) after
    each mainshock. Each origin is assigned to exactly ONE post-event window
    label (0-24h, 1-7d, 8-30d, 31-90d) based on its lag.

    Background origins are placed every `background_origin_step_days` days
    throughout the catalog, EXCLUDING any origin that falls within 90d after
    a mainshock.
    """
    hy = HORIZON_YEARS[horizon]
    horizon_td = timedelta(days=hy * 365.25)
    if catalog_start is None:
        catalog_start = min(e.origin_time_utc for e in events)
    if catalog_end is None:
        catalog_end = max(e.origin_time_utc for e in events)

    # Identify mainshocks
    mainshocks = sorted(
        [e for e in events
         if (e.mw if e.mw is not None else e.original_magnitude) >= mainshock_definition],
        key=lambda e: e.origin_time_utc,
    )

    def _preceding_counts(t_origin: datetime) -> tuple[int, int, int, int, int]:
        """Count events in the 1d/7d/30d/90d preceding the origin, and mainshocks in 90d."""
        n1 = n7 = n30 = n90 = nm90 = 0
        for e in events:
            if e.origin_time_utc >= t_origin:
                break
            dt = (t_origin - e.origin_time_utc).total_seconds() / 86400.0
            if dt <= 1.0:
                n1 += 1
            if dt <= 7.0:
                n7 += 1
            if dt <= 30.0:
                n30 += 1
            if dt <= 90.0:
                n90 += 1
                if (e.mw if e.mw is not None else e.original_magnitude) >= mainshock_definition:
                    nm90 += 1
        return n1, n7, n30, n90, nm90

    def _most_recent_mainshock(t_origin: datetime) -> tuple[Optional[datetime], Optional[float], float]:
        """Return (time, mag, days_since) of most recent mainshock before origin, or (None, None, inf)."""
        recent = None
        for ms in reversed(mainshocks):
            if ms.origin_time_utc < t_origin:
                recent = ms
                break
        if recent is None:
            return None, None, float("inf")
        dt = (t_origin - recent.origin_time_utc).total_seconds() / 86400.0
        return recent.origin_time_utc, (recent.mw if recent.mw is not None else recent.original_magnitude), dt

    origins: list[ConditionedOrigin] = []

    # --- Post-mainshock origins: at lags 1d, 7d, 30d, 90d after each mainshock ---
    # Each lag maps to a non-overlapping window label.
    lag_to_label = [(1.0, "0-24h"), (7.0, "1-7d"), (30.0, "8-30d"), (90.0, "31-90d")]
    for ms in mainshocks:
        for lag_days, label in lag_to_label:
            t0 = ms.origin_time_utc + timedelta(days=lag_days)
            if t0 < catalog_start or t0 + horizon_td > catalog_end:
                continue
            n1, n7, n30, n90, nm90 = _preceding_counts(t0)
            origins.append(ConditionedOrigin(
                origin_time=t0, horizon=horizon, threshold=threshold,
                most_recent_mainshock_time=ms.origin_time_utc,
                most_recent_mainshock_mag=(ms.mw if ms.mw is not None else ms.original_magnitude),
                time_since_mainshock_days=lag_days,
                is_post_mainshock=True, is_background=False,
                post_event_window_label=label,
                mainshock_definition=mainshock_definition,
                n_events_preceding_1d=n1, n_events_preceding_7d=n7,
                n_events_preceding_30d=n30, n_events_preceding_90d=n90,
                n_mainshocks_preceding_90d=nm90,
            ))

    # --- Background origins: every background_origin_step_days, excluding 90d after any mainshock ---
    ms_times = [ms.origin_time_utc for ms in mainshocks]
    t = catalog_start + timedelta(days=365)   # leave 1 year for initial training
    step = timedelta(days=background_origin_step_days)
    while t + horizon_td <= catalog_end:
        # Check no mainshock in the 90d before t
        is_post = any(0 <= (t - mst).total_seconds() / 86400.0 <= 90.0 for mst in ms_times)
        if not is_post:
            n1, n7, n30, n90, nm90 = _preceding_counts(t)
            mst, msmag, dt_ms = _most_recent_mainshock(t)
            origins.append(ConditionedOrigin(
                origin_time=t, horizon=horizon, threshold=threshold,
                most_recent_mainshock_time=mst,
                most_recent_mainshock_mag=msmag,
                time_since_mainshock_days=dt_ms if math.isfinite(dt_ms) else None,
                is_post_mainshock=False, is_background=True,
                post_event_window_label="background",
                mainshock_definition=mainshock_definition,
                n_events_preceding_1d=n1, n_events_preceding_7d=n7,
                n_events_preceding_30d=n30, n_events_preceding_90d=n90,
                n_mainshocks_preceding_90d=nm90,
            ))
        t += step

    return origins


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_origins(
    events: list[CanonicalEvent],
    origins: list[ConditionedOrigin],
    threshold: float,
    horizon: str,
    Mc: float,
    forced_params: ETASParams,
    catalog_start: datetime,
) -> None:
    """Score each origin with three models: Poisson, locally-fitted ETAS (MLE),
    and externally-informed ETAS (forced params). Mutates origins in place.

    CRITICAL: The Poisson rate is computed PER ORIGIN using an expanding
    window (all events before that origin's time). This is the correct
    chronological approach. Using a single fixed rate from the shortest
    training window would unfairly penalize the Poisson baseline.
    """
    hy = HORIZON_YEARS[horizon]
    horizon_td = timedelta(days=hy * 365.25)

    # Fit the locally-fitted ETAS ONCE on the pre-test period (training window
    # ends before the first origin). This is the "locally fitted ETAS" model;
    # it will be K≈0 per Stage 5. (ETAS refitting per origin is too expensive;
    # the single fit is chronological because it ends before the first origin.)
    if origins:
        fit_end = min(o.origin_time for o in origins) - timedelta(days=1)
        train_events = [e for e in events if e.origin_time_utc < fit_end]
        local_fit = fit_etas_mle(train_events, Mc=Mc, background_kind="kde",
                                  spatial_kernel="powerlaw", t_end=fit_end)

    for o in origins:
        # PER-ORIGIN expanding-window training set
        train_for_forecast = [e for e in events if e.origin_time_utc < o.origin_time]
        cat = prepare_catalog(train_for_forecast, Mc=Mc, t_end=o.origin_time)
        if cat["n"] == 0:
            continue
        # PER-ORIGIN Poisson rate (expanding window)
        train_above = [e for e in train_for_forecast
                       if (e.mw if e.mw is not None else e.original_magnitude) >= threshold]
        train_span_years = (o.origin_time - catalog_start).total_seconds() / (365.25 * 86400)
        poisson_rate = len(train_above) / max(train_span_years, 1e-6)

        # Locally-fitted ETAS (K≈0) — uses the single pre-test fit
        mle_model = ETASModel(params=local_fit.params, background=local_fit.background,
                              bbox=(20.0, 28.0, 88.0, 96.0),
                              fit_info={"b_value": _b_from_catalog(train_for_forecast, Mc)})
        _, p_mle = forecast_temporal(
            mle_model, cat["times_days"], cat["lats"], cat["lons"], cat["mags"],
            forecast_start_days=cat["t_end_days"],
            horizon_days=hy * 365.25, threshold=threshold,
        )
        # Externally-informed ETAS (forced params) — rebuild background with
        # the training data so μ is local, but triggering params are external.
        forced_bg = KDEBackground.build(
            cat["lats"], cat["lons"],
            mu_total_per_year=max(forced_params.mu_total_per_year, 0.1),
            bbox=(20.0, 28.0, 88.0, 96.0),
        ) if len(cat["lats"]) > 5 else UniformBackground.build(
            forced_params.mu_total_per_year, (20.0, 28.0, 88.0, 96.0))
        forced_model = ETASModel(params=forced_params, background=forced_bg,
                                  bbox=(20.0, 28.0, 88.0, 96.0),
                                  fit_info={"b_value": _b_from_catalog(train_for_forecast, Mc),
                                            "externally_informed": True})
        _, p_forced = forecast_temporal(
            forced_model, cat["times_days"], cat["lats"], cat["lons"], cat["mags"],
            forecast_start_days=cat["t_end_days"],
            horizon_days=hy * 365.25, threshold=threshold,
        )
        # Poisson — per-origin rate
        p_pois = 1.0 - math.exp(-poisson_rate * hy)
        # Observation
        obs_events = [e for e in events
                      if o.origin_time <= e.origin_time_utc < o.origin_time + horizon_td
                      and (e.mw if e.mw is not None else e.original_magnitude) >= threshold]
        o.forecast_probability_etas_mle = p_mle
        o.forecast_probability_etas_forced = p_forced
        o.forecast_probability_poisson = p_pois
        o.n_observed_in_horizon = len(obs_events)
        o.observed_binary = 1 if obs_events else 0
        o.n_train_events = cat["n"]


def aggregate_results(
    origins: list[ConditionedOrigin],
    mainshock_definition: float,
    threshold: float,
    horizon: str,
) -> list[ConditionedBacktestResult]:
    """Aggregate scored origins by post_event_window_label."""
    results = []
    labels = ["0-24h", "1-7d", "8-30d", "31-90d", "background"]
    for label in labels:
        sub = [o for o in origins if o.post_event_window_label == label]
        if not sub:
            continue
        mle_arr = np.array([o.forecast_probability_etas_mle for o in sub])
        forced_arr = np.array([o.forecast_probability_etas_forced for o in sub])
        pois_arr = np.array([o.forecast_probability_poisson for o in sub])
        obs_arr = np.array([o.observed_binary for o in sub], dtype=float)
        n_pos = int(obs_arr.sum())
        base_rate = n_pos / len(sub) if sub else 0.0

        b_mle = brier_score(mle_arr, obs_arr)
        b_forced = brier_score(forced_arr, obs_arr)
        b_pois = brier_score(pois_arr, obs_arr)
        ll_mle = log_likelihood_score(mle_arr, obs_arr)
        ll_forced = log_likelihood_score(forced_arr, obs_arr)
        ll_pois = log_likelihood_score(pois_arr, obs_arr)

        eps = 1e-12
        def _ig(forecast, ref):
            return float(np.mean(
                obs_arr * np.log(np.clip(forecast, eps, 1 - eps) / np.clip(ref, eps, 1 - eps)) +
                (1 - obs_arr) * np.log(np.clip(1 - forecast, eps, 1 - eps) / np.clip(1 - ref, eps, 1 - eps))
            ))
        ig_mle = _ig(mle_arr, pois_arr)
        ig_forced = _ig(forced_arr, pois_arr)

        auc_mle = roc_auc(mle_arr, obs_arr)
        auc_forced = roc_auc(forced_arr, obs_arr)
        auc_pois = roc_auc(pois_arr, obs_arr)

        rel_mle = reliability_diagram(mle_arr, obs_arr, n_bins=5)
        rel_forced = reliability_diagram(forced_arr, obs_arr, n_bins=5)
        rel_pois = reliability_diagram(pois_arr, obs_arr, n_bins=5)
        def _cal(rel):
            vals = [abs(b[3] - b[4]) for b in rel if b[2] > 0 and not math.isnan(b[3])]
            return float(np.mean(vals)) if vals else float("nan")
        cal_mle = _cal(rel_mle)
        cal_forced = _cal(rel_forced)
        cal_pois = _cal(rel_pois)

        notes = []
        if n_pos < 5:
            notes.append(f"Only {n_pos} positive observations; high variance.")
        if b_forced < b_pois:
            notes.append(f"Forced-ETAS BEATS Poisson (Brier {b_forced:.4f} < {b_pois:.4f}).")
        else:
            notes.append(f"Forced-ETAS does NOT beat Poisson (Brier {b_forced:.4f} >= {b_pois:.4f}).")
        if b_mle < b_pois - 1e-6:
            notes.append(f"Locally-fitted ETAS BEATS Poisson (Brier {b_mle:.4f} < {b_pois:.4f}).")
        else:
            notes.append(f"Locally-fitted ETAS does NOT beat Poisson (Brier {b_mle:.4f} >= {b_pois:.4f}).")
        notes.append("ROC-AUC is a SECONDARY diagnostic only.")

        results.append(ConditionedBacktestResult(
            mainshock_definition=mainshock_definition,
            threshold=threshold, horizon=horizon, window_label=label,
            n_origins=len(sub), n_positive=n_pos, base_rate=base_rate,
            mean_etas_mle_prob=float(mle_arr.mean()),
            mean_etas_forced_prob=float(forced_arr.mean()),
            mean_poisson_prob=float(pois_arr.mean()),
            brier_etas_mle=b_mle, brier_etas_forced=b_forced, brier_poisson=b_pois,
            loglik_etas_mle=ll_mle, loglik_etas_forced=ll_forced, loglik_poisson=ll_pois,
            ig_etas_mle_vs_poisson=ig_mle, ig_etas_forced_vs_poisson=ig_forced,
            roc_auc_etas_mle=auc_mle, roc_auc_etas_forced=auc_forced, roc_auc_poisson=auc_pois,
            calibration_error_etas_mle=cal_mle,
            calibration_error_etas_forced=cal_forced,
            calibration_error_poisson=cal_pois,
            sharpness_etas_mle=float(np.std(mle_arr)),
            sharpness_etas_forced=float(np.std(forced_arr)),
            sharpness_poisson=float(np.std(pois_arr)),
            origins=sub, notes=notes,
        ))
    return results


def _b_from_catalog(events, Mc):
    mags = np.array([e.mw if e.mw is not None else e.original_magnitude for e in events])
    mags = mags[mags >= Mc - 0.05]
    if len(mags) < 20:
        return 1.0
    mean_m = float(np.mean(mags))
    denom = mean_m - (Mc - 0.05)
    if denom <= 0:
        return 1.0
    return math.log10(math.e) / denom


def run_full_conditioned_backtest(
    events: list[CanonicalEvent],
    thresholds: list[float],
    horizons: list[str],
    mainshock_definitions: list[float],
    Mc: float = 4.5,
    forced_params: Optional[ETASParams] = None,
    catalog_start: Optional[datetime] = None,
    max_origins_per_window: int = 200,
) -> list[ConditionedBacktestResult]:
    """Run the full event-conditioned backtest matrix.

    For each (mainshock_definition, threshold, horizon), builds origins,
    scores them with all three models, and aggregates by post-event window.

    ``max_origins_per_window`` caps the number of origins scored per
    (window_label) to keep runtime manageable. Sampling is deterministic
    (seed=42). The cap is a runtime compromise, NOT a scientific filter —
    all origins are still constructed; only the scoring is subsampled.
    """
    if forced_params is None:
        forced_params = ETASParams(
            mu_total_per_year=10.0, K=0.02, alpha=0.8, c_days=0.05, p=1.1,
            sigma_km=10.0, gamma=0.5, q=1.0, Mc=Mc,
            spatial_kernel="powerlaw",
            fixed_parameters={"K": 0.02, "alpha": 0.8, "c_days": 0.05,
                              "p": 1.1, "sigma_km": 10.0, "gamma": 0.5, "q": 1.0},
        )
    if catalog_start is None:
        catalog_start = min(e.origin_time_utc for e in events)

    rng = np.random.default_rng(42)
    all_results = []
    for ms_def in mainshock_definitions:
        for th in thresholds:
            for h in horizons:
                origins = build_conditioned_origins(
                    events, horizon=h, threshold=th,
                    mainshock_definition=ms_def,
                    catalog_start=catalog_start,
                )
                # Subsample per window_label if exceeds cap
                from collections import defaultdict
                by_label = defaultdict(list)
                for o in origins:
                    by_label[o.post_event_window_label].append(o)
                sampled = []
                for label, lst in by_label.items():
                    if len(lst) > max_origins_per_window:
                        idx = rng.choice(len(lst), max_origins_per_window, replace=False)
                        sampled.extend([lst[i] for i in idx])
                    else:
                        sampled.extend(lst)
                score_origins(events, sampled, threshold=th, horizon=h, Mc=Mc,
                              forced_params=forced_params, catalog_start=catalog_start)
                results = aggregate_results(sampled, ms_def, th, h)
                all_results.extend(results)
    return all_results
