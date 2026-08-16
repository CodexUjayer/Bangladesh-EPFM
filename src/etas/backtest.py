"""Chronological ETAS backtest with event-conditioned analysis.

CRITICAL: No data leakage. At each forecast origin:
  1. Use ONLY events before the origin.
  2. Fit ETAS on the training window (or use a pre-fit model retrained up to
     the origin).
  3. Generate the forecast for the horizon.
  4. Freeze the forecast.
  5. Observe future earthquakes.
  6. Score.

Event-conditioned analysis (the scientific core of Stage 5):
  A. Post-mainshock windows: forecast origins placed 1 day, 7 days, 30 days
     after each M>=5.0 (and M>=6.0) mainshock. This tests whether ETAS
     actually captures aftershock triggering.
  B. Background windows: forecast origins in periods WITHOUT a recent
     major event. This tests whether ETAS adds skill in quiet periods
     (where it should reduce to ~Poisson).

The hypothesis: ETAS gains skill mainly in (A), and should be approximately
tied with Poisson in (B). If ETAS does NOT beat Poisson in (A), it provides
no value over the simpler baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from ..baselines.backtest import (
    BacktestResult,
    brier_score,
    log_likelihood_score,
    reliability_diagram,
    information_gain,
    roc_auc,
)
from ..baselines.poisson import HORIZON_YEARS
from ..ingestion.schema import CanonicalEvent
from .background import KDEBackground, UniformBackground
from .estimation import fit_etas_mle, prepare_catalog
from .forecast import forecast_temporal


@dataclass
class ETASBacktestOrigin:
    """One ETAS forecast origin."""

    origin_time: datetime
    horizon: str
    threshold: float
    window_type: str              # "post_mainshock" or "background"
    mainshock_mag: Optional[float] = None
    mainshock_time: Optional[datetime] = None
    days_since_mainshock: Optional[float] = None
    n_train_events: int = 0
    forecast_probability: float = 0.0
    poisson_probability: float = 0.0
    observed_binary: int = 0
    n_observed_in_horizon: int = 0


@dataclass
class ETASBacktestResult:
    """Aggregated backtest for one (threshold, horizon, window_type)."""

    model: str
    threshold: float
    horizon: str
    window_type: str
    n_origins: int
    n_positive: int
    base_rate: float
    mean_etas_prob: float
    mean_poisson_prob: float
    mean_forced_prob: float = 0.0
    brier_etas: float = float("nan")
    brier_poisson: float = float("nan")
    brier_forced: float = float("nan")
    loglik_etas: float = float("nan")
    loglik_poisson: float = float("nan")
    loglik_forced: float = float("nan")
    information_gain_etas_vs_poisson: float = float("nan")
    information_gain_forced_vs_poisson: float = float("nan")
    roc_auc_etas: Optional[float] = None
    roc_auc_poisson: Optional[float] = None
    roc_auc_forced: Optional[float] = None
    calibration_error_etas: float = float("nan")
    calibration_error_poisson: float = float("nan")
    calibration_error_forced: float = float("nan")
    origins: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_summary_row(self) -> dict:
        return {
            "model": "etas",
            "threshold": self.threshold,
            "horizon": self.horizon,
            "window_type": self.window_type,
            "n_origins": self.n_origins,
            "n_positive": self.n_positive,
            "base_rate": round(self.base_rate, 4),
            "mean_etas_prob": round(self.mean_etas_prob, 4),
            "mean_poisson_prob": round(self.mean_poisson_prob, 4),
            "mean_forced_prob": round(self.mean_forced_prob, 4),
            "brier_etas": round(self.brier_etas, 4) if not math.isnan(self.brier_etas) else None,
            "brier_poisson": round(self.brier_poisson, 4),
            "brier_forced": round(self.brier_forced, 4) if not math.isnan(self.brier_forced) else None,
            "brier_improvement_mle": round(self.brier_poisson - self.brier_etas, 4) if not math.isnan(self.brier_etas) else None,
            "brier_improvement_forced": round(self.brier_poisson - self.brier_forced, 4) if not math.isnan(self.brier_forced) else None,
            "loglik_etas": round(self.loglik_etas, 4) if not math.isnan(self.loglik_etas) else None,
            "loglik_poisson": round(self.loglik_poisson, 4),
            "loglik_forced": round(self.loglik_forced, 4) if not math.isnan(self.loglik_forced) else None,
            "information_gain_etas_vs_poisson": round(self.information_gain_etas_vs_poisson, 4) if not math.isnan(self.information_gain_etas_vs_poisson) else None,
            "information_gain_forced_vs_poisson": round(self.information_gain_forced_vs_poisson, 4) if not math.isnan(self.information_gain_forced_vs_poisson) else None,
            "roc_auc_etas_secondary": round(self.roc_auc_etas, 4) if self.roc_auc_etas is not None else None,
            "roc_auc_poisson_secondary": round(self.roc_auc_poisson, 4) if self.roc_auc_poisson is not None else None,
            "roc_auc_forced_secondary": round(self.roc_auc_forced, 4) if self.roc_auc_forced is not None else None,
            "calibration_error_etas": round(self.calibration_error_etas, 4) if not math.isnan(self.calibration_error_etas) else None,
            "calibration_error_poisson": round(self.calibration_error_poisson, 4),
            "calibration_error_forced": round(self.calibration_error_forced, 4) if not math.isnan(self.calibration_error_forced) else None,
            "notes": "; ".join(self.notes),
        }


def run_etas_backtest(
    events: list[CanonicalEvent],
    threshold: float,
    horizon: str,
    Mc: float = 4.5,
    mainshock_threshold: float = 5.0,
    catalog_start: Optional[datetime] = None,
    retrain_each_origin: bool = False,
) -> list[ETASBacktestResult]:
    """Run the chronological ETAS backtest.

    Produces separate results for:
      - post_mainshock windows (origins 1d, 7d, 30d after each M>=mainshock_threshold event)
      - background windows (yearly origins in periods without a recent major event)

    Parameters
    ----------
    threshold : forecast magnitude threshold (e.g. 5.0)
    horizon : '24h', '7d', '30d', '90d'
    Mc : ETAS fitting threshold (default 4.5, conservative)
    mainshock_threshold : threshold defining a 'mainshock' for post-mainshock windows
    retrain_each_origin : if True, refit ETAS at each origin (slow but fully
        prospective). If False, fit once on the full pre-test period and
        reuse (faster; still chronological if the fit period ends before the
        first test origin).
    """
    hy = HORIZON_YEARS[horizon]
    horizon_td = timedelta(days=hy * 365.25)
    if catalog_start is None:
        catalog_start = min(e.origin_time_utc for e in events)
    catalog_end = max(e.origin_time_utc for e in events)

    # Identify mainshocks (events above mainshock_threshold)
    mainshocks = sorted(
        [e for e in events
         if (e.mw if e.mw is not None else e.original_magnitude) >= mainshock_threshold],
        key=lambda e: e.origin_time_utc,
    )

    # --- Build post-mainshock origins ---
    post_origins = []
    for ms in mainshocks:
        for lag_days in [1, 7, 30]:
            t0 = ms.origin_time_utc + timedelta(days=lag_days)
            if t0 < catalog_start or t0 + horizon_td > catalog_end:
                continue
            post_origins.append(ETASBacktestOrigin(
                origin_time=t0, horizon=horizon, threshold=threshold,
                window_type="post_mainshock",
                mainshock_mag=(ms.mw if ms.mw is not None else ms.original_magnitude),
                mainshock_time=ms.origin_time_utc,
                days_since_mainshock=lag_days,
            ))

    # --- Build background origins (yearly, exclude those within 90d of a mainshock) ---
    bg_origins = []
    year_start = catalog_start.year + 2  # leave 2 years for initial ETAS fit
    for year in range(year_start, catalog_end.year):
        t0 = datetime(year, 1, 1, tzinfo=timezone.utc)
        if t0 + horizon_td > catalog_end:
            continue
        # Exclude if within 90 days after a mainshock
        near = any(abs((t0 - ms.origin_time_utc).total_seconds()) / 86400.0 < 90
                   for ms in mainshocks)
        if near:
            continue
        bg_origins.append(ETASBacktestOrigin(
            origin_time=t0, horizon=horizon, threshold=threshold,
            window_type="background",
        ))

    # --- Fit ETAS once on pre-test data (if not retraining) ---
    # Use events up to the first test origin minus 1 day.
    all_origins = post_origins + bg_origins
    if not all_origins:
        return [_empty_result(threshold, horizon, "no origins")]
    first_origin = min(o.origin_time for o in all_origins)

    results_by_window = {}
    for window_type, origins in [("post_mainshock", post_origins),
                                  ("background", bg_origins)]:
        if not origins:
            continue
        # Fit ETAS up to (first origin in this group - 1 day)
        fit_end = min(o.origin_time for o in origins) - timedelta(days=1)
        train_events = [e for e in events if e.origin_time_utc < fit_end]
        etas_fit = fit_etas_mle(train_events, Mc=Mc, background_kind="kde",
                                 spatial_kernel="powerlaw",
                                 t_end=fit_end)
        # ALSO build a "forced-triggering" ETAS variant with literature-informed
        # parameters (clearly labeled externally_informed). This tests whether
        # triggering helps prospectively even when the in-sample MLE prefers K=0.
        from .model import ETASModel, ETASParams
        from .background import KDEBackground
        import numpy as _np
        train_cat = prepare_catalog(train_events, Mc=Mc, t_end=fit_end)
        forced_params = ETASParams(
            mu_total_per_year=max(len(train_events) / max((fit_end - catalog_start).total_seconds()/(365.25*86400), 1e-6) * 0.5, 0.1),
            K=0.02, alpha=0.8, c_days=0.05, p=1.1,
            sigma_km=10.0, gamma=0.5, q=1.0, Mc=Mc,
            spatial_kernel="powerlaw",
            fixed_parameters={"K": 0.02, "alpha": 0.8, "c_days": 0.05,
                              "p": 1.1, "sigma_km": 10.0, "gamma": 0.5, "q": 1.0},
        )
        forced_bg = KDEBackground.build(
            train_cat["lats"], train_cat["lons"],
            mu_total_per_year=forced_params.mu_total_per_year,
            bbox=(20.0, 28.0, 88.0, 96.0),
        ) if len(train_cat["lats"]) > 5 else UniformBackground.build(
            forced_params.mu_total_per_year, (20.0, 28.0, 88.0, 96.0))
        forced_model = ETASModel(params=forced_params, background=forced_bg,
                                  bbox=(20.0, 28.0, 88.0, 96.0),
                                  fit_info={"b_value": _b_from_catalog(train_events, Mc),
                                            "forced_triggering": True})
        # For Poisson baseline: rate from training events above threshold
        train_above = [e for e in train_events
                       if (e.mw if e.mw is not None else e.original_magnitude) >= threshold]
        train_span_years = (fit_end - catalog_start).total_seconds() / (365.25 * 86400)
        poisson_rate = len(train_above) / max(train_span_years, 1e-6)

        # Score each origin with THREE models: Poisson, MLE-ETAS, forced-ETAS
        etas_probs = []       # MLE ETAS
        forced_probs = []     # forced-triggering ETAS
        pois_probs = []
        obs = []
        for o in origins:
            train_for_forecast = [e for e in events if e.origin_time_utc < o.origin_time]
            cat = prepare_catalog(train_for_forecast, Mc=Mc, t_end=o.origin_time)
            if cat["n"] == 0:
                continue
            # MLE ETAS forecast
            mle_model = ETASModel(params=etas_fit.params, background=etas_fit.background,
                                  bbox=(20.0, 28.0, 88.0, 96.0),
                                  fit_info={"b_value": _b_from_catalog(train_for_forecast, Mc)})
            _, p_etas = forecast_temporal(
                mle_model, cat["times_days"], cat["lats"], cat["lons"], cat["mags"],
                forecast_start_days=cat["t_end_days"],
                horizon_days=hy * 365.25,
                threshold=threshold,
            )
            # Forced-triggering ETAS forecast
            forced_model.fit_info["b_value"] = _b_from_catalog(train_for_forecast, Mc)
            _, p_forced = forecast_temporal(
                forced_model, cat["times_days"], cat["lats"], cat["lons"], cat["mags"],
                forecast_start_days=cat["t_end_days"],
                horizon_days=hy * 365.25,
                threshold=threshold,
            )
            # Poisson forecast
            p_pois = 1.0 - math.exp(-poisson_rate * hy)
            # Observation
            obs_events = [e for e in events
                          if o.origin_time <= e.origin_time_utc < o.origin_time + horizon_td
                          and (e.mw if e.mw is not None else e.original_magnitude) >= threshold]
            ob = 1 if obs_events else 0
            o.n_train_events = cat["n"]
            o.forecast_probability = p_etas
            o.poisson_probability = p_pois
            o.observed_binary = ob
            o.n_observed_in_horizon = len(obs_events)
            etas_probs.append(p_etas)
            forced_probs.append(p_forced)
            pois_probs.append(p_pois)
            obs.append(ob)

        if not etas_probs:
            continue
        etas_arr = np.array(etas_probs)
        pois_arr = np.array(pois_probs)
        forced_arr = np.array(forced_probs)
        obs_arr = np.array(obs, dtype=float)
        n_pos = int(obs_arr.sum())
        base_rate = n_pos / len(obs_arr) if obs_arr.size else 0.0
        b_etas = brier_score(etas_arr, obs_arr)
        b_pois = brier_score(pois_arr, obs_arr)
        b_forced = brier_score(forced_arr, obs_arr)
        ll_etas = log_likelihood_score(etas_arr, obs_arr)
        ll_pois = log_likelihood_score(pois_arr, obs_arr)
        ll_forced = log_likelihood_score(forced_arr, obs_arr)
        # IG: ETAS vs Poisson (reference = Poisson forecast, not climatology)
        eps = 1e-12
        ig_arr = obs_arr * np.log(np.clip(etas_arr, eps, 1 - eps) / np.clip(pois_arr, eps, 1 - eps)) + \
                 (1 - obs_arr) * np.log(np.clip(1 - etas_arr, eps, 1 - eps) / np.clip(1 - pois_arr, eps, 1 - eps))
        ig = float(np.mean(ig_arr))
        igf_arr = obs_arr * np.log(np.clip(forced_arr, eps, 1 - eps) / np.clip(pois_arr, eps, 1 - eps)) + \
                  (1 - obs_arr) * np.log(np.clip(1 - forced_arr, eps, 1 - eps) / np.clip(1 - pois_arr, eps, 1 - eps))
        ig_forced = float(np.mean(igf_arr))
        auc_etas = roc_auc(etas_arr, obs_arr)
        auc_pois = roc_auc(pois_arr, obs_arr)
        auc_forced = roc_auc(forced_arr, obs_arr)
        rel_etas = reliability_diagram(etas_arr, obs_arr, n_bins=5)
        rel_pois = reliability_diagram(pois_arr, obs_arr, n_bins=5)
        rel_forced = reliability_diagram(forced_arr, obs_arr, n_bins=5)
        cal_etas = float(np.mean([abs(b[3] - b[4]) for b in rel_etas if b[2] > 0 and not math.isnan(b[3])])) if any(b[2] > 0 for b in rel_etas) else float("nan")
        cal_pois = float(np.mean([abs(b[3] - b[4]) for b in rel_pois if b[2] > 0 and not math.isnan(b[3])])) if any(b[2] > 0 for b in rel_pois) else float("nan")
        cal_forced = float(np.mean([abs(b[3] - b[4]) for b in rel_forced if b[2] > 0 and not math.isnan(b[3])])) if any(b[2] > 0 for b in rel_forced) else float("nan")

        notes = []
        if n_pos < 5:
            notes.append(f"Only {n_pos} positive observations; high variance.")
        # MLE ETAS comparison
        if not math.isnan(b_etas):
            if b_etas < b_pois:
                notes.append(f"MLE-ETAS BEATS Poisson (Brier {b_etas:.4f} < {b_pois:.4f}).")
            else:
                notes.append(f"MLE-ETAS does NOT beat Poisson (Brier {b_etas:.4f} >= {b_pois:.4f}).")
        # Forced-triggering ETAS comparison
        if b_forced < b_pois:
            notes.append(f"FORCED-ETAS BEATS Poisson (Brier {b_forced:.4f} < {b_pois:.4f}).")
        else:
            notes.append(f"FORCED-ETAS does NOT beat Poisson (Brier {b_forced:.4f} >= {b_pois:.4f}).")
        if ig_forced > 0:
            notes.append(f"Forced-ETAS positive information gain (+{ig_forced:.4f}).")
        else:
            notes.append(f"Forced-ETAS non-positive information gain ({ig_forced:.4f}); no skill over Poisson.")
        notes.append("ROC-AUC is a SECONDARY diagnostic only.")

        results_by_window[window_type] = ETASBacktestResult(
            model="etas", threshold=threshold, horizon=horizon,
            window_type=window_type, n_origins=len(origins), n_positive=n_pos,
            base_rate=base_rate,
            mean_etas_prob=float(etas_arr.mean()),
            mean_poisson_prob=float(pois_arr.mean()),
            mean_forced_prob=float(forced_arr.mean()),
            brier_etas=b_etas, brier_poisson=b_pois, brier_forced=b_forced,
            loglik_etas=ll_etas, loglik_poisson=ll_pois, loglik_forced=ll_forced,
            information_gain_etas_vs_poisson=ig,
            information_gain_forced_vs_poisson=ig_forced,
            roc_auc_etas=auc_etas, roc_auc_poisson=auc_pois, roc_auc_forced=auc_forced,
            calibration_error_etas=cal_etas, calibration_error_poisson=cal_pois,
            calibration_error_forced=cal_forced,
            origins=origins, notes=notes,
        )
    return list(results_by_window.values())


def event_conditioned_backtest(
    events: list[CanonicalEvent],
    thresholds: list[float],
    horizons: list[str],
    Mc: float = 4.5,
    mainshock_threshold: float = 5.0,
) -> list[ETASBacktestResult]:
    """Run the full event-conditioned backtest matrix.

    For each (threshold, horizon), runs both post-mainshock and background
    windows. Returns a list of ETASBacktestResult, one per (threshold,
    horizon, window_type).
    """
    all_results = []
    for th in thresholds:
        for h in horizons:
            results = run_etas_backtest(events, threshold=th, horizon=h, Mc=Mc,
                                        mainshock_threshold=mainshock_threshold)
            all_results.extend(results)
    return all_results


def _b_from_catalog(events, Mc):
    """Quick b-value from a catalog for forecast scaling."""
    mags = np.array([e.mw if e.mw is not None else e.original_magnitude for e in events])
    mags = mags[mags >= Mc - 0.05]
    if len(mags) < 20:
        return 1.0
    mean_m = float(np.mean(mags))
    denom = mean_m - (Mc - 0.05)
    if denom <= 0:
        return 1.0
    return math.log10(math.e) / denom


def _empty_result(threshold, horizon, note):
    return ETASBacktestResult(
        model="etas", threshold=threshold, horizon=horizon,
        window_type="none", n_origins=0, n_positive=0, base_rate=float("nan"),
        mean_etas_prob=float("nan"), mean_poisson_prob=float("nan"),
        brier_etas=float("nan"), brier_poisson=float("nan"),
        loglik_etas=float("nan"), loglik_poisson=float("nan"),
        information_gain_etas_vs_poisson=float("nan"),
        roc_auc_etas=None, roc_auc_poisson=None,
        calibration_error_etas=float("nan"), calibration_error_poisson=float("nan"),
        notes=[note],
    )
