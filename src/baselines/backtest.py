"""Chronological backtesting framework for Stage 4 baselines.

NEVER randomly shuffle earthquake events. Earthquake forecasting is a
temporal prediction problem. This module implements:

  - expanding-window chronological backtesting
  - per-origin: train on data before origin, forecast next horizon,
    compare to actual observations
  - evaluation: Brier score, log-likelihood, reliability, information gain
  - ROC-AUC reported ONLY as a secondary diagnostic (rare-event ROC-AUC is
    misleading and not the primary measure)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from ..ingestion.schema import CanonicalEvent
from .poisson import HORIZON_YEARS, estimate_temporal_poisson


@dataclass
class BacktestOrigin:
    """One forecast origin in the chronological backtest."""

    origin_time: datetime
    horizon: str
    horizon_years: float
    threshold: float
    n_train_events: int
    train_exposure_years: float
    forecast_rate_per_year: float
    forecast_probability: float
    forecast_ci: tuple[float, float]
    n_observed_in_horizon: int
    observed_binary: int               # 1 if >=1 event occurred
    model: str = "temporal_poisson"


@dataclass
class BacktestResult:
    """Aggregated backtest metrics for one (model, threshold, horizon)."""

    model: str
    threshold: float
    horizon: str
    n_origins: int
    n_positive: int                    # origins where >=1 event occurred
    base_rate: float                   # n_positive / n_origins
    mean_forecast_probability: float
    brier: float
    log_likelihood: float
    information_gain_vs_climatology: float
    roc_auc: Optional[float]           # secondary diagnostic only
    reliability_bins: list             # list of (bin_lo, bin_hi, n, mean_p, obs_freq)
    calibration_error: float           # mean abs(mean_p - obs_freq) over bins
    origins: list[BacktestOrigin] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_summary_row(self) -> dict:
        return {
            "model": self.model,
            "threshold": self.threshold,
            "horizon": self.horizon,
            "n_origins": self.n_origins,
            "n_positive": self.n_positive,
            "base_rate": round(self.base_rate, 4),
            "mean_forecast_probability": round(self.mean_forecast_probability, 4),
            "brier_score": round(self.brier, 4),
            "log_likelihood": round(self.log_likelihood, 4),
            "information_gain_vs_climatology": round(self.information_gain_vs_climatology, 4),
            "roc_auc_secondary": round(self.roc_auc, 4) if self.roc_auc is not None else None,
            "calibration_error": round(self.calibration_error, 4),
            "notes": "; ".join(self.notes),
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def brier_score(forecasts: np.ndarray, observations: np.ndarray) -> float:
    """Mean squared error of probabilistic forecasts."""
    f = np.asarray(forecasts, dtype=float)
    o = np.asarray(observations, dtype=float)
    return float(np.mean((f - o) ** 2))


def log_likelihood_score(forecasts: np.ndarray, observations: np.ndarray,
                         eps: float = 1e-12) -> float:
    """Mean Bernoulli log-likelihood. Higher is better."""
    f = np.clip(np.asarray(forecasts, dtype=float), eps, 1 - eps)
    o = np.asarray(observations, dtype=float)
    return float(np.mean(o * np.log(f) + (1 - o) * np.log(1 - f)))


def reliability_diagram(forecasts: np.ndarray, observations: np.ndarray,
                        n_bins: int = 10) -> list[tuple[float, float, int, float, float]]:
    """Bin forecasts and return (bin_lo, bin_hi, n, mean_forecast, observed_freq)."""
    f = np.asarray(forecasts, dtype=float)
    o = np.asarray(observations, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (f >= lo) & (f < hi) if i < n_bins - 1 else (f >= lo) & (f <= hi)
        if mask.sum() > 0:
            out.append((float(lo), float(hi), int(mask.sum()),
                        float(f[mask].mean()), float(o[mask].mean())))
        else:
            out.append((float(lo), float(hi), 0, float("nan"), float("nan")))
    return out


def information_gain(forecasts: np.ndarray, observations: np.ndarray,
                     reference: float, eps: float = 1e-12) -> float:
    """Per-sample information gain over a reference (climatology) probability.

    IG = mean[ y*log(p/p_ref) + (1-y)*log((1-p)/(1-p_ref)) ]
    Positive means the model beats the reference; negative means worse.
    """
    f = np.clip(np.asarray(forecasts, dtype=float), eps, 1 - eps)
    o = np.asarray(observations, dtype=float)
    ref = max(min(reference, 1 - eps), eps)
    return float(np.mean(
        o * np.log(f / ref) + (1 - o) * np.log((1 - f) / (1 - ref))
    ))


def roc_auc(forecasts: np.ndarray, observations: np.ndarray) -> Optional[float]:
    """ROC-AUC. Returns None if only one class is present.

    WARNING: ROC-AUC is reported as a SECONDARY diagnostic only. For rare
    events it can be misleadingly high and is not the primary measure.
    """
    f = np.asarray(forecasts, dtype=float)
    o = np.asarray(observations, dtype=int)
    if o.sum() == 0 or o.sum() == len(o):
        return None
    # Mann-Whitney U formulation with proper tie handling via scipy.rankdata.
    from scipy import stats
    n1 = int(o.sum())
    n0 = len(o) - n1
    ranks = stats.rankdata(f)
    sum_ranks_pos = ranks[o == 1].sum()
    auc = (sum_ranks_pos - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(auc)


# ---------------------------------------------------------------------------
# Chronological backtest runner
# ---------------------------------------------------------------------------


def run_chronological_backtest(
    events: list[CanonicalEvent],
    threshold: float,
    horizon: str,
    origin_start_year: int = 1995,
    origin_end_year: int = 2024,
    origin_step_years: int = 1,
    catalog_start_time: Optional[datetime] = None,
) -> BacktestResult:
    """Run an expanding-window chronological backtest.

    For each forecast origin t0 (yearly from origin_start_year to
    origin_end_year):
      1. Training set = events before t0 (and above threshold).
      2. Estimate Poisson rate from training set.
      3. Forecast P(N>=1 in [t0, t0+horizon)).
      4. Observation = whether any event above threshold occurred in
         [t0, t0+horizon).
      5. Record (forecast_prob, observed_binary).

    The training window EXPANDS (always from catalog start to t0), so later
    origins have more data. No shuffling. No future leakage.

    Parameters
    ----------
    threshold : magnitude threshold for the baseline
    horizon : one of '24h', '7d', '30d', '90d', '1y'
    origin_start_year, origin_end_year : yearly origins in [start, end)
    origin_step_years : step between origins (default 1 year)
    catalog_start_time : if given, training exposure computed from this to
        t0; otherwise from the earliest event in `events`.
    """
    hy = HORIZON_YEARS[horizon]
    horizon_td = timedelta(days=hy * 365.25)

    if catalog_start_time is None:
        catalog_start_time = min(e.origin_time_utc for e in events)

    origins: list[BacktestOrigin] = []
    for year in range(origin_start_year, origin_end_year, origin_step_years):
        t0 = datetime(year, 1, 1, tzinfo=timezone.utc)
        t1 = t0 + horizon_td
        # Training: events before t0
        train_events = [e for e in events if e.origin_time_utc < t0]
        if not train_events:
            continue
        # Estimate Poisson rate from training set
        train_res = estimate_temporal_poisson(
            train_events, threshold=threshold,
            start_time=catalog_start_time, end_time=t0,
            ci_method="garwood",
        )
        # Forecast
        p = train_res.rate.probability_at_least_one(hy)
        plo, phi = train_res.rate.probability_ci(hy)
        # Observation: events above threshold in [t0, t1)
        obs_events = [e for e in events if t0 <= e.origin_time_utc < t1
                      and (e.mw if e.mw is not None else e.original_magnitude) >= threshold]
        n_obs = len(obs_events)
        observed = 1 if n_obs > 0 else 0
        origins.append(BacktestOrigin(
            origin_time=t0, horizon=horizon, horizon_years=hy,
            threshold=threshold,
            n_train_events=train_res.rate.n_observed,
            train_exposure_years=train_res.rate.exposure_years,
            forecast_rate_per_year=train_res.rate.rate_per_year,
            forecast_probability=p,
            forecast_ci=(plo, phi),
            n_observed_in_horizon=n_obs,
            observed_binary=observed,
            model="temporal_poisson",
        ))

    if not origins:
        return BacktestResult(
            model="temporal_poisson", threshold=threshold, horizon=horizon,
            n_origins=0, n_positive=0, base_rate=float("nan"),
            mean_forecast_probability=float("nan"), brier=float("nan"),
            log_likelihood=float("nan"),
            information_gain_vs_climatology=float("nan"),
            roc_auc=None, reliability_bins=[], calibration_error=float("nan"),
            notes=["No valid backtest origins produced."],
        )

    forecasts = np.array([o.forecast_probability for o in origins])
    observations = np.array([o.observed_binary for o in origins], dtype=float)
    n_pos = int(observations.sum())
    base_rate = n_pos / len(origins)
    mean_p = float(forecasts.mean())

    bs = brier_score(forecasts, observations)
    ll = log_likelihood_score(forecasts, observations)
    ig = information_gain(forecasts, observations, reference=base_rate)
    auc = roc_auc(forecasts, observations)
    rel = reliability_diagram(forecasts, observations, n_bins=5)
    # Calibration error: mean |mean_p - obs_freq| over non-empty bins
    cal_errors = [abs(b[3] - b[4]) for b in rel if b[2] > 0 and not math.isnan(b[3])]
    cal_err = float(np.mean(cal_errors)) if cal_errors else float("nan")

    notes = []
    if n_pos == 0:
        notes.append(
            "ZERO positive observations across all origins; Brier/log-lik are "
            "computable but ROC-AUC is undefined and information gain is "
            "degenerate. This reflects the rarity of events at this threshold."
        )
    if n_pos < 10:
        notes.append(
            f"Only {n_pos} positive observations; metric estimates have high "
            "variance. Treat with caution."
        )
    notes.append(
        "ROC-AUC is reported as a SECONDARY diagnostic only; for rare events "
        "it can be misleading and is not the primary measure."
    )
    notes.append(
        "Information gain is computed against the climatology (base rate) "
        "reference, which is the appropriate null for rare-event forecasting."
    )

    return BacktestResult(
        model="temporal_poisson", threshold=threshold, horizon=horizon,
        n_origins=len(origins), n_positive=n_pos, base_rate=base_rate,
        mean_forecast_probability=mean_p, brier=bs, log_likelihood=ll,
        information_gain_vs_climatology=ig, roc_auc=auc,
        reliability_bins=rel, calibration_error=cal_err,
        origins=origins, notes=notes,
    )
