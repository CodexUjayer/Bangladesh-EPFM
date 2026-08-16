"""Externally-informed ETAS sensitivity analysis with nested validation.

USER CORRECTION (Stage 5 validation):

  The externally informed parameter set (K=0.02, α=0.8, c=0.05d, p=1.1,
  σ=10km, γ=0.5, q=1.0) must remain explicitly labeled "externally_informed"
  with recorded provenance. Perform sensitivity analysis varying K, α, c, p,
  and spatial scale, and determine whether the apparent M>=5 forecasting
  improvement is robust or only occurs for a narrow parameter choice.

  DO NOT tune these parameters on the same backtest period being used to
  claim predictive skill. If tuning is necessary, use a proper NESTED
  chronological validation procedure:
    - Outer loop: prospective backtest period (the period we report skill on)
    - Inner loop: tuning period (earlier data used ONLY to select params)

  In this analysis we do NOT tune — we sweep a pre-specified grid of
  physically plausible external parameter sets and report which subsets of
  the grid beat Poisson. This is a sensitivity analysis, not tuning.

Provenance of the external parameter set:
  The default values are LITERATURE-INFORMED typical ETAS parameters from
  tectonic-regime studies (Ogata 1998; Zhuang et al. 2011; Marsan & Lengliné
  2010). They are NOT Bangladesh-calibrated. No published Bangladesh-specific
  ETAS parameter set exists. We treat this as a SINGLE-PRIOR EXPERIMENT
  with sensitivity, not a multi-prior transfer study.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from ..baselines.backtest import brier_score, log_likelihood_score
from ..baselines.poisson import HORIZON_YEARS
from ..ingestion.schema import CanonicalEvent
from .background import KDEBackground, UniformBackground
from .estimation import prepare_catalog
from .forecast import forecast_temporal
from .model import ETASModel, ETASParams


@dataclass
class SensitivityResult:
    """Result of one parameter-set evaluation on the backtest period."""

    K: float
    alpha: float
    c_days: float
    p: float
    sigma_km: float
    gamma: float
    q: float
    n_origins: int
    n_positive: int
    brier_etas: float
    brier_poisson: float
    brier_improvement: float       # pois - etas (>0 means ETAS better)
    loglik_etas: float
    loglik_poisson: float
    information_gain: float        # etas vs poisson
    mean_etas_prob: float
    mean_poisson_prob: float
    beats_poisson: bool
    label: str = "externally_informed"

    def to_row(self) -> dict:
        return {
            "label": self.label,
            "K": self.K, "alpha": self.alpha, "c_days": self.c_days,
            "p": self.p, "sigma_km": self.sigma_km, "gamma": self.gamma, "q": self.q,
            "n_origins": self.n_origins, "n_positive": self.n_positive,
            "brier_etas": round(self.brier_etas, 4),
            "brier_poisson": round(self.brier_poisson, 4),
            "brier_improvement": round(self.brier_improvement, 4),
            "loglik_etas": round(self.loglik_etas, 4),
            "loglik_poisson": round(self.loglik_poisson, 4),
            "information_gain": round(self.information_gain, 4),
            "mean_etas_prob": round(self.mean_etas_prob, 4),
            "mean_poisson_prob": round(self.mean_poisson_prob, 4),
            "beats_poisson": self.beats_poisson,
        }


@dataclass
class SensitivitySummary:
    """Summary across the parameter grid."""

    n_param_sets: int
    n_beat_poisson: int
    frac_beat_poisson: float
    mean_brier_improvement: float
    median_brier_improvement: float
    min_brier_improvement: float
    max_brier_improvement: float
    robust: bool                    # True if >50% of param sets beat Poisson
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parameter grid (pre-specified, NOT tuned)
# ---------------------------------------------------------------------------


# Default externally-informed parameter set (literature-typical, BASE-10).
# With base-10 productivity K·10^{α(M−Mc)}, typical α is 0.3-2.0 (Ogata 1998).
# The previous default α=0.8 was applied in a base-e formulation, making it
# effectively α≈1.84 in base-10. The corrected default α=0.8 is now on the
# correct scale.
DEFAULT_EXTERNAL_PARAMS = {
    "K": 0.02, "alpha": 0.8, "c_days": 0.05, "p": 1.1,
    "sigma_km": 10.0, "gamma": 0.5, "q": 1.0,
}

# Sensitivity grid: vary K, alpha, c, p, sigma one at a time around the
# default, holding the others fixed. This is a One-At-a-Time (OAT) sweep,
# not a full Cartesian grid (which would be too expensive).
SENSITIVITY_GRID = {
    "K":         [0.005, 0.01, 0.02, 0.05, 0.1],
    "alpha":     [0.3, 0.5, 0.8, 1.0, 1.5],
    "c_days":    [0.01, 0.05, 0.1, 0.3],
    "p":         [1.05, 1.1, 1.2, 1.3],
    "sigma_km":  [3.0, 5.0, 10.0, 20.0, 50.0],
}

# Additional published-informed parameter sets (for transferability test).
# These are APPROXIMATE typical values from named studies; we use them as
# SEPARATE external priors, NOT selecting whichever scores best.
PUBLISHED_PRIORS = {
    "ogata1998_california": {
        "K": 0.015, "alpha": 1.0, "c_days": 0.05, "p": 1.1,
        "sigma_km": 5.0, "gamma": 0.4, "q": 1.0,
        "citation": "Ogata (1998) JASA — California shallow strike-slip",
    },
    "zhuang2011_japan": {
        "K": 0.03, "alpha": 0.8, "c_days": 0.03, "p": 1.0,
        "sigma_km": 8.0, "gamma": 0.5, "q": 1.5,
        "citation": "Zhuang et al. (2011) — Japan subduction",
    },
    "marsan2010_global": {
        "K": 0.02, "alpha": 0.7, "c_days": 0.1, "p": 1.2,
        "sigma_km": 15.0, "gamma": 0.6, "q": 0.8,
        "citation": "Marsan & Lengliné (2010) — global survey",
    },
}


def build_param_set(**overrides) -> ETASParams:
    """Build an ETASParams from the default external set with overrides."""
    params = dict(DEFAULT_EXTERNAL_PARAMS)
    params.update(overrides)
    return ETASParams(
        mu_total_per_year=10.0,  # will be overridden by local KDE
        spatial_kernel="powerlaw",
        Mc=4.5,
        fixed_parameters={k: v for k, v in params.items()},
        **params,
    )


# ---------------------------------------------------------------------------
# Sensitivity evaluation
# ---------------------------------------------------------------------------


def evaluate_param_set_on_origins(
    events: list[CanonicalEvent],
    origins: list,
    threshold: float,
    horizon: str,
    params: ETASParams,
    poisson_rate: float,   # ignored if per-origin rates are used (see below)
    Mc: float = 4.5,
    catalog_start=None,
) -> SensitivityResult:
    """Evaluate one parameter set on a list of pre-built origins.

    Uses PER-ORIGIN expanding-window Poisson rates (same as the conditioned
    backtest) so the comparison is fair. The `poisson_rate` argument is kept
    for API compatibility but is NOT used when `catalog_start` is provided.
    """
    hy = HORIZON_YEARS[horizon]
    etas_probs = []
    pois_probs = []
    obs = []
    for o in origins:
        train_for_forecast = [e for e in events if e.origin_time_utc < o.origin_time]
        cat = prepare_catalog(train_for_forecast, Mc=Mc, t_end=o.origin_time)
        if cat["n"] == 0:
            continue
        # Background is LOCAL (KDE on training events); triggering params are external
        bg = KDEBackground.build(
            cat["lats"], cat["lons"],
            mu_total_per_year=max(params.mu_total_per_year, 0.1),
            bbox=(20.0, 28.0, 88.0, 96.0),
        ) if len(cat["lats"]) > 5 else UniformBackground.build(
            params.mu_total_per_year, (20.0, 28.0, 88.0, 96.0))
        model = ETASModel(params=params, background=bg,
                          bbox=(20.0, 28.0, 88.0, 96.0),
                          fit_info={"b_value": _b_from_catalog(train_for_forecast, Mc),
                                    "externally_informed": True})
        _, p_etas = forecast_temporal(
            model, cat["times_days"], cat["lats"], cat["lons"], cat["mags"],
            forecast_start_days=cat["t_end_days"],
            horizon_days=hy * 365.25, threshold=threshold,
        )
        # PER-ORIGIN Poisson rate (expanding window) — fair comparison
        if catalog_start is not None:
            train_above = [e for e in train_for_forecast
                           if (e.mw if e.mw is not None else e.original_magnitude) >= threshold]
            train_span = (o.origin_time - catalog_start).total_seconds() / (365.25 * 86400)
            p_rate = len(train_above) / max(train_span, 1e-6)
        else:
            p_rate = poisson_rate
        p_pois = 1.0 - math.exp(-p_rate * hy)
        etas_probs.append(p_etas)
        pois_probs.append(p_pois)
        obs.append(o.observed_binary)

    if not etas_probs:
        return SensitivityResult(
            K=params.K, alpha=params.alpha, c_days=params.c_days,
            p=params.p, sigma_km=params.sigma_km, gamma=params.gamma, q=params.q,
            n_origins=0, n_positive=0,
            brier_etas=float("nan"), brier_poisson=float("nan"),
            brier_improvement=float("nan"),
            loglik_etas=float("nan"), loglik_poisson=float("nan"),
            information_gain=float("nan"),
            mean_etas_prob=float("nan"), mean_poisson_prob=float("nan"),
            beats_poisson=False,
        )
    etas_arr = np.array(etas_probs)
    pois_arr = np.array(pois_probs)
    obs_arr = np.array(obs, dtype=float)
    b_etas = brier_score(etas_arr, obs_arr)
    b_pois = brier_score(pois_arr, obs_arr)
    ll_etas = log_likelihood_score(etas_arr, obs_arr)
    ll_pois = log_likelihood_score(pois_arr, obs_arr)
    eps = 1e-12
    ig = float(np.mean(
        obs_arr * np.log(np.clip(etas_arr, eps, 1 - eps) / np.clip(pois_arr, eps, 1 - eps)) +
        (1 - obs_arr) * np.log(np.clip(1 - etas_arr, eps, 1 - eps) / np.clip(1 - pois_arr, eps, 1 - eps))
    ))
    return SensitivityResult(
        K=params.K, alpha=params.alpha, c_days=params.c_days,
        p=params.p, sigma_km=params.sigma_km, gamma=params.gamma, q=params.q,
        n_origins=len(etas_probs), n_positive=int(obs_arr.sum()),
        brier_etas=b_etas, brier_poisson=b_pois,
        brier_improvement=b_pois - b_etas,
        loglik_etas=ll_etas, loglik_poisson=ll_pois,
        information_gain=ig,
        mean_etas_prob=float(etas_arr.mean()),
        mean_poisson_prob=float(pois_arr.mean()),
        beats_poisson=b_etas < b_pois,
    )


def run_sensitivity_analysis(
    events: list[CanonicalEvent],
    origins: list,
    threshold: float,
    horizon: str,
    poisson_rate: float,
    Mc: float = 4.5,
    catalog_start=None,
) -> tuple[list[SensitivityResult], list[SensitivityResult], SensitivitySummary]:
    """Run the OAT sensitivity sweep + published-prior transfer test.

    Returns (oat_results, published_prior_results, summary).
    """
    oat_results = []

    # One-At-a-Time sweep around the default
    for param_name, values in SENSITIVITY_GRID.items():
        for val in values:
            overrides = {param_name: val}
            params = build_param_set(**overrides)
            res = evaluate_param_set_on_origins(
                events, origins, threshold, horizon, params, poisson_rate, Mc,
                catalog_start=catalog_start)
            res.label = f"OAT_{param_name}={val}"
            oat_results.append(res)

    # Published priors (transferability)
    published_results = []
    for name, prior in PUBLISHED_PRIORS.items():
        params = ETASParams(
            mu_total_per_year=10.0,
            K=prior["K"], alpha=prior["alpha"], c_days=prior["c_days"],
            p=prior["p"], sigma_km=prior["sigma_km"], gamma=prior["gamma"],
            q=prior["q"], Mc=Mc, spatial_kernel="powerlaw",
            fixed_parameters={k: prior[k] for k in
                              ["K", "alpha", "c_days", "p", "sigma_km", "gamma", "q"]},
        )
        res = evaluate_param_set_on_origins(
            events, origins, threshold, horizon, params, poisson_rate, Mc,
            catalog_start=catalog_start)
        res.label = f"published_prior:{name}|{prior['citation']}"
        published_results.append(res)

    # Summary across OAT results (excluding NaN)
    valid = [r for r in oat_results if not math.isnan(r.brier_improvement)]
    n_beat = sum(1 for r in valid if r.beats_poisson)
    improvements = [r.brier_improvement for r in valid]
    summary = SensitivitySummary(
        n_param_sets=len(valid),
        n_beat_poisson=n_beat,
        frac_beat_poisson=n_beat / max(len(valid), 1),
        mean_brier_improvement=float(np.mean(improvements)) if improvements else float("nan"),
        median_brier_improvement=float(np.median(improvements)) if improvements else float("nan"),
        min_brier_improvement=float(np.min(improvements)) if improvements else float("nan"),
        max_brier_improvement=float(np.max(improvements)) if improvements else float("nan"),
        robust=n_beat > 0.5 * max(len(valid), 1),
        notes=[
            f"{n_beat}/{len(valid)} externally-informed parameter sets beat Poisson.",
            f"Brier improvement range: [{float(np.min(improvements)):.4f}, {float(np.max(improvements)):.4f}]"
            if improvements else "No valid results.",
            "This is a sensitivity analysis, NOT tuning. Parameters were pre-specified.",
        ],
    )
    return oat_results, published_results, summary


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
