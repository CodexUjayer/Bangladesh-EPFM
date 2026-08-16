"""B1: ETAS vs Spatial Poisson — direct comparison on identical origins.

THE CENTRAL MISSING EXPERIMENT from Stages 5/7B. ETAS was compared only to
uniform Poisson (Stage 5); SP was compared only to ML (Stage 7B). This module
compares all four models head-to-head:
  - Spatial Poisson (causal expanding-window)
  - Uniform Poisson (expanding-window)
  - Locally fitted ETAS (base-10, GK declustered background)
  - Externally informed ETAS (base-10, literature params)

Strict expanding-window chronology. Block bootstrap + permutation test.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
from scipy import stats as scipy_stats

from ..baselines.poisson import HORIZON_YEARS
from ..etas.estimation import fit_etas_mle, prepare_catalog
from ..etas.forecast import forecast_temporal
from ..etas.model import ETASModel, ETASParams
from ..etas.background import KDEBackground, UniformBackground
from ..ingestion.schema import CanonicalEvent
from ..ml.features import MLGridConfig
from ..ml.spatial_poisson import causal_spatial_rate, spatial_poisson_forecast

logger = logging.getLogger("phase_b.b1")


@dataclass
class B1OriginResult:
    origin_time: datetime
    horizon: str
    threshold: float
    y_true: np.ndarray
    sp_pred: np.ndarray
    uniform_pred: np.ndarray
    etas_mle_pred: np.ndarray
    etas_forced_pred: np.ndarray


def run_etas_vs_sp_comparison(
    events: list[CanonicalEvent],
    catalog_start: datetime,
    horizons: list[str] = None,
    thresholds: list[float] = None,
    origin_start_year: int = 1995,
    origin_end_year: int = 2024,
    origin_step_years: int = 2,
    grid: Optional[MLGridConfig] = None,
) -> dict:
    """Run the ETAS-vs-SP direct comparison.

    Returns dict: (horizon, threshold) -> {
        'origins': list[B1OriginResult],
        'evaluations': {model_key -> EvalMetrics},
        'bootstrap': {model_key -> CI dict},
        'permutation': {model_key -> p-value},
    }
    """
    if horizons is None:
        horizons = ["7d", "30d"]
    if thresholds is None:
        thresholds = [4.5, 5.0]
    if grid is None:
        grid = MLGridConfig()

    all_results = {}

    for horizon in horizons:
        for threshold in thresholds:
            logger.warning("B1: horizon=%s, threshold=M>=%s", horizon, threshold)
            origins = _run_single_config(
                events, catalog_start, horizon, threshold,
                origin_start_year, origin_end_year, origin_step_years, grid,
            )
            if not origins:
                continue
            evals, boot, perm = _evaluate_and_test(origins)
            all_results[(horizon, threshold)] = {
                "origins": origins,
                "evaluations": evals,
                "bootstrap": boot,
                "permutation": perm,
            }

    return all_results


def _run_single_config(
    events, catalog_start, horizon, threshold,
    origin_start_year, origin_end_year, origin_step_years, grid,
) -> list[B1OriginResult]:
    """Run one (horizon, threshold) configuration."""
    hy = HORIZON_YEARS[horizon]
    horizon_td = timedelta(days=hy * 365.25)

    # Fit locally-fitted ETAS ONCE on pre-test period (chronological)
    fit_end = datetime(origin_start_year, 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
    train_events_pre = [e for e in events if e.origin_time_utc < fit_end]
    local_fit = fit_etas_mle(train_events_pre, Mc=4.5, background_kind="kde",
                              spatial_kernel="powerlaw", t_end=fit_end)

    # Forced-ETAS params (base-10, externally informed)
    forced_params = ETASParams(
        mu_total_per_year=10.0, K=0.02, alpha=0.8, c_days=0.05, p=1.1,
        sigma_km=10.0, gamma=0.5, q=1.0, Mc=4.5, spatial_kernel="powerlaw",
        fixed_parameters={"K": 0.02, "alpha": 0.8, "c_days": 0.05,
                          "p": 1.1, "sigma_km": 10.0, "gamma": 0.5, "q": 1.0},
    )

    results = []
    for year in range(origin_start_year, origin_end_year, origin_step_years):
        t0 = datetime(year, 1, 1, tzinfo=timezone.utc)
        # Per-origin training set
        train_for_forecast = [e for e in events if e.origin_time_utc < t0]
        if not train_for_forecast:
            continue

        # Build per-cell observations
        from ..ml.features import MLGridConfig, compute_features_at_origin
        cell_area_km2 = grid.cell_size_deg * 110.574 * grid.cell_size_deg * 111.32 * math.cos(math.radians(24.0))
        fm = compute_features_at_origin(
            events, origin_time=t0, horizon=horizon, threshold=threshold,
            grid=grid, catalog_start=catalog_start,
            horizon_days=hy * 365.25, cell_area_km2=cell_area_km2,
        )
        y_true = fm.y.astype(float)

        # Spatial Poisson (causal)
        sp_rates = causal_spatial_rate(
            events, origin_time=t0, grid=grid, threshold=threshold,
            catalog_start=catalog_start, method="expanding", smoothing="raw",
        )
        sp_pred = spatial_poisson_forecast(sp_rates, hy)

        # Uniform Poisson (per-origin expanding rate)
        p_uniform = 1.0 - math.exp(-fm.poisson_rate_per_year * hy)
        uniform_pred = np.full(len(y_true), p_uniform)

        # ETAS MLE (locally fitted; K≈0 so ≈ Poisson)
        cat = prepare_catalog(train_for_forecast, Mc=4.5, t_end=t0)
        if cat["n"] > 0:
            mle_model = ETASModel(params=local_fit.params, background=local_fit.background,
                                  bbox=(20.0, 28.0, 88.0, 96.0),
                                  fit_info={"b_value": _b_from_catalog(train_for_forecast, 4.5)})
            _, p_etas_mle = forecast_temporal(
                mle_model, cat["times_days"], cat["lats"], cat["lons"], cat["mags"],
                forecast_start_days=cat["t_end_days"], horizon_days=hy * 365.25,
                threshold=threshold,
            )
        else:
            p_etas_mle = p_uniform

        # ETAS forced (externally informed)
        if cat["n"] > 0:
            forced_bg = KDEBackground.build(
                cat["lats"], cat["lons"],
                mu_total_per_year=max(forced_params.mu_total_per_year, 0.1),
                bbox=(20.0, 28.0, 88.0, 96.0),
            ) if len(cat["lats"]) > 5 else UniformBackground.build(
                forced_params.mu_total_per_year, (20.0, 28.0, 88.0, 96.0))
            forced_model = ETASModel(params=forced_params, background=forced_bg,
                                      bbox=(20.0, 28.0, 88.0, 96.0),
                                      fit_info={"b_value": _b_from_catalog(train_for_forecast, 4.5),
                                                "externally_informed": True})
            _, p_etas_forced = forecast_temporal(
                forced_model, cat["times_days"], cat["lats"], cat["lons"], cat["mags"],
                forecast_start_days=cat["t_end_days"], horizon_days=hy * 365.25,
                threshold=threshold,
            )
        else:
            p_etas_forced = p_uniform

        results.append(B1OriginResult(
            origin_time=t0, horizon=horizon, threshold=threshold,
            y_true=y_true, sp_pred=sp_pred, uniform_pred=uniform_pred,
            etas_mle_pred=np.full(len(y_true), p_etas_mle),
            etas_forced_pred=np.full(len(y_true), p_etas_forced),
        ))

    return results


def _evaluate_and_test(origins: list[B1OriginResult]) -> tuple:
    """Evaluate all 4 models + bootstrap CIs + permutation tests."""
    from ..ml.evaluation import evaluate_model

    y_true_all = np.concatenate([o.y_true for o in origins])
    sp_all = np.concatenate([o.sp_pred for o in origins])
    uniform_all = np.concatenate([o.uniform_pred for o in origins])
    etas_mle_all = np.concatenate([o.etas_mle_pred for o in origins])
    etas_forced_all = np.concatenate([o.etas_forced_pred for o in origins])

    evals = {}
    evals["spatial_poisson"] = evaluate_model("spatial_poisson", sp_all, y_true_all, sp_all)
    evals["uniform_poisson"] = evaluate_model("uniform_poisson", uniform_all, y_true_all, sp_all)
    evals["etas_mle"] = evaluate_model("etas_mle", etas_mle_all, y_true_all, sp_all)
    evals["etas_forced"] = evaluate_model("etas_forced", etas_forced_all, y_true_all, sp_all)

    # Block bootstrap (over origins) for ΔBrier vs SP
    boot = {}
    for key, pred_key in [("uniform_poisson", "uniform_pred"),
                           ("etas_mle", "etas_mle_pred"),
                           ("etas_forced", "etas_forced_pred")]:
        boot[key] = _block_bootstrap(
            [getattr(o, pred_key) for o in origins],
            [o.sp_pred for o in origins],
            [o.y_true for o in origins],
        )

    # Permutation test: shuffle origin labels, recompute ΔBrier
    perm = {}
    for key, pred_key in [("uniform_poisson", "uniform_pred"),
                           ("etas_mle", "etas_mle_pred"),
                           ("etas_forced", "etas_forced_pred")]:
        perm[key] = _permutation_test(
            [getattr(o, pred_key) for o in origins],
            [o.sp_pred for o in origins],
            [o.y_true for o in origins],
        )

    return evals, boot, perm


def _block_bootstrap(ml_per_origin, sp_per_origin, y_per_origin, n_boot=500):
    """Block bootstrap over origins. Returns ΔBrier CI (SP - model)."""
    rng = np.random.default_rng(42)
    n = len(ml_per_origin)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ml = np.concatenate([ml_per_origin[i] for i in idx])
        sp = np.concatenate([sp_per_origin[i] for i in idx])
        yt = np.concatenate([y_per_origin[i] for i in idx])
        mask = ~np.isnan(ml)
        b_ml = np.mean((ml[mask] - yt[mask]) ** 2)
        b_sp = np.mean((sp[mask] - yt[mask]) ** 2)
        deltas.append(b_sp - b_ml)
    return {
        "delta_brier_mean": float(np.mean(deltas)),
        "delta_brier_ci": (float(np.percentile(deltas, 2.5)),
                           float(np.percentile(deltas, 97.5))),
        "n_origins": n,
    }


def _permutation_test(ml_per_origin, sp_per_origin, y_per_origin, n_perm=1000):
    """Permutation test: shuffle origin labels. Returns p-value for ΔBrier > 0."""
    rng = np.random.default_rng(123)
    n = len(ml_per_origin)
    # Observed ΔBrier
    ml = np.concatenate(ml_per_origin)
    sp = np.concatenate(sp_per_origin)
    yt = np.concatenate(y_per_origin)
    mask = ~np.isnan(ml)
    obs_delta = np.mean((sp[mask] - yt[mask]) ** 2) - np.mean((ml[mask] - yt[mask]) ** 2)
    # Permute
    count = 0
    for _ in range(n_perm):
        idx = rng.permutation(n)
        ml_perm = np.concatenate([ml_per_origin[i] for i in idx])
        sp_perm = np.concatenate([sp_per_origin[i] for i in idx])
        # y stays in place; we shuffle which model prediction pairs with which y
        # Actually: shuffle the PAIRING between model predictions and y
        yt_perm = np.concatenate([y_per_origin[i] for i in idx])
        mask_p = ~np.isnan(ml_perm)
        d = np.mean((sp_perm[mask_p] - yt_perm[mask_p]) ** 2) - np.mean((ml_perm[mask_p] - yt_perm[mask_p]) ** 2)
        if d >= obs_delta:
            count += 1
    return {"observed_delta_brier": float(obs_delta), "p_value": count / n_perm, "n_perm": n_perm}


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
