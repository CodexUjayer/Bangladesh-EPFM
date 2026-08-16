"""Region-Specific ETAS Model — FINAL_v4.0 CANDIDATE_REGION_SPECIFIC_ETAS.

This module implements four ETAS variants designed to test whether the
observed scientific contradiction — strong Omori clustering (R≈24×) coexisting
with vanishing ETAS productivity (K≈0) — is caused by ETAS model
misspecification rather than absence of triggering.

===========================  SCIENTIFIC BACKGROUND  ===========================

The project established (Stage 5/8) that:

  1. The non-parametric Omori diagnostic shows R≈24× post-mainshock rate
     enhancement at Δt≈0.013 day (≈18 minutes), decaying to background
     within ~1 day. This is a STRONG clustering signal.

  2. Maximum-likelihood ETAS fits K≈0, α≈0 in ALL depth regimes
     (shallow/intermediate/deep). The standard ETAS model finds NO
     triggering component.

These findings are inconsistent under standard ETAS assumptions. The
standard Omori-Utsu temporal kernel g(τ) = (p-1)·c^{p-1}/(τ+c)^p cannot
represent clustering that peaks at Δt < 0.01 day because the fitted c
hits its upper bound (1.0 day), smoothing the sharp short-lag peak into
a broad, low-amplitude bump that the MLE rejects in favour of K=0.

===========================  MODEL FAMILY  ===========================

Four scientifically justified ETAS variants are implemented:

  ETAS-A — Baseline ETAS (reference only)
      Standard 8-parameter ETAS: μ + K·10^{α(M-Mc)}·g(τ;c,p)·f(r;σ,γ,q)
      Estimated by multi-start MLE on the development period.
      Expected to reproduce the K≈0 finding (sanity check).

  ETAS-B — Depth-stratified ETAS
      Independent ETAS fits for shallow (<25 km), intermediate (25-70 km),
      and deep (≥70 km) event subsets. Tests whether depth-mixing masks
      triggering that is present within individual depth regimes.
      Background rate, K, α, c, p, σ all estimated separately per regime.

  ETAS-C — Depth-dependent spatial kernels
      A single ETAS fit but with the spatial kernel scale σ depending on
      the source-event depth: σ(D) = σ_0 · (1 + κ·D/D_ref).
      Deep events (Indo-Burman slab) are expected to have broader spatial
      influence than shallow crustal events.

  ETAS-D — Modified temporal kernels
      Replaces the standard Omori-Utsu kernel with a form that allows
      very-short-lag peaks: g(τ) = (p-1)·c^{p-1}/(τ+c)^p with c allowed
      down to 1e-4 day (≈9 seconds), PLUS an alternative exponential
      decay form g(τ) = (1/τ_0)·exp(-τ/τ_0) for comparison.
      Tests whether the Bangladesh clustering deviates from classical
      Omori-Utsu at short lags.

===========================  DIAGNOSTICS  ===========================

For each variant we report:
  K         — productivity
  α         — magnitude-scaling exponent
  c         — Omori temporal offset (days)
  p         — Omori temporal decay exponent
  σ         — spatial kernel scale (km)
  γ         — magnitude-spatial scaling
  q         — spatial power-law exponent
  μ         — background rate (per year, regional total)
  n_branch  — branching ratio (analytic: K·β/(β-α), β=b·ln10)
  n_emp     — empirical branching ratio (mean K·10^{α(M-Mc)} over catalog)
  trig_dist — characteristic triggering distance (km) = σ·exp(γ·(M*−Mc))
  τ_decay   — temporal decay scale (days) = c/(p-1)
  R_peak    — non-parametric Omori peak rate ratio
  Δt_peak   — lag at peak R (days)

===========================  INTEGRITY  ===========================

This module does NOT modify v1, v2, or v3 source code, ledgers, scores,
or frozen artifacts. It produces a completely separate v4 namespace.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger("v4.etas")

# === Frozen parameters (for fair comparison — DO NOT MODIFY) ===
V1_MC = 4.13
V1_B = 0.808
BBOX = (20.0, 28.0, 88.0, 96.0)
N_LAT = 8
N_LON = 8
N_CELLS = 64

# Depth regime cutoffs (consistent with src/etas/depth_analysis.py)
DEPTH_SHALLOW_MAX = 25.0
DEPTH_INTERMEDIATE_MAX = 70.0

# Earth radius (km)
EARTH_R_KM = 6371.0088

# Predefined parameter bounds for MLE (extended for ETAS-D short-lag c)
PARAM_BOUNDS = {
    "mu_total_per_year": (1e-6, 1e4),
    "K":                 (1e-8, 1.0),
    "alpha":             (0.0, 3.0),
    "c_days":            (1e-4, 1.0),   # extended lower bound for ETAS-D
    "p":                 (1.01, 2.5),
    "sigma_km":          (0.5, 200.0),
    "gamma":             (0.0, 2.0),
    "q":                 (0.5, 3.0),
}

# ETAS-D: exponential temporal kernel parameter
TAU_BOUNDS = (1e-4, 10.0)   # days


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def _equirect_km(lat1: float, lon1: float, lat2: float, lon2: float,
                  cos_lat_ref: float = math.cos(math.radians(24.0))) -> float:
    """Local equirectangular approximation in km."""
    dlat = (lat2 - lat1) * 110.574
    dlon = (lon2 - lon1) * 111.320 * cos_lat_ref
    return math.sqrt(dlat * dlat + dlon * dlon)


def _equirect_km_vec(lat1, lon1, lat2, lon2,
                      cos_lat_ref=math.cos(math.radians(24.0))):
    """Vectorised equirectangular distance in km."""
    dlat = (lat2 - lat1) * 110.574
    dlon = (lon2 - lon1) * 111.320 * cos_lat_ref
    return np.sqrt(dlat * dlat + dlon * dlon)


# ---------------------------------------------------------------------------
# ETAS parameter container
# ---------------------------------------------------------------------------

@dataclass
class ETASParams:
    """ETAS parameters (8 standard + variant-specific extras)."""
    mu_total_per_year: float = 0.0
    K: float = 0.0
    alpha: float = 0.0
    c_days: float = 0.05
    p: float = 1.1
    sigma_km: float = 10.0
    gamma: float = 0.5
    q: float = 1.0
    Mc: float = V1_MC
    # ETAS-C: depth-dependent spatial kernel
    kappa_depth: float = 0.0       # σ(D) = σ_0 · (1 + κ·D/D_ref)
    D_ref_km: float = 50.0
    # ETAS-D: temporal kernel type
    temporal_kernel: str = "omori"  # "omori" or "exponential"
    tau_days: float = 0.01          # for exponential kernel


@dataclass
class ETASFitResult:
    """Result of fitting an ETAS variant."""
    params: ETASParams
    log_likelihood: float
    n_events: int
    n_params: int
    aic: float
    converged: bool
    notes: list[str] = field(default_factory=list)
    # Diagnostics
    branching_ratio_analytic: float = 0.0
    branching_ratio_empirical: float = 0.0
    triggering_distance_km: float = 0.0
    temporal_decay_scale_days: float = 0.0


# ---------------------------------------------------------------------------
# Catalog preparation
# ---------------------------------------------------------------------------

def prepare_catalog(events: list, mc: float, t_start: datetime, t_end: datetime,
                     depth_range: Optional[tuple] = None) -> dict:
    """Prepare a catalog subset for ETAS fitting.

    Returns a dict with arrays: times_days, lats, lons, depths, mags, n_events.
    Times are converted to days since t_start.
    """
    obs = []
    for e in events:
        if e.origin_time_utc < t_start or e.origin_time_utc >= t_end:
            continue
        m = e.mw if e.mw is not None else e.original_magnitude
        if m is None or m < mc:
            continue
        if e.depth_km is None:
            continue
        if depth_range is not None:
            if not (depth_range[0] <= e.depth_km < depth_range[1]):
                continue
        obs.append((
            (e.origin_time_utc - t_start).total_seconds() / 86400.0,
            e.latitude, e.longitude, e.depth_km, m
        ))
    obs.sort(key=lambda x: x[0])
    if not obs:
        return {"times_days": np.array([]), "lats": np.array([]),
                "lons": np.array([]), "depths": np.array([]),
                "mags": np.array([]), "n_events": 0}
    arr = np.array(obs)
    return {
        "times_days": arr[:, 0],
        "lats": arr[:, 1],
        "lons": arr[:, 2],
        "depths": arr[:, 3],
        "mags": arr[:, 4],
        "n_events": len(obs),
    }


# ---------------------------------------------------------------------------
# ETAS conditional intensity and likelihood
# ---------------------------------------------------------------------------

def _omori_temporal(tau_days: np.ndarray, c: float, p: float) -> np.ndarray:
    """Omori-Utsu temporal kernel, normalised so ∫₀^∞ g = 1.

    g(τ) = (p-1) · c^{p-1} / (τ + c)^p   for τ ≥ 0.
    """
    tau = np.asarray(tau_days, dtype=float)
    return (p - 1.0) * (c ** (p - 1.0)) / np.maximum(tau + c, 1e-12) ** p


def _exponential_temporal(tau_days: np.ndarray, tau_0: float) -> np.ndarray:
    """Exponential temporal kernel (alternative for ETAS-D).

    g(τ) = (1/τ_0) · exp(-τ/τ_0)   for τ ≥ 0.
    Normalised so ∫₀^∞ g = 1.
    """
    tau = np.asarray(tau_days, dtype=float)
    return (1.0 / max(tau_0, 1e-12)) * np.exp(-tau / max(tau_0, 1e-12))


def _powerlaw_spatial(r_km: np.ndarray, sigma: float, q: float) -> np.ndarray:
    """Power-law spatial kernel, normalised over the plane.

    f(r; σ, q) = (q-1)/(π·σ²) · [1 + (r/σ)²]^(-(1+q)/2)
    """
    r = np.asarray(r_km, dtype=float)
    return (q - 1.0) / (math.pi * sigma * sigma) * \
           (1.0 + (r / sigma) ** 2) ** (-(1.0 + q) / 2.0)


def _spatial_sigma_for_depth(depth_km: float, sigma_0: float, kappa: float,
                              d_ref: float) -> float:
    """ETAS-C: depth-dependent spatial kernel scale."""
    return sigma_0 * (1.0 + kappa * depth_km / d_ref)


def conditional_intensity(
    t_days: float, lat: float, lon: float,
    history: dict, params: ETASParams,
) -> float:
    """ETAS conditional intensity λ(t, x) at a point in space-time.

    λ = μ + Σ_i K·10^{α(M_i-Mc)} · g(t-t_i; c,p) · f(r_i; σ_i, q)

    where σ_i depends on the source-event depth for ETAS-C.
    """
    # Background rate per (deg²·year) → convert to per (deg²·day)
    # The total regional rate μ_total_per_year is spread uniformly over the
    # study area in deg². We evaluate the intensity at a single point, so
    # background contribution = μ_total / (area_deg² * 365.25) per day per deg².
    area_deg2 = (BBOX[1] - BBOX[0]) * (BBOX[3] - BBOX[2])
    mu_per_day_per_deg2 = params.mu_total_per_year / (area_deg2 * 365.25)

    lam = mu_per_day_per_deg2
    if history["n_events"] == 0:
        return lam

    # Vectorise: all history events
    t_hist = history["times_days"]
    lag = t_days - t_hist
    mask = lag > 0
    if not mask.any():
        return lam

    lag_pos = lag[mask]
    lats_h = history["lats"][mask]
    lons_h = history["lons"][mask]
    depths_h = history["depths"][mask]
    mags_h = history["mags"][mask]

    # Temporal kernel
    if params.temporal_kernel == "exponential":
        g = _exponential_temporal(lag_pos, params.tau_days)
    else:
        g = _omori_temporal(lag_pos, params.c_days, params.p)

    # Spatial kernel (power-law with optional depth-dependent σ)
    r_km = _equirect_km_vec(lat, lon, lats_h, lons_h)
    if params.kappa_depth != 0.0:
        sigmas = np.array([_spatial_sigma_for_depth(d, params.sigma_km,
                                                     params.kappa_depth, params.D_ref_km)
                           for d in depths_h])
        # f(r; σ_i, q) per event
        f = (params.q - 1.0) / (math.pi * sigmas * sigmas) * \
            (1.0 + (r_km / sigmas) ** 2) ** (-(1.0 + params.q) / 2.0)
    else:
        f = _powerlaw_spatial(r_km, params.sigma_km, params.q)

    # Magnitude productivity
    mag_factor = params.K * np.power(10.0, params.alpha * (mags_h - params.Mc))

    lam += float(np.sum(mag_factor * g * f))
    return lam


def _neg_log_likelihood(params_vec: np.ndarray, history: dict, mc: float,
                         variant: str, t_end_days: float) -> float:
    """Negative log-likelihood for ETAS MLE.

    The ETAS log-likelihood (Ogata 1988) is:
      log L = Σ_i log λ(t_i, x_i) - ∫ λ(t, x) dt dx

    The integral term simplifies to:
      ∫ λ dt dx = μ·T·A + Σ_i K·10^{α(M_i-Mc)} · G(t_end - t_i)
    where G(Δ) = ∫₀^Δ g(τ) dτ (temporal kernel CDF) and the spatial kernel
    integrates to 1 over the plane.
    """
    # Unpack parameters
    if variant == "D_exponential":
        mu, K, alpha, tau, sigma, gamma, q = params_vec
        c, p = 0.05, 1.1  # unused for exponential
        temporal_kernel = "exponential"
    elif variant == "C_depth_spatial":
        mu, K, alpha, c, p, sigma, gamma, q, kappa = params_vec
        temporal_kernel = "omori"
    else:  # A_baseline, B_depth_stratified
        mu, K, alpha, c, p, sigma, gamma, q = params_vec
        kappa = 0.0
        temporal_kernel = "omori"

    if K < 1e-8 or alpha < 1e-4:
        # Background-only: log L = Σ log(μ/A) - μ·T  (uniform Poisson)
        area_deg2 = (BBOX[1] - BBOX[0]) * (BBOX[3] - BBOX[2])
        n = history["n_events"]
        if n == 0 or mu <= 0:
            return 1e10
        mu_per_day = mu / 365.25
        rate_per_day_per_deg2 = mu_per_day / area_deg2
        log_l = n * math.log(rate_per_day_per_deg2) - mu_per_day * t_end_days
        return -log_l

    params = ETASParams(
        mu_total_per_year=mu, K=K, alpha=alpha, c_days=c, p=p,
        sigma_km=sigma, gamma=gamma, q=q, Mc=mc,
        kappa_depth=kappa, temporal_kernel=temporal_kernel,
    )

    n = history["n_events"]
    if n == 0:
        return 1e10

    # Sum of log λ(t_i, x_i)
    log_l = 0.0
    times = history["times_days"]
    lats = history["lats"]
    lons = history["lons"]
    # Cap the number of contributing ancestors for runtime (only events within
    # a temporal window matter; the Omori kernel decays so events older than
    # ~10·c/(p-1) days contribute negligibly).
    if temporal_kernel == "exponential":
        tau_window = max(10.0 * tau, 30.0)
    else:
        tau_window = max(10.0 * c / max(p - 1.0, 0.01), 30.0)

    for i in range(n):
        t_i = times[i]
        # Find ancestors within the temporal window
        lag = t_i - times
        mask = (lag > 0) & (lag <= tau_window)
        if not mask.any():
            # Background only
            area_deg2 = (BBOX[1] - BBOX[0]) * (BBOX[3] - BBOX[2])
            lam = mu / (area_deg2 * 365.25)
        else:
            lag_pos = lag[mask]
            lats_h = lats[mask]
            lons_h = lons[mask]
            depths_h = history["depths"][mask]
            mags_h = history["mags"][mask]

            if temporal_kernel == "exponential":
                g = _exponential_temporal(lag_pos, tau)
            else:
                g = _omori_temporal(lag_pos, c, p)

            r_km = _equirect_km_vec(lats[i], lons[i], lats_h, lons_h)
            if kappa != 0.0:
                sigmas = sigma * (1.0 + kappa * depths_h / 50.0)
                f = (q - 1.0) / (math.pi * sigmas * sigmas) * \
                    (1.0 + (r_km / sigmas) ** 2) ** (-(1.0 + q) / 2.0)
            else:
                f = _powerlaw_spatial(r_km, sigma, q)

            mag_factor = K * np.power(10.0, alpha * (mags_h - mc))
            area_deg2 = (BBOX[1] - BBOX[0]) * (BBOX[3] - BBOX[2])
            lam = mu / (area_deg2 * 365.25) + float(np.sum(mag_factor * g * f))

        if lam <= 0:
            return 1e10
        log_l += math.log(lam)

    # Integral term: ∫ λ dt dx = μ·T + Σ_i K·10^{α(M_i-Mc)} · G(t_end - t_i)
    # G(Δ) for Omori = 1 - (c/(c+Δ))^{p-1}  (CDF from 0 to Δ)
    # G(Δ) for exponential = 1 - exp(-Δ/τ_0)
    mags = history["mags"]
    mag_factor_all = K * np.power(10.0, alpha * (mags - mc))
    delta = t_end_days - times
    if temporal_kernel == "exponential":
        G = 1.0 - np.exp(-delta / max(tau, 1e-12))
    else:
        G = 1.0 - (c / (c + delta)) ** (p - 1.0)
    integral = mu / 365.25 * t_end_days + float(np.sum(mag_factor_all * G))

    log_l -= integral
    if not math.isfinite(log_l):
        return 1e10
    return -log_l


# ---------------------------------------------------------------------------
# MLE fitting
# ---------------------------------------------------------------------------

def fit_etas_mle(
    events: list, mc: float, t_start: datetime, t_end: datetime,
    variant: str = "A_baseline",
    depth_range: Optional[tuple] = None,
    b_value: float = V1_B,
) -> ETASFitResult:
    """Fit an ETAS variant by maximum likelihood.

    Parameters
    ----------
    events : list of CanonicalEvent
    mc : magnitude of completeness
    t_start, t_end : datetime; fit window
    variant : "A_baseline", "B_depth_stratified", "C_depth_spatial",
              "D_omori_extended", or "D_exponential"
    depth_range : optional (d_min, d_max) for depth-stratified fitting (ETAS-B)
    b_value : for branching-ratio calculation
    """
    history = prepare_catalog(events, mc, t_start, t_end, depth_range=depth_range)
    n = history["n_events"]
    t_end_days = (t_end - t_start).total_seconds() / 86400.0

    if n < 10:
        return ETASFitResult(
            params=ETASParams(Mc=mc), log_likelihood=0.0, n_events=n,
            n_params=0, aic=0.0, converged=False,
            notes=[f"Insufficient events (n={n})"],
        )

    # Naive background rate (events per year)
    naive_rate = n / max(t_end_days / 365.25, 1e-6)

    # Multi-start initialisation
    if variant == "D_exponential":
        bounds = [
            PARAM_BOUNDS["mu_total_per_year"],
            PARAM_BOUNDS["K"],
            PARAM_BOUNDS["alpha"],
            TAU_BOUNDS,
            PARAM_BOUNDS["sigma_km"],
            PARAM_BOUNDS["gamma"],
            PARAM_BOUNDS["q"],
        ]
        starts = [
            [naive_rate, 0.02, 0.8, 0.01, 10.0, 0.5, 1.0],
            [naive_rate, 1e-8, 0.0, 0.01, 10.0, 0.5, 1.0],   # background-only
            [naive_rate, 0.05, 1.0, 0.005, 20.0, 0.4, 1.5],
            [naive_rate * 0.5, 0.01, 0.5, 0.05, 5.0, 0.6, 0.8],
        ]
        n_params = 7
    elif variant == "C_depth_spatial":
        bounds = [
            PARAM_BOUNDS["mu_total_per_year"],
            PARAM_BOUNDS["K"],
            PARAM_BOUNDS["alpha"],
            PARAM_BOUNDS["c_days"],
            PARAM_BOUNDS["p"],
            PARAM_BOUNDS["sigma_km"],
            PARAM_BOUNDS["gamma"],
            PARAM_BOUNDS["q"],
            (0.0, 2.0),   # kappa_depth
        ]
        starts = [
            [naive_rate, 0.02, 0.8, 0.05, 1.1, 10.0, 0.5, 1.0, 0.0],
            [naive_rate, 1e-8, 0.0, 0.05, 1.1, 10.0, 0.5, 1.0, 0.0],
            [naive_rate, 0.05, 1.0, 0.01, 1.2, 20.0, 0.4, 1.5, 0.5],
            [naive_rate * 0.5, 0.01, 0.5, 0.1, 1.15, 5.0, 0.6, 0.8, 0.3],
        ]
        n_params = 9
    else:  # A_baseline, B_depth_stratified, D_omori_extended
        bounds = [
            PARAM_BOUNDS["mu_total_per_year"],
            PARAM_BOUNDS["K"],
            PARAM_BOUNDS["alpha"],
            PARAM_BOUNDS["c_days"],
            PARAM_BOUNDS["p"],
            PARAM_BOUNDS["sigma_km"],
            PARAM_BOUNDS["gamma"],
            PARAM_BOUNDS["q"],
        ]
        starts = [
            [naive_rate, 0.02, 0.8, 0.05, 1.1, 10.0, 0.5, 1.0],
            [naive_rate, 1e-8, 0.0, 0.05, 1.1, 10.0, 0.5, 1.0],
            [naive_rate, 0.05, 1.0, 0.01, 1.2, 20.0, 0.4, 1.5],
            [naive_rate * 0.5, 0.01, 0.5, 0.1, 1.15, 5.0, 0.6, 0.8],
            [naive_rate, 0.1, 1.5, 0.005, 1.05, 15.0, 0.3, 1.2],
        ]
        n_params = 8

    best = None
    best_nll = float("inf")
    for x0 in starts:
        try:
            res = minimize(
                _neg_log_likelihood, x0=np.array(x0),
                args=(history, mc, variant, t_end_days),
                method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 200, "ftol": 1e-6},
            )
            if res.fun < best_nll and math.isfinite(res.fun):
                best_nll = res.fun
                best = res.x
        except Exception as e:
            logger.debug("Start %s failed: %s", x0, e)
            continue

    if best is None:
        # Fallback: background-only
        return ETASFitResult(
            params=ETASParams(mu_total_per_year=naive_rate, Mc=mc),
            log_likelihood=-naive_rate * t_end_days / 365.25 + n * math.log(
                naive_rate / ((BBOX[1]-BBOX[0])*(BBOX[3]-BBOX[2]) * 365.25)
            ),
            n_events=n, n_params=1,
            aic=2*1 - 2*(-naive_rate * t_end_days / 365.25),
            converged=False,
            notes=["MLE failed; background-only fallback"],
        )

    # Build params
    if variant == "D_exponential":
        params = ETASParams(
            mu_total_per_year=best[0], K=best[1], alpha=best[2],
            tau_days=best[3], sigma_km=best[4], gamma=best[5], q=best[6],
            Mc=mc, temporal_kernel="exponential",
        )
    elif variant == "C_depth_spatial":
        params = ETASParams(
            mu_total_per_year=best[0], K=best[1], alpha=best[2],
            c_days=best[3], p=best[4], sigma_km=best[5], gamma=best[6],
            q=best[7], Mc=mc, kappa_depth=best[8], temporal_kernel="omori",
        )
    else:
        params = ETASParams(
            mu_total_per_year=best[0], K=best[1], alpha=best[2],
            c_days=best[3], p=best[4], sigma_km=best[5], gamma=best[6],
            q=best[7], Mc=mc, temporal_kernel="omori",
        )

    # Diagnostics
    K = params.K
    alpha = params.alpha
    beta = b_value * math.log(10.0)
    if alpha > 0 and beta > alpha:
        # Analytic branching ratio: n = K · β/(β-α)  (base-10 productivity,
        # GR expectation of 10^{α(M-Mc)} = β/(β-α) when α<β).
        br_analytic = K * beta / (beta - alpha)
    else:
        br_analytic = float("inf") if alpha >= beta > 0 else 0.0
    br_empirical = float(np.mean(params.K * np.power(10.0, params.alpha * (history["mags"] - mc)))) if n > 0 else 0.0
    # Triggering distance: σ·exp(γ·(M*−Mc)) for a typical M*=5.0 event
    M_star = 5.0
    trig_dist = params.sigma_km * math.exp(params.gamma * (M_star - mc))
    # Temporal decay scale: c/(p-1) for Omori; τ_0 for exponential
    if params.temporal_kernel == "exponential":
        tau_decay = params.tau_days
    else:
        tau_decay = params.c_days / max(params.p - 1.0, 0.01)

    log_l = -best_nll
    aic = 2 * n_params - 2 * log_l

    notes = []
    if params.K <= 1e-6:
        notes.append("K≈0: no triggering detected by MLE")
    if params.alpha <= 1e-4:
        notes.append("α≈0: no magnitude scaling")
    if params.c_days >= 0.99:
        notes.append("c at upper bound: Omori kernel cannot capture short-lag peak")
    if params.K <= 1e-6 and params.alpha <= 1e-4:
        notes.append("Model reduces to background Poisson")

    return ETASFitResult(
        params=params, log_likelihood=log_l, n_events=n,
        n_params=n_params, aic=aic, converged=True, notes=notes,
        branching_ratio_analytic=br_analytic,
        branching_ratio_empirical=br_empirical,
        triggering_distance_km=trig_dist,
        temporal_decay_scale_days=tau_decay,
    )


# ---------------------------------------------------------------------------
# Forecast generation
# ---------------------------------------------------------------------------

@dataclass
class ETASForecast:
    """Per-cell forecast from an ETAS variant."""
    threshold: float
    horizon: str
    horizon_years: float
    regional_probability: float
    regional_expected_count: float
    cell_probs: np.ndarray   # shape (N_CELLS,)
    cell_expected_counts: np.ndarray


def forecast_etas(
    fit: ETASFitResult,
    events: list,
    t0: datetime,
    t_min: datetime,
    threshold: float,
    horizon: str,
    horizon_years: float,
    cell_size_deg: float = 1.0,
) -> ETASForecast:
    """Generate a forecast from a fitted ETAS model.

    Computes the expected number of events ≥ threshold in [t0, t0+Δt) per
    cell, then P(N≥1) = 1 - exp(-E[N]).

    The ETAS expected count integrates the conditional intensity over the
    forecast window. For the background component: E_bg = μ·Δt (regional).
    For the triggered component: Σ_i K·10^{α(M_i-Mc)} · G(t0+Δt - t_i)
    where G is the Omori/exponential CDF. The triggered contribution is
    distributed across cells via the spatial kernel centred on each ancestor.
    """
    params = fit.params
    hy = horizon_years
    horizon_days = hy * 365.25

    # Cell centres
    qlats = np.array([BBOX[0] + (i + 0.5) * cell_size_deg for i in range(N_LAT)])
    qlons = np.array([BBOX[2] + (j + 0.5) * cell_size_deg for j in range(N_LON)])

    # Background expected count per cell (uniform): μ·Δt / N_CELLS,
    # then scaled by magnitude threshold (GR).
    # E[N≥threshold] = E[N≥Mc] · 10^{-b·(threshold-Mc)}
    mag_scale = math.pow(10.0, -V1_B * (threshold - params.Mc))
    bg_total = params.mu_total_per_year * hy * mag_scale
    bg_per_cell = bg_total / N_CELLS

    cell_expected = np.full(N_CELLS, bg_per_cell)

    # Triggered contribution from ancestors before t0
    history = [e for e in events if e.origin_time_utc < t0]
    above = []
    for e in history:
        m = e.mw if e.mw is not None else e.original_magnitude
        if m is not None and m >= params.Mc and e.depth_km is not None:
            above.append((
                (e.origin_time_utc - t_min).total_seconds() / 86400.0,
                e.latitude, e.longitude, e.depth_km, m
            ))
    above.sort(key=lambda x: x[0])

    t0_days = (t0 - t_min).total_seconds() / 86400.0
    t_end_days = t0_days + horizon_days

    if params.K > 1e-8 and params.alpha > 1e-4 and above:
        # For each ancestor, compute G(t0+Δt - t_i) and distribute across cells
        for (t_i, lat_i, lon_i, depth_i, mag_i) in above:
            delta = t_end_days - t_i
            if delta <= 0:
                continue
            if params.temporal_kernel == "exponential":
                G = 1.0 - math.exp(-delta / max(params.tau_days, 1e-12))
                # Only consider recent ancestors (within ~10·τ)
                if t0_days - t_i > 10.0 * params.tau_days:
                    continue
            else:
                G = 1.0 - (params.c_days / (params.c_days + delta)) ** (params.p - 1.0)
                tau_window = max(10.0 * params.c_days / max(params.p - 1.0, 0.01), 30.0)
                if t0_days - t_i > tau_window:
                    continue

            mag_factor = params.K * math.pow(10.0, params.alpha * (mag_i - params.Mc))
            # Total expected events from this ancestor (above Mc):
            expected_from_i = mag_factor * G
            # Scale to threshold
            expected_from_i *= mag_scale

            if params.kappa_depth != 0.0:
                sigma_i = _spatial_sigma_for_depth(depth_i, params.sigma_km,
                                                    params.kappa_depth, params.D_ref_km)
            else:
                sigma_i = params.sigma_km

            # Distribute across cells via spatial kernel
            for idx in range(N_CELLS):
                i_lat = idx // N_LON
                i_lon = idx % N_LON
                r_km = _equirect_km(qlats[i_lat], qlons[i_lon], lat_i, lon_i)
                f = _powerlaw_spatial(np.array([r_km]), sigma_i, params.q)[0]
                cell_expected[idx] += expected_from_i * f * (cell_size_deg ** 2)
                # The f is per km²; multiply by cell area in deg² → need km²
                # Actually f has units 1/km², and we want expected count per cell.
                # Cell area in km²: (1° lat × 1° lon at 24°N) ≈ 110.574 × 101.32 km²
                # So expected count per cell = expected_from_i · f · cell_area_km²

            # The above loop is slow; let's vectorise it below instead.
        # Note: the per-cell loop above is the conceptual form; the actual
        # implementation uses the vectorised version below.

    # Vectorised triggered contribution (replaces the slow loop above)
    cell_expected = np.full(N_CELLS, bg_per_cell)
    if params.K > 1e-8 and params.alpha > 1e-4 and above:
        cell_area_km2 = cell_size_deg * 110.574 * cell_size_deg * 111.320 * math.cos(math.radians(24.0))
        # Build query grid
        qgrid_lat, qgrid_lon = np.meshgrid(qlats, qlons, indexing="ij")
        qflat_lat = qgrid_lat.flatten()
        qflat_lon = qgrid_lon.flatten()

        for (t_i, lat_i, lon_i, depth_i, mag_i) in above:
            delta = t_end_days - t_i
            if delta <= 0:
                continue
            if params.temporal_kernel == "exponential":
                if t0_days - t_i > 10.0 * params.tau_days:
                    continue
                G = 1.0 - math.exp(-delta / max(params.tau_days, 1e-12))
            else:
                tau_window = max(10.0 * params.c_days / max(params.p - 1.0, 0.01), 30.0)
                if t0_days - t_i > tau_window:
                    continue
                G = 1.0 - (params.c_days / (params.c_days + delta)) ** (params.p - 1.0)

            mag_factor = params.K * math.pow(10.0, params.alpha * (mag_i - params.Mc))
            expected_from_i = mag_factor * G * mag_scale

            sigma_i = (_spatial_sigma_for_depth(depth_i, params.sigma_km,
                                                  params.kappa_depth, params.D_ref_km)
                       if params.kappa_depth != 0.0 else params.sigma_km)
            r_km = _equirect_km_vec(qflat_lat, qflat_lon, lat_i, lon_i)
            f = _powerlaw_spatial(r_km, sigma_i, params.q)
            cell_expected += expected_from_i * f * cell_area_km2

    # Cap expected counts to avoid numerical issues
    cell_expected = np.maximum(cell_expected, 0.0)
    cell_probs = 1.0 - np.exp(-cell_expected)
    cell_probs = np.clip(cell_probs, 0.0, 1.0)

    total_expected = float(np.sum(cell_expected))
    regional_p = 1.0 - math.exp(-total_expected)

    return ETASForecast(
        threshold=threshold, horizon=horizon, horizon_years=hy,
        regional_probability=round(regional_p, 6),
        regional_expected_count=round(total_expected, 4),
        cell_probs=cell_probs,
        cell_expected_counts=cell_expected,
    )


# ---------------------------------------------------------------------------
# Omori diagnostic (non-parametric R(Δt))
# ---------------------------------------------------------------------------

def compute_omori_diagnostic(
    events: list, mainshock_threshold: float = 5.0,
    target_threshold: float = V1_MC,
    t_start: Optional[datetime] = None,
    t_end: Optional[datetime] = None,
    max_lag_days: float = 30.0,
    n_bins: int = 20,
) -> dict:
    """Non-parametric Omori diagnostic: R(Δt) = observed rate / background rate.

    For each mainshock (M ≥ mainshock_threshold), count subsequent target
    events (M ≥ target_threshold) in log-spaced time bins. R(Δt) is the
    ratio of observed post-event rate to the overall catalog rate.

    Returns dict with: R_per_bin, bin_centers_days, peak_R, peak_lag_days,
    n_mainshocks, n_targets, omori_like.
    """
    if t_start is None:
        t_start = min(e.origin_time_utc for e in events)
    if t_end is None:
        t_end = max(e.origin_time_utc for e in events)

    mainshocks = [e for e in events
                  if t_start <= e.origin_time_utc <= t_end
                  and (e.mw if e.mw is not None else e.original_magnitude) is not None
                  and (e.mw if e.mw is not None else e.original_magnitude) >= mainshock_threshold]
    targets = [e for e in events
               if t_start <= e.origin_time_utc <= t_end
               and (e.mw if e.mw is not None else e.original_magnitude) is not None
               and (e.mw if e.mw is not None else e.original_magnitude) >= target_threshold]

    if not mainshocks or not targets:
        return {"peak_R": 0.0, "peak_lag_days": 0.0, "n_mainshocks": 0,
                "n_targets": 0, "omori_like": False}

    catalog_span_days = max((t_end - t_start).total_seconds() / 86400.0, 1e-6)
    bg_rate = len(targets) / catalog_span_days  # events per day

    # Log-spaced bins from 1e-3 day (≈86s) to max_lag_days
    log_min = -3.0   # 10^-3 day
    log_max = math.log10(max_lag_days)
    bin_edges = np.logspace(log_min, log_max, n_bins + 1)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # geometric mean

    counts = np.zeros(n_bins)
    exposure = np.zeros(n_bins)  # total exposure time per bin

    for ms in mainshocks:
        ms_t = ms.origin_time_utc
        for tgt in targets:
            lag_days = (tgt.origin_time_utc - ms_t).total_seconds() / 86400.0
            if lag_days <= 0:
                continue
            # Censor at catalog end
            time_to_end = (t_end - ms_t).total_seconds() / 86400.0
            if lag_days > time_to_end:
                continue
            # Find bin
            for i in range(n_bins):
                if bin_edges[i] <= lag_days < bin_edges[i + 1]:
                    counts[i] += 1
                    break
        # Exposure: each mainshock contributes its survival time per bin
        time_to_end = (t_end - ms_t).total_seconds() / 86400.0
        for i in range(n_bins):
            bin_lo = bin_edges[i]
            bin_hi = bin_edges[i + 1]
            # Exposure in this bin = min(bin_hi, time_to_end) - bin_lo, if positive
            if time_to_end > bin_lo:
                exposure[i] += min(bin_hi, time_to_end) - bin_lo

    # Rate per bin
    observed_rate = np.where(exposure > 0, counts / exposure, 0.0)
    R = observed_rate / max(bg_rate, 1e-12)

    # Peak
    peak_idx = int(np.argmax(R))
    peak_R = float(R[peak_idx])
    peak_lag = float(bin_centers[peak_idx])

    # Omori-like: R > 2 in any bin with center < 1 day
    omori_like = bool(np.any((R > 2.0) & (bin_centers < 1.0)))

    return {
        "R_per_bin": R.tolist(),
        "bin_centers_days": bin_centers.tolist(),
        "bin_edges_days": bin_edges.tolist(),
        "peak_R": round(peak_R, 4),
        "peak_lag_days": round(peak_lag, 6),
        "n_mainshocks": len(mainshocks),
        "n_targets": len(targets),
        "bg_rate_per_day": round(bg_rate, 6),
        "counts_per_bin": counts.astype(int).tolist(),
        "exposure_days_per_bin": [round(float(x), 4) for x in exposure],
        "omori_like": omori_like,
    }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 7) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() > 0:
            e += abs(float(probs[mask].mean()) - float(y_true[mask].mean())) * mask.sum() / len(probs)
    return e


def evaluate_forecast(probs: np.ndarray, y_true: np.ndarray) -> dict:
    eps = 1e-12
    brier = float(np.mean((probs - y_true) ** 2))
    f = np.clip(probs, eps, 1 - eps)
    log_lik = float(np.mean(y_true * np.log(f) + (1 - y_true) * np.log(1 - f)))
    ece = _ece(probs, y_true)
    sharpness = float(np.std(probs))
    return {
        "brier": round(brier, 6),
        "log_lik": round(log_lik, 6),
        "ece": round(ece, 6),
        "sharpness": round(sharpness, 6),
        "n_positive": int(y_true.sum()),
        "n_cells": len(y_true),
    }


def block_bootstrap_delta(
    v4_probs_per_origin: list, baseline_probs_per_origin: list,
    y_true_per_origin: list, n_bootstrap: int = 500, seed: int = 42,
) -> dict:
    """Block bootstrap over forecast ORIGINS for ΔBrier and Δlog-lik.

    ΔBrier = Brier_baseline - Brier_v4  (positive = v4 better)
    """
    rng = np.random.default_rng(seed)
    n = len(v4_probs_per_origin)
    if n == 0:
        return {"delta_brier_mean": 0.0, "delta_brier_ci": [0.0, 0.0],
                "delta_log_lik_mean": 0.0, "delta_log_lik_ci": [0.0, 0.0],
                "n_bootstrap": n_bootstrap, "n_origins": 0}
    eps = 1e-12
    deltas_brier = []
    deltas_ll = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        v4_all = np.concatenate([v4_probs_per_origin[i] for i in idx])
        b_all = np.concatenate([baseline_probs_per_origin[i] for i in idx])
        yt_all = np.concatenate([y_true_per_origin[i] for i in idx])
        b_v4 = np.mean((v4_all - yt_all) ** 2)
        b_b = np.mean((b_all - yt_all) ** 2)
        deltas_brier.append(b_b - b_v4)
        f_v4 = np.clip(v4_all, eps, 1 - eps)
        f_b = np.clip(b_all, eps, 1 - eps)
        ll_v4 = np.mean(yt_all * np.log(f_v4) + (1 - yt_all) * np.log(1 - f_v4))
        ll_b = np.mean(yt_all * np.log(f_b) + (1 - yt_all) * np.log(1 - f_b))
        deltas_ll.append(ll_v4 - ll_b)
    return {
        "delta_brier_mean": round(float(np.mean(deltas_brier)), 6),
        "delta_brier_ci": [round(float(np.percentile(deltas_brier, 2.5)), 6),
                           round(float(np.percentile(deltas_brier, 97.5)), 6)],
        "delta_log_lik_mean": round(float(np.mean(deltas_ll)), 6),
        "delta_log_lik_ci": [round(float(np.percentile(deltas_ll, 2.5)), 6),
                             round(float(np.percentile(deltas_ll, 97.5)), 6)],
        "n_bootstrap": n_bootstrap,
        "n_origins": n,
    }


def permutation_test_delta(
    v4_probs_per_origin: list, baseline_probs_per_origin: list,
    y_true_per_origin: list, n_permutations: int = 1000, seed: int = 42,
) -> dict:
    """Permutation test for ΔBrier under the null of no difference."""
    rng = np.random.default_rng(seed)
    n = len(v4_probs_per_origin)
    if n == 0:
        return {"p_value": 1.0, "n_permutations": n_permutations, "n_origins": 0,
                "observed_delta_brier": 0.0}
    def _delta(v4_list, b_list, y_list):
        v4_all = np.concatenate(v4_list)
        b_all = np.concatenate(b_list)
        yt_all = np.concatenate(y_list)
        return np.mean((b_all - yt_all)**2) - np.mean((v4_all - yt_all)**2)
    observed = _delta(v4_probs_per_origin, baseline_probs_per_origin, y_true_per_origin)
    perm_deltas = np.zeros(n_permutations)
    for p in range(n_permutations):
        swap = rng.random(n) < 0.5
        v4_perm = [baseline_probs_per_origin[i] if swap[i] else v4_probs_per_origin[i] for i in range(n)]
        b_perm = [v4_probs_per_origin[i] if swap[i] else baseline_probs_per_origin[i] for i in range(n)]
        perm_deltas[p] = _delta(v4_perm, b_perm, y_true_per_origin)
    p_value = float(np.mean(np.abs(perm_deltas) >= abs(observed)))
    return {
        "p_value": round(p_value, 4),
        "n_permutations": n_permutations,
        "n_origins": n,
        "observed_delta_brier": round(float(observed), 6),
    }


# ---------------------------------------------------------------------------
# Posterior predictive check
# ---------------------------------------------------------------------------

def posterior_predictive_check(
    fit: ETASFitResult, events: list, t_start: datetime, t_end: datetime,
    mc: float, n_sims: int = 200, seed: int = 42,
) -> dict:
    """Simulate catalogs from the fitted ETAS model and compare statistics."""
    rng = np.random.default_rng(seed)
    history = prepare_catalog(events, mc, t_start, t_end)
    n_obs = history["n_events"]
    if n_obs == 0:
        return {"observed_total": 0, "sim_total_ci": [0, 0], "pass": True}

    t_span_days = (t_end - t_start).total_seconds() / 86400.0
    obs_depths = history["depths"]
    obs_mags = history["mags"]

    # Observed statistics
    obs_total = n_obs
    obs_mean_depth = float(np.mean(obs_depths)) if n_obs > 0 else 0.0
    obs_median_iet_days = 0.0
    if n_obs > 1:
        iets = np.diff(history["times_days"])
        iets = iets[iets > 0]
        obs_median_iet_days = float(np.median(iets)) if len(iets) > 0 else 0.0

    # Simulate: thinning is expensive; use a simpler approach.
    # For a Poisson background (K≈0 case), simulate n ~ Poisson(μ·T) with
    # uniform spatial and GR magnitude distribution.
    # For K>0, add a triggering cascade via branching.
    sim_totals = []
    sim_mean_depths = []
    sim_median_iets = []
    for s in range(n_sims):
        if fit.params.K <= 1e-6:
            # Pure Poisson
            mu_total = fit.params.mu_total_per_year * (t_span_days / 365.25)
            n_sim = rng.poisson(max(mu_total, 0.0))
        else:
            # Branching: background ~ Poisson(μ·T), each event triggers
            # n_trig ~ mean(K·10^{α(M-Mc})) offspring. Approximate.
            mu_total = fit.params.mu_total_per_year * (t_span_days / 365.25)
            n_bg = rng.poisson(max(mu_total, 0.0))
            # Approximate total = n_bg / (1 - branching_ratio) if br < 1
            br = min(fit.branching_ratio_analytic, 0.95) if fit.branching_ratio_analytic < 1 else 0.95
            n_sim = int(n_bg / max(1.0 - br, 0.05))
        sim_totals.append(n_sim)
        # Depth: sample from observed depth distribution
        if n_sim > 0 and n_obs > 0:
            sim_depths = rng.choice(obs_depths, size=n_sim, replace=True)
            sim_mean_depths.append(float(np.mean(sim_depths)))
            # IET: exponential with rate = n_sim / T
            sim_times = np.sort(rng.uniform(0, t_span_days, size=n_sim))
            if n_sim > 1:
                sim_iets = np.diff(sim_times)
                sim_iets = sim_iets[sim_iets > 0]
                sim_median_iets.append(float(np.median(sim_iets)) if len(sim_iets) > 0 else 0.0)
            else:
                sim_median_iets.append(0.0)
        else:
            sim_mean_depths.append(0.0)
            sim_median_iets.append(0.0)

    sim_totals = np.array(sim_totals)
    sim_mean_depths = np.array(sim_mean_depths)
    sim_median_iets = np.array(sim_median_iets)

    total_ci = [int(np.percentile(sim_totals, 2.5)), int(np.percentile(sim_totals, 97.5))]
    depth_ci = [float(np.percentile(sim_mean_depths, 2.5)),
                float(np.percentile(sim_mean_depths, 97.5))]
    iet_ci = [float(np.percentile(sim_median_iets, 2.5)),
              float(np.percentile(sim_median_iets, 97.5))]

    return {
        "observed_total": int(obs_total),
        "sim_total_mean": round(float(np.mean(sim_totals)), 1),
        "sim_total_ci": total_ci,
        "observed_mean_depth": round(obs_mean_depth, 2),
        "sim_mean_depth_mean": round(float(np.mean(sim_mean_depths)), 2),
        "sim_mean_depth_ci": [round(depth_ci[0], 2), round(depth_ci[1], 2)],
        "observed_median_iet_days": round(obs_median_iet_days, 4),
        "sim_median_iet_days_mean": round(float(np.mean(sim_median_iets)), 4),
        "sim_median_iet_days_ci": [round(iet_ci[0], 4), round(iet_ci[1], 4)],
        "total_pass": bool(total_ci[0] <= obs_total <= total_ci[1]),
        "depth_pass": bool(depth_ci[0] <= obs_mean_depth <= depth_ci[1]),
        "iet_pass": bool(iet_ci[0] <= obs_median_iet_days <= iet_ci[1]),
    }


# ---------------------------------------------------------------------------
# Benjamini-Hochberg multiple comparison correction
# ---------------------------------------------------------------------------

def benjamini_hochberg(p_values: list, alpha: float = 0.05) -> dict:
    """Apply Benjamini-Hochberg FDR correction.

    Returns dict with: rejected (list of bool), critical_values, n_rejected.
    """
    n = len(p_values)
    if n == 0:
        return {"rejected": [], "critical_values": [], "n_rejected": 0}
    # Sort p-values
    order = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]
    critical = np.array([(i + 1) / n * alpha for i in range(n)])
    # Find the largest k where sorted_p[k] <= critical[k]
    rejected_sorted = sorted_p <= critical
    # Once we stop rejecting, all subsequent are not rejected
    if rejected_sorted.any():
        last_reject = np.where(rejected_sorted)[0][-1]
        rejected_sorted[:last_reject + 1] = True
        rejected_sorted[last_reject + 1:] = False
    # Unsort
    rejected = np.zeros(n, dtype=bool)
    rejected[order] = rejected_sorted
    return {
        "rejected": rejected.tolist(),
        "critical_values": critical.tolist(),
        "n_rejected": int(rejected.sum()),
        "alpha": alpha,
    }
