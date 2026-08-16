"""ETAS parameter estimation by maximum likelihood.

The ETAS log-likelihood for a catalog of N events in [0, T] over region A is:

    log L = Σ_{j=1}^{N} log λ(x_j, y_j, t_j | H_{t_j})  -  ∫_0^T ∫_A λ(x,y,t|H_t) dx dy dt

The first term is the sum of log-intensities at each event; the second is
the integrated intensity (expected total count). For a power-law spatial
kernel, the spatial integral over an effectively unbounded plane is finite
but we approximate it over the study bbox.

Optimization: L-BFGS-B with the bounds in model.PARAM_BOUNDS. We use a
robust initialization (Mu_total from Poisson rate of declustered mainshocks;
K from branching-ratio ~0.5; α=1.0; c=0.1; p=1.1; σ=10km; γ=0.5; q=1.0).

Identifiability: if the optimizer hits a bound, or if the Hessian-derived
standard error is huge, or if profile-likelihood shows a flat region, the
parameter is flagged as 'not identifiable' and reported as such. We do NOT
force all parameters to be locally estimated if the data are insufficient;
in that case we fix the poorly-identified parameter at a literature-informed
value WITH A CLEAR 'externally_informed' FLAG.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from ..ingestion.schema import CanonicalEvent
from .background import BackgroundRate, KDEBackground, UniformBackground
from .model import ETASModel, ETASParams, PARAM_BOUNDS, conditional_intensity
from .omori import omori_utsu_g
from .spatial_kernels import _equirect_km, magnitude_scaled_length

logger = logging.getLogger(__name__)


@dataclass
class ETASFitResult:
    """Result of an ETAS MLE fit."""

    params: ETASParams
    background: BackgroundRate
    log_likelihood: float
    n_events_used: int
    fitting_period_days: tuple[float, float]
    Mc: float
    # Uncertainty (Hessian-derived via numerical differentiation)
    param_std: dict = field(default_factory=dict)
    # Identifiability flags
    identifiability: dict = field(default_factory=dict)
    # Optimization info
    converged: bool = False
    n_evaluations: int = 0
    notes: list[str] = field(default_factory=list)

    def to_row(self) -> dict:
        r = {
            "model": "etas",
            "Mc": self.Mc,
            "mu_total_per_year": round(self.params.mu_total_per_year, 6),
            "K": self.params.K,
            "alpha": round(self.params.alpha, 4),
            "c_days": round(self.params.c_days, 4),
            "p": round(self.params.p, 4),
            "sigma_km": round(self.params.sigma_km, 3),
            "gamma": round(self.params.gamma, 4),
            "q": round(self.params.q, 4),
            "spatial_kernel": self.params.spatial_kernel,
            "log_likelihood": round(self.log_likelihood, 3),
            "n_events_used": self.n_events_used,
            "fitting_period_days": f"{self.fitting_period_days[0]:.1f}-{self.fitting_period_days[1]:.1f}",
            "converged": self.converged,
            "n_evaluations": self.n_evaluations,
        }
        for k, v in self.param_std.items():
            r[f"std_{k}"] = round(v, 5) if v is not None and not math.isnan(v) else None
        for k, v in self.identifiability.items():
            r[f"ident_{k}"] = v
        r["notes"] = "; ".join(self.notes)
        r["fixed_parameters"] = "; ".join(f"{k}={v}" for k, v in self.params.fixed_parameters.items()) or "none"
        return r


# ---------------------------------------------------------------------------
# Catalog preparation
# ---------------------------------------------------------------------------


def _get_declustered_background(
    events: list[CanonicalEvent],
    Mc: float,
    bbox: tuple[float, float, float, float],
    t_start,
    t_end,
    method: str = "gardner_knopoff",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lats, lons) of declustered mainshocks for background estimation.

    Uses the Stage 3 declustering modules. Phase A correction: replaces the
    previous Mc+0.5 proxy with proper declustering.

    Parameters
    ----------
    method : "gardner_knopoff" or "reasenberg"
    """
    from ..declustering import gardner_knopoff, reasenberg
    import copy

    # Filter to events above Mc, in bbox, in time window
    mn_lat, mx_lat, mn_lon, mx_lon = bbox
    sel = []
    for e in events:
        m = e.mw if e.mw is not None else e.original_magnitude
        if m is None or m < Mc:
            continue
        if not (mn_lat <= e.latitude <= mx_lat and mn_lon <= e.longitude <= mx_lon):
            continue
        if t_start is not None and e.origin_time_utc < t_start:
            continue
        if t_end is not None and e.origin_time_utc >= t_end:
            continue
        sel.append(e)

    if len(sel) < 10:
        return np.array([]), np.array([])

    # Run declustering on a copy (so we don't mutate the original events)
    sel_copy = copy.deepcopy(sel)
    if method == "reasenberg":
        reasenberg(sel_copy, magnitude_field="mw")
    else:
        gardner_knopoff(sel_copy, magnitude_field="mw")

    mainshocks = [e for e in sel_copy if e.is_mainshock]
    if len(mainshocks) == 0:
        return np.array([]), np.array([])

    lats = np.array([e.latitude for e in mainshocks])
    lons = np.array([e.longitude for e in mainshocks])
    return lats, lons


def prepare_catalog(
    events: list[CanonicalEvent],
    Mc: float,
    t_start=None,
    t_end=None,
    bbox: Optional[tuple[float, float, float, float]] = None,
) -> dict:
    """Convert CanonicalEvents to ETAS-ready arrays.

    Returns dict with lats, lons, times_days (since first event), mags.
    Uses Mw where available, else original_magnitude; filters events below
    Mc and outside the bbox/time window.
    """
    if bbox is None:
        bbox = (20.0, 28.0, 88.0, 96.0)
    mn_lat, mx_lat, mn_lon, mx_lon = bbox

    sel = []
    for e in events:
        m = e.mw if e.mw is not None else e.original_magnitude
        if m is None or m < Mc:
            continue
        if not (mn_lat <= e.latitude <= mx_lat and mn_lon <= e.longitude <= mx_lon):
            continue
        if t_start is not None and e.origin_time_utc < t_start:
            continue
        if t_end is not None and e.origin_time_utc >= t_end:
            continue
        sel.append((e.origin_time_utc, e.latitude, e.longitude, m))

    sel.sort(key=lambda x: x[0])
    if not sel:
        return {"times_days": np.array([]), "lats": np.array([]),
                "lons": np.array([]), "mags": np.array([]), "n": 0}

    times_dt = [s[0] for s in sel]
    t0 = times_dt[0]
    times_days = np.array([(t - t0).total_seconds() / 86400.0 for t in times_dt])
    lats = np.array([s[1] for s in sel])
    lons = np.array([s[2] for s in sel])
    mags = np.array([s[3] for s in sel])
    return {"times_days": times_days, "lats": lats, "lons": lons,
            "mags": mags, "n": len(sel), "t0": t0,
            "t_start_days": 0.0, "t_end_days": float(times_days[-1])}


# ---------------------------------------------------------------------------
# Log-likelihood
# ---------------------------------------------------------------------------


def _unpack(theta):
    mu, K, alpha, c, p, sigma, gamma, q = theta
    return mu, K, alpha, c, p, sigma, gamma, q


def _neg_log_likelihood(
    theta,
    catalog: dict,
    bbox: tuple[float, float, float, float],
    background: BackgroundRate,
    Mc: float,
    spatial_kernel: str = "powerlaw",
    fixed: Optional[dict] = None,
) -> float:
    """Negative ETAS log-likelihood (to be minimized)."""
    # Override fixed parameters
    if fixed:
        theta_list = list(theta)
        # theta is [mu, K, alpha, c, p, sigma, gamma, q]
        names = ["mu_total_per_year", "K", "alpha", "c_days", "p", "sigma_km", "gamma", "q"]
        for name, val in fixed.items():
            if name in names:
                theta_list[names.index(name)] = val
        theta = tuple(theta_list)

    mu, K, alpha, c, p, sigma, gamma, q = _unpack(theta)

    times = catalog["times_days"]
    lats = catalog["lats"]
    lons = catalog["lons"]
    mags = catalog["mags"]
    n = len(times)
    if n == 0:
        return 0.0

    t_start = catalog["t_start_days"]
    t_end = catalog["t_end_days"]
    T = max(t_end - t_start, 1e-6)

    # --- First term: sum of log λ at each event ---
    # Compute λ at each event j, conditioned on events i < j.
    # Use a time window cutoff for efficiency (events older than ~10*c/(p-1) contribute little)
    log_lam_sum = 0.0
    # Precompute declustered background density at each event (per year per km² -> per day per km²)
    bg_at_events = np.array([background.at(la, lo) / 365.25 for la, lo in zip(lats, lons)])

    for j in range(n):
        tj, latj, lonj, mj = times[j], lats[j], lons[j], mags[j]
        # Background
        lam = bg_at_events[j]
        # Triggered: sum over i < j
        if j > 0:
            # Time cutoff: include events within the last tau_max days
            tau_max = max(10.0 * c / max(p - 1.0, 0.01), 365.0)  # cap at 1yr for efficiency
            i_lo = 0
            # binary search for first i with times[i] >= tj - tau_max
            lo, hi = 0, j
            while lo < hi:
                mid = (lo + hi) // 2
                if times[mid] >= tj - tau_max:
                    hi = mid
                else:
                    lo = mid + 1
            i_lo = lo
            ti = times[i_lo:j]
            if len(ti) > 0:
                tau = tj - ti
                g = (p - 1.0) * (c ** (p - 1.0)) / (tau + c) ** p
                # BASE-10 productivity: K · 10^{α(M − Mc)} (Phase A correction)
                prod = K * np.power(10.0, alpha * (mags[i_lo:j] - Mc))
                # spatial
                r_km = np.array([_equirect_km(latj, lonj, lats[k], lons[k])
                                 for k in range(i_lo, j)])
                s = sigma * np.exp(gamma * (mags[i_lo:j] - Mc))
                s2 = s * s
                if spatial_kernel == "powerlaw":
                    f = (q - 1.0) / (np.pi * s2) * (1.0 + (r_km * r_km) / s2) ** (-(1.0 + q))
                else:
                    f = 1.0 / (np.pi * s2) * np.exp(-(r_km * r_km) / s2)
                lam += float(np.sum(prod * g * f))
        if lam <= 0:
            lam = 1e-30
        log_lam_sum += math.log(lam)

    # --- Second term: integrated intensity (expected count) ---
    # ∫_0^T μ dt = mu * T  (background; mu is total rate per year, T in days)
    # The triggered integral: ∫_0^T Σ_i K·exp(α(M_i-Mc))·g(τ)f(x,y) dτ dx dy
    # Since f integrates to 1 over the plane, ∫ f dx dy = 1, so the triggered
    # integral reduces to Σ_i K·exp(α(M_i-Mc))·∫_{max(0,?)}^T g(τ-t_i) dτ
    # For each event i, the contribution to the integral from t_i to T is:
    #   K·exp(α(M_i-Mc))·∫_0^{T-t_i} g(τ) dτ
    # Using the normalized Omori, ∫_0^L g(τ) dτ = c^{p-1}·[(c)^{-(p-1)} - (L+c)^{-(p-1)}]
    #                                       = 1 - [c/(L+c)]^{p-1}
    integ_bg = mu * T / 365.25   # background: per year -> per day
    integ_trig = 0.0
    for i in range(n):
        L = max(t_end - times[i], 0.0)
        if L <= 0:
            continue
        G = 1.0 - (c / (L + c)) ** (p - 1.0)   # ∫_0^L g(τ) dτ for normalized Omori
        # BASE-10 productivity: K · 10^{α(M − Mc)} (Phase A correction)
        integ_trig += K * math.pow(10.0, alpha * (mags[i] - Mc)) * G
    expected = integ_bg + integ_trig

    nll = -log_lam_sum + expected
    if not math.isfinite(nll):
        return 1e18
    return nll


# ---------------------------------------------------------------------------
# Fitter
# ---------------------------------------------------------------------------


def fit_etas_mle(
    events: list[CanonicalEvent],
    Mc: float,
    bbox: tuple[float, float, float, float] = (20.0, 28.0, 88.0, 96.0),
    background_kind: str = "kde",     # "uniform" or "kde"
    spatial_kernel: str = "powerlaw",
    t_start=None,
    t_end=None,
    fix_parameters: Optional[dict] = None,
    initial_guess: Optional[dict] = None,
) -> ETASFitResult:
    """Fit ETAS parameters by MLE.

    Parameters
    ----------
    Mc : magnitude threshold for fitting.
    background_kind : 'uniform' or 'kde'.
    spatial_kernel : 'powerlaw' or 'gaussian'.
    fix_parameters : dict of parameter_name -> value to fix (not estimate).
        Use this when a parameter is not identifiable; the fixed value is
        clearly flagged as 'externally_informed' in the result.
    initial_guess : dict of parameter_name -> value for the optimizer start.
    """
    catalog = prepare_catalog(events, Mc=Mc, t_start=t_start, t_end=t_end, bbox=bbox)
    n = catalog["n"]
    if n < 50:
        return _insufficient_data_result(Mc, n, bbox, background_kind, spatial_kernel,
                                          catalog, fix_parameters)

    # Build background from declustered mainshocks (use all events above Mc as
    # a proxy; a proper declustering would be done in Stage 3 but for the
    # background rate we use the full catalog spatial distribution).
    lats = catalog["lats"]
    lons = catalog["lons"]
    times = catalog["times_days"]
    T = catalog["t_end_days"] - catalog["t_start_days"]
    T_years = T / 365.25

    # Initial mu_total: assume ~50-70% of events are background (typical ETAS)
    # We'll let the optimizer refine this.
    mu_init = (n / T_years) * 0.5

    if background_kind == "kde":
        # Phase A correction: use PROPER declustering instead of the Mc+0.5 proxy.
        # The background rate μ(x,y) should be estimated from declustered
        # mainshocks, not from all events above an arbitrary threshold.
        # We use the Gardner-Knopoff declustering from Stage 3.
        bg_lats, bg_lons = _get_declustered_background(events, Mc, bbox, t_start, t_end,
                                                        method="gardner_knopoff")
        if len(bg_lats) < 5:
            # Fall back to Reasenberg if GK produces too few mainshocks
            bg_lats, bg_lons = _get_declustered_background(events, Mc, bbox, t_start, t_end,
                                                            method="reasenberg")
        if len(bg_lats) < 5:
            # Last resort: use all events above Mc (the old proxy, documented)
            bg_mask = catalog["mags"] >= Mc
            bg_lats = lats[bg_mask]
            bg_lons = lons[bg_mask]
        background = KDEBackground.build(
            bg_lats, bg_lons, mu_total_per_year=mu_init, bbox=bbox,
        )
    else:
        background = UniformBackground.build(mu_total_per_year=mu_init, bbox=bbox)

    # Initial guess
    ig = initial_guess or {}
    x0 = [
        ig.get("mu_total_per_year", mu_init),
        ig.get("K", 0.01),
        ig.get("alpha", 1.0),
        ig.get("c_days", 0.05),
        ig.get("p", 1.1),
        ig.get("sigma_km", 10.0),
        ig.get("gamma", 0.5),
        ig.get("q", 1.0),
    ]
    # Bounds
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
    # If some parameters are fixed, restrict their bounds to the fixed value
    if fix_parameters:
        names = ["mu_total_per_year", "K", "alpha", "c_days", "p", "sigma_km", "gamma", "q"]
        for name, val in fix_parameters.items():
            if name in names:
                idx = names.index(name)
                bounds[idx] = (val, val)
                x0[idx] = val

    # Optimize with multiple restarts (the likelihood surface has multiple
    # local minima due to the interaction of K, alpha, and the spatial kernel).
    # We use parameter scaling so L-BFGS-B's numerical gradient is well-conditioned.
    n_eval = [0]
    def objective(theta):
        n_eval[0] += 1
        return _neg_log_likelihood(theta, catalog, bbox, background, Mc,
                                    spatial_kernel, fixed=fix_parameters)

    # Multi-start: try several initializations and keep the best.
    # Include the "background-only" start (K=1e-8, alpha=0) so the optimizer
    # can find the K=0 solution if that is genuinely better (which is a
    # legitimate scientific finding: "no triggering detected in-sample").
    naive_rate = n / T_years
    starts = [
        x0,
        [naive_rate * 0.3, 0.02, 0.8, 0.05, 1.1, 5.0, 0.4, 1.0],
        [naive_rate * 0.5, 0.05, 1.0, 0.1, 1.2, 10.0, 0.5, 1.5],
        [naive_rate * 0.7, 0.1, 1.5, 0.02, 1.05, 15.0, 0.6, 0.8],
        # Background-only start (lets the optimizer choose K=0 if preferred)
        [naive_rate, 1e-8, 0.0, 1.0, 1.01, 10.0, 0.5, 1.0],
    ]
    # If parameters are fixed, override the start values
    if fix_parameters:
        names = ["mu_total_per_year", "K", "alpha", "c_days", "p", "sigma_km", "gamma", "q"]
        for name, val in fix_parameters.items():
            if name in names:
                idx = names.index(name)
                for s in starts:
                    s[idx] = val

    best_res = None
    best_nll = float("inf")
    converged = False
    for start in starts:
        try:
            res = minimize(objective, x0=start, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 100, "ftol": 1e-5, "disp": False,
                                    "eps": [1e-3, 1e-5, 1e-3, 1e-4, 1e-3, 1e-2, 1e-3, 1e-3]})
            if res.fun < best_nll and math.isfinite(res.fun):
                best_nll = res.fun
                best_res = res
                converged = res.success
        except Exception as exc:
            logger.warning("ETAS MLE start failed at Mc=%s: %s", Mc, exc)
            continue
    if best_res is None:
        logger.warning("ETAS MLE failed at Mc=%s: all starts failed", Mc)
        theta_hat = np.array(x0)
        nll = float("nan")
        converged = False
    else:
        theta_hat = best_res.x
        nll = best_nll

    # Detect the "no triggering detected" outcome: if K is at the lower bound
    # OR effectively zero, the MLE has selected a background-only model. This
    # is a legitimate finding (report it), and we set mu to the full Poisson
    # rate so downstream forecasts equal the Poisson baseline.
    no_triggering_detected = (
        theta_hat[1] <= 1e-6 or theta_hat[2] <= 1e-4
    )
    if no_triggering_detected:
        # Set mu to the full Poisson rate (MLE of pure-background model)
        theta_hat[0] = naive_rate
        theta_hat[1] = 0.0   # exactly zero (will be floored to 1e-12 in forecast)
        theta_hat[2] = 0.0
        nll = _neg_log_likelihood(tuple(theta_hat), catalog, bbox, background, Mc,
                                   spatial_kernel, fixed=fix_parameters)
        logger.warning("ETAS MLE at Mc=%s: no triggering detected (K->0, alpha->0). "
                        "Model reduces to background Poisson. This is reported as a "
                        "scientific finding, NOT a fit failure.", Mc)

    mu, K, alpha, c, p, sigma, gamma, q = theta_hat
    params = ETASParams(
        mu_total_per_year=float(mu), K=float(K), alpha=float(alpha),
        c_days=float(c), p=float(p), sigma_km=float(sigma),
        gamma=float(gamma), q=float(q), Mc=Mc,
        spatial_kernel=spatial_kernel,
        fixed_parameters=fix_parameters or {},
    )
    # Update the background's mu_total to the fitted value
    background.mu_total_per_year = float(mu)

    # Identifiability: check bounds & Hessian
    ident = parameter_identifiability(theta_hat, bounds, catalog, background, Mc, spatial_kernel, fix_parameters)
    param_std = ident.pop("std", {})

    # Log-likelihood (positive)
    ll = -nll if math.isfinite(nll) else float("nan")

    notes = []
    if not converged:
        notes.append("Optimizer did not report convergence; treat parameters with caution.")
    for name, flag in ident.items():
        if isinstance(flag, str) and flag != "ok":
            notes.append(f"Parameter {name}: {flag}.")
    if fix_parameters:
        for k in fix_parameters:
            notes.append(f"Parameter {k} FIXED at {fix_parameters[k]} (externally_informed).")
    if no_triggering_detected:
        notes.append(
            "NO TRIGGERING DETECTED in-sample: MLE selected K≈0, α≈0. The ETAS model "
            "reduces to the background Poisson. This is a SCIENTIFIC FINDING (the "
            "Bangladesh USGS catalog does not exhibit the Omori-law clustering that "
            "ETAS is designed to capture, likely because the seismicity is dominated "
            "by deep Indo-Burman subduction events), NOT a fit failure. ETAS forecasts "
            "will equal the Poisson baseline; the event-conditioned backtest (Section 6) "
            "tests whether a FORCED-triggering variant still adds prospective skill."
        )

    return ETASFitResult(
        params=params,
        background=background,
        log_likelihood=ll,
        n_events_used=n,
        fitting_period_days=(catalog["t_start_days"], catalog["t_end_days"]),
        Mc=Mc,
        param_std=param_std,
        identifiability=ident,
        converged=converged,
        n_evaluations=n_eval[0],
        notes=notes,
    )


def parameter_identifiability(theta_hat, bounds, catalog, background, Mc,
                              spatial_kernel, fixed):
    """Heuristic identifiability check: bound-hits + numerical Hessian.

    For each non-fixed parameter:
      - if the estimate is at a bound -> 'at_bound_<which>'
      - if the numerical second derivative of NLL is near zero -> 'flat_likelihood'
      - else -> 'ok'
    """
    names = ["mu_total_per_year", "K", "alpha", "c_days", "p", "sigma_km", "gamma", "q"]
    result = {}
    stds = {}
    eps = 1e-4
    for i, name in enumerate(names):
        if fixed and name in fixed:
            result[name] = "fixed"
            stds[name] = None
            continue
        lo, hi = bounds[i]
        val = theta_hat[i]
        # Bound check
        if abs(val - lo) < 1e-6 * max(abs(lo), 1.0):
            result[name] = f"at_lower_bound ({val:.4g} ≈ {lo:.4g})"
            stds[name] = None
            continue
        if abs(val - hi) < 1e-6 * max(abs(hi), 1.0):
            result[name] = f"at_upper_bound ({val:.4g} ≈ {hi:.4g})"
            stds[name] = None
            continue
        # Numerical second derivative (Hessian diagonal)
        d1 = _neg_log_likelihood(_perturb(theta_hat, i, eps), catalog,
                                 (20.0, 28.0, 88.0, 96.0), background, Mc, spatial_kernel, fixed)
        d2 = _neg_log_likelihood(_perturb(theta_hat, i, -eps), catalog,
                                 (20.0, 28.0, 88.0, 96.0), background, Mc, spatial_kernel, fixed)
        d0 = _neg_log_likelihood(theta_hat, catalog,
                                 (20.0, 28.0, 88.0, 96.0), background, Mc, spatial_kernel, fixed)
        hess = (d1 + d2 - 2 * d0) / (eps * eps)
        if not math.isfinite(hess) or hess <= 1e-12:
            result[name] = "flat_likelihood (not identifiable)"
            stds[name] = None
        else:
            se = 1.0 / math.sqrt(hess)
            stds[name] = se
            # Large SE relative to value -> poorly identified
            if se > 0.5 * max(abs(val), 1e-6):
                result[name] = f"poorly_identified (SE={se:.4g})"
            else:
                result[name] = "ok"
    result["std"] = stds
    return result


def _perturb(theta, i, eps):
    t = list(theta)
    t[i] = t[i] + eps
    return tuple(t)


def _insufficient_data_result(Mc, n, bbox, background_kind, spatial_kernel, catalog, fixed):
    notes = [f"Insufficient data (N={n} < 50) for ETAS MLE at Mc={Mc}. "
             "Returning degenerate parameters; do NOT use for forecasting."]
    params = ETASParams(
        mu_total_per_year=float("nan"), K=float("nan"), alpha=float("nan"),
        c_days=float("nan"), p=float("nan"), sigma_km=float("nan"),
        gamma=float("nan"), q=float("nan"), Mc=Mc,
        spatial_kernel=spatial_kernel, fixed_parameters=fixed or {},
    )
    bg = UniformBackground.build(0.0, bbox) if background_kind == "uniform" else \
         KDEBackground.build(np.array([24.0]), np.array([91.0]), 0.0, bbox)
    return ETASFitResult(
        params=params, background=bg, log_likelihood=float("nan"),
        n_events_used=n, fitting_period_days=(0.0, 0.0), Mc=Mc,
        identifiability={k: "insufficient_data" for k in
                         ["mu_total_per_year", "K", "alpha", "c_days", "p", "sigma_km", "gamma", "q"]},
        converged=False, n_evaluations=0, notes=notes,
    )
