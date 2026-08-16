"""Parallel prospective v1 vs v2 forecasting pipeline.

Generates BOTH v1 (Spatial Poisson) and v2 (Bayesian hierarchical) forecasts
from the SAME catalog snapshot at the SAME timestamp. Saves to separate
immutable ledgers. Scores both against the SAME future observations.

FINAL_v1.0_FROZEN remains production.
FINAL_v2.0_CANDIDATE_BAYESIAN_SPATIAL remains non-production candidate.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import build_canonical_events, read_usgs_csv
from src.phase_c.isc_reader import read_isc_text
from src.baselines.poisson import HORIZON_YEARS
from src.baselines.uncertainty import poisson_rate_ci_garwood
from src.ml.features import MLGridConfig
from src.ml.spatial_poisson import causal_spatial_rate, spatial_poisson_forecast
from v2_candidates.bayesian_spatial.model import (
    BayesianSpatialConfig,
    fit_bayesian_hierarchical,
    compute_probabilities,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("parallel")

MODEL_V1 = "FINAL_v1.0_FROZEN"
MODEL_V2 = "FINAL_v2.0_CANDIDATE_BAYESIAN_SPATIAL"
FROZEN_MC = 4.13
FROZEN_B = 0.808
GRID = MLGridConfig()
BBOX = (20.0, 28.0, 88.0, 96.0)
CATALOG_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
LEDGER_V1 = Path(__file__).resolve().parent / "forecast_ledger" / "v1"
LEDGER_V2 = Path(__file__).resolve().parent / "forecast_ledger" / "v2_bayesian"
LEDGER_V1.mkdir(parents=True, exist_ok=True)
LEDGER_V2.mkdir(parents=True, exist_ok=True)

FORECAST_CONFIGS = [
    {"threshold": 4.5, "horizon": "7d"},
    {"threshold": 4.5, "horizon": "30d"},
    {"threshold": 5.0, "horizon": "7d"},
    {"threshold": 5.0, "horizon": "30d"},
]

# v2 promotion evidence levels
V2_EVIDENCE_LEVELS = {
    0: {"min_evaluated": 0, "description": "0 evaluated forecasts"},
    1: {"min_evaluated": 1, "description": "1-4 evaluated forecasts — early monitoring"},
    2: {"min_evaluated": 5, "description": "5-9 evaluated forecasts — initial evidence"},
    3: {"min_evaluated": 10, "description": "10-19 evaluated forecasts — meaningful but limited"},
    4: {"min_evaluated": 20, "description": ">=20 evaluated — eligible for formal promotion assessment"},
}

# Promotion criteria (predefined, not adjustable)
PROMOTION_CRITERIA = [
    "At least 20 evaluated forecast origins",
    "No statistically significant degradation in Brier score",
    "No statistically significant degradation in log score",
    "Demonstrable improvement in calibration OR uncertainty quality",
    "Appropriate posterior predictive interval coverage (count-based)",
    "No material loss of sharpness",
    "Stable results across M>=4.5 and M>=5.0",
    "Stable results across 7d and 30d",
    "No evidence of data leakage",
    "No dependence on a single unusual event",
    "Results remain consistent over time",
    "Improvement survives multiple-comparison considerations",
]


def _compute_hash(record: dict) -> str:
    """SHA-256 hash of forecast content (excluding hash/score fields)."""
    content = {k: v for k, v in record.items()
               if k not in ("forecast_hash", "score", "scored", "scored_at", "integrity_warning")}
    return hashlib.sha256(json.dumps(content, sort_keys=True, default=str).encode()).hexdigest()[:16]


def fetch_usgs(start: str, end: str) -> list:
    url = (f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv"
           f"&starttime={start}&endtime={end}&minmagnitude=2.5"
           f"&minlatitude=20&maxlatitude=28&minlongitude=88&maxlongitude=96"
           f"&eventtype=earthquake&orderby=time-asc")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "bd-eq-forecast/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read().decode("utf-8")
        tmp = CATALOG_DIR / "usgs" / "usgs_live_latest.csv"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(raw, encoding="utf-8")
        return read_usgs_csv(tmp)
    except Exception as e:
        logger.error("USGS fetch failed: %s", e)
        return []


def fetch_isc(start: str, end: str) -> list:
    url = (f"http://www.isc.ac.uk/fdsnws/event/1/query?format=text"
           f"&starttime={start}&endtime={end}&minlat=20&maxlat=28&minlon=88&maxlon=96"
           f"&minmag=3.0&orderby=time-asc")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "bd-eq-forecast/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read().decode("utf-8")
        tmp = CATALOG_DIR / "isc" / "isc_live_latest.txt"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(raw, encoding="utf-8")
        return read_isc_text(tmp)
    except Exception as e:
        logger.error("ISC fetch failed: %s", e)
        return []


def load_catalog() -> list:
    usgs_file = CATALOG_DIR / "usgs" / "usgs_bangladesh_1973_2025_m25.csv"
    isc_file = CATALOG_DIR / "isc" / "isc_bangladesh_1973_2025_m3.txt"
    obs = []
    if usgs_file.exists():
        obs.extend(read_usgs_csv(usgs_file))
    if isc_file.exists():
        obs.extend(read_isc_text(isc_file))
    return obs


def generate_dual_forecast(events: list, catalog_start: datetime) -> dict:
    """Generate both v1 and v2 forecasts from the same catalog snapshot."""
    ts = datetime.now(timezone.utc)
    v1_forecasts = {}
    v2_forecasts = {}

    for cfg in FORECAST_CONFIGS:
        th = cfg["threshold"]
        hz = cfg["horizon"]
        hy = HORIZON_YEARS[hz]
        key = f"M{th}_{hz}"

        # === v1: Spatial Poisson ===
        sp_rates = causal_spatial_rate(
            events, origin_time=ts, grid=GRID, threshold=th,
            catalog_start=catalog_start, method="expanding", smoothing="raw")
        v1_probs = spatial_poisson_forecast(sp_rates, hy)
        exposure = (ts - catalog_start).total_seconds() / (365.25 * 86400)
        v1_cells = []
        for i in range(64):
            n_cell = int(sp_rates[i] * exposure)
            ci = poisson_rate_ci_garwood(n_cell, exposure)
            p_lo = max(1.0 - math.exp(-ci[1] * hy), 0.0)
            p_hi = min(1.0 - math.exp(-ci[0] * hy), 1.0)
            v1_cells.append({
                "cell_id": f"cell_{i//8:02d}_{i%8:02d}",
                "lat_center": round(20.0 + (i//8 + 0.5) * 1.0, 2),
                "lon_center": round(88.0 + (i%8 + 0.5) * 1.0, 2),
                "rate_per_year": round(float(sp_rates[i]), 6),
                "probability": round(float(v1_probs[i]), 6),
                "probability_lower": round(float(p_lo), 6),
                "probability_upper": round(float(p_hi), 6),
                # Count-based CI for posterior predictive coverage
                "count_lower": max(int(ci[0] * hy), 0),
                "count_upper": max(int(math.ceil(ci[1] * hy)), 0),
            })
        v1_forecasts[key] = {
            "threshold": th, "horizon": hz, "horizon_years": hy,
            "regional_rate": round(float(np.sum(sp_rates)), 4),
            "regional_probability": round(1.0 - math.exp(-float(np.sum(sp_rates)) * hy), 6),
            "n_cells": 64, "cells": v1_cells,
        }

        # === v2: Bayesian hierarchical ===
        config = BayesianSpatialConfig(mc=FROZEN_MC)
        cells_b, alpha_p, beta_p, exp_yr = fit_bayesian_hierarchical(
            events, threshold=th, catalog_start=catalog_start,
            forecast_origin=ts, config=config)
        compute_probabilities(cells_b, hy, config)

        v2_cells = []
        for c in cells_b:
            # Count-based posterior predictive: N ~ Poisson(λ * Δt)
            # Posterior on λ is Gamma(α, β), so posterior predictive on N
            # is Negative Binomial(r=α, p=β/(β+Δt))
            # Count quantiles from the NB distribution
            nb_r = c.alpha
            nb_p = c.beta / (c.beta + hy)
            count_dist = scipy_stats.nbinom(nb_r, nb_p)
            count_50 = int(count_dist.ppf(0.25))
            count_80 = int(count_dist.ppf(0.10))
            count_90 = int(count_dist.ppf(0.05))
            count_95 = int(count_dist.ppf(0.025))
            count_50_hi = int(count_dist.ppf(0.75))
            count_80_hi = int(count_dist.ppf(0.90))
            count_90_hi = int(count_dist.ppf(0.95))
            count_95_hi = int(count_dist.ppf(0.975))

            v2_cells.append({
                "cell_id": c.cell_id,
                "lat_center": round(c.lat_center, 2),
                "lon_center": round(c.lon_center, 2),
                "rate_mean": round(c.rate_mean, 6),
                "rate_lower": round(c.rate_lower, 6),
                "rate_upper": round(c.rate_upper, 6),
                "prob_mean": round(c.prob_mean, 6),
                "prob_median": round(c.prob_median, 6),
                "prob_lower": round(c.prob_lower, 6),
                "prob_upper": round(c.prob_upper, 6),
                # Count-based posterior predictive intervals
                "count_50_lower": count_50, "count_50_upper": count_50_hi,
                "count_80_lower": count_80, "count_80_upper": count_80_hi,
                "count_90_lower": count_90, "count_90_upper": count_90_hi,
                "count_95_lower": count_95, "count_95_upper": count_95_hi,
                "alpha": round(c.alpha, 4), "beta": round(c.beta, 4),
            })

        total_rate_v2 = sum(c.rate_mean for c in cells_b)
        v2_forecasts[key] = {
            "threshold": th, "horizon": hz, "horizon_years": hy,
            "prior_alpha": round(alpha_p, 6), "prior_beta": round(beta_p, 6),
            "regional_rate": round(total_rate_v2, 4),
            "regional_probability": round(1.0 - math.exp(-total_rate_v2 * hy), 6),
            "n_cells": 64, "cells": v2_cells,
        }

    return {"timestamp": ts, "v1": v1_forecasts, "v2": v2_forecasts,
            "catalog_n_events": len(events),
            "catalog_start": catalog_start.isoformat()}


def save_dual_forecast(dual: dict) -> tuple[str, str]:
    """Save v1 and v2 forecasts to separate immutable ledgers."""
    ts = dual["timestamp"]
    filename = f"forecast_{ts.strftime('%Y-%m-%d_%H%M%S')}.json"
    common = {
        "forecast_timestamp": ts.isoformat(),
        "frozen_mc": FROZEN_MC,
        "frozen_b": FROZEN_B,
        "catalog_n_events": dual["catalog_n_events"],
        "catalog_start": dual["catalog_start"],
        "grid": {"cell_size_deg": 1.0, "n_cells": 64, "bbox": list(BBOX)},
    }

    # v1 record
    v1_record = {**common, "model_version": MODEL_V1, "forecasts": dual["v1"],
                 "warnings": ["PROBABILISTIC FORECAST — not deterministic prediction.",
                              "FINAL_v1.0_FROZEN — PRODUCTION MODEL"]}
    v1_record["forecast_hash"] = _compute_hash(v1_record)
    v1_path = LEDGER_V1 / filename
    v1_path.write_text(json.dumps(v1_record, indent=2, default=str), encoding="utf-8")

    # v2 record
    v2_record = {**common, "model_version": MODEL_V2, "forecasts": dual["v2"],
                 "warnings": ["CANDIDATE — NOT PRODUCTION.",
                              "FINAL_v2.0_CANDIDATE_BAYESIAN_SPATIAL.",
                              "Do not use for operational decisions."]}
    v2_record["forecast_hash"] = _compute_hash(v2_record)
    v2_path = LEDGER_V2 / filename
    v2_path.write_text(json.dumps(v2_record, indent=2, default=str), encoding="utf-8")

    logger.info("v1 saved: %s (hash: %s)", v1_path.name, v1_record["forecast_hash"])
    logger.info("v2 saved: %s (hash: %s)", v2_path.name, v2_record["forecast_hash"])
    return v1_path.name, v2_path.name


def score_completed_forecasts(events: list) -> list:
    """Score both v1 and v2 forecasts whose windows have completed."""
    now = datetime.now(timezone.utc)
    scored = []

    v1_files = sorted(LEDGER_V1.glob("forecast_*.json"))
    v2_files = sorted(LEDGER_V2.glob("forecast_*.json"))

    # Match by filename
    v1_map = {f.name: f for f in v1_files}
    v2_map = {f.name: f for f in v2_files}
    common_files = sorted(set(v1_map.keys()) & set(v2_map.keys()))

    for fname in common_files:
        v1_rec = json.loads(v1_map[fname].read_text(encoding="utf-8"))
        v2_rec = json.loads(v2_map[fname].read_text(encoding="utf-8"))

        if v1_rec.get("scored") or v2_rec.get("scored"):
            continue

        fc_ts = datetime.fromisoformat(v1_rec["forecast_timestamp"])
        any_scored = False

        for cfg_key in FORECAST_CONFIGS:
            key = f"M{cfg_key['threshold']}_{cfg_key['horizon']}"
            hy = HORIZON_YEARS[cfg_key["horizon"]]
            th = cfg_key["threshold"]
            horizon_end = fc_ts + timedelta(days=hy * 365.25)
            if horizon_end > now:
                continue

            # Count actual events per cell
            actual_counts = np.zeros(64, dtype=int)
            for e in events:
                if e.origin_time_utc < fc_ts or e.origin_time_utc >= horizon_end:
                    continue
                m = e.mw if e.mw is not None else e.original_magnitude
                if m is not None and m >= th:
                    i_lat = min(int((e.latitude - 20) / 1.0), 7)
                    i_lon = min(int((e.longitude - 88) / 1.0), 7)
                    actual_counts[max(i_lat,0)*8 + max(i_lon,0)] += 1

            y_binary = (actual_counts > 0).astype(float)

            for model_key, rec in [("v1", v1_rec), ("v2", v2_rec)]:
                fc = rec.get("forecasts", {}).get(key)
                if not fc:
                    continue

                cells = fc["cells"]
                probs = np.array([c["probability"] if model_key == "v1" else c["prob_mean"] for c in cells])

                # Brier (binary)
                eps = 1e-12
                brier = float(np.mean((probs - y_binary) ** 2))
                ll = float(np.mean(y_binary * np.log(np.clip(probs, eps, 1-eps)) +
                                   (1-y_binary) * np.log(np.clip(1-probs, eps, 1-eps))))

                # ECE (7 bins)
                bins = np.linspace(0, 1, 8)
                ece = 0.0
                for i in range(len(bins)-1):
                    mask = (probs >= bins[i]) & (probs < bins[i+1])
                    if mask.sum() > 0:
                        ece += abs(float(probs[mask].mean()) - float(y_binary[mask].mean())) * mask.sum() / len(probs)

                sharpness = float(np.std(probs))

                # Hit/false-alarm (P >= 10% threshold)
                alarm = 0.10
                hits = int(np.sum((probs >= alarm) & (y_binary == 1)))
                fa = int(np.sum((probs >= alarm) & (y_binary == 0)))
                misses = int(np.sum((probs < alarm) & (y_binary == 1)))
                cn = int(np.sum((probs < alarm) & (y_binary == 0)))

                # Count-based posterior predictive coverage (CORRECT uncertainty eval)
                # For each cell, check if observed count falls within the count CI
                coverage_results = {}
                for level, lo_key, hi_key in [
                    (0.50, "count_50_lower", "count_50_upper"),
                    (0.80, "count_80_lower", "count_80_upper"),
                    (0.90, "count_90_lower", "count_90_upper"),
                    (0.95, "count_95_lower", "count_95_upper"),
                ]:
                    if model_key == "v1":
                        # v1 doesn't have count CIs; use Poisson CI on rate * Δt
                        # Approximate using the probability CI
                        # For Poisson: count CI ≈ -ln(1-P_lo) to -ln(1-P_hi) but inverted
                        # Simpler: use the Garwood CI already computed
                        # The v1 cells have count_lower/count_upper (95% only)
                        if level == 0.95:
                            lo_arr = np.array([c.get("count_lower", 0) for c in cells])
                            hi_arr = np.array([c.get("count_upper", 0) for c in cells])
                        else:
                            # Approximate by scaling
                            z_ratio = scipy_stats.norm.ppf((1+level)/2) / scipy_stats.norm.ppf(0.975)
                            lo95 = np.array([c.get("count_lower", 0) for c in cells])
                            hi95 = np.array([c.get("count_upper", 0) for c in cells])
                            mid = (lo95 + hi95) / 2
                            lo_arr = np.maximum(mid - (mid - lo95) * z_ratio, 0).astype(int)
                            hi_arr = (mid + (hi95 - mid) * z_ratio).astype(int)
                    else:
                        lo_arr = np.array([c.get(lo_key, 0) for c in cells])
                        hi_arr = np.array([c.get(hi_key, 0) for c in cells])

                    covered = (actual_counts >= lo_arr) & (actual_counts <= hi_arr)
                    cov = float(np.mean(covered))
                    width = float(np.mean(hi_arr - lo_arr))

                    # Interval score (Gneiting & Raftery 2007)
                    # IS = (hi - lo) + 2/α * (lo - y) * I(y < lo) + 2/α * (y - hi) * I(y > hi)
                    alpha_is = 1 - level
                    penalty_lo = np.where(actual_counts < lo_arr, (2/alpha_is) * (lo_arr - actual_counts), 0)
                    penalty_hi = np.where(actual_counts > hi_arr, (2/alpha_is) * (actual_counts - hi_arr), 0)
                    interval_score = float(np.mean((hi_arr - lo_arr) + penalty_lo + penalty_hi))

                    coverage_results[f"{int(level*100)}%"] = {
                        "nominal": level,
                        "empirical_coverage": round(cov, 4),
                        "coverage_error": round(abs(cov - level), 4),
                        "interval_width": round(width, 4),
                        "interval_score": round(interval_score, 4),
                    }

                fc["score"] = {
                    "scored_at": now.isoformat(),
                    "horizon_end": horizon_end.isoformat(),
                    "n_actual_events": int(actual_counts.sum()),
                    "n_cells_with_events": int(y_binary.sum()),
                    "brier_score": round(brier, 6),
                    "log_likelihood": round(ll, 6),
                    "ece": round(ece, 6),
                    "sharpness": round(sharpness, 6),
                    "hits": hits, "false_alarms": fa,
                    "misses": misses, "correct_negatives": cn,
                    "count_coverage": coverage_results,
                }
                any_scored = True

        if any_scored:
            v1_rec["scored"] = True
            v1_rec["scored_at"] = now.isoformat()
            v1_map[fname].write_text(json.dumps(v1_rec, indent=2, default=str), encoding="utf-8")
            v2_rec["scored"] = True
            v2_rec["scored_at"] = now.isoformat()
            v2_map[fname].write_text(json.dumps(v2_rec, indent=2, default=str), encoding="utf-8")
            scored.append(fname)
            logger.info("Scored: %s", fname)

    return scored


def get_comparison_summary() -> dict:
    """Get cumulative v1 vs v2 comparison summary."""
    v1_scores = []
    v2_scores = []

    v1_files = sorted(LEDGER_V1.glob("forecast_*.json"))
    v2_files = sorted(LEDGER_V2.glob("forecast_*.json"))
    v1_map = {f.name: f for f in v1_files}
    v2_map = {f.name: f for f in v2_files}
    common = sorted(set(v1_map.keys()) & set(v2_map.keys()))

    for fname in common:
        v1_rec = json.loads(v1_map[fname].read_text(encoding="utf-8"))
        v2_rec = json.loads(v2_map[fname].read_text(encoding="utf-8"))
        if not v1_rec.get("scored"):
            continue
        for cfg_key in ["M4.5_7d", "M4.5_30d", "M5.0_7d", "M5.0_30d"]:
            v1_fc = v1_rec.get("forecasts", {}).get(cfg_key, {})
            v2_fc = v2_rec.get("forecasts", {}).get(cfg_key, {})
            s1 = v1_fc.get("score")
            s2 = v2_fc.get("score")
            if s1 and s2:
                v1_scores.append({"config": cfg_key, "file": fname, **s1})
                v2_scores.append({"config": cfg_key, "file": fname, **s2})

    n_issued = len(common)
    n_evaluated = len(v1_scores) // 4 if v1_scores else 0  # 4 configs per forecast

    # Evidence level
    level = 0
    for lv in sorted(V2_EVIDENCE_LEVELS.keys(), reverse=True):
        if n_evaluated >= V2_EVIDENCE_LEVELS[lv]["min_evaluated"]:
            level = lv
            break

    if not v1_scores:
        return {
            "n_issued": n_issued, "n_evaluated": 0,
            "evidence_level": level,
            "evidence_description": V2_EVIDENCE_LEVELS[level]["description"],
            "v1_status": "PRODUCTION", "v2_status": "CANDIDATE",
            "insufficient": True,
            "message": "No completed forecast windows scored yet.",
            "promotion_criteria": PROMOTION_CRITERIA,
        }

    # Cumulative metrics per config
    by_config = {}
    for cfg_key in ["M4.5_7d", "M4.5_30d", "M5.0_7d", "M5.0_30d"]:
        v1_cfg = [s for s in v1_scores if s["config"] == cfg_key]
        v2_cfg = [s for s in v2_scores if s["config"] == cfg_key]
        if not v1_cfg:
            continue

        briers_v1 = [s["brier_score"] for s in v1_cfg]
        briers_v2 = [s["brier_score"] for s in v2_cfg]
        lls_v1 = [s["log_likelihood"] for s in v1_cfg]
        lls_v2 = [s["log_likelihood"] for s in v2_cfg]
        eces_v1 = [s["ece"] for s in v1_cfg]
        eces_v2 = [s["ece"] for s in v2_cfg]
        sharps_v1 = [s["sharpness"] for s in v1_cfg]
        sharps_v2 = [s["sharpness"] for s in v2_cfg]

        # Count coverage at 95%
        cov95_v1 = [s.get("count_coverage", {}).get("95%", {}).get("empirical_coverage", 0) for s in v1_cfg]
        cov95_v2 = [s.get("count_coverage", {}).get("95%", {}).get("empirical_coverage", 0) for s in v2_cfg]
        is95_v1 = [s.get("count_coverage", {}).get("95%", {}).get("interval_score", 0) for s in v1_cfg]
        is95_v2 = [s.get("count_coverage", {}).get("95%", {}).get("interval_score", 0) for s in v2_cfg]

        by_config[cfg_key] = {
            "n_evaluated": len(v1_cfg),
            "brier_v1": round(float(np.mean(briers_v1)), 6),
            "brier_v2": round(float(np.mean(briers_v2)), 6),
            "delta_brier": round(float(np.mean(briers_v1)) - float(np.mean(briers_v2)), 6),
            "log_lik_v1": round(float(np.mean(lls_v1)), 6),
            "log_lik_v2": round(float(np.mean(lls_v2)), 6),
            "delta_ll": round(float(np.mean(lls_v2)) - float(np.mean(lls_v1)), 6),
            "ece_v1": round(float(np.mean(eces_v1)), 6),
            "ece_v2": round(float(np.mean(eces_v2)), 6),
            "delta_ece": round(float(np.mean(eces_v1)) - float(np.mean(eces_v2)), 6),
            "sharpness_v1": round(float(np.mean(sharps_v1)), 6),
            "sharpness_v2": round(float(np.mean(sharps_v2)), 6),
            "coverage_95_v1": round(float(np.mean(cov95_v1)), 4),
            "coverage_95_v2": round(float(np.mean(cov95_v2)), 4),
            "interval_score_v1": round(float(np.mean(is95_v1)), 4),
            "interval_score_v2": round(float(np.mean(is95_v2)), 4),
        }

    # Bootstrap CI on ΔBrier
    boot_ci = None
    if n_evaluated >= 3:
        rng = np.random.default_rng(42)
        all_delta_brier = []
        for cfg_key in by_config:
            v1_cfg = [s["brier_score"] for s in v1_scores if s["config"] == cfg_key]
            v2_cfg = [s["brier_score"] for s in v2_scores if s["config"] == cfg_key]
            for i in range(len(v1_cfg)):
                all_delta_brier.append(v1_cfg[i] - v2_cfg[i])
        if len(all_delta_brier) >= 3:
            boot_deltas = []
            for _ in range(500):
                idx = rng.integers(0, len(all_delta_brier), size=len(all_delta_brier))
                boot_deltas.append(np.mean([all_delta_brier[i] for i in idx]))
            boot_ci = [round(float(np.percentile(boot_deltas, 2.5)), 6),
                       round(float(np.percentile(boot_deltas, 97.5)), 6)]

    return {
        "n_issued": n_issued, "n_evaluated": n_evaluated,
        "evidence_level": level,
        "evidence_description": V2_EVIDENCE_LEVELS[level]["description"],
        "v1_status": "PRODUCTION", "v2_status": "CANDIDATE",
        "insufficient": n_evaluated < 20,
        "by_config": by_config,
        "bootstrap_delta_brier_ci": boot_ci,
        "promotion_criteria": PROMOTION_CRITERIA,
        "promotion_eligible": n_evaluated >= 20,
        "all_v1_scores": v1_scores[:20],  # Latest 20 for dashboard
        "all_v2_scores": v2_scores[:20],
    }


def run_parallel_pipeline() -> dict:
    """Run the dual v1+v2 forecast pipeline."""
    logger.info("=== PARALLEL v1+v2 PIPELINE START ===")
    now = datetime.now(timezone.utc)
    start_date = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = (now + timedelta(days=1)).strftime("%Y-%m-%d")

    new_usgs = fetch_usgs(start_date, end_date)
    new_isc = fetch_isc(start_date, end_date)
    existing = load_catalog()
    all_obs = existing + new_usgs + new_isc
    events = build_canonical_events(all_obs, time_window_s=120.0, spatial_window_km=50.0)
    logger.info("Catalog: %d events", len(events))

    if not events:
        return {"status": "error", "reason": "no events"}

    catalog_start = min(e.origin_time_utc for e in events)
    dual = generate_dual_forecast(events, catalog_start)
    v1_file, v2_file = save_dual_forecast(dual)

    # Score any completed forecasts
    scored = score_completed_forecasts(events)

    # Save latest for dashboard
    latest = {
        "timestamp": dual["timestamp"].isoformat(),
        "catalog_n_events": dual["catalog_n_events"],
        "v1_forecasts": dual["v1"],
        "v2_forecasts": dual["v2"],
        "comparison": get_comparison_summary(),
    }
    latest_path = Path(__file__).resolve().parent / "latest_parallel.json"
    latest_path.write_text(json.dumps(latest, indent=2, default=str), encoding="utf-8")

    logger.info("=== PARALLEL PIPELINE COMPLETE ===")
    return {
        "status": "success",
        "v1_file": v1_file, "v2_file": v2_file,
        "catalog_n_events": len(events),
        "scored": scored,
        "comparison": get_comparison_summary(),
    }


if __name__ == "__main__":
    result = run_parallel_pipeline()
    print(json.dumps(result, indent=2, default=str))
