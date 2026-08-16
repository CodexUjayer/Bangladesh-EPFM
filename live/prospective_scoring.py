"""Prospective scoring engine for FINAL_v1.0_FROZEN.

Scores completed forecast windows against actual observations.
NEVER modifies frozen forecasts. Only appends scores.

Flow: FORECAST MADE → FORECAST LOCKED → OUTCOME WINDOW PASSES → OUTCOME OBSERVED → FORECAST SCORED
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("phase_e")

LEDGER_DIR = Path(__file__).resolve().parent / "forecast_ledger"
LEDGER_DIR.mkdir(parents=True, exist_ok=True)

# Production forecast ledger subdirectories (v1 = PRODUCTION, v2 = CANDIDATE).
# The prospective scoring engine scans BOTH subdirectories so that the dashboard
# can display real prospective evidence. Previously the engine globbed only the
# top-level directory and found 0 forecasts — a bug that is now fixed.
LEDGER_V1_DIR = LEDGER_DIR / "v1"
LEDGER_V2_DIR = LEDGER_DIR / "v2_bayesian"
LEDGER_V1_DIR.mkdir(parents=True, exist_ok=True)
LEDGER_V2_DIR.mkdir(parents=True, exist_ok=True)


def _all_ledger_files() -> list:
    """Return all forecast JSON files across v1 and v2 ledger subdirectories,
    sorted by filename. Each entry is a Path. The v1 production ledger takes
    precedence in display order.
    """
    files = list(LEDGER_V1_DIR.glob("forecast_*.json")) + \
            list(LEDGER_V2_DIR.glob("forecast_*.json"))
    return sorted(files)


def _v1_ledger_files() -> list:
    """Return v1 production ledger files only, sorted."""
    return sorted(LEDGER_V1_DIR.glob("forecast_*.json"))

# Reliability bins
RELIABILITY_BINS = [(0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40),
                    (0.40, 0.60), (0.60, 0.80), (0.80, 1.01)]

# Evidence level thresholds
EVIDENCE_LEVELS = {
    0: {"min_forecasts": 0, "description": "No completed prospective forecasts"},
    1: {"min_forecasts": 1, "min_evaluated": 1, "description": "Early monitoring — insufficient evidence"},
    2: {"min_forecasts": 5, "min_evaluated": 3, "description": "Initial prospective evidence — uncertainty remains substantial"},
    3: {"min_forecasts": 20, "min_evaluated": 10, "description": "Meaningful prospective evaluation"},
    4: {"min_forecasts": 50, "min_evaluated": 30, "description": "Strong prospective evidence"},
}


def compute_forecast_hash(record: dict) -> str:
    """Compute SHA-256 hash of forecast content for integrity verification.
    Excludes the hash field, score fields, and integrity warnings."""
    content = {k: v for k, v in record.items()
               if k not in ("forecast_hash", "score", "scored", "scored_at", "integrity_warning")}
    return hashlib.sha256(json.dumps(content, sort_keys=True, default=str).encode()).hexdigest()[:16]


def score_completed_forecasts(events: list) -> list:
    """Score all archived forecasts whose horizon has completed.

    For each forecast in the ledger:
    1. Check if the forecast window has ended
    2. If yes and not already scored, count actual events
    3. Compute Brier, log-likelihood, calibration, sharpness
    4. Append score (never modify original forecast)
    """
    now = datetime.now(timezone.utc)
    scored_summaries = []

    for ledger_file in sorted(_all_ledger_files()):
        record = json.loads(ledger_file.read_text(encoding="utf-8"))

        # Skip if already scored
        if record.get("scored"):
            continue

        forecast_ts_str = record.get("forecast_timestamp")
        if not forecast_ts_str:
            continue
        forecast_ts = datetime.fromisoformat(forecast_ts_str)

        # Verify forecast integrity (hash check)
        stored_hash = record.get("forecast_hash")
        if stored_hash:
            recomputed = compute_forecast_hash(record)
            if stored_hash != recomputed:
                logger.warning("Hash mismatch for %s — forecast may have been modified", ledger_file.name)
                record["integrity_warning"] = "Hash mismatch detected"

        forecasts = record.get("forecasts", {})
        any_scored = False

        for config_key, fc in forecasts.items():
            horizon_years = fc.get("horizon_years", 0)
            horizon_end = forecast_ts + timedelta(days=horizon_years * 365.25)

            if horizon_end > now:
                continue  # Window not yet completed

            threshold = fc["threshold"]

            # Count actual events in [forecast_ts, horizon_end) above threshold
            actual_cells = []
            for e in events:
                if e.origin_time_utc < forecast_ts or e.origin_time_utc >= horizon_end:
                    continue
                mag = e.mw if e.mw is not None else e.original_magnitude
                if mag is not None and mag >= threshold:
                    # Find cell
                    i_lat = min(int((e.latitude - 20.0) / 1.0), 7)
                    i_lon = min(int((e.longitude - 88.0) / 1.0), 7)
                    cell_idx = max(i_lat, 0) * 8 + max(i_lon, 0)
                    actual_cells.append(cell_idx)

            # Per-cell binary outcomes
            y_true = np.zeros(64, dtype=float)
            for idx in actual_cells:
                if 0 <= idx < 64:
                    y_true[idx] = 1.0

            # Forecast probabilities
            y_pred = np.array([c["probability"] for c in fc["cells"]])
            y_pred_lo = np.array([c["probability_lower"] for c in fc["cells"]])
            y_pred_hi = np.array([c["probability_upper"] for c in fc["cells"]])

            # Metrics
            eps = 1e-12
            f = np.clip(y_pred, eps, 1 - eps)
            brier = float(np.mean((y_pred - y_true) ** 2))
            log_lik = float(np.mean(y_true * np.log(f) + (1 - y_true) * np.log(1 - f)))
            n_positive = int(y_true.sum())
            n_expected = float(np.sum(y_pred))  # Expected total count ≈ sum of P
            sharpness = float(np.std(y_pred))

            # Reliability bins
            reliability_bins = []
            for lo, hi in RELIABILITY_BINS:
                mask = (y_pred >= lo) & (y_pred < hi)
                n_in_bin = int(mask.sum())
                if n_in_bin > 0:
                    mean_pred = float(y_pred[mask].mean())
                    obs_freq = float(y_true[mask].mean())
                    reliability_bins.append({
                        "bin": f"{lo:.2f}-{hi:.2f}",
                        "n": n_in_bin,
                        "mean_predicted": round(mean_pred, 4),
                        "observed_frequency": round(obs_freq, 4),
                        "calibration_error": round(abs(mean_pred - obs_freq), 4),
                    })

            # ECE (expected calibration error)
            total_cells = 64
            ece = sum(b["calibration_error"] * b["n"] / total_cells for b in reliability_bins) if reliability_bins else 0.0

            # Hit/miss (for cells with P > some threshold)
            alarm_threshold = 0.10
            hits = int(np.sum((y_pred >= alarm_threshold) & (y_true == 1)))
            false_alarms = int(np.sum((y_pred >= alarm_threshold) & (y_true == 0)))
            misses = int(np.sum((y_pred < alarm_threshold) & (y_true == 1)))
            correct_negatives = int(np.sum((y_pred < alarm_threshold) & (y_true == 0)))

            fc["score"] = {
                "scored_at": now.isoformat(),
                "horizon_end": horizon_end.isoformat(),
                "n_actual_events": n_positive,
                "n_expected_events": round(n_expected, 2),
                "brier_score": round(brier, 6),
                "log_likelihood": round(log_lik, 6),
                "ece": round(ece, 6),
                "sharpness": round(sharpness, 6),
                "reliability_bins": reliability_bins,
                "hits": hits,
                "false_alarms": false_alarms,
                "misses": misses,
                "correct_negatives": correct_negatives,
            }
            any_scored = True
            scored_summaries.append({
                "ledger_file": ledger_file.name,
                "config": config_key,
                "forecast_time": forecast_ts.isoformat(),
                "horizon_end": horizon_end.isoformat(),
                "n_events": n_positive,
                "n_expected": round(n_expected, 2),
                "brier": round(brier, 6),
                "log_likelihood": round(log_lik, 6),
                "ece": round(ece, 6),
            })

        if any_scored:
            record["scored"] = True
            record["scored_at"] = now.isoformat()
            # Preserve original forecast hash (don't recompute after scoring)
            ledger_file.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
            logger.info("Scored: %s", ledger_file.name)

    return scored_summaries


def get_scoring_summary() -> dict:
    """Get summary of all scored forecasts + cumulative metrics + evidence level."""
    all_scores = []
    for ledger_file in sorted(_all_ledger_files()):
        record = json.loads(ledger_file.read_text(encoding="utf-8"))
        if not record.get("scored"):
            continue
        for config_key, fc in record.get("forecasts", {}).items():
            s = fc.get("score")
            if s:
                all_scores.append({
                    "config": config_key,
                    "forecast_time": record["forecast_timestamp"],
                    "brier": s["brier_score"],
                    "log_likelihood": s["log_likelihood"],
                    "ece": s["ece"],
                    "n_events": s["n_actual_events"],
                    "n_expected": s["n_expected_events"],
                    "sharpness": s["sharpness"],
                })

    n_forecasts = len(list(_all_ledger_files()))
    n_evaluated = len(all_scores)

    # Determine evidence level
    level = 0
    for lv in sorted(EVIDENCE_LEVELS.keys(), reverse=True):
        req = EVIDENCE_LEVELS[lv]
        if n_forecasts >= req.get("min_forecasts", 0) and n_evaluated >= req.get("min_evaluated", 0):
            level = lv
            break

    if not all_scores:
        return {
            "n_forecasts_issued": n_forecasts,
            "n_forecasts_evaluated": 0,
            "evidence_level": level,
            "evidence_description": EVIDENCE_LEVELS[level]["description"],
            "message": "No completed prospective forecast windows scored yet.",
            "insufficient": True,
        }

    briers = [s["brier"] for s in all_scores]
    log_liks = [s["log_likelihood"] for s in all_scores]
    eces = [s["ece"] for s in all_scores]
    sharpnesses = [s["sharpness"] for s in all_scores]

    # Per-config breakdown
    by_config = {}
    for s in all_scores:
        cfg = s["config"]
        if cfg not in by_config:
            by_config[cfg] = {"scores": [], "briers": [], "log_liks": [], "eces": []}
        by_config[cfg]["scores"].append(s)
        by_config[cfg]["briers"].append(s["brier"])
        by_config[cfg]["log_liks"].append(s["log_likelihood"])
        by_config[cfg]["eces"].append(s["ece"])

    config_summary = {}
    for cfg, data in by_config.items():
        config_summary[cfg] = {
            "n_evaluated": len(data["scores"]),
            "mean_brier": round(float(np.mean(data["briers"])), 6),
            "mean_log_lik": round(float(np.mean(data["log_liks"])), 6),
            "mean_ece": round(float(np.mean(data["eces"])), 6),
            "total_events": sum(s["n_events"] for s in data["scores"]),
            "total_expected": round(sum(s["n_expected"] for s in data["scores"]), 2),
        }

    # Bootstrap CI on mean Brier (if enough data)
    brier_ci = None
    if len(briers) >= 5:
        rng = np.random.default_rng(42)
        boot_means = []
        for _ in range(500):
            sample = rng.choice(briers, size=len(briers), replace=True)
            boot_means.append(np.mean(sample))
        brier_ci = [round(float(np.percentile(boot_means, 2.5)), 6),
                    round(float(np.percentile(boot_means, 97.5)), 6)]

    # Aggregated reliability bins
    agg_bins = {f"{lo:.2f}-{hi:.2f}": {"n": 0, "pred_sum": 0.0, "obs_sum": 0.0}
                for lo, hi in RELIABILITY_BINS}
    for s in all_scores:
        # We need to re-aggregate from the ledger (per-forecast bins)
        pass  # Aggregation done from ledger directly in dashboard

    return {
        "n_forecasts_issued": n_forecasts,
        "n_forecasts_evaluated": n_evaluated,
        "evidence_level": level,
        "evidence_description": EVIDENCE_LEVELS[level]["description"],
        "insufficient": n_evaluated < 10,
        "cumulative": {
            "mean_brier": round(float(np.mean(briers)), 6),
            "brier_ci": brier_ci,
            "mean_log_lik": round(float(np.mean(log_liks)), 6),
            "mean_ece": round(float(np.mean(eces)), 6),
            "mean_sharpness": round(float(np.mean(sharpnesses)), 6),
            "total_observed_events": sum(s["n_events"] for s in all_scores),
            "total_expected_events": round(sum(s["n_expected"] for s in all_scores), 2),
        },
        "by_config": config_summary,
        "all_scores": all_scores,
    }


def get_forecast_history() -> list:
    """Get list of all forecasts in the ledger for the history viewer."""
    history = []
    for ledger_file in sorted(_all_ledger_files(), reverse=True):
        record = json.loads(ledger_file.read_text(encoding="utf-8"))
        entry = {
            "file": ledger_file.name,
            "timestamp": record.get("forecast_timestamp"),
            "model_version": record.get("model_version"),
            "catalog_n_events": record.get("catalog_n_events"),
            "scored": record.get("scored", False),
            "configs": list(record.get("forecasts", {}).keys()),
        }
        if record.get("scored"):
            scores = {}
            for cfg, fc in record.get("forecasts", {}).items():
                s = fc.get("score")
                if s:
                    scores[cfg] = {
                        "brier": s["brier_score"],
                        "n_events": s["n_actual_events"],
                        "n_expected": s["n_expected_events"],
                    }
            entry["scores"] = scores
        history.append(entry)
    return history


def get_data_quality_status() -> dict:
    """Check data source availability and catalog quality."""
    import urllib.request
    now = datetime.now(timezone.utc)

    # Check USGS
    usgs_ok = False
    try:
        req = urllib.request.Request(
            "https://earthquake.usgs.gov/fdsnws/event/1/count?starttime=2024-01-01&endtime=2024-02-01&minmagnitude=4.0",
            headers={"User-Agent": "eq-monitor/1.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            json.loads(r.read())
        usgs_ok = True
    except:
        pass

    # Check ISC
    isc_ok = False
    try:
        req = urllib.request.Request(
            "http://www.isc.ac.uk/fdsnws/event/1/query?format=text&starttime=2024-01-01&endtime=2024-02-01&minlat=20&maxlat=28&minlon=88&maxlon=96&minmag=4.0",
            headers={"User-Agent": "eq-monitor/1.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read(100)
        isc_ok = True
    except:
        pass

    # Check latest forecast age
    latest_path = Path(__file__).resolve().parent / "latest_forecast.json"
    forecast_age_hours = None
    if latest_path.exists():
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        ts = datetime.fromisoformat(latest["forecast_timestamp"])
        forecast_age_hours = (now - ts).total_seconds() / 3600

    return {
        "usgs_available": usgs_ok,
        "isc_available": isc_ok,
        "forecast_age_hours": round(forecast_age_hours, 1) if forecast_age_hours else None,
        "n_ledger_forecasts": len(list(_all_ledger_files())),
        "timestamp": now.isoformat(),
        "warnings": [
            "USGS unavailable" if not usgs_ok else None,
            "ISC unavailable" if not isc_ok else None,
            f"Forecast is {forecast_age_hours:.0f}h old" if forecast_age_hours and forecast_age_hours > 24 else None,
        ],
        "warnings_clean": [w for w in [
            "USGS unavailable" if not usgs_ok else None,
            "ISC unavailable" if not isc_ok else None,
            f"Forecast is {forecast_age_hours:.0f}h old" if forecast_age_hours and forecast_age_hours > 24 else None,
        ] if w],
    }


def check_forecast_integrity() -> dict:
    """Verify forecast integrity: bounds, hashes, immutability."""
    issues = []
    n_checked = 0

    for ledger_file in sorted(_all_ledger_files()):
        n_checked += 1
        record = json.loads(ledger_file.read_text(encoding="utf-8"))

        # Check model version
        if record.get("model_version") != "FINAL_v1.0_FROZEN":
            issues.append(f"{ledger_file.name}: wrong model version")

        # Check probabilities in [0,1]
        for cfg, fc in record.get("forecasts", {}).items():
            for cell in fc.get("cells", []):
                p = cell.get("probability", 0)
                if p < 0 or p > 1:
                    issues.append(f"{ledger_file.name}/{cfg}/{cell['cell_id']}: P out of bounds ({p})")
                lo = cell.get("probability_lower", 0)
                hi = cell.get("probability_upper", 0)
                if lo > p or p > hi:
                    if not (lo == 0 and hi == 0 and p == 0):  # Allow all-zero
                        issues.append(f"{ledger_file.name}/{cfg}/{cell['cell_id']}: UI bounds violation")

        # Check hash if present
        stored_hash = record.get("forecast_hash")
        if stored_hash:
            recomputed = compute_forecast_hash(record)
            if stored_hash != recomputed:
                issues.append(f"{ledger_file.name}: hash mismatch (forecast may have been modified)")

    return {
        "n_checked": n_checked,
        "n_issues": len(issues),
        "issues": issues[:10],  # First 10
        "all_ok": len(issues) == 0,
    }


if __name__ == "__main__":
    # Run scoring on any completed forecasts
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.ingestion import build_canonical_events, read_usgs_csv
    from src.phase_c.isc_reader import read_isc_text

    usgs = read_usgs_csv(Path(__file__).resolve().parent.parent / "data/raw/usgs/usgs_bangladesh_1973_2025_m25.csv")
    isc = read_isc_text(Path(__file__).resolve().parent.parent / "data/raw/isc/isc_bangladesh_1973_2025_m3.txt")
    events = build_canonical_events(usgs + isc, time_window_s=120.0, spatial_window_km=50.0)

    scored = score_completed_forecasts(events)
    summary = get_scoring_summary()
    integrity = check_forecast_integrity()
    dq = get_data_quality_status()

    print(json.dumps({
        "scored": scored,
        "summary": summary,
        "integrity": integrity,
        "data_quality": dq,
    }, indent=2, default=str))
