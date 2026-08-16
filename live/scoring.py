"""Prospective scoring infrastructure for archived forecasts.

Evaluates completed forecast windows against actual observations.
NEVER modifies frozen forecasts. Only scores them against future outcomes.

Flow: FORECAST MADE → OUTCOME OBSERVED LATER → FORECAST SCORED
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("live.scoring")

LEDGER_DIR = Path(__file__).resolve().parent / "forecast_ledger"


def score_completed_forecasts(events: list) -> list:
    """Score all archived forecasts whose horizon has completed.

    For each forecast in the ledger:
    1. Check if the forecast window has ended (forecast_time + horizon < now)
    2. If yes and not already scored, count actual events in the window
    3. Compute Brier score, log-likelihood, calibration
    4. Save the score alongside the forecast (append, never overwrite)

    Returns list of scored forecast summaries.
    """
    now = datetime.now(timezone.utc)
    scored = []

    for ledger_file in sorted(LEDGER_DIR.glob("forecast_*.json")):
        record = json.loads(ledger_file.read_text(encoding="utf-8"))
        if record.get("scored"):
            continue  # Already scored

        forecast_ts = datetime.fromisoformat(record["forecast_timestamp"])
        forecasts = record.get("forecasts", {})

        for config_key, fc in forecasts.items():
            horizon_years = fc.get("horizon_years", 0)
            horizon_end = forecast_ts + timedelta(days=horizon_years * 365.25)

            if horizon_end > now:
                continue  # Window not yet completed

            threshold = fc["threshold"]
            # Count actual events in [forecast_ts, horizon_end) above threshold
            actual_events = []
            for e in events:
                if e.origin_time_utc < forecast_ts or e.origin_time_utc >= horizon_end:
                    continue
                mag = e.mw if e.mw is not None else e.original_magnitude
                if mag is not None and mag >= threshold:
                    i_lat = int((e.latitude - 20.0) / 1.0)
                    i_lon = int((e.longitude - 88.0) / 1.0)
                    cell_idx = i_lat * 8 + i_lon
                    actual_events.append(cell_idx)

            # Per-cell binary outcomes
            y_true = np.zeros(64, dtype=float)
            for idx in actual_events:
                if 0 <= idx < 64:
                    y_true[idx] = 1.0

            # Per-cell forecast probabilities
            y_pred = np.array([c["probability"] for c in fc["cells"]])

            # Metrics
            brier = float(np.mean((y_pred - y_true) ** 2))
            eps = 1e-12
            f = np.clip(y_pred, eps, 1 - eps)
            log_lik = float(np.mean(y_true * np.log(f) + (1 - y_true) * np.log(1 - f)))
            n_positive = int(y_true.sum())

            # Save score
            fc["score"] = {
                "scored_at": now.isoformat(),
                "horizon_end": horizon_end.isoformat(),
                "n_actual_events": n_positive,
                "brier_score": round(brier, 6),
                "log_likelihood": round(log_lik, 6),
                "n_cells_with_events": n_positive,
            }

            scored.append({
                "ledger_file": ledger_file.name,
                "config": config_key,
                "forecast_time": forecast_ts.isoformat(),
                "horizon_end": horizon_end.isoformat(),
                "n_events": n_positive,
                "brier": round(brier, 6),
                "log_likelihood": round(log_lik, 6),
            })

        # Mark as scored and re-save (append score, don't modify forecast)
        record["scored"] = True
        record["scored_at"] = now.isoformat()
        ledger_file.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        logger.info("Scored: %s", ledger_file.name)

    return scored


def get_scoring_summary() -> dict:
    """Get a summary of all scored forecasts."""
    scores = []
    for ledger_file in sorted(LEDGER_DIR.glob("forecast_*.json")):
        record = json.loads(ledger_file.read_text(encoding="utf-8"))
        if not record.get("scored"):
            continue
        for config_key, fc in record.get("forecasts", {}).items():
            s = fc.get("score")
            if s:
                scores.append({
                    "config": config_key,
                    "forecast_time": record["forecast_timestamp"],
                    "brier": s["brier_score"],
                    "log_likelihood": s["log_likelihood"],
                    "n_events": s["n_actual_events"],
                })

    if not scores:
        return {"n_scored": 0, "message": "No completed forecast windows scored yet."}

    briers = [s["brier"] for s in scores]
    return {
        "n_scored": len(scores),
        "mean_brier": float(np.mean(briers)),
        "median_brier": float(np.median(briers)),
        "scores": scores,
    }
