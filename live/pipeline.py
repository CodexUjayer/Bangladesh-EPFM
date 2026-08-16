"""Live forecasting pipeline for FINAL_v1.0_FROZEN.

Fetches new earthquake data from USGS FDSN → deduplicates → runs the
frozen Spatial Poisson model → generates forecasts → saves to immutable ledger.

NEVER modifies the frozen model. NEVER overwrites historical forecasts.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import build_canonical_events, read_usgs_csv
from src.phase_c.isc_reader import read_isc_text
from src.completeness.mc import estimate_completeness
from src.baselines.poisson import HORIZON_YEARS
from src.ml.features import MLGridConfig
from src.ml.spatial_poisson import causal_spatial_rate, spatial_poisson_forecast
from src.baselines.uncertainty import poisson_rate_ci_garwood

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("live")

# === FROZEN MODEL PARAMETERS (DO NOT CHANGE) ===
MODEL_VERSION = "FINAL_v1.0_FROZEN"
FROZEN_MC = 4.13
FROZEN_B = 0.808
GRID = MLGridConfig()  # 1.0° grid, 64 cells
BBOX = (20.0, 28.0, 88.0, 96.0)
CATALOG_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
LEDGER_DIR = Path(__file__).resolve().parent / "forecast_ledger"
LEDGER_DIR.mkdir(parents=True, exist_ok=True)

# Forecast configurations
FORECAST_CONFIGS = [
    {"threshold": 4.5, "horizon": "7d"},
    {"threshold": 4.5, "horizon": "30d"},
    {"threshold": 5.0, "horizon": "7d"},
    {"threshold": 5.0, "horizon": "30d"},
]


def fetch_usgs_catalog(start_date: str, end_date: str) -> list:
    """Fetch earthquakes from USGS FDSN API."""
    url = (
        f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv"
        f"&starttime={start_date}&endtime={end_date}"
        f"&minmagnitude=2.5&minlatitude=20&maxlatitude=28"
        f"&minlongitude=88&maxlongitude=96&eventtype=earthquake&orderby=time-asc"
    )
    logger.info("Fetching USGS: %s", url[:80])
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "bangladesh-eq-forecast-live/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
        # Save to temp file for the reader
        tmp = CATALOG_DIR / "usgs" / "usgs_live_latest.csv"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(raw, encoding="utf-8")
        obs = read_usgs_csv(tmp)
        logger.info("USGS: fetched %d events", len(obs))
        return obs
    except Exception as e:
        logger.error("USGS fetch failed: %s", e)
        return []


def fetch_isc_catalog(start_date: str, end_date: str) -> list:
    """Fetch earthquakes from ISC FDSN API."""
    url = (
        f"http://www.isc.ac.uk/fdsnws/event/1/query?format=text"
        f"&starttime={start_date}&endtime={end_date}"
        f"&minlat=20&maxlat=28&minlon=88&maxlon=96&minmag=3.0&orderby=time-asc"
    )
    logger.info("Fetching ISC: %s", url[:80])
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "bangladesh-eq-forecast-live/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
        tmp = CATALOG_DIR / "isc" / "isc_live_latest.txt"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(raw, encoding="utf-8")
        obs = read_isc_text(tmp)
        logger.info("ISC: fetched %d events", len(obs))
        return obs
    except Exception as e:
        logger.error("ISC fetch failed: %s", e)
        return []


def load_existing_catalog() -> list:
    """Load the existing merged USGS+ISC catalog."""
    usgs_file = CATALOG_DIR / "usgs" / "usgs_bangladesh_1973_2025_m25.csv"
    isc_file = CATALOG_DIR / "isc" / "isc_bangladesh_1973_2025_m3.txt"
    obs = []
    if usgs_file.exists():
        obs.extend(read_usgs_csv(usgs_file))
    if isc_file.exists():
        obs.extend(read_isc_text(isc_file))
    return obs


def merge_catalogs(existing_obs: list, new_usgs: list, new_isc: list) -> list:
    """Merge existing + new observations using the project's canonical matching."""
    all_obs = existing_obs + new_usgs + new_isc
    events = build_canonical_events(all_obs, time_window_s=120.0, spatial_window_km=50.0)
    logger.info("Merged catalog: %d canonical events", len(events))
    return events


def generate_forecast(events: list, config: dict, catalog_start: datetime) -> dict:
    """Generate a forecast using the FROZEN Spatial Poisson model."""
    threshold = config["threshold"]
    horizon = config["horizon"]
    hy = HORIZON_YEARS[horizon]

    # Causal spatial rate
    rates = causal_spatial_rate(
        events, origin_time=datetime.now(timezone.utc), grid=GRID,
        threshold=threshold, catalog_start=catalog_start,
        method="expanding", smoothing="raw",
    )
    probs = spatial_poisson_forecast(rates, hy)

    # Per-cell output
    lats_grid, lons_grid = GRID.cell_centers()
    cells = []
    for i in range(GRID.n_cells):
        i_lat = i // GRID.n_lon
        i_lon = i % GRID.n_lon
        # Uncertainty (Garwood exact Poisson CI on the rate)
        n_cell = int(rates[i] * (datetime.now(timezone.utc) - catalog_start).total_seconds() / (365.25 * 86400))
        ci = poisson_rate_ci_garwood(n_cell, (datetime.now(timezone.utc) - catalog_start).total_seconds() / (365.25 * 86400))
        p_lo = 1.0 - math.exp(-ci[1] * hy)  # upper rate → lower P bound inverted
        p_hi = 1.0 - math.exp(-ci[0] * hy)  # Actually: higher rate = higher P
        p_lo = 1.0 - math.exp(-max(ci[0], 0) * hy)
        p_hi = 1.0 - math.exp(-ci[1] * hy)
        cells.append({
            "cell_id": f"cell_{i_lat:02d}_{i_lon:02d}",
            "lat_center": round(lats_grid[i_lat], 2),
            "lon_center": round(lons_grid[i_lon], 2),
            "rate_per_year": round(float(rates[i]), 6),
            "probability": round(float(probs[i]), 6),
            "probability_lower": round(max(float(p_lo), 0.0), 6),
            "probability_upper": round(min(float(p_hi), 1.0), 6),
        })

    # Regional summary
    total_rate = float(np.sum(rates))
    p_regional = 1.0 - math.exp(-total_rate * hy)

    return {
        "threshold": threshold,
        "horizon": horizon,
        "horizon_years": round(hy, 6),
        "regional_rate_per_year": round(total_rate, 4),
        "regional_probability": round(p_regional, 6),
        "n_cells": GRID.n_cells,
        "cells": cells,
    }


def save_forecast_ledger(forecast: dict, events: list) -> str:
    """Save forecast as an immutable record. Returns the filename."""
    import hashlib
    ts = datetime.now(timezone.utc)
    filename = f"forecast_{ts.strftime('%Y-%m-%d_%H%M%S')}.json"
    filepath = LEDGER_DIR / filename

    record = {
        "forecast_timestamp": ts.isoformat(),
        "model_version": MODEL_VERSION,
        "frozen_mc": FROZEN_MC,
        "frozen_b": FROZEN_B,
        "catalog_version": f"USGS+ISC merged, {len(events)} events",
        "catalog_snapshot": {
            "n_events": len(events),
            "time_range": [
                min(e.origin_time_utc for e in events).isoformat(),
                max(e.origin_time_utc for e in events).isoformat(),
            ],
        },
        "grid": {
            "cell_size_deg": GRID.cell_size_deg,
            "n_cells": GRID.n_cells,
            "bbox": list(BBOX),
        },
        "forecasts": forecast,
        "data_quality": {
            "usgs_fetched": len(new_usgs) if 'new_usgs' in dir() else 0,
            "isc_fetched": len(new_isc) if 'new_isc' in dir() else 0,
            "catalog_n_events": len(events),
        },
        "warnings": [
            "This is a PROBABILISTIC FORECAST, not deterministic earthquake prediction.",
            "Probabilities are not guarantees.",
            "Rare-event probabilities have substantial uncertainty.",
            f"Model: {MODEL_VERSION} (frozen; do not modify).",
            f"Mc={FROZEN_MC}, b={FROZEN_B}.",
        ],
    }

    # Compute hash of the forecast content (excluding hash, score, scored fields)
    # This must match the compute_forecast_hash function in prospective_scoring.py
    hash_content = {k: v for k, v in record.items()
                    if k not in ("forecast_hash", "score", "scored", "scored_at", "integrity_warning")}
    record["forecast_hash"] = hashlib.sha256(
        json.dumps(hash_content, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]

    filepath.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    logger.info("Forecast saved to immutable ledger: %s (hash: %s)", filename, record["forecast_hash"])
    return filename


def run_pipeline() -> dict:
    """Run the complete live forecasting pipeline.

    Returns a summary of what was done.
    """
    logger.info("=== LIVE PIPELINE START ===")
    ts_start = datetime.now(timezone.utc)

    # 1. Fetch new data
    now = datetime.now(timezone.utc)
    start_date = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = (now + timedelta(days=1)).strftime("%Y-%m-%d")

    new_usgs = fetch_usgs_catalog(start_date, end_date)
    new_isc = fetch_isc_catalog(start_date, end_date)

    # 2. Merge with existing catalog
    existing = load_existing_catalog()
    events = merge_catalogs(existing, new_usgs, new_isc)

    if not events:
        logger.error("No events in catalog; cannot generate forecast")
        return {"status": "error", "reason": "no events"}

    catalog_start = min(e.origin_time_utc for e in events)

    # 3. Generate forecasts
    forecasts = {}
    for config in FORECAST_CONFIGS:
        key = f"M{config['threshold']}_{config['horizon']}"
        logger.info("Generating forecast: %s", key)
        forecasts[key] = generate_forecast(events, config, catalog_start)

    # 4. Save to immutable ledger
    ledger_file = save_forecast_ledger(forecasts, events)

    # 5. Save latest forecast for dashboard (separate from ledger)
    latest = {
        "forecast_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION,
        "frozen_mc": FROZEN_MC,
        "frozen_b": FROZEN_B,
        "catalog_n_events": len(events),
        "catalog_start": catalog_start.isoformat(),
        "catalog_end": max(e.origin_time_utc for e in events).isoformat(),
        "forecasts": forecasts,
        "recent_earthquakes": [
            {
                "event_id": e.canonical_id,
                "time": e.origin_time_utc.isoformat(),
                "latitude": round(e.latitude, 4),
                "longitude": round(e.longitude, 4),
                "depth_km": round(e.depth_km, 1),
                "magnitude": round(e.mw if e.mw else e.original_magnitude, 1),
                "magnitude_type": e.original_magnitude_type,
                "sources": e.source_catalogs,
            }
            for e in sorted(events, key=lambda x: x.origin_time_utc, reverse=True)[:20]
        ],
        "warnings": [
            "This is a PROBABILISTIC FORECAST, not deterministic earthquake prediction.",
            "Probabilities are not guarantees.",
            "Rare-event probabilities have substantial uncertainty.",
            f"Model: {MODEL_VERSION} (frozen; do not modify).",
        ],
    }
    latest_path = Path(__file__).resolve().parent / "latest_forecast.json"
    latest_path.write_text(json.dumps(latest, indent=2, default=str), encoding="utf-8")
    logger.info("Latest forecast saved for dashboard: %s", latest_path)

    ts_end = datetime.now(timezone.utc)
    summary = {
        "status": "success",
        "pipeline_start": ts_start.isoformat(),
        "pipeline_end": ts_end.isoformat(),
        "duration_s": (ts_end - ts_start).total_seconds(),
        "usgs_fetched": len(new_usgs),
        "isc_fetched": len(new_isc),
        "catalog_n_events": len(events),
        "ledger_file": ledger_file,
        "model_version": MODEL_VERSION,
    }
    logger.info("=== LIVE PIPELINE COMPLETE ===")
    return summary


def get_latest_forecast() -> Optional[dict]:
    """Get the latest forecast for the dashboard."""
    latest_path = Path(__file__).resolve().parent / "latest_forecast.json"
    if latest_path.exists():
        return json.loads(latest_path.read_text(encoding="utf-8"))
    return None


def get_ledger_forecasts() -> list:
    """List all forecasts in the immutable ledger."""
    files = sorted(LEDGER_DIR.glob("forecast_*.json"), reverse=True)
    return [f.name for f in files]


if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps(result, indent=2))
