# LIVE SYSTEM README — Bangladesh Earthquake Forecasting

## FINAL_v1.0_FROZEN — Model Development CLOSED

This is the live deployment of the frozen Bangladesh earthquake forecasting system. The scientific model is immutable and will not be modified.

## What This System Does

1. **Fetches** new earthquake data from USGS and ISC FDSN APIs
2. **Merges** with the existing catalog (5,779+ events) using canonical event matching
3. **Runs** the frozen Spatial Poisson model to generate probabilistic forecasts
4. **Saves** every forecast to an immutable ledger (never overwrites)
5. **Serves** an interactive web dashboard with probability maps and recent earthquakes
6. **Scores** completed forecast windows against actual outcomes (prospective evaluation)

## Architecture

```
USGS FDSN API ─┐
               ├─→ Pipeline (Python) ─→ Frozen Spatial Poisson ─→ Forecast JSON
ISC FDSN API ─┘         │                    (FINAL_v1.0_FROZEN)       │
                        │                                               ├──→ Immutable Ledger
                        └──→ latest_forecast.json ──→ Next.js API ──→ Dashboard
                                                        │
                                                        └──→ Prospective Scoring
```

## How to Run

### 1. Start the Dashboard

```bash
cd /home/z/my-project
npx next dev
```

The dashboard will be available at `http://localhost:3000`.

### 2. Run the Forecast Pipeline (Manual)

```bash
cd /home/z/my-project/bangladesh_eq_forecast
python3 live/pipeline.py
```

This fetches new data, generates forecasts, and saves to the ledger.

### 3. Trigger Update from Dashboard

Click the "Update Now" button on the dashboard. This calls the API endpoint `/api/update` which runs the Python pipeline.

### 4. Automatic Updates (Cron)

```bash
# Add to crontab for hourly updates:
0 * * * * cd /home/z/my-project/bangladesh_eq_forecast && python3 live/pipeline.py >> /tmp/eq_pipeline.log 2>&1
```

## Dashboard Features

- **Probability Map**: 8×8 grid (1° cells) colored by P(≥1 event) per cell
- **Threshold Selector**: M≥4.5 or M≥5.0
- **Horizon Selector**: 7 days or 30 days
- **Cell Inspection**: Click any cell for detailed probability and uncertainty
- **Recent Earthquakes**: Latest 20 events with magnitude, location, depth, source
- **Model Information**: Version, Mc, b-value, catalog size, data sources
- **Warnings**: Prominent disclaimers about probabilistic vs deterministic forecasting

## Forecast Outputs

Each forecast provides per-cell:
- `probability`: P(≥1 event ≥ M_threshold in horizon)
- `rate_per_year`: Expected earthquake rate
- `probability_lower` / `probability_upper`: 95% uncertainty interval
- `forecast_start` / `forecast_end`: Time window
- `model_version`: FINAL_v1.0_FROZEN
- `catalog_version`: Source and event count

## Immutable Forecast Ledger

Location: `bangladesh_eq_forecast/live/forecast_ledger/`

Each forecast is saved as `forecast_YYYY-MM-DD_HHMMSS.json` and is NEVER overwritten. This enables genuine prospective evaluation.

## Data Sources

| Source | Status | Update Frequency |
|--------|--------|-------------------|
| USGS ComCat | ✅ Live (FDSN API) | Every pipeline run |
| ISC Bulletin | ✅ Live (FDSN API) | Every pipeline run |
| GCMT | ❌ Unavailable | — |
| BMD | ❌ Unavailable | — |

## Safety Warnings

- This system does NOT predict the exact time or location of earthquakes
- Probabilities are NOT guarantees
- Rare-event probabilities have substantial uncertainty
- M≥6.5+ forecasts should not be presented as precise
- This is NOT an official warning system
- The model is based on available catalog data and known limitations

## Model Version

**FINAL_v1.0_FROZEN** — Spatial Poisson
- Mc ≈ 4.13
- b ≈ 0.808
- Grid: 1.0° × 1.0° (64 cells)
- Catalog: USGS + ISC merged (5,779+ events)

If future research produces an improved model, it becomes `FINAL_v2.0` and must be evaluated independently against v1.0.
