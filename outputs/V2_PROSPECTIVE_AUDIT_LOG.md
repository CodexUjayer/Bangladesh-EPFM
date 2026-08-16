# V2 Prospective Audit Log

## Independent Audit Trail for v1 vs v2 Parallel Prospective Monitoring

Every forecast and score is recorded with full provenance to enable reconstruction.

## Forecast Records

### v1 (Production) Ledger: `live/forecast_ledger/v1/`

| File | Timestamp | Model | Catalog N | Hash | Scored |
|------|-----------|-------|-----------|------|--------|
| forecast_2026-08-11_082108.json | 2026-08-11T08:21:08 | FINAL_v1.0_FROZEN | 5781 | (pre-hash) | No |
| forecast_2026-08-12_091636.json | 2026-08-12T09:16:36 | FINAL_v1.0_FROZEN | 5781 | ✓ | No |

### v2 (Candidate) Ledger: `live/forecast_ledger/v2_bayesian/`

| File | Timestamp | Model | Catalog N | Hash | Scored |
|------|-----------|-------|-----------|------|--------|
| forecast_2026-08-12_091636.json | 2026-08-12T09:16:36 | FINAL_v2.0_CANDIDATE | 5781 | ✓ | No |

## Scoring Methodology

- **Scoring version**: v2_prospective_v1.0
- **Binary metrics**: Brier, log-likelihood, ECE (7-bin), sharpness, hit/FA/miss/CN
- **Count coverage**: 50/80/90/95% posterior predictive intervals (NegativeBinomial for v2; Garwood for v1)
- **Interval score**: Gneiting & Raftery (2007) proper interval score
- **Bootstrap**: 500 resamples over forecast origins, seed=42

## Audit Trail Fields

Each forecast record contains:
- `forecast_timestamp`: when the forecast was issued
- `model_version`: FINAL_v1.0_FROZEN or FINAL_v2.0_CANDIDATE_BAYESIAN_SPATIAL
- `catalog_n_events`: number of events in catalog snapshot
- `catalog_start`: earliest event time in catalog
- `forecast_hash`: SHA-256 of forecast content (excluding hash/score fields)
- `frozen_mc`: 4.13 (immutable)
- `frozen_b`: 0.808 (immutable)
- `grid`: 1.0° × 1.0° (64 cells, immutable)
- `forecasts`: per-config cell probabilities and uncertainty
- `scored`: boolean (set when window completes and score is appended)
- `scored_at`: timestamp of scoring
- `warnings`: model status warnings

## Integrity Rules

1. Forecasts are NEVER overwritten (unique timestamp filenames)
2. Scores are APPENDED (never replacing original forecast values)
3. Hash is computed BEFORE scoring (scoring doesn't break integrity)
4. Both v1 and v2 use the SAME catalog snapshot
5. Both v1 and v2 are scored against the SAME future observations
6. No model tuning is permitted during prospective monitoring
7. If a methodological issue is discovered, create FINAL_v2.1_CANDIDATE separately

## Current Status

- v1 forecasts issued: 2
- v2 forecasts issued: 1
- Windows evaluated: 0
- Evidence level: 0
- V2 STATUS: PROSPECTIVE MONITORING
