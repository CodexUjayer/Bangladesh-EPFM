# STAGE 4 — Statistical Baseline Layer (Poisson + Gutenberg-Richter)

> Generated 2026-08-09T07:45:52.065393+00:00.

## 0. Probability interpretation discipline

Throughout this report, the following are kept DISTINCT and never conflated:

- **Rate λ** (events per year): estimated as N / T. NOT a probability.
- **Expected count over horizon Δt**: λ × Δt. NOT a probability.
- **P(N ≥ 1 | Δt)**: 1 − exp(−λΔt). This is a probability.
- **Cell probability**: same formula with the cell's λ.
- Everything is **conditional on the observed catalog and the working Mc**.
- The word **'risk' is NOT used** (risk requires exposure/vulnerability, which is not modeled in Stage 4).

## 1. Catalog and thresholding

- Catalog file: `/home/z/my-project/bangladesh_eq_forecast/data/raw/usgs/usgs_bangladesh_1973_2025_m25.csv`
- Catalog version: usgs_bangladesh_1973_2025_m25 (USGS ComCat M>=2.5 query; floor M3.2)
- N events (full M≥2.5 query): 2,293
- Time range: 1973-02-10T04:25:29.700000+00:00 -> 2024-12-20T13:13:08.914000+00:00
- Exposure (catalog span): **51.86 years**
- **Working modeling threshold: M ≥ 4.5** (conservative working modeling threshold under the current USGS-only catalog limitations; NOT a definitively validated regional Mc).
- Mc uncertainty (from Stage 3 audit): working range **M3.5–4.5**; USGS ComCat floor ~M3.2; true Mc unverifiable below ~M3.5 without BMD/ISC.
- Mc sensitivity scenarios tested: [4.0, 4.5, 5.0]

## 2. Temporal Poisson baseline

Rate λ = N / T (events per year). P(N ≥ 1 | Δt) = 1 − exp(−λΔt).

| Threshold | N obs | Exposure (yr) | λ (1/yr) | 95% CI on λ | P(≥1) 24h | P(≥1) 7d | P(≥1) 30d | P(≥1) 90d | P(≥1) 1y |
|-----------|-------|---------------|----------|-------------|-----------|-----------|-----------|-----------|----------|
| M≥4.5 | 1987 | 51.86 | 38.3157 | [36.6493, 40.0383] | 0.0996 | 0.5202 | 0.9570 | 0.9999 | 1.0000 |
| M≥5.0 | 640 | 51.86 | 12.3412 | [11.4035, 13.3355] | 0.0332 | 0.2106 | 0.6371 | 0.9522 | 1.0000 |
| M≥5.5 | 96 | 51.86 | 1.8512 | [1.4995, 2.2606] | 0.0051 | 0.0349 | 0.1411 | 0.3663 | 0.8429 |
| M≥6.0 | 24 | 51.86 | 0.4628 | [0.2965, 0.6886] | 0.0013 | 0.0088 | 0.0373 | 0.1078 | 0.3705 |
| M≥6.5 | 9 | 51.86 | 0.1735 | [0.0794, 0.3294] | 0.0005 | 0.0033 | 0.0142 | 0.0419 | 0.1593 |
| M≥7.0 | 2 | 51.86 | 0.0386 | [0.0047, 0.1393] | 0.0001 | 0.0007 | 0.0032 | 0.0095 | 0.0378 |

**Notes:**
- Expected count (λΔt) is NOT shown in the probability columns; those are P(N ≥ 1).
- For small N, the 95% CI on λ spans a wide range; the probability CI (not shown in this table, see CSV) is correspondingly wide.
- M≥6.5: Small sample (N=9); rate and probability CIs are wide. Treat point estimates as indicative only.
- M≥7.0: Small sample (N=2); rate and probability CIs are wide. Treat point estimates as indicative only.

## 3. Gutenberg-Richter model (MLE)

Fitted by Aki-Utsu maximum-likelihood (NOT visual line fit). Shi-Bolt (1982) standard error; bootstrap 95% CI on b.

| Mc | b (MLE) | σ_b (Shi-Bolt) | b 95% CI | a (at Mc) | σ_a | N used | M range | Notes |
|----|----------|----------------|----------|-----------|-----|--------|---------|-------|
| 4.0 | 0.493 | 0.004 | [0.485, 0.501] | 5.331 | 0.019 | 2286 | 4.00-7.30 | Sensitivity scenario: Mc=4.0. Stage 3 established Mc is a working range (M3.5-4.5), NOT a validated threshold. |
| 4.5 | 0.951 | 0.015 | [0.923, 0.981] | 7.579 | 0.067 | 1985 | 4.51-7.30 | Sensitivity scenario: Mc=4.5. Stage 3 established Mc is a working range (M3.5-4.5), NOT a validated threshold. |
| 5.0 | 1.427 | 0.056 | [1.329, 1.538] | 9.939 | 0.281 | 639 | 5.00-7.30 | Sensitivity scenario: Mc=5.0. Stage 3 established Mc is a working range (M3.5-4.5), NOT a validated threshold. |

**Sensitivity interpretation:** We do NOT pick the Mc that gives the most attractive b-value. The three scenarios are reported jointly as a sensitivity analysis. The Stage 3 audit established Mc is a working range (M3.5-4.5), not a validated threshold; the b-value variation across Mc scenarios reflects this uncertainty.

**Per-scenario notes:**
- **Mc=4.0**: b=0.493 is anomalously LOW. This is because Mc=4.0 sits AT or BELOW the catalog's effective floor (USGS ComCat floor ~M3.2; Stage 3 working range M3.5-4.5). The FMD is truncated there, so the Aki-Utsu MLE (which assumes complete sampling above Mc) is BIASED. This b-value should NOT be used for extrapolation; it is reported only to show the sensitivity.
- **Mc=4.5**: b=0.951 is the WORKING estimate. This is the conservative modeling threshold; the FMD above 4.5 is robustly sampled in USGS ComCat. This is the b-value used for the primary Poisson/GR baseline.
- **Mc=5.0**: b=1.427 uses fewer events (more robust completeness) but loses the smaller-magnitude leverage. The higher b-value reflects the steeper tail when small events are excluded; this is a known MLE sensitivity, not necessarily a better estimate.

## 4. Mc sensitivity (probability of larger events)

How the estimated probability of M≥6.0 events (1-year horizon) changes under each Mc scenario, using the GR model extrapolated from each threshold:

| Mc scenario | b | N≥Mc | Predicted N≥6.0 | Predicted rate≥6.0 (1/yr) | P(≥1 M≥6.0 | 1yr) |
|------------|----|------|-----------------|---------------------------|----------------------|
| 4.0 | 0.493 | 2286 | 236.04 | 4.5516 | 0.9894 |
| 4.5 | 0.951 | 1985 | 74.22 | 1.4312 | 0.7610 |
| 5.0 | 1.427 | 639 | 23.92 | 0.4613 | 0.3695 |

**Interpretation:** Lower Mc uses more events but risks including incomplete data; higher Mc is more robust but uses fewer events. The spread in predicted P(≥1 M≥6.0 | 1yr) across scenarios is a direct measure of how much the Mc uncertainty propagates into the forecast.

## 5. Spatial baseline

- Grid: 1.0° resolution → 64 cells.
- Cells with ≥1 event above M≥4.5: 61 / 64
- Cells flagged low-statistics (N < 5): 20
- Exposure: 51.86 years.
- Coarse grid chosen deliberately; finer resolution is NOT automatically better and would inflate the number of low-statistics cells.

Top 10 cells by event count (M≥4.5):

| Cell | Lat | Lon | N | λ (1/yr) | 95% CI | λ density (1/km²/yr) | Mean M | Max M | Low-stat |
|------|-----|-----|---|----------|---------|----------------------|--------|--------|----------|
| cell_03_06 | 23.50 | 94.50 | 228 | 4.3966 | [3.8534, 4.9956] | 3.91e-04 | 4.89 | 6.90 | False |
| cell_04_06 | 24.50 | 94.50 | 205 | 3.9531 | [3.4394, 4.5225] | 3.52e-04 | 4.90 | 6.20 | False |
| cell_02_06 | 22.50 | 94.50 | 177 | 3.4131 | [2.9377, 3.9443] | 3.04e-04 | 4.85 | 5.90 | False |
| cell_04_07 | 24.50 | 95.50 | 99 | 1.9090 | [1.5603, 2.3136] | 1.70e-04 | 4.97 | 6.40 | False |
| cell_05_07 | 25.50 | 95.50 | 97 | 1.8705 | [1.5255, 2.2712] | 1.66e-04 | 4.90 | 7.30 | False |
| cell_03_05 | 23.50 | 93.50 | 94 | 1.8126 | [1.4734, 2.2076] | 1.61e-04 | 4.89 | 5.60 | False |
| cell_06_04 | 26.50 | 92.50 | 80 | 1.5427 | [1.2318, 1.9093] | 1.37e-04 | 4.89 | 6.00 | False |
| cell_01_06 | 21.50 | 94.50 | 73 | 1.4077 | [1.1119, 1.7592] | 1.25e-04 | 4.90 | 6.50 | False |
| cell_07_04 | 27.50 | 92.50 | 72 | 1.3884 | [1.0949, 1.7377] | 1.23e-04 | 4.91 | 5.54 | False |
| cell_02_05 | 22.50 | 93.50 | 66 | 1.2727 | [0.9928, 1.6084] | 1.13e-04 | 4.87 | 6.20 | False |

## 6. Spatial + magnitude forecast (example cells)

Full table in `outputs/stage4_probability_maps/`. Example: top 5 cells by P(≥1 M≥5.0 | 7d).

| Cell | Lat | Lon | N (M≥5.0) | λ (1/yr) | Expected (7d) | P(≥1 | 7d) | 95% UI | Low-stat |
|------|-----|-----|-----------|----------|---------------|-----------|---------|----------|
| cell_03_06 | 23.50 | 94.50 | 67 | 1.2920 | 0.0248 | 0.0245 | [0.0192, 0.0308] | False |
| cell_04_06 | 24.50 | 94.50 | 66 | 1.2727 | 0.0244 | 0.0241 | [0.0188, 0.0304] | False |
| cell_02_06 | 22.50 | 94.50 | 45 | 0.8677 | 0.0166 | 0.0165 | [0.0122, 0.0218] | False |
| cell_04_07 | 24.50 | 95.50 | 35 | 0.6749 | 0.0129 | 0.0129 | [0.0091, 0.0176] | False |
| cell_03_05 | 23.50 | 93.50 | 31 | 0.5978 | 0.0115 | 0.0114 | [0.0079, 0.0159] | False |

- **Expected count (λΔt) is NOT a probability** and is reported separately from P(≥1).
- Low-statistics cells have wide uncertainty intervals; their point probabilities are indicative only.

## 7. Large-event limitation (M≥6.5, M≥7.0)

For rare large events, ordinary frequentist precision is NOT achievable. We report N, exposure, rate CI under three priors (Garwood frequentist, Jeffreys Bayesian, Uniform Bayesian), and prior sensitivity.

| Threshold | N obs | Exposure (yr) | λ (1/yr) | Garwood 95% CI | Jeffreys 95% CI | Uniform 95% CI | Prior ratio | Sufficient? |
|-----------|-------|---------------|----------|----------------|------------------|-----------------|-------------|-------------|
| M≥6.5 | 9 | 51.86 | 0.1735 | [0.0794, 0.3294] | [0.0859, 0.3167] | [0.0925, 0.3294] | 0.961 | NO |
| M≥7.0 | 2 | 51.86 | 0.0386 | [0.0047, 0.1393] | [0.0080, 0.1237] | [0.0119, 0.1393] | 0.888 | NO |

**Notes:**
- M≥6.5: Only 9 events above M6.5 in 51.9 years. Frequentist precision is limited; Bayesian CIs (Jeffreys) are reported alongside the frequentist (Garwood) interval.
- M≥6.5: Prior sensitivity: Jeffreys vs uniform upper-bound ratio = 0.961 (close to 1 means prior-insensitive).
- M≥7.0: Only 2 event(s) observed above M7.0 in 51.9 years. Rate CI spans a large range; treat any point probability as indicative only, with wide uncertainty.
- M≥7.0: Prior sensitivity: Jeffreys vs uniform upper-bound ratio = 0.888 (close to 1 means prior-insensitive).

## 8. Chronological backtesting

Expanding-window chronological backtest. For each yearly origin (1995-2024), train on all events before the origin, forecast the next horizon, compare to observation. NO shuffling.

| Model | Threshold | Horizon | N origins | N positive | Base rate | Mean forecast P | Brier | Log-lik | IG vs climatology | ROC-AUC (sec.) | Cal. error |
|-------|-----------|---------|-----------|------------|-----------|-----------------|-------|---------|--------------------|-----------------|------------|
| temporal_poisson | M≥4.5 | 7d | 29 | 16 | 0.5517 | 0.4473 | 0.2538 | -0.7008 | -0.013 | 0.5817 | 0.1048 |
| temporal_poisson | M≥4.5 | 30d | 29 | 27 | 0.931 | 0.9192 | 0.0625 | -0.2384 | 0.0125 | 0.7222 | 0.0119 |
| temporal_poisson | M≥4.5 | 90d | 29 | 29 | 1.0 | 0.9994 | 0.0 | -0.0006 | -0.0006 | None | 0.0006 |
| temporal_poisson | M≥5.0 | 7d | 29 | 6 | 0.2069 | 0.2136 | 0.1639 | -0.5091 | 0.0007 | 0.5362 | 0.0067 |
| temporal_poisson | M≥5.0 | 30d | 29 | 15 | 0.5172 | 0.6427 | 0.2655 | -0.7257 | -0.0331 | 0.4905 | 0.1254 |
| temporal_poisson | M≥5.0 | 90d | 29 | 28 | 0.9655 | 0.9542 | 0.0336 | -0.1534 | -0.0034 | 0.2143 | 0.0113 |
| temporal_poisson | M≥5.5 | 30d | 29 | 4 | 0.1379 | 0.1417 | 0.1185 | -0.3994 | 0.0018 | 0.59 | 0.0038 |
| temporal_poisson | M≥6.0 | 90d | 29 | 2 | 0.069 | 0.1136 | 0.068 | -0.2713 | -0.0204 | 0.0741 | 0.0446 |

**Notes:**
- ROC-AUC is reported as a SECONDARY diagnostic only; for rare events it can be misleadingly high.
- Information gain is against the climatology (base rate) reference, the appropriate null for rare-event forecasting.
- Brier score and log-likelihood are the primary probabilistic metrics.
- M≥4.5, 7d: ROC-AUC is reported as a SECONDARY diagnostic only; for rare events it can be misleading and is not the primary measure.
- M≥4.5, 7d: Information gain is computed against the climatology (base rate) reference, which is the appropriate null for rare-event forecasting.
- M≥4.5, 30d: ROC-AUC is reported as a SECONDARY diagnostic only; for rare events it can be misleading and is not the primary measure.
- M≥4.5, 30d: Information gain is computed against the climatology (base rate) reference, which is the appropriate null for rare-event forecasting.
- M≥4.5, 90d: ROC-AUC is reported as a SECONDARY diagnostic only; for rare events it can be misleading and is not the primary measure.
- M≥4.5, 90d: Information gain is computed against the climatology (base rate) reference, which is the appropriate null for rare-event forecasting.
- M≥5.0, 7d: Only 6 positive observations; metric estimates have high variance. Treat with caution.
- M≥5.0, 7d: ROC-AUC is reported as a SECONDARY diagnostic only; for rare events it can be misleading and is not the primary measure.
- M≥5.0, 7d: Information gain is computed against the climatology (base rate) reference, which is the appropriate null for rare-event forecasting.
- M≥5.0, 30d: ROC-AUC is reported as a SECONDARY diagnostic only; for rare events it can be misleading and is not the primary measure.
- M≥5.0, 30d: Information gain is computed against the climatology (base rate) reference, which is the appropriate null for rare-event forecasting.
- M≥5.0, 90d: ROC-AUC is reported as a SECONDARY diagnostic only; for rare events it can be misleading and is not the primary measure.
- M≥5.0, 90d: Information gain is computed against the climatology (base rate) reference, which is the appropriate null for rare-event forecasting.
- M≥5.5, 30d: Only 4 positive observations; metric estimates have high variance. Treat with caution.
- M≥5.5, 30d: ROC-AUC is reported as a SECONDARY diagnostic only; for rare events it can be misleading and is not the primary measure.
- M≥5.5, 30d: Information gain is computed against the climatology (base rate) reference, which is the appropriate null for rare-event forecasting.
- M≥6.0, 90d: Only 2 positive observations; metric estimates have high variance. Treat with caution.
- M≥6.0, 90d: ROC-AUC is reported as a SECONDARY diagnostic only; for rare events it can be misleading and is not the primary measure.
- M≥6.0, 90d: Information gain is computed against the climatology (base rate) reference, which is the appropriate null for rare-event forecasting.

## 9. Baseline comparison table (summary)

| Model | Threshold | Horizon | Expected rate (1/yr) | P(≥1) | Brier | Log-lik | Calibration | 95% UI on P |
|-------|-----------|---------|----------------------|-------|-------|---------|-------------|-------------|
| temporal_poisson | M≥4.5 | 24h | 38.3157 | 0.0996 | — | — | — | [0.0955, 0.1038] |
| temporal_poisson | M≥4.5 | 7d | 38.3157 | 0.5202 | — | — | — | [0.5046, 0.5358] |
| temporal_poisson | M≥4.5 | 30d | 38.3157 | 0.9570 | — | — | — | [0.9507, 0.9627] |
| temporal_poisson | M≥4.5 | 90d | 38.3157 | 0.9999 | — | — | — | [0.9999, 0.9999] |
| temporal_poisson | M≥4.5 | 1y | 38.3157 | 1.0000 | — | — | — | [1.0000, 1.0000] |
| temporal_poisson | M≥5.0 | 24h | 12.3412 | 0.0332 | — | — | — | [0.0307, 0.0359] |
| temporal_poisson | M≥5.0 | 7d | 12.3412 | 0.2106 | — | — | — | [0.1963, 0.2255] |
| temporal_poisson | M≥5.0 | 30d | 12.3412 | 0.6371 | — | — | — | [0.6081, 0.6656] |
| temporal_poisson | M≥5.0 | 90d | 12.3412 | 0.9522 | — | — | — | [0.9398, 0.9626] |
| temporal_poisson | M≥5.0 | 1y | 12.3412 | 1.0000 | — | — | — | [1.0000, 1.0000] |
| temporal_poisson | M≥5.5 | 24h | 1.8512 | 0.0051 | — | — | — | [0.0041, 0.0062] |
| temporal_poisson | M≥5.5 | 7d | 1.8512 | 0.0349 | — | — | — | [0.0283, 0.0424] |
| temporal_poisson | M≥5.5 | 30d | 1.8512 | 0.1411 | — | — | — | [0.1159, 0.1695] |
| temporal_poisson | M≥5.5 | 90d | 1.8512 | 0.3663 | — | — | — | [0.3089, 0.4271] |
| temporal_poisson | M≥5.5 | 1y | 1.8512 | 0.8429 | — | — | — | [0.7768, 0.8957] |
| temporal_poisson | M≥6.0 | 24h | 0.4628 | 0.0013 | — | — | — | [0.0008, 0.0019] |
| temporal_poisson | M≥6.0 | 7d | 0.4628 | 0.0088 | — | — | — | [0.0057, 0.0131] |
| temporal_poisson | M≥6.0 | 30d | 0.4628 | 0.0373 | — | — | — | [0.0241, 0.0550] |
| temporal_poisson | M≥6.0 | 90d | 0.4628 | 0.1078 | — | — | — | [0.0705, 0.1561] |
| temporal_poisson | M≥6.0 | 1y | 0.4628 | 0.3705 | — | — | — | [0.2566, 0.4977] |
| temporal_poisson | M≥6.5 | 24h | 0.1735 | 0.0005 | — | — | — | [0.0002, 0.0009] |
| temporal_poisson | M≥6.5 | 7d | 0.1735 | 0.0033 | — | — | — | [0.0015, 0.0063] |
| temporal_poisson | M≥6.5 | 30d | 0.1735 | 0.0142 | — | — | — | [0.0065, 0.0267] |
| temporal_poisson | M≥6.5 | 90d | 0.1735 | 0.0419 | — | — | — | [0.0194, 0.0780] |
| temporal_poisson | M≥6.5 | 1y | 0.1735 | 0.1593 | — | — | — | [0.0763, 0.2807] |
| temporal_poisson | M≥7.0 | 24h | 0.0386 | 0.0001 | — | — | — | [0.0000, 0.0004] |
| temporal_poisson | M≥7.0 | 7d | 0.0386 | 0.0007 | — | — | — | [0.0001, 0.0027] |
| temporal_poisson | M≥7.0 | 30d | 0.0386 | 0.0032 | — | — | — | [0.0004, 0.0114] |
| temporal_poisson | M≥7.0 | 90d | 0.0386 | 0.0095 | — | — | — | [0.0012, 0.0337] |
| temporal_poisson | M≥7.0 | 1y | 0.0386 | 0.0378 | — | — | — | [0.0047, 0.1300] |
| temporal_poisson (backtested) | M≥4.5 | 7d | 38.3157 | 0.4473 | 0.2538 | -0.7008 | 0.1048 | [0.3662, 0.5288] |
| temporal_poisson (backtested) | M≥4.5 | 30d | 38.3157 | 0.9192 | 0.0625 | -0.2384 | 0.0119 | [0.8583, 0.9603] |
| temporal_poisson (backtested) | M≥4.5 | 90d | 38.3157 | 0.9994 | 0.0000 | -0.0006 | 0.0006 | [0.9972, 0.9999] |
| temporal_poisson (backtested) | M≥5.0 | 7d | 12.3412 | 0.2136 | 0.1639 | -0.5091 | 0.0067 | [0.1874, 0.2551] |
| temporal_poisson (backtested) | M≥5.0 | 30d | 12.3412 | 0.6427 | 0.2655 | -0.7257 | 0.1254 | [0.5891, 0.7170] |
| temporal_poisson (backtested) | M≥5.0 | 90d | 12.3412 | 0.9542 | 0.0336 | -0.1534 | 0.0113 | [0.9306, 0.9773] |
| temporal_poisson (backtested) | M≥5.5 | 30d | 1.8512 | 0.1417 | 0.1185 | -0.3994 | 0.0038 | [0.0995, 0.2031] |
| temporal_poisson (backtested) | M≥6.0 | 90d | 0.4628 | 0.1136 | 0.0680 | -0.2713 | 0.0446 | [0.0560, 0.2173] |

## 10. Reproducibility

All results record: catalog version, filtering threshold, Mc scenario, date range, geographic region, forecast horizon, model version, parameter estimates. See `outputs/stage4_model_metadata.json`.

**Artifacts:**
- `outputs/stage4_report.md` (this file)
- `outputs/stage4_baseline_results.csv` (flat results table)
- `outputs/stage4_probability_maps/` (per threshold×horizon cell forecasts)
- `outputs/stage4_backtest/` (per threshold×horizon backtest details)
- `outputs/stage4_model_metadata.json`

## 11. Scientific conclusion

**How much predictive skill can we obtain from historical seismicity alone, before adding aftershock triggering, physics-based stress, or machine learning?**

The temporal Poisson baseline captures only the **long-term average rate**. It has NO short-term time-dependence: after a large earthquake, the Poisson forecast for the next 7 days is identical to any other 7-day window. This is the central limitation that Stage 5 ETAS must address.

Specifically:
- The Poisson baseline's Brier score and log-likelihood reflect essentially the **climatology** (base rate). Information gain over climatology is expected to be ~0 by construction (Poisson rates are constant in time).
- The spatial baseline captures **where** events are more likely (Indo-Burman fold belt, Arakan megathrust) but not **when**.
- The GR model gives a **magnitude-distribution** prior, useful for extrapolating rates to larger magnitudes, but with large uncertainty for M≥6.5 where observations are few (Section 7).
- The Mc uncertainty (M3.5-4.5 working range) propagates into a non-trivial spread in b-value and hence in extrapolated large-event probabilities (Section 4).

**Expected Stage 5 improvement:** ETAS should beat this baseline primarily on **short-term horizons (24h, 7d) after mainshocks**, where Omori-law aftershock decay creates real time-dependence that Poisson cannot capture. On 90-day and 1-year horizons, the Poisson baseline may be competitive because aftershock sequences have largely decayed. Any ML model in Stage 7 must be evaluated against THIS baseline first; if it does not beat Poisson+Brier on the short horizons, it adds no skill.

**This is the bar Stage 5 must clear.**