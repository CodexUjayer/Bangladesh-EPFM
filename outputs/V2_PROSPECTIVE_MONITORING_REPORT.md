# V2 Prospective Monitoring Report

## FINAL_v1.0_FROZEN vs FINAL_v2.0_CANDIDATE_BAYESIAN_SPATIAL

## 1. Objective

Deploy v2 as a parallel candidate forecast stream alongside v1. Both generate independent forecasts from the same catalog snapshot, scored against the same future observations. Collect genuine prospective evidence to determine whether Bayesian hierarchical spatial modeling improves reliability over the frozen Spatial Poisson model.

## 2. v1/v2 Architecture

| Component | v1 (Production) | v2 (Candidate) |
|-----------|----------------|----------------|
| Model | Spatial Poisson | Bayesian hierarchical Gamma-Poisson |
| Rate estimation | MLE: N_i / T | Posterior: Gamma(α+N_i, β+T) |
| Uncertainty | Garwood exact Poisson CI | Full posterior distribution |
| Hyperparameters | None | Empirical Bayes (α, β from cross-cell rates) |
| Count CI | Poisson CI on rate×Δt | Negative Binomial posterior predictive |
| Version | FINAL_v1.0_FROZEN | FINAL_v2.0_CANDIDATE_BAYESIAN_SPATIAL |
| Status | PRODUCTION | CANDIDATE — NOT PRODUCTION |

## 3. Immutable Forecast Design

- v1 ledger: `live/forecast_ledger/v1/`
- v2 ledger: `live/forecast_ledger/v2_bayesian/`
- Each forecast: `forecast_YYYY-MM-DD_HHMMSS.json` (never overwritten)
- SHA-256 hash for integrity verification
- Both models use the SAME catalog snapshot, timestamp, grid, thresholds, horizons

## 4. Prospective Scoring Methodology

### Probability calibration (binary)
- Brier score: mean((P_i - Y_i)²) where Y_i ∈ {0,1}
- Log-likelihood: Bernoulli
- ECE: 7-bin reliability
- Reliability diagrams

### Count-based posterior predictive coverage (CORRECTED)
The previous experiment incorrectly evaluated narrow probability intervals against binary outcomes (0% coverage). This was methodologically wrong.

**Corrected approach**: Compare observed event COUNTS per cell against posterior predictive count intervals.

For v2 (Bayesian): Posterior predictive on N ~ NegativeBinomial(r=α, p=β/(β+Δt))
- 50% interval: [NB.ppf(0.25), NB.ppf(0.75)]
- 80% interval: [NB.ppf(0.10), NB.ppf(0.90)]
- 90% interval: [NB.ppf(0.05), NB.ppf(0.95)]
- 95% interval: [NB.ppf(0.025), NB.ppf(0.975)]

For v1 (Poisson): Garwood exact CI on count = rate × Δt

Coverage = fraction of cells where observed count falls within the interval.

### Interval score (Gneiting & Raftery 2007)
IS = (hi - lo) + (2/α)·(lo - y)·I(y < lo) + (2/α)·(y - hi)·I(y > hi)

## 5. Uncertainty Methodology

**Clearly separated:**
- **Probability calibration**: evaluated via Brier, log-score, ECE, reliability diagrams
- **Posterior predictive count uncertainty**: evaluated via coverage at 50/80/90/95% levels + interval score

## 6. Promotion Criteria (Predefined)

v2 may replace v1 ONLY if ALL 12 conditions are met:
1. ≥20 evaluated forecast origins
2. No significant Brier degradation
3. No significant log-score degradation
4. Demonstrable calibration OR uncertainty improvement
5. Appropriate posterior predictive coverage (count-based)
6. No material sharpness loss
7. Stable across M≥4.5 and M≥5.0
8. Stable across 7d and 30d
9. No data leakage
10. No dependence on single unusual event
11. Results consistent over time
12. Survives multiple-comparison correction

## 7. Data Quality Controls
- USGS/ISC API availability monitored
- Forecast age tracked (>24h warning)
- Hash integrity verified (SHA-256)
- Catalog snapshot versioned

## 8. Current Evidence Level

- **Forecasts issued**: 1 (v1 and v2)
- **Forecasts evaluated**: 0
- **Evidence Level**: 0 (No completed prospective forecasts)
- **Status**: INSUFFICIENT PROSPECTIVE DATA — MONITORING CONTINUES

## 9. Current Results

No completed forecast windows. No Brier/log/ECE/coverage metrics available yet.

## 10. Limitations
- 0 completed prospective windows (cannot assess reliability)
- 7-day windows complete after 7 days; 30-day after 30 days
- Need ≥20 evaluated windows for promotion eligibility
- Rare-event coverage may be dominated by zero-count cells

## 11. Scientific Interpretation

The previous retrospective experiment found v2 ≈ v1 in predictive skill (ΔBrier ≈ 0) with marginally better calibration. The v2 model provides theoretically better uncertainty representation (full posterior vs point+CI). However, retrospective performance does NOT prove prospective reliability. This experiment will collect genuine prospective evidence.

## 12. Promotion Status

**V2 STATUS: PROSPECTIVE MONITORING**

v2 remains a CANDIDATE. No promotion decision will be made until ≥20 evaluated forecast windows exist and ALL 12 predefined criteria are independently verified.

**FINAL_v1.0_FROZEN remains the production model.**
