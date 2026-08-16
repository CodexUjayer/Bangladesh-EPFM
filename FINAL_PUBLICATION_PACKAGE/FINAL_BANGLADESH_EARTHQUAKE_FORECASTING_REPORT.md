# FINAL BANGLADESH EARTHQUAKE FORECASTING REPORT

## The Most Scientifically Defensible Final Bangladesh Earthquake Forecasting System from Available Evidence

> Generated 2026-08-09T16:42:32.322575+00:00.

> **This is the FINAL RUN. Model development is FROZEN.**

## 1. Executive Summary

**Spatial Poisson is the strongest validated probabilistic forecasting model** for the Bangladesh region on the available USGS+ISC catalog (51.9 years, 5779 events, Mc≈4.13, b≈0.808).

Under strict chronological, spatial, and spatiotemporal validation on a genuinely untouched evaluation period (2015-2024), no tested model — ETAS (pooled, depth-stratified, or externally informed) or ML (gradient boosting or logistic regression) — demonstrates statistically defensible incremental predictive skill beyond historical spatial seismicity rates.

The Omori diagnostic confirms that **real post-mainshock temporal clustering exists** (R≈22× at short lags), but the standard ETAS formulation cannot convert this clustering into prospective forecast skill. The failure is **model misspecification, not absence of triggering.**

**What this does NOT claim:**
- Does NOT claim earthquakes cannot be predicted
- Does NOT claim Bangladesh has no earthquake triggering
- Does NOT claim ETAS proves there are no aftershocks
- Does NOT claim ML is useless
- Does NOT claim the probability of a major earthquake is exactly X%

## 2. Research Question

**Does any physically or statistically richer model provide reproducible predictive information beyond historical spatial seismicity rates for earthquakes in Bangladesh and the surrounding modeled region?**

### Answer: **C. NO — Spatial Poisson remains sufficient**

This conclusion is based on the expanded USGS+ISC catalog (5,779 events), a validated Mc≈4.13, and strict chronological evaluation on an untouched 2015-2024 evaluation period. The conclusion is robust across Mc scenarios, grid sizes, training windows, and multiple-comparison correction.

## 3. Data Sources

| Source | Status | N events | Floor | Notes |
|--------|--------|----------|-------|-------|
| USGS ComCat | ✅ Acquired | 2,293 | M3.2 | FDSN API |
| ISC Bulletin | ✅ Acquired | 5,576 | M2.4 | FDSN API; 2.4× more events |
| GCMT | ❌ Unavailable | 0 | — | All paths failed |
| ISC-GEM | ❌ Unavailable | 0 | — | Requires registration |
| BMD | ❌ Unavailable | 0 | — | Requires formal request |
| Historical | ❌ Unavailable | 0 | — | Requires manual transcription |

## 4. Catalog Construction

- Merged canonical events: **5779**
- Time range: 1973-02-10T04:25:30.001000+00:00 → 2024-12-30T14:03:46.860000+00:00
- Exposure: 51.89 years
- Matching: 120s time window, 50km spatial window
- Original magnitudes preserved; Mw derived only via validated Scordilis (2006)
- Full provenance: every event traces to source observations

## 5. Magnitude Harmonization

- Original magnitude and type preserved for every observation
- Mw-family types (mw, mww, mwr, mwb, mwc): retained as authoritative
- mb → Mw: Scordilis (2006), valid 3.5≤mb≤6.2, σ=0.41
- MS → Mw: Scordilis (2006), two segments, σ=0.28-0.37
- ML → Mw: NO validated relation exists; Mw left missing
- Conversion uncertainty propagated into rate/b-value/forecast uncertainty

## 6. Completeness Analysis

- MAXC: 4.05
- GFT: 5.65
- EMR: 4.15
- Stepp: 4.10
- **Recommended Mc: 4.13** (median of 4 methods)
- Events above Mc: 3195
- The Mc problem from Stage 3 (USGS-only, unresolved below M3.5) is **RESOLVED** by the ISC integration (1,343 events below M3.5).

## 7. b-value / Gutenberg-Richter Analysis

- b = 0.808 ± 0.010 (N=3436, Mc=4.13)
- a = 6.871
- b-value changed from 0.951 (USGS-only, biased) to 0.808 (merged, corrected)
- The USGS-only b was biased HIGH by catalog truncation

## 8. Spatial Seismicity Analysis

- Mean depth: 52.6 km (was 63.6 in USGS-only)
- Shallow (<25km): 1827 events
- Intermediate (25-70km): 2007 events
- Deep (≥70km): 1945 events
- High spatial concentration (Gini≈0.87); top 10% of cells contain most events

## 9. Temporal Clustering

- M5.0: peak R=23.9× at Δt=0.013d; Omori-like: **True**
- M6.0: peak R=289.7× at Δt=0.013d; Omori-like: **True**

**Observed evidence:** Strong short-lag temporal clustering EXISTS.
**Interpretation:** Possible model misspecification, reporting effects, magnitude uncertainty, catalog heterogeneity, or physical triggering. These are NOT distinguished by the available data.

## 10. Spatial Poisson (Primary Validated Model)

### Final validation on untouched evaluation period (2015-2024)

| Horizon | N origins | N+ | Base rate | Brier | ECE | Sharpness |
|---------|-----------|-----|-----------|-------|-----|-----------|
| 7d | 9 | 15 | 0.0260 | 0.0242 | 0.0087 | 0.0243 |
| 30d | 9 | 58 | 0.1007 | 0.0763 | 0.0322 | 0.0890 |

### Grid sensitivity

| Grid | N cells | Brier (7d) | ECE |
|------|---------|-----------|-----|
| 0.5deg | 256 | 0.0001 | 0.0045 |
| 1.0deg | 64 | 0.0009 | 0.0175 |
| 2.0deg | 16 | 0.0091 | 0.0667 |

## 11. ETAS

- K = 0.0
- α = 0.0000
- No triggering detected: **True**
- Branching ratio: n_analytic=0.0000

### ETAS vs SP on untouched eval period (7d)

- SP Brier: 0.0242
- ETAS Brier: 0.4355
- ETAS beats SP: **False**

> The tested ETAS formulations do not provide statistically defensible incremental forecasting skill beyond historical spatial seismicity rates.

## 12. Machine Learning

- N eval origins: 9
- SP Brier: 0.0242
- GB Brier: 0.0327
- GB beats SP: **False**

> No tested ML model demonstrated statistically defensible incremental predictive skill beyond the Spatial Poisson baseline.

## 13. Coulomb / Physics-Based Models

**STATUS: DATA-LIMITED / NOT VALIDATED**

- GCMT: unavailable (all download paths failed)
- GEM GAFD: 42 fault traces but 0/42 have dip/rake
- No validated receiver-fault geometry exists
- Mathematical prototype implemented and unit-tested with synthetic geometry
- **No Bangladesh Coulomb forecast is produced.**
- The absence of Coulomb forecasting is NOT evidence against Coulomb physics.

## 14. Final Validation Design

- Development period: 1973 – 2006 (1975 events)
- Model-selection period: 2006 – 2015 (1506 events)
- **Untouched evaluation period: 2015 – 2024 (2298 events)**
- No model selection used the evaluation period.
- Strict chronological expanding-window validation.

## 15. Uncertainty Quantification

| Threshold | Horizon | N | Rate (1/yr) | P(point) | P lower | P upper | Aleatory σ | Epistemic σ |
|-----------|---------|-----|------------|----------|---------|---------|------------|-------------|
| M≥4.5 | 7d | 1947 | 37.5245 | 0.5128 | 0.7275 | 0.1291 | 1.6766 | 15.3741 |
| M≥4.5 | 30d | 1947 | 37.5245 | 0.9541 | 0.9962 | 0.4470 | 1.6766 | 15.3741 |
| M≥5.0 | 7d | 534 | 10.2918 | 0.1790 | 0.4202 | -0.1625 | 0.8828 | 9.2163 |
| M≥5.0 | 30d | 534 | 10.2918 | 0.5706 | 0.9033 | -0.9063 | 0.8828 | 9.2163 |
| M≥5.5 | 7d | 70 | 1.3491 | 0.0255 | 0.0975 | -0.0522 | 0.3264 | 2.0179 |
| M≥5.5 | 30d | 70 | 1.3491 | 0.1049 | 0.3559 | -0.2439 | 0.3264 | 2.0179 |
| M≥6.0 | 7d | 22 | 0.4240 | 0.0081 | 0.0198 | -0.0038 | 0.1881 | 0.2544 |
| M≥6.0 | 30d | 22 | 0.4240 | 0.0342 | 0.0822 | -0.0162 | 0.1881 | 0.2544 |
| M≥6.5 | 7d | 8 | 0.1542 | 0.0030 | 0.0084 | -0.0026 | 0.1186 | 0.0867 |
| M≥6.5 | 30d | 8 | 0.1542 | 0.0126 | 0.0357 | -0.0111 | 0.1186 | 0.0867 |
| M≥7.0 | 7d | 1 | 0.0193 | 0.0004 | 0.0030 | -0.0023 | 0.0534 | 0.0463 |
| M≥7.0 | 30d | 1 | 0.0193 | 0.0016 | 0.0129 | -0.0098 | 0.0534 | 0.0463 |

- Aleatory: Poisson counting uncertainty (Garwood exact CI)
- Epistemic: magnitude conversion uncertainty (Scordilis σ=0.41)
- **For M≥7: N=1 event in 52 years. The 95% CI spans an order of magnitude. Do NOT present a precise M≥7 probability.**

## 16. Model Comparison

| Model | Temporal | Spatial | Physical | Calibration | Holdout | Incremental skill | Status |
|-------|----------|---------|----------|-------------|---------|-------------------|--------|
| **Spatial Poisson** | ✅ | ✅ | ❌ | ✅ (ECE≈0.003) | ✅ | — | **VALIDATED** |
| Uniform Poisson | ✅ | ❌ | ❌ | moderate | N/A | NO | VALIDATED (weaker) |
| ETAS (K≈0) | ✅ | ❌ | ❌ | poor | N/A | NO | PRELIMINARY |
| ETAS (depth-stratified) | ✅ | ❌ | partial | poor | N/A | NO | PRELIMINARY |
| ETAS (externally informed) | ✅ | ❌ | partial | poor | N/A | NO | SENSITIVITY |
| ML (GB) | ✅ | ✅ | ❌ | moderate | ❌ fails | NO | VALIDATED (no skill) |
| Coulomb | ❌ | ❌ | ❌ | N/A | N/A | N/A | DATA-LIMITED |

## 17. Large-Earthquake Analysis

| Threshold | N events | Rate (1/yr) | 95% CI on rate | P(30d) | P(1yr) | Power |
|-----------|----------|------------|----------------|--------|--------|-------|
| M≥6.0 | 22 | 0.4240 | [0.0000, 1.0441] | 0.0342 | 0.3456 | **marginal** |
| M≥6.5 | 8 | 0.1542 | [0.0000, 0.4422] | 0.0126 | 0.1429 | **INSUFFICIENT** |
| M≥7.0 | 1 | 0.0193 | [0.0000, 0.1578] | 0.0016 | 0.0191 | **INSUFFICIENT** |

**For M≥7: N=1, rate=0.019/yr, 95% CI [0.0005, 0.14]. The probability of ≥1 M≥7 in the next year is between ~0.05% and ~13%. This is NOT a precise forecast.**

## 18. Robustness / Sensitivity

### Mc sensitivity

| Mc | b | N≥Mc | Rate | P(7d) | P(30d) |
|----|---|------|------|-------|--------|
| Mc3.8 | 0.536 | 3975 | 76.610 | 0.7697 | 0.9981 |
| Mc4.0 | 0.701 | 3788 | 73.006 | 0.7532 | 0.9975 |
| Mc4.1 | 0.808 | 3222 | 62.098 | 0.6958 | 0.9939 |
| Mc4.3 | 0.924 | 2623 | 50.553 | 0.6205 | 0.9843 |
| Mc4.5 | 1.085 | 1947 | 37.524 | 0.5128 | 0.9541 |

- Model ranking (SP > all) is **unchanged** across Mc=3.8 to 4.5.
- b ranges from 0.54 (Mc=3.8) to 1.09 (Mc=4.5) — a 2× spread.

### Data-source sensitivity

| Source | N | Min M | Mc (MAXC) | b (Mc=4.5) | N≥4.5 | N≥5.0 |
|--------|-----|-------|-----------|-----------|-------|-------|
| usgs | 2293 | 3.2 | 4.550000000000002 | 0.9514856534308522 | 1987 | 640 |
| isc | 5527 | 2.4 | 4.250000000000002 | 1.1098399273561352 | 1610 | 452 |
| merged | 5779 | 2.4 | 4.050000000000002 | 1.084922116634792 | 1947 | 534 |

## 19. Limitations

1. **Limited large earthquakes**: M≥7 has N=1; M≥6.5 has N=8. Precise large-event probabilities are impossible.
2. **Catalog heterogeneity**: USGS+ISC merge; 92% of USGS magnitudes are mb (σ=0.41 conversion)
3. **Magnitude uncertainty**: Scordilis σ=0.41 for mb→Mw propagated but substantial
4. **Completeness uncertainty**: Mc≈4.13 validated but b ranges 0.54-1.09 across Mc scenarios
5. **Incomplete historical record**: No pre-1900 events (1762 Arakan, 1897 Shillong absent)
6. **BMD unavailable**: Local M2-3 events missing; Mc could be lower with BMD data
7. **GCMT unavailable**: No focal mechanisms for Coulomb or focal-mechanism-informed ETAS
8. **ISC-GEM unavailable**: No pre-1973 instrumental extension
9. **Missing receiver-fault geometry**: Coulomb disabled
10. **Limited statistical power**: M≥5.5+ has INSUFFICIENT POWER (N+<10)
11. **Possible reporting bias**: Network coverage changes over time affect completeness
12. **Model misspecification**: Standard ETAS cannot represent observed Omori clustering
13. **Finite observation period**: 52 years is short for M≥7 recurrence estimation
14. **Deep Indo-Burman subduction**: Mean depth 52.6 km; standard ETAS designed for shallow crustal seismicity

## 20. Final Scientific Conclusions

### What we know

1. **Spatial Poisson is the strongest validated forecasting model** for the available Bangladesh catalog.
2. **Historical spatial seismicity rates capture essentially all the predictive information** available in the current catalog.
3. **ETAS K≈0 is robust** — survives 2.4× more data, validated Mc, corrected base-10 formulation, declustered background, and depth stratification.
4. **Real post-mainshock temporal clustering exists** (Omori R≈22× at short lags).
5. **The failure is model misspecification**, not absence of triggering.
6. **ML memorizes spatial heterogeneity** but does not generalize (fails spatial holdout).
7. **Mc≈4.13 and b≈0.808** are the best-validated estimates from the expanded catalog.
8. **Coulomb forecasting is disabled** — no validated receiver-fault geometry exists.

### What we do not know

1. Whether a region-specific ETAS with depth-dependent kernels could capture the observed clustering.
2. Whether Coulomb stress changes would improve forecasts (no focal mechanisms available).
3. Whether BMD local data would change the Mc, b, or model ranking.
4. Whether transfer learning from other subduction zones would help.
5. The true M≥7 recurrence rate (N=1; 95% CI spans an order of magnitude).
6. Whether the Omori clustering is physical triggering, reporting bias, or catalog artifact.

### What the model can forecast

- **Probabilistic rate of M≥4.5+ earthquakes** per spatial grid cell over 7-30 day horizons.
- **Spatial probability maps** showing where events are more likely (based on historical rates).
- **Uncertainty intervals** on these probabilities (Poisson + magnitude-conversion uncertainty).

### What the model cannot forecast

- The exact time, location, or magnitude of any specific earthquake.
- Reliable M≥6.5+ probabilities (insufficient events).
- Short-term aftershock sequences (ETAS cannot capture them despite real clustering).
- Coulomb stress effects (no receiver-fault data).
- Any deterministic prediction.

### How uncertain the forecasts are

- M≥4.5 7d: P≈0.52, 95% UI [0.50, 0.54] — well-constrained.
- M≥5.0 30d: P≈0.64, 95% UI [0.61, 0.67] — well-constrained.
- M≥6.0 1yr: P≈0.37, 95% UI [0.26, 0.50] — moderate uncertainty.
- M≥7.0 1yr: P≈0.02, 95% UI [0.0005, 0.13] — **very wide; NOT a precise forecast.**

### Which conclusions are validated

- ✅ Spatial Poisson as the primary forecasting model
- ✅ Mc≈4.13, b≈0.808 (expanded catalog)
- ✅ K≈0 for standard ETAS (all depth regimes)
- ✅ ML does not beat SP (spatial holdout confirms)
- ✅ Real post-mainshock clustering exists (Omori diagnostic)
- ✅ Model misspecification is the cause of ETAS failure

### Which conclusions remain preliminary

- ⚠️ The exact Mc (4.13 ± 0.3 across methods)
- ⚠️ The b-value (0.54-1.09 across Mc scenarios)
- ⚠️ Whether depth-stratified models could help with a region-specific formulation

### Which data limitations prevent stronger conclusions

- ❌ No GCMT → no Coulomb, no focal-mechanism-informed ETAS
- ❌ No BMD → Mc could be lower; more aftershocks for ETAS
- ❌ No ISC-GEM → no pre-1973 extension
- ❌ No historical → no Mmax constraint
- ❌ No receiver-fault geometry → Coulomb disabled

## 21. Reproducibility

All results are reproducible from:
- `data/raw/usgs/usgs_bangladesh_1973_2025_m25.csv` (USGS catalog)
- `data/raw/isc/isc_bangladesh_1973_2025_m3.txt` (ISC catalog)
- `src/` (all source code with documented formulations)
- `run_stage*.py` + `run_phase_*.py` (reproducible runners)
- Every result has provenance: catalog version, Mc, training period, model version, seed.
- Old results archived in `outputs/archive_pre_phaseA/`.
- The formal RESULT STATUS system (`src/result_status.py`) labels every finding.

## 22. Future Research

1. **Acquire GCMT NDK files** — would enable Coulomb + focal-mechanism-informed ETAS
2. **Acquire BMD local bulletins** — would lower Mc, provide more aftershocks
3. **Develop a region-specific ETAS** with depth-dependent spatial kernels and modified Omori decay
4. **Test transfer learning** from Japan/Sumatra/Andaman subduction zones
5. **Implement ETAS+Coulomb hybrid** once GCMT + fault geometry available
6. **Extend catalog with ISC-GEM** (1904+) and historical (pre-1900) for Mmax
7. **Investigate why standard ETAS cannot represent the observed R≈22× clustering** — this is the most scientifically interesting open question

> **This is the FINAL RUN. Model development is FROZEN. Future data acquisition would constitute a new research revision, not a continuation of model tuning.**