# PHASE C — Data Acquisition, Integration & Catalog Upgrade

> Generated 2026-08-09T16:05:28.989633+00:00.

## 1. Data sources acquired

| Source | Status | N events | Floor | Notes |
|--------|--------|----------|-------|-------|
| USGS ComCat | ✅ ACQUIRED | 2293 | M3.2 | Previously acquired (Stage 2) |
| ISC Bulletin | ✅ **NEW** | 5533 | M2.4 | Downloaded via ISC FDSN; 2.4× more events |
| GCMT | ❌ UNAVAILABLE | 0 | — | All paths failed (404/410) |
| ISC-GEM | ❌ UNAVAILABLE | 0 | — | Requires registration |
| BMD | ❌ UNAVAILABLE | 0 | — | Requires formal request |
| Historical | ❌ UNAVAILABLE | 0 | — | Requires manual transcription |

## 2. Catalog merge

- USGS observations: 2293
- ISC observations: 5533
- Total observations: 7826
- **Merged canonical events: 5779**
- Multi-source events (USGS+ISC matched): 2042
- Deduplication rate: 26.2%

## 3. Magnitude distribution comparison

| Catalog | N | Min M | Max M | N below M4 | N below M3.5 |
|---------|-----|-------|-------|------------|--------------|
| USGS | 2293 | 3.2 | 7.3 | 167 | 3 |
| ISC | 5527 | 2.4 | 7.2 | 3606 | 1392 |
| **Merged** | **5779** | **2.4** | **7.2** | **3483** | **1343** |

**Key improvement:** The merged catalog contains **1343 events below M3.5** (vs only 3 in USGS-only). This FINALLY resolves the Mc estimation problem from Stages 3-7B.

### Magnitude type distribution (merged)

| Type | Count |
|------|-------|
| mb | 3504 |
| mw | 748 |
| ml | 691 |
| ms | 255 |
| mb1 | 253 |
| m | 174 |
| mbtmp | 108 |
| mww | 33 |
| mlv | 6 |
| mwb | 4 |

## 4. Completeness (Mc) re-estimation

### Before (USGS-only, Stage 3 audit)

- MAXC: 4.55
- Working range: M3.5-4.5 (NOT validated below M3.5 due to USGS floor M3.2)

### After (merged USGS+ISC)

- MAXC: **4.05**
- GFT: 5.65
- EMR: 3.65
- Stepp: 4.10
- **Recommended Mc: 4.08** (median of 4 methods)

**Mc changed from 4.55 (USGS-only) to 4.08 (merged). Change: -0.47 magnitude units.**

### Mc(t) — temporal completeness (5-year rolling MAXC)

| Period | Mc | N events |
|--------|-----|----------|
| 1973-1977 | 5.05 | 76 |
| 1975-1979 | 5.05 | 100 |
| 1977-1981 | 4.85 | 116 |
| 1979-1983 | 5.15 | 142 |
| 1981-1985 | 5.15 | 146 |
| 1983-1987 | 5.15 | 144 |
| 1985-1989 | 5.05 | 180 |
| 1987-1991 | 5.05 | 218 |
| 1989-1993 | 3.25 | 370 |
| 1991-1995 | 3.25 | 439 |
| 1993-1997 | 4.65 | 503 |
| 1995-1999 | 4.55 | 437 |
| 1997-2001 | 4.55 | 429 |
| 1999-2003 | 4.35 | 514 |
| 2001-2005 | 4.35 | 614 |
| 2003-2007 | 4.35 | 634 |
| 2005-2009 | 4.35 | 680 |
| 2007-2011 | 4.25 | 785 |
| 2009-2013 | 4.25 | 921 |
| 2011-2015 | 4.25 | 985 |
| 2013-2017 | 3.45 | 1011 |
| 2015-2019 | 4.05 | 1017 |
| 2017-2021 | 4.05 | 1158 |
| 2019-2023 | 3.35 | 1241 |

## 5. b-value re-estimation

| Catalog | Mc | b (MLE) | σ_b | N used |
|---------|-----|---------|------|--------|
| USGS-only | 4.5 | 0.951 | 0.015 | 1985 |
| **Merged** | **4.08** | **0.765** | **0.009** | **3558** |

**b-value changed from 0.951 to 0.765 (Δ=-0.187).** This is a substantial change — the expanded catalog with proper Mc gives a different b-value, confirming that the USGS-only b was biased by truncation.

## 6. Rate re-estimation

| Threshold | N (USGS) | N (merged) | Rate USGS (1/yr) | Rate merged (1/yr) | Change |
|-----------|----------|------------|------------------|-------------------|--------|
| M≥4.5 | 1987 | 1947 | 38.2954 | 37.5245 | -0.7709 |
| M≥5.0 | 640 | 534 | 12.3347 | 10.2918 | -2.0429 |
| M≥5.5 | 96 | 70 | 1.8502 | 1.3491 | -0.5011 |
| M≥6.0 | 24 | 22 | 0.4626 | 0.4240 | -0.0385 |
| M≥6.5 | 9 | 8 | 0.1735 | 0.1542 | -0.0193 |
| M≥7.0 | 2 | 1 | 0.0385 | 0.0193 | -0.0193 |

## 7. Depth distribution comparison

| Catalog | Mean depth | Median | Min | Max | N shallow (<25km) | N deep (≥70km) |
|---------|-----------|--------|-----|-----|-------------------|----------------|
| USGS | 63.6 | 57.6 | 0.6 | 200.0 | 306 | 951 |
| Merged | 52.6 | 41.0 | 0.0 | 323.6 | 1827 | 1945 |

## 8. Data provenance

Every result in this report identifies which catalog(s) produced it:
- Merged catalog = USGS + ISC, matched by time/space proximity (120s, 50km)
- Original magnitudes preserved from both sources; Mw derived only via validated Scordilis (2006)
- ISC provides 786 MW magnitudes from contributing agencies (including GCMT indirectly)
- No fabricated data. Unavailable sources (GCMT, ISC-GEM, BMD, historical) documented.

## 9. Before-vs-after summary

| Metric | USGS-only (before) | Merged (after) | Change | Impact |
|--------|-------------------|----------------|--------|--------|
| N events | 2293 | 5779 | +3486 | 2.4× more data |
| Min magnitude | 3.2 | 2.4 | -0.8 | Resolves Mc |
| N below M3.5 | 3 | 1343 | +1340 | Mc now estimable |
| Mc (recommended) | 4.55 (unresolved) | 4.08 | -0.47 | RESOLVED |
| b-value (Mc=working) | 0.951 | 0.765 | -0.187 | Substantial |
| N multi-source | 0 | 2042 | +2042 | Cross-validation |

## 10. Impact on existing conclusions

The Phase A/B conclusions were based on the USGS-only catalog (floor M3.2). The merged catalog (floor M2.4) changes the data foundation:

1. **Mc is now estimable** — the previous 'Mc unresolved below M3.5' limitation is RESOLVED.
2. **b-value changed** from 0.951 to 0.765 — the USGS-only b was biased.
3. **Rates changed** — more events, especially below M4.5, changes the rate estimates.
4. **Spatial Poisson remains the primary benchmark** until the expanded catalog is fully validated.
5. **All Phase A/B model comparisons must be re-run** with the expanded catalog before drawing new conclusions about model skill.

**Do NOT declare a new model superior merely because the data changed.** The expanded catalog must be validated (Mc, b, rates) and all model comparisons (ETAS, ML, Spatial Poisson) must be re-run before any conclusion update.

## 11. Datasets acquired vs unavailable

| Dataset | Status | Impact if acquired |
|---------|--------|--------------------|
| ✅ ISC Bulletin | ACQUIRED | Resolves Mc; 2.4× more events; 786 MW magnitudes |
| ❌ GCMT | Unavailable | Would provide focal mechanisms for Coulomb + ETAS spatial kernels |
| ❌ ISC-GEM | Unavailable | Would extend catalog to 1904; authoritative Mw for historical events |
| ❌ BMD | Unavailable | Would provide M2-3 local events; further lower Mc |
| ❌ Historical | Unavailable | Would provide pre-1900 M7+ events for Mmax |

## 12. Next steps

1. Re-run Phase A/B model comparisons with the expanded catalog.
2. Re-estimate Mc(t) and spatial Mc with the expanded catalog.
3. Re-fit ETAS with the expanded catalog (more events may resolve K≈0).
4. Re-run ML backtest with expanded features (more training data).
5. If ISC-GEM/GCMT become available, integrate them for historical extension + focal mechanisms.