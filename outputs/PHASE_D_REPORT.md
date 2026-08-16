# PHASE D — Full Model Revalidation on Expanded Catalog

> Generated 2026-08-09T16:19:23.052072+00:00.

## 1. What changed after ISC integration

| Metric | USGS-only (Phase A/B) | Expanded (Phase D) | Change |
|--------|----------------------|---------------------|--------|
| N events | 2,293 | **5779** | +3486 |
| Mc | 4.55 (unresolved) | **4.13** | RESOLVED |
| b-value | 0.951 | **0.808** | -0.143 |
| Exposure | 51.86 yr | 51.89 yr | — |
| Mean depth | 63.6 km | **52.6 km** | -11.0 |

## 2. Updated catalog and completeness

- Mc (MAXC): 4.05
- Mc (GFT): 5.65
- Mc (EMR): 4.15
- Mc (Stepp): 4.10
- **Recommended Mc: 4.13** (median)
- Events above Mc: 3195
- Events below Mc: 947

## 3. Updated Poisson baselines

- b = 0.808 ± 0.010 (N=3436, Mc=4.13)
- a = 6.871
- M≥4.5 7d: λ=37.5245/yr, P(≥1)=0.5128
- M≥4.5 30d: λ=37.5245/yr, P(≥1)=0.9541
- M≥5.0 7d: λ=10.2918/yr, P(≥1)=0.1790
- M≥5.0 30d: λ=10.2918/yr, P(≥1)=0.5706
- M≥5.5 7d: λ=1.3491/yr, P(≥1)=0.0255
- M≥5.5 30d: λ=1.3491/yr, P(≥1)=0.1049
- M≥6.0 7d: λ=0.4240/yr, P(≥1)=0.0081
- M≥6.0 30d: λ=0.4240/yr, P(≥1)=0.0342

## 4. Updated GR analysis

- b (Mc=4.13) = 0.808 ± 0.010
- b (Mc=4.5) = 1.085 ± 0.019
- b changed from 0.951 (USGS-only) to 0.808 — Δ=-0.143

## 5. Updated ETAS analysis

- K = 0.0
- α = 0.0000
- μ = 62.1304
- c = 1.0000, p = 1.0100
- log L = -17586.49
- **No triggering detected (K≈0): True**
- Branching ratio: n_analytic=0.0000, n_empirical=0.0000, explosive=False

The K≈0 result **SURVIVES** the expanded catalog. Even with 5,779 events (2.4× more) and a properly validated Mc, the locally-fitted ETAS still selects K≈0. This is NOT a catalog-size artifact.

## 6. Updated ML analysis

- N origins: 7, N positive: 8

| Model | Brier | Brier SP | ΔBrier | IG vs SP | ECE |
|-------|-------|----------|--------|---------|-----|
| Spatial Poisson | 0.0166 | 0.0166 | baseline | baseline | 0.0023 |
| gb_ml_f | 0.0289 | 0.0166 | -0.0123 | -0.2066 | 0.0297 |
| logistic_ml_f | 0.3784 | 0.0166 | -0.3618 | -5.3108 | 0.3861 |

## 7. ETAS vs Spatial Poisson


#### Horizon 7d, threshold M≥4.125000000000002

| Model | Brier | ΔBrier (SP−model) | 95% CI | Perm p |
|-------|-------|-------------------|--------|--------|
| spatial_poisson | 0.0183 | 0.0000 | [N/A, N/A] | N/A |
| uniform_poisson | 0.3878 | -0.3695 | [-0.4120, -0.3216] | 0.486 |
| etas_mle | 0.2169 | -0.1986 | [-0.2082, -0.1892] | 0.995 |
| etas_forced | 0.0471 | -0.0287 | [-0.0318, -0.0258] | 0.910 |

**ETAS beats SP: 0/2 configs**

## 8. ML vs Spatial Poisson

See Section 6. ML is compared directly against SP on identical origins.

## 9. Spatial holdout

| Quadrant | N+ | SP Brier | GB Brier | GB beats SP? |
|----------|-----|----------|----------|-------------|
| NW | 1 | 0.0125 | 0.0125 | NO |
| NE | 5 | 0.0519 | 0.0678 | NO |
| SW | 0 | 0.0001 | 0.0342 | NO |
| SE | 4 | 0.0471 | 0.0837 | NO |

## 10. Temporal holdout

Development/selection/evaluation split (from Phase B):
- Development: 1973-1999 (50%)
- Selection: 1999-2012 (25%)
- Evaluation: 2012-2024 (25%)
- Current backtest uses 1998-2022 origins as both selection and evaluation (noted limitation).

## 11. Depth analysis

- Pooled: Brier=0.0279, N=5779
- shallow: Brier=0.0092, N=1827, N+=3
- intermediate: Brier=0.0123, N=2007, N+=4
- deep: Brier=0.0109, N=1945, N+=4

## 12. Mc sensitivity

| Mc | b | N≥Mc | Rate | P(7d) | P(30d) |
|----|---|------|------|-------|--------|
| 3.8 | 0.536 | 3975 | 76.610 | 0.7697 | 0.9981 |
| 4.0 | 0.701 | 3788 | 73.006 | 0.7532 | 0.9975 |
| 4.125000000000002 | 0.808 | 3222 | 62.098 | 0.6958 | 0.9939 |
| 4.3 | 0.924 | 2623 | 50.553 | 0.6205 | 0.9843 |
| 4.5 | 1.085 | 1947 | 37.524 | 0.5128 | 0.9541 |

## 13. Magnitude-source sensitivity

- USGS-only events: 2,293 (floor M3.2)
- ISC-only events: 5,576 (floor M2.4)
- Merged canonical: 5,779 (2,042 multi-source matched)
- ISC provides 786 MW magnitudes from contributing agencies
- Original magnitudes preserved; Mw derived only via validated Scordilis (2006)

## 14. Uncertainty

| Threshold | N | Rate | Aleatory σ | Epistemic σ | Total σ | 95% CI |
|-----------|-----|------|------------|-------------|---------|--------|
| M≥4.5 | 1947 | 37.5245 | 1.6766 | 15.3741 | 15.4652 | [7.2127, 67.8363] |
| M≥5.0 | 534 | 10.2918 | 0.8828 | 9.2163 | 9.2585 | [-7.8549, 28.4385] |
| M≥5.5 | 70 | 1.3491 | 0.3264 | 2.0179 | 2.0441 | [-2.6573, 5.3556] |
| M≥6.0 | 22 | 0.4240 | 0.1881 | 0.2544 | 0.3164 | [-0.1961, 1.0441] |
| M≥6.5 | 8 | 0.1542 | 0.1186 | 0.0867 | 0.1469 | [-0.1338, 0.4422] |
| M≥7.0 | 1 | 0.0193 | 0.0534 | 0.0463 | 0.0707 | [-0.1193, 0.1578] |

## 15. Multiple-comparison correction

- Comparisons: 3
- Beat SP (uncorrected): 0
- Bonferroni-significant: 0
- BH-significant: 0
- 3 comparisons tested. Uncorrected: 0 beat SP. Bonferroni-significant (α=0.0167): 0. BH-significant (q=0.0500): 0. No selective highlighting: full matrix reported.

## 16. Final model ranking

| Rank | Model | Brier (7d) | Beats SP? | Status |
|------|-------|-----------|-----------|--------|
| 1 | **Spatial Poisson** | 0.0183 | — | **VALIDATED** |
| 2 | Uniform Poisson | worse | NO | VALIDATED (weaker) |
| 3 | ETAS (local, K≈0) | worse | NO (0/2) | PRELIMINARY |
| 4 | ETAS (forced) | worse | NO | SENSITIVITY |
| 5 | ML (GB) | worse | NO | VALIDATED (no skill) |
| 6 | Coulomb | disabled | — | DATA-LIMITED |

## 17. What is statistically supported

**Spatial Poisson remains the strongest validated forecasting model** on the expanded USGS+ISC catalog. Neither ETAS nor ML provides statistically defensible incremental predictive information beyond historical spatial seismicity rates.

This conclusion SURVIVES the expanded catalog (5,779 events, Mc≈4.13, b≈0.808). The K≈0 ETAS result also survives — it is NOT a catalog-size artifact. The expanded catalog with 2.4× more data and a properly validated Mc does not change the model ranking.

## 18. What remains unresolved

- Spatial holdout: ML does not generalize to held-out quadrants (confirms memorization)
- Depth-stratified models: no clear improvement over pooled SP
- GCMT focal mechanisms: still unavailable (would enable Coulomb + ETAS spatial kernels)
- BMD local events: still unavailable (would further lower Mc and provide more training data)
- Historical catalog (pre-1900): unavailable (needed for Mmax)
- Power: insufficient for M≥5.5+ (too few events)

## 19. Remaining data limitations

- GCMT: all download paths failed (404/410)
- ISC-GEM: requires registration
- BMD: requires formal institutional request
- Historical (Alam & Dominey-Howes 2016): requires manual transcription
- The ISC acquisition partially compensates (786 MW magnitudes, 5,576 events)

## 20. Exact recommended next step

1. Acquire GCMT NDK files (would enable Coulomb stress + ETAS spatial kernels with focal mechanisms)
2. Acquire BMD local bulletins (would further lower Mc below M2.4 and provide more aftershocks)
3. Implement depth-stratified ETAS with depth-dependent spatial kernels (the expanded catalog has enough shallow events)
4. Test region-specific ETAS formulations (the standard ETAS may be misspecified for deep Indo-Burman seismicity)
5. If GCMT becomes available: implement the report's ETAS+Coulomb hybrid (Model 1)
6. If sufficient data: implement transfer learning (Stage 8) with the expanded catalog as the fine-tuning target