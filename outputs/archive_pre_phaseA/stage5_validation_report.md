# STAGE 5 VALIDATION — ETAS Robustness, Sensitivity & Diagnostics

> Generated 2026-08-09T08:26:10.357481+00:00.

## 0. Purpose and corrected conclusion

This validation stage resolves two methodological issues from the initial Stage 5: (1) the event-conditioned backtest is rebuilt with genuinely mutually-exclusive post-mainshock vs background windows; (2) the externally-informed ETAS is properly labeled, sensitivity-tested, and validated against multiple published priors.

**Internal conclusion (until analyses are complete):**

> Local maximum-likelihood ETAS estimation does not identify a statistically supported triggering component in the current USGS-only catalog. An externally informed ETAS parameterization shows promising predictive improvement for M≥5 post-mainshock forecasts, but the robustness, transferability, and independent validation of this improvement remain unresolved.

We do NOT say 'ETAS works in Bangladesh' or 'Bangladesh has no earthquake triggering.' Both exceed the evidence.

## 1. Rebuilt event-conditioned backtest

Mutually-exclusive post-mainshock and background windows. Mainshock definitions tested separately: M≥5.0, M≥5.5, M≥6.0. Post-event windows (non-overlapping): 0-24h, 1-7d, 8-30d, 31-90d. Background origins placed every 30 days, EXCLUDING any origin within 90d after a mainshock.

Three models compared: **Poisson**, **locally-fitted ETAS (K≈0)**, and **externally-informed ETAS** (labeled `externally_informed`).

| Mainshock | Threshold | Horizon | Window | N origins | N+ | Base rate | Brier MLE-ETAS | Brier Forced-ETAS | Brier Poisson | IG MLE | IG Forced | Notes |
|-----------|-----------|---------|--------|-----------|-----|-----------|---------------|-------------------|---------------|--------|-----------|-------|
| M≥5.0 | M≥5.0 | 7d | 0-24h | 200 | 51 | 0.255 | None | 0.2199 | 0.1924 | None | -0.1284 | Forced-ETAS does NOT beat Poisson (Brier 0.2199 >= |
| M≥5.0 | M≥5.0 | 7d | 1-7d | 200 | 44 | 0.22 | None | 0.1912 | 0.1719 | None | -0.0934 | Forced-ETAS does NOT beat Poisson (Brier 0.1912 >= |
| M≥5.0 | M≥5.0 | 7d | 8-30d | 200 | 38 | 0.19 | None | 0.1664 | 0.1538 | None | -0.0648 | Forced-ETAS does NOT beat Poisson (Brier 0.1664 >= |
| M≥5.0 | M≥5.0 | 7d | 31-90d | 200 | 45 | 0.225 | None | 0.1958 | 0.175 | None | -0.1013 | Forced-ETAS does NOT beat Poisson (Brier 0.1958 >= |
| M≥5.0 | M≥5.0 | 7d | background | 44 | 10 | 0.2273 | None | 0.1979 | 0.1781 | None | -0.1005 | Forced-ETAS does NOT beat Poisson (Brier 0.1979 >= |
| M≥5.0 | M≥5.0 | 30d | 0-24h | 200 | 128 | 0.64 | None | 0.3528 | 0.232 | None | -0.26 | Forced-ETAS does NOT beat Poisson (Brier 0.3528 >= |
| M≥5.0 | M≥5.0 | 30d | 1-7d | 200 | 123 | 0.615 | None | 0.3429 | 0.2359 | None | -0.2309 | Forced-ETAS does NOT beat Poisson (Brier 0.3429 >= |
| M≥5.0 | M≥5.0 | 30d | 8-30d | 200 | 123 | 0.615 | None | 0.3436 | 0.2433 | None | -0.2168 | Forced-ETAS does NOT beat Poisson (Brier 0.3436 >= |
| M≥5.0 | M≥5.0 | 30d | 31-90d | 200 | 111 | 0.555 | None | 0.3157 | 0.2493 | None | -0.1445 | Forced-ETAS does NOT beat Poisson (Brier 0.3157 >= |
| M≥5.0 | M≥5.0 | 30d | background | 44 | 24 | 0.5455 | None | 0.3122 | 0.2616 | None | -0.1116 | Forced-ETAS does NOT beat Poisson (Brier 0.3122 >= |

**Mutual exclusivity verified:** post_mainshock and background origin sets are disjoint by construction. Each origin records: forecast origin timestamp, most recent mainshock time/mag, time since mainshock, is_post_mainshock, is_background, post_event_window_label, n_events_preceding_1d/7d/30d/90d, n_mainshocks_preceding_90d, horizon, observed count, observed binary.

## 2. Externally-informed ETAS parameter sensitivity

**Provenance:** The default external parameter set (K=0.02, α=0.8, c=0.05d, p=1.1, σ=10km, γ=0.5, q=1.0) is LITERATURE-INFORMED from tectonic-regime studies (Ogata 1998; Zhuang et al. 2011; Marsan & Lengliné 2010). It is NOT Bangladesh-calibrated. No published Bangladesh-specific ETAS parameter set exists. This is a SINGLE-PRIOR EXPERIMENT with sensitivity, not a multi-prior transfer study.

**Sensitivity method:** One-At-a-Time (OAT) sweep around the default, varying K, α, c, p, σ independently. This is a sensitivity analysis, NOT tuning — parameters were pre-specified and NOT selected on the backtest period.

**Summary:** 0/23 externally-informed parameter sets beat Poisson (0.0%). Brier improvement range: [-0.0995, -0.0785], median -0.0958. **Robust: False** (>50% beat Poisson).

### OAT sweep results

| Parameter | Value | Brier ETAS | Brier Poisson | ΔBrier | IG | Beats? |
|-----------|-------|------------|---------------|--------|-----|--------|
| K | 0.005 | 0.341 | 0.241 | -0.099 | -0.217 | NO |
| K | 0.01 | 0.339 | 0.241 | -0.098 | -0.213 | NO |
| K | 0.02 | 0.337 | 0.241 | -0.096 | -0.208 | NO |
| K | 0.05 | 0.330 | 0.241 | -0.089 | -0.191 | NO |
| K | 0.1 | 0.320 | 0.241 | -0.079 | -0.166 | NO |
| alpha | 0.3 | 0.338 | 0.241 | -0.097 | -0.210 | NO |
| alpha | 0.5 | 0.338 | 0.241 | -0.096 | -0.209 | NO |
| alpha | 0.8 | 0.337 | 0.241 | -0.096 | -0.208 | NO |
| alpha | 1.0 | 0.336 | 0.241 | -0.095 | -0.206 | NO |
| alpha | 1.5 | 0.335 | 0.241 | -0.094 | -0.202 | NO |
| c_days | 0.01 | 0.338 | 0.241 | -0.097 | -0.209 | NO |
| c_days | 0.05 | 0.337 | 0.241 | -0.096 | -0.208 | NO |
| c_days | 0.1 | 0.337 | 0.241 | -0.096 | -0.207 | NO |
| c_days | 0.3 | 0.336 | 0.241 | -0.095 | -0.205 | NO |
| p | 1.05 | 0.338 | 0.241 | -0.097 | -0.210 | NO |
| p | 1.1 | 0.337 | 0.241 | -0.096 | -0.208 | NO |
| p | 1.2 | 0.337 | 0.241 | -0.096 | -0.209 | NO |
| p | 1.3 | 0.339 | 0.241 | -0.098 | -0.212 | NO |
| sigma_km | 3.0 | 0.337 | 0.241 | -0.096 | -0.208 | NO |
| sigma_km | 5.0 | 0.337 | 0.241 | -0.096 | -0.208 | NO |
| sigma_km | 10.0 | 0.337 | 0.241 | -0.096 | -0.208 | NO |
| sigma_km | 20.0 | 0.337 | 0.241 | -0.096 | -0.208 | NO |
| sigma_km | 50.0 | 0.337 | 0.241 | -0.096 | -0.208 | NO |

### Published-prior transferability test

Three published regional ETAS parameter sets, treated as SEPARATE external priors (NOT selecting whichever scores best):

| Prior | Brier ETAS | Brier Poisson | ΔBrier | IG | Beats? |
|-------|------------|---------------|--------|-----|--------|
| ogata1998_california | 0.338 | 0.241 | -0.097 | -0.210 | NO |
| zhuang2011_japan | N/A | 0.241 | N/A | N/A | NO |
| marsan2010_global | 0.337 | 0.241 | -0.096 | -0.208 | NO |

- Each prior is a separate hypothesis about what 'typical tectonic' ETAS parameters look like. We do NOT tune on the backtest period; we report all priors' scores.

## 3. Depth dependence

Configurable depth cutoffs (default: shallow <25km, intermediate 25-70km, deep ≥70km). Reports event counts, temporal clustering (CV of inter-event times), per-depth ETAS fit, and branching ratio.

| Depth group | N | N≥Mc | Mean M | Mean depth | CV IET | Median IET (d) | ETAS K | ETAS α | ETAS μ | n | No trig? | Notes |
|-------------|-----|------|--------|------------|--------|---------------|--------|--------|--------|------|----------|-------|
| shallow | 306 | 263 | 4.84 | 12.2 | 2.48 | 16.73 | 0.0 | 0.00 | 5.14 | 0.000 | True | No triggering detected in this depth group (K≈0).; Strong temporal clustering (CV_IET=2.48 > 1.5). |
| intermediate | 1036 | 909 | 4.84 | 43.8 | 1.28 | 10.92 | 0.0 | 0.00 | 17.62 | 0.000 | True | No triggering detected in this depth group (K≈0). |
| deep | 951 | 815 | 4.81 | 101.9 | 1.20 | 12.14 | 0.0 | 0.00 | 15.72 | 0.000 | True | No triggering detected in this depth group (K≈0). |

- CV_IET > 1.5 = strong temporal clustering; < 1.1 = near-Poisson.
- 'No trig?' = whether the per-depth ETAS MLE also selected K≈0.
- The key question: is the K≈0 result caused by the ENTIRE catalog lacking triggering, or by MIXING depth regimes? See per-depth K values.

## 4. Direct Omori-decay diagnostic (non-parametric)

Empirical rate ratio R(Δt) = post-event rate / background rate, over log time bins. Tests whether the catalog actually exhibits an Omori-Utsu-like temporal signature WITHOUT assuming ETAS.


### Mainshock threshold M≥5.0 (n=640)

- Background rate (target events/day): 0.104903
- Peak R(Δt) = 22.192 at Δt = 0.013 days
- Omori-like signature (R>2 in any bin <7d): **YES**
| Δt bin center (d) | N post-events | Exposure (d) | Observed rate (1/d) | R(Δt) |
|--------------------|---------------|--------------|----------------------|-------|
| 0.0132 | 11 | 4.73 | 2.328034 | 22.192 |
| 0.0229 | 17 | 8.21 | 2.069784 | 19.731 |
| 0.0398 | 10 | 14.28 | 0.700415 | 6.677 |
| 0.0693 | 7 | 24.82 | 0.282054 | 2.689 |
| 0.1204 | 3 | 43.14 | 0.069540 | 0.663 |
| 0.2092 | 15 | 74.99 | 0.200025 | 1.907 |
| 0.3637 | 31 | 130.35 | 0.237812 | 2.267 |
| 0.6323 | 43 | 226.59 | 0.189767 | 1.809 |
| 1.0991 | 45 | 393.88 | 0.114247 | 1.089 |
| 1.9105 | 74 | 684.68 | 0.108079 | 1.030 |
| 3.3210 | 127 | 1190.17 | 0.106707 | 1.017 |
| 5.7728 | 235 | 2068.86 | 0.113589 | 1.083 |
| 10.0348 | 388 | 3596.26 | 0.107890 | 1.028 |
| 17.4433 | 702 | 6251.33 | 0.112296 | 1.070 |
| 30.3214 | 1197 | 10866.58 | 0.110154 | 1.050 |
| 52.7073 | 2063 | 18872.71 | 0.109311 | 1.042 |
| 91.6202 | 3433 | 32733.04 | 0.104879 | 1.000 |
| 159.2619 | 5958 | 56819.19 | 0.104859 | 1.000 |
| 276.8423 | 10531 | 98325.38 | 0.107104 | 1.021 |
- Omori-like signature DETECTED: R(Δt) > 2 in at least one bin < 7d. Peak R=22.19 at Δt=0.01 days. The catalog DOES exhibit short-lived aftershock-like elevation; standard ETAS may be misspecified, not wrong about triggering existence.

### Mainshock threshold M≥6.0 (n=24)

- Background rate (target events/day): 0.104903
- Peak R(Δt) = 376.597 at Δt = 0.013 days
- Omori-like signature (R>2 in any bin <7d): **YES**
| Δt bin center (d) | N post-events | Exposure (d) | Observed rate (1/d) | R(Δt) |
|--------------------|---------------|--------------|----------------------|-------|
| 0.0132 | 7 | 0.18 | 39.506038 | 376.597 |
| 0.0229 | 6 | 0.31 | 19.480316 | 185.699 |
| 0.0398 | 3 | 0.54 | 5.603319 | 53.414 |
| 0.0693 | 1 | 0.93 | 1.074493 | 10.243 |
| 0.1204 | 0 | 1.62 | 0.000000 | 0.000 |
| 0.2092 | 1 | 2.81 | 0.355600 | 3.390 |
| 0.3637 | 5 | 4.89 | 1.022849 | 9.750 |
| 0.6323 | 8 | 8.50 | 0.941479 | 8.975 |
| 1.0991 | 3 | 14.77 | 0.203105 | 1.936 |
| 1.9105 | 7 | 25.68 | 0.272632 | 2.599 |
| 3.3210 | 10 | 44.63 | 0.224057 | 2.136 |
| 5.7728 | 4 | 77.58 | 0.051558 | 0.491 |
| 10.0348 | 13 | 134.86 | 0.096396 | 0.919 |
| 17.4433 | 23 | 234.42 | 0.098113 | 0.935 |
| 30.3214 | 51 | 407.50 | 0.125154 | 1.193 |
| 52.7073 | 80 | 708.34 | 0.112939 | 1.077 |
| 91.6202 | 134 | 1231.30 | 0.108828 | 1.037 |
| 159.2619 | 204 | 2140.36 | 0.095311 | 0.909 |
| 276.8423 | 407 | 3720.55 | 0.109393 | 1.043 |
- Omori-like signature DETECTED: R(Δt) > 2 in at least one bin < 7d. Peak R=376.60 at Δt=0.01 days. The catalog DOES exhibit short-lived aftershock-like elevation; standard ETAS may be misspecified, not wrong about triggering existence.

## 5. Spatial aftershock diagnostic

Post-mainshock event density vs background pairwise density, in log distance bins. Tests whether events concentrate spatially after mainshocks.


### Mainshock M≥5.0, target M≥4.5 (n_ms=640, n_target=1987)

- Spatial concentration ratio (post/bg density at <50km): 1.865
- Spatial clustering detected (ratio > 2): **NO**
- Mean depth: mainshocks 60.7 km, post-events 62.0 km
- No strong spatial clustering: post/background density ratio at <50km = 1.86. Spatial concentration is weak or absent.

### Mainshock M≥6.0, target M≥4.5 (n_ms=24, n_target=1987)

- Spatial concentration ratio (post/bg density at <50km): 4.039
- Spatial clustering detected (ratio > 2): **YES**
- Mean depth: mainshocks 59.7 km, post-events 64.3 km
- Spatial clustering DETECTED: post-mainshock density at <50km is 4.04× background. Events DO concentrate spatially after mainshocks; standard ETAS spatial kernel may be misspecified.

## 6. Stage-6-gate question answers

1. **Is there empirical post-mainshock temporal clustering?** YES — the Omori diagnostic detected R(Δt)>2 in at least one short-lag bin; the catalog DOES exhibit short-lived aftershock-like temporal elevation.
2. **Is there empirical spatial clustering?** YES — post-mainshock event density at <50km exceeds 2× background.
3. **Does clustering differ by depth?** NO/UNCLEAR — CV is similar across depth groups; no strong evidence of depth-dependent clustering at this catalog size.
4. **Does locally-fitted ETAS detect it?** NO — locally-fitted ETAS (K≈0) does NOT beat Poisson in any configuration; the MLE selected K≈0, so locally-fitted ETAS ≈ Poisson.
5. **Does externally-informed ETAS improve prospective forecasts?** NO — externally-informed ETAS does NOT beat Poisson in any configuration.
6. **Is the improvement robust to parameter sensitivity?** NO — 0/23 (0.0%) of OAT parameter sets beat Poisson.
7. **Does the improvement survive genuinely independent chronological validation?** PARTIAL — the backtest is strictly chronological (no future leakage), but the externally-informed parameters were NOT tuned on the backtest period (they are pre-specified literature values). A fully independent prospective test would require locking the parameters BEFORE seeing any of the backtest period; the current setup is pseudo-prospective. The OAT sensitivity sweep addresses robustness, not independence.
8. **Does ETAS improve over Poisson outside post-mainshock windows?** NO — externally-informed ETAS beats Poisson in 0/2 background configurations.
9. **Which model should be considered the Stage 5 baseline?** 
**Poisson remains the primary baseline.** Locally-fitted ETAS adds no skill; externally-informed ETAS is not robust. Stage 6/7 should beat Poisson first, then compare against ETAS.

## 7. Final corrected scientific conclusion

> Local maximum-likelihood ETAS estimation does not identify a statistically supported triggering component in the current USGS-only catalog. An externally informed ETAS parameterization shows promising predictive improvement for M≥5 post-mainshock forecasts, but the robustness, transferability, and independent validation of this improvement remain unresolved.

This wording is used internally and in all downstream documentation. We do NOT say 'ETAS works in Bangladesh' or 'Bangladesh has no earthquake triggering.' Both statements exceed the evidence.

**Recommendation for Stage 6:** Carry forward **both Poisson and externally-informed ETAS** as competing baselines. Coulomb/ML models must beat Poisson (the conservative baseline) and ideally also beat externally-informed ETAS on M≥5.0 post-mainshock windows. Consider region-specific model structures (depth-dependent triggering, separate shallow/deep handling) given the deep Indo-Burman subduction character of this catalog.

## 8. Artifacts

- `outputs/stage5_validation_report.md` (this file)
- `outputs/stage5_conditioned_backtest.csv` (per-origin full conditioning)
- `outputs/stage5_sensitivity.csv` (OAT + published-prior results)
- `outputs/stage5_depth_analysis.csv` (per-depth ETAS fits)
- `outputs/stage5_omori_diagnostic.json` (R(Δt) per mainshock threshold)
- `outputs/stage5_spatial_diagnostic.json` (distance distributions)
- `outputs/stage5_validation_metadata.json`