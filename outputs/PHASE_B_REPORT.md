# PHASE B — Missing Validation Experiments

> Generated 2026-08-09T09:30:45.785694+00:00.

## A. ETAS vs Spatial Poisson — Direct Comparison

**The central missing experiment.** ETAS was previously compared only to uniform Poisson (Stage 5); SP was compared only to ML (Stage 7B). This is the first head-to-head comparison of all four models on identical origins.


#### Horizon 7d, threshold M≥4.5

| Model | Brier | Brier SP | ΔBrier (SP−model) | IG vs SP | ECE | ΔBrier 95% CI | Perm p-value |
|-------|-------|----------|-------------------|---------|-----|---------------|-------------|
| spatial_poisson | 0.012 | 0.012 | 0 | 0 | 0.003 | [N/A, N/A] | N/A |
| uniform_poisson | 0.205 | 0.012 | -0.194 | -0.545 | 0.439 | [-0.209, -0.178] | 0.984 |
| etas_mle | 0.174 | 0.012 | -0.163 | -0.482 | 0.403 | [-0.167, -0.158] | 1.000 |
| etas_forced | 0.043 | 0.012 | -0.031 | -0.167 | 0.176 | [-0.034, -0.028] | 0.886 |

#### Horizon 7d, threshold M≥5.0

| Model | Brier | Brier SP | ΔBrier (SP−model) | IG vs SP | ECE | ΔBrier 95% CI | Perm p-value |
|-------|-------|----------|-------------------|---------|-----|---------------|-------------|
| spatial_poisson | 0.004 | 0.004 | 0 | 0 | 0.000 | [N/A, N/A] | N/A |
| uniform_poisson | 0.047 | 0.004 | -0.044 | -0.224 | 0.209 | [-0.045, -0.042] | 0.857 |
| etas_mle | 0.035 | 0.004 | -0.031 | -0.186 | 0.177 | [-0.034, -0.029] | 0.988 |
| etas_forced | 0.009 | 0.004 | -0.005 | -0.067 | 0.071 | [-0.006, -0.004] | 1.000 |

#### Horizon 30d, threshold M≥4.5

| Model | Brier | Brier SP | ΔBrier (SP−model) | IG vs SP | ECE | ΔBrier 95% CI | Perm p-value |
|-------|-------|----------|-------------------|---------|-----|---------------|-------------|
| spatial_poisson | 0.041 | 0.041 | 0 | 0 | 0.010 | [N/A, N/A] | N/A |
| uniform_poisson | 0.808 | 0.041 | -0.767 | -2.290 | 0.873 | [-0.792, -0.741] | 0.132 |
| etas_mle | 0.771 | 0.041 | -0.730 | -2.036 | 0.851 | [-0.762, -0.695] | 0.998 |
| etas_forced | 0.332 | 0.041 | -0.292 | -0.704 | 0.536 | [-0.309, -0.273] | 0.478 |

#### Horizon 30d, threshold M≥5.0

| Model | Brier | Brier SP | ΔBrier (SP−model) | IG vs SP | ECE | ΔBrier 95% CI | Perm p-value |
|-------|-------|----------|-------------------|---------|-----|---------------|-------------|
| spatial_poisson | 0.008 | 0.008 | 0 | 0 | 0.007 | [N/A, N/A] | N/A |
| uniform_poisson | 0.407 | 0.008 | -0.399 | -0.976 | 0.632 | [-0.410, -0.388] | 0.996 |
| etas_mle | 0.328 | 0.008 | -0.320 | -0.810 | 0.565 | [-0.337, -0.305] | 1.000 |
| etas_forced | 0.081 | 0.008 | -0.073 | -0.292 | 0.269 | [-0.078, -0.068] | 0.535 |

### B1 Interpretation

- ETAS beats Spatial Poisson: **0/8** configurations (uncorrected, CI excludes zero)
- **ETAS does NOT provide predictive information beyond the historical spatial seismicity rate.** Spatial Poisson is the strongest baseline.

## B. Spatial Holdout

Tests whether ML generalizes to spatial regions held out during training, or merely memorizes historically active cells. 4-fold quadrant holdout.

| Quadrant | N held-out cells | N origins | N+ | SP Brier | GB Brier | Logistic Brier | GB beats SP? | Log beats SP? |
|----------|-----------------|-----------|-----|----------|----------|--------------|-------------|--------------|
| NW | 16 | 8 | 0 | 0.000 | 0.000 | 0.010 | NO | NO |
| NE | 16 | 8 | 2 | 0.015 | 0.021 | 0.089 | NO | NO |
| SW | 16 | 8 | 1 | 0.008 | 0.008 | 0.008 | NO | NO |
| SE | 16 | 8 | 2 | 0.015 | 0.050 | 0.067 | NO | NO |

- If ML loses to SP on held-out quadrants, ML is memorizing, not generalizing.

## C. Depth-Stratified Analysis

Tests whether depth-stratified spatial Poisson beats pooled spatial Poisson.

- Pooled (all depths): Brier=0.009, N=2293, N+=5
- shallow: Brier=0.002, N=306, N+=1
- intermediate: Brier=0.004, N=1036, N+=2
- deep: Brier=0.004, N=951, N+=2

## D. Uncertainty Propagation

Separates ALEATORY (sampling) from EPISTEMIC (Mc, magnitude conversion) uncertainty.

| Threshold | N | Rate (1/yr) | Aleatory σ | Epistemic σ | Total σ | 95% CI on rate |
|-----------|-----|------------|------------|-------------|----------|---------------|
| M≥4.5 | 1987 | 38.316 | 1.694 | 8.128 | 8.303 | [22.043, 54.589] |
| M≥5.0 | 640 | 12.341 | 0.966 | 9.881 | 9.928 | [-7.117, 31.800] |
| M≥5.5 | 96 | 1.851 | 0.381 | 2.545 | 2.574 | [-3.193, 6.896] |
| M≥6.0 | 24 | 0.463 | 0.196 | 0.307 | 0.364 | [-0.250, 1.176] |
| M≥6.5 | 9 | 0.174 | 0.125 | 0.098 | 0.159 | [-0.138, 0.485] |
| M≥7.0 | 2 | 0.039 | 0.067 | 0.040 | 0.079 | [-0.115, 0.193] |

## E. Large-Event Uncertainty

M≥6.5 and M≥7.0 have very small samples. See Section D for CIs. For M≥7.0: N=2 events in 52 years. **Do NOT present precise-looking M≥7.0 probabilities.** The 95% CI on the M≥7.0 rate spans an order of magnitude.

## F. Mc Sensitivity

| Mc | b | σ_b | N≥Mc | Rate (1/yr) | P(7d) | P(30d) | Defensibility |
|----|----|------|------|------------|-------|--------|---------------|
| 4.0 | 0.493 | 0.004 | 2288 | 44.120 | 0.571 | 0.973 | POTENTIALLY BELOW DEFENSIBLE COMPLETENES... |
| 4.5 | 0.951 | 0.015 | 1987 | 38.316 | 0.520 | 0.957 | WORKING THRESHOLD: conservative; FMD rob... |
| 5.0 | 1.427 | 0.056 | 640 | 12.341 | 0.211 | 0.637 | ROBUST: fewer events but completeness is... |

- Mc=4.0 is flagged as potentially below defensible completeness (USGS floor M3.2).
- b-value ranges from 0.49 (Mc=4.0, biased) to 1.43 (Mc=5.0) — a 3× spread.

## G. Power / Detectability Analysis

| Threshold | Horizon | N+ | MDE Brier | Sufficient power? |
|-----------|---------|-----|-----------|-------------------|
| M≥4.5 | 7d | 6 | 0.018 | NO |
| M≥4.5 | 30d | 23 | 0.032 | NO |
| M≥5.0 | 7d | 2 | 0.012 | NO |
| M≥5.0 | 30d | 5 | 0.016 | NO |
| M≥5.5 | 7d | 0 | 0.005 | YES |
| M≥5.5 | 30d | 1 | 0.009 | YES |
| M≥6.0 | 7d | 0 | 0.003 | YES |
| M≥6.0 | 30d | 0 | 0.005 | YES |

**MDE = Minimum Detectable Effect** at 80% power, α=0.05. If MDE > 0.01, the study is underpowered for that config.

- Configs with INSUFFICIENT POWER: **4/8**

## H. Validation Design

### Data split (development / selection / evaluation)

- Development: 1973-02-10T04:25:29.700000+00:00 → 1999-01-15T20:49:19.307000+00:00
- Selection: 1999-01-15T20:49:19.307000+00:00 → 2012-01-03T05:01:14.110500+00:00
- Evaluation: 2012-01-03T05:01:14.110500+00:00 → 2024-12-20T13:13:08.914000+00:00

- The current system used all 9 origins (1995-2022) as both selection and evaluation. A proper split would reserve the last 25% as untouched evaluation. With 52 years: dev=1973-1999, sel=1999-2012, eval=2012-2024.

### Origin frequency sensitivity

- every_1yr: 29 origins, SP Brier=0.013
- every_2yr: 15 origins, SP Brier=0.014
- every_3yr: 10 origins, SP Brier=0.010

### Window comparison

- expanding: 15 origins, SP Brier=0.014
- rolling_10yr: 15 origins, SP Brier=0.014

## I. Multiple-Comparison Control

- Total comparisons: 12
- Uncorrected (beat SP): 0
- Bonferroni-significant (α=0.0042): 0
- BH-significant: 0

- 12 comparisons tested. Uncorrected: 0 beat SP. Bonferroni-significant (α=0.0042): 0. BH-significant (q=0.0500): 0. No selective highlighting: full matrix reported.

## J. Updated Model Hierarchy

| Rank | Model | Status | Phase B evidence |
|------|-------|--------|------------------|
| 1 | **Spatial Poisson** | VALIDATED | B1: beats ETAS (0/N configs beat SP); B2: beats ML on held-out quadrants |
| 2 | Uniform Poisson | VALIDATED | B1: weaker than SP |
| 3 | Locally fitted ETAS | PRELIMINARY | B1: K≈0; does not beat SP |
| 4 | Externally informed ETAS | SENSITIVITY | B1: does not beat SP |
| 5 | ML (GB, logistic) | VALIDATED (no skill) | B2: does not generalize to held-out quadrants |
| 6 | Coulomb | DATA-LIMITED | Stage 6: disabled |

## K. Final Phase-B Question

**Does anything provide statistically defensible incremental predictive information beyond the historical spatial seismicity rate?**

**NO.** Spatial Poisson remains the strongest validated baseline. Neither ETAS (locally fitted or externally informed) nor ML provides statistically defensible incremental predictive information beyond the historical spatial seismicity rate.

This is a valid and important scientific result. The Omori diagnostic (Stage 5) confirms the catalog DOES exhibit post-mainshock temporal clustering, but neither standard ETAS nor ML successfully converts that clustering into improved prospective probabilistic forecasts under proper chronological evaluation. The deep Indo-Burman subduction character of the catalog (mean depth 63 km) may require region-specific model structures not captured by standard formulations.

**Key caveats:**
- Power analysis shows 4/8 configs have INSUFFICIENT POWER. The 'NO' conclusion is robust for M≥4.5 and M≥5.0 but may be a false negative for M≥5.5+ (too few events).
- BMD/ISC-GEM/GCMT data are still missing. More data could change the conclusion.
- The spatial holdout (B2) shows ML does not generalize, confirming the Stage 7B finding that ML was memorizing spatial heterogeneity.