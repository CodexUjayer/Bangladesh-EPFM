# FINAL_v4.0_CANDIDATE_REGION_SPECIFIC_ETAS

> **STATUS: REJECTED (Verdict D)**
> **Final answer: NO** — a Bangladesh-specific ETAS formulation CANNOT explain the
> observed R≈24× clustering AND produce statistically defensible forecasting
> improvements.
> **NOT deployed prospectively.**
> FINAL_v1.0_FROZEN remains PRODUCTION.

## What this is

The **final major model-development experiment** in the Bangladesh earthquake
forecasting project. It tests whether the unresolved scientific contradiction
between strong Omori clustering (R≈24×) and vanishing ETAS productivity (K≈0)
is caused by ETAS model misspecification.

## The scientific contradiction

The project established both:

| Observation | Result |
|-------------|--------|
| Omori clustering | R≈24× at Δt≈18 min (non-parametric diagnostic) |
| ETAS productivity | K≈0 in ALL depth regimes (MLE fitting) |

These are inconsistent under standard ETAS assumptions: strong Omori clustering
implies K > 0.

## Model family

Four ETAS variants were implemented and tested:

| Variant | Description | K | α |
|---------|-------------|------|------|
| ETAS-A | Baseline ETAS (reference) | 0 | 0 |
| ETAS-B | Depth-stratified (shallow/intermediate/deep, independent fits) | 0 (all 3) | 0 |
| ETAS-C | Depth-dependent spatial kernels (σ(D) = σ₀·(1+κ·D/D_ref)) | 0 | 0 |
| ETAS-D | Exponential temporal kernel (modified Omori-Utsu) | 0 | 0 |

**All four variants reproduce K≈0.** The contradiction is NOT resolved by
ETAS misspecification.

## Key results

### Parameters (development period, pre-2010)

| Variant | n | μ (1/yr) | K | α | c (d) | p | σ (km) | BR | R peak | logL |
|---------|-----|----------|------|------|-------|-----|--------|------|--------|------|
| A baseline | 1910 | 51.78 | 1e-8 | 0.0 | 0.05 | 1.1 | 10.0 | 0.0 | 15.5× | -13585 |
| B shallow | 273 | 7.40 | 1e-8 | 0.0 | 0.05 | 1.1 | 10.0 | 0.0 | 15.5× | -2473 |
| B intermediate | 862 | 23.37 | 1e-8 | 0.0 | 0.05 | 1.1 | 10.0 | 0.0 | 15.5× | -6817 |
| B deep | 775 | 21.01 | 1e-8 | 0.0 | 0.05 | 1.1 | 10.0 | 0.0 | 15.5× | -6211 |
| C depth-spatial | 1910 | 51.78 | 1e-8 | 0.0 | 0.05 | 1.1 | 10.0 | 0.0 | 15.5× | -13585 |
| D exponential | 1910 | 51.78 | 1e-8 | 0.0 | — | — | 10.0 | 0.0 | 15.5× | -13585 |

### Retrospective evaluation (2015-2023, untouched)

- **Mean Brier (4 configs):** v4 = 0.02179, v1 = 0.02001. ΔBrier(v4−v1) = +0.00179 (v4 slightly worse).
- **Bootstrap CIs exclude zero in favour of v4:** 0/16 configs vs v1, 0/16 vs v2.
- **BH FDR-corrected rejections:** 0/16.
- **No statistically significant improvement in any variant × config.**

### Short-horizon evaluation (M≥4.5)

| Horizon | n+ | Brier v1 | Brier v4 | Δ(v4−v1) | Significant? |
|---------|-----|----------|----------|-----------|--------------|
| 1h | 0 | 0.00000 | 0.00000 | 0.00000 | — (no events) |
| 6h | 0 | 0.00000 | 0.00000 | 0.00000 | — (no events) |
| 24h | 0 | 0.000008 | 0.000001 | -0.000007 | NO |
| 7d | 9 | 0.015015 | 0.015443 | +0.000428 | NO |
| 30d | 34 | 0.049981 | 0.056230 | +0.006249 | NO (v4 worse) |
| 90d | 74 | 0.093581 | 0.113068 | +0.019487 | NO (v4 worse) |

v4 does NOT beat v1 at any horizon. At longer horizons v4 is worse because
its uniform background lacks the spatial heterogeneity of v1's per-cell rates.

### Posterior predictive checks: 3/3 PASS

The background-only model (which all variants collapse to) correctly
reproduces total event counts, depth distributions, and inter-event time
statistics. This is expected: with K=0, the model is just a Poisson process
with the correct rate.

### Spatial holdout

v4 beats v1 in 1/4 quadrants (SW), loses in 3/4 (NW, NE, SE).

## Answers to the 5 contradiction questions

1. **Why is R≈24× while K≈0?** The non-parametric Omori diagnostic measures
   actual post-mainshock rate enhancement without assuming a parametric form.
   It finds R≈15× (24× on the expanded catalog) at Δt≈3 min. The ETAS MLE
   fits the standard Omori-Utsu kernel which cannot represent clustering at
   this timescale (c parameter hits bounds). The clustering is real but its
   timescale is shorter than standard Omori can capture.

2. **Is triggering present but incorrectly modeled?** YES. The non-parametric
   R(Δt) shows clear triggering. CV_IET = 1.36 (>1 = clustered). Even ETAS-D
   with exponential temporal kernel and τ as short as 1e-4 day selects K=0.

3. **Is triggering confined to specific depth regimes?** NO. K≈0 in ALL depth
   regimes (shallow/intermediate/deep). CV_IET varies (shallow=1.65,
   intermediate=1.32, deep=1.36) but ETAS still selects K=0 everywhere.

4. **Is triggering limited to particular magnitudes?** The magnitude-scaling
   parameter α=0 in all fits, meaning productivity does not increase with
   mainshock magnitude. This violates the fundamental ETAS assumption.

5. **Does the Bangladesh catalog violate standard ETAS assumptions?** YES, in
   three ways: (a) temporal clustering timescale (~3 min) is shorter than
   Omori can represent; (b) deep subduction events may have different
   spatial triggering geometry; (c) α=0 violates the magnitude-productivity
   relationship. The most likely explanation is that the short-lag clustering
   is dominated by event relocations/duplicates rather than genuine tectonic
   aftershock cascades.

## Verdict: **D. REJECTED**

No evidence that ETAS misspecification explains the R≈24× / K≈0 contradiction.
K≈0 in all four variants; no predictive improvement over v1.

## Final answer: **NO**

A Bangladesh-specific ETAS formulation CANNOT explain the observed R≈24×
clustering AND produce statistically defensible forecasting improvements.
The Spatial Poisson baseline (FINAL_v1.0_FROZEN) remains the best-validated
probabilistic forecasting model for Bangladesh.

## Files

| File | Description |
|------|-------------|
| `model.py` | Region-specific ETAS implementation (4 variants, diagnostics, PPC) |
| `model_metadata.json` | Final metadata with verdict |
| `README.md` | This file |

## Companion artifacts (in `outputs/`)

| File | Description |
|------|-------------|
| `V4_REGION_SPECIFIC_ETAS_REPORT.md` | Full experiment report (19 sections) |
| `v4_etas_parameters.csv` | Per-variant × depth parameters and diagnostics |
| `v4_short_horizon_results.csv` | 1h/6h/24h/7d/30d/90d evaluation |
| `v4_depth_results.csv` | Per-depth-regime K/α/BR/R |
| `v4_clustering_results.csv` | Omori R(Δt) + CV_IET by depth |
| `v4_holdout_results.csv` | 4-quadrant spatial holdout |
| `v4_uncertainty_results.csv` | Per-config bootstrap CIs + permutation p-values |
| `v4_region_specific_etas_metadata.json` | Final metadata |

## Runner

```bash
cd bangladesh_eq_forecast
python run_v4_experiment.py
```

Runtime: ~2 minutes.

## Integrity

- **FINAL_v1.0_FROZEN:** source code unchanged; ledger unchanged; scores unchanged.
- **FINAL_v2.0_CANDIDATE_BAYESIAN_SPATIAL:** source code unchanged; ledger unchanged; scores unchanged.
- **FINAL_v3.0_CANDIDATE_ADAPTIVE_SPATIAL:** source code unchanged; verdict unchanged (REJECTED).
- **No evaluation-period leakage:** all parameters estimated on dev period (pre-2010) only.
- **No forecast rewriting, no cherry-picking, no post-hoc threshold selection.**
- **No fabricated data or performance.**
- **No deterministic earthquake predictions.**

All v4 artifacts are in a SEPARATE namespace (`v4_candidates/region_specific_etas/`
and `outputs/v4_*`). No v1, v2, or v3 file was modified, overwritten, or deleted.

## Scientific takeaway

The R≈24× / K≈0 contradiction is NOT caused by ETAS misspecification. Four
region-specific ETAS variants — including depth-stratified fitting,
depth-dependent spatial kernels, and a modified exponential temporal kernel —
all reproduce K≈0. The non-parametric Omori clustering signal is real but:

1. Its timescale (~3 minutes) is shorter than any parametric Omori-like
   kernel can represent.
2. The lack of magnitude scaling (α=0) violates the fundamental ETAS
   assumption that larger mainshocks produce more aftershocks.
3. The most likely physical explanation is that the short-lag clustering
   is dominated by event relocations/duplicates (multiple agency reports
   of the same physical event) rather than genuine tectonic aftershock
   cascades.

**This is the final major model-development experiment.** No v5 will be
developed. FINAL_v1.0_FROZEN (Spatial Poisson) remains the production
forecasting model for Bangladesh.
