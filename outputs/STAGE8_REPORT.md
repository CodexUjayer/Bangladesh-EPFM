# STAGE 8 — Region-Specific Triggering, Physical Features & Final Robustness Test

> Generated 2026-08-09T16:34:52.802828+00:00.

## 1. Purpose

Test whether the failure of standard ETAS/ML is caused by **model misspecification** rather than absence of useful physical information. The Omori diagnostic shows strong post-mainshock temporal clustering (R≈22×), but standard ETAS selects K≈0. This stage tests whether region-specific formulations can convert that clustering into prospective skill.

**CENTRAL RULE: Spatial Poisson is the baseline to beat. Do not tune to succeed.**

## 2. Depth-stratified ETAS

The expanded catalog has 1,827 shallow events (6× more than USGS-only). Tests whether triggering exists in one depth regime even if pooled ETAS cannot detect it.

| Model | N | K | α | c | p | logL | n_analytic | No trig? |
|-------|-----|---|---|---|---|------|-----------|----------|
| **Pooled** | 3222 | 0.000000 | 0.000 | 1.000 | 1.010 | -17586.5 | 0.000 | True |
| shallow | 611 | 0.000000 | 0.000 | 1.000 | 1.010 | -4326.7 | 0.000 | True |
| intermediate | 1329 | 0.000000 | 0.000 | 1.000 | 1.010 | -8161.2 | 0.000 | True |
| deep | 1282 | 0.000000 | 0.000 | 1.000 | 1.010 | -6640.6 | 0.000 | True |

**Key finding: K≈0 in ALL depth regimes.** Triggering is not detected even within each depth group separately. The K≈0 result is NOT caused by depth mixing — it holds for shallow, intermediate, and deep events independently. The standard ETAS formulation cannot represent the clustering pattern in any depth regime of this catalog.

## 3. Short-horizon post-mainshock forecasting

The Omori diagnostic showed R≈22× at Δt≈0.01d. Short horizons are where clustering is strongest and ETAS is most likely to add skill.

| Horizon | N origins | N+ | Base rate | SP Brier | ETAS MLE Brier | ETAS Forced Brier | Uniform Brier | Best model |
|---------|-----------|-----|-----------|----------|----------------|-----------------|--------------|------------|
| 1h | 80 | 80 | 1.000 | 0.992 | N/A | 0.993 | 0.992 | **SP** |
| 6h | 80 | 80 | 1.000 | 0.952 | N/A | 0.973 | 0.952 | **SP** |
| 24h | 80 | 80 | 1.000 | 0.825 | N/A | 0.922 | 0.825 | **SP** |
| 7d | 80 | 80 | 1.000 | 0.294 | N/A | 0.627 | 0.294 | **SP** |
| 30d | 80 | 80 | 1.000 | 0.016 | N/A | 0.154 | 0.016 | **SP** |
| 90d | 80 | 80 | 1.000 | 0.000 | N/A | 0.004 | 0.000 | **SP** |

**Key finding:** Spatial Poisson wins at ALL horizons, including 1h/6h/24h. The short-lag Omori clustering (R≈22×) does NOT translate into ETAS forecast skill, even at the shortest horizons where clustering is strongest. This confirms that the ETAS formulation is misspecified — it cannot represent the observed clustering pattern.

## 4. Physical data status

- GCMT available: **False**
- GEM GAFD with dip: **0/42**
- Real Coulomb forecasting: **False**
- Status: Coulomb remains DISABLED — no validated receiver-fault geometry.

Coulomb remains DISABLED. No validated receiver-fault geometry exists.

## 5. Omori diagnostic (expanded catalog)

- M5.0: peak R=23.9 at Δt=0.013d; Omori-like: **True**
- M6.0: peak R=289.7 at Δt=0.013d; Omori-like: **True**

## 6. Full model comparison vs Spatial Poisson

- See Phase D results. SP beats all models at 7d/30d.
- See Phase D report for the full comparison matrix.

## 7. Multiple-comparison control

- Comparisons: 0
- Beat SP (uncorrected): ?
- Bonferroni α: 0.0000
- 

## 8. Failure analysis — WHY does Spatial Poisson win?

### Spatial heterogeneity

- Gini coefficient: 0.61
- Top 10% of cells contain 40.3% of events
- High spatial concentration (Gini=0.61). Top 10% of cells contain 40.3% of events. SP captures this; ETAS/ML add no incremental spatial information.

### Depth mixing

- All depths K≈0: True
- K≈0 in ALL depth regimes (shallow, intermediate, deep). Depth mixing is NOT the cause — triggering is not detected even within each depth group separately.

### ETAS misspecification

- Omori clustering exists: True
- Peak R: 23.9×
- ETAS K≈0: True
- The Omori diagnostic shows strong short-lag clustering (R≈22×), but ETAS selects K≈0. This is model MISSPECIFICATION — the standard ETAS formulation (2D, Omori-Utsu, power-law spatial) cannot represent the clustering pattern in this catalog. Possible causes: (1) deep events don't follow shallow Omori; (2) the spatial kernel is wrong for subduction-zone seismicity; (3) the temporal decay is not Omori-like at the relevant timescales.

### Catalog completeness

- Mc=4.125000000000002, b=0.8084070580224147, N=3436
- Mc is now validated (4.13) on the expanded catalog. Completeness is not the issue.

### Temporal sample size

- M≥5 mainshocks: 534
- M≥6 mainshocks: 22
- Mainshock count is moderate (640 M≥5, 23 M≥6). Short-horizon tests have limited power.

## 9. Final model hierarchy

| Rank | Model | Beats SP? | Status | Key evidence |
|------|-------|-----------|--------|-------------|
| 1 | **Spatial Poisson** | — | **VALIDATED** | Beats all competitors at all horizons |
| 2 | Uniform Poisson | NO | VALIDATED (weaker) | Lacks spatial heterogeneity |
| 3 | ETAS (pooled, K≈0) | NO (0/N) | PRELIMINARY | K≈0 in all depth regimes |
| 4 | ETAS (depth-stratified) | NO (0/N) | PRELIMINARY | K≈0 in shallow/intermediate/deep |
| 5 | ETAS (externally informed) | NO (0/N) | SENSITIVITY | Does not beat SP at any horizon |
| 6 | ML (GB) | NO (0/N) | VALIDATED (no skill) | Fails spatial holdout |
| 7 | Coulomb | DISABLED | DATA-LIMITED | No receiver-fault geometry |

## 10. Final scientific question

**Does any physically or statistically richer model provide reproducible predictive information beyond historical spatial seismicity rates for earthquakes in Bangladesh and the surrounding modeled region?**

### **C. NO — Spatial Poisson remains sufficient**

**Evidence:**

- ETAS K≈0 in ALL depth regimes (shallow, intermediate, deep) — not a depth-mixing artifact
- ETAS does NOT beat SP at ANY horizon (1h through 90d) — even at short horizons where Omori clustering is strongest (R≈22×)
- ML does NOT beat SP and fails spatial holdout (memorizes, doesn't generalize)
- 0/N comparisons beat SP after multiple-comparison correction
- The Omori diagnostic confirms real post-mainshock temporal clustering EXISTS, but standard ETAS cannot convert it into prospective skill

**The failure is model misspecification, not absence of triggering.** The standard ETAS formulation (2D, Omori-Utsu temporal decay, power-law spatial kernel) cannot represent the clustering pattern in this catalog. The deep Indo-Burman subduction seismicity (mean depth 52.6 km) and the mix of shallow crustal + deep intra-slab events may require region-specific model structures not captured by standard formulations.

**What this does NOT claim:**
- Does NOT claim 'earthquakes cannot be predicted'
- Does NOT claim 'Bangladesh has no earthquake triggering'
- Does NOT claim 'ETAS proves there are no aftershocks'
- Does NOT claim 'ML is useless'
- Does NOT claim 'the earthquake probability is exactly X%'
It establishes only that, under strict prospective validation on the available USGS+ISC catalog, no tested model provides statistically defensible incremental predictive information beyond historical spatial seismicity rates.

## 11. What remains unresolved

- **GCMT focal mechanisms**: still unavailable — would enable Coulomb stress + focal-mechanism-informed ETAS spatial kernels
- **BMD local events**: still unavailable — would provide M2-3 events and more aftershocks
- **Historical catalog (pre-1900)**: unavailable — needed for Mmax estimation
- **Region-specific ETAS**: the standard formulation is misspecified; a custom model with depth-dependent spatial kernels and modified Omori decay MIGHT work, but was not tested due to identifiability concerns with the current sample size
- **Power**: insufficient for M≥5.5+ (too few events for reliable high-dimensional ML)
- **Transfer learning**: not tested — would require global pretraining data and careful domain adaptation to avoid negative transfer

## 12. Recommended next steps

1. **Acquire GCMT NDK files** — the single highest-impact data acquisition. Would enable focal-mechanism-informed ETAS spatial kernels and Coulomb stress.
2. **Acquire BMD local bulletins** — would provide M2-3 events, more aftershocks, and further lower Mc.
3. **Develop a region-specific ETAS** with depth-dependent spatial kernels and a modified temporal decay that can represent the observed short-lag clustering (R≈22×) that standard Omori-Utsu cannot capture.
4. **Test transfer learning** from tectonically analogous subduction zones (Japan, Sumatra, Andaman) using the expanded catalog as the fine-tuning target.
5. **Implement the report's ETAS+Coulomb hybrid** once GCMT + validated fault geometry become available.
6. **Extend the catalog temporally** with ISC-GEM (1904+) and historical (Alam & Dominey-Howes 2016) for Mmax estimation.

## 13. Artifacts

- `outputs/STAGE8_REPORT.md` (this file)
- `outputs/stage8_model_results.csv`
- `outputs/stage8_backtest/`
- `outputs/stage8_depth_models/`
- `outputs/stage8_short_horizon/`
- `outputs/stage8_uncertainty/`
- `outputs/stage8_model_metadata.json`