# Bayesian Hierarchical Spatial Seismicity-Rate Model — FINAL_v2.0 CANDIDATE

## Model: v2.0_CANDIDATE_BAYESIAN_SPATIAL

## Status: EXPERIMENTAL — NOT production

## Control: FINAL_v1.0_FROZEN (immutable, must not be modified)

## Model Structure

```
N_i ~ Poisson(T * λ_i)
λ_i ~ Gamma(α, β)    (hierarchical prior)

Empirical Bayes hyperparameters:
  α = μ_rate² / σ_rate²    (shape, from cross-cell rate distribution)
  β = μ_rate / σ_rate²     (rate)

Posterior:
  λ_i | N_i, T ~ Gamma(α + N_i, β + T)

Posterior predictive:
  P(≥1 event in Δt) = 1 - (β/(β+Δt))^(α+N_i)
```

## Key Differences from v1.0

| Feature | v1.0 (Spatial Poisson) | v2.0 (Bayesian Hierarchical) |
|---------|----------------------|------------------------------|
| Rate estimation | MLE: N_i / T | Posterior: Gamma(α+N_i, β+T) |
| Uncertainty | Garwood exact Poisson CI | Full posterior distribution |
| Between-cell sharing | None | Hierarchical shrinkage via α, β |
| Low-stat cells | Wide CI, zero rate possible | Pulled toward regional mean |
| Epistemic uncertainty | Not captured | Captured via posterior |
| Probability | 1 - exp(-λ̂Δt) | 1 - (β/(β+Δt))^(α+N) |

## Priors

- **Empirical Bayes** (default): α, β estimated from cross-cell rate distribution
  - Justification: data-driven, adapts to catalog; standard for spatial smoothing (Clayton & Kaldor 1987)
- **Fixed weakly informative** (sensitivity): α=1.0, β=0.1
  - Justification: represents weak prior belief; no information leakage

## Inference Method

Conjugate Gamma-Poisson (analytical; no MCMC needed):
- Computationally efficient (<1 second per forecast)
- No convergence issues
- Exact posterior (not approximate)

## Validation

- Development: 1973–2006
- Selection: 2006–2015
- Evaluation: 2015–2024 (untouched, same as v1.0)
- Prospective: separate experimental ledger (does not contaminate v1.0)

## Random Seed

42 (for reproducibility of posterior sampling)
