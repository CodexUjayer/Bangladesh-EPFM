"""Stage 4 report generator and artifact saver.

Produces:
  outputs/stage4_report.md            — full narrative report
  outputs/stage4_baseline_results.csv — flat table of all baseline results
  outputs/stage4_probability_maps/    — per (threshold, horizon) cell forecasts CSV
  outputs/stage4_backtest/            — per (threshold, horizon) backtest details
  outputs/stage4_model_metadata.json  — catalog version, thresholds, Mc, etc.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from ..ingestion.schema import CanonicalEvent
from .backtest import run_chronological_backtest
from .forecast import SpatialForecast, forecast_spatial
from .gutenberg_richter import GRResult, fit_gr_multiple_thresholds
from .large_events import LargeEventAssessment, assess_large_events
from .poisson import HORIZON_YEARS, TemporalPoissonResult, estimate_temporal_poisson
from .spatial import GridConfig, SpatialGrid, build_spatial_grid


def _fmt(x, nd=4):
    if x is None:
        return "N/A"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "N/A"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    return f"{x:.{nd}f}"


def generate_stage4_report(
    events: list[CanonicalEvent],
    working_threshold: float,
    mc_scenarios: list[float],
    poisson_results: list[TemporalPoissonResult],
    gr_results: list[GRResult],
    spatial_grid: SpatialGrid,
    spatial_forecast: SpatialForecast,
    large_event_assessments: list[LargeEventAssessment],
    backtest_results: list,
    catalog_metadata: dict,
) -> str:
    """Generate the full Stage 4 markdown report."""
    md = []
    md.append("# STAGE 4 — Statistical Baseline Layer (Poisson + Gutenberg-Richter)\n")
    md.append(f"> Generated {datetime.now(timezone.utc).isoformat()}.\n")

    md.append("## 0. Probability interpretation discipline\n")
    md.append("Throughout this report, the following are kept DISTINCT and never conflated:\n")
    md.append("- **Rate λ** (events per year): estimated as N / T. NOT a probability.")
    md.append("- **Expected count over horizon Δt**: λ × Δt. NOT a probability.")
    md.append("- **P(N ≥ 1 | Δt)**: 1 − exp(−λΔt). This is a probability.")
    md.append("- **Cell probability**: same formula with the cell's λ.")
    md.append("- Everything is **conditional on the observed catalog and the working Mc**.")
    md.append("- The word **'risk' is NOT used** (risk requires exposure/vulnerability, "
              "which is not modeled in Stage 4).\n")

    md.append("## 1. Catalog and thresholding\n")
    md.append(f"- Catalog file: `{catalog_metadata.get('catalog_file', 'N/A')}`")
    md.append(f"- Catalog version: {catalog_metadata.get('catalog_version', 'N/A')}")
    md.append(f"- N events (full M≥2.5 query): {catalog_metadata.get('n_events_total', 'N/A'):,}")
    md.append(f"- Time range: {catalog_metadata.get('time_range', 'N/A')}")
    md.append(f"- Exposure (catalog span): **{catalog_metadata.get('exposure_years', 'N/A'):.2f} years**")
    md.append(f"- **Working modeling threshold: M ≥ {working_threshold}** "
              f"(conservative working modeling threshold under the current USGS-only "
              f"catalog limitations; NOT a definitively validated regional Mc).")
    md.append(f"- Mc uncertainty (from Stage 3 audit): working range **M3.5–4.5**; "
              f"USGS ComCat floor ~M3.2; true Mc unverifiable below ~M3.5 without BMD/ISC.")
    md.append(f"- Mc sensitivity scenarios tested: {mc_scenarios}\n")

    # ---- 2. Temporal Poisson ----
    md.append("## 2. Temporal Poisson baseline\n")
    md.append("Rate λ = N / T (events per year). P(N ≥ 1 | Δt) = 1 − exp(−λΔt).\n")
    md.append("| Threshold | N obs | Exposure (yr) | λ (1/yr) | 95% CI on λ | "
              "P(≥1) 24h | P(≥1) 7d | P(≥1) 30d | P(≥1) 90d | P(≥1) 1y |")
    md.append("|-----------|-------|---------------|----------|-------------|"
              "-----------|-----------|-----------|-----------|----------|")
    for r in poisson_results:
        pr = r.rate
        p24 = pr.probability_at_least_one(HORIZON_YEARS["24h"])
        p7 = pr.probability_at_least_one(HORIZON_YEARS["7d"])
        p30 = pr.probability_at_least_one(HORIZON_YEARS["30d"])
        p90 = pr.probability_at_least_one(HORIZON_YEARS["90d"])
        p1y = pr.probability_at_least_one(HORIZON_YEARS["1y"])
        md.append(f"| M≥{pr.threshold} | {pr.n_observed} | {pr.exposure_years:.2f} | "
                  f"{pr.rate_per_year:.4f} | [{pr.rate_ci_lower:.4f}, {pr.rate_ci_upper:.4f}] | "
                  f"{p24:.4f} | {p7:.4f} | {p30:.4f} | {p90:.4f} | {p1y:.4f} |")
    md.append("\n**Notes:**")
    md.append("- Expected count (λΔt) is NOT shown in the probability columns; "
              "those are P(N ≥ 1).")
    md.append("- For small N, the 95% CI on λ spans a wide range; the probability "
              "CI (not shown in this table, see CSV) is correspondingly wide.")
    for r in poisson_results:
        if r.notes:
            for n in r.notes:
                md.append(f"- M≥{r.threshold}: {n}")

    # ---- 3. Gutenberg-Richter ----
    md.append("\n## 3. Gutenberg-Richter model (MLE)\n")
    md.append("Fitted by Aki-Utsu maximum-likelihood (NOT visual line fit). "
              "Shi-Bolt (1982) standard error; bootstrap 95% CI on b.\n")
    md.append("| Mc | b (MLE) | σ_b (Shi-Bolt) | b 95% CI | a (at Mc) | σ_a | "
              "N used | M range | Notes |")
    md.append("|----|----------|----------------|----------|-----------|-----|"
              "--------|---------|-------|")
    for g in gr_results:
        notes_str = "; ".join(g.notes[:2]) if g.notes else ""
        md.append(f"| {g.threshold} | {_fmt(g.b_mle,3)} | {_fmt(g.b_sigma_shibolt,3)} | "
                  f"[{_fmt(g.b_ci_lower,3)}, {_fmt(g.b_ci_upper,3)}] | "
                  f"{_fmt(g.a_value,3)} | {_fmt(g.a_sigma,3)} | {g.n_events_used} | "
                  f"{g.magnitude_range[0]:.2f}-{g.magnitude_range[1]:.2f} | {notes_str} |")
    md.append("\n**Sensitivity interpretation:** We do NOT pick the Mc that gives the "
              "most attractive b-value. The three scenarios are reported jointly as a "
              "sensitivity analysis. The Stage 3 audit established Mc is a working "
              "range (M3.5-4.5), not a validated threshold; the b-value variation "
              "across Mc scenarios reflects this uncertainty.")
    # Add explicit per-scenario interpretation
    md.append("\n**Per-scenario notes:**")
    for g in gr_results:
        if g.threshold == 4.0:
            md.append(f"- **Mc=4.0**: b={_fmt(g.b_mle,3)} is anomalously LOW. This is "
                      "because Mc=4.0 sits AT or BELOW the catalog's effective floor "
                      "(USGS ComCat floor ~M3.2; Stage 3 working range M3.5-4.5). The "
                      "FMD is truncated there, so the Aki-Utsu MLE (which assumes "
                      "complete sampling above Mc) is BIASED. This b-value should NOT "
                      "be used for extrapolation; it is reported only to show the "
                      "sensitivity.")
        elif g.threshold == 4.5:
            md.append(f"- **Mc=4.5**: b={_fmt(g.b_mle,3)} is the WORKING estimate. "
                      "This is the conservative modeling threshold; the FMD above 4.5 "
                      "is robustly sampled in USGS ComCat. This is the b-value used "
                      "for the primary Poisson/GR baseline.")
        elif g.threshold == 5.0:
            md.append(f"- **Mc=5.0**: b={_fmt(g.b_mle,3)} uses fewer events (more "
                      "robust completeness) but loses the smaller-magnitude leverage. "
                      "The higher b-value reflects the steeper tail when small events "
                      "are excluded; this is a known MLE sensitivity, not necessarily "
                      "a better estimate.")

    # ---- 4. Mc sensitivity ----
    md.append("\n## 4. Mc sensitivity (probability of larger events)\n")
    md.append("How the estimated probability of M≥6.0 events (1-year horizon) "
              "changes under each Mc scenario, using the GR model extrapolated from "
              "each threshold:\n")
    md.append("| Mc scenario | b | N≥Mc | Predicted N≥6.0 | Predicted rate≥6.0 (1/yr) | "
              "P(≥1 M≥6.0 | 1yr) |")
    md.append("|------------|----|------|-----------------|---------------------------|"
              "----------------------|")
    exposure = catalog_metadata.get("exposure_years", 1.0)
    for g in gr_results:
        if math.isnan(g.b_mle):
            md.append(f"| {g.threshold} | N/A | {g.n_events_used} | N/A | N/A | N/A |")
            continue
        n_pred_6 = g.n_predicted_above(6.0)
        rate_6 = n_pred_6 / exposure if exposure > 0 else float("nan")
        p_6 = 1.0 - math.exp(-rate_6 * 1.0)
        md.append(f"| {g.threshold} | {g.b_mle:.3f} | {g.n_events_used} | "
                  f"{n_pred_6:.2f} | {rate_6:.4f} | {p_6:.4f} |")
    md.append("\n**Interpretation:** Lower Mc uses more events but risks including "
              "incomplete data; higher Mc is more robust but uses fewer events. The "
              "spread in predicted P(≥1 M≥6.0 | 1yr) across scenarios is a direct "
              "measure of how much the Mc uncertainty propagates into the forecast.")

    # ---- 5. Spatial baseline ----
    md.append("\n## 5. Spatial baseline\n")
    md.append(f"- Grid: {spatial_grid.config.cell_size_deg}° resolution → "
              f"{len(spatial_grid.cells)} cells.")
    md.append(f"- Cells with ≥1 event above M≥{working_threshold}: "
              f"{spatial_grid.n_cells_with_events} / {len(spatial_grid.cells)}")
    md.append(f"- Cells flagged low-statistics (N < {spatial_grid.config.min_events_for_stable_rate}): "
              f"{spatial_grid.n_cells_low_statistics}")
    md.append(f"- Exposure: {spatial_grid.exposure_years:.2f} years.")
    md.append("- Coarse grid chosen deliberately; finer resolution is NOT "
              "automatically better and would inflate the number of low-statistics "
              "cells.\n")
    md.append(f"Top 10 cells by event count (M≥{working_threshold}):\n")
    md.append("| Cell | Lat | Lon | N | λ (1/yr) | 95% CI | λ density (1/km²/yr) | "
              "Mean M | Max M | Low-stat |")
    md.append("|------|-----|-----|---|----------|---------|----------------------|"
              "--------|--------|----------|")
    top_cells = sorted(spatial_grid.cells, key=lambda c: -c.n_events)[:10]
    for c in top_cells:
        md.append(f"| {c.cell_id} | {c.lat_center:.2f} | {c.lon_center:.2f} | "
                  f"{c.n_events} | {c.rate_per_year:.4f} | "
                  f"[{c.rate_ci_lower:.4f}, {c.rate_ci_upper:.4f}] | "
                  f"{c.rate_density_per_km2_per_year:.2e} | "
                  f"{_fmt(c.mean_magnitude,2)} | {_fmt(c.max_magnitude,2)} | "
                  f"{c.low_statistics} |")

    # ---- 6. Spatial + magnitude forecast ----
    md.append("\n## 6. Spatial + magnitude forecast (example cells)\n")
    md.append(f"Full table in `outputs/stage4_probability_maps/`. Example: top 5 "
              f"cells by P(≥1 M≥5.0 | 7d).\n")
    md.append("| Cell | Lat | Lon | N (M≥5.0) | λ (1/yr) | Expected (7d) | "
              "P(≥1 | 7d) | 95% UI | Low-stat |")
    md.append("|------|-----|-----|-----------|----------|---------------|"
              "-----------|---------|----------|")
    f75 = [f for f in spatial_forecast.forecasts
           if f.threshold == 5.0 and f.horizon == "7d" and f.n_events_above_threshold > 0]
    f75.sort(key=lambda x: -x.probability_at_least_one)
    for f in f75[:5]:
        md.append(f"| {f.cell_id} | {f.lat_center:.2f} | {f.lon_center:.2f} | "
                  f"{f.n_events_above_threshold} | {f.expected_rate_per_year:.4f} | "
                  f"{f.expected_count:.4f} | {f.probability_at_least_one:.4f} | "
                  f"[{f.probability_ci_lower:.4f}, {f.probability_ci_upper:.4f}] | "
                  f"{f.low_statistics} |")
    md.append("\n- **Expected count (λΔt) is NOT a probability** and is reported "
              "separately from P(≥1).")
    md.append("- Low-statistics cells have wide uncertainty intervals; their point "
              "probabilities are indicative only.")

    # ---- 7. Large-event limitation ----
    md.append("\n## 7. Large-event limitation (M≥6.5, M≥7.0)\n")
    md.append("For rare large events, ordinary frequentist precision is NOT "
              "achievable. We report N, exposure, rate CI under three priors "
              "(Garwood frequentist, Jeffreys Bayesian, Uniform Bayesian), and "
              "prior sensitivity.\n")
    md.append("| Threshold | N obs | Exposure (yr) | λ (1/yr) | Garwood 95% CI | "
              "Jeffreys 95% CI | Uniform 95% CI | Prior ratio | Sufficient? |")
    md.append("|-----------|-------|---------------|----------|----------------|"
              "------------------|-----------------|-------------|-------------|")
    for a in large_event_assessments:
        md.append(f"| M≥{a.threshold} | {a.n_observed} | {a.exposure_years:.2f} | "
                  f"{a.rate_per_year:.4f} | "
                  f"[{a.rate_ci_garwood[0]:.4f}, {a.rate_ci_garwood[1]:.4f}] | "
                  f"[{a.rate_ci_jeffreys[0]:.4f}, {a.rate_ci_jeffreys[1]:.4f}] | "
                  f"[{a.rate_ci_uniform_prior[0]:.4f}, {a.rate_ci_uniform_prior[1]:.4f}] | "
                  f"{a.prior_sensitivity_ratio:.3f} | "
                  f"{'YES' if a.sufficient_for_frequentist_precision else 'NO'} |")
    md.append("\n**Notes:**")
    for a in large_event_assessments:
        for n in a.notes:
            md.append(f"- M≥{a.threshold}: {n}")

    # ---- 8. Backtesting ----
    md.append("\n## 8. Chronological backtesting\n")
    md.append("Expanding-window chronological backtest. For each yearly origin "
              "(1995-2024), train on all events before the origin, forecast the "
              "next horizon, compare to observation. NO shuffling.\n")
    md.append("| Model | Threshold | Horizon | N origins | N positive | Base rate | "
              "Mean forecast P | Brier | Log-lik | IG vs climatology | ROC-AUC (sec.) | "
              "Cal. error |")
    md.append("|-------|-----------|---------|-----------|------------|-----------|"
              "-----------------|-------|---------|--------------------|-----------------|"
              "------------|")
    for bt in backtest_results:
        s = bt.to_summary_row()
        md.append(f"| {s['model']} | M≥{s['threshold']} | {s['horizon']} | "
                  f"{s['n_origins']} | {s['n_positive']} | {s['base_rate']} | "
                  f"{s['mean_forecast_probability']} | {s['brier_score']} | "
                  f"{s['log_likelihood']} | {s['information_gain_vs_climatology']} | "
                  f"{s['roc_auc_secondary']} | {s['calibration_error']} |")
    md.append("\n**Notes:**")
    md.append("- ROC-AUC is reported as a SECONDARY diagnostic only; for rare "
              "events it can be misleadingly high.")
    md.append("- Information gain is against the climatology (base rate) reference, "
              "the appropriate null for rare-event forecasting.")
    md.append("- Brier score and log-likelihood are the primary probabilistic metrics.")
    for bt in backtest_results:
        for n in bt.notes:
            md.append(f"- M≥{bt.threshold}, {bt.horizon}: {n}")

    # ---- 9. Baseline comparison ----
    md.append("\n## 9. Baseline comparison table (summary)\n")
    md.append("| Model | Threshold | Horizon | Expected rate (1/yr) | P(≥1) | "
              "Brier | Log-lik | Calibration | 95% UI on P |")
    md.append("|-------|-----------|---------|----------------------|-------|"
              "-------|---------|-------------|-------------|")
    # Poisson rows
    for r in poisson_results:
        for hname in ["24h", "7d", "30d", "90d", "1y"]:
            hr = r.horizon_results[hname]
            md.append(f"| temporal_poisson | M≥{r.threshold} | {hname} | "
                      f"{r.rate.rate_per_year:.4f} | {hr['P_ge1']:.4f} | "
                      f"— | — | — | [{hr['P_ge1_ci'][0]:.4f}, {hr['P_ge1_ci'][1]:.4f}] |")
    # Backtest rows (Brier, log-lik, calibration)
    for bt in backtest_results:
        # find matching poisson rate
        pr = next((r for r in poisson_results if r.threshold == bt.threshold), None)
        rate_str = f"{pr.rate.rate_per_year:.4f}" if pr else "—"
        # mean forecast P and its CI from origins
        if bt.origins:
            ps = [o.forecast_probability for o in bt.origins]
            mean_p = sum(ps) / len(ps)
            ci_lo = min(o.forecast_ci[0] for o in bt.origins)
            ci_hi = max(o.forecast_ci[1] for o in bt.origins)
        else:
            mean_p = float("nan"); ci_lo = ci_hi = float("nan")
        md.append(f"| temporal_poisson (backtested) | M≥{bt.threshold} | {bt.horizon} | "
                  f"{rate_str} | {mean_p:.4f} | {bt.brier:.4f} | {bt.log_likelihood:.4f} | "
                  f"{bt.calibration_error:.4f} | [{ci_lo:.4f}, {ci_hi:.4f}] |")

    # ---- 10. Reproducibility ----
    md.append("\n## 10. Reproducibility\n")
    md.append("All results record: catalog version, filtering threshold, Mc scenario, "
              "date range, geographic region, forecast horizon, model version, parameter "
              "estimates. See `outputs/stage4_model_metadata.json`.\n")
    md.append("**Artifacts:**")
    md.append("- `outputs/stage4_report.md` (this file)")
    md.append("- `outputs/stage4_baseline_results.csv` (flat results table)")
    md.append("- `outputs/stage4_probability_maps/` (per threshold×horizon cell forecasts)")
    md.append("- `outputs/stage4_backtest/` (per threshold×horizon backtest details)")
    md.append("- `outputs/stage4_model_metadata.json`")

    # ---- 11. Scientific conclusion ----
    md.append("\n## 11. Scientific conclusion\n")
    md.append("**How much predictive skill can we obtain from historical seismicity "
              "alone, before adding aftershock triggering, physics-based stress, or "
              "machine learning?**\n")
    md.append("The temporal Poisson baseline captures only the **long-term average "
              "rate**. It has NO short-term time-dependence: after a large earthquake, "
              "the Poisson forecast for the next 7 days is identical to any other "
              "7-day window. This is the central limitation that Stage 5 ETAS must "
              "address.\n")
    md.append("Specifically:")
    md.append("- The Poisson baseline's Brier score and log-likelihood reflect "
              "essentially the **climatology** (base rate). Information gain over "
              "climatology is expected to be ~0 by construction (Poisson rates are "
              "constant in time).")
    md.append("- The spatial baseline captures **where** events are more likely "
              "(Indo-Burman fold belt, Arakan megathrust) but not **when**.")
    md.append("- The GR model gives a **magnitude-distribution** prior, useful for "
              "extrapolating rates to larger magnitudes, but with large uncertainty "
              "for M≥6.5 where observations are few (Section 7).")
    md.append("- The Mc uncertainty (M3.5-4.5 working range) propagates into a "
              "non-trivial spread in b-value and hence in extrapolated large-event "
              "probabilities (Section 4).")
    md.append("\n**Expected Stage 5 improvement:** ETAS should beat this baseline "
              "primarily on **short-term horizons (24h, 7d) after mainshocks**, where "
              "Omori-law aftershock decay creates real time-dependence that Poisson "
              "cannot capture. On 90-day and 1-year horizons, the Poisson baseline "
              "may be competitive because aftershock sequences have largely decayed. "
              "Any ML model in Stage 7 must be evaluated against THIS baseline first; "
              "if it does not beat Poisson+Brier on the short horizons, it adds no "
              "skill.\n")
    md.append("**This is the bar Stage 5 must clear.**")

    return "\n".join(md)


def save_stage4_artifacts(
    events: list[CanonicalEvent],
    report_md: str,
    poisson_results: list[TemporalPoissonResult],
    gr_results: list[GRResult],
    spatial_grid: SpatialGrid,
    spatial_forecast: SpatialForecast,
    large_event_assessments: list[LargeEventAssessment],
    backtest_results: list,
    catalog_metadata: dict,
    output_dir: str | Path,
) -> None:
    """Save all Stage 4 artifacts to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage4_report.md").write_text(report_md, encoding="utf-8")

    # Flat results CSV
    rows = []
    for r in poisson_results:
        rows.append(r.to_row())
    for g in gr_results:
        rows.append(g.to_row())
    for a in large_event_assessments:
        rows.append(a.to_row())
    for bt in backtest_results:
        rows.append(bt.to_summary_row())
    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        with (out / "stage4_baseline_results.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # Probability maps: one CSV per (threshold, horizon)
    pmap_dir = out / "stage4_probability_maps"
    pmap_dir.mkdir(exist_ok=True)
    by_th_h = {}
    for f in spatial_forecast.forecasts:
        by_th_h.setdefault((f.threshold, f.horizon), []).append(f)
    for (th, h), fs in by_th_h.items():
        fname = pmap_dir / f"forecast_M{th:.1f}_{h}.csv"
        with fname.open("w", encoding="utf-8", newline="") as fcsv:
            if fs:
                w = csv.DictWriter(fcsv, fieldnames=list(fs[0].to_row().keys()))
                w.writeheader()
                for f in fs:
                    w.writerow(f.to_row())

    # Backtest details
    bt_dir = out / "stage4_backtest"
    bt_dir.mkdir(exist_ok=True)
    for bt in backtest_results:
        fname = bt_dir / f"backtest_M{bt.threshold:.1f}_{bt.horizon}.csv"
        with fname.open("w", encoding="utf-8", newline="") as fcsv:
            w = csv.DictWriter(fcsv, fieldnames=[
                "origin_time", "horizon", "threshold", "n_train_events",
                "train_exposure_years", "forecast_rate_per_year",
                "forecast_probability", "forecast_ci_lower", "forecast_ci_upper",
                "n_observed_in_horizon", "observed_binary",
            ])
            w.writeheader()
            for o in bt.origins:
                w.writerow({
                    "origin_time": o.origin_time.isoformat(),
                    "horizon": o.horizon,
                    "threshold": o.threshold,
                    "n_train_events": o.n_train_events,
                    "train_exposure_years": round(o.train_exposure_years, 3),
                    "forecast_rate_per_year": round(o.forecast_rate_per_year, 6),
                    "forecast_probability": round(o.forecast_probability, 6),
                    "forecast_ci_lower": round(o.forecast_ci[0], 6),
                    "forecast_ci_upper": round(o.forecast_ci[1], 6),
                    "n_observed_in_horizon": o.n_observed_in_horizon,
                    "observed_binary": o.observed_binary,
                })

    # Model metadata
    metadata = {
        "stage": 4,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_metadata": catalog_metadata,
        "model_version": "stage4_temporal_poisson_gr_v0.1",
        "probability_interpretation": {
            "rate_lambda": "events per year, N/T",
            "expected_count": "lambda * horizon (NOT a probability)",
            "probability_at_least_one": "1 - exp(-lambda * horizon)",
            "conditional_on": "observed catalog and working Mc",
        },
        "working_threshold": catalog_metadata.get("working_threshold"),
        "mc_scenarios": catalog_metadata.get("mc_scenarios"),
        "mc_caveat": "Mc is a working range (M3.5-4.5), NOT a validated threshold. "
                     "USGS ComCat floor ~M3.2. True Mc unverifiable below ~M3.5 "
                     "without BMD/ISC data.",
        "poisson_thresholds": [r.threshold for r in poisson_results],
        "gr_thresholds": [g.threshold for g in gr_results],
        "large_event_thresholds": [a.threshold for a in large_event_assessments],
        "backtest_configs": [
            {"threshold": bt.threshold, "horizon": bt.horizon,
             "n_origins": bt.n_origins}
            for bt in backtest_results
        ],
        "spatial_grid": {
            "cell_size_deg": spatial_grid.config.cell_size_deg,
            "n_cells": len(spatial_grid.cells),
            "n_cells_with_events": spatial_grid.n_cells_with_events,
            "n_cells_low_statistics": spatial_grid.n_cells_low_statistics,
        },
    }
    (out / "stage4_model_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
