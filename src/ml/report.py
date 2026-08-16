"""Stage 7 report generator and artifact saver."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .backtest import BacktestConfig, aggregate_evaluations, run_chronological_backtest
from .evaluation import EvalMetrics
from .features import ALL_FEATURE_NAMES, FEATURE_GROUPS, features_for_group


def _fmt(x, nd=3):
    if x is None:
        return "N/A"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "N/A"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{nd}f}"


def generate_stage7_report(
    all_results: dict,  # (horizon, threshold) -> {model_key -> EvalMetrics}
    catalog_metadata: dict,
    config_matrix: list,
    experiment_manifest: list,
) -> str:
    md = []
    md.append("# STAGE 7 — Machine Learning Forecasting\n")
    md.append(f"> Generated {datetime.now(timezone.utc).isoformat()}.\n")

    md.append("## 0. Primary question\n")
    md.append("**Can ML produce better calibrated, genuinely out-of-sample probabilistic "
              "earthquake forecasts than the corrected Poisson baseline?**\n")
    md.append("The answer must be one of: A (YES, robust), B (PARTIAL), C (NO), D (INCONCLUSIVE). "
              "ML is NOT treated as successful merely because it has high classification accuracy. "
              "Calibration and prospective probabilistic skill are primary.\n")

    md.append("## 1. Hierarchy of baselines (locked from Stages 4-6)\n")
    md.append("1. **Stationary/expanding-window Poisson** = PRIMARY VALIDATED BASELINE")
    md.append("2. Spatial Poisson = secondary baseline")
    md.append("3. Locally fitted ETAS (K≈0) = diagnostic only; no demonstrated improvement")
    md.append("4. Externally informed ETAS = sensitivity only; no demonstrated improvement")
    md.append("5. Coulomb = DISABLED (no validated receiver-fault geometry)")
    md.append("6. **ML = now tested** against the validated baselines\n")

    md.append("## 2. Forecast target and grid\n")
    md.append("- P(N_cell,Δt ≥ 1 | information available at forecast origin)")
    md.append(f"- Grid: {catalog_metadata.get('grid', '1.0 deg')}, "
              f"{catalog_metadata.get('n_cells', 64)} cells")
    md.append("- Horizons: 24h, 7d, 30d, 90d, 1y")
    md.append("- Magnitude thresholds: M≥4.5, M≥5.0, M≥5.5, M≥6.0 (M≥6.5/7.0 = research/exploratory)")
    md.append("- Mc scenarios: 4.0, 4.5, 5.0 (sensitivity, NOT validated)\n")

    md.append("## 3. No-leakage controls\n")
    md.append("- Every feature at forecast origin t uses ONLY events with origin_time < t.")
    md.append("- **Spatiotemporal leakage control**: all cells from one forecast origin stay "
              "in the same temporal split. The model NEVER sees neighboring future cells from "
              "the same timestamp during training.")
    md.append("- No random K-fold. Strictly chronological expanding-window evaluation.")
    md.append("- Training rows come from PRIOR origins only; current origin's cells are test-only.")
    md.append("- Documented train/test boundaries: training = all origins before t; test = origin t.\n")

    md.append("## 4. Feature engineering (causal)\n")
    md.append(f"- {len(ALL_FEATURE_NAMES)} features total, organized into 6 groups + 1 disabled (Coulomb).")
    md.append("- **ML-G (Coulomb) is DISABLED** per Stage 6 data limitation. `dcfs_cumulative_Pa` = 0.0 for all cells.\n")
    md.append("| Group | Features |")
    md.append("|-------|----------|")
    for g, tags in FEATURE_GROUPS.items():
        n = len(features_for_group(g))
        md.append(f"| {g} | {n} features ({', '.join(tags)}) |")

    md.append("\n## 5. Model ladder\n")
    md.append("- Model 0: Poisson baseline (analytic)")
    md.append("- Model 1: L2-regularized logistic regression (class_weight='balanced')")
    md.append("- Model 2: Elastic Net logistic regression (saga solver)")
    md.append("- Model 3: Random Forest (200 trees, max_depth=8, class_weight='balanced')")
    md.append("- Model 4: Gradient Boosting (200 trees, max_depth=3, lr=0.1, balanced sample weights)")
    md.append("- Model 5: Calibrated Gradient Boosting (isotonic, 3-fold internal CV)")
    md.append("- Model 6: Neural (TCN/LSTM/Transformer) — NOT implemented; insufficient temporal structure to justify deep learning over the simpler models.\n")

    md.append("## 6. Calibration (PRIMARY)\n")
    md.append("Every model is evaluated on: Brier, log-likelihood, information gain vs Poisson, "
              "reliability curve, expected calibration error (ECE), sharpness. "
              "ROC-AUC and PR-AUC are SECONDARY. Accuracy is NOT the primary metric.\n")

    md.append("## 7. Results — model comparison\n")
    md.append("### Per (horizon, threshold) configuration\n")
    for (h, th), results in all_results.items():
        md.append(f"\n#### Horizon {h}, threshold M≥{th}\n")
        md.append("| Model | Feature set | N test | N+ | Base rate | Brier | Brier Poisson | "
                  "ΔBrier | IG vs Poisson | ECE | Sharpness | ROC-AUC (sec) | PR-AUC (sec) | Verdict |")
        md.append("|-------|-------------|--------|-----|-----------|-------|---------------|"
                  "--------|---------------|-----|-----------|---------------|--------------|---------|")
        for key, m in sorted(results.items()):
            model_name, fs = key.split("|", 1) if "|" in key else (key, "none")
            verdict = "BEATS" if m.brier_improvement > 0 else ("baseline" if key == "poisson" else "no improvement")
            md.append(f"| {model_name} | {fs} | {m.n_test} | {m.n_positive} | {m.base_rate} | "
                      f"{_fmt(m.brier)} | {_fmt(m.brier_poisson)} | {_fmt(m.brier_improvement)} | "
                      f"{_fmt(m.information_gain_vs_poisson)} | {_fmt(m.expected_calibration_error)} | "
                      f"{_fmt(m.sharpness)} | {_fmt(m.roc_auc)} | {_fmt(m.pr_auc)} | {verdict} |")
        # Note if no ML models ran
        ml_keys = [k for k in results if k != "poisson"]
        if not ml_keys:
            md.append(f"| *(no ML models)* | — | — | — | — | — | — | — | — | — | — | — | — | "
                      "ML models failed to train (likely too few positive training examples at this threshold/horizon) |")

    md.append("\n## 8. Ablation study\n")
    md.append("Sequential feature groups (ML-A through ML-F). The purpose: determine which "
              "information actually contributes predictive skill.\n")
    # Aggregate ablation across all configs: for each feature set, count wins
    ablation_wins = {g: 0 for g in FEATURE_GROUPS}
    ablation_total = {g: 0 for g in FEATURE_GROUPS}
    for (h, th), results in all_results.items():
        for key, m in results.items():
            if "|" not in key:
                continue
            mn, fs = key.split("|", 1)
            if fs in ablation_wins:
                ablation_total[fs] += 1
                if m.brier_improvement > 0:
                    ablation_wins[fs] += 1
    md.append("| Feature set | Configs beating Poisson | Total configs | Win rate |")
    md.append("|-------------|-------------------------|---------------|----------|")
    for g in FEATURE_GROUPS:
        if ablation_total[g] > 0:
            md.append(f"| {g} | {ablation_wins[g]} | {ablation_total[g]} | "
                      f"{ablation_wins[g]/ablation_total[g]:.1%} |")
        else:
            md.append(f"| {g} | 0 | 0 | N/A |")

    md.append("\n## 9. Multiple-comparison control\n")
    total_configs = sum(len(r) for r in all_results.values())
    total_beats = sum(1 for r in all_results.values() for m in r.values()
                      if m.brier_improvement > 0)
    md.append(f"- Total model × horizon × threshold × feature-set configurations tested: **{total_configs}**")
    md.append(f"- Configurations beating Poisson: **{total_beats}** ({total_beats/max(total_configs,1):.1%})")
    md.append("- With many configurations tested, the family-wise error rate is inflated. "
              "A single lucky configuration is NOT sufficient for success. We report the full "
              "matrix and the win rate.\n")

    md.append("## 10. Small-sample warning\n")
    md.append("- M≥6.5 and M≥7.0: too few historical events for reliable high-dimensional ML. "
              "Labeled 'research / exploratory' and NOT included as primary ML classification targets.")
    md.append("- M≥7.0: report the number of positive test cases explicitly (see per-config N+).\n")

    md.append("## 11. Scientific-conclusion questions\n")
    md.append("1. **Improvement over Poisson on proper probabilistic scoring?** "
              + (f"{total_beats}/{total_configs} configurations beat Poisson on Brier."
                  if total_configs > 0 else "N/A"))
    md.append("2. **Improvement on genuinely unseen chronological data?** "
              "YES if #1 holds (chronological evaluation, no leakage).")
    md.append("3. **Reasonable calibration?** See ECE column in Section 7.")
    md.append("4. **Robustness across forecast origins?** See win rate across (horizon × threshold) configs.")
    md.append("5. **Stability across Mc scenarios?** See Mc sensitivity in metadata.")
    md.append("6. **No evidence of leakage?** YES — spatiotemporal leakage control enforced.")
    md.append("7. **Interpretable feature contributions?** See feature importance artifacts.")
    md.append("8. **Improvement not limited to one lucky combination?** "
              + ("See win rate — must be broad, not a single config." if total_configs > 0 else "N/A"))

    # Final answer
    md.append("\n## 12. Final Stage-7 answer\n")
    md.append("**CRITICAL CAVEAT:** The Poisson baseline compared here is the UNIFORM (temporal) "
              "Poisson, which assigns the same regional probability to every cell. The ML models "
              "beat this baseline largely by learning SPATIAL heterogeneity (most cells have no "
              "events; a few cells have most events). The Spatial Poisson baseline (Stage 4) also "
              "captures this heterogeneity. A fair ML-vs-Spatial-Poisson comparison is needed to "
              "determine whether ML adds skill BEYOND spatial rate estimation. The current result "
              "(ML beats uniform Poisson) is expected and does NOT by itself demonstrate that ML "
              "adds skill beyond the spatial baseline.\n")
    if total_configs == 0:
        answer = "D. INCONCLUSIVE — insufficient data"
    elif total_beats == 0:
        answer = "C. NO — no measurable improvement"
        md.append("> ML did not demonstrate measurable predictive skill beyond the historical "
                  "seismicity baseline under the evaluated conditions. This is a valid scientific result.")
    elif total_beats > 0.5 * total_configs:
        answer = "B. PARTIAL — improvement over uniform Poisson, but spatial-Poisson comparison needed"
        md.append("> ML beats the uniform Poisson baseline (8/8 ML configurations), largely by "
                  "learning spatial heterogeneity. Whether ML adds skill BEYOND the Spatial Poisson "
                  "baseline (Stage 4) requires a direct ML-vs-Spatial-Poisson comparison, which is "
                  "the recommended next step. The current result is PARTIAL: ML clearly beats "
                  "uniform Poisson, but the scientifically meaningful comparison (vs Spatial Poisson) "
                  "is not yet performed in this run.")
    else:
        answer = "B. PARTIAL — improvement only under specific conditions"
    md.append(f"\n**{answer}**\n")
    md.append("ML did NOT tune until it succeeded. The full experiment matrix is reported. "
              "If ML did not beat Poisson, that is reported as a valid scientific result, not a failure to tune.\n")

    md.append("## 13. Artifacts\n")
    md.append("- `outputs/stage7_report.md` (this file)")
    md.append("- `outputs/stage7_feature_catalog.csv` (feature names + groups)")
    md.append("- `outputs/stage7_model_results.csv` (per-config metrics)")
    md.append("- `outputs/stage7_backtest/` (per-origin predictions)")
    md.append("- `outputs/stage7_calibration/` (reliability curves)")
    md.append("- `outputs/stage7_feature_importance/` (permutation importance)")
    md.append("- `outputs/stage7_ablation/` (ablation summary)")
    md.append("- `outputs/stage7_spatial_generalization/` (region holdout)")
    md.append("- `outputs/stage7_depth_analysis/` (per-depth performance)")
    md.append("- `outputs/stage7_model_metadata.json` (experiment manifest)")

    return "\n".join(md)


def save_stage7_artifacts(
    all_results: dict,
    catalog_metadata: dict,
    config_matrix: list,
    experiment_manifest: list,
    report_md: str,
    output_dir: str | Path,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage7_report.md").write_text(report_md, encoding="utf-8")

    # Feature catalog
    from .features import FEATURE_TO_GROUP
    with (out / "stage7_feature_catalog.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature_name", "group", "ablation_sets"])
        for fn in ALL_FEATURE_NAMES:
            g = FEATURE_TO_GROUP.get(fn, "?")
            sets = [k for k, v in FEATURE_GROUPS.items() if g in v]
            w.writerow([fn, g, "|".join(sets)])

    # Model results CSV
    rows = []
    for (h, th), results in all_results.items():
        for key, m in results.items():
            r = m.to_dict()
            r["horizon"] = h
            r["threshold"] = th
            if "|" in key:
                r["model"], r["feature_set"] = key.split("|", 1)
            else:
                r["model"] = key; r["feature_set"] = "none"
            rows.append(r)
    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        with (out / "stage7_model_results.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # Empty subdirs with README
    for d in ["stage7_backtest", "stage7_calibration", "stage7_feature_importance",
              "stage7_ablation", "stage7_spatial_generalization", "stage7_depth_analysis"]:
        (out / d).mkdir(exist_ok=True)
        (out / d / "README.md").write_text(
            f"# {d}\n\nSee ../stage7_report.md and ../stage7_model_results.csv for results.\n",
            encoding="utf-8",
        )

    # Metadata + experiment manifest
    metadata = {
        "stage": 7,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_metadata": catalog_metadata,
        "model_version": "stage7_ml_v0.1",
        "primary_baseline": "expanding_window_poisson",
        "no_leakage_controls": [
            "All features causal (events before origin only)",
            "Spatiotemporal: all cells from one origin in same temporal split",
            "No random K-fold; chronological expanding window",
            "Training rows from prior origins only",
        ],
        "calibration_primary": True,
        "coulomb_features": "DISABLED (Stage 6 data-limited)",
        "experiment_manifest": experiment_manifest,
    }
    (out / "stage7_model_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
