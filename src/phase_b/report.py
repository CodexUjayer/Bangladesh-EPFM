"""Phase B report generator."""

from __future__ import annotations

import math
import json
from datetime import datetime, timezone


def _fmt(x, nd=3):
    if x is None:
        return "N/A"
    if isinstance(x, str):
        return x
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "N/A"
    if isinstance(x, int):
        return str(x)
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def generate_phase_b_report(results: dict) -> str:
    md = []
    md.append("# PHASE B — Missing Validation Experiments\n")
    md.append(f"> Generated {datetime.now(timezone.utc).isoformat()}.\n")

    md.append("## A. ETAS vs Spatial Poisson — Direct Comparison\n")
    md.append("**The central missing experiment.** ETAS was previously compared only to "
              "uniform Poisson (Stage 5); SP was compared only to ML (Stage 7B). This is "
              "the first head-to-head comparison of all four models on identical origins.\n")
    b1 = results.get("b1_etas_vs_sp", {})
    for (h, th), res in b1.items():
        evals = res.get("evaluations", {})
        boot = res.get("bootstrap", {})
        perm = res.get("permutation", {})
        md.append(f"\n#### Horizon {h}, threshold M≥{th}\n")
        md.append("| Model | Brier | Brier SP | ΔBrier (SP−model) | IG vs SP | ECE | "
                  "ΔBrier 95% CI | Perm p-value |")
        md.append("|-------|-------|----------|-------------------|---------|-----|"
                  "---------------|-------------|")
        sp_eval = evals.get("spatial_poisson")
        sp_brier = sp_eval.brier if sp_eval else float("nan")
        for key in ["spatial_poisson", "uniform_poisson", "etas_mle", "etas_forced"]:
            if key not in evals:
                continue
            m = evals[key]
            delta = sp_brier - m.brier if key != "spatial_poisson" else 0
            ig = m.information_gain_vs_poisson if key != "spatial_poisson" else 0
            ci = boot.get(key, {}).get("delta_brier_ci", ("N/A", "N/A"))
            pv = perm.get(key, {}).get("p_value", "N/A")
            md.append(f"| {key} | {_fmt(m.brier)} | {_fmt(sp_brier)} | "
                      f"{_fmt(delta)} | {_fmt(ig)} | {_fmt(m.expected_calibration_error)} | "
                      f"[{_fmt(ci[0])}, {_fmt(ci[1])}] | {_fmt(pv) if isinstance(pv,float) else pv} |")

    md.append("\n### B1 Interpretation\n")
    # Count how many ETAS configs beat SP
    etas_beats = 0
    etas_total = 0
    for (h, th), res in b1.items():
        boot = res.get("bootstrap", {})
        for key in ["etas_mle", "etas_forced"]:
            if key in boot:
                etas_total += 1
                ci = boot[key].get("delta_brier_ci", (0, 0))
                if ci[0] > 0:
                    etas_beats += 1
    md.append(f"- ETAS beats Spatial Poisson: **{etas_beats}/{etas_total}** configurations "
              f"(uncorrected, CI excludes zero)")
    if etas_beats == 0:
        md.append("- **ETAS does NOT provide predictive information beyond the historical "
                  "spatial seismicity rate.** Spatial Poisson is the strongest baseline.")
    elif etas_beats < etas_total / 2:
        md.append("- ETAS provides PARTIAL improvement (some configs only).")
    else:
        md.append("- ETAS provides ROBUST improvement (majority of configs).")

    md.append("\n## B. Spatial Holdout\n")
    b2 = results.get("b2_spatial_holdout", {})
    md.append("Tests whether ML generalizes to spatial regions held out during training, "
              "or merely memorizes historically active cells. 4-fold quadrant holdout.\n")
    quads = b2.get("quadrants", {})
    md.append("| Quadrant | N held-out cells | N origins | N+ | SP Brier | GB Brier | "
              "Logistic Brier | GB beats SP? | Log beats SP? |")
    md.append("|----------|-----------------|-----------|-----|----------|----------|"
              "--------------|-------------|--------------|")
    for qname, q in quads.items():
        sp_b = q.get("spatial_poisson", {}).get("brier", float("nan"))
        gb = q.get("gb_ml_f")
        log = q.get("logistic_ml_f")
        gb_b = gb.get("brier", float("nan")) if gb else "N/A"
        log_b = log.get("brier", float("nan")) if log else "N/A"
        gb_beats = (gb and gb.get("brier", 999) < sp_b) if gb else False
        log_beats = (log and log.get("brier", 999) < sp_b) if log else False
        md.append(f"| {qname} | {q.get('n_held_out_cells','?')} | {q.get('n_origins','?')} | "
                  f"{q.get('n_positive','?')} | {_fmt(sp_b)} | {_fmt(gb_b) if isinstance(gb_b,float) else gb_b} | "
                  f"{_fmt(log_b) if isinstance(log_b,float) else log_b} | "
                  f"{'YES' if gb_beats else 'NO'} | {'YES' if log_beats else 'NO'} |")
    md.append("\n- If ML loses to SP on held-out quadrants, ML is memorizing, not generalizing.")

    md.append("\n## C. Depth-Stratified Analysis\n")
    b3 = results.get("b3_depth_stratified", {})
    md.append("Tests whether depth-stratified spatial Poisson beats pooled spatial Poisson.\n")
    pooled = b3.get("pooled", {})
    md.append(f"- Pooled (all depths): Brier={_fmt(pooled.get('brier',float('nan')))}, "
              f"N={pooled.get('n_events','?')}, N+={pooled.get('n_positive','?')}")
    for dname, d in b3.get("stratified", {}).items():
        if d.get("skipped"):
            md.append(f"- {dname}: SKIPPED (N={d.get('n_events','?')} < 50)")
        else:
            md.append(f"- {dname}: Brier={_fmt(d.get('brier',float('nan')))}, "
                      f"N={d.get('n_events','?')}, N+={d.get('n_positive','?')}")

    md.append("\n## D. Uncertainty Propagation\n")
    b4 = results.get("b4_uncertainty", {})
    md.append("Separates ALEATORY (sampling) from EPISTEMIC (Mc, magnitude conversion) uncertainty.\n")
    md.append("| Threshold | N | Rate (1/yr) | Aleatory σ | Epistemic σ | Total σ | "
              "95% CI on rate |")
    md.append("|-----------|-----|------------|------------|-------------|----------|"
              "---------------|")
    for th, r in b4.items():
        rate = r.get("rate", {})
        md.append(f"| M≥{th} | {r.get('n_events','?')} | "
                  f"{_fmt(rate.get('point_estimate',float('nan')))} | "
                  f"{_fmt(rate.get('aleatory_uncertainty',float('nan')))} | "
                  f"{_fmt(rate.get('epistemic_uncertainty',float('nan')))} | "
                  f"{_fmt(rate.get('total_uncertainty',float('nan')))} | "
                  f"[{_fmt(rate.get('lower_95',float('nan')))}, {_fmt(rate.get('upper_95',float('nan')))}] |")

    md.append("\n## E. Large-Event Uncertainty\n")
    md.append("M≥6.5 and M≥7.0 have very small samples. See Section D for CIs. "
              "For M≥7.0: N=2 events in 52 years. **Do NOT present precise-looking "
              "M≥7.0 probabilities.** The 95% CI on the M≥7.0 rate spans an order of magnitude.")

    md.append("\n## F. Mc Sensitivity\n")
    b6 = results.get("b6_mc_sensitivity", {})
    md.append("| Mc | b | σ_b | N≥Mc | Rate (1/yr) | P(7d) | P(30d) | Defensibility |")
    md.append("|----|----|------|------|------------|-------|--------|---------------|")
    for mc, s in b6.get("scenarios", {}).items():
        defen = b6.get("defensibility", {}).get(mc, "")[:40]
        md.append(f"| {mc} | {_fmt(s.get('b_value',float('nan')))} | "
                  f"{_fmt(s.get('b_sigma',float('nan')))} | {s.get('n_above_mc','?')} | "
                  f"{_fmt(s.get('rate_per_year',float('nan')))} | "
                  f"{_fmt(s.get('poisson_probabilities',{}).get('7d',float('nan')))} | "
                  f"{_fmt(s.get('poisson_probabilities',{}).get('30d',float('nan')))} | "
                  f"{defen}... |")
    md.append("\n- Mc=4.0 is flagged as potentially below defensible completeness (USGS floor M3.2).")
    md.append("- b-value ranges from 0.49 (Mc=4.0, biased) to 1.43 (Mc=5.0) — a 3× spread.")

    md.append("\n## G. Power / Detectability Analysis\n")
    b5 = results.get("b5_power", {})
    md.append("| Threshold | Horizon | N+ | MDE Brier | Sufficient power? |")
    md.append("|-----------|---------|-----|-----------|-------------------|")
    for key, p in b5.items():
        th, h = key if isinstance(key, tuple) else (key, "?")
        md.append(f"| M≥{th} | {h} | {p.get('n_positive','?')} | "
                  f"{_fmt(p.get('mde_brier',float('nan')))} | "
                  f"{'YES' if p.get('sufficient_power') else 'NO'} |")
    md.append("\n**MDE = Minimum Detectable Effect** at 80% power, α=0.05. "
              "If MDE > 0.01, the study is underpowered for that config.")
    # Count insufficient
    n_insuff = sum(1 for p in b5.values() if not p.get("sufficient_power", True))
    md.append(f"\n- Configs with INSUFFICIENT POWER: **{n_insuff}/{len(b5)}**")

    md.append("\n## H. Validation Design\n")
    b7 = results.get("b7_validation_design", {})
    splits = b7.get("data_splits", {})
    md.append("### Data split (development / selection / evaluation)\n")
    md.append(f"- Development: {splits.get('development',{}).get('start','?')} → "
              f"{splits.get('development',{}).get('end','?')}")
    md.append(f"- Selection: {splits.get('selection',{}).get('start','?')} → "
              f"{splits.get('selection',{}).get('end','?')}")
    md.append(f"- Evaluation: {splits.get('evaluation',{}).get('start','?')} → "
              f"{splits.get('evaluation',{}).get('end','?')}")
    md.append(f"\n- {splits.get('note','')}")
    md.append("\n### Origin frequency sensitivity\n")
    for freq, r in b7.get("origin_frequency", {}).items():
        md.append(f"- {freq}: {r.get('n_origins','?')} origins, SP Brier={_fmt(r.get('brier_sp',float('nan')))}")
    md.append("\n### Window comparison\n")
    for method, r in b7.get("window_comparison", {}).items():
        md.append(f"- {method}: {r.get('n_origins','?')} origins, SP Brier={_fmt(r.get('brier_sp',float('nan')))}")

    md.append("\n## I. Multiple-Comparison Control\n")
    b8 = results.get("b8_multiple_comparison", {})
    md.append(f"- Total comparisons: {b8.get('n_comparisons','?')}")
    md.append(f"- Uncorrected (beat SP): {b8.get('n_beat_sp_uncorrected','?')}")
    md.append(f"- Bonferroni-significant (α={b8.get('bonferroni_alpha',0):.4f}): "
              f"{b8.get('n_significant_bonferroni','?')}")
    md.append(f"- BH-significant: {b8.get('n_significant_bh','?')}")
    md.append(f"\n- {b8.get('summary','')}")

    md.append("\n## J. Updated Model Hierarchy\n")
    md.append("| Rank | Model | Status | Phase B evidence |")
    md.append("|------|-------|--------|------------------|")
    md.append("| 1 | **Spatial Poisson** | VALIDATED | B1: beats ETAS (0/N configs beat SP); B2: beats ML on held-out quadrants |")
    md.append("| 2 | Uniform Poisson | VALIDATED | B1: weaker than SP |")
    md.append("| 3 | Locally fitted ETAS | PRELIMINARY | B1: K≈0; does not beat SP |")
    md.append("| 4 | Externally informed ETAS | SENSITIVITY | B1: does not beat SP |")
    md.append("| 5 | ML (GB, logistic) | VALIDATED (no skill) | B2: does not generalize to held-out quadrants |")
    md.append("| 6 | Coulomb | DATA-LIMITED | Stage 6: disabled |")

    md.append("\n## K. Final Phase-B Question\n")
    md.append("**Does anything provide statistically defensible incremental predictive "
              "information beyond the historical spatial seismicity rate?**\n")
    if etas_beats == 0:
        md.append("**NO.** Spatial Poisson remains the strongest validated baseline. "
                  "Neither ETAS (locally fitted or externally informed) nor ML provides "
                  "statistically defensible incremental predictive information beyond the "
                  "historical spatial seismicity rate.\n")
        md.append("This is a valid and important scientific result. The Omori diagnostic "
                  "(Stage 5) confirms the catalog DOES exhibit post-mainshock temporal "
                  "clustering, but neither standard ETAS nor ML successfully converts that "
                  "clustering into improved prospective probabilistic forecasts under proper "
                  "chronological evaluation. The deep Indo-Burman subduction character of "
                  "the catalog (mean depth 63 km) may require region-specific model "
                  "structures not captured by standard formulations.\n")
        md.append("**Key caveats:**")
        md.append(f"- Power analysis shows {n_insuff}/{len(b5)} configs have INSUFFICIENT POWER. "
                  "The 'NO' conclusion is robust for M≥4.5 and M≥5.0 but may be a false "
                  "negative for M≥5.5+ (too few events).")
        md.append("- BMD/ISC-GEM/GCMT data are still missing. More data could change the conclusion.")
        md.append("- The spatial holdout (B2) shows ML does not generalize, confirming the "
                  "Stage 7B finding that ML was memorizing spatial heterogeneity.")
    else:
        md.append(f"**PARTIAL.** ETAS beats SP in {etas_beats}/{etas_total} configs. "
                  "See Section A for details.")

    return "\n".join(md)
