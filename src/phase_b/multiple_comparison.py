"""B8: Multiple-comparison control.

The project now has many models × thresholds × horizons × depth regimes ×
Mc scenarios × spatial configurations. This module:
  1. Reports the FULL tested matrix (no selective highlighting)
  2. Applies Bonferroni and Benjamini-Hochberg corrections
  3. Reports how many configurations beat Spatial Poisson and how many fail
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

logger = logging.getLogger("phase_b.b8")


def run_multiple_comparison_control(
    all_results: dict,  # (horizon, threshold) -> {evaluations, bootstrap}
    alpha: float = 0.05,
) -> dict:
    """Apply multiple-comparison control to the full experiment matrix.

    Parameters
    ----------
    all_results : the output of run_etas_vs_sp_comparison (B1)
    alpha : family-wise error rate
    """
    # Collect all p-values from permutation tests and bootstrap CIs
    comparisons = []
    for (h, th), res in all_results.items():
        boot = res.get("bootstrap", {})
        perm = res.get("permutation", {})
        for model_key in boot:
            if model_key == "spatial_poisson":
                continue
            b = boot[model_key]
            p = perm.get(model_key, {})
            comparisons.append({
                "horizon": h,
                "threshold": th,
                "model": model_key,
                "delta_brier": b.get("delta_brier_mean", float("nan")),
                "ci_lower": b.get("delta_brier_ci", (float("nan"),))[0],
                "ci_upper": b.get("delta_brier_ci", (float("nan"), float("nan")))[1],
                "p_value": p.get("p_value", float("nan")),
                "beats_sp": b.get("delta_brier_ci", (0, 0))[0] > 0,
            })

    n_comparisons = len(comparisons)
    if n_comparisons == 0:
        return {"n_comparisons": 0, "note": "No comparisons to correct."}

    # Bonferroni: α/n per comparison
    bonferroni_alpha = alpha / n_comparisons
    # Benjamini-Hochberg: sort p-values, find largest k where p(k) <= k/n * α
    p_values = [c["p_value"] for c in comparisons if not math.isnan(c["p_value"])]
    p_values_sorted = sorted(p_values)
    bh_threshold = alpha
    for k, p in enumerate(p_values_sorted, 1):
        if p <= (k / len(p_values_sorted)) * alpha:
            bh_threshold = p
        else:
            break

    # Apply corrections
    for c in comparisons:
        c["bonferroni_significant"] = (not math.isnan(c["p_value"])) and c["p_value"] < bonferroni_alpha
        c["bh_significant"] = (not math.isnan(c["p_value"])) and c["p_value"] <= bh_threshold

    # Summary
    n_beat_sp = sum(1 for c in comparisons if c["beats_sp"])
    n_bonf = sum(1 for c in comparisons if c["bonferroni_significant"])
    n_bh = sum(1 for c in comparisons if c["bh_significant"])

    return {
        "n_comparisons": n_comparisons,
        "alpha": alpha,
        "bonferroni_alpha": bonferroni_alpha,
        "bh_threshold": bh_threshold,
        "n_beat_sp_uncorrected": n_beat_sp,
        "n_significant_bonferroni": n_bonf,
        "n_significant_bh": n_bh,
        "comparisons": comparisons,
        "summary": (
            f"{n_comparisons} comparisons tested. "
            f"Uncorrected: {n_beat_sp} beat SP. "
            f"Bonferroni-significant (α={bonferroni_alpha:.4f}): {n_bonf}. "
            f"BH-significant (q={bh_threshold:.4f}): {n_bh}. "
            f"No selective highlighting: full matrix reported."
        ),
    }
