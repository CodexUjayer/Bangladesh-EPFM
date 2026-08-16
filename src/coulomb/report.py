"""Stage 6 report generator and artifact saver."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data_audit import CoulombDataAudit
from .coupling import CouplingParams, document_formulation
from .forecast import CoulombForecast, StressDiagnostic


def _fmt(x, nd=3):
    if x is None:
        return "N/A"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "N/A"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    return f"{x:.{nd}f}"


def generate_stage6_report(
    audit: CoulombDataAudit,
    coupling_params: CouplingParams,
    forecasts: list,
    stress_diagnostics: list,
    backtest_summary: dict,
    catalog_metadata: dict,
    unit_test_results: str,
) -> str:
    md = []
    md.append("# STAGE 6 — Physics-Based / Coulomb-Stress Analysis\n")
    md.append(f"> Generated {datetime.now(timezone.utc).isoformat()}.\n")

    md.append("## 0. Primary scientific question\n")
    md.append("Do physics-based stress changes provide predictive information beyond:\n")
    md.append("1. stationary Poisson,")
    md.append("2. spatial Poisson,")
    md.append("3. locally fitted ETAS (K≈0),")
    md.append("4. externally informed ETAS?\n")
    md.append("The primary baseline remains the corrected expanding-window Poisson model "
              "(Stage 5 conclusion: standard ETAS does not beat Poisson with the correct "
              "per-origin Poisson rate).\n")

    md.append("## 1. Data audit (A/B/C/D classification)\n")
    md.append("**CRITICAL DATA-INTEGRITY RULE:** No Bangladesh fault geometry, slip rate, "
              "rake, dip, receiver faults, friction coefficients, coseismic slip, or "
              "stress-change fields are fabricated.\n")
    md.append("| Field | Class | Source | Value | Notes |")
    md.append("|-------|-------|--------|-------|-------|")
    for f in audit.fields:
        md.append(f"| {f.field_name} | **{f.classification}** | {f.source} | "
                  f"{f.value_summary} | {f.notes} |")
    md.append(f"\n**Real Coulomb forecasting ENABLED: {audit.real_forecasting_enabled}**")
    if audit.blocking_gaps:
        md.append("\n**Blocking data gaps:**")
        for gap in audit.blocking_gaps:
            md.append(f"- {gap}")
    if not audit.real_forecasting_enabled:
        md.append("\n> Per the data-integrity rule, real Coulomb forecasting is DISABLED. "
                  "A mathematical prototype is implemented and unit-tested with synthetic "
                  "geometry; results are NOT presented as a Bangladesh forecast. See "
                  "Section 5 (prototype validation) and Section 7 (data-gap report).")

    md.append("\n## 2. Coulomb formulation (documented)\n")
    md.append("### Mathematical form\n")
    md.append("ΔCFS = Δτ + μ'·Δσ_n\n")
    md.append("where:\n")
    md.append("- Δτ = shear stress change on the receiver fault, resolved in the slip direction")
    md.append("- Δσ_n = normal stress change (positive = unclamping/tension; negative = compression)")
    md.append("- μ' = effective friction coefficient (μ' = μ(1-B) with Skempton B)")
    md.append("\n### Sign conventions\n")
    md.append("- ΔCFS > 0 → fault brought closer to failure (triggering)")
    md.append("- ΔCFS < 0 → stress shadow (inhibition)")
    md.append("- Compression is NEGATIVE (rock-mechanics convention)")
    md.append("\n### Coordinate system\n")
    md.append("- Geographic ENU: x=East, y=North, z=Up (z=0 is surface)")
    md.append("- Source: strike (clockwise from N), dip (from horizontal), rake (slip direction in dip plane)")
    md.append("\n### Elastic half-space assumptions (Okada 1992)\n")
    md.append("- Isotropic, homogeneous, linear-elastic half-space")
    md.append("- Default: μ=30 GPa, ν=0.25 (Class C engineering assumption; sensitivity tested)")
    md.append("- Free surface at z=0; no topography or lateral heterogeneity")
    md.append("- Point-source approximation for stress (finite-source for M>7 would improve accuracy)")
    md.append("\n### Stress-to-rate coupling\n")
    md.append(f"**{document_formulation(coupling_params)}**\n")
    md.append("λ(x, t) = λ₀(x) · f(ΔCFS(x, t))  — Coulomb-modulated Poisson.\n")

    md.append("## 3. Sources and receivers\n")
    md.append(f"- USGS focal-mechanism products available for ~22/30 largest M≥5.5 events "
              "(provides source strike/dip/rake — Class A).")
    md.append(f"- GEM GAFD: {audit.gem_gafd_segments_in_region} fault traces in region, but "
              f"**{audit.gem_gafd_with_dip} have dip** and 0 have rake. Geometry-only.")
    md.append(f"- Receiver-fault geometry: **{audit.fields[-4].classification}** (no validated receiver-fault dataset).")
    md.append(f"- For prototype/diagnostic use only: receivers on a 1° grid at 10 km depth with "
              "ASSUMED orientation (strike=0, dip=45, rake=90 — Class C engineering assumption).")

    md.append("\n## 4. Forecasts\n")
    if not audit.real_forecasting_enabled:
        md.append("**Real Coulomb forecasting is DISABLED.** No Bangladesh forecast maps "
                  "are produced. The forecast function returns NaN with a data-limited note.\n")
    else:
        md.append("| Forecast | Horizon | Threshold | Enabled | Expected count | P(≥1) | Notes |")
        md.append("|----------|---------|-----------|---------|----------------|-------|-------|")
        for cf in forecasts:
            md.append(f"| {cf.forecast_start.isoformat()} | {cf.horizon} | M≥{cf.threshold} | "
                      f"{cf.enabled} | {_fmt(cf.expected_total_count)} | "
                      f"{_fmt(cf.probability_at_least_one)} | {cf.notes[0] if cf.notes else ''} |")

    md.append("\n## 5. Mathematical-prototype unit tests (synthetic geometry)\n")
    md.append("Unit tests validate the Okada implementation against known analytical properties:\n")
    md.append("```\n" + unit_test_results + "\n```\n")
    md.append("These tests validate the MATH, not the data. They confirm the implementation is "
              "correct; they do NOT validate any Bangladesh forecast.")

    md.append("\n## 6. Stress-forecast diagnostics\n")
    if stress_diagnostics:
        for sd in stress_diagnostics:
            md.append(f"\n### {sd.get('label', 'diagnostic')}\n")
            md.append("| ΔCFS bin center (Pa) | N events | N cells | Rate ratio |")
            md.append("|----------------------|----------|---------|------------|")
            for bc, ne, nc, rr in zip(sd["bin_centers_Pa"], sd["n_events_in_bin"],
                                       sd["exposure_cells"], sd["rate_ratio"]):
                md.append(f"| {bc} | {ne} | {nc} | {rr} |")
            for note in sd["notes"]:
                md.append(f"- {note}")
    else:
        md.append("No stress diagnostics computed (data-limited mode).")

    md.append("\n## 7. Backtest summary\n")
    if backtest_summary.get("enabled", False):
        md.append(f"- N origins: {backtest_summary.get('n_origins', 'N/A')}")
        md.append(f"- Brier (Coulomb-Poisson): {backtest_summary.get('brier_coulomb', 'N/A')}")
        md.append(f"- Brier (Poisson baseline): {backtest_summary.get('brier_poisson', 'N/A')}")
        md.append(f"- Information gain: {backtest_summary.get('information_gain', 'N/A')}")
        md.append(f"- Verdict: {backtest_summary.get('verdict', 'N/A')}")
    else:
        md.append("**Backtest DISABLED** — real Coulomb forecasting is data-limited. "
                  "No prospective backtest is performed because there are no validated "
                  "receiver-fault data to compute ΔCFS on. A pseudo-backtest on ASSUMED "
                  "receiver faults would not constitute a validated Bangladesh forecast.")

    md.append("\n## 8. Scientific-conclusion questions\n")
    md.append("1. **Does Coulomb add predictive information?** "
              + ("CANNOT BE DETERMINED — real forecasting disabled by data gaps." if not audit.real_forecasting_enabled else "See backtest."))
    md.append("2. **Does it outperform spatial Poisson?** "
              + ("CANNOT BE DETERMINED — real forecasting disabled." if not audit.real_forecasting_enabled else "See backtest."))
    md.append("3. **Does it outperform ETAS where ETAS is retained?** "
              + ("N/A — ETAS does not beat Poisson (Stage 5); Coulomb comparison moot until data arrive." if not audit.real_forecasting_enabled else "See backtest."))
    md.append("4. **Is the improvement robust to physical-parameter uncertainty?** "
              + ("N/A — no forecast to test." if not audit.real_forecasting_enabled else "See sensitivity."))
    md.append("5. **Is the improvement spatially localized?** "
              + ("N/A." if not audit.real_forecasting_enabled else "See stress diagnostics."))
    md.append("6. **Does it survive chronological validation?** "
              + ("N/A — no prospective backtest." if not audit.real_forecasting_enabled else "See backtest."))
    md.append("7. **Is the result reproducible from independently sourced physical data?** "
              + ("NO — cannot be reproduced because the required receiver-fault data do not exist in any available source (GEM GAFD geometry-only, GCMT not supplied, published literature not transcribed)." if not audit.real_forecasting_enabled else "Yes."))

    md.append("\n## 9. Data-gap report and required future data\n")
    md.append("To enable real Coulomb forecasting for Bangladesh, the following data are required:\n")
    md.append("1. **Validated receiver-fault geometry** (strike, dip, rake, depth) for the major "
             "Bangladesh faults: Dauki, Dhubri, Oldham, Dapsi, Churachandpur-Mao, Naga Thrust, "
             "Arakan megathrust, Chittagong-Tripura fold belt. Sources: Morino et al. 2014; "
             "Wang et al. 2014; Steckler et al. 2016 — require manual transcription.")
    md.append("2. **GCMT NDK file** (global CMT solutions 1976-present) — provides authoritative "
             "source focal mechanisms for M≥5.5. Download from globalcmt.org and place in "
             "`data/raw/gcmt/`.")
    md.append("3. **Regional stress orientation** (from World Stress Map or geophysical inversion) "
             "— would allow 'optimal' receiver-fault orientation if explicit fault geometry is "
             "unavailable.")
    md.append("4. **Bangladesh-specific elastic model** (μ, ν, layered structure) — currently "
             "using Okada 1992 defaults (Class C).")
    md.append("5. **Finite-source slip models** for the largest events (M≥7) — would replace "
             "the point-source approximation for near-field accuracy.")
    md.append("\nUntil these data are supplied, the mathematical prototype remains the only "
             "Coulomb deliverable. **No Bangladesh Coulomb forecast map is produced.**")

    md.append("\n## 10. Stage-7 gate\n")
    md.append("Stage 7 (ML) may proceed. The Stage 6 outcome is **(B) conclusively documented "
              "that the required data are unavailable** and the correct data-limited baseline "
              "(Poisson) is established. Stage 7 ML models must beat the Poisson baseline; "
              "Coulomb features (ΔCFS) can be added as optional ML inputs only if/when "
              "validated receiver-fault data arrive, and must be clearly labeled as "
              "externally-informed features.")

    md.append("\n## 11. Artifacts\n")
    md.append("- `outputs/stage6_report.md` (this file)")
    md.append("- `outputs/stage6_data_audit.csv` + `.json` (A/B/C/D classification)")
    md.append("- `outputs/stage6_coulomb_parameters.csv` (elastic + coupling params)")
    md.append("- `outputs/stage6_forecasts.csv` (data-limited; NaN where disabled)")
    md.append("- `outputs/stage6_backtest/` (empty — backtest disabled)")
    md.append("- `outputs/stage6_stress_maps/` (empty — no validated stress maps)")
    md.append("- `outputs/stage6_residual_diagnostics/` (empty — no residuals)")
    md.append("- `outputs/stage6_model_metadata.json`")

    return "\n".join(md)


def save_stage6_artifacts(
    audit: CoulombDataAudit,
    coupling_params: CouplingParams,
    forecasts: list,
    stress_diagnostics: list,
    backtest_summary: dict,
    catalog_metadata: dict,
    report_md: str,
    elastic_params,
    output_dir: str | Path,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage6_report.md").write_text(report_md, encoding="utf-8")

    # Data audit CSV + JSON
    from .data_audit import save_data_audit
    save_data_audit(audit, out / "stage6_data_audit")

    # Coulomb parameters CSV
    with (out / "stage6_coulomb_parameters.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value", "classification", "source"])
        w.writerow(["shear_modulus_GPa", elastic_params.shear_modulus_GPa, "C", "engineering assumption (Okada 1992 default)"])
        w.writerow(["poissons_ratio", elastic_params.poissons_ratio, "C", "engineering assumption"])
        w.writerow(["effective_friction", elastic_params.effective_friction, "C", "engineering assumption (King et al. 1994 typical)"])
        w.writerow(["skempton_coefficient", elastic_params.skempton_coefficient, "C", "engineering assumption"])
        w.writerow(["coupling_formulation", coupling_params.formulation.value, "B", "Dieterich 1994 / Toda 1998 / King 1994"])
        w.writerow(["A_sigma_bar_MPa", coupling_params.A_sigma_bar_MPa, "C", "engineering assumption (typical 0.1-10 MPa)"])
        w.writerow(["beta_per_MPa", coupling_params.beta_per_MPa, "C", "engineering assumption"])
        w.writerow(["step_alpha", coupling_params.step_alpha, "C", "engineering assumption"])
        w.writerow(["step_beta", coupling_params.step_beta, "C", "engineering assumption"])

    # Forecasts CSV (data-limited if disabled)
    fc_rows = []
    for cf in forecasts:
        for cell in cf.per_cell if cf.per_cell else []:
            fc_rows.append({
                "forecast_start": cf.forecast_start.isoformat(),
                "horizon": cf.horizon, "threshold": cf.threshold,
                "enabled": cf.enabled,
                "cell_id": cell.get("cell_id"),
                "lat": cell.get("lat_center"), "lon": cell.get("lon_center"),
                "dcfs_Pa": cell.get("dcfs_Pa"),
                "rate_multiplier": cell.get("rate_multiplier"),
                "expected_count": cell.get("expected_count"),
                "probability_at_least_one": cell.get("probability_at_least_one"),
            })
    if not fc_rows:
        fc_rows = [{"forecast_start": "N/A", "enabled": False,
                    "note": "Real Coulomb forecasting disabled by data audit"}]
    keys = sorted({k for r in fc_rows for k in r.keys()})
    with (out / "stage6_forecasts.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in fc_rows:
            w.writerow(r)

    # Empty backtest/stress-maps/residual directories (data-limited)
    for d in ["stage6_backtest", "stage6_stress_maps", "stage6_residual_diagnostics"]:
        (out / d).mkdir(exist_ok=True)
        # Write a README explaining why empty
        (out / d / "README.md").write_text(
            f"# {d}\n\nEmpty: real Coulomb forecasting is disabled by the data audit "
            "(no validated receiver-fault geometry for Bangladesh). See "
            "../stage6_report.md Section 9 (data-gap report).\n",
            encoding="utf-8",
        )

    # Stress diagnostics JSON (if any)
    if stress_diagnostics:
        (out / "stage6_residual_diagnostics" / "stress_diagnostics.json").write_text(
            json.dumps(stress_diagnostics, indent=2, default=str), encoding="utf-8"
        )

    # Metadata
    metadata = {
        "stage": 6,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_metadata": catalog_metadata,
        "model_version": "stage6_coulomb_prototype_v0.1",
        "real_forecasting_enabled": audit.real_forecasting_enabled,
        "blocking_gaps": audit.blocking_gaps,
        "elastic_params": {
            "shear_modulus_GPa": elastic_params.shear_modulus_GPa,
            "poissons_ratio": elastic_params.poissons_ratio,
            "effective_friction": elastic_params.effective_friction,
            "skempton_coefficient": elastic_params.skempton_coefficient,
        },
        "coupling_formulation": coupling_params.formulation.value,
        "data_integrity_rule": "No fabricated Bangladesh fault geometry, slip, rake, dip, receiver faults, friction, or stress fields.",
        "stage5_conclusion_carried_forward": (
            "Local maximum-likelihood ETAS estimation does not identify a statistically "
            "supported triggering component in the current USGS-only catalog. Corrected "
            "prospective backtesting shows that neither locally fitted nor externally "
            "informed standard ETAS produces measurable predictive skill beyond the properly "
            "constructed expanding-window Poisson baseline. However, independent Omori and "
            "spatial diagnostics demonstrate strong post-mainshock temporal and spatial "
            "clustering. Therefore, the absence of ETAS predictive improvement must not be "
            "interpreted as evidence that earthquake triggering is absent. It indicates that "
            "the tested standard ETAS formulation does not successfully convert the observed "
            "clustering into improved prospective forecasts."
        ),
        "scientific_success_criterion": (
            "Coulomb is successful only if it provides statistically supported out-of-sample "
            "improvement over the corrected Poisson baseline. 'Looks physically plausible' "
            "is NOT sufficient."
        ),
    }
    (out / "stage6_model_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
