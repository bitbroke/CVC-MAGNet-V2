"""Run native-resolution evaluations across S0, S1, and S2 checkpoints and synthesize a unified comparison report."""

import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_cvc_masks_native import (
    NativeMaskEvaluator,
    compute_patient_bootstrap_ci,
    freeze_and_save_audit_panel,
)


def run_all_evaluations():
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = [
        {
            "tag": "S0_frozen_final",
            "path": str(PROJECT_ROOT / "runs" / "pilot_experiment_expanded" / "checkpoints" / "final_coarse_dual_head.pt"),
            "img_size": 512,
            "type": "S0 Frozen Dual-Head (Final)",
        },
        {
            "tag": "S1_pretrained_final",
            "path": str(PROJECT_ROOT / "runs" / "s1_pretrained_segmentation" / "checkpoints" / "final_segmentation_v3.pt"),
            "img_size": 512,
            "type": "S1 Pretrained V3 512px (Final)",
        },
        {
            "tag": "S1_pretrained_best",
            "path": str(PROJECT_ROOT / "runs" / "s1_pretrained_segmentation" / "checkpoints" / "best_segmentation_v3.pt"),
            "img_size": 512,
            "type": "S1 Pretrained V3 512px (Best Val Soft Dice)",
        },
        {
            "tag": "S2_higher_res_final",
            "path": str(PROJECT_ROOT / "runs" / "s2_higher_resolution_768" / "checkpoints" / "final_segmentation_v3.pt"),
            "img_size": 768,
            "type": "S2 Pretrained V3 768px (Final)",
        },
        {
            "tag": "S2_higher_res_best",
            "path": str(PROJECT_ROOT / "runs" / "s2_higher_resolution_768" / "checkpoints" / "best_segmentation_v3.pt"),
            "img_size": 768,
            "type": "S2 Pretrained V3 768px (Best Val Soft Dice)",
        },
    ]

    eval_results = {}
    for ckpt in checkpoints:
        tag = ckpt["tag"]
        print(f"\n==================================================", flush=True)
        print(f"Evaluating: {tag} ({ckpt['type']})", flush=True)
        print(f"==================================================", flush=True)

        evaluator = NativeMaskEvaluator(
            checkpoint_path=ckpt["path"],
            inventory_parquet=str(PROJECT_ROOT / "data" / "annotation_inventory.parquet"),
            study_inventory_parquet=str(PROJECT_ROOT / "data" / "study_inventory.parquet"),
            splits_json=str(PROJECT_ROOT / "data" / "splits.json"),
            images_dir=r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train",
            img_size=ckpt["img_size"],
        )
        res = evaluator.evaluate_split(split_name="dev_pilot_val")
        eval_results[tag] = res

        # Save individual JSON
        out_json = reports_dir / f"native_eval_{tag}.json"
        summary_to_save = {
            "checkpoint": res["checkpoint"],
            "checkpoint_sha256": res["checkpoint_sha256"],
            "tag": tag,
            "type": ckpt["type"],
            "img_size": res["img_size"],
            "evaluation_grid": res["evaluation_grid"],
            "total_studies_evaluated": res["total_studies_evaluated"],
            "summary_by_threshold": res["summary_by_threshold"],
        }
        with open(out_json, "w") as f:
            json.dump(summary_to_save, f, indent=2)

        # Persist full per-study records (JSON and CSV) for complete reproducibility
        full_per_study_json = reports_dir / f"native_eval_{tag}_per_study.json"
        with open(full_per_study_json, "w") as f:
            json.dump(res["per_study_results"], f)

        # Export full per-study measurements table at tau=0.50
        all_study_records = []
        for sdata in res["per_study_results"]:
            sid = sdata["study_id"]
            t50 = sdata["threshold_metrics"]["tau_50"]
            tp = t50["tp"]
            fp = t50["fp"]
            fn = t50["fn"]
            prec = float(tp / max(tp + fp, 1))
            rec = float(tp / max(tp + fn, 1))
            cov10 = float(np.mean([cov["cov_10px"] for cov in t50["instance_coverages"]])) if t50["instance_coverages"] else float("nan")
            all_study_records.append({
                "study_id": sid,
                "patient_id": sdata.get("patient_id", ""),
                "is_positive": sdata["is_positive"],
                "binary_dice": t50["binary_dice"],
                "soft_dice": sdata.get("soft_dice_native", float("nan")),
                "precision": prec,
                "recall": rec,
                "centerline_coverage_10px": cov10,
                "false_foreground_area_px": fp,
                "false_foreground_area_pct": t50["false_area_pct_radiograph"],
                "orig_height": sdata["orig_size"][1],
                "orig_width": sdata["orig_size"][0],
            })
        df_all = pd.DataFrame(all_study_records)
        df_all.to_csv(reports_dir / f"native_eval_{tag}_all_studies_tau50.csv", index=False)


        freeze_and_save_audit_panel(res["per_study_results"], str(reports_dir), tag, threshold=0.50)

    # Compute bootstrap CI
    print("\nComputing Paired Patient-Cluster Bootstrap CIs...", flush=True)
    bootstrap_s0_s1 = compute_patient_bootstrap_ci(
        eval_results["S0_frozen_final"]["per_study_results"],
        eval_results["S1_pretrained_final"]["per_study_results"],
        t_key_a="tau_50",
        t_key_b="tau_50",
    )
    bootstrap_s1_s2 = compute_patient_bootstrap_ci(
        eval_results["S1_pretrained_final"]["per_study_results"],
        eval_results["S2_higher_res_final"]["per_study_results"],
        t_key_a="tau_50",
        t_key_b="tau_50",
    )

    # Persist bootstrap CI outputs for offline inspection
    with open(reports_dir / "bootstrap_ci_s0_s1.json", "w") as f:
        json.dump(bootstrap_s0_s1, f, indent=2)

    with open(reports_dir / "bootstrap_ci_s1_s2.json", "w") as f:
        json.dump(bootstrap_s1_s2, f, indent=2)



    # Build Unified Comparison Markdown Report
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    md_lines = [
        "# Canonical Native-Resolution CVC Segmentation Benchmark Report",
        "",
        "**Evaluation Methodology**:",
        "- **Grid**: All predictions un-letterboxed via bilinear interpolation to exact native radiograph pixel coordinates.",
        "- **Target**: Ground truth polylines rasterized at 10 native pixels on identical native image grids.",
        "- **Cohort**: `dev_pilot_val` (193 studies: 185 positive, 8 verified negative; zero padding contamination).",
        "- **Metrics**: Macro Binary Dice, Macro Soft Dice, Pooled Pixel Precision/Recall, True Native Centerline Coverage at 10px, Native False Area (px and % of radiograph area).",
        "",
        "## 1. Primary Fixed-Budget Comparison (1,000 Updates, Final Checkpoints)",
        "",
        "| Model | Resolution | $\\tau$ | Binary Dice | Soft Dice | Precision | Recall | Cov @ 10px | Missed @ 10px | Neg False Px | Neg False % |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    models_to_tabulate = [
        ("S0 Frozen Reference (Final)", "S0_frozen_final"),
        ("S1 Pretrained Baseline (Final)", "S1_pretrained_final"),
        ("S2 Higher-Res 768px (Final)", "S2_higher_res_final"),
    ]

    for model_label, m_tag in models_to_tabulate:
        res = eval_results[m_tag]
        sz = res["img_size"]
        for tau in [0.30, 0.50, 0.60]:
            t_key = f"tau_{int(tau*100)}"
            s = res["summary_by_threshold"][t_key]
            cov10 = s["centerline_coverage"]["cov_10px"] * 100.0
            miss10 = s["missed_catheter_rates"]["missed_rate_10px"] * 100.0
            n_px = s["negatives"]["mean_false_area_native_px"]
            n_pct = s["negatives"]["mean_false_area_pct_radiograph"]
            md_lines.append(
                f"| **{model_label}** | {sz}×{sz} | **{tau:.2f}** | {s['macro_binary_dice']:.4f} | {s['macro_soft_dice']:.4f} | "
                f"{s['pooled_precision']*100:.2f}% | {s['pooled_recall']*100:.2f}% | {cov10:.1f}% | {miss10:.1f}% | "
                f"{n_px:,.0f} px | {n_pct:.3f}% |"
            )

    md_lines.extend([
        "",
        "## 2. Development Best-Checkpoint Comparison (Selected by Validation Soft Dice)",
        "",
        "| Model | Resolution | $\\tau$ | Binary Dice | Soft Dice | Precision | Recall | Cov @ 10px | Missed @ 10px | Neg False Px | Neg False % |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    best_models = [
        ("S1 Pretrained Baseline (Best)", "S1_pretrained_best"),
        ("S2 Higher-Res 768px (Best)", "S2_higher_res_best"),
    ]
    for model_label, m_tag in best_models:
        res = eval_results[m_tag]
        sz = res["img_size"]
        for tau in [0.30, 0.50, 0.60]:
            t_key = f"tau_{int(tau*100)}"
            s = res["summary_by_threshold"][t_key]
            cov10 = s["centerline_coverage"]["cov_10px"] * 100.0
            miss10 = s["missed_catheter_rates"]["missed_rate_10px"] * 100.0
            n_px = s["negatives"]["mean_false_area_native_px"]
            n_pct = s["negatives"]["mean_false_area_pct_radiograph"]
            md_lines.append(
                f"| **{model_label}** | {sz}×{sz} | **{tau:.2f}** | {s['macro_binary_dice']:.4f} | {s['macro_soft_dice']:.4f} | "
                f"{s['pooled_precision']*100:.2f}% | {s['pooled_recall']*100:.2f}% | {cov10:.1f}% | {miss10:.1f}% | "
                f"{n_px:,.0f} px | {n_pct:.3f}% |"
            )

    md_lines.extend([
        "",
        "## 3. Matched-Recall Operating Point Analysis",
        "",
        "To prevent selective threshold bias, comparing models at matched recall regimes:",
        "",
        "### High-Recall Regime (Recall ~ 77–80%):",
    ])

    # Find closest threshold for each
    for m_label, m_tag in models_to_tabulate:
        res = eval_results[m_tag]
        # find threshold closest to 0.78 recall
        best_diff = 999.0
        best_t = None
        for tau in thresholds:
            t_k = f"tau_{int(tau*100)}"
            rec = res["summary_by_threshold"][t_k]["pooled_recall"]
            if abs(rec - 0.78) < best_diff:
                best_diff = abs(rec - 0.78)
                best_t = tau
        s = res["summary_by_threshold"][f"tau_{int(best_t*100)}"]
        md_lines.append(
            f"- **{m_label}** ($\\tau={best_t:.2f}$): Recall = **{s['pooled_recall']*100:.2f}%**, "
            f"Precision = **{s['pooled_precision']*100:.2f}%**, Binary Dice = **{s['macro_binary_dice']:.4f}**, "
            f"Centerline Cov @ 10px = **{s['centerline_coverage']['cov_10px']*100:.1f}%**, "
            f"Neg False Area = **{s['negatives']['mean_false_area_pct_radiograph']:.3f}%** ({s['negatives']['mean_false_area_native_px']:,.0f} px)"
        )

    md_lines.extend([
        "",
        "### Balanced-Recall Regime (Recall ~ 65–68%):",
    ])
    for m_label, m_tag in models_to_tabulate:
        res = eval_results[m_tag]
        best_diff = 999.0
        best_t = None
        for tau in thresholds:
            t_k = f"tau_{int(tau*100)}"
            rec = res["summary_by_threshold"][t_k]["pooled_recall"]
            if abs(rec - 0.66) < best_diff:
                best_diff = abs(rec - 0.66)
                best_t = tau
        s = res["summary_by_threshold"][f"tau_{int(best_t*100)}"]
        md_lines.append(
            f"- **{m_label}** ($\\tau={best_t:.2f}$): Recall = **{s['pooled_recall']*100:.2f}%**, "
            f"Precision = **{s['pooled_precision']*100:.2f}%**, Binary Dice = **{s['macro_binary_dice']:.4f}**, "
            f"Centerline Cov @ 10px = **{s['centerline_coverage']['cov_10px']*100:.1f}%**, "
            f"Neg False Area = **{s['negatives']['mean_false_area_pct_radiograph']:.3f}%** ({s['negatives']['mean_false_area_native_px']:,.0f} px)"
        )

    md_lines.extend([
        "",
        "## 4. Paired Patient-Cluster Bootstrap Confidence Intervals (95% CI, $\\tau=0.50$)",
        "",
        f"- **S1 Final vs S0 Frozen** (N={bootstrap_s0_s1['n_patients']} patients, {bootstrap_s0_s1['n_pairs']} positive studies):",
        f"  - $\\Delta$ Macro Binary Dice: **[{bootstrap_s0_s1['delta_macro_binary_dice_95ci'][0]:+.4f}, {bootstrap_s0_s1['delta_macro_binary_dice_95ci'][1]:+.4f}]**",
        f"  - $\\Delta$ Pooled Precision: **[{bootstrap_s0_s1['delta_pooled_precision_95ci'][0]*100:+.2f}%, {bootstrap_s0_s1['delta_pooled_precision_95ci'][1]*100:+.2f}%]**",
        f"  - $\\Delta$ Pooled Recall: **[{bootstrap_s0_s1['delta_pooled_recall_95ci'][0]*100:+.2f}%, {bootstrap_s0_s1['delta_pooled_recall_95ci'][1]*100:+.2f}%]**",
        "",
        f"- **S2 Final vs S1 Final** (N={bootstrap_s1_s2['n_patients']} patients, {bootstrap_s1_s2['n_pairs']} positive studies):",
        f"  - $\\Delta$ Macro Binary Dice: **[{bootstrap_s1_s2['delta_macro_binary_dice_95ci'][0]:+.4f}, {bootstrap_s1_s2['delta_macro_binary_dice_95ci'][1]:+.4f}]**",
        f"  - $\\Delta$ Pooled Precision: **[{bootstrap_s1_s2['delta_pooled_precision_95ci'][0]*100:+.2f}%, {bootstrap_s1_s2['delta_pooled_precision_95ci'][1]*100:+.2f}%]**",
        f"  - $\\Delta$ Pooled Recall: **[{bootstrap_s1_s2['delta_pooled_recall_95ci'][0]*100:+.2f}%, {bootstrap_s1_s2['delta_pooled_recall_95ci'][1]*100:+.2f}%]**",
        "",
        "## 5. Audit Panel Integrity",
        f"- Frozen Audit Panel established at `reports/frozen_50_study_audit_panel_ids.json` (50 studies total: 10 multi-CVC, 10 abnormal, 10 cluttered, 10 single-CVC, 8 negatives, 2 other).",
        f"- Separate per-model measurements saved: `audit_panel_measurements_S0_frozen_final.csv`, `audit_panel_measurements_S1_pretrained_final.csv`, `audit_panel_measurements_S2_higher_res_final.csv`.",
    ])

    report_md_path = reports_dir / "native_resolution_comparative_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"\nSaved Canonical Native Benchmark Report to {report_md_path}", flush=True)


if __name__ == "__main__":
    run_all_evaluations()
