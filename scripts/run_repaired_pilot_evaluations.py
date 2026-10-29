"""Run native-resolution evaluations on repaired S1 (512px) and S2 (768px) pilot models."""

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


def run_repaired_evaluations():
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    repaired_checkpoints = [
        {
            "tag": "S1_repaired_final",
            "path": str(PROJECT_ROOT / "runs" / "s1_repaired_512" / "checkpoints" / "final_segmentation_v3.pt"),
            "img_size": 512,
            "type": "S1 Repaired V3 512px (Final)",
        },
        {
            "tag": "S1_repaired_best",
            "path": str(PROJECT_ROOT / "runs" / "s1_repaired_512" / "checkpoints" / "best_segmentation_v3.pt"),
            "img_size": 512,
            "type": "S1 Repaired V3 512px (Best Val Soft Dice)",
        },
        {
            "tag": "S2_repaired_final",
            "path": str(PROJECT_ROOT / "runs" / "s2_repaired_768" / "checkpoints" / "final_segmentation_v3.pt"),
            "img_size": 768,
            "type": "S2 Repaired V3 768px (Final)",
        },
        {
            "tag": "S2_repaired_best",
            "path": str(PROJECT_ROOT / "runs" / "s2_repaired_768" / "checkpoints" / "best_segmentation_v3.pt"),
            "img_size": 768,
            "type": "S2 Repaired V3 768px (Best Val Soft Dice)",
        },
    ]

    eval_results = {}

    # Load precomputed S0 results for reference
    s0_summary_path = reports_dir / "native_eval_S0_frozen_final.json"
    s0_per_study_path = reports_dir / "native_eval_S0_frozen_final_per_study.json"
    if s0_summary_path.exists() and s0_per_study_path.exists():
        print("Loading baseline S0_frozen_final from previous native evaluation...", flush=True)
        with open(s0_summary_path, encoding="utf-8") as f:
            s0_summary = json.load(f)
        with open(s0_per_study_path, encoding="utf-8") as f:
            s0_per_study = json.load(f)
        eval_results["S0_frozen_final"] = {
            "checkpoint": s0_summary["checkpoint"],
            "checkpoint_sha256": s0_summary["checkpoint_sha256"],
            "img_size": s0_summary["img_size"],
            "summary_by_threshold": s0_summary["summary_by_threshold"],
            "per_study_results": s0_per_study,
        }
    else:
        print("Running native evaluation on S0_frozen_final...", flush=True)
        evaluator_s0 = NativeMaskEvaluator(
            checkpoint_path=str(PROJECT_ROOT / "runs" / "pilot_experiment_expanded" / "checkpoints" / "final_coarse_dual_head.pt"),
            inventory_parquet=str(PROJECT_ROOT / "data" / "annotation_inventory.parquet"),
            study_inventory_parquet=str(PROJECT_ROOT / "data" / "study_inventory.parquet"),
            splits_json=str(PROJECT_ROOT / "data" / "splits.json"),
            images_dir=r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train",
            img_size=512,
        )
        res_s0 = evaluator_s0.evaluate_split(split_name="dev_pilot_val")
        eval_results["S0_frozen_final"] = res_s0

    for ckpt in repaired_checkpoints:
        tag = ckpt["tag"]
        out_json = reports_dir / f"native_eval_{tag}.json"
        full_per_study_json = reports_dir / f"native_eval_{tag}_per_study.json"

        if out_json.exists() and full_per_study_json.exists():
            print(f"Loading cached native evaluation for {tag}...", flush=True)
            with open(out_json, encoding="utf-8") as f:
                s_data = json.load(f)
            with open(full_per_study_json, encoding="utf-8") as f:
                ps_data = json.load(f)
            eval_results[tag] = {
                "checkpoint": s_data["checkpoint"],
                "checkpoint_sha256": s_data["checkpoint_sha256"],
                "img_size": s_data["img_size"],
                "summary_by_threshold": s_data["summary_by_threshold"],
                "per_study_results": ps_data,
            }
            continue

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

        # Save individual summary JSON
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
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary_to_save, f, indent=2)

        # Persist full per-study records (JSON and CSV)
        full_per_study_json = reports_dir / f"native_eval_{tag}_per_study.json"
        with open(full_per_study_json, "w", encoding="utf-8") as f:
            json.dump(res["per_study_results"], f)

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
    print("\nComputing Paired Patient-Cluster Bootstrap CIs for Repaired Runs...", flush=True)
    bootstrap_s0_s1_repaired = compute_patient_bootstrap_ci(
        eval_results["S0_frozen_final"]["per_study_results"],
        eval_results["S1_repaired_final"]["per_study_results"],
        t_key_a="tau_50",
        t_key_b="tau_50",
    )
    bootstrap_s1_s2_repaired_final = compute_patient_bootstrap_ci(
        eval_results["S1_repaired_final"]["per_study_results"],
        eval_results["S2_repaired_final"]["per_study_results"],
        t_key_a="tau_50",
        t_key_b="tau_50",
    )
    bootstrap_s1_s2_repaired_best = compute_patient_bootstrap_ci(
        eval_results["S1_repaired_best"]["per_study_results"],
        eval_results["S2_repaired_best"]["per_study_results"],
        t_key_a="tau_50",
        t_key_b="tau_50",
    )

    with open(reports_dir / "bootstrap_ci_s0_s1_repaired.json", "w", encoding="utf-8") as f:
        json.dump(bootstrap_s0_s1_repaired, f, indent=2)

    with open(reports_dir / "bootstrap_ci_s1_s2_repaired_final.json", "w", encoding="utf-8") as f:
        json.dump(bootstrap_s1_s2_repaired_final, f, indent=2)

    with open(reports_dir / "bootstrap_ci_s1_s2_repaired_best.json", "w", encoding="utf-8") as f:
        json.dump(bootstrap_s1_s2_repaired_best, f, indent=2)

    # Build Comparative Report Markdown
    md_lines = [
        "# Repaired Native-Resolution CVC Segmentation Benchmark Report",
        "",
        "**Evaluation Methodology**:",
        "- **Rigorous Experimental Parity**: Exactly 1,000 updates, identical initial weights (`runs/initial_weights.pt`), identical effective batch size (4), identical exposure sequence and per-exposure augmentation parameters.",
        "- **Native Resolution Grid**: All predictions unletterboxed to original radiograph pixels and evaluated against 10px ground truth rasterized masks.",
        "- **Evaluation Cohort**: `dev_pilot_val` (193 studies: 185 positive, 8 verified negative; 132 unique patients; zero padding contamination).",
        "- **Metrics**: Macro Binary Dice, Native Soft Dice, Pooled Pixel Precision/Recall, Centerline Coverage @ 10px, Missed-path rate, False foreground on negative radiographs.",
        "",
        "## 1. Primary Repaired Comparison (1,000 Updates, Final Checkpoints)",
        "",
        "| Model | Resolution | $\\tau$ | Binary Dice | Soft Dice | Precision | Recall | Cov @ 10px | Missed @ 10px | Neg False Px | Neg False % |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    models_to_tabulate = [
        ("S0 Frozen Reference", "S0_frozen_final"),
        ("S1 Repaired 512px (Final)", "S1_repaired_final"),
        ("S2 Repaired 768px (Final)", "S2_repaired_final"),
    ]

    for model_label, m_tag in models_to_tabulate:
        res = eval_results[m_tag]
        sz = res["img_size"]
        for tau in [0.30, 0.40, 0.50, 0.60, 0.70]:
            t_key = f"tau_{int(tau*100)}"
            s = res["summary_by_threshold"][t_key]
            cov10 = s["centerline_coverage"]["cov_10px"] * 100.0
            miss10 = s.get("missed_catheter_rates", {}).get("missed_rate_10px", 0.0) * 100.0
            neg_fp = s["negatives"].get("mean_false_area_native_px", s["negatives"].get("mean_false_positive_pixels", 0.0))
            neg_pct = s["negatives"].get("mean_false_area_pct_radiograph", s["negatives"].get("mean_false_positive_area_pct_radiograph", 0.0))
            soft_d = s.get("macro_soft_dice", s.get("soft_dice_native_macro", float("nan")))
            if np.isnan(soft_d):
                soft_d = res.get("summary_by_threshold", {}).get("tau_50", {}).get("macro_soft_dice", float("nan"))
            md_lines.append(
                f"| **{model_label}** | {sz}×{sz} | **{tau:.2f}** | {s['macro_binary_dice']:.4f} | {soft_d:.4f} | "
                f"{s['pooled_precision']*100:.2f}% | {s['pooled_recall']*100:.2f}% | {cov10:.1f}% | {miss10:.1f}% | "
                f"{int(neg_fp):,} px | {neg_pct:.3f}% |"
            )

    md_lines.extend([
        "",
        "## 2. Repaired Best-Checkpoint Comparison (Selected by Val Soft Dice)",
        "",
        "| Model | Resolution | $\\tau$ | Binary Dice | Soft Dice | Precision | Recall | Cov @ 10px | Missed @ 10px | Neg False Px | Neg False % |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    best_models_to_tabulate = [
        ("S1 Repaired 512px (Best)", "S1_repaired_best"),
        ("S2 Repaired 768px (Best)", "S2_repaired_best"),
    ]

    for model_label, m_tag in best_models_to_tabulate:
        res = eval_results[m_tag]
        sz = res["img_size"]
        for tau in [0.30, 0.40, 0.50, 0.60, 0.70]:
            t_key = f"tau_{int(tau*100)}"
            s = res["summary_by_threshold"][t_key]
            cov10 = s["centerline_coverage"]["cov_10px"] * 100.0
            miss10 = s.get("missed_catheter_rates", {}).get("missed_rate_10px", 0.0) * 100.0
            neg_fp = s["negatives"].get("mean_false_area_native_px", s["negatives"].get("mean_false_positive_pixels", 0.0))
            neg_pct = s["negatives"].get("mean_false_area_pct_radiograph", s["negatives"].get("mean_false_positive_area_pct_radiograph", 0.0))
            soft_d = s.get("macro_soft_dice", s.get("soft_dice_native_macro", float("nan")))
            if np.isnan(soft_d):
                soft_d = res.get("summary_by_threshold", {}).get("tau_50", {}).get("macro_soft_dice", float("nan"))
            md_lines.append(
                f"| **{model_label}** | {sz}×{sz} | **{tau:.2f}** | {s['macro_binary_dice']:.4f} | {soft_d:.4f} | "
                f"{s['pooled_precision']*100:.2f}% | {s['pooled_recall']*100:.2f}% | {cov10:.1f}% | {miss10:.1f}% | "
                f"{int(neg_fp):,} px | {neg_pct:.3f}% |"
            )

    md_lines.extend([
        "",
        "## 3. Matched-Recall Operating Point Analysis",
        "",
        "Comparing models at matched recall regimes to eliminate operating point bias:",
        "",
        "### High-Recall Operating Regime (Recall ~ 76–80%):",
    ])

    for model_label, m_tag in [("S0 Frozen Reference", "S0_frozen_final"), ("S1 Repaired Final", "S1_repaired_final"), ("S2 Repaired Final", "S2_repaired_final")]:
        s = eval_results[m_tag]["summary_by_threshold"]["tau_30"]
        cov10 = s["centerline_coverage"]["cov_10px"] * 100.0
        neg_pct = s["negatives"].get("mean_false_area_pct_radiograph", s["negatives"].get("mean_false_positive_area_pct_radiograph", 0.0))
        neg_px = int(s["negatives"].get("mean_false_area_native_px", s["negatives"].get("mean_false_positive_pixels", 0.0)))
        md_lines.append(
            f"- **{model_label}** ($\\tau=0.30$): Recall = **{s['pooled_recall']*100:.2f}%**, Precision = **{s['pooled_precision']*100:.2f}%**, "
            f"Binary Dice = **{s['macro_binary_dice']:.4f}**, Centerline Cov @ 10px = **{cov10:.1f}%**, Neg False Area = **{neg_pct:.3f}%** ({neg_px:,} px)"
        )

    md_lines.extend([
        "",
        "### Balanced Operating Regime (Recall ~ 65–68%):",
    ])

    for model_label, m_tag in [("S0 Frozen Reference", "S0_frozen_final"), ("S1 Repaired Final", "S1_repaired_final"), ("S2 Repaired Final", "S2_repaired_final")]:
        s = eval_results[m_tag]["summary_by_threshold"]["tau_50"]
        cov10 = s["centerline_coverage"]["cov_10px"] * 100.0
        neg_pct = s["negatives"].get("mean_false_area_pct_radiograph", s["negatives"].get("mean_false_positive_area_pct_radiograph", 0.0))
        neg_px = int(s["negatives"].get("mean_false_area_native_px", s["negatives"].get("mean_false_positive_pixels", 0.0)))
        md_lines.append(
            f"- **{model_label}** ($\\tau=0.50$): Recall = **{s['pooled_recall']*100:.2f}%**, Precision = **{s['pooled_precision']*100:.2f}%**, "
            f"Binary Dice = **{s['macro_binary_dice']:.4f}**, Centerline Cov @ 10px = **{cov10:.1f}%**, Neg False Area = **{neg_pct:.3f}%** ({neg_px:,} px)"
        )

    md_lines.extend([
        "",
        "## 4. Paired Patient-Cluster Bootstrap Confidence Intervals (95% CI, $\\tau=0.50$)",
        "",
        f"- **S1 Repaired Final vs S0 Frozen Reference** (N={bootstrap_s0_s1_repaired['n_patients']} patients, {bootstrap_s0_s1_repaired['n_pairs']} positive studies):",
        f"  - $\\Delta$ Macro Binary Dice: **[{bootstrap_s0_s1_repaired['delta_macro_binary_dice_95ci'][0]:+.4f}, {bootstrap_s0_s1_repaired['delta_macro_binary_dice_95ci'][1]:+.4f}]**",
        f"  - $\\Delta$ Pooled Precision: **[{bootstrap_s0_s1_repaired['delta_pooled_precision_95ci'][0]*100:+.2f}%, {bootstrap_s0_s1_repaired['delta_pooled_precision_95ci'][1]*100:+.2f}%]**",
        f"  - $\\Delta$ Pooled Recall: **[{bootstrap_s0_s1_repaired['delta_pooled_recall_95ci'][0]*100:+.2f}%, {bootstrap_s0_s1_repaired['delta_pooled_recall_95ci'][1]*100:+.2f}%]**",
        "",
        f"- **S2 Repaired Final vs S1 Repaired Final** (N={bootstrap_s1_s2_repaired_final['n_patients']} patients, {bootstrap_s1_s2_repaired_final['n_pairs']} positive studies):",
        f"  - $\\Delta$ Macro Binary Dice: **[{bootstrap_s1_s2_repaired_final['delta_macro_binary_dice_95ci'][0]:+.4f}, {bootstrap_s1_s2_repaired_final['delta_macro_binary_dice_95ci'][1]:+.4f}]**",
        f"  - $\\Delta$ Pooled Precision: **[{bootstrap_s1_s2_repaired_final['delta_pooled_precision_95ci'][0]*100:+.2f}%, {bootstrap_s1_s2_repaired_final['delta_pooled_precision_95ci'][1]*100:+.2f}%]**",
        f"  - $\\Delta$ Pooled Recall: **[{bootstrap_s1_s2_repaired_final['delta_pooled_recall_95ci'][0]*100:+.2f}%, {bootstrap_s1_s2_repaired_final['delta_pooled_recall_95ci'][1]*100:+.2f}%]**",
        "",
        f"- **S2 Repaired Best vs S1 Repaired Best** (N={bootstrap_s1_s2_repaired_best['n_patients']} patients, {bootstrap_s1_s2_repaired_best['n_pairs']} positive studies):",
        f"  - $\\Delta$ Macro Binary Dice: **[{bootstrap_s1_s2_repaired_best['delta_macro_binary_dice_95ci'][0]:+.4f}, {bootstrap_s1_s2_repaired_best['delta_macro_binary_dice_95ci'][1]:+.4f}]**",
        f"  - $\\Delta$ Pooled Precision: **[{bootstrap_s1_s2_repaired_best['delta_pooled_precision_95ci'][0]*100:+.2f}%, {bootstrap_s1_s2_repaired_best['delta_pooled_precision_95ci'][1]*100:+.2f}%]**",
        f"  - $\\Delta$ Pooled Recall: **[{bootstrap_s1_s2_repaired_best['delta_pooled_recall_95ci'][0]*100:+.2f}%, {bootstrap_s1_s2_repaired_best['delta_pooled_recall_95ci'][1]*100:+.2f}%]**",
        "",
        "## 5. Audit Panel & Reproducibility Provenance",
        f"- Full per-study measurements saved for each model in `reports/native_eval_<tag>_per_study.json` and `reports/native_eval_<tag>_all_studies_tau50.csv`.",
        f"- Frozen Audit Panel evaluated and persisted: `audit_panel_measurements_S1_repaired_final.csv`, `audit_panel_measurements_S2_repaired_final.csv`.",
    ])

    report_path = reports_dir / "repaired_native_resolution_comparative_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"\nSaved Repaired Native Benchmark Report to {report_path}", flush=True)


if __name__ == "__main__":
    run_repaired_evaluations()
