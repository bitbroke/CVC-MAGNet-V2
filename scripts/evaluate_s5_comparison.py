"""Native-Resolution Evaluator and Benchmark for S5 2,500-Update Expansion Comparison.

Compares:
1. S3 Reference Baseline (768px, 1,000 updates continued from S2)
2. S4 Arm B Experiment (768px, 1,000 updates, Selective Hard Neg, Seed 43)
3. S5 Arm A: Duration Control (400 pos + 396 neg = 796 studies, 2,500 updates @ 768px, 50% Top-99 Hard Neg)
4. S5 Arm B: Expanded Cohort (800 pos + 396 neg = 1,196 studies, 2,500 updates @ 768px, 50% Top-99 Hard Neg)

Evaluated at native resolution on frozen dev_pilot_val (193 studies, 185 positive, 8 negative, 132 unique patients).
"""

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


def run_s5_evaluation():
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_to_eval = [
        {
            "tag": "S5_duration_control_final",
            "path": str(PROJECT_ROOT / "runs" / "s5_duration_control_2500" / "checkpoints" / "final_segmentation_v3.pt"),
            "img_size": 768,
            "name": "S5 Arm A: Duration Control (400 pos, 2,500 updates)",
        },
        {
            "tag": "S5_expanded_cohort_final",
            "path": str(PROJECT_ROOT / "runs" / "s5_expanded_cohort_2500" / "checkpoints" / "final_segmentation_v3.pt"),
            "img_size": 768,
            "name": "S5 Arm B: Expanded Cohort (800 pos, 2,500 updates)",
        },
    ]

    eval_results = {}

    # 1. Load cached S3 Reference Baseline
    s3_summary_path = reports_dir / "native_eval_S3_experiment_final.json"
    s3_per_study_path = reports_dir / "native_eval_S3_experiment_final_per_study.json"
    if s3_summary_path.exists() and s3_per_study_path.exists():
        print("Loading cached S3 Reference Baseline...", flush=True)
        with open(s3_summary_path, encoding="utf-8") as f:
            s3_summary = json.load(f)
        with open(s3_per_study_path, encoding="utf-8") as f:
            s3_per_study = json.load(f)
        eval_results["S3_reference"] = {
            "checkpoint": s3_summary["checkpoint"],
            "checkpoint_sha256": s3_summary["checkpoint_sha256"],
            "img_size": s3_summary["img_size"],
            "summary_by_threshold": s3_summary["summary_by_threshold"],
            "per_study_results": s3_per_study,
        }

    # 2. Load cached S4 Seed 43 Experiment Baseline
    s4_summary_path = reports_dir / "native_eval_S4_experiment_seed43_final.json"
    s4_per_study_path = reports_dir / "native_eval_S4_experiment_seed43_final_per_study.json"
    if s4_summary_path.exists() and s4_per_study_path.exists():
        print("Loading cached S4 Seed 43 Experiment Baseline...", flush=True)
        with open(s4_summary_path, encoding="utf-8") as f:
            s4_summary = json.load(f)
        with open(s4_per_study_path, encoding="utf-8") as f:
            s4_per_study = json.load(f)
        eval_results["S4_seed43"] = {
            "checkpoint": s4_summary["checkpoint"],
            "checkpoint_sha256": s4_summary["checkpoint_sha256"],
            "img_size": s4_summary["img_size"],
            "summary_by_threshold": s4_summary["summary_by_threshold"],
            "per_study_results": s4_per_study,
        }

    # 3. Evaluate S5 checkpoints
    for ckpt in checkpoints_to_eval:
        tag = ckpt["tag"]
        out_json = reports_dir / f"native_eval_{tag}.json"
        full_per_study_json = reports_dir / f"native_eval_{tag}_per_study.json"

        if out_json.exists() and full_per_study_json.exists():
            print(f"Loading cached evaluation for {tag}...", flush=True)
            with open(out_json, encoding="utf-8") as f:
                res_sum = json.load(f)
            with open(full_per_study_json, encoding="utf-8") as f:
                res_per_study = json.load(f)
            eval_results[tag] = {
                "checkpoint": res_sum["checkpoint"],
                "checkpoint_sha256": res_sum["checkpoint_sha256"],
                "img_size": res_sum["img_size"],
                "summary_by_threshold": res_sum["summary_by_threshold"],
                "per_study_results": res_per_study,
            }
        else:
            print(f"\nRunning native evaluation for {tag} ({ckpt['path']})...", flush=True)
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

            with open(full_per_study_json, "w", encoding="utf-8") as f:
                json.dump(res["per_study_results"], f, indent=2)

            res_copy = dict(res)
            res_copy.pop("per_study_results", None)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(res_copy, f, indent=2)

    # 4. Bootstrap Confidence Intervals: Arm B (Expanded) vs Arm A (Duration Control)
    print("\nComputing Patient-Cluster Bootstrap CIs (Expanded vs Duration Control)...", flush=True)
    ci_expanded_vs_control = compute_patient_bootstrap_ci(
        per_study_a=eval_results["S5_duration_control_final"]["per_study_results"],
        per_study_b=eval_results["S5_expanded_cohort_final"]["per_study_results"],
        t_key_a="tau_50",
        t_key_b="tau_50",
        n_bootstrap=1000,
        seed=42,
    )
    with open(reports_dir / "bootstrap_ci_s5_expanded_vs_control.json", "w", encoding="utf-8") as f:
        json.dump(ci_expanded_vs_control, f, indent=2)

    # 5. Extract Metrics Table at tau=0.50
    table_rows = []
    cohort_keys = [
        ("S3_reference", "S3 Frozen Baseline (1,000 updates, 400 pos)", "-"),
        ("S4_seed43", "S4 Seed-43 Confirmed (1,000 updates, 400 pos, 50% Top-99 Hard)", "43"),
        ("S5_duration_control_final", "S5 Arm A: Duration Control (2,500 updates, 400 pos, 50% Top-99 Hard)", "43"),
        ("S5_expanded_cohort_final", "S5 Arm B: Expanded Cohort (2,500 updates, 800 pos, 50% Top-99 Hard)", "43"),
    ]

    for key, name, seed in cohort_keys:
        if key in eval_results:
            s50 = eval_results[key]["summary_by_threshold"]["tau_50"]
            table_rows.append({
                "Arm / Model": name,
                "Seed": seed,
                "Binary Dice": f"{s50['macro_binary_dice']:.4f}",
                "Precision": f"{s50['pooled_precision']*100:.2f}%",
                "Recall": f"{s50['pooled_recall']*100:.2f}%",
                "Cov @ 10px": f"{s50['centerline_coverage']['cov_10px']*100:.1f}%",
                "Missed @ 10px": f"{s50['missed_catheter_rates']['missed_rate_10px']*100:.1f}%",
                "Neg False Px": f"{s50['negatives']['mean_false_area_native_px']:.0f} px",
                "Neg False %": f"{s50['negatives']['mean_false_area_pct_radiograph']:.3f}%",
            })

    df_table = pd.DataFrame(table_rows)

    # 6. Generate Markdown Report
    ci_dice_low, ci_dice_high = ci_expanded_vs_control.get("delta_macro_binary_dice_95ci", [0.0, 0.0])
    ci_prec_low, ci_prec_high = ci_expanded_vs_control.get("delta_pooled_precision_95ci", [0.0, 0.0])
    ci_rec_low, ci_rec_high = ci_expanded_vs_control.get("delta_pooled_recall_95ci", [0.0, 0.0])

    s5_ctrl_50 = eval_results["S5_duration_control_final"]["summary_by_threshold"]["tau_50"]
    s5_exp_50 = eval_results["S5_expanded_cohort_final"]["summary_by_threshold"]["tau_50"]

    diff_dice_pt = s5_exp_50["macro_binary_dice"] - s5_ctrl_50["macro_binary_dice"]
    diff_prec_pt = s5_exp_50["pooled_precision"] - s5_ctrl_50["pooled_precision"]
    diff_rec_pt = s5_exp_50["pooled_recall"] - s5_ctrl_50["pooled_recall"]

    report_md = f"""# S5 Cohort Expansion Benchmark Report: Duration Control vs Expanded Cohort

**Date**: 2026-10-17  
**Evaluation Protocol**: Native radiograph grid on frozen `dev_pilot_val` (193 studies: 185 positive, 8 negative, 132 unique patients).  
**Experimental Design**:
- **Initialization**: Identical frozen S3 baseline (`runs/s3_experiment_hard_neg`).
- **Update Budget**: Exactly 2,500 successful optimizer updates for both arms.
- **Negative Sampling**: Exactly identical 50% Top-99 hard negative + 50% uniform negative policy (396 confirmed negatives).
- **Arm A (Duration Control)**: Current cohort (400 positive studies + 396 negatives = 796 studies).
- **Arm B (Expanded Cohort)**: Audited 2x cohort (800 positive studies + 396 negatives = 1,196 studies).
- **Geometry & Distal Tip Audit**: Repaired native-dimension checks; 35 boundary-reaching instances safely clipped for segmentation while preserving raw annotations; clipped endpoints explicitly non-equivalent to distal tips.

---

## 1. Primary Benchmark Comparison at $\\tau=0.50$ (Native Grid)

{df_table.to_markdown(index=False)}

---

## 2. Statistical Paired Differences (Expanded Cohort vs Duration Control at $\\tau=0.50$)

- **$\\Delta$ Macro Binary Dice**: **{diff_dice_pt:+.4f}** [95% CI: {ci_dice_low:+.4f}, {ci_dice_high:+.4f}]
- **$\\Delta$ Pooled Precision**: **{diff_prec_pt*100:+.2f}%** [95% CI: {ci_prec_low*100:+.2f}%, {ci_prec_high*100:+.2f}%]
- **$\\Delta$ Pooled Recall**: **{diff_rec_pt*100:+.2f}%** [95% CI: {ci_rec_low*100:+.2f}%, {ci_rec_high*100:+.2f}%]

---

## 3. Disentangling Data Volume vs Training Duration

1. **Duration Effect (S4 1,000 updates $\\rightarrow$ S5 Arm A 2,500 updates on 400 Positives)**:
   - Evaluates the effect of extended training on fixed data volume.
2. **Data Expansion Effect (S5 Arm A 400 Positives $\\rightarrow$ S5 Arm B 800 Positives at 2,500 updates)**:
   - Isolates the marginal value of 2x annotated positive data when training duration is held constant at 2,500 updates.
"""

    report_path = reports_dir / "s5_expansion_comparison_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\nSaved S5 benchmark report to: {report_path}", flush=True)


if __name__ == "__main__":
    run_s5_evaluation()
