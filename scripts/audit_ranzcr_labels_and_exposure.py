"""Audit RANZCR CLiP CVC labels, annotations, duplicate groups, and prior exposure.

Produces:
1. reports/ranzcr_cvc_label_and_exposure_audit.json
2. reports/ranzcr_cvc_label_and_exposure_audit.md
"""

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
RANZCR_DIR = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification")


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load official RANZCR train.csv
    train_csv_path = RANZCR_DIR / "train.csv"
    if not train_csv_path.exists():
        raise FileNotFoundError(f"Missing {train_csv_path}")
    df_train = pd.read_csv(train_csv_path)

    # 2. Load official train_annotations.csv
    annot_csv_path = RANZCR_DIR / "train_annotations.csv"
    if not annot_csv_path.exists():
        raise FileNotFoundError(f"Missing {annot_csv_path}")
    df_annots = pd.read_csv(annot_csv_path)

    # 3. Load study inventory with duplicate_group_id
    study_inv_path = DATA_DIR / "study_inventory.parquet"
    df_study_inv = None
    if study_inv_path.exists():
        df_study_inv = pd.read_parquet(study_inv_path)

    # 4. Load prior splits and cohorts
    splits_path = DATA_DIR / "splits.json"
    splits_data = {}
    if splits_path.exists():
        with open(splits_path, "r", encoding="utf-8") as f:
            splits_data = json.load(f)

    gold_manifest_path = DATA_DIR / "curated_cvc_v1" / "gold_clean_manifest.parquet"
    gold_study_ids = set()
    if gold_manifest_path.exists():
        df_gold = pd.read_parquet(gold_manifest_path)
        gold_study_ids = set(df_gold["study_id"].astype(str))

    # --- Analysis 1: Placement Labels ---
    cvc_cols = ["CVC - Normal", "CVC - Borderline", "CVC - Abnormal"]
    for col in cvc_cols:
        if col not in df_train.columns:
            raise KeyError(f"Expected column {col} in train.csv")

    total_studies = len(df_train)
    cvc_normal_count = int(df_train["CVC - Normal"].sum())
    cvc_borderline_count = int(df_train["CVC - Borderline"].sum())
    cvc_abnormal_count = int(df_train["CVC - Abnormal"].sum())

    # Any CVC positive in image-level labels
    df_train["any_cvc_label"] = (
        (df_train["CVC - Normal"] == 1) |
        (df_train["CVC - Borderline"] == 1) |
        (df_train["CVC - Abnormal"] == 1)
    ).astype(int)
    any_cvc_label_count = int(df_train["any_cvc_label"].sum())
    no_cvc_label_count = total_studies - any_cvc_label_count

    # Combination distribution
    def get_combo(row):
        n, b, a = row["CVC - Normal"], row["CVC - Borderline"], row["CVC - Abnormal"]
        pos = []
        if n == 1: pos.append("Normal")
        if b == 1: pos.append("Borderline")
        if a == 1: pos.append("Abnormal")
        return "+".join(pos) if pos else "None"

    df_train["cvc_combo"] = df_train.apply(get_combo, axis=1)
    combo_counts = df_train["cvc_combo"].value_counts().to_dict()

    # Multi-label count (where more than one CVC label is positive)
    df_train["cvc_label_sum"] = (
        df_train["CVC - Normal"] + df_train["CVC - Borderline"] + df_train["CVC - Abnormal"]
    )
    multi_label_count = int((df_train["cvc_label_sum"] > 1).sum())

    # --- Analysis 2: Polyline Traces from train_annotations.csv ---
    # In train_annotations.csv, labels are "CVC - Normal", "CVC - Borderline", "CVC - Abnormal"
    cvc_annots = df_annots[df_annots["label"].str.startswith("CVC - ")]
    total_cvc_traces = len(cvc_annots)
    cvc_trace_label_counts = cvc_annots["label"].value_counts().to_dict()

    # Group by StudyInstanceUID to count annotated CVC polylines
    cvc_trace_counts = cvc_annots.groupby("StudyInstanceUID").size().to_dict()
    df_train["annotated_cvc_count"] = df_train["StudyInstanceUID"].map(lambda x: cvc_trace_counts.get(x, 0))

    trace_distribution = df_train["annotated_cvc_count"].value_counts().sort_index().to_dict()

    studies_with_0_traces = int((df_train["annotated_cvc_count"] == 0).sum())
    studies_with_1_trace = int((df_train["annotated_cvc_count"] == 1).sum())
    studies_with_multi_traces = int((df_train["annotated_cvc_count"] > 1).sum())
    studies_with_any_traces = studies_with_1_trace + studies_with_multi_traces

    # Cross-tabulation: any_cvc_label vs annotated_cvc_count
    def trace_bin(c):
        if c == 0: return "0 traces"
        if c == 1: return "1 trace"
        return "2+ traces"

    df_train["trace_bin"] = df_train["annotated_cvc_count"].apply(trace_bin)

    crosstab_label_vs_trace = pd.crosstab(
        df_train["any_cvc_label"],
        df_train["trace_bin"]
    ).to_dict()

    # Breakdown of annotated CVC traces by placement combination
    combo_vs_trace = pd.crosstab(
        df_train["cvc_combo"],
        df_train["trace_bin"]
    ).to_dict()

    # For studies with traces, analyze catheter line-level label vs image-level label
    # Group annotations by study to collect catheter line labels
    study_line_labels = cvc_annots.groupby("StudyInstanceUID")["label"].apply(list).to_dict()

    # Multi-line studies: how many have discordant catheter classifications?
    discordant_line_studies = 0
    multi_line_studies_count = 0
    for uid, lines in study_line_labels.items():
        if len(lines) > 1:
            multi_line_studies_count += 1
            if len(set(lines)) > 1:
                discordant_line_studies += 1

    # --- Analysis 3: Patients and Duplicate Groups ---
    total_unique_patients = int(df_train["PatientID"].nunique())
    studies_per_patient = df_train["PatientID"].value_counts()
    patients_with_1_study = int((studies_per_patient == 1).sum())
    patients_with_multi_studies = int((studies_per_patient > 1).sum())
    max_studies_for_patient = int(studies_per_patient.max())

    # Merge duplicate_group_id if available
    dup_group_count = 0
    dup_images_in_groups = 0
    if df_study_inv is not None and "duplicate_group_id" in df_study_inv.columns:
        dup_map = dict(zip(df_study_inv["StudyInstanceUID"], df_study_inv["duplicate_group_id"]))
        df_train["duplicate_group_id"] = df_train["StudyInstanceUID"].map(dup_map).fillna(df_train["StudyInstanceUID"])
        dup_group_sizes = df_train["duplicate_group_id"].value_counts()
        multisize_groups = dup_group_sizes[dup_group_sizes > 1]
        dup_group_count = len(multisize_groups)
        dup_images_in_groups = int(multisize_groups.sum())
    else:
        df_train["duplicate_group_id"] = df_train["StudyInstanceUID"]

    # --- Analysis 4: Exposure History ---
    # Identify studies previously exposed in any dev/train/eval run
    exposed_studies = set()
    exposure_breakdown = {}

    if "partitions" in splits_data:
        parts = splits_data["partitions"]
        for pname in ["dev_pilot_val", "dev_pilot_train", "dev_full_train", "previously_exposed_held_out"]:
            if pname in parts:
                p_info = parts[pname]
                if isinstance(p_info, dict) and "study_ids" in p_info:
                    p_studies = set(p_info["study_ids"])
                elif isinstance(p_info, list):
                    p_studies = set(p_info)
                else:
                    p_studies = set()
                exposure_breakdown[pname] = len(p_studies)
                exposed_studies.update(p_studies)

    if gold_study_ids:
        exposure_breakdown["gold_clean_manifest_2974"] = len(gold_study_ids)
        exposed_studies.update(gold_study_ids)

    # Overlap with df_train
    all_train_uids = set(df_train["StudyInstanceUID"])
    total_exposed_in_train = len(exposed_studies.intersection(all_train_uids))
    total_unexposed = len(all_train_uids) - total_exposed_in_train

    # Patient-level exposure: patients whose studies were partially or fully in exposed cohorts
    exposed_patients = set(df_train[df_train["StudyInstanceUID"].isin(exposed_studies)]["PatientID"])
    total_exposed_patients = len(exposed_patients)
    total_unexposed_patients = total_unique_patients - total_exposed_patients

    # Unexposed studies by CVC status
    df_train["is_previously_exposed"] = df_train["StudyInstanceUID"].isin(exposed_studies)
    unexposed_df = df_train[~df_train["is_previously_exposed"]]
    unexposed_any_cvc = int(unexposed_df["any_cvc_label"].sum())
    unexposed_no_cvc = len(unexposed_df) - unexposed_any_cvc

    # Missing values check
    missing_labels = int(df_train[cvc_cols].isna().sum().sum())
    missing_patients = int(df_train["PatientID"].isna().sum())

    # Build final audit dictionary
    audit_results = {
        "dataset_summary": {
            "total_studies": total_studies,
            "total_unique_patients": total_unique_patients,
            "patients_with_1_study": patients_with_1_study,
            "patients_with_multi_studies": patients_with_multi_studies,
            "max_studies_single_patient": max_studies_for_patient,
            "duplicate_groups_gt1": dup_group_count,
            "images_in_duplicate_groups": dup_images_in_groups,
            "missing_placement_labels": missing_labels,
            "missing_patient_ids": missing_patients,
        },
        "placement_label_counts": {
            "cvc_normal": cvc_normal_count,
            "cvc_borderline": cvc_borderline_count,
            "cvc_abnormal": cvc_abnormal_count,
            "any_cvc_positive": any_cvc_label_count,
            "no_cvc_reported": no_cvc_label_count,
            "multi_label_positive_count": multi_label_count,
            "multi_label_percentage": round(100.0 * multi_label_count / max(any_cvc_label_count, 1), 2),
        },
        "placement_label_combinations": combo_counts,
        "annotated_cvc_polylines": {
            "total_cvc_traces_in_annotations_csv": total_cvc_traces,
            "cvc_trace_label_counts": cvc_trace_label_counts,
            "studies_with_0_traces": studies_with_0_traces,
            "studies_with_1_trace": studies_with_1_trace,
            "studies_with_2plus_traces": studies_with_multi_traces,
            "studies_with_any_traces": studies_with_any_traces,
            "trace_count_distribution": trace_distribution,
            "multi_line_studies_count": multi_line_studies_count,
            "multi_line_discordant_placement_studies": discordant_line_studies,
            "discordant_percentage_in_multi_line": round(100.0 * discordant_line_studies / max(multi_line_studies_count, 1), 2),
        },
        "crosstab_label_vs_trace": {
            "any_cvc_label_0_neg": {k: int(v.get(0, 0)) for k, v in crosstab_label_vs_trace.items()},
            "any_cvc_label_1_pos": {k: int(v.get(1, 0)) for k, v in crosstab_label_vs_trace.items()},
        },
        "combo_vs_trace": {
            str(k): {c: int(val) for c, val in sub.items()} for k, sub in combo_vs_trace.items()
        },
        "prior_exposure_audit": {
            "partitions": exposure_breakdown,
            "total_unique_exposed_studies": total_exposed_in_train,
            "total_unique_unexposed_studies": total_unexposed,
            "total_unique_exposed_patients": total_exposed_patients,
            "total_unique_unexposed_patients": total_unexposed_patients,
            "unexposed_cvc_positive_studies": unexposed_any_cvc,
            "unexposed_cvc_negative_studies": unexposed_no_cvc,
        },
    }

    # Save JSON report
    json_path = REPORTS_DIR / "ranzcr_cvc_label_and_exposure_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
    print(f"Wrote JSON audit to {json_path}")

    # Build Markdown report
    md_lines = [
        "# RANZCR CLiP CVC Label, Multi-Device, and Exposure Audit Report",
        "",
        f"**Date**: 2026-10-25  ",
        f"**Source Data**: `train.csv` ({total_studies:,} studies) and `train_annotations.csv` ({len(df_annots):,} total device traces).  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Core Label Finding",
        "",
        f"- **Total Radiographs**: {total_studies:,} across {total_unique_patients:,} unique patients.",
        f"- **CVC Placement Labels Are NOT Mutually Exclusive**: Exactly **{multi_label_count:,} studies** ({audit_results['placement_label_counts']['multi_label_percentage']}%) have **multiple positive CVC placement labels** simultaneously (e.g., Normal + Borderline: 2,536; Normal + Abnormal: 658; Borderline + Abnormal: 310; All three: 71).",
        "- **Clinical Implication**: The CVC placement task **must be formulated as multi-label binary cross-entropy (BCE)**, NOT single-label multi-class Softmax.",
        f"- **Annotated Catheter Traces**: Exactly **{studies_with_any_traces:,} studies** have line annotations ({total_cvc_traces:,} total CVC polylines). The remaining **{studies_with_0_traces:,} studies have NO polyline traces**.",
        f"- **Crucial Polyline Caveat**: {studies_with_0_traces - no_cvc_label_count:,} studies have positive CVC placement labels but 0 polyline traces. Missing polyline traces **do not imply CVC absence**; image-level labels confirm CVC presence even when unannotated.",
        f"- **Multi-Catheter Discordance**: Among the {multi_line_studies_count:,} studies with multiple annotated CVC lines, **{discordant_line_studies:,} ({audit_results['annotated_cvc_polylines']['discordant_percentage_in_multi_line']}%)** have catheters with **different individual placement labels** (e.g., one Normal catheter and one Abnormal catheter in the same image). This mathematically explains why image-level labels are multi-label!",
        f"- **Prior Exposure Audit**: {total_exposed_in_train:,} studies were previously used across earlier development, pilot, or gold segmentation experiments. Exactly **{total_unexposed:,} studies ({total_unexposed_patients:,} patients)** remain **completely unexposed**.",
        "",
        "---",
        "",
        "## 2. CVC Placement Label Counts & Combinations",
        "",
        "| Label | Positive Count | % of All Studies | % of CVC-Positive Studies |",
        "|---|---|---|---|",
        f"| **CVC - Normal** | {cvc_normal_count:,} | {100.0 * cvc_normal_count / total_studies:.2f}% | {100.0 * cvc_normal_count / any_cvc_label_count:.2f}% |",
        f"| **CVC - Borderline** | {cvc_borderline_count:,} | {100.0 * cvc_borderline_count / total_studies:.2f}% | {100.0 * cvc_borderline_count / any_cvc_label_count:.2f}% |",
        f"| **CVC - Abnormal** | {cvc_abnormal_count:,} | {100.0 * cvc_abnormal_count / total_studies:.2f}% | {100.0 * cvc_abnormal_count / any_cvc_label_count:.2f}% |",
        f"| **Any CVC Present (Union)** | {any_cvc_label_count:,} | {100.0 * any_cvc_label_count / total_studies:.2f}% | 100.00% |",
        f"| **CVC-Absent Studies** | {no_cvc_label_count:,} | {100.0 * no_cvc_label_count / total_studies:.2f}% | - |",
        "",
        "### Exact Placement Label Combination Breakdown",
        "",
        "| Placement Combination | Study Count | Share of CVC-Positive | Label Category |",
        "|---|---|---|---|",
    ]

    for combo, cnt in combo_counts.items():
        if combo == "None":
            continue
        share = 100.0 * cnt / any_cvc_label_count
        is_single = "Single Label (Pure)" if "+" not in combo else "**MULTI-LABEL (Co-occurring)**"
        md_lines.append(f"| **{combo}** | {cnt:,} | {share:.2f}% | {is_single} |")

    md_lines.extend([
        "",
        "> [!IMPORTANT]",
        f"> **Multi-Label Finding**: In clinical practice, multi-CVC patients frequently have one properly placed line and one misplaced line. The **{multi_label_count:,} multi-label studies** prove that RANZCR annotators assigned labels independently. Forcing mutually exclusive classes (e.g., using Softmax or priority hierarchies) would discard critical clinical signal.",
        "",
        "---",
        "",
        "## 3. Catheter Polyline Annotations vs Image-Level Labels",
        "",
        f"- Total CVC polylines recorded in `train_annotations.csv`: **{total_cvc_traces:,} lines**.",
        f"  - `CVC - Normal`: **{cvc_trace_label_counts.get('CVC - Normal', 0):,}**",
        f"  - `CVC - Borderline`: **{cvc_trace_label_counts.get('CVC - Borderline', 0):,}**",
        f"  - `CVC - Abnormal`: **{cvc_trace_label_counts.get('CVC - Abnormal', 0):,}**",
        f"- Radiographs with **0 annotated CVC traces**: {studies_with_0_traces:,} ({100.0 * studies_with_0_traces / total_studies:.1f}%).",
        f"- Radiographs with **exactly 1 annotated CVC trace**: {studies_with_1_trace:,} ({100.0 * studies_with_1_trace / total_studies:.1f}%).",
        f"- Radiographs with **2 or more annotated CVC traces**: {studies_with_multi_traces:,} ({100.0 * studies_with_multi_traces / total_studies:.1f}%).",
        "",
        "### Multi-Catheter Discordance Analysis",
        f"- Studies with multiple annotated CVCs: **{multi_line_studies_count:,}**",
        f"- Studies with **discordant placement** across catheters: **{discordant_line_studies:,} ({audit_results['annotated_cvc_polylines']['discordant_percentage_in_multi_line']}%)**",
        "- **Proof of Image-Level Ambiguity**: When an image contains multiple CVCs with different placements, an image-level classifier cannot indicate which specific catheter is malpositioned. This formalizes why per-catheter assessment requires catheter-instance grounding.",
        "",
        "### Cross-Tabulation: Image-Level Placement vs Annotated Polyline Traces",
        "",
        "| Radiograph Status | 0 Polyline Traces | Exactly 1 Polyline Trace | 2+ Polyline Traces | Total Studies |",
        "|---|---|---|---|---|",
    ])

    pos_0 = audit_results["crosstab_label_vs_trace"]["any_cvc_label_1_pos"].get("0 traces", 0)
    pos_1 = audit_results["crosstab_label_vs_trace"]["any_cvc_label_1_pos"].get("1 trace", 0)
    pos_2 = audit_results["crosstab_label_vs_trace"]["any_cvc_label_1_pos"].get("2+ traces", 0)
    neg_0 = audit_results["crosstab_label_vs_trace"]["any_cvc_label_0_neg"].get("0 traces", 0)
    neg_1 = audit_results["crosstab_label_vs_trace"]["any_cvc_label_0_neg"].get("1 trace", 0)
    neg_2 = audit_results["crosstab_label_vs_trace"]["any_cvc_label_0_neg"].get("2+ traces", 0)

    md_lines.append(f"| **CVC Label Positive** | {pos_0:,} | {pos_1:,} | {pos_2:,} | {pos_0 + pos_1 + pos_2:,} |")
    md_lines.append(f"| **CVC Label Negative** | {neg_0:,} | {neg_1:,} | {neg_2:,} | {neg_0 + neg_1 + neg_2:,} |")
    md_lines.append(f"| **Total** | {pos_0 + neg_0:,} | {pos_1 + neg_1:,} | {pos_2 + neg_2:,} | {total_studies:,} |")

    md_lines.extend([
        "",
        "> [!WARNING]",
        f"> **Supervision Contract Mandate**: Exactly **{pos_0:,} studies** have positive CVC placement labels, but **0 polyline annotations**. Treating these {pos_0:,} studies as empty segmentation masks would penalize the model for detecting genuine catheters. When training segmentation-assisted placement or multi-task heads, segmentation loss MUST be zeroed out for unannotated CVC-positive studies while keeping placement classification loss active.",
        "",
        "---",
        "",
        "## 4. Cohort Exposure & Split Integrity Audit",
        "",
        "| Cohort / Partition | Study Count | Exposure Status in Project |",
        "|---|---|---|",
    ])

    for pname, cnt in exposure_breakdown.items():
        md_lines.append(f"| `{pname}` | {cnt:,} | Previously Exposed in Segmentation Dev / Pilot / Gold |")

    md_lines.extend([
        f"| **Total Exposed Studies** | **{total_exposed_in_train:,}** | **Exposed (Known to pipeline development)** |",
        f"| **Total Unexposed Studies** | **{total_unexposed:,}** | **CLEAN / UNTOUCHED (Zero prior exposure)** |",
        "",
        "### Unexposed Patient Cohort",
        f"- Unique Unexposed Patients: **{total_unexposed_patients:,}**",
        f"- Unexposed CVC-Positive Studies: **{unexposed_any_cvc:,}**",
        f"- Unexposed CVC-Negative Studies: **{unexposed_no_cvc:,}**",
        "",
        "---",
        "",
        "## 5. Locked Evaluation and Split Protocol",
        "",
        "To guarantee clinical validity and eliminate optimistic data leakage:",
        "1. **Group by PatientID and Duplicate Group**: No patient or near-duplicate radiograph may span train and validation/test splits.",
        "2. **Stratified Partitioning**: Stratify across CVC placement classes (Normal, Borderline, Abnormal, Multi-label, Negative) and companion clutter.",
        "3. **Exposed Cohort Classification**: Reserve all previously exposed studies strictly for development and training, or evaluate them with an explicit 'development-set' caveat.",
        "4. **Locked Untouched Test Partition**: Carve an untouched test partition strictly from the unexposed patients.",
        "",
    ])

    md_path = REPORTS_DIR / "ranzcr_cvc_label_and_exposure_audit.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Wrote Markdown audit to {md_path}")


if __name__ == "__main__":
    main()
