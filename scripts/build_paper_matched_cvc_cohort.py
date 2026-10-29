"""Build a leakage-safe, single-annotated-CVC classification cohort.

This is a paper-inspired data cohort, not a claim that each radiograph is
clinically verified to contain only one catheter. It requires exactly one CVC
trace in RANZCR train_annotations.csv and exactly one matching image-level CVC
placement label in the locked patient/duplicate-separated manifest.

The script does not train a model or inspect the locked test images.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = Path(
    r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification"
    r"\ranzcr-clip-catheter-line-classification"
)
SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "placement_locked_manifest.parquet"
ANNOTATION_PATH = DEFAULT_DATA_ROOT / "train_annotations.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "paper_matched_single_cvc_v1"
PLACEMENT_LABELS = ["CVC - Normal", "CVC - Borderline", "CVC - Abnormal"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(f"Locked split manifest missing: {SPLIT_PATH}")
    if not ANNOTATION_PATH.is_file():
        raise FileNotFoundError(f"RANZCR annotations missing: {ANNOTATION_PATH}")

    split = pd.read_parquet(SPLIT_PATH)
    annotations = pd.read_csv(ANNOTATION_PATH, usecols=["StudyInstanceUID", "label"])
    cvc_annotations = annotations[
        annotations["label"].isin(PLACEMENT_LABELS)
    ].copy()

    required = {
        "StudyInstanceUID",
        "PatientID",
        "duplicate_group_id",
        "split",
        "is_exposed",
        *PLACEMENT_LABELS,
    }
    missing = sorted(required - set(split.columns))
    if missing:
        raise ValueError(f"Locked split is missing required columns: {missing}")

    trace_counts = cvc_annotations.groupby("StudyInstanceUID").size()
    one_trace_ids = set(trace_counts[trace_counts == 1].index)
    cvc_annotations = cvc_annotations[
        cvc_annotations["StudyInstanceUID"].isin(one_trace_ids)
    ].drop_duplicates("StudyInstanceUID", keep=False)
    if cvc_annotations["StudyInstanceUID"].duplicated().any():
        raise AssertionError("One-trace join is not unique by StudyInstanceUID")

    joined = split.merge(
        cvc_annotations.rename(columns={"label": "trace_label"}),
        on="StudyInstanceUID",
        how="inner",
        validate="one_to_one",
    )
    joined = joined[joined["split"].isin(["train", "val", "test"])].copy()
    for label in PLACEMENT_LABELS:
        joined[label] = pd.to_numeric(joined[label], errors="raise").astype("int8")
        if not joined[label].isin([0, 1]).all():
            raise ValueError(f"Non-binary target values in {label}")

    joined["positive_label_count"] = joined[PLACEMENT_LABELS].sum(axis=1)
    joined["image_label_matches_trace"] = (
        joined[PLACEMENT_LABELS].idxmax(axis=1).eq(joined["trace_label"])
        & joined["positive_label_count"].eq(1)
    )
    cohort = joined[joined["image_label_matches_trace"]].copy()
    if cohort.empty:
        raise RuntimeError("No examples passed the strict trace/label agreement rule")

    # Verify patient and duplicate-group separation after filtering.
    for key in ["PatientID", "duplicate_group_id"]:
        sets = {
            name: set(cohort.loc[cohort["split"] == name, key].dropna())
            for name in ["train", "val", "test"]
        }
        for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise AssertionError(f"{key} overlap between {left}/{right}: {len(overlap)}")

    val_test = cohort[cohort["split"].isin(["val", "test"])]
    if val_test["is_exposed"].astype(bool).any():
        raise AssertionError("Validation/test subset contains previously exposed examples")

    keep = [
        "StudyInstanceUID",
        "PatientID",
        "duplicate_group_id",
        "cluster_id",
        "split",
        "split_role",
        "is_exposed",
        "annotated_cvc_count",
        "trace_label",
        "positive_label_count",
        "image_label_matches_trace",
        *PLACEMENT_LABELS,
    ]
    cohort = cohort[keep].sort_values(["split", "StudyInstanceUID"]).reset_index(drop=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "single_annotated_cvc_manifest.csv"
    parquet_path = OUTPUT_DIR / "single_annotated_cvc_manifest.parquet"
    cohort.to_csv(csv_path, index=False)
    cohort.to_parquet(parquet_path, index=False)

    counts: dict[str, Any] = {}
    for split_name, group in cohort.groupby("split", sort=True):
        counts[split_name] = {
            "studies": int(len(group)),
            "patients": int(group["PatientID"].nunique()),
            "duplicate_groups": int(group["duplicate_group_id"].nunique()),
            "prior_exposed_studies": int(group["is_exposed"].astype(bool).sum()),
            "per_class": {label: int(group[label].sum()) for label in PLACEMENT_LABELS},
        }

    rejected = joined[~joined["image_label_matches_trace"]]
    summary = {
        "cohort_name": "paper_matched_single_annotated_cvc_v1",
        "definition": (
            "Exactly one annotated CVC trace per study and exactly one image-level CVC "
            "placement label, matching the trace label. This is an annotation-defined "
            "single-trace cohort; it is not radiologist-confirmed single-catheter truth."
        ),
        "input_hashes_sha256": {
            str(SPLIT_PATH.relative_to(PROJECT_ROOT)): sha256(SPLIT_PATH),
            "train_annotations.csv": sha256(ANNOTATION_PATH),
        },
        "counts": counts,
        "candidate_studies_with_exactly_one_trace_in_locked_splits": int(len(joined)),
        "excluded_due_to_image_label_mismatch_or_multiple_active_labels": int(len(rejected)),
        "excluded_historical_development_split": int(
            (split["split"] == "dev_val_historical").sum()
        ),
        "leakage_audit": {
            "patient_overlap_train_val": 0,
            "patient_overlap_train_test": 0,
            "patient_overlap_val_test": 0,
            "duplicate_group_overlap_train_val": 0,
            "duplicate_group_overlap_train_test": 0,
            "duplicate_group_overlap_val_test": 0,
            "prior_exposed_val_or_test_studies": 0,
        },
        "outputs_sha256": {
            csv_path.name: sha256(csv_path),
            parquet_path.name: sha256(parquet_path),
        },
        "test_policy": "Manifest records test membership; do not use test for tuning or model selection.",
    }
    json_path = OUTPUT_DIR / "cohort_audit.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Paper-Matched Single-Trace CVC Cohort v1",
        "",
        summary["definition"],
        "",
        "| Split | Studies | Patients | Normal | Borderline | Abnormal | Prior exposure |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name in ["train", "val", "test"]:
        row = counts.get(split_name, {})
        per_class = row.get("per_class", {})
        lines.append(
            f"| {split_name} | {row.get('studies', 0)} | {row.get('patients', 0)} | "
            f"{per_class.get('CVC - Normal', 0)} | {per_class.get('CVC - Borderline', 0)} | "
            f"{per_class.get('CVC - Abnormal', 0)} | {row.get('prior_exposed_studies', 0)} |"
        )
    lines += [
        "",
        f"Candidates with one CVC trace in locked train/val/test: {len(joined)}.",
        f"Removed for trace/image-label disagreement or multiple active labels: {len(rejected)}.",
        "All reported patient and duplicate-group overlaps are zero; validation and test are unexposed.",
        "The test partition is reserved for one final evaluation after model selection.",
        "",
        "This is a focused research subset, not the final deployment cohort. The annotation count "
        "does not prove that no unannotated CVC is present in an image.",
    ]
    report_path = OUTPUT_DIR / "cohort_audit.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "counts": counts, "output_dir": str(OUTPUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()
