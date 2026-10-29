"""Build leakage-safe placement-classification cohort variants for ablations.

This script only reads labels/manifests and writes derived manifests. It does not
read images, train models, alter existing checkpoints, or access test labels for
model selection. The locked split assignment is inherited unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(
    r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification"
    r"\ranzcr-clip-catheter-line-classification"
)
SPLIT = ROOT / "data/splits/placement_locked_manifest.parquet"
ANNOTATIONS = DATA_ROOT / "train_annotations.csv"
OUT = ROOT / "experiments/placement_cohort_matrix_v1"
LABELS = ["CVC - Normal", "CVC - Borderline", "CVC - Abnormal"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def audit_split(df: pd.DataFrame, name: str) -> dict:
    result = {}
    for part in ("train", "val", "test"):
        p = df[df.split == part]
        result[part] = {
            "studies": int(len(p)),
            "patients": int(p.PatientID.nunique()),
            "duplicate_groups": int(p.duplicate_group_id.nunique()),
            "prior_exposed": int(p.is_exposed.astype(bool).sum()),
            "label_counts": {label: int(p[label].sum()) for label in LABELS},
            "multi_label_studies": int((p[LABELS].sum(axis=1) > 1).sum()),
        }
    for key in ("PatientID", "duplicate_group_id"):
        parts = {s: set(df.loc[df.split == s, key].dropna()) for s in ("train", "val", "test")}
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            if parts[left] & parts[right]:
                raise AssertionError(f"{name}: {key} leakage {left}/{right}")
    if df[df.split.isin(["val", "test"])].is_exposed.astype(bool).any():
        raise AssertionError(f"{name}: validation/test contains previously exposed studies")
    return result


def main() -> None:
    split = pd.read_parquet(SPLIT)
    ann = pd.read_csv(ANNOTATIONS, usecols=["StudyInstanceUID", "label"])
    ann = ann[ann.label.isin(LABELS)].copy()
    counts = ann.groupby("StudyInstanceUID").size().rename("trace_count")
    joined = split.merge(counts, on="StudyInstanceUID", how="left", validate="one_to_one")
    joined["trace_count"] = joined.trace_count.fillna(0).astype("int16")
    joined = joined[joined.split.isin(["train", "val", "test"])].copy()
    joined[LABELS] = joined[LABELS].apply(pd.to_numeric, errors="raise").astype("int8")
    joined["positive_label_count"] = joined[LABELS].sum(axis=1).astype("int8")

    # Variant A: all studies with at least one recorded CVC trace. This keeps
    # the original multi-label image-level targets, including multi-CVC cases.
    traced = joined[joined.trace_count >= 1].copy()

    # Variant B: strict single-trace, single-label, trace/label concordant cases.
    ann_one = ann.groupby("StudyInstanceUID").filter(lambda g: len(g) == 1)
    ann_one = ann_one.drop_duplicates("StudyInstanceUID", keep=False).rename(columns={"label": "trace_label"})
    one = joined[joined.trace_count == 1].merge(
        ann_one[["StudyInstanceUID", "trace_label"]],
        on="StudyInstanceUID", how="inner", validate="one_to_one",
    )
    one["trace_label_matches"] = one.apply(
        lambda r: int(r["trace_label"] == next((c for c in LABELS if r[c] == 1), "")
                      and r["positive_label_count"] == 1), axis=1
    )
    one = one[one.trace_label_matches == 1].copy()
    one["binary_target"] = (one["CVC - Normal"] == 0).astype("int8")
    one["binary_target_name"] = one.binary_target.map({0: "normal", 1: "borderline_or_abnormal"})

    OUT.mkdir(parents=True, exist_ok=True)
    variants = {
        "all_cvc_traced_multilabel": traced,
        "single_cvc_binary_normal_vs_non_normal": one,
    }
    reports = {}
    for name, frame in variants.items():
        frame = frame.sort_values(["split", "StudyInstanceUID"]).reset_index(drop=True)
        frame.to_parquet(OUT / f"{name}.parquet", index=False)
        frame.to_csv(OUT / f"{name}.csv", index=False)
        reports[name] = {
            "definition": (
                "At least one annotated CVC trace; original three image-level labels retained."
                if name == "all_cvc_traced_multilabel"
                else "Exactly one CVC trace, exactly one concordant image-level label; Normal=0, Borderline or Abnormal=1."
            ),
            "counts": audit_split(frame, name),
            "manifest_sha256": digest(OUT / f"{name}.parquet"),
        }
    metadata = {
        "title": "Placement cohort ablation matrix v1",
        "inputs": {
            "locked_split_sha256": digest(SPLIT),
            "train_annotations_sha256": digest(ANNOTATIONS),
        },
        "variants": reports,
        "cautions": [
            "An annotation trace count is not proof that the image contains exactly that many visible catheters.",
            "Image-level placement labels are not catheter-specific outcomes in multi-CVC studies.",
            "The locked test subset must remain unused until final model selection is complete.",
            "The binary target means not-normal by dataset label; it is not a millimetre-based placement judgment.",
        ],
    }
    (OUT / "cohort_matrix_audit.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    lines = ["# Placement Cohort Matrix v1", "", "Derived without changing split assignments or checkpoints.", ""]
    for name, payload in reports.items():
        lines += [f"## {name}", "", payload["definition"], "", "| Split | Studies | Patients | Normal | Borderline | Abnormal | Multi-label | Prior exposed |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for part, row in payload["counts"].items():
            lc = row["label_counts"]
            lines.append(f"| {part} | {row['studies']} | {row['patients']} | {lc[LABELS[0]]} | {lc[LABELS[1]]} | {lc[LABELS[2]]} | {row['multi_label_studies']} | {row['prior_exposed']} |")
        lines.append("")
    lines += ["## Interpretation safeguards", "", *[f"- {x}" for x in metadata["cautions"]], ""]
    (OUT / "cohort_matrix_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "complete", "output_dir": str(OUT), "variants": reports}, indent=2))


if __name__ == "__main__":
    main()
