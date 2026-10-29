from __future__ import annotations
import json
from pathlib import Path
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "curated_cvc_v1"
QUEUE = OUT / "carina_placement_pilot_50.json"
CSV = OUT / "carina_placement_pilot_50.csv"
SEED = 20261025

if QUEUE.exists() or CSV.exists():
    raise SystemExit(f"Refusing to overwrite existing pilot queue: {QUEUE} / {CSV}")
manifest = pd.read_parquet(ROOT / "data" / "curated_cvc_v1" / "study_manifest.parquet")
ann = pd.read_parquet(ROOT / "data" / "annotation_inventory.parquet")
splits = json.loads((ROOT / "data" / "splits.json").read_text(encoding="utf-8"))
val_ids = set(map(str, splits["partitions"]["dev_pilot_val"]["study_ids"]))
train_ids = set(map(str, splits["partitions"]["dev_full_train"]["study_ids"]))
positive = manifest[(manifest.study_id.astype(str).isin(train_ids)) & manifest.has_cvc.astype(bool)].copy()
positive = positive[positive.supervision_quality_group.isin(["Group A", "Group B"])].copy()
# The queue is for reviewing real inventory polylines. Filter to studies with at least one CVC annotation row.
annotated_ids = set(ann.loc[ann.device_type.eq("CVC"), "study_id"].astype(str))
positive = positive[positive.study_id.astype(str).isin(annotated_ids)].copy()
positive["study_id"] = positive.study_id.astype(str)
positive["patient_id"] = positive.patient_id.astype(str)
positive["duplicate_group_id"] = positive.duplicate_group_id.astype(str)
positive = positive[~positive.study_id.isin(val_ids)]
positive = positive.sort_values("study_id").drop_duplicates(["patient_id", "duplicate_group_id"])

# Fixed strata are review-queue sampling descriptors, not clinical labels.
strata = {
    "multi_cvc": positive[positive.num_annotated_cvcs.ge(2)],
    "single_cluttered": positive[positive.num_annotated_cvcs.eq(1) & positive.has_other_lines.astype(bool)],
    "single_low_clutter": positive[positive.num_annotated_cvcs.eq(1) & ~positive.has_other_lines.astype(bool)],
    "border_or_short_trace": positive[positive.num_annotated_cvcs.ge(1) & (positive.has_border_touch.astype(bool) | positive.min_path_length_px.lt(800))],
    "placement_mix": positive[positive.num_annotated_cvcs.ge(1)],
}
quotas = [("multi_cvc", 10), ("single_cluttered", 10), ("single_low_clutter", 10), ("border_or_short_trace", 10), ("placement_mix", 10)]
used_ids, used_patients, used_dups, selected = set(), set(), set(), []
for stratum, quota in quotas:
    group = strata[stratum].sample(frac=1, random_state=SEED + len(selected))
    count = 0
    for _, row in group.iterrows():
        sid, patient, dup = row.study_id, row.patient_id, row.duplicate_group_id
        if sid in used_ids or patient in used_patients or dup in used_dups:
            continue
        selected.append((row, stratum))
        used_ids.add(sid); used_patients.add(patient); used_dups.add(dup)
        count += 1
        if count == quota:
            break
    if count != quota:
        raise RuntimeError(f"Could only select {count}/{quota} for stratum {stratum}")

records = []
for row, stratum in selected:
    records.append({
        "study_id": row.study_id,
        "patient_id": row.patient_id,
        "duplicate_group_id": row.duplicate_group_id,
        "queue_stratum": stratum,
        "num_annotated_cvcs": int(row.num_annotated_cvcs),
        "has_other_lines": bool(row.has_other_lines),
        "cvc_normal": int(row.cvc_normal),
        "cvc_borderline": int(row.cvc_borderline),
        "cvc_abnormal": int(row.cvc_abnormal),
        "image_width": int(row.image_width),
        "image_height": int(row.image_height),
        "readability_status": row.readability_status,
        "has_border_touch": bool(row.has_border_touch),
        "min_path_length_px": float(row.min_path_length_px),
        "split_membership": row.split_membership,
        "selection_note": "Stratum is a sampling descriptor; placement fields are dataset labels and require independent visual review.",
    })
df = pd.DataFrame(records)
assert len(df) == 50
assert df.study_id.nunique() == df.patient_id.nunique() == df.duplicate_group_id.nunique() == 50
assert set(df.study_id).isdisjoint(val_ids)
assert set(df.study_id).issubset(train_ids)
# Explicitly audit leakage at patient and duplicate-group level against the frozen validation registry.
val = manifest[manifest.study_id.astype(str).isin(val_ids)]
assert set(df.patient_id).isdisjoint(set(val.patient_id.astype(str)))
assert set(df.duplicate_group_id).isdisjoint(set(val.duplicate_group_id.astype(str)))
df.to_csv(CSV, index=False)
meta = {
    "queue_name": "carina_placement_pilot_50",
    "purpose": "Training-only dual-review calibration of carina point visibility/location and per-CVC placement labels.",
    "seed": SEED,
    "source_manifest": "data/curated_cvc_v1/study_manifest.parquet",
    "source_inventory": "data/annotation_inventory.parquet",
    "split": "dev_full_train",
    "validation_overlap_studies": 0,
    "validation_overlap_patients": 0,
    "validation_overlap_duplicate_groups": 0,
    "study_count": len(df),
    "unique_patient_count": int(df.patient_id.nunique()),
    "unique_duplicate_group_count": int(df.duplicate_group_id.nunique()),
    "stratum_counts": df.queue_stratum.value_counts().to_dict(),
    "placement_source_fields_are_not_review_truth": True,
    "image_quality_note": "Source readability_status is uniformly readable; border contact and short annotated path are difficulty proxies, not image-quality labels.",
    "csv_sha256": hashlib.sha256(CSV.read_bytes()).hexdigest(),
    "study_ids": df.study_id.tolist(),
    "studies": df.study_id.tolist(),
}
QUEUE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(json.dumps({k: meta[k] for k in ["queue_name", "study_count", "unique_patient_count", "validation_overlap_studies", "validation_overlap_patients", "validation_overlap_duplicate_groups", "stratum_counts", "csv_sha256"]}, indent=2))
