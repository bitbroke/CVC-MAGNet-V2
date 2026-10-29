"""Inspect cohort availability and check candidates for training expansion."""

import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
data_dir = PROJECT_ROOT / "data"

splits = json.load(open(data_dir / "splits.json"))
train_sids = set(splits["partitions"]["dev_pilot_train"]["study_ids"])
val_sids = set(splits["partitions"]["dev_pilot_val"]["study_ids"])

study_inv = pd.read_parquet(data_dir / "study_inventory.parquet")
ann_inv = pd.read_parquet(data_dir / "annotation_inventory.parquet")

study_to_pid = dict(zip(study_inv["StudyInstanceUID"], study_inv["PatientID"]))
val_pids = {study_to_pid[sid] for sid in val_sids if sid in study_to_pid}

cvc_ann = ann_inv[ann_inv["device_type"] == "CVC"].copy()
ann_study_ids = set(cvc_ann["study_id"])

print(f"Total CVC-annotated studies in annotation_inventory: {len(ann_study_ids)}")
print(f"Studies in pilot train: {len(train_sids)}")
print(f"  Positives in pilot train: {len(train_sids.intersection(ann_study_ids))}")
print(f"  Negatives in pilot train: {len(train_sids - ann_study_ids)}")
print(f"Studies in pilot val: {len(val_sids)}")
print(f"  Positives in pilot val: {len(val_sids.intersection(ann_study_ids))}")
print(f"  Negatives in pilot val: {len(val_sids - ann_study_ids)}")

# Available candidates outside val and train
val_patient_studies = set(study_inv[study_inv["PatientID"].isin(val_pids)]["StudyInstanceUID"])
available_positives = ann_study_ids - val_patient_studies - train_sids
print(f"Additional annotated positive studies available (strictly outside val patients & train): {len(available_positives)}")

confirmed_negs = set(study_inv[study_inv["has_cvc"] == False]["StudyInstanceUID"])
available_negs = confirmed_negs - val_patient_studies - train_sids
print(f"Additional confirmed negative studies available (strictly outside val patients & train): {len(available_negs)}")
