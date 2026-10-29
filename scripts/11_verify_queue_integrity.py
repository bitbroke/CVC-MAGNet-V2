"""Audit and Enforce Queue Integrity.

Verifies:
1. Patient ID separation between training queue (200 positive + 400 negative) and validation (193 studies in dev_pilot_val).
2. Duplicate-group separation between training queue and validation.
3. Separation from frozen pilot training (the 23 positive studies).
4. If any collisions exist, replaces offending studies with clean candidates from dev_full_train.
5. Re-saves queue files and documents the 235 catheters as 'inventory-listed'.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"C:\Ankshit\CVC-MAGNet-V2")

def audit_queue_integrity():
    print("=" * 80)
    print("AUDITING ANNOTATION QUEUE INTEGRITY & SEPARATION")
    print("=" * 80)

    inv_df = pd.read_parquet(PROJECT_ROOT / "data" / "annotation_inventory.parquet")
    study_df = pd.read_parquet(PROJECT_ROOT / "data" / "study_inventory.parquet")
    reviewed_df = pd.read_parquet(PROJECT_ROOT / "data" / "reviewed_distal_tips_v3.parquet")
    
    with open(PROJECT_ROOT / "data" / "splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)

    with open(PROJECT_ROOT / "data" / "annotation_expansion_queue.json", "r", encoding="utf-8") as f:
        queue_data = json.load(f)

    # 1. Validation cohort patient and duplicate-group sets
    val_sids = set(splits["partitions"]["dev_pilot_val"]["study_ids"])
    val_studies = study_df[study_df["StudyInstanceUID"].isin(val_sids)].copy()
    val_patients = set(val_studies["PatientID"].dropna())
    val_dup_groups = set(val_studies["duplicate_group_id"].dropna())

    print(f"Validation Cohort ({len(val_sids)} studies):")
    print(f"  Unique Patients:        {len(val_patients)}")
    print(f"  Unique Duplicate Groups: {len(val_dup_groups)}")

    # 2. Frozen pilot training positive studies
    frozen_pilot_train_sids = set(reviewed_df[reviewed_df["split"] == "train"]["study_id"])
    frozen_pilot_studies = study_df[study_df["StudyInstanceUID"].isin(frozen_pilot_train_sids)].copy()
    frozen_patients = set(frozen_pilot_studies["PatientID"].dropna())
    frozen_dup_groups = set(frozen_pilot_studies["duplicate_group_id"].dropna())

    print(f"\nFrozen Pilot Train Cohort ({len(frozen_pilot_train_sids)} positive studies):")
    print(f"  Unique Patients:        {len(frozen_patients)}")
    print(f"  Unique Duplicate Groups: {len(frozen_dup_groups)}")

    # 3. Check Positive Annotation Queue (200 studies)
    pos_queue_sids = queue_data["positive_study_ids"]
    pos_studies = study_df[study_df["StudyInstanceUID"].isin(pos_queue_sids)].copy()
    pos_patients = set(pos_studies["PatientID"].dropna())
    pos_dup_groups = set(pos_studies["duplicate_group_id"].dropna())

    print(f"\nPositive Annotation Queue ({len(pos_queue_sids)} studies):")
    print(f"  Unique Patients:        {len(pos_patients)}")
    print(f"  Unique Duplicate Groups: {len(pos_dup_groups)}")

    # Check collisions with Validation
    val_patient_collisions = pos_patients.intersection(val_patients)
    val_dup_collisions = pos_dup_groups.intersection(val_dup_groups)
    print(f"\nCollisions with Validation:")
    print(f"  Patient ID collisions:         {len(val_patient_collisions)} -> {val_patient_collisions}")
    print(f"  Duplicate Group collisions:    {len(val_dup_collisions)} -> {val_dup_collisions}")

    # Check collisions with Frozen Pilot Train
    train_patient_collisions = pos_patients.intersection(frozen_patients)
    train_dup_collisions = pos_dup_groups.intersection(frozen_dup_groups)
    print(f"Collisions with Frozen Pilot Train:")
    print(f"  Patient ID collisions:         {len(train_patient_collisions)} -> {train_patient_collisions}")
    print(f"  Duplicate Group collisions:    {len(train_dup_collisions)} -> {train_dup_collisions}")

    # 4. Check Negative Annotation Queue (400 studies)
    neg_queue_sids = queue_data["negative_study_ids"]
    neg_studies = study_df[study_df["StudyInstanceUID"].isin(neg_queue_sids)].copy()
    neg_patients = set(neg_studies["PatientID"].dropna())
    neg_dup_groups = set(neg_studies["duplicate_group_id"].dropna())

    neg_val_patient_collisions = neg_patients.intersection(val_patients)
    neg_val_dup_collisions = neg_dup_groups.intersection(val_dup_groups)
    print(f"\nNegative Queue Collisions with Validation:")
    print(f"  Patient ID collisions:         {len(neg_val_patient_collisions)} -> {neg_val_patient_collisions}")
    print(f"  Duplicate Group collisions:    {len(neg_val_dup_collisions)} -> {neg_val_dup_collisions}")

    # If any collisions exist, swap them out cleanly
    all_banned_patients = val_patients | frozen_patients
    all_banned_dup_groups = val_dup_groups | frozen_dup_groups
    all_banned_sids = val_sids | frozen_pilot_train_sids

    full_train_sids = set(splits["partitions"]["dev_full_train"]["study_ids"])

    # Replacement logic for positive studies if needed
    cleaned_pos_sids = []
    replaced_count = 0
    for sid in pos_queue_sids:
        s_row = study_df[study_df["StudyInstanceUID"] == sid].iloc[0]
        pid = s_row["PatientID"]
        did = s_row["duplicate_group_id"]
        if sid in all_banned_sids or pid in all_banned_patients or did in all_banned_dup_groups:
            replaced_count += 1
            print(f"Flagged collision: Study {sid[:12]} (Patient {pid}, Group {did})")
        else:
            cleaned_pos_sids.append(sid)

    if replaced_count > 0:
        print(f"Replacing {replaced_count} colliding positive studies...")
        # Find replacement candidates
        cvc_inv = inv_df[inv_df["device_type"] == "CVC"].copy()
        current_patients = set(study_df[study_df["StudyInstanceUID"].isin(cleaned_pos_sids)]["PatientID"])
        current_dups = set(study_df[study_df["StudyInstanceUID"].isin(cleaned_pos_sids)]["duplicate_group_id"])

        candidate_rows = study_df[
            study_df["StudyInstanceUID"].isin(full_train_sids) &
            study_df["has_cvc"] &
            (~study_df["StudyInstanceUID"].isin(all_banned_sids)) &
            (~study_df["PatientID"].isin(all_banned_patients | current_patients)) &
            (~study_df["duplicate_group_id"].isin(all_banned_dup_groups | current_dups))
        ]
        replacements = candidate_rows.sample(n=replaced_count, random_state=42)["StudyInstanceUID"].tolist()
        cleaned_pos_sids.extend(replacements)

    # Replacement logic for negative studies if needed
    cleaned_neg_sids = []
    replaced_neg_count = 0
    for sid in neg_queue_sids:
        s_row = study_df[study_df["StudyInstanceUID"] == sid].iloc[0]
        pid = s_row["PatientID"]
        did = s_row["duplicate_group_id"]
        if sid in all_banned_sids or pid in all_banned_patients or did in all_banned_dup_groups:
            replaced_neg_count += 1
        else:
            cleaned_neg_sids.append(sid)

    if replaced_neg_count > 0:
        print(f"Replacing {replaced_neg_count} colliding negative studies...")
        current_neg_patients = set(study_df[study_df["StudyInstanceUID"].isin(cleaned_neg_sids)]["PatientID"])
        current_neg_dups = set(study_df[study_df["StudyInstanceUID"].isin(cleaned_neg_sids)]["duplicate_group_id"])
        neg_candidates = study_df[
            study_df["is_study_level_negative"] &
            (~study_df["has_cvc"]) &
            (~study_df["StudyInstanceUID"].isin(all_banned_sids)) &
            (~study_df["PatientID"].isin(all_banned_patients | current_neg_patients)) &
            (~study_df["duplicate_group_id"].isin(all_banned_dup_groups | current_neg_dups))
        ]
        replacements_neg = neg_candidates.sample(n=replaced_neg_count, random_state=42)["StudyInstanceUID"].tolist()
        cleaned_neg_sids.extend(replacements_neg)

    # Assert 100% strict separation
    final_pos_studies = study_df[study_df["StudyInstanceUID"].isin(cleaned_pos_sids)]
    final_neg_studies = study_df[study_df["StudyInstanceUID"].isin(cleaned_neg_sids)]

    assert len(set(final_pos_studies["PatientID"]).intersection(val_patients)) == 0, "Patient leak in positive queue!"
    assert len(set(final_pos_studies["duplicate_group_id"]).intersection(val_dup_groups)) == 0, "Dup group leak in positive queue!"
    assert len(set(final_neg_studies["PatientID"]).intersection(val_patients)) == 0, "Patient leak in negative queue!"
    assert len(set(final_neg_studies["duplicate_group_id"]).intersection(val_dup_groups)) == 0, "Dup group leak in negative queue!"

    print("\n" + "=" * 80)
    print("QUEUE INTEGRITY AUDIT PASSED: 100% AIRTIGHT SEPARATION VERIFIED")
    print(f"  Positive studies: {len(cleaned_pos_sids)} (Zero patient/dup overlap with validation or pilot)")
    print(f"  Negative studies: {len(cleaned_neg_sids)} (Zero patient/dup overlap with validation or pilot)")
    print("=" * 80)

    # Re-generate queues with updated lists
    pos_targets_records = []
    for sid in cleaned_pos_sids:
        scvcs = inv_df[(inv_df["study_id"] == sid) & (inv_df["device_type"] == "CVC")]
        for _, crow in scvcs.iterrows():
            pos_targets_records.append({
                "study_id": sid,
                "patient_id": crow["patient_id"],
                "catheter_instance_id": crow["catheter_instance_id"],
                "instance_key": crow["instance_key"],
                "label": crow["label"],
                "image_width": crow["image_width"],
                "image_height": crow["image_height"],
                "polyline_length_px": round(crow["polyline_length_px"], 1),
                "heuristic_tip_x": crow["tip_x"],
                "heuristic_tip_y": crow["tip_y"],
                "reviewed_tip_x": None,
                "reviewed_tip_y": None,
                "review_state": "queued",  # assessable, uncertain, outside_image, annotation_incorrect
                "review_status": "queued_for_review",
            })

    pos_targets_df = pd.DataFrame(pos_targets_records)
    print(f"Inventory-listed CVC catheters: {len(pos_targets_df)}")

    # Update Queue JSON
    queue_dict = {
        "description": "Expanded pilot annotation queue (200 positive studies + 400 confirmed negatives)",
        "queue_integrity_verified": True,
        "patient_separation_verified": True,
        "duplicate_group_separation_verified": True,
        "positive_studies_count": len(cleaned_pos_sids),
        "inventory_listed_cvc_catheters_count": len(pos_targets_df),
        "negative_studies_count": len(cleaned_neg_sids),
        "positive_study_ids": cleaned_pos_sids,
        "negative_study_ids": cleaned_neg_sids,
        "annotation_contract": {
            "completeness_rule": "Every visible CVC catheter in each study must be reviewed. Document inventory-listed catheters as well as any uncataloged visible CVCs.",
            "coordinate_rule": "No automatically accepted endpoint suggestions. Reviewer must explicitly place and attest coordinates.",
            "states": ["assessable", "uncertain", "outside_image", "annotation_incorrect"],
            "blinding_rule": "Reviewers must work without model prediction overlays.",
        },
    }

    queue_json_path = PROJECT_ROOT / "data" / "annotation_expansion_queue.json"
    with open(queue_json_path, "w", encoding="utf-8") as f:
        json.dump(queue_dict, f, indent=2)

    pos_targets_csv = PROJECT_ROOT / "data" / "annotation_expansion_positive_targets.csv"
    pos_targets_df.to_csv(pos_targets_csv, index=False)

    neg_df = pd.DataFrame({"study_id": cleaned_neg_sids, "device_type": "None", "has_cvc": False})
    neg_csv = PROJECT_ROOT / "data" / "annotation_expansion_negatives.csv"
    neg_df.to_csv(neg_csv, index=False)

    print("Updated queue files successfully saved.")

if __name__ == "__main__":
    audit_queue_integrity()
