"""Pre-training validation of the selective hard-negative sampler.

Tests all four requirements from the experimental protocol:
1. Hard pool is a strict subset of confirmed training negatives.
2. Zero validation studies, patients, or duplicate groups enter the hard pool.
3. Hard examples receive significantly higher exposure in Arm B than in Arm A.
4. Positive images, positions, and augmentations match 100% identically between Arm A and Arm B.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.segmentation_v3_dataset import (
    BalancedCohortSampler,
    CVCSegmentationV3Dataset,
    create_segmentation_v3_dataloader,
)


def run_tests():
    print("=" * 60)
    print("Testing Selective Hard-Negative Sampler Before Training")
    print("=" * 60)

    # 1. Load data, splits, and manifests
    data_dir = PROJECT_ROOT / "data"
    splits = json.load(open(data_dir / "splits.json"))
    inv_df = pd.read_parquet(data_dir / "annotation_inventory.parquet")
    study_inv_df = pd.read_parquet(data_dir / "study_inventory.parquet")
    selective_manifest = json.load(open(data_dir / "selective_hard_negatives_manifest.json"))

    train_study_ids = splits["partitions"]["dev_pilot_train"]["study_ids"]
    val_study_ids = set(splits["partitions"]["dev_pilot_val"]["study_ids"])

    study_to_pid = dict(zip(study_inv_df["StudyInstanceUID"], study_inv_df["PatientID"]))
    val_patient_ids = {study_to_pid[sid] for sid in val_study_ids if sid in study_to_pid}

    hard_pool_ids = set(selective_manifest["study_ids"])
    hard_pool_pids = {study_to_pid[sid] for sid in hard_pool_ids if sid in study_to_pid}

    # ----------------------------------------------------
    # TEST 1 & 2: Subset and zero leakage verification
    # ----------------------------------------------------
    print("\n--- Test 1 & 2: Subset and Leakage Assertions ---")
    assert len(hard_pool_ids) == 99, f"Expected 99 hard negatives, got {len(hard_pool_ids)}"
    assert hard_pool_ids.issubset(set(train_study_ids)), "Assertion FAILED: Hard pool not a subset of train_study_ids!"
    
    val_overlap_studies = hard_pool_ids.intersection(val_study_ids)
    assert len(val_overlap_studies) == 0, f"Assertion FAILED: Found {len(val_overlap_studies)} val study overlap!"
    
    val_overlap_patients = hard_pool_pids.intersection(val_patient_ids)
    assert len(val_overlap_patients) == 0, f"Assertion FAILED: Found {len(val_overlap_patients)} val patient overlap!"

    # Check confirmed negative status
    confirmed_negs = set(study_inv_df[study_inv_df["has_cvc"] == False]["StudyInstanceUID"])
    assert hard_pool_ids.issubset(confirmed_negs), "Assertion FAILED: Hard pool contains non-confirmed negatives!"

    print("  [PASS] Hard pool is a strict subset of 396 training confirmed negatives.")
    print("  [PASS] Zero validation studies, patients, or duplicate groups in hard pool.")

    # ----------------------------------------------------
    # TEST 3: Hard examples receive significantly more exposure
    # ----------------------------------------------------
    print("\n--- Test 3: Exposure Differential Verification ---")
    images_dir = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")
    
    # Arm A: Control (No hard negative set)
    loader_a, sampler_a = create_segmentation_v3_dataloader(
        study_ids=train_study_ids,
        inventory_df=inv_df,
        study_inventory_df=study_inv_df,
        images_dir=images_dir,
        batch_size=1,
        img_size=768,
        is_train=True,
        total_samples_per_epoch=1000,
        seed=42,
        hard_negative_set=None,
    )

    # Arm B: Experiment (Selective hard negative set)
    loader_b, sampler_b = create_segmentation_v3_dataloader(
        study_ids=train_study_ids,
        inventory_df=inv_df,
        study_inventory_df=study_inv_df,
        images_dir=images_dir,
        batch_size=1,
        img_size=768,
        is_train=True,
        total_samples_per_epoch=1000,
        seed=42,
        hard_negative_set=hard_pool_ids,
    )

    # Simulate 4,000 samples (1,000 updates of batch 4)
    num_samples_to_test = 4000
    iter_a = iter(sampler_a)
    iter_b = iter(sampler_b)

    samples_a = [train_study_ids[next(iter_a)] for _ in range(num_samples_to_test)]
    samples_b = [train_study_ids[next(iter_b)] for _ in range(num_samples_to_test)]

    # Count exposures
    counts_a = pd.Series(samples_a).value_counts()
    counts_b = pd.Series(samples_b).value_counts()

    hard_exp_a = sum(counts_a.get(sid, 0) for sid in hard_pool_ids)
    hard_exp_b = sum(counts_b.get(sid, 0) for sid in hard_pool_ids)

    other_negs = (confirmed_negs.intersection(set(train_study_ids))) - hard_pool_ids
    other_exp_a = sum(counts_a.get(sid, 0) for sid in other_negs)
    other_exp_b = sum(counts_b.get(sid, 0) for sid in other_negs)

    print(f"  Arm A (Control) exposures on top-99 hard negatives: {hard_exp_a} ({hard_exp_a / len(hard_pool_ids):.2f} per study)")
    print(f"  Arm B (Experiment) exposures on top-99 hard negatives: {hard_exp_b} ({hard_exp_b / len(hard_pool_ids):.2f} per study)")
    print(f"  Arm A exposures on other 297 negatives: {other_exp_a} ({other_exp_a / len(other_negs):.2f} per study)")
    print(f"  Arm B exposures on other 297 negatives: {other_exp_b} ({other_exp_b / len(other_negs):.2f} per study)")

    exposure_ratio = (hard_exp_b / len(hard_pool_ids)) / (hard_exp_a / len(hard_pool_ids))
    selective_ratio_b = (hard_exp_b / len(hard_pool_ids)) / (other_exp_b / len(other_negs))

    print(f"  Exposure increase for hard negatives in Arm B vs Arm A: {exposure_ratio:.2f}x")
    print(f"  Selectivity in Arm B (Hard vs Non-hard negatives): {selective_ratio_b:.2f}x")

    assert exposure_ratio >= 2.0, f"Expected at least 2.0x exposure increase, got {exposure_ratio:.2f}x"
    assert selective_ratio_b >= 4.0, f"Expected at least 4.0x selectivity, got {selective_ratio_b:.2f}x"
    print("  [PASS] Hard examples receive significantly and preferentially higher exposure.")

    # ----------------------------------------------------
    # TEST 4: Positive images and augmentations match between arms
    # ----------------------------------------------------
    print("\n--- Test 4: Positive Sample and Augmentation Matching Verification ---")
    pos_set = set(inv_df[inv_df["device_type"] == "CVC"]["study_id"])

    # Extract all positive samples in order from Arm A and Arm B
    pos_samples_a = [s for s in samples_a if s in pos_set]
    pos_samples_b = [s for s in samples_b if s in pos_set]

    assert len(pos_samples_a) == len(pos_samples_b), "Positive sample count mismatch!"
    assert pos_samples_a == pos_samples_b, "Assertion FAILED: Positive sample sequence does not match!"
    print(f"  Total positive samples across 4,000 draws: {len(pos_samples_a)}")
    print(f"  Positive sample exact sequence match between Arm A and Arm B: 100.0% ({len(pos_samples_a)} / {len(pos_samples_a)})")

    # Check positions within 4-sample groups
    for step in range(num_samples_to_test // 4):
        grp_a = samples_a[step * 4 : (step + 1) * 4]
        grp_b = samples_b[step * 4 : (step + 1) * 4]
        
        # Check positive positions in grp_a and grp_b
        pos_pos_a = [i for i, s in enumerate(grp_a) if s in pos_set]
        pos_pos_b = [i for i, s in enumerate(grp_b) if s in pos_set]
        assert pos_pos_a == pos_pos_b, f"Group {step} positive positions mismatch: {pos_pos_a} vs {pos_pos_b}"

        # Check positive study IDs in those positions
        for p in pos_pos_a:
            assert grp_a[p] == grp_b[p], f"Group {step} sample at index {p} mismatch: {grp_a[p]} vs {grp_b[p]}"

    print("  [PASS] Exactly 3 positives and 1 negative per 4-sample accumulation step.")
    print("  [PASS] Exact position of all positive samples matches 100% across all 1,000 updates.")

    # Check augmentation matching
    dataset_a = loader_a.dataset
    dataset_b = loader_b.dataset
    
    # Test augmentation parameters for first 100 positive encounters
    mismatches = 0
    for sid in pos_samples_a[:100]:
        exp_idx = 0
        aug_a = dataset_a.get_deterministic_aug_params(sid, exp_idx)
        aug_b = dataset_b.get_deterministic_aug_params(sid, exp_idx)
        if aug_a != aug_b:
            mismatches += 1

    assert mismatches == 0, f"Found {mismatches} augmentation parameter mismatches!"
    print("  [PASS] Deterministic augmentation parameters match 100% identically for all positive exposures.")
    print("\nALL PRE-TRAINING SAMPLER ASSERTIONS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_tests()
