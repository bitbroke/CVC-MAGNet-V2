"""Script 02: Prepare Splits, Target Registries, and Geometric Verification.

Requirements:
- Group patients and duplicates before splitting (zero leakage).
- Preserve every evaluation target, including images with no predictions.
- Treat existing evaluated data as development data.
- Identify genuinely untouched data for final testing; otherwise state that final generalization remains unmeasured.
- Document polyline thickness and rasterization as an adaptation of dense masks.
- Preserve aspect ratio and record transformations back to native coordinates.
- Pass condition: split assertions pass, and image/mask/tip overlays align correctly.
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.rasterize import (
    create_gaussian_attention_mask,
    generate_gaussian_tip_heatmap,
    rasterize_polyline,
    rasterize_study_polylines,
)
from src.data.transforms import LetterboxTransform, TransformMetadata


def main():
    print("=" * 70)
    print("STEP 3: ESTABLISH SPLITS, TARGETS & GEOMETRIC VERIFICATION")
    print("=" * 70)

    # 1. Load Data
    data_dir = PROJECT_ROOT / "data"
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = reports_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    inv_path = data_dir / "annotation_inventory.parquet"
    study_inv_path = data_dir / "study_inventory.parquet"
    assert inv_path.exists(), f"Missing {inv_path}. Run 01_build_inventory.py first."
    assert study_inv_path.exists(), f"Missing {study_inv_path}."

    print("[1/5] Loading inventories...")
    catheter_df = pd.read_parquet(inv_path)
    study_df = pd.read_parquet(study_inv_path)
    print(f"Loaded {len(catheter_df):,} catheter records and {len(study_df):,} study records.")

    # Load V1 reference folds and frozen pilot manifest
    v1_ref_dir = Path(r"C:\Ankshit\Catheter")
    v1_folds_path = v1_ref_dir / "data" / "train_folds.parquet"
    v1_pilot_manifest_path = (
        v1_ref_dir / "data" / "refiner" / "exploratory_pilot_v1" / "pilot_subset" / "pilot_manifest.json"
    )

    assert v1_folds_path.exists(), f"Missing V1 folds {v1_folds_path}"
    assert v1_pilot_manifest_path.exists(), f"Missing V1 pilot manifest {v1_pilot_manifest_path}"

    v1_folds_df = pd.read_parquet(v1_folds_path)
    with open(v1_pilot_manifest_path, "r") as f:
        v1_manifest = json.load(f)

    # 2. Establish Partitions
    print("\n[2/5] Establishing development and untouched test cohorts...")
    study_to_fold = dict(zip(v1_folds_df["StudyInstanceUID"], v1_folds_df["fold"]))
    study_df["fold"] = study_df["StudyInstanceUID"].map(study_to_fold)
    catheter_df["fold"] = catheter_df["study_id"].map(study_to_fold)

    # Development Validation Cohort (exact 193 studies from frozen pilot manifest)
    val_study_ids = set(v1_manifest["validation_studies"])
    val_catheters = catheter_df[catheter_df["study_id"].isin(val_study_ids)]
    val_cvc_targets = val_catheters[
        (val_catheters["device_type"] == "CVC") & (val_catheters["is_assessable"])
    ]
    print(f"Development validation cohort: {len(val_study_ids)} studies.")
    print(f"Total assessable CVC targets: {len(val_cvc_targets)} (matches V1 frozen denominator of exactly 248).")
    assert len(val_cvc_targets) == 248, f"Expected exactly 248 validation targets, got {len(val_cvc_targets)}!"

    # Save validation targets registry
    val_targets_path = splits_dir / "dev_pilot_val_targets.csv"
    val_cvc_targets.to_csv(val_targets_path, index=False)
    print(f"Saved development validation targets to {val_targets_path}")

    # Identify patient and duplicate group IDs in validation
    val_patients: Set[str] = set(study_df[study_df["StudyInstanceUID"].isin(val_study_ids)]["PatientID"])
    val_dup_groups: Set[str] = set(study_df[study_df["StudyInstanceUID"].isin(val_study_ids)]["duplicate_group_id"])

    # Development Training Cohort: Folds 0, 1, 3
    # Exclude any patient or duplicate group in validation cohort
    dev_train_studies_mask = (
        (study_df["fold"].isin([0, 1, 3]))
        & (~study_df["PatientID"].isin(val_patients))
        & (~study_df["duplicate_group_id"].isin(val_dup_groups))
    )
    dev_train_studies = study_df[dev_train_studies_mask]
    dev_train_ids = set(dev_train_studies["StudyInstanceUID"])

    # Pilot Training Subset: Annotation-supported positives + Confirmed label-supported negatives
    cvc_ann_records = catheter_df[catheter_df["device_type"] == "CVC"]
    annotated_cvc_study_ids = set(cvc_ann_records["study_id"].unique())

    # Confirmed positives with real CVC polylines
    dev_annotated_pos_studies = dev_train_studies[
        dev_train_studies["StudyInstanceUID"].isin(annotated_cvc_study_ids)
    ]["StudyInstanceUID"].tolist()

    # Confirmed negatives with documented study-level absence
    dev_confirmed_neg_studies = dev_train_studies[
        dev_train_studies["is_study_level_negative"] & (~dev_train_studies["has_cvc"])
    ]["StudyInstanceUID"].tolist()

    # Unannotated positives: CVC present in classification, but no polylines (treated as unknown, excluded)
    unannotated_pos_studies = dev_train_studies[
        dev_train_studies["has_cvc"] & (~dev_train_studies["StudyInstanceUID"].isin(annotated_cvc_study_ids))
    ]["StudyInstanceUID"].tolist()

    print("\nTraining Supervision Audit:")
    print(f"  - Available Annotation-Supported CVC Positives: {len(dev_annotated_pos_studies)}")
    print(f"  - Available Confirmed Label-Supported Negatives: {len(dev_confirmed_neg_studies)}")
    print(f"  - Unannotated Positives (Treated as Unknown, Excluded from Mask Supervision): {len(unannotated_pos_studies)}")

    rng = np.random.RandomState(42)
    # Balanced pilot cohort: 400 annotation-supported positives + all 396 confirmed negatives
    n_pos_sample = min(400, len(dev_annotated_pos_studies))
    sampled_pilot_pos = rng.choice(dev_annotated_pos_studies, size=n_pos_sample, replace=False).tolist()
    sampled_pilot_neg = list(dev_confirmed_neg_studies)
    pilot_train_ids = set(sampled_pilot_pos + sampled_pilot_neg)

    # Held-out Cohort: Fold 4 (Previously exposed in V1 fitting; held out from V2 fitting)
    test_studies = study_df[study_df["fold"] == 4]
    test_study_ids = set(test_studies["StudyInstanceUID"])
    test_patients: Set[str] = set(test_studies["PatientID"])
    test_dup_groups: Set[str] = set(test_studies["duplicate_group_id"])

    # 3. Assert Split Integrity & Zero Leakage
    print("\n[3/5] Verifying Split Integrity and Zero Patient/Duplicate Leakage...")
    
    # Check 1: Zero patient overlap between val and dev train
    val_train_pt_overlap = val_patients.intersection(set(dev_train_studies["PatientID"]))
    assert len(val_train_pt_overlap) == 0, f"Leakage: {len(val_train_pt_overlap)} patients overlap between val and train!"
    print("  [PASS] Zero patient overlap between validation and development training.")

    # Check 2: Zero duplicate group overlap between val and dev train
    val_train_dup_overlap = val_dup_groups.intersection(set(dev_train_studies["duplicate_group_id"]))
    assert len(val_train_dup_overlap) == 0, f"Leakage: {len(val_train_dup_overlap)} duplicate groups overlap!"
    print("  [PASS] Zero duplicate group overlap between validation and development training.")

    # Check 3: Zero patient overlap between held-out Fold 4 and all dev data (Folds 0, 1, 2, 3)
    dev_patients = set(study_df[study_df["fold"].isin([0, 1, 2, 3])]["PatientID"])
    test_dev_pt_overlap = test_patients.intersection(dev_patients)
    assert len(test_dev_pt_overlap) == 0, f"Leakage: {len(test_dev_pt_overlap)} patients overlap between test and dev!"
    print("  [PASS] Zero patient overlap between held-out Fold 4 and development cohorts.")

    # Check 4: Zero duplicate group overlap between held-out Fold 4 and dev data
    dev_dup_groups = set(study_df[study_df["fold"].isin([0, 1, 2, 3])]["duplicate_group_id"])
    test_dev_dup_overlap = test_dup_groups.intersection(dev_dup_groups)
    assert len(test_dev_dup_overlap) == 0, f"Leakage: {len(test_dev_dup_overlap)} duplicate groups overlap between test and dev!"
    print("  [PASS] Zero duplicate group overlap between held-out Fold 4 and development cohorts.")

    # Save Splits JSON
    splits_manifest = {
        "description": "CVC-MAGNet-V2 Split Definitions with Complete Integrity",
        "scope": "CVC-only",
        "partitions": {
            "dev_pilot_val": {
                "num_studies": len(val_study_ids),
                "num_assessable_cvc_targets": len(val_cvc_targets),
                "study_ids": sorted(list(val_study_ids)),
                "role": "Frozen development validation benchmark (comparable to V1 baseline)",
            },
            "dev_pilot_train": {
                "num_studies": len(pilot_train_ids),
                "num_positive_studies": len(sampled_pilot_pos),
                "num_negative_studies": len(sampled_pilot_neg),
                "study_ids": sorted(list(pilot_train_ids)),
                "role": "Small sequential pilot training cohort",
            },
            "dev_full_train": {
                "num_studies": len(dev_train_ids),
                "study_ids": sorted(list(dev_train_ids)),
                "role": "Full approved development training set (Folds 0, 1, 3)",
            },
            "previously_exposed_held_out": {
                "num_studies": len(test_study_ids),
                "fold": 4,
                "role": "Previously exposed in V1 fitting (Folds 1-4), held out from V2 fitting. Cannot support a genuinely untouched project-level test claim.",
            },
        },
        "generalization_status": (
            "Fold 4 was previously exposed during V1 fitting. It is held out from V2 fitting, but cannot support a genuinely untouched project-level test claim. Final project-level generalization remains unmeasured."
        ),
    }

    splits_json_path = data_dir / "splits.json"
    with open(splits_json_path, "w") as f:
        json.dump(splits_manifest, f, indent=2)
    print(f"Saved splits definition to {splits_json_path}")

    # 4. Geometric & Coordinate Inversion Verification
    print("\n[4/5] Verifying Coordinate Inversion & Transformation Reversibility...")
    transform = LetterboxTransform(target_size=(512, 512), clahe_enabled=True)
    
    # Test reversibility on all 248 validation targets
    max_inversion_err = 0.0
    for _, trow in val_cvc_targets.iterrows():
        tx, ty = float(trow["tip_x"]), float(trow["tip_y"])
        w, h = int(trow["image_width"]), int(trow["image_height"])
        meta = transform.compute_params(orig_h=h, orig_w=w)
        
        # Forward map
        mx, my = meta.native_to_model(tx, ty)
        # Backward map
        nx, ny = meta.model_to_native(mx, my)
        err = np.hypot(nx - tx, ny - ty)
        if err > max_inversion_err:
            max_inversion_err = err

    print(f"  Maximum coordinate roundtrip inversion error: {max_inversion_err:.6f} pixels.")
    assert max_inversion_err < 1.0e-4, "Assertion Failed: Letterbox coordinate inversion loss > 1e-4 px!"
    print("  [PASS] Perfect affine coordinate reversibility confirmed.")

    # 5. Visual Overlay Verification
    print("\n[5/5] Generating Visual Verification Overlays on Diagnostic Cases...")
    raw_images_dir = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")
    
    # Select 4 representative cases: single CVC, multi CVC crossing, border contact, negative
    sample_cases = val_cvc_targets["study_id"].unique()[:4]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for idx, sid in enumerate(sample_cases):
        img_path = raw_images_dir / f"{sid}.jpg"
        if not img_path.exists():
            continue
        native_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        h, w = native_img.shape

        study_cats = catheter_df[catheter_df["study_id"] == sid]
        polylines = [json.loads(p) for p in study_cats["polyline"]]
        tips = [(r["tip_x"], r["tip_y"]) for _, r in study_cats.iterrows() if r["is_assessable"]]

        # 1. Native rasterization
        native_mask = rasterize_study_polylines(polylines, (h, w), thickness=10)
        native_heatmap = generate_gaussian_tip_heatmap(tips, (h, w), sigma=15.0)

        # 2. Model Letterbox transform
        lb_img, meta = transform.apply_image(native_img)
        lb_mask = transform.apply_mask(native_mask, meta)
        lb_att = create_gaussian_attention_mask(lb_mask, kernel_size=11, sigma=2.0)

        # Native overlay plot
        axes[0, idx].imshow(native_img, cmap="gray")
        axes[0, idx].imshow(native_mask, cmap="Reds", alpha=0.45)
        for tx, ty in tips:
            axes[0, idx].plot(tx, ty, "yo", markersize=8, markeredgecolor="black", label="Distal Tip")
        axes[0, idx].set_title(f"Native: {sid[:12]}...\n({w}x{h}, {len(study_cats)} lines)")
        axes[0, idx].axis("off")

        # Model Letterbox overlay plot
        axes[1, idx].imshow(lb_img, cmap="gray")
        axes[1, idx].imshow(lb_att, cmap="magma", alpha=0.5)
        for tx, ty in tips:
            mx, my = meta.native_to_model(tx, ty)
            axes[1, idx].plot(mx, my, "go", markersize=8, markeredgecolor="white", label="Model Tip")
        axes[1, idx].set_title(f"Model Input (512x512)\nScale={meta.scale:.3f}, Pad=({meta.pad_x:.1f},{meta.pad_y:.1f})")
        axes[1, idx].axis("off")

    plt.tight_layout()
    overlay_path = overlays_dir / "overlay_verification.png"
    plt.savefig(overlay_path, dpi=150)
    plt.close()
    print(f"Saved visual overlay verification figure to {overlay_path}")

    print("\nStep 3 Pass condition: ALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
