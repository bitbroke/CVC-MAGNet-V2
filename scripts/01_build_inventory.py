"""Script 01: Build Annotation Inventory for CVC-MAGNet-V2.

Builds a comprehensive, clinically auditable inventory of all catheter instances
and study-level records.

Fields per catheter:
- Study, patient, duplicate-group IDs (Split integrity)
- Catheter instance ID and instance key (Preserve identity)
- Polyline and image dimensions (Segmentation supervision)
- Tip coordinate and annotation source (Distinguish reviewed from inferred tips)
- Assessability and review status (Handle unknown or cropped tips)
- Image-level CVC presence (Support negative supervision)

Pass conditions:
- Valid coordinates within [0, width) and [0, height) for assessable tips.
- Consistent identities (no duplicate instance keys).
- Explicit unknown states for unassessable / cropped tips.
- Documented annotation sources (expert_reviewed vs heuristic_border_distance).
"""
import ast
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tip_rules import resolve_catheter_tip


def get_image_size(img_path: Path) -> Tuple[int, int]:
    """Return (width, height) without decoding pixel raster.
    
    Raises FileNotFoundError or RuntimeError if the image cannot be read.
    Never invents fallback dimensions.
    """
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found on disk: {img_path}")
    try:
        with Image.open(img_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        raise RuntimeError(f"Failed to read image dimensions from {img_path}: {e}") from e


def build_duplicate_mapping(train_df: pd.DataFrame, duplicate_csv: Path) -> Dict[str, str]:
    """Map each PatientID to a connected component group ID based on shared image hashes."""
    G = nx.Graph()
    for pt in train_df["PatientID"].unique():
        G.add_node(pt)

    if duplicate_csv.exists():
        print(f"Loading exact image duplicates from {duplicate_csv}...")
        dup_df = pd.read_csv(duplicate_csv)
        for sha, group in dup_df.groupby("sha256"):
            pts = group["PatientID"].unique().tolist()
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    G.add_edge(pts[i], pts[j])

    pt_to_group = {}
    for component in nx.connected_components(G):
        group_id = min(component)
        for pt in component:
            pt_to_group[pt] = group_id

    return pt_to_group


def main():
    print("=" * 70)
    print("STEP 2: BUILD ANNOTATION INVENTORY (CVC-MAGNet-V2)")
    print("=" * 70)

    # 1. Load Paths
    raw_dir = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification")
    train_csv = raw_dir / "train.csv"
    annotations_csv = raw_dir / "train_annotations.csv"
    train_images_dir = raw_dir / "train"

    v1_ref_dir = Path(r"C:\Ankshit\Catheter")
    duplicate_csv = v1_ref_dir / "outputs" / "exact_duplicate_images.csv"
    expert_manifest_csv = v1_ref_dir / "reports" / "expert_review_manifest.csv"

    out_data_dir = PROJECT_ROOT / "data"
    out_data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading metadata from {train_csv}...")
    train_df = pd.read_csv(train_csv)
    print(f"Loaded {len(train_df):,} studies across {train_df['PatientID'].nunique():,} unique patients.")

    # 2. Build Duplicate Groups
    print("[2/6] Establishing duplicate groups for split integrity...")
    pt_to_group = build_duplicate_mapping(train_df, duplicate_csv)
    train_df["duplicate_group_id"] = train_df["PatientID"].map(pt_to_group)
    print(f"Mapped {train_df['PatientID'].nunique():,} patients to {train_df['duplicate_group_id'].nunique():,} isolated groups.")

    # Study-level CVC presence and label indicators
    cvc_label_cols = ["CVC - Normal", "CVC - Borderline", "CVC - Abnormal"]
    other_label_cols = [
        "ETT - Normal", "ETT - Borderline", "ETT - Abnormal",
        "NGT - Normal", "NGT - Borderline", "NGT - Abnormal", "NGT - Incompletely Imaged",
        "Swan Ganz Catheter Present"
    ]
    train_df["has_cvc"] = train_df[cvc_label_cols].sum(axis=1) > 0
    train_df["has_other_lines"] = train_df[other_label_cols].sum(axis=1) > 0
    train_df["is_study_level_negative"] = (~train_df["has_cvc"]) & (train_df["has_other_lines"])

    study_meta = train_df.set_index("StudyInstanceUID")

    # 3. Load Expert Review Manifest (Audit Ground Truth)
    print(f"[3/6] Loading expert review audit records from {expert_manifest_csv}...")
    expert_lookup = {}
    if expert_manifest_csv.exists():
        expert_df = pd.read_csv(expert_manifest_csv)
        for _, erow in expert_df.iterrows():
            # Key by (study_id, instance_id)
            sid = erow["study_id"]
            iid = int(erow["instance_id"])
            expert_lookup[(sid, iid)] = erow
        print(f"Loaded {len(expert_lookup)} audited cases ({len([k for k, v in expert_lookup.items() if not pd.isna(v.get('expert_distal_tip_x'))])} with distal tips).")
    else:
        print("WARNING: expert_review_manifest.csv not found; proceeding with unreviewed heuristic labels.")

    # 4. Load Polylines and Query Image Dimensions
    print(f"[4/6] Loading polyline annotations from {annotations_csv}...")
    ann_df = pd.read_csv(annotations_csv)
    unique_studies = ann_df["StudyInstanceUID"].unique()
    print(f"Querying image dimensions for {len(unique_studies):,} annotated studies using parallel workers...")

    study_sizes = {}
    def fetch_size(uid):
        return uid, get_image_size(train_images_dir / f"{uid}.jpg")

    with ThreadPoolExecutor(max_workers=16) as executor:
        for uid, size in executor.map(fetch_size, unique_studies):
            study_sizes[uid] = size

    # 5. Build Catheter Inventory
    print("[5/6] Building catheter instance inventory with explicit provenance...")
    inventory_rows = []
    
    # Track CVC counts per study
    study_cvc_counts = {}

    for uid, grp in ann_df.groupby("StudyInstanceUID", sort=False):
        if uid not in study_sizes:
            raise RuntimeError(f"Dimensions missing for study UID {uid}; dimensions must be explicitly determined from disk.")
        orig_w, orig_h = study_sizes[uid]
        pt_id = study_meta.loc[uid, "PatientID"] if uid in study_meta.index else "UNKNOWN"
        dup_grp_id = study_meta.loc[uid, "duplicate_group_id"] if uid in study_meta.index else pt_id
        has_cvc = bool(study_meta.loc[uid, "has_cvc"]) if uid in study_meta.index else False
        is_neg = bool(study_meta.loc[uid, "is_study_level_negative"]) if uid in study_meta.index else False

        cvc_count = 0
        instance_id = 0

        for _, row in grp.iterrows():
            instance_id += 1
            raw_pts = ast.literal_eval(row["data"])
            pts_np = np.asarray(raw_pts, dtype=np.float32)
            dev_label = row["label"]
            dev_type = "Swan" if "Swan" in dev_label else dev_label[:3]

            if dev_type == "CVC":
                cvc_count += 1

            # Check if this instance has an expert review audit
            expert_match = expert_lookup.get((uid, instance_id))

            # Default geometric resolution
            res = resolve_catheter_tip(pts_np, dev_type, image_shape=(orig_h, orig_w))

            # Check if this instance has actual documented clinical review
            reviewer_id = str(expert_match.get("reviewer_id", "")) if expert_match is not None else ""
            has_clinical_review = (
                expert_match is not None
                and not pd.isna(expert_match.get("expert_distal_tip_x"))
                and ("awaiting" not in reviewer_id.lower() and "automated" not in reviewer_id.lower() and reviewer_id != "")
            )

            if has_clinical_review:
                tip_x = float(expert_match["expert_distal_tip_x"])
                tip_y = float(expert_match["expert_distal_tip_y"])
                ann_source = "clinical_expert_verified"
                review_status = "documented_clinical_review"
                is_assessable = True
                is_oob = False
                is_border = False
                expert_note = str(expert_match.get("expert_notes", ""))
                expert_fail_class = str(expert_match.get("expert_failure_classification", "SUCCESS"))
            elif expert_match is not None and not pd.isna(expert_match.get("expert_distal_tip_x")):
                # Populated in diagnostic review packet, but strictly automated provenance awaiting audit
                tip_x = float(expert_match["expert_distal_tip_x"])
                tip_y = float(expert_match["expert_distal_tip_y"])
                ann_source = "diagnostic_packet_awaiting_audit"
                review_status = "unreviewed_automated_pipeline"
                is_assessable = True
                is_oob = False
                is_border = False
                expert_note = str(expert_match.get("expert_notes", ""))
                expert_fail_class = str(expert_match.get("expert_failure_classification", "UNREVIEWED"))
            else:
                # Unreviewed heuristic endpoint - DO NOT automatically promote to verified
                if res["is_assessable"]:
                    tip_x = float(res["tip"][0])
                    tip_y = float(res["tip"][1])
                    ann_source = "heuristic_border_distance"
                    review_status = "unreviewed_heuristic"
                    is_assessable = True
                    is_oob = False
                    is_border = False
                else:
                    # Unassessable / border / out of bounds
                    tip_x = np.nan
                    tip_y = np.nan
                    ann_source = "unassessable_crop"
                    review_status = "unassessable"
                    is_assessable = False
                    is_oob = res["is_out_of_bounds"]
                    is_border = res["is_border_contact"]
                expert_note = ""
                expert_fail_class = "UNREVIEWED"

            instance_key = f"{uid}_{instance_id}"

            inventory_rows.append({
                "study_id": uid,
                "patient_id": pt_id,
                "duplicate_group_id": dup_grp_id,
                "catheter_instance_id": instance_id,
                "instance_key": instance_key,
                "device_type": dev_type,
                "label": dev_label,
                "polyline": json.dumps(raw_pts),
                "num_points": len(raw_pts),
                "polyline_length_px": res["length_px"],
                "image_width": orig_w,
                "image_height": orig_h,
                "tip_x": tip_x,
                "tip_y": tip_y,
                "entry_x": float(res["entry"][0]),
                "entry_y": float(res["entry"][1]),
                "tip_endpoint_index": res["tip_idx"],
                "annotation_source": ann_source,
                "review_status": review_status,
                "is_assessable": is_assessable,
                "is_out_of_bounds": is_oob,
                "is_border_contact": is_border,
                "image_has_cvc": has_cvc,
                "is_study_level_negative": is_neg,
                "has_clinical_expert_verification": bool(has_clinical_review),
                "expert_notes": expert_note,
                "expert_failure_classification": expert_fail_class,
            })

        study_cvc_counts[uid] = cvc_count

    catheter_df = pd.DataFrame(inventory_rows)

    # Attach exact cvc_count to catheter records
    catheter_df["image_cvc_count"] = catheter_df["study_id"].map(study_cvc_counts)

    # 6. Verify Pass Conditions
    print("[6/6] Verifying Pass Conditions on Catheter Inventory...")
    
    # 1. Unique identities
    assert catheter_df["instance_key"].nunique() == len(catheter_df), "Assertion Failed: Duplicate instance keys detected!"
    print("  [PASS] Instance keys are uniquely preserved.")

    # 2. Coordinates within image bounds for assessable tips
    assessable_mask = catheter_df["is_assessable"]
    assessable_df = catheter_df[assessable_mask]
    
    assert not assessable_df["tip_x"].isna().any(), "Assertion Failed: Assessable tips contain NaN tip_x!"
    assert not assessable_df["tip_y"].isna().any(), "Assertion Failed: Assessable tips contain NaN tip_y!"
    
    valid_x = (assessable_df["tip_x"] >= 0) & (assessable_df["tip_x"] < assessable_df["image_width"])
    valid_y = (assessable_df["tip_y"] >= 0) & (assessable_df["tip_y"] < assessable_df["image_height"])
    assert valid_x.all(), "Assertion Failed: Assessable tips have out-of-bounds tip_x!"
    assert valid_y.all(), "Assertion Failed: Assessable tips have out-of-bounds tip_y!"
    print("  [PASS] All assessable coordinates strictly bounded within image width and height.")

    # 3. Explicit unknown / NaN for unassessable tips
    unassessable_df = catheter_df[~assessable_mask]
    assert unassessable_df["tip_x"].isna().all(), "Assertion Failed: Unassessable tips must have NaN coordinates!"
    assert unassessable_df["tip_y"].isna().all(), "Assertion Failed: Unassessable tips must have NaN coordinates!"
    print("  [PASS] Unassessable tips have explicit NaN coordinates.")

    # 4. Documented annotation sources
    sources = catheter_df["annotation_source"].value_counts().to_dict()
    print(f"  [PASS] Documented annotation sources: {sources}")

    # 5. Provenance check: zero unverified automated endpoints promoted to clinical expert verified
    num_expert_verified = int(catheter_df["has_clinical_expert_verification"].sum())
    assert num_expert_verified == 0, f"Error: {num_expert_verified} automated endpoints inappropriately promoted to expert verified!"
    print(f"  [PASS] Provenance verified: {num_expert_verified} endpoints marked as clinical expert verified (all remain automated/unreviewed).")

    # Summary Statistics
    print("\nInventory Breakdown by Device & Assessability:")
    print(catheter_df.groupby(["device_type", "is_assessable", "annotation_source"]).size().to_string())

    # Save catheter inventory
    inv_path = out_data_dir / "annotation_inventory.parquet"
    catheter_df.to_parquet(inv_path, index=False)
    print(f"\nSaved {len(catheter_df):,} catheter records to {inv_path}")

    # Build and save study inventory (30,083 studies) for full split & negative supervision support
    train_df["cvc_count"] = train_df["StudyInstanceUID"].map(lambda u: study_cvc_counts.get(u, 0))
    study_inv_path = out_data_dir / "study_inventory.parquet"
    train_df.to_parquet(study_inv_path, index=False)
    print(f"Saved {len(train_df):,} study records to {study_inv_path}")

    print("\nStep 2 Pass condition: ALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
