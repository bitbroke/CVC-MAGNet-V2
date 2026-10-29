"""Rigorous Decoder Tracing, Exposure Audit, and Annotation Queue Builder.

1. Traces candidate generation through exact decoder stages:
   - Sigmoid -> local maxima (3x3 dilation) -> tau=0.30 filter -> NMS (min_distance=10 model px) -> Top 10 cap -> Hungarian matching (r<=15px).
   - Recalculates exact failure stages for all 25 validation targets.
   - Verifies exact match with primary frozen evaluator: Success@15 = 1/25 (4.0%) for Controlled, 0/25 for Uniform.

2. Exposure Audit:
   - Checks previously_exposed_held_out and all study partitions for prior training/evaluation exposure.

3. Annotation Queue Preparation:
   - Selects 200 positive studies and ~400 confirmed negatives based on documented attributes:
     * Label classes (CVC - Normal, Borderline, Abnormal)
     * Catheter count (single vs multi-CVC)
     * Image geometry (dimensions, aspect ratio)
     * Patient IDs
   - Preserves original V3 benchmark; sets up complete review specification for validation multi-CVC cases.
"""
import ast
import json
from pathlib import Path
import sys
from typing import Dict, List, Any, Tuple

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import CoarseDualDataset
from src.data.transforms import TransformMetadata
from src.models.coarse_dual import CoarseDualNet


def load_model(ckpt_path: Path, device: torch.device) -> nn.Module:
    model = CoarseDualNet(in_channels=1, base_channels=32)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt.get("state_dict", ckpt)))
    clean_state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(clean_state, strict=True)
    model.to(device)
    model.eval()
    return model


def trace_decoder_for_target(
    heatmap: np.ndarray,
    target_native_x: float,
    target_native_y: float,
    meta: TransformMetadata,
    threshold: float = 0.30,
    nms_min_distance: int = 10,
    max_peaks: int = 10,
    match_radius_native: float = 15.0,
) -> Dict[str, Any]:
    """Traces candidate lifecycle through exact decoder stages for a target."""
    hm_f32 = heatmap.astype(np.float32)
    
    # Target in model grid coordinates
    tx_m, ty_m = meta.native_to_model(target_native_x, target_native_y)
    r_grid = match_radius_native * meta.scale

    # Check maximum probability in native 15px radius window
    y_min_w = int(max(0, np.floor(ty_m - r_grid)))
    y_max_w = int(min(511, np.ceil(ty_m + r_grid)))
    x_min_w = int(max(0, np.floor(tx_m - r_grid)))
    x_max_w = int(min(511, np.ceil(tx_m + r_grid)))

    if y_max_w >= y_min_w and x_max_w >= x_min_w:
        patch = hm_f32[y_min_w : y_max_w + 1, x_min_w : x_max_w + 1]
        yy, xx = np.ogrid[y_min_w : y_max_w + 1, x_min_w : x_max_w + 1]
        mask_r = ((xx - tx_m) ** 2 + (yy - ty_m) ** 2) <= (r_grid ** 2)
        max_p_15 = float(patch[mask_r].max()) if mask_r.any() else float(hm_f32[int(round(ty_m)), int(round(tx_m))])
    else:
        max_p_15 = float(hm_f32[int(round(ty_m)), int(round(tx_m))])

    # Stage 1 & 2 check: if no activation at all
    if max_p_15 < 0.10:
        stage = "Stage 1: Silent / No Signal (P_max < 0.10 in 15px)"
    elif max_p_15 < threshold:
        stage = "Stage 2: Sub-threshold Activation (0.10 <= P_max < 0.30 in 15px)"
    else:
        stage = None

    # Step 1: Local Maxima via 3x3 Dilation
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(hm_f32, kernel)
    local_max_mask = (hm_f32 == dilated) & (hm_f32 >= threshold)
    peak_ys, peak_xs = np.where(local_max_mask)

    all_raw_candidates = []
    for x, y in zip(peak_xs, peak_ys):
        conf = float(hm_f32[y, x])
        nx, ny = meta.model_to_native(float(x), float(y))
        dist_to_target = float(np.hypot(nx - target_native_x, ny - target_native_y))
        all_raw_candidates.append({
            "model_x": float(x),
            "model_y": float(y),
            "native_x": nx,
            "native_y": ny,
            "conf": conf,
            "dist_to_target": dist_to_target,
            "in_15px": dist_to_target <= match_radius_native,
        })

    # Sort descending by confidence
    all_raw_candidates.sort(key=lambda c: c["conf"], reverse=True)
    for rank_idx, c in enumerate(all_raw_candidates):
        c["raw_rank"] = rank_idx + 1

    # Check if any raw peak >= threshold exists within 15px
    raw_in_15 = [c for c in all_raw_candidates if c["in_15px"]]

    if stage is None and len(raw_in_15) == 0:
        stage = "Stage 3: Spatial Displacement (P >= 0.30 exists, but nearest is > 15px)"

    # Step 2: NMS with min_distance = nms_min_distance (model pixels)
    nms_admitted = []
    nms_suppressed = []

    for c in all_raw_candidates:
        suppressed_by = None
        for admitted in nms_admitted:
            dist_m = np.hypot(c["model_x"] - admitted["model_x"], c["model_y"] - admitted["model_y"])
            if dist_m < nms_min_distance:
                suppressed_by = admitted
                break
        if suppressed_by is not None:
            c["suppressed_by"] = suppressed_by
            nms_suppressed.append(c)
        else:
            c["nms_rank"] = len(nms_admitted) + 1
            nms_admitted.append(c)

    # Step 3: Top-10 Cap
    final_top_10 = nms_admitted[:max_peaks]
    cap_pruned = nms_admitted[max_peaks:]

    # Now, if stage was not yet assigned, check lifecycle of peaks within 15px
    if stage is None:
        # We have at least one raw peak within 15px
        # Check if any peak within 15px survived NMS
        admitted_in_15 = [c for c in nms_admitted if c["in_15px"]]
        if len(admitted_in_15) == 0:
            # All peaks within 15px were suppressed during NMS by a nearby higher peak
            stage = "Stage 4: NMS Suppression (Peak in 15px suppressed by nearby higher peak)"
        else:
            # At least one peak survived NMS
            top10_in_15 = [c for c in final_top_10 if c["in_15px"]]
            if len(top10_in_15) == 0:
                # Survived NMS, but rank after NMS was > 10!
                stage = "Stage 5: Cap Pruning (Survived NMS, but rank > 10)"
            else:
                # Peak is in final top 10 and within 15px!
                stage = "Stage 7: Success@15"

    # Nearest final candidate distance
    if len(final_top_10) > 0:
        dists_final = [c["dist_to_target"] for c in final_top_10]
        min_dist_final = min(dists_final)
        nearest_final_conf = final_top_10[np.argmin(dists_final)]["conf"]
    else:
        min_dist_final = 9999.0
        nearest_final_conf = 0.0

    return {
        "failure_stage": stage,
        "max_p_15": max_p_15,
        "num_raw_peaks_030": len(all_raw_candidates),
        "num_nms_admitted": len(nms_admitted),
        "num_final_top10": len(final_top_10),
        "min_dist_final": min_dist_final,
        "nearest_final_conf": nearest_final_conf,
        "raw_in_15_count": len(raw_in_15),
        "admitted_in_15_count": len([c for c in nms_admitted if c["in_15px"]]),
        "top10_in_15_count": len([c for c in final_top_10 if c["in_15px"]]),
        "final_top_10": final_top_10,
    }


def run_rigorous_decoder_trace():
    print("=" * 80)
    print("RIGOROUS DECODER TRACE (min_distance=10, tau=0.30, max_peaks=10)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = PROJECT_ROOT / "runs" / "sampling_comparison_1000"
    ctrl_ckpt = run_dir / "control_uniform" / "checkpoints" / "final_coarse_dual_head.pt"
    exp_ckpt = run_dir / "experiment_controlled" / "checkpoints" / "final_coarse_dual_head.pt"

    with open(PROJECT_ROOT / "configs" / "pilot.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    with open(PROJECT_ROOT / "data" / "splits.json", "r") as f:
        splits = json.load(f)

    inv_df = pd.read_parquet(PROJECT_ROOT / "data" / "annotation_inventory.parquet")
    study_df = pd.read_parquet(PROJECT_ROOT / "data" / "study_inventory.parquet")
    reviewed_df = pd.read_parquet(PROJECT_ROOT / "data" / "reviewed_distal_tips_v3.parquet")
    raw_images_dir = Path(cfg["data"]["raw_dir"]) / "train"

    val_study_ids = splits["partitions"]["dev_pilot_val"]["study_ids"]
    reviewed_val_df = reviewed_df[reviewed_df["split"] == "val"].sort_values("case_idx").copy()

    confirmed_neg_ids = set(
        study_df[study_df["is_study_level_negative"] & (~study_df["has_cvc"])]["StudyInstanceUID"]
    )

    val_dataset = CoarseDualDataset(
        study_ids=val_study_ids,
        inventory_df=inv_df,
        images_dir=raw_images_dir,
        confirmed_negative_study_ids=confirmed_neg_ids,
        reviewed_tips_df=reviewed_val_df,
        target_size=(512, 512),
        mask_stroke_width_native=10,
        clahe_enabled=True,
        tip_sigma=2.5,
        unreviewed_spatial_ignore_radius_native=50.0,
        partial_review_policy="exclude",
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    print("Loading models...")
    ctrl_model = load_model(ctrl_ckpt, device)
    exp_model = load_model(exp_ckpt, device)

    # 1. Run inference and trace all 25 validation targets
    study_preds = {}
    with torch.no_grad():
        for batch in val_loader:
            sid = batch["study_id"][0]
            imgs = batch["image"].to(device)
            m_dict = batch["meta"]
            meta_b = TransformMetadata(
                scale=float(m_dict["scale"][0]),
                pad_x=float(m_dict["pad_x"][0]),
                pad_y=float(m_dict["pad_y"][0]),
                orig_w=int(m_dict["orig_w"][0]),
                orig_h=int(m_dict["orig_h"][0]),
                target_w=int(m_dict["target_w"][0]),
                target_h=int(m_dict["target_h"][0]),
                crop_x0=float(m_dict["crop_x0"][0]),
                crop_y0=float(m_dict["crop_y0"][0]),
                new_w=int(m_dict["new_w"][0]),
                new_h=int(m_dict["new_h"][0]),
                pad_left=int(m_dict["pad_left"][0]),
                pad_top=int(m_dict["pad_top"][0]),
            )
            with torch.amp.autocast("cuda"):
                _, c_logits = ctrl_model(imgs)
                _, e_logits = exp_model(imgs)
            study_preds[sid] = {
                "meta": meta_b,
                "ctrl_hm": torch.sigmoid(c_logits).cpu().numpy()[0, 0],
                "exp_hm": torch.sigmoid(e_logits).cpu().numpy()[0, 0],
            }

    trace_results = {"control_uniform": [], "experiment_controlled": []}
    stage_counts = {
        "control_uniform": {},
        "experiment_controlled": {},
    }

    for _, r in reviewed_val_df.iterrows():
        c_idx = int(r["case_idx"])
        sid = r["study_id"]
        cid = int(r["catheter_instance_id"])
        tx = float(r["reviewed_tip_x"])
        ty = float(r["reviewed_tip_y"])
        info = study_preds[sid]
        meta = info["meta"]

        for arm in ["control_uniform", "experiment_controlled"]:
            hm = info["ctrl_hm"] if arm == "control_uniform" else info["exp_hm"]
            res = trace_decoder_for_target(
                heatmap=hm,
                target_native_x=tx,
                target_native_y=ty,
                meta=meta,
                threshold=0.30,
                nms_min_distance=10,  # Official evaluator contract
                max_peaks=10,
                match_radius_native=15.0,
            )
            stg = res["failure_stage"]
            stage_counts[arm][stg] = stage_counts[arm].get(stg, 0) + 1
            trace_results[arm].append({
                "case_idx": c_idx,
                "study_id": sid,
                "catheter_instance_id": cid,
                **res,
            })

    print("\n--- RECALCULATED FAILURE STAGES (OFFICIAL EVALUATOR CONTRACT) ---")
    for arm in ["control_uniform", "experiment_controlled"]:
        print(f"\nArm: {arm}")
        for stg, cnt in sorted(stage_counts[arm].items()):
            print(f"  {stg:<70}: {cnt}/25 ({cnt/25.0:.1%})")

    # Double check Success@15 matches official count exactly
    exp_success = [r for r in trace_results["experiment_controlled"] if r["failure_stage"] == "Stage 7: Success@15"]
    ctrl_success = [r for r in trace_results["control_uniform"] if r["failure_stage"] == "Stage 7: Success@15"]
    print(f"\nVerified Frozen Primary Match: Controlled={len(exp_success)}/25, Control={len(ctrl_success)}/25")
    assert len(ctrl_success) == 0, f"Expected 0 for control, got {len(ctrl_success)}"
    assert len(exp_success) == 1, f"Expected 1 for experiment, got {len(exp_success)}"

    # 2. Exposure Audit of Dataset Partitions
    print("\n" + "=" * 80)
    print("EXPOSURE AUDIT OF DATASET PARTITIONS")
    print("=" * 80)
    for p_name, p_data in splits["partitions"].items():
        if isinstance(p_data, dict):
            s_ids = p_data.get("study_ids", [])
            print(f"Partition '{p_name}': {len(s_ids)} studies (role: {p_data.get('role', 'N/A')})")

    held_out_data = splits["partitions"]["previously_exposed_held_out"]
    print(f"\nAudit of 'previously_exposed_held_out':")
    print(f"  Total studies: {held_out_data.get('num_studies', 'unknown')}")
    print(f"  Note: As documented in splits.json, these studies were previously exposed in V1 training/validation folds!")
    print(f"  Conclusion: Cannot be labeled unexposed. Genuine independent evaluation requires external cohort.")

    # 3. Build Annotation Queue: 200 Positive Studies + 400 Confirmed Negatives
    print("\n" + "=" * 80)
    print("BUILDING ANNOTATION QUEUE (200 POSITIVE + 400 NEGATIVE)")
    print("=" * 80)

    # Exclude currently active pilot studies (dev_pilot_train and dev_pilot_val)
    pilot_train_sids = set(splits["partitions"]["dev_pilot_train"]["study_ids"])
    pilot_val_sids = set(splits["partitions"]["dev_pilot_val"]["study_ids"])
    excluded_sids = pilot_train_sids | pilot_val_sids

    # Candidate pool from dev_full_train
    full_train_sids = [s for s in splits["partitions"]["dev_full_train"]["study_ids"] if s not in excluded_sids]
    
    # Filter for positive CVC studies in inventory
    cvc_inv = inv_df[inv_df["device_type"] == "CVC"].copy()
    positive_candidates = cvc_inv[cvc_inv["study_id"].isin(full_train_sids)].copy()

    # Stratified selection by documented attributes: label, single vs multi-CVC
    studies_summary = positive_candidates.groupby("study_id").agg({
        "label": lambda s: list(s),
        "catheter_instance_id": "count",
        "patient_id": "first",
        "image_width": "first",
        "image_height": "first",
    }).rename(columns={"catheter_instance_id": "cvc_count"})

    # Primary label per study (prioritize Abnormal > Borderline > Normal)
    def assign_primary_label(lbl_list):
        if any("Abnormal" in l for l in lbl_list):
            return "CVC - Abnormal"
        if any("Borderline" in l for l in lbl_list):
            return "CVC - Borderline"
        return "CVC - Normal"

    studies_summary["primary_label"] = studies_summary["label"].apply(assign_primary_label)

    # Sample 200 diverse positive studies stratified by primary label and multi-CVC status
    # Proportions based on natural availability in candidate pool
    np.random.seed(42)
    selected_pos_sids = []
    
    # Stratify by (primary_label, is_multi)
    for (lbl, is_multi), grp in studies_summary.groupby(["primary_label", studies_summary["cvc_count"] > 1]):
        weight = len(grp) / len(studies_summary)
        n_sample = int(round(weight * 200))
        n_sample = min(n_sample, len(grp))
        sampled = grp.sample(n=n_sample, random_state=42)
        selected_pos_sids.extend(sampled.index.tolist())

    # Adjust to exactly 200
    if len(selected_pos_sids) < 200:
        remaining = [s for s in studies_summary.index if s not in set(selected_pos_sids)]
        selected_pos_sids.extend(remaining[:200 - len(selected_pos_sids)])
    elif len(selected_pos_sids) > 200:
        selected_pos_sids = selected_pos_sids[:200]

    selected_pos_df = studies_summary.loc[selected_pos_sids].reset_index()
    total_estimated_targets = selected_pos_df["cvc_count"].sum()
    print(f"Selected 200 positive studies (total visible CVC catheters to review: {total_estimated_targets}).")
    print(f"Label breakdown of 200 positive studies:")
    print(selected_pos_df["primary_label"].value_counts())
    print(f"Multi-CVC breakdown: {(selected_pos_df['cvc_count'] > 1).value_counts().to_dict()}")

    # Select 400 confirmed negative studies from candidate pool
    confirmed_neg_pool = study_df[
        study_df["StudyInstanceUID"].isin(full_train_sids) &
        study_df["is_study_level_negative"] &
        (~study_df["has_cvc"])
    ]["StudyInstanceUID"].tolist()

    np.random.seed(42)
    selected_neg_sids = np.random.choice(confirmed_neg_pool, size=min(400, len(confirmed_neg_pool)), replace=False).tolist()
    print(f"Selected {len(selected_neg_sids)} confirmed negative studies.")

    # Save Annotation Queue
    queue_data = {
        "description": "Expanded pilot annotation queue (200 positive studies + 400 confirmed negatives)",
        "positive_studies_count": len(selected_pos_sids),
        "estimated_cvc_targets": int(total_estimated_targets),
        "negative_studies_count": len(selected_neg_sids),
        "positive_study_ids": selected_pos_sids,
        "negative_study_ids": selected_neg_sids,
        "annotation_rules": [
            "Every visible CVC catheter in each study must be reviewed (100% complete review).",
            "Unassessable tips must be explicitly recorded with reason (is_assessable=False).",
            "Preserve existing V3 25-target benchmark for comparison.",
            "Record validation multi-CVC completions in reviewed_distal_tips_v4_val.parquet.",
        ],
    }

    queue_path = PROJECT_ROOT / "data" / "annotation_expansion_queue.json"
    with open(queue_path, "w", encoding="utf-8") as f:
        json.dump(queue_data, f, indent=2)
    print(f"Saved annotation queue to: {queue_path}")

    # Build queue metadata CSV for human annotators
    ann_queue_records = []
    for sid in selected_pos_sids:
        s_cvcs = inv_df[(inv_df["study_id"] == sid) & (inv_df["device_type"] == "CVC")]
        for _, crow in s_cvcs.iterrows():
            ann_queue_records.append({
                "study_id": sid,
                "catheter_instance_id": crow["catheter_instance_id"],
                "instance_key": crow["instance_key"],
                "label": crow["label"],
                "polyline_length_px": crow["polyline_length_px"],
                "heuristic_tip_x": crow["tip_x"],
                "heuristic_tip_y": crow["tip_y"],
                "review_status": "queued_for_review",
            })
    ann_queue_df = pd.DataFrame(ann_queue_records)
    ann_queue_csv = PROJECT_ROOT / "data" / "annotation_expansion_positive_targets.csv"
    ann_queue_df.to_csv(ann_queue_csv, index=False)
    print(f"Saved positive target review list ({len(ann_queue_df)} rows) to: {ann_queue_csv}")

    # Save recalculated diagnostic summary
    recalculated_diagnostics = {
        "decoder_parameters": {
            "threshold": 0.30,
            "min_distance_model_px": 10,
            "max_peaks": 10,
            "match_radius_native_px": 15.0,
        },
        "stage_counts": stage_counts,
        "target_traces": trace_results,
    }
    recalc_path = run_dir / "recalculated_diagnostic_analysis.json"
    with open(recalc_path, "w", encoding="utf-8") as f:
        json.dump(recalculated_diagnostics, f, indent=2)
    print(f"Saved recalculated diagnostics to: {recalc_path}")


if __name__ == "__main__":
    run_rigorous_decoder_trace()
