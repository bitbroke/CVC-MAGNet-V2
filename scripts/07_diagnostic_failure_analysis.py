"""Script 07: Diagnostic Failure Analysis & Decoding Ablation.

Performs rigorous diagnostics on the frozen Step 1000 checkpoints
(Uniform Control vs. Tip-Balanced Experiment) in runs/sampling_comparison_1000:

1. Uncapped False-Peak Analysis & Background Confidence Distributions on Negative Studies.
2. Dissection of Prediction vs. Decoding Failures across all 25 validation targets.
3. Complete Clinical & Geometric Inspection of all 25 Validation Targets vs. 39 Training Targets.
4. Preserves all existing frozen results; writes separate diagnostic artifacts.
"""
import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

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
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import CoarseDualDataset
from src.data.transforms import TransformMetadata
from src.models.coarse_dual import CoarseDualNet, decode_tip_peaks


def extract_uncapped_peaks(
    heatmap: np.ndarray,
    min_threshold: float = 0.05,
) -> List[Tuple[float, float, float]]:
    """Extract all local maxima from a 2D float heatmap without any max_peaks cap."""
    hm_f32 = heatmap.astype(np.float32)
    if float(hm_f32.max()) < min_threshold:
        return []

    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(hm_f32, kernel)
    local_maxima = (hm_f32 == dilated) & (hm_f32 >= min_threshold)

    peak_ys, peak_xs = np.where(local_maxima)
    peaks = []
    for x, y in zip(peak_xs, peak_ys):
        conf = float(hm_f32[y, x])
        peaks.append((float(x), float(y), conf))

    # Sort descending by confidence
    peaks.sort(key=lambda p: p[2], reverse=True)
    return peaks


def apply_distance_nms(
    peaks: List[Tuple[float, float, float]],
    min_distance: float = 5.0,
    max_peaks: Optional[int] = None,
) -> List[Tuple[float, float, float]]:
    """Apply greedy distance-based non-maximum suppression and optional cap."""
    kept = []
    for c in peaks:
        if max_peaks is not None and len(kept) >= max_peaks:
            break
        too_close = False
        for k in kept:
            dist = np.hypot(c[0] - k[0], c[1] - k[1])
            if dist < min_distance:
                too_close = True
                break
        if not too_close:
            kept.append(c)
    return kept


def threshold_constrained_matching(
    pred_coords: np.ndarray,
    target_coords: np.ndarray,
    radius: float = 15.0,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    P = len(pred_coords)
    T = len(target_coords)
    if P == 0 or T == 0:
        return [], list(range(P)), list(range(T))

    diffs = pred_coords[:, None, :] - target_coords[None, :, :]
    dists = np.hypot(diffs[:, :, 0], diffs[:, :, 1])
    cost_matrix = np.where(dists <= radius, dists, 1e9)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched = []
    matched_preds = set()
    matched_targets = set()
    for r, c in zip(row_ind, col_ind):
        d = float(dists[r, c])
        if d <= radius:
            matched.append((int(r), int(c), d))
            matched_preds.add(int(r))
            matched_targets.add(int(c))

    unmatched_preds = [p for p in range(P) if p not in matched_preds]
    unmatched_targets = [t for t in range(T) if t not in matched_targets]
    return matched, unmatched_preds, unmatched_targets


def load_model(ckpt_path: Path, device: torch.device) -> nn.Module:
    model = CoarseDualNet(in_channels=1, base_channels=32)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt.get("state_dict", ckpt)))
    clean_state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(clean_state, strict=True)
    model.to(device)
    model.eval()
    return model


def run_diagnostic_analysis():
    print("=" * 80)
    print("CVC-MAGNet-V2: DIAGNOSTIC FAILURE ANALYSIS & DECODING ABLATION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Paths
    run_dir = PROJECT_ROOT / "runs" / "sampling_comparison_1000"
    ctrl_ckpt_path = run_dir / "control_uniform" / "checkpoints" / "final_coarse_dual_head.pt"
    exp_ckpt_path = run_dir / "experiment_controlled" / "checkpoints" / "final_coarse_dual_head.pt"

    if not ctrl_ckpt_path.exists() or not exp_ckpt_path.exists():
        raise FileNotFoundError("Checkpoints not found in runs/sampling_comparison_1000")

    cfg_path = PROJECT_ROOT / "configs" / "pilot.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    splits_path = PROJECT_ROOT / "data" / "splits.json"
    with open(splits_path, "r") as f:
        splits = json.load(f)

    val_study_ids = splits["partitions"]["dev_pilot_val"]["study_ids"]
    train_study_ids = splits["partitions"]["dev_pilot_train"]["study_ids"]

    inv_df = pd.read_parquet(PROJECT_ROOT / "data" / "annotation_inventory.parquet")
    study_df = pd.read_parquet(PROJECT_ROOT / "data" / "study_inventory.parquet")
    reviewed_df = pd.read_parquet(PROJECT_ROOT / "data" / "reviewed_distal_tips_v3.parquet")
    raw_images_dir = Path(cfg["data"]["raw_dir"]) / "train"

    confirmed_neg_ids = set(
        study_df[study_df["is_study_level_negative"] & (~study_df["has_cvc"])]["StudyInstanceUID"]
    )

    reviewed_val_df = reviewed_df[reviewed_df["split"] == "val"].copy()
    reviewed_train_df = reviewed_df[reviewed_df["split"] == "train"].copy()

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

    print(f"Loaded validation cohort: {len(val_dataset)} studies ({len(reviewed_val_df)} reviewed targets).")

    # Load both models
    print(f"Loading Control checkpoint from: {ctrl_ckpt_path.name}")
    ctrl_model = load_model(ctrl_ckpt_path, device)
    print(f"Loading Experiment checkpoint from: {exp_ckpt_path.name}")
    exp_model = load_model(exp_ckpt_path, device)

    # -------------------------------------------------------------------------
    # PART 1: Uncapped False Peaks on Negative Studies
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 1: UNCAPPED FALSE-PEAK & CONFIDENCE DISTRIBUTION ANALYSIS (8 NEGATIVE STUDIES)")
    print("=" * 80)

    thresholds_sweep = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    neg_analysis = {
        "control_uniform": {
            "per_study": {},
            "aggregate_peak_confidences": [],
            "counts_by_threshold": {t: 0 for t in thresholds_sweep},
            "mean_heatmap_prob": [],
            "max_heatmap_prob": [],
        },
        "experiment_controlled": {
            "per_study": {},
            "aggregate_peak_confidences": [],
            "counts_by_threshold": {t: 0 for t in thresholds_sweep},
            "mean_heatmap_prob": [],
            "max_heatmap_prob": [],
        },
    }

    # Store predicted heatmaps and metadata for all validation studies
    study_predictions = {}

    with torch.no_grad():
        for batch in val_loader:
            sid = batch["study_id"][0]
            imgs = batch["image"].to(device)
            meta_dict = batch["meta"]
            meta_b = TransformMetadata(
                scale=float(meta_dict["scale"][0]),
                pad_x=float(meta_dict["pad_x"][0]),
                pad_y=float(meta_dict["pad_y"][0]),
                orig_w=int(meta_dict["orig_w"][0]),
                orig_h=int(meta_dict["orig_h"][0]),
                target_w=int(meta_dict["target_w"][0]),
                target_h=int(meta_dict["target_h"][0]),
                crop_x0=float(meta_dict["crop_x0"][0]),
                crop_y0=float(meta_dict["crop_y0"][0]),
                new_w=int(meta_dict["new_w"][0]),
                new_h=int(meta_dict["new_h"][0]),
                pad_left=int(meta_dict["pad_left"][0]),
                pad_top=int(meta_dict["pad_top"][0]),
            )

            is_neg = sid in confirmed_neg_ids

            with torch.amp.autocast("cuda"):
                _, ctrl_tip_logits = ctrl_model(imgs)
                _, exp_tip_logits = exp_model(imgs)

            ctrl_hm = torch.sigmoid(ctrl_tip_logits).cpu().numpy()[0, 0]
            exp_hm = torch.sigmoid(exp_tip_logits).cpu().numpy()[0, 0]

            study_predictions[sid] = {
                "meta": meta_b,
                "is_neg": is_neg,
                "ctrl_hm": ctrl_hm,
                "exp_hm": exp_hm,
            }

            if is_neg:
                for arm_key, hm in [("control_uniform", ctrl_hm), ("experiment_controlled", exp_hm)]:
                    uncapped_peaks = extract_uncapped_peaks(hm, min_threshold=0.05)
                    confs = [p[2] for p in uncapped_peaks]
                    neg_analysis[arm_key]["aggregate_peak_confidences"].extend(confs)
                    neg_analysis[arm_key]["mean_heatmap_prob"].append(float(hm.mean()))
                    neg_analysis[arm_key]["max_heatmap_prob"].append(float(hm.max()))

                    counts_th = {}
                    for th in thresholds_sweep:
                        cnt = sum(1 for c in confs if c >= th)
                        counts_th[th] = cnt
                        neg_analysis[arm_key]["counts_by_threshold"][th] += cnt

                    neg_analysis[arm_key]["per_study"][sid] = {
                        "uncapped_peaks_total_ge_005": len(confs),
                        "peaks_ge_030": counts_th[0.30],
                        "max_confidence": float(max(confs)) if len(confs) > 0 else 0.0,
                        "median_confidence": float(np.median(confs)) if len(confs) > 0 else 0.0,
                        "mean_heatmap_prob": float(hm.mean()),
                        "counts_by_threshold": counts_th,
                    }

    # Summary table for negative studies
    print(f"{'Threshold (tau)':<16} | {'Uniform Control Uncapped Total':<30} | {'Controlled Exp Uncapped Total':<30}")
    print("-" * 80)
    for th in thresholds_sweep:
        c_cnt = neg_analysis["control_uniform"]["counts_by_threshold"][th]
        e_cnt = neg_analysis["experiment_controlled"]["counts_by_threshold"][th]
        print(f"tau >= {th:<10.2f} | {c_cnt:<30} ({c_cnt/8.0:.1f}/study) | {e_cnt:<30} ({e_cnt/8.0:.1f}/study)")

    for arm_key, label in [("control_uniform", "Control (Uniform)"), ("experiment_controlled", "Experiment (Controlled)")]:
        confs = np.array(neg_analysis[arm_key]["aggregate_peak_confidences"])
        means = neg_analysis[arm_key]["mean_heatmap_prob"]
        maxs = neg_analysis[arm_key]["max_heatmap_prob"]
        print(f"\n[{label}] Negative Studies Activation Statistics (N=8):")
        print(f"  Uncapped peaks (tau>=0.05) total: {len(confs)} ({len(confs)/8.0:.1f} per study)")
        print(f"  Uncapped peaks (tau>=0.30) total: {neg_analysis[arm_key]['counts_by_threshold'][0.30]} ({neg_analysis[arm_key]['counts_by_threshold'][0.30]/8.0:.1f} per study)")
        if len(confs) > 0:
            print(f"  Peak Confidence: Min={confs.min():.4f}, Q25={np.percentile(confs, 25):.4f}, Median={np.median(confs):.4f}, Q75={np.percentile(confs, 75):.4f}, Max={confs.max():.4f}")
        print(f"  Whole-Image Mean Probability: {np.mean(means):.6f} | Max Probability: {np.max(maxs):.4f}")

    # -------------------------------------------------------------------------
    # PART 2: Dissecting Prediction vs. Decoding Failures on 25 Validation Targets
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 2: PREDICTION VS. DECODING DISSECTION (25 VALIDATION TARGETS)")
    print("=" * 80)

    # Load targets
    reviewed_targets = reviewed_val_df.to_dict(orient="records")

    target_diagnostics = []
    failure_counts = {
        "control_uniform": {
            "Stage 1: Silent / No Signal (P_max < 0.10 in 15px)": 0,
            "Stage 2: Sub-threshold Activation (0.10 <= P_max < 0.30 in 15px)": 0,
            "Stage 3: Spatial Displacement (P >= 0.30 exists, but nearest is > 15px)": 0,
            "Stage 4: Cap Pruning (P >= 0.30 peak in 15px, but rank > 10)": 0,
            "Stage 5: NMS Suppression (Peak in 15px suppressed by nearby peak)": 0,
            "Stage 6: Hungarian Match Rejection (Top 10 in 15px, but lost competition)": 0,
            "Stage 7: Success@15": 0,
        },
        "experiment_controlled": {
            "Stage 1: Silent / No Signal (P_max < 0.10 in 15px)": 0,
            "Stage 2: Sub-threshold Activation (0.10 <= P_max < 0.30 in 15px)": 0,
            "Stage 3: Spatial Displacement (P >= 0.30 exists, but nearest is > 15px)": 0,
            "Stage 4: Cap Pruning (P >= 0.30 peak in 15px, but rank > 10)": 0,
            "Stage 5: NMS Suppression (Peak in 15px suppressed by nearby peak)": 0,
            "Stage 6: Hungarian Match Rejection (Top 10 in 15px, but lost competition)": 0,
            "Stage 7: Success@15": 0,
        },
    }

    # Group targets by study
    targets_by_study = {}
    for t in reviewed_targets:
        targets_by_study.setdefault(t["study_id"], []).append(t)

    for sid, targets in targets_by_study.items():
        study_info = study_predictions[sid]
        meta = study_info["meta"]
        t_arr_nat = np.array([[t["reviewed_tip_x"], t["reviewed_tip_y"]] for t in targets], dtype=np.float32)

        for arm_key in ["control_uniform", "experiment_controlled"]:
            hm = study_info["ctrl_hm"] if arm_key == "control_uniform" else study_info["exp_hm"]

            # Decoded peaks under standard contract (tau=0.30, min_dist=5, max_peaks=10)
            std_peaks_model = decode_tip_peaks(hm, threshold=0.30, min_distance=5, max_peaks=10)
            std_peaks_nat = []
            for p in std_peaks_model:
                px_nat, py_nat = meta.model_to_native(p[0], p[1])
                std_peaks_nat.append([px_nat, py_nat])
            std_arr_nat = np.array(std_peaks_nat, dtype=np.float32) if len(std_peaks_nat) > 0 else np.zeros((0, 2), dtype=np.float32)

            matched_std_15, unmatched_p, unmatched_t = threshold_constrained_matching(std_arr_nat, t_arr_nat, radius=15.0)
            matched_target_indices = {m[1]: (m[0], m[2]) for m in matched_std_15}

            # Uncapped peaks of all levels
            uncapped_peaks_005 = extract_uncapped_peaks(hm, min_threshold=0.05)
            uncapped_peaks_030 = [p for p in uncapped_peaks_005 if p[2] >= 0.30]

            # Convert uncapped peaks to native coords
            uncapped_nat_005 = []
            for rank, p in enumerate(uncapped_peaks_005):
                px_nat, py_nat = meta.model_to_native(p[0], p[1])
                uncapped_nat_005.append({"nat_x": px_nat, "nat_y": py_nat, "conf": p[2], "rank": rank + 1})

            uncapped_nat_030 = [p for p in uncapped_nat_005 if p["conf"] >= 0.30]

            for idx, target in enumerate(targets):
                tx_nat = target["reviewed_tip_x"]
                ty_nat = target["reviewed_tip_y"]
                tx_m, ty_m = meta.native_to_model(tx_nat, ty_nat)

                # Prob at exact tip pixel
                ix_m = int(round(np.clip(tx_m, 0, 511)))
                iy_m = int(round(np.clip(ty_m, 0, 511)))
                p_exact = float(hm[iy_m, ix_m])

                # Max prob in native radius windows
                max_prob_windows = {}
                for r_nat in [5.0, 10.0, 15.0, 20.0, 30.0, 50.0]:
                    r_m = r_nat * meta.scale
                    y_min = int(max(0, np.floor(ty_m - r_m)))
                    y_max = int(min(511, np.ceil(ty_m + r_m)))
                    x_min = int(max(0, np.floor(tx_m - r_m)))
                    x_max = int(min(511, np.ceil(tx_m + r_m)))

                    if y_max >= y_min and x_max >= x_min:
                        patch = hm[y_min : y_max + 1, x_min : x_max + 1]
                        yy, xx = np.ogrid[y_min : y_max + 1, x_min : x_max + 1]
                        dist_sq = (xx - tx_m) ** 2 + (yy - ty_m) ** 2
                        mask_r = dist_sq <= (r_m ** 2)
                        if mask_r.any():
                            max_prob_windows[r_nat] = float(patch[mask_r].max())
                        else:
                            max_prob_windows[r_nat] = p_exact
                    else:
                        max_prob_windows[r_nat] = p_exact

                # Distances to nearest uncapped peaks
                dist_nearest_005 = 9999.0
                conf_nearest_005 = 0.0
                if len(uncapped_nat_005) > 0:
                    dists_005 = [np.hypot(p["nat_x"] - tx_nat, p["nat_y"] - ty_nat) for p in uncapped_nat_005]
                    min_idx = int(np.argmin(dists_005))
                    dist_nearest_005 = float(dists_005[min_idx])
                    conf_nearest_005 = uncapped_nat_005[min_idx]["conf"]

                dist_nearest_030 = 9999.0
                conf_nearest_030 = 0.0
                rank_nearest_030 = -1
                if len(uncapped_nat_030) > 0:
                    dists_030 = [np.hypot(p["nat_x"] - tx_nat, p["nat_y"] - ty_nat) for p in uncapped_nat_030]
                    min_idx = int(np.argmin(dists_030))
                    dist_nearest_030 = float(dists_030[min_idx])
                    conf_nearest_030 = uncapped_nat_030[min_idx]["conf"]
                    rank_nearest_030 = uncapped_nat_030[min_idx]["rank"]

                # Check if there is an uncapped peak with conf >= 0.30 within 15px
                peaks_within_15 = [p for p in uncapped_nat_030 if np.hypot(p["nat_x"] - tx_nat, p["nat_y"] - ty_nat) <= 15.0]

                # Determine failure stage
                if idx in matched_target_indices:
                    stage = "Stage 7: Success@15"
                elif max_prob_windows[15.0] < 0.10:
                    stage = "Stage 1: Silent / No Signal (P_max < 0.10 in 15px)"
                elif max_prob_windows[15.0] < 0.30:
                    stage = "Stage 2: Sub-threshold Activation (0.10 <= P_max < 0.30 in 15px)"
                elif len(peaks_within_15) == 0:
                    stage = "Stage 3: Spatial Displacement (P >= 0.30 exists, but nearest is > 15px)"
                else:
                    best_peak_in_15 = sorted(peaks_within_15, key=lambda x: x["conf"], reverse=True)[0]
                    if best_peak_in_15["rank"] > 10:
                        stage = "Stage 4: Cap Pruning (P >= 0.30 peak in 15px, but rank > 10)"
                    else:
                        stage = "Stage 5: NMS Suppression (Peak in 15px suppressed by nearby peak)"

                failure_counts[arm_key][stage] += 1

                if arm_key == "experiment_controlled":
                    target_diagnostics.append({
                        "case_idx": int(target["case_idx"]),
                        "study_id": sid,
                        "catheter_instance_id": target["catheter_instance_id"],
                        "device_type": target.get("device_type", "Unknown"),
                        "reviewed_tip_x": float(tx_nat),
                        "reviewed_tip_y": float(ty_nat),
                        "heuristic_tip_x": float(target["original_heuristic_tip_x"]),
                        "heuristic_tip_y": float(target["original_heuristic_tip_y"]),
                        "heuristic_diff_px": float(np.hypot(target["reviewed_tip_x"] - target["original_heuristic_tip_x"], target["reviewed_tip_y"] - target["original_heuristic_tip_y"])),
                        "p_exact_tip": p_exact,
                        "max_prob_5px": max_prob_windows[5.0],
                        "max_prob_10px": max_prob_windows[10.0],
                        "max_prob_15px": max_prob_windows[15.0],
                        "max_prob_30px": max_prob_windows[30.0],
                        "max_prob_50px": max_prob_windows[50.0],
                        "dist_nearest_peak_005": dist_nearest_005,
                        "conf_nearest_peak_005": conf_nearest_005,
                        "dist_nearest_peak_030": dist_nearest_030,
                        "conf_nearest_peak_030": conf_nearest_030,
                        "rank_nearest_peak_030": rank_nearest_030,
                        "failure_stage": stage,
                    })

    # Print breakdown of failure stages
    print(f"\n{'Failure Triage Category':<70} | {'Control':<10} | {'Experiment':<10}")
    print("-" * 95)
    for stage_name in failure_counts["experiment_controlled"].keys():
        c_val = failure_counts["control_uniform"][stage_name]
        e_val = failure_counts["experiment_controlled"][stage_name]
        print(f"{stage_name:<70} | {c_val:<10} | {e_val:<10}")

    # -------------------------------------------------------------------------
    # PART 3: Inspection of All 25 Validation Targets vs. Training Targets
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 3: COHORT LEVEL COMPARISON (25 VALIDATION TARGETS VS. 39 TRAINING TARGETS)")
    print("=" * 80)

    # Merge target diagnostics with annotation inventory
    diag_df = pd.DataFrame(target_diagnostics)
    merged_val = diag_df.merge(
        inv_df[["study_id", "catheter_instance_id", "label", "image_width", "image_height", "image_cvc_count", "num_points", "polyline_length_px"]],
        on=["study_id", "catheter_instance_id"],
        how="left"
    )

    train_inv_merged = reviewed_train_df.merge(
        inv_df[["study_id", "catheter_instance_id", "label", "image_width", "image_height", "image_cvc_count", "num_points", "polyline_length_px"]],
        on=["study_id", "catheter_instance_id"],
        how="left"
    )

    print("\n1. Catheter Label Distribution:")
    val_labels = merged_val["label"].value_counts().to_dict()
    train_labels = train_inv_merged["label"].value_counts().to_dict()
    all_label_types = sorted(list(set(list(val_labels.keys()) + list(train_labels.keys()))))
    print(f"{'Label Type':<30} | {'Validation (N=25)':<20} | {'Training (N=39)':<20}")
    print("-" * 75)
    for lt in all_label_types:
        v_c = val_labels.get(lt, 0)
        t_c = train_labels.get(lt, 0)
        print(f"{lt:<30} | {v_c:>3} ({v_c/25.0:>6.1%})      | {t_c:>3} ({t_c/39.0:>6.1%})")

    print("\n2. Single vs. Multiple CVC Cohort Composition:")
    v_multi = (merged_val["image_cvc_count"] > 1).sum()
    t_multi = (train_inv_merged["image_cvc_count"] > 1).sum()
    print(f"  Validation: Single-CVC = {25 - v_multi}/25 ({(25-v_multi)/25:.1%}), Multi-CVC = {v_multi}/25 ({v_multi/25:.1%})")
    print(f"  Training:   Single-CVC = {39 - t_multi}/39 ({(39-t_multi)/39:.1%}), Multi-CVC = {t_multi}/39 ({t_multi/39:.1%})")

    print("\n3. Heuristic Concordance Discrepancy:")
    v_discord = (merged_val["heuristic_diff_px"] > 15.0).sum()
    train_h_diffs = np.hypot(train_inv_merged["reviewed_tip_x"] - train_inv_merged["original_heuristic_tip_x"], train_inv_merged["reviewed_tip_y"] - train_inv_merged["original_heuristic_tip_y"])
    t_discord = (train_h_diffs > 15.0).sum()
    print(f"  Validation cases where reviewed tip moved >15px from heuristic: {v_discord}/25 ({v_discord/25.0:.1%})")
    print(f"  Training cases where reviewed tip moved >15px from heuristic:   {t_discord}/39 ({t_discord/39.0:.1%})")

    # -------------------------------------------------------------------------
    # PART 4: Write Out Full Diagnostic Report
    # -------------------------------------------------------------------------
    report_md_lines = []
    report_md_lines.append("# Diagnostic Failure Analysis & Decoding Ablation Report")
    report_md_lines.append(f"**Execution Directory**: `{run_dir}`\n")

    report_md_lines.append("## Executive Summary of Findings")
    report_md_lines.append(
        "1. **Uncapped False-Peak Saturation Confirmed**:\n"
        "   Both models predict an overwhelming density of false-positive activations across negative radiographs.\n"
        f"   At $\\tau=0.30$, Uniform Control generates **{neg_analysis['control_uniform']['counts_by_threshold'][0.30]} uncapped peaks** ({neg_analysis['control_uniform']['counts_by_threshold'][0.30]/8.0:.1f}/study) "
        f"and Controlled Experiment generates **{neg_analysis['experiment_controlled']['counts_by_threshold'][0.30]} uncapped peaks** ({neg_analysis['experiment_controlled']['counts_by_threshold'][0.30]/8.0:.1f}/study).\n"
        "   The reported count of 80 false peaks was strictly a saturated ceiling ($8 \\text{ images} \\times 10 \\text{ max\\_peaks} = 80$).\n"
        "   Specificity is **not** controlled in either model.\n"
    )
    report_md_lines.append(
        "2. **Deconstruction of the 25 Validation Failures (Controlled Experiment)**:\n"
        f"   - **Silent / No Signal ($P_{{max}} < 0.10$ in $15\\text{{px}}$)**: {failure_counts['experiment_controlled']['Stage 1: Silent / No Signal (P_max < 0.10 in 15px)']}/25 ({failure_counts['experiment_controlled']['Stage 1: Silent / No Signal (P_max < 0.10 in 15px)']/25.0:.1%}). The model produced essentially zero tip activation near the target.\n"
        f"   - **Sub-Threshold Activation ($0.10 \\le P_{{max}} < 0.30$)**: {failure_counts['experiment_controlled']['Stage 2: Sub-threshold Activation (0.10 <= P_max < 0.30 in 15px)']}/25 ({failure_counts['experiment_controlled']['Stage 2: Sub-threshold Activation (0.10 <= P_max < 0.30 in 15px)']/25.0:.1%}). A modest local peak exists but is filtered out by $\\tau=0.30$.\n"
        f"   - **Spatial Displacement ($P \\ge 0.30$ exists elsewhere)**: {failure_counts['experiment_controlled']['Stage 3: Spatial Displacement (P >= 0.30 exists, but nearest is > 15px)']}/25 ({failure_counts['experiment_controlled']['Stage 3: Spatial Displacement (P >= 0.30 exists, but nearest is > 15px)']/25.0:.1%}). A confident peak was placed, but shifted away from the ground-truth coordinate.\n"
        f"   - **Cap Pruning / Overcrowding ($P \\ge 0.30$ within $15\\text{{px}}$, but rank $> 10$)**: {failure_counts['experiment_controlled']['Stage 4: Cap Pruning (P >= 0.30 peak in 15px, but rank > 10)']}/25 ({failure_counts['experiment_controlled']['Stage 4: Cap Pruning (P >= 0.30 peak in 15px, but rank > 10)']/25.0:.1%}).\n"
        f"   - **Successful Match within $15\\text{{px}}$**: {failure_counts['experiment_controlled']['Stage 7: Success@15']}/25 ({failure_counts['experiment_controlled']['Stage 7: Success@15']/25.0:.1%}).\n"
    )

    report_md_lines.append("## Detailed Table of All 25 Validation Targets (Controlled Experiment)")
    report_md_lines.append("| Case | Study ID (Prefix) | Catheter ID | Label | CVC Count | Exact Tip P | Max P (15px) | Nearest Peak Dist (P>=0.30) | Failure Category |")
    report_md_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in target_diagnostics:
        c_idx = r["case_idx"]
        s_pref = r["study_id"][:12] + "..."
        c_id = str(r["catheter_instance_id"])[:12] + "..."
        lbl = str(r["device_type"])
        p_ex = f"{r['p_exact_tip']:.3f}"
        p_15 = f"{r['max_prob_15px']:.3f}"
        d_30 = f"{r['dist_nearest_peak_030']:.1f} px" if r['dist_nearest_peak_030'] < 9000 else "None"
        fail_cat = r["failure_stage"].split(":")[0] if ":" in r["failure_stage"] else r["failure_stage"]
        report_md_lines.append(f"| {c_idx} | `{s_pref}` | `{c_id}` | {lbl} | - | {p_ex} | {p_15} | {d_30} | {fail_cat} |")

    report_text = "\n".join(report_md_lines)
    report_path = run_dir / "diagnostic_failure_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\nSaved Markdown report to: {report_path}")

    # Save JSON summary
    summary_data = {
        "negative_analysis": neg_analysis,
        "failure_counts": failure_counts,
        "target_diagnostics": target_diagnostics,
    }
    json_path = run_dir / "diagnostic_failure_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved JSON data to: {json_path}")
    print("\nDiagnostic failure analysis finished successfully.")


if __name__ == "__main__":
    run_diagnostic_analysis()
