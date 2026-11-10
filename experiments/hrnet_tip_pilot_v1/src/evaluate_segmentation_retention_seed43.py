"""
Evaluate segmentation retention for HRNet Seed 43 on dev_pilot_val (193 studies).
Compares frozen reference HRNet Seed 43 baseline vs T1 Joint HRNet Seed 43.
Performs paired per-study analysis to detect any lost catheter segments.
"""
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree
import torch
import torch.nn as nn

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXP_ROOT = PROJECT_ROOT / "experiments" / "hrnet_tip_pilot_v1"
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")

from src.data.rasterize import rasterize_study_polylines
from src.data.transforms import LetterboxTransform
from experiments.hrnet_tip_pilot_v1.frozen_reference.code.model_registry import HRNetW18Net
from experiments.hrnet_tip_pilot_v1.src.models.hrnet_joint_tip import HRNetJointTipNet


def sample_polyline_points(polyline: List[List[float]], step_px: float = 2.0) -> np.ndarray:
    pts = np.array(polyline, dtype=np.float32)
    if len(pts) < 2:
        return pts
    sampled = []
    for i in range(len(pts) - 1):
        p1, p2 = pts[i], pts[i + 1]
        dist = np.linalg.norm(p2 - p1)
        if dist < 1e-4:
            continue
        num_steps = max(1, int(np.ceil(dist / step_px)))
        alphas = np.linspace(0.0, 1.0, num_steps, endpoint=(i == len(pts) - 2))
        for a in alphas:
            sampled.append(p1 + a * (p2 - p1))
    return np.array(sampled, dtype=np.float32) if sampled else pts


def evaluate_both_models_on_dev_val(
    baseline_model: nn.Module,
    t1_model: nn.Module,
    device: torch.device,
    threshold: float = 0.50,
) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]]:
    transform = LetterboxTransform(target_size=(768, 768), clahe_enabled=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    inv_df = pd.read_parquet(DATA_DIR / "annotation_inventory.parquet")
    with open(DATA_DIR / "splits.json", "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    val_study_ids = splits_data["partitions"]["dev_pilot_val"]["study_ids"]

    polylines_by_study: Dict[str, List[List[List[float]]]] = {}
    for _, row in inv_df[(inv_df["device_type"] == "CVC") & (inv_df["study_id"].isin(val_study_ids))].iterrows():
        sid = str(row["study_id"])
        p_raw = row["polyline"]
        poly = json.loads(p_raw) if isinstance(p_raw, str) else p_raw
        polylines_by_study.setdefault(sid, []).append(poly)

    baseline_model.eval()
    t1_model.eval()

    paired_records = []
    lost_catheters = []

    b_macro_dices, t_macro_dices = [], []
    b_coverage_list, t_coverage_list = [], []
    b_missed_catheter_count, t_missed_catheter_count = 0, 0
    total_catheters = 0

    b_total_inter, b_total_pred, b_total_gt = 0, 0, 0
    t_total_inter, t_total_pred, t_total_gt = 0, 0, 0

    b_neg_fp_pixels, t_neg_fp_pixels, neg_valid_pixels = 0, 0, 0
    num_pos = 0
    num_neg = 0

    print(f"Starting paired segmentation evaluation on {len(val_study_ids)} dev_pilot_val studies...")
    with torch.no_grad():
        for i, sid in enumerate(val_study_ids):
            img_path = IMAGES_DIR / f"{sid}.jpg"
            if not img_path.exists():
                continue
            pil_im = Image.open(img_path).convert("L")
            im_np = np.array(pil_im)
            orig_h, orig_w = im_np.shape

            polylines = polylines_by_study.get(sid, [])
            is_pos = len(polylines) > 0

            native_gt = (
                rasterize_study_polylines(polylines, (orig_h, orig_w), thickness=10)
                if is_pos
                else np.zeros((orig_h, orig_w), dtype=np.uint8)
            )

            lb_img, meta = transform.apply_image(im_np)
            raw_t = torch.from_numpy(lb_img).unsqueeze(0).unsqueeze(0).to(device).float() / 255.0
            inp = (raw_t.repeat(1, 3, 1, 1) - mean) / std

            # 1. Baseline forward pass
            b_out = baseline_model(inp)
            b_logits = b_out["seg_logits"] if isinstance(b_out, dict) else b_out
            b_prob_map = torch.sigmoid(b_logits.float()).squeeze().cpu().numpy().astype(np.float32)
            b_valid_crop = b_prob_map[meta.pad_top : meta.pad_top + meta.new_h, meta.pad_left : meta.pad_left + meta.new_w]
            b_prob_native = cv2.resize(b_valid_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            b_bin_native = (b_prob_native >= threshold).astype(np.uint8)

            # 2. T1 forward pass
            t_out = t1_model(inp)
            t_logits = t_out["seg_logits"] if isinstance(t_out, dict) else t_out
            t_prob_map = torch.sigmoid(t_logits.float()).squeeze().cpu().numpy().astype(np.float32)
            t_valid_crop = t_prob_map[meta.pad_top : meta.pad_top + meta.new_h, meta.pad_left : meta.pad_left + meta.new_w]
            t_prob_native = cv2.resize(t_valid_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            t_bin_native = (t_prob_native >= threshold).astype(np.uint8)

            if is_pos:
                num_pos += 1
                gt_fg = native_gt > 0
                b_pred_fg = b_bin_native > 0
                t_pred_fg = t_bin_native > 0

                b_inter = int(np.logical_and(gt_fg, b_pred_fg).sum())
                b_pred_c = int(b_pred_fg.sum())
                gt_c = int(gt_fg.sum())
                b_dice = (2.0 * b_inter) / (b_pred_c + gt_c + 1e-6)
                b_macro_dices.append(float(b_dice))
                b_total_inter += b_inter
                b_total_pred += b_pred_c
                b_total_gt += gt_c

                t_inter = int(np.logical_and(gt_fg, t_pred_fg).sum())
                t_pred_c = int(t_pred_fg.sum())
                t_dice = (2.0 * t_inter) / (t_pred_c + gt_c + 1e-6)
                t_macro_dices.append(float(t_dice))
                t_total_inter += t_inter
                t_total_pred += t_pred_c
                t_total_gt += gt_c

                # Centerline coverage
                margin = 15
                study_b_covs, study_t_covs = [], []

                for c_idx, poly in enumerate(polylines):
                    pts = sample_polyline_points(poly, step_px=2.0)
                    in_bounds = (pts[:, 0] >= 0) & (pts[:, 0] < orig_w) & (pts[:, 1] >= 0) & (pts[:, 1] < orig_h)
                    valid_pts = pts[in_bounds]
                    total_catheters += 1
                    if len(valid_pts) == 0:
                        b_coverage_list.append(0.0)
                        t_coverage_list.append(0.0)
                        b_missed_catheter_count += 1
                        t_missed_catheter_count += 1
                        continue

                    min_x = max(0, int(np.floor(np.min(valid_pts[:, 0]))) - margin)
                    max_x = min(orig_w, int(np.ceil(np.max(valid_pts[:, 0]))) + margin)
                    min_y = max(0, int(np.floor(np.min(valid_pts[:, 1]))) - margin)
                    max_y = min(orig_h, int(np.ceil(np.max(valid_pts[:, 1]))) + margin)

                    # Baseline coverage for this catheter
                    sub_b_pred = b_pred_fg[min_y:max_y, min_x:max_x]
                    sub_b_y, sub_b_x = np.nonzero(sub_b_pred)
                    if len(sub_b_x) == 0:
                        b_cov = 0.0
                    else:
                        fg_b_pts = np.column_stack([sub_b_x + min_x, sub_b_y + min_y])
                        tree_b = cKDTree(fg_b_pts)
                        dists_b, _ = tree_b.query(valid_pts, k=1)
                        b_cov = float(np.mean(dists_b <= 10.0))

                    # T1 coverage for this catheter
                    sub_t_pred = t_pred_fg[min_y:max_y, min_x:max_x]
                    sub_t_y, sub_t_x = np.nonzero(sub_t_pred)
                    if len(sub_t_x) == 0:
                        t_cov = 0.0
                    else:
                        fg_t_pts = np.column_stack([sub_t_x + min_x, sub_t_y + min_y])
                        tree_t = cKDTree(fg_t_pts)
                        dists_t, _ = tree_t.query(valid_pts, k=1)
                        t_cov = float(np.mean(dists_t <= 10.0))

                    b_coverage_list.append(b_cov)
                    t_coverage_list.append(t_cov)
                    study_b_covs.append(b_cov)
                    study_t_covs.append(t_cov)

                    if b_cov < 0.30:
                        b_missed_catheter_count += 1
                    if t_cov < 0.30:
                        t_missed_catheter_count += 1

                    # Check for lost catheter (baseline detected cov>=0.50, but T1 missed cov<0.50)
                    if b_cov >= 0.50 and t_cov < 0.50:
                        lost_catheters.append({
                            "study_id": sid,
                            "catheter_index": c_idx,
                            "baseline_coverage": float(b_cov),
                            "t1_coverage": float(t_cov),
                            "coverage_drop": float(b_cov - t_cov),
                            "baseline_dice": float(b_dice),
                            "t1_dice": float(t_dice),
                        })

                paired_records.append({
                    "study_id": sid,
                    "is_negative": False,
                    "num_catheters": len(polylines),
                    "baseline_dice": b_dice,
                    "t1_dice": t_dice,
                    "dice_diff": t_dice - b_dice,
                    "baseline_mean_cov": np.mean(study_b_covs) if study_b_covs else 0.0,
                    "t1_mean_cov": np.mean(study_t_covs) if study_t_covs else 0.0,
                    "cov_diff": (np.mean(study_t_covs) - np.mean(study_b_covs)) if study_b_covs else 0.0,
                })
            else:
                num_neg += 1
                b_fp = int(b_bin_native.sum())
                t_fp = int(t_bin_native.sum())
                v_pix = orig_h * orig_w
                b_neg_fp_pixels += b_fp
                t_neg_fp_pixels += t_fp
                neg_valid_pixels += v_pix

                paired_records.append({
                    "study_id": sid,
                    "is_negative": True,
                    "num_catheters": 0,
                    "baseline_dice": 1.0 if b_fp == 0 else 0.0,
                    "t1_dice": 1.0 if t_fp == 0 else 0.0,
                    "dice_diff": (1.0 if t_fp == 0 else 0.0) - (1.0 if b_fp == 0 else 0.0),
                    "baseline_mean_cov": 1.0,
                    "t1_mean_cov": 1.0,
                    "cov_diff": 0.0,
                })

            if (i + 1) % 50 == 0 or (i + 1) == len(val_study_ids):
                print(f"Evaluated {i + 1}/{len(val_study_ids)} studies...")

    b_metrics = {
        "num_positive_studies": num_pos,
        "num_negative_studies": num_neg,
        "macro_dice": float(np.mean(b_macro_dices)) if b_macro_dices else 0.0,
        "centerline_coverage_10px": float(np.mean(b_coverage_list)) if b_coverage_list else 0.0,
        "precision": float(b_total_inter / (b_total_pred + 1e-6)),
        "recall": float(b_total_inter / (b_total_gt + 1e-6)),
        "missed_catheter_count": b_missed_catheter_count,
        "total_catheters_evaluated": total_catheters,
        "missed_catheter_rate_pct": float((b_missed_catheter_count / total_catheters) * 100.0) if total_catheters > 0 else 0.0,
        "negative_fp_area_pct": float((b_neg_fp_pixels / (neg_valid_pixels + 1e-6)) * 100.0),
    }

    t_metrics = {
        "num_positive_studies": num_pos,
        "num_negative_studies": num_neg,
        "macro_dice": float(np.mean(t_macro_dices)) if t_macro_dices else 0.0,
        "centerline_coverage_10px": float(np.mean(t_coverage_list)) if t_coverage_list else 0.0,
        "precision": float(t_total_inter / (t_total_pred + 1e-6)),
        "recall": float(t_total_inter / (t_total_gt + 1e-6)),
        "missed_catheter_count": t_missed_catheter_count,
        "total_catheters_evaluated": total_catheters,
        "missed_catheter_rate_pct": float((t_missed_catheter_count / total_catheters) * 100.0) if total_catheters > 0 else 0.0,
        "negative_fp_area_pct": float((t_neg_fp_pixels / (neg_valid_pixels + 1e-6)) * 100.0),
    }

    df_paired = pd.DataFrame(paired_records)
    return b_metrics, t_metrics, df_paired, lost_catheters


def run_seed43_retention():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Seed 43 Segmentation Retention on {device}...")

    # Load frozen reference HRNet Seed 43 baseline
    b_ckpt_path = EXP_ROOT / "frozen_reference" / "checkpoints" / "hrnet_seed43_final.pt"
    print(f"Loading baseline checkpoint: {b_ckpt_path}")
    b_ckpt = torch.load(b_ckpt_path, map_location=device, weights_only=False)
    baseline_model = HRNetW18Net(pretrained=False)
    b_state = b_ckpt.get("model_state_dict", b_ckpt)
    baseline_model.load_state_dict(b_state)
    baseline_model.to(device)

    # Load T1 Joint HRNet Seed 43 model
    t_ckpt_path = EXP_ROOT / "runs" / "seed_43" / "t1_joint_hrnet" / "checkpoints" / "final_model.pt"
    print(f"Loading T1 checkpoint: {t_ckpt_path}")
    t_ckpt = torch.load(t_ckpt_path, map_location=device, weights_only=False)
    t1_model = HRNetJointTipNet(pretrained=False)
    t_state = t_ckpt.get("model_state_dict", t_ckpt)
    t1_model.load_state_dict(t_state)
    t1_model.to(device)

    b_metrics, t_metrics, df_paired, lost_caths = evaluate_both_models_on_dev_val(
        baseline_model, t1_model, device=device
    )

    dice_diff = t_metrics["macro_dice"] - b_metrics["macro_dice"]
    cov_diff = t_metrics["centerline_coverage_10px"] - b_metrics["centerline_coverage_10px"]

    print("\n================== SEED 43 SEGMENTATION RETENTION RESULTS ==================")
    print(f"Frozen HRNet Seed 43 Baseline: Macro Dice = {b_metrics['macro_dice']:.4f}, Coverage@10px = {b_metrics['centerline_coverage_10px']:.4f}, Recall = {b_metrics['recall']:.4f}, Missed = {b_metrics['missed_catheter_count']}/{b_metrics['total_catheters_evaluated']} ({b_metrics['missed_catheter_rate_pct']:.2f}%)")
    print(f"T1 Joint HRNet Seed 43:        Macro Dice = {t_metrics['macro_dice']:.4f}, Coverage@10px = {t_metrics['centerline_coverage_10px']:.4f}, Recall = {t_metrics['recall']:.4f}, Missed = {t_metrics['missed_catheter_count']}/{t_metrics['total_catheters_evaluated']} ({t_metrics['missed_catheter_rate_pct']:.2f}%)")
    print(f"Dice difference:     {dice_diff:+.4f} ({dice_diff / b_metrics['macro_dice'] * 100:+.2f}%)")
    print(f"Coverage difference: {cov_diff:+.4f} ({cov_diff / b_metrics['centerline_coverage_10px'] * 100:+.2f}%)")
    print(f"Total lost catheters (baseline covered >=50%, T1 missed <50%): {len(lost_caths)}")
    for lc in lost_caths:
        print(f"  Lost catheter in study {lc['study_id'][:16]}... index={lc['catheter_index']}: baseline_cov={lc['baseline_coverage']:.2f} -> t1_cov={lc['t1_coverage']:.2f}")

    # Paired distribution
    pos_paired = df_paired[~df_paired["is_negative"]]
    dice_diffs = pos_paired["dice_diff"].values
    print(f"\nPaired Dice Differences (N={len(pos_paired)}):")
    print(f"  Mean diff: {np.mean(dice_diffs):+.4f} | Median diff: {np.median(dice_diffs):+.4f}")
    print(f"  Improved studies (Dice diff > +0.01): {sum(dice_diffs > 0.01)}")
    print(f"  Degraded studies (Dice diff < -0.01): {sum(dice_diffs < -0.01)}")
    print(f"  Unchanged studies (|Dice diff| <= 0.01): {sum(np.abs(dice_diffs) <= 0.01)}")

    damage_detected = (dice_diff < -0.05) or (cov_diff < -0.05) or (len(lost_caths) > 5)

    report = {
        "seed": 43,
        "frozen_baseline_seed_43": b_metrics,
        "t1_joint_hrnet_seed_43": t_metrics,
        "dice_difference": dice_diff,
        "coverage_difference": cov_diff,
        "damage_detected_predefined_tolerance": damage_detected,
        "lost_catheters_count": len(lost_caths),
        "lost_catheters_details": lost_caths,
        "paired_dice_distribution": {
            "mean_diff": float(np.mean(dice_diffs)),
            "median_diff": float(np.median(dice_diffs)),
            "improved_studies": int(sum(dice_diffs > 0.01)),
            "degraded_studies": int(sum(dice_diffs < -0.01)),
            "unchanged_studies": int(sum(np.abs(dice_diffs) <= 0.01)),
        }
    }

    out_json = EXP_ROOT / "reports" / "t1_segmentation_retention_seed_43.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    out_csv = EXP_ROOT / "reports" / "seed_43_segmentation_paired_analysis.csv"
    df_paired.to_csv(out_csv, index=False)

    print(f"\nSaved Seed 43 retention report to: {out_json}")
    print(f"Saved Seed 43 paired CSV to: {out_csv}")


if __name__ == "__main__":
    run_seed43_retention()
