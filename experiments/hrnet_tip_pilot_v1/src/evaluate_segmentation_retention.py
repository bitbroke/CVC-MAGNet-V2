"""Evaluation of segmentation retention on dev_pilot_val for T1 Joint HRNet.

Verifies Section 12 & 13 progression rule:
"Segmentation Dice and coverage: Detect damage to the existing task"
"Advance T1 when it does not materially damage catheter segmentation"
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

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


def evaluate_segmentation_on_dev_val(
    model: nn.Module,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    threshold: float = 0.50,
) -> Dict[str, Any]:
    transform = LetterboxTransform(target_size=(768, 768), clahe_enabled=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    inv_df = pd.read_parquet(DATA_DIR / "annotation_inventory.parquet")
    study_df = pd.read_parquet(DATA_DIR / "study_inventory.parquet")
    with open(DATA_DIR / "splits.json", "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    val_study_ids = splits_data["partitions"]["dev_pilot_val"]["study_ids"]

    polylines_by_study: Dict[str, List[List[List[float]]]] = {}
    for _, row in inv_df[(inv_df["device_type"] == "CVC") & (inv_df["study_id"].isin(val_study_ids))].iterrows():
        sid = str(row["study_id"])
        p_raw = row["polyline"]
        poly = json.loads(p_raw) if isinstance(p_raw, str) else p_raw
        polylines_by_study.setdefault(sid, []).append(poly)

    model.eval()
    macro_dices = []
    coverage_10px_list = []
    missed_catheter_count = 0
    total_catheters = 0
    total_inter = 0
    total_pred = 0
    total_gt = 0
    neg_fp_pixels = 0
    neg_valid_pixels = 0
    num_pos = 0
    num_neg = 0

    print(f"Starting segmentation retention evaluation on {len(val_study_ids)} dev_pilot_val studies...")
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

            native_gt = rasterize_study_polylines(polylines, (orig_h, orig_w), thickness=10) if is_pos else np.zeros((orig_h, orig_w), dtype=np.uint8)

            lb_img, meta = transform.apply_image(im_np)
            raw_t = torch.from_numpy(lb_img).unsqueeze(0).unsqueeze(0).to(device).float() / 255.0
            inp = (raw_t.repeat(1, 3, 1, 1) - mean) / std

            out = model(inp)
            logits = out["seg_logits"] if isinstance(out, dict) else out
            prob_map = torch.sigmoid(logits.float()).squeeze().cpu().numpy().astype(np.float32)

            valid_crop = prob_map[meta.pad_top : meta.pad_top + meta.new_h, meta.pad_left : meta.pad_left + meta.new_w]
            prob_native = cv2.resize(valid_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            bin_native = (prob_native >= threshold).astype(np.uint8)

            if is_pos:
                num_pos += 1
                gt_fg = native_gt > 0
                pred_fg = bin_native > 0
                inter = int(np.logical_and(gt_fg, pred_fg).sum())
                pred_c = int(pred_fg.sum())
                gt_c = int(gt_fg.sum())

                dice = (2.0 * inter) / (pred_c + gt_c + 1e-6)
                macro_dices.append(float(dice))
                total_inter += inter
                total_pred += pred_c
                total_gt += gt_c

                margin = 15
                pred_area = pred_c
                for poly in polylines:
                    pts = sample_polyline_points(poly, step_px=2.0)
                    in_bounds = (pts[:, 0] >= 0) & (pts[:, 0] < orig_w) & (pts[:, 1] >= 0) & (pts[:, 1] < orig_h)
                    valid_pts = pts[in_bounds]
                    total_catheters += 1
                    if len(valid_pts) == 0 or pred_area == 0:
                        coverage_10px_list.append(0.0)
                        missed_catheter_count += 1
                        continue

                    min_x = max(0, int(np.floor(np.min(valid_pts[:, 0]))) - margin)
                    max_x = min(orig_w, int(np.ceil(np.max(valid_pts[:, 0]))) + margin)
                    min_y = max(0, int(np.floor(np.min(valid_pts[:, 1]))) - margin)
                    max_y = min(orig_h, int(np.ceil(np.max(valid_pts[:, 1]))) + margin)

                    sub_pred = pred_fg[min_y:max_y, min_x:max_x]
                    sub_y, sub_x = np.nonzero(sub_pred)
                    if len(sub_x) == 0:
                        cov = 0.0
                    else:
                        fg_sub_pts = np.column_stack([sub_x + min_x, sub_y + min_y])
                        tree = cKDTree(fg_sub_pts)
                        dists, _ = tree.query(valid_pts, k=1)
                        cov = float(np.mean(dists <= 10.0))

                    coverage_10px_list.append(cov)
                    if cov < 0.30:
                        missed_catheter_count += 1
            else:
                num_neg += 1
                neg_fp_pixels += int(bin_native.sum())
                neg_valid_pixels += orig_h * orig_w

            if (i + 1) % 50 == 0 or (i + 1) == len(val_study_ids):
                print(f"Evaluated {i + 1}/{len(val_study_ids)} studies...")

    macro_dice = float(np.mean(macro_dices)) if macro_dices else 0.0
    mean_coverage = float(np.mean(coverage_10px_list)) if coverage_10px_list else 0.0
    precision = float(total_inter / (total_pred + 1e-6))
    recall = float(total_inter / (total_gt + 1e-6))
    missed_rate = float((missed_catheter_count / total_catheters) * 100.0) if total_catheters > 0 else 0.0
    neg_fp_pct = float((neg_fp_pixels / (neg_valid_pixels + 1e-6)) * 100.0)

    results = {
        "num_positive_studies": num_pos,
        "num_negative_studies": num_neg,
        "macro_dice": macro_dice,
        "centerline_coverage_10px": mean_coverage,
        "precision": precision,
        "recall": recall,
        "missed_catheter_rate_pct": missed_rate,
        "total_catheters_evaluated": total_catheters,
        "negative_fp_area_pct": neg_fp_pct,
    }
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating segmentation retention on {device}...")

    # Load frozen reference HRNet baseline metrics for comparison
    frozen_baseline = {
        "macro_dice": 0.473204,
        "centerline_coverage_10px": 0.871774,
        "precision": 0.343782,
        "recall": 0.836118,
        "missed_catheter_rate_pct": 3.629,
        "negative_fp_area_pct": 0.042,
    }

    ckpt_path = EXP_ROOT / "runs" / "seed_42" / "t1_joint_hrnet" / "checkpoints" / "final_model.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = HRNetJointTipNet(pretrained=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)

    t1_final_metrics = evaluate_segmentation_on_dev_val(model, device=device)

    print("\n--- SEGMENTATION RETENTION SUMMARY ---")
    print(f"Frozen HRNet Seed 42 Baseline: Macro Dice = {frozen_baseline['macro_dice']:.4f}, Coverage@10px = {frozen_baseline['centerline_coverage_10px']:.4f}, Recall = {frozen_baseline['recall']:.4f}")
    print(f"T1 Joint HRNet (Update 1000):   Macro Dice = {t1_final_metrics['macro_dice']:.4f}, Coverage@10px = {t1_final_metrics['centerline_coverage_10px']:.4f}, Recall = {t1_final_metrics['recall']:.4f}")

    dice_diff = t1_final_metrics["macro_dice"] - frozen_baseline["macro_dice"]
    cov_diff = t1_final_metrics["centerline_coverage_10px"] - frozen_baseline["centerline_coverage_10px"]
    print(f"Dice difference: {dice_diff:+.4f} ({dice_diff / frozen_baseline['macro_dice'] * 100:+.2f}%)")
    print(f"Coverage difference: {cov_diff:+.4f} ({cov_diff / frozen_baseline['centerline_coverage_10px'] * 100:+.2f}%)")

    damage_detected = (dice_diff < -0.05) or (cov_diff < -0.05)
    print(f"Material segmentation damage (>5% drop): {'YES (DAMAGE DETECTED)' if damage_detected else 'NO (SEGMENTATION PRESERVED)'}")

    report = {
        "frozen_baseline_seed_42": frozen_baseline,
        "t1_joint_hrnet_update_1000": t1_final_metrics,
        "dice_difference": dice_diff,
        "coverage_difference": cov_diff,
        "segmentation_damaged": damage_detected,
    }

    out_file = EXP_ROOT / "reports" / "t1_segmentation_retention_seed_42.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved retention report to {out_file}")


if __name__ == "__main__":
    main()
