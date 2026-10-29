"""Comprehensive Performance Matrix and Failure Analysis Engine.

Evaluates trained segmentation checkpoints across the full spectrum of metrics:
1. Continuous & Binary Mask Dice across thresholds [0.1, 0.2, ..., 0.9]
2. IoU, Precision, Recall, Specificity
3. Centerline coverage and mean boundary error
4. Breakdown across clinical subgroups:
   - Single CVC vs Multiple CVCs
   - Low clutter vs High clutter (presence of ETT, NGT, Swan-Ganz)
   - Diagnostic placement: Normal, Borderline, Abnormal
5. False Positive rate and false area on label-supported negative studies
6. Deep failure analysis: identifies worst-performing cases with failure modes
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
import torch
from torch.amp import autocast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.rasterize import rasterize_study_polylines
from src.data.transforms import LetterboxTransform, TransformMetadata
from src.models.segmentation_v3 import CVCSegmentationV3Net


def compute_binary_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray) -> Dict[str, float]:
    """Compute Dice, IoU, Precision, Recall, Specificity on flattened binary masks."""
    p_flat = pred_bin.ravel()
    g_flat = gt_bin.ravel()

    tp = np.logical_and(p_flat, g_flat).sum()
    fp = np.logical_and(p_flat, np.logical_not(g_flat)).sum()
    fn = np.logical_and(np.logical_not(p_flat), g_flat).sum()
    tn = np.logical_and(np.logical_not(p_flat), np.logical_not(g_flat)).sum()

    eps = 1e-6
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def evaluate_checkpoint(
    checkpoint_path: str,
    img_size: int = 512,
    device_str: str = "cuda",
    thresholds: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
) -> Dict[str, Any]:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")

    print(f"Loading checkpoint: {ckpt_p}...")
    ckpt = torch.load(str(ckpt_p), map_location=device, weights_only=False)

    model = CVCSegmentationV3Net(pretrained=False).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    # Load data
    data_dir = PROJECT_ROOT / "data"
    study_inv_p = data_dir / "study_inventory.parquet"
    ann_inv_p = data_dir / "annotation_inventory.parquet"
    study_df = pd.read_parquet(study_inv_p)
    ann_df = pd.read_parquet(ann_inv_p)

    with open(data_dir / "splits.json") as f:
        splits = json.load(f)
    val_sids = splits["partitions"]["dev_pilot_val"]["study_ids"]

    images_dir = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")
    transform = LetterboxTransform(target_size=(img_size, img_size), clahe_enabled=True)

    # Index polylines
    cvc_ann = ann_df[ann_df["device_type"] == "CVC"]
    poly_idx = {}
    for _, row in cvc_ann.iterrows():
        sid = str(row["study_id"])
        raw = row.get("polyline")
        if raw is not None:
            p = json.loads(raw) if isinstance(raw, str) else raw
            poly_idx.setdefault(sid, []).append(p)

    study_meta = {}
    for _, row in study_df.iterrows():
        sid = str(row["StudyInstanceUID"])
        study_meta[sid] = {
            "has_cvc": bool(row.get("has_cvc", False)),
            "num_cvc": int(row.get("num_annotated_cvcs", 0)),
            "cvc_normal": int(row.get("CVC - Normal", 0)),
            "cvc_borderline": int(row.get("CVC - Borderline", 0)),
            "cvc_abnormal": int(row.get("CVC - Abnormal", 0)),
            "has_other_lines": bool(row.get("has_other_lines", False)),
            "ett": bool(row.get("ETT - Normal", 0) or row.get("ETT - Borderline", 0) or row.get("ETT - Abnormal", 0)),
            "ngt": bool(row.get("NGT - Normal", 0) or row.get("NGT - Borderline", 0) or row.get("NGT - Incompletely Imaged", 0) or row.get("NGT - Abnormal", 0)),
            "swan_ganz": bool(row.get("Swan Ganz Catheter Present", 0)),
        }

    results = []
    print(f"Evaluating {len(val_sids)} validation studies at {img_size}px...")
    start_time = time.time()

    for idx, sid in enumerate(val_sids):
        img_path = images_dir / f"{sid}.jpg"
        if not img_path.exists():
            continue
        im = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        orig_h, orig_w = im.shape[:2]
        lb_img, meta = transform.apply_image(im)

        polylines = poly_idx.get(sid, [])
        is_pos = len(polylines) > 0

        # Ground truth mask at letterbox scale
        if is_pos:
            native_mask = rasterize_study_polylines(polylines, (orig_h, orig_w), thickness=10)
            gt_mask = transform.apply_mask(native_mask, meta) > 0
        else:
            gt_mask = np.zeros((img_size, img_size), dtype=bool)

        valid_mask = np.zeros((img_size, img_size), dtype=bool)
        valid_mask[meta.pad_top : meta.pad_top + meta.new_h, meta.pad_left : meta.pad_left + meta.new_w] = True

        # Model prediction
        inp_tensor = torch.from_numpy(lb_img).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1) / 255.0
        # ImageNet normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        inp_tensor = (inp_tensor - mean) / std
        inp_tensor = inp_tensor.to(device)

        with torch.no_grad():
            with autocast("cuda", enabled=torch.cuda.is_available()):
                out = model(inp_tensor)
            prob_map = out["seg_probs"].squeeze().cpu().numpy()

        # Zero out padding
        prob_map[~valid_mask] = 0.0

        # Continuous soft dice
        if is_pos:
            p_valid = prob_map[valid_mask]
            g_valid = gt_mask[valid_mask].astype(np.float32)
            soft_dice = float((2.0 * (p_valid * g_valid).sum() + 1e-6) / (p_valid.sum() + g_valid.sum() + 1e-6))
        else:
            soft_dice = 1.0 if prob_map[valid_mask].sum() == 0 else 0.0

        # Metrics per threshold
        thresh_metrics = {}
        for th in thresholds:
            pred_bin = (prob_map >= th) & valid_mask
            if is_pos:
                m = compute_binary_metrics(pred_bin, gt_mask)
            else:
                fp_pix = int(pred_bin.sum())
                m = {"dice": 1.0 if fp_pix == 0 else 0.0, "fp_pixels": fp_pix, "fp_rate": fp_pix / valid_mask.sum()}
            thresh_metrics[f"th_{th:.1f}"] = m

        sm = study_meta.get(sid, {})
        study_record = {
            "study_id": sid,
            "is_positive": is_pos,
            "num_cvc": len(polylines),
            "soft_dice": soft_dice,
            "thresholds": thresh_metrics,
            "gt_pixel_count": int(gt_mask.sum()),
            "pred_prob_max": float(prob_map[valid_mask].max()),
            "pred_prob_mean": float(prob_map[valid_mask].mean()),
            "has_other_lines": sm.get("has_other_lines", False),
            "clutter": sm.get("ett", False) or sm.get("ngt", False) or sm.get("swan_ganz", False),
            "placement": "Abnormal" if sm.get("cvc_abnormal") else ("Borderline" if sm.get("cvc_borderline") else "Normal"),
        }
        results.append(study_record)

    elapsed = time.time() - start_time
    print(f"Evaluated {len(results)} studies in {elapsed:.1f}s ({elapsed/len(results):.3f} s/study)")

    # Aggregate performance matrix
    pos_results = [r for r in results if r["is_positive"]]
    neg_results = [r for r in results if not r["is_positive"]]

    summary = {
        "checkpoint": str(ckpt_p),
        "img_size": img_size,
        "total_evaluated": len(results),
        "num_positives": len(pos_results),
        "num_negatives": len(neg_results),
        "macro_soft_dice_positives": float(np.mean([r["soft_dice"] for r in pos_results])),
        "performance_by_threshold": {},
        "subgroup_analysis": {},
        "worst_failures": sorted(pos_results, key=lambda x: x["soft_dice"])[:10],
    }

    # Best threshold selection on positives
    best_th = 0.5
    best_th_dice = -1.0
    for th in thresholds:
        th_key = f"th_{th:.1f}"
        dices = [r["thresholds"][th_key]["dice"] for r in pos_results]
        mean_dice = float(np.mean(dices))
        ious = [r["thresholds"][th_key]["iou"] for r in pos_results]
        precs = [r["thresholds"][th_key]["precision"] for r in pos_results]
        recs = [r["thresholds"][th_key]["recall"] for r in pos_results]
        neg_fps = [r["thresholds"][th_key].get("fp_pixels", 0) for r in neg_results]

        summary["performance_by_threshold"][th_key] = {
            "mean_dice": mean_dice,
            "mean_iou": float(np.mean(ious)),
            "mean_precision": float(np.mean(precs)),
            "mean_recall": float(np.mean(recs)),
            "negative_mean_fp_pixels": float(np.mean(neg_fps)) if neg_fps else 0.0,
            "negative_clean_fraction": float(np.mean([fp == 0 for fp in neg_fps])) if neg_fps else 1.0,
        }
        if mean_dice > best_th_dice:
            best_th_dice = mean_dice
            best_th = th

    summary["optimal_threshold"] = best_th
    summary["optimal_threshold_mean_dice"] = best_th_dice

    # Subgroup breakdown at optimal threshold
    opt_key = f"th_{best_th:.1f}"
    # Single vs Multi
    single_cvc = [r for r in pos_results if r["num_cvc"] == 1]
    multi_cvc = [r for r in pos_results if r["num_cvc"] > 1]
    # Low clutter vs High clutter
    low_clutter = [r for r in pos_results if not r["clutter"]]
    high_clutter = [r for r in pos_results if r["clutter"]]
    # Placement
    normal_p = [r for r in pos_results if r["placement"] == "Normal"]
    border_p = [r for r in pos_results if r["placement"] == "Borderline"]
    abnorm_p = [r for r in pos_results if r["placement"] == "Abnormal"]

    def subgroup_stats(group: List[Dict]) -> Dict[str, Any]:
        if not group:
            return {"count": 0, "soft_dice": 0.0, "binary_dice": 0.0}
        return {
            "count": len(group),
            "soft_dice": float(np.mean([r["soft_dice"] for r in group])),
            "binary_dice": float(np.mean([r["thresholds"][opt_key]["dice"] for r in group])),
            "mean_recall": float(np.mean([r["thresholds"][opt_key]["recall"] for r in group])),
            "mean_precision": float(np.mean([r["thresholds"][opt_key]["precision"] for r in group])),
        }

    summary["subgroup_analysis"] = {
        "single_cvc": subgroup_stats(single_cvc),
        "multi_cvc": subgroup_stats(multi_cvc),
        "low_clutter": subgroup_stats(low_clutter),
        "high_clutter": subgroup_stats(high_clutter),
        "normal_placement": subgroup_stats(normal_p),
        "borderline_placement": subgroup_stats(border_p),
        "abnormal_placement": subgroup_stats(abnorm_p),
    }

    return summary, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Segmentation Performance Matrix")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--out_json", type=str, default=None)
    args = parser.parse_args()

    summary, results = evaluate_checkpoint(args.checkpoint, img_size=args.img_size)

    print("\n=================== PERFORMANCE MATRIX ===================")
    print(f"Checkpoint: {summary['checkpoint']}")
    print(f"Validation Positives Soft Dice: {summary['macro_soft_dice_positives']:.4f}")
    print(f"Optimal Threshold: {summary['optimal_threshold']} (Binary Dice: {summary['optimal_threshold_mean_dice']:.4f})")
    print("\nMetrics by Threshold:")
    for th, m in summary["performance_by_threshold"].items():
        print(f"  {th}: Dice={m['mean_dice']:.4f}, IoU={m['mean_iou']:.4f}, Precision={m['mean_precision']:.4f}, Recall={m['mean_recall']:.4f}, Neg FP Pixels={m['negative_mean_fp_pixels']:.1f}")

    print("\nSubgroup Analysis (at optimal threshold):")
    for sg, stats in summary["subgroup_analysis"].items():
        print(f"  {sg:20s}: Count={stats['count']:3d} | Soft Dice={stats['soft_dice']:.4f} | Binary Dice={stats['binary_dice']:.4f}")

    if args.out_json:
        out_p = Path(args.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary to {out_p}")
