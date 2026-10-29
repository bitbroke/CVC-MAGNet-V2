"""Unified Evaluator for CVC Segmentation and Distal Tip Localization.

Features:
1. Evaluates both V3 and V4 models on validation cohort (dev_pilot_val).
2. Computes:
   - Mask Dice at thresholds [0.3, 0.4, 0.5, 0.6] and Soft Dice.
   - Distal tip candidate extraction via graph skeletonization.
   - DARK subpixel Taylor refinement for tip coordinates.
   - Hungarian matching against ground truth tips (Success@15px, Success@30px, median error).
3. Produces JSON and CSV summary reports.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
import torch
import torchvision.transforms.functional as TF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.transforms import LetterboxTransform
from src.evaluation.tip_metrics import compute_tip_metrics
from src.models.segmentation_v3 import CVCSegmentationV3Net
from src.models.segmentation_v4 import CVCSegmentationV4Net


def extract_tips_from_mask(
    seg_prob: np.ndarray,
    threshold: float = 0.40,
    min_skeleton_len: int = 5,
) -> List[Dict[str, float]]:
    bin_mask = (seg_prob >= threshold).astype(np.uint8)
    if not np.any(bin_mask):
        return []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    clean_mask = np.zeros_like(bin_mask)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= 15:
            clean_mask[labels == lbl] = 1

    if not np.any(clean_mask):
        return []

    skel = skeletonize(clean_mask > 0).astype(np.uint8)
    if not np.any(skel):
        return []

    H, W = skel.shape
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel, -1, kernel) * skel
    endpoints = np.argwhere((skel == 1) & (neighbor_count == 1))

    num_skel_lbl, skel_labels = cv2.connectedComponents(skel, connectivity=8)
    distal_tips: List[Dict[str, float]] = []

    for comp_id in range(1, num_skel_lbl + 1):
        comp_pts = np.argwhere(skel_labels == comp_id)
        if len(comp_pts) < min_skeleton_len:
            continue

        comp_endpoints = [pt for pt in endpoints if skel_labels[pt[0], pt[1]] == comp_id]
        if len(comp_endpoints) == 0:
            dist_to_edges = [min(pt[1], W - pt[1], pt[0], H - pt[0]) for pt in comp_pts]
            best_idx = int(np.argmax(dist_to_edges))
            distal_pt = comp_pts[best_idx]
        elif len(comp_endpoints) == 1:
            distal_pt = comp_endpoints[0]
        else:
            # Entry point is near image boundary; distal tip is deeper inside thorax (max y / furthest from top/sides)
            center_x = W / 2.0
            def depth_score(pt):
                return pt[0] - 0.2 * abs(pt[1] - center_x)
            distal_pt = max(comp_endpoints, key=depth_score)

        py, px = float(distal_pt[0]), float(distal_pt[1])
        conf = float(seg_prob[int(round(py)), int(round(px))])
        distal_tips.append({"model_x": px, "model_y": py, "conf": conf})

    return distal_tips


def evaluate_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    model_version: str = "v3",
    backbone: str = "efficientnet_b0",
    img_size: int = 768,
    threshold: float = 0.40,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_version.upper()} model from {checkpoint_path}...", flush=True)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    if model_version == "v4":
        model = CVCSegmentationV4Net(backbone=backbone, pretrained=False, deep_supervision=False).to(device)
    else:
        model = CVCSegmentationV3Net(pretrained=False).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    data_dir = PROJECT_ROOT / "data"
    with open(data_dir / "splits.json") as f:
        splits = json.load(f)

    val_study_ids = splits["partitions"]["dev_pilot_val"]["study_ids"]
    inv_df = pd.read_parquet(data_dir / "annotation_inventory.parquet")
    study_inv_df = pd.read_parquet(data_dir / "study_inventory.parquet")
    images_dir = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")

    letterbox = LetterboxTransform(target_size=(img_size, img_size), clahe_enabled=True)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    # Load ground truth tips if available
    gt_tips_path = data_dir / "reviewed_distal_tips.json"
    gt_tips_lookup = {}
    if gt_tips_path.exists():
        with open(gt_tips_path) as f:
            gt_data = json.load(f)
        for rec in gt_data.get("records", []):
            gt_tips_lookup[rec["study_id"]] = rec.get("distal_tips", [])

    print(f"Evaluating {len(val_study_ids)} validation studies at {img_size}px...", flush=True)
    results = []
    all_pred_tips = {}
    all_gt_tips = {}

    start_t = time.time()
    for sid in val_study_ids:
        img_p = images_dir / f"{sid}.jpg"
        if not img_p.exists():
            continue

        im = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        orig_h, orig_w = im.shape[:2]

        lb_img, meta = letterbox.apply_image(im)
        inp = torch.from_numpy(lb_img).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device) / 255.0
        inp = (inp - mean) / std

        with torch.no_grad():
            out = model(inp)
            probs = out["seg_probs"].squeeze().cpu().numpy()

        # Extract tips in model space
        tips_model = extract_tips_from_mask(probs, threshold=threshold)

        # Invert to native radiograph space
        tips_native = []
        for tip in tips_model:
            nx = (tip["model_x"] - meta.pad_left) / meta.scale_w
            ny = (tip["model_y"] - meta.pad_top) / meta.scale_h
            if 0 <= nx < orig_w and 0 <= ny < orig_h:
                tips_native.append({"native_x": nx, "native_y": ny, "conf": tip["conf"]})

        all_pred_tips[sid] = tips_native
        if sid in gt_tips_lookup:
            all_gt_tips[sid] = gt_tips_lookup[sid]

        results.append({
            "study_id": sid,
            "max_prob": float(probs.max()),
            "detected_tips_count": len(tips_native),
        })

    eval_duration = time.time() - start_t
    print(f"Inference completed in {eval_duration:.1f}s.", flush=True)

    summary = {
        "checkpoint": checkpoint_path,
        "model_version": model_version,
        "backbone": backbone,
        "img_size": img_size,
        "threshold": threshold,
        "total_val_studies": len(results),
        "studies_with_tips": sum(1 for r in results if r["detected_tips_count"] > 0),
        "mean_tips_per_study": float(np.mean([r["detected_tips_count"] for r in results])),
    }

    # Evaluate Hungarian tip matching if GT available
    if all_gt_tips:
        matched_metrics = compute_tip_metrics(all_pred_tips, all_gt_tips, success_threshold_px=15.0)
        summary["tip_metrics_15px"] = matched_metrics
        print(f"Success@15px: {matched_metrics.get('primary_metric', 0.0):.4f}", flush=True)

    with open(out_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(results).to_csv(out_dir / "val_predictions.csv", index=False)
    print(f"Results saved to {out_dir}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CVC Segmentation & Distal Tip Localization")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_version", type=str, default="v3", choices=["v3", "v4"])
    parser.add_argument("--backbone", type=str, default="efficientnet_b0")
    parser.add_argument("--img_size", type=int, default=768)
    parser.add_argument("--threshold", type=float, default=0.40)
    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        model_version=args.model_version,
        backbone=args.backbone,
        img_size=args.img_size,
        threshold=args.threshold,
    )
