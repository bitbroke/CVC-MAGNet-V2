"""Score and rank training-set confirmed negatives using the repaired 768 model.

This script executes Item 2 & 3 of the false-positive suppression protocol:
1. Loads the 396 confirmed-negative training images from dev_pilot_train.
2. Runs inference using the repaired 768 checkpoint (without test-set leakage).
3. Computes false foreground metrics (max probability, false pixels, false area % at tau=0.30, 0.50, 0.70).
4. Ranks all 396 images from hardest to easiest.
5. Saves ranked manifest data/hard_negatives_manifest.json and reports/hard_negatives_ranked.csv.
6. Generates visual overlay inspection panels for the top 15 worst false-positive studies.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.transforms import LetterboxTransform
from src.models.segmentation_v3 import CVCSegmentationV3Net


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def score_training_negatives(
    checkpoint_path: str = r"C:\Ankshit\CVC-MAGNet-V2\runs\s2_repaired_768\checkpoints\final_segmentation_v3.pt",
    img_size: int = 768,
    output_manifest: str = r"C:\Ankshit\CVC-MAGNet-V2\data\hard_negatives_manifest.json",
    output_csv: str = r"C:\Ankshit\CVC-MAGNet-V2\reports\hard_negatives_ranked.csv",
    overlays_dir: str = r"C:\Ankshit\CVC-MAGNet-V2\reports\overlays\hard_negatives",
    top_n_overlays: int = 15,
):
    print("=" * 60)
    print("Scoring Training-Set Confirmed Negatives (Hard Negative Mining)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # 1. Load splits and inventories
    data_dir = PROJECT_ROOT / "data"
    splits_path = data_dir / "splits.json"
    study_inv_path = data_dir / "study_inventory.parquet"

    with open(splits_path) as f:
        splits = json.load(f)

    train_study_ids = set(splits["partitions"]["dev_pilot_train"]["study_ids"])
    study_inv_df = pd.read_parquet(study_inv_path)

    # Filter strictly for confirmed training negatives
    neg_df = study_inv_df[
        (study_inv_df["StudyInstanceUID"].isin(train_study_ids)) & (study_inv_df["has_cvc"] == False)
    ].copy()

    total_negatives = len(neg_df)
    print(f"Identified {total_negatives} confirmed-negative training studies in dev_pilot_train.", flush=True)
    if total_negatives == 0:
        raise RuntimeError("No confirmed-negative training studies found in dev_pilot_train!")

    images_dir = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")

    # 2. Load model
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt_sha256 = compute_sha256(str(ckpt_path))
    print(f"Loading checkpoint: {ckpt_path.name} (SHA256: {ckpt_sha256[:16]}...)", flush=True)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    model = CVCSegmentationV3Net(pretrained=False, freeze_encoder_bn=True).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    transform = LetterboxTransform(target_size=(img_size, img_size), clahe_enabled=True)

    # 3. Score all 396 negative images
    records = []
    print(f"Beginning inference across {total_negatives} images at {img_size}x{img_size}...", flush=True)

    for i, (_, row) in enumerate(neg_df.iterrows(), 1):
        sid = str(row["StudyInstanceUID"])
        pid = str(row.get("PatientID", ""))
        has_other = bool(row.get("has_other_lines", False))

        img_path = images_dir / f"{sid}.jpg"
        if not img_path.exists():
            print(f"Warning: Missing image file {img_path}")
            continue

        im = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if im is None:
            print(f"Warning: Failed to decode image {img_path}")
            continue

        orig_h, orig_w = im.shape[:2]
        lb_img, meta = transform.apply_image(im)

        # Build valid mask (letterbox region excluding padding)
        valid_mask = np.zeros((img_size, img_size), dtype=np.float32)
        valid_mask[meta.pad_top : meta.pad_top + meta.new_h, meta.pad_left : meta.pad_left + meta.new_w] = 1.0
        total_valid = int(valid_mask.sum())

        inp_t = torch.from_numpy(lb_img).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
        inp_3ch = inp_t.repeat(1, 3, 1, 1)
        inp_norm = (inp_3ch - mean) / std

        with torch.no_grad(), autocast("cuda", enabled=torch.cuda.is_available()):
            out = model(inp_norm)
            logits = out["seg_logits"].float()
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        probs_valid = probs[valid_mask == 1.0]

        max_prob = float(probs_valid.max()) if len(probs_valid) > 0 else 0.0
        mean_prob = float(probs_valid.mean()) if len(probs_valid) > 0 else 0.0
        p99_prob = float(np.percentile(probs_valid, 99.0)) if len(probs_valid) > 0 else 0.0

        fp_30 = int((probs_valid >= 0.30).sum())
        fp_50 = int((probs_valid >= 0.50).sum())
        fp_70 = int((probs_valid >= 0.70).sum())

        fp_area_pct_50 = float(fp_50 / max(total_valid, 1) * 100.0)

        # Non-CVC line presence flags
        has_ett = bool(row.get("ETT - Normal", 0) or row.get("ETT - Borderline", 0) or row.get("ETT - Abnormal", 0))
        has_ngt = bool(row.get("NGT - Normal", 0) or row.get("NGT - Borderline", 0) or row.get("NGT - Abnormal", 0) or row.get("NGT - Incompletely Imaged", 0))
        has_swan = bool(row.get("Swan Ganz Catheter Present", 0))

        records.append({
            "study_id": sid,
            "patient_id": pid,
            "orig_height": orig_h,
            "orig_width": orig_w,
            "has_other_lines": has_other,
            "has_ett": has_ett,
            "has_ngt": has_ngt,
            "has_swan": has_swan,
            "max_prob": max_prob,
            "mean_prob": mean_prob,
            "p99_prob": p99_prob,
            "false_pixels_tau30": fp_30,
            "false_pixels_tau50": fp_50,
            "false_pixels_tau70": fp_70,
            "false_area_pct_tau50": fp_area_pct_50,
            "total_valid_pixels": total_valid,
        })

        if i % 50 == 0 or i == total_negatives:
            print(f"Scored {i}/{total_negatives} negatives...", flush=True)

    # 4. Sort and rank
    records.sort(key=lambda r: (r["false_pixels_tau50"], r["max_prob"]), reverse=True)

    for rank, r in enumerate(records, 1):
        r["rank"] = rank
        # Define hardness tier
        if r["false_pixels_tau50"] >= 1000:
            r["tier"] = "severe_hard"
        elif r["false_pixels_tau50"] >= 200:
            r["tier"] = "moderate_hard"
        elif r["false_pixels_tau50"] > 0 or r["max_prob"] >= 0.30:
            r["tier"] = "mild_hard"
        else:
            r["tier"] = "clean_negative"

    # 5. Export CSV (tabular)
    csv_rows = []
    for r in records:
        csv_rows.append({
            "rank": r["rank"],
            "tier": r["tier"],
            "study_id": r["study_id"],
            "patient_id": r["patient_id"],
            "false_pixels_tau50": r["false_pixels_tau50"],
            "false_area_pct_tau50": f"{r['false_area_pct_tau50']:.4f}%",
            "max_prob": f"{r['max_prob']:.4f}",
            "p99_prob": f"{r['p99_prob']:.4f}",
            "mean_prob": f"{r['mean_prob']:.6f}",
            "false_pixels_tau30": r["false_pixels_tau30"],
            "false_pixels_tau70": r["false_pixels_tau70"],
            "has_other_lines": r["has_other_lines"],
            "has_ett": r["has_ett"],
            "has_ngt": r["has_ngt"],
            "has_swan": r["has_swan"],
            "orig_size": f"{r['orig_width']}x{r['orig_height']}",
        })

    df_out = pd.DataFrame(csv_rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"\nSaved ranked CSV to {output_csv}", flush=True)

    # 6. Export JSON Manifest
    manifest_data = {
        "model_checkpoint": str(ckpt_path),
        "model_checkpoint_sha256": ckpt_sha256,
        "total_training_negatives_evaluated": len(records),
        "summary": {
            "severe_hard_count": sum(1 for r in records if r["tier"] == "severe_hard"),
            "moderate_hard_count": sum(1 for r in records if r["tier"] == "moderate_hard"),
            "mild_hard_count": sum(1 for r in records if r["tier"] == "mild_hard"),
            "clean_negative_count": sum(1 for r in records if r["tier"] == "clean_negative"),
            "studies_with_false_foreground_tau50": sum(1 for r in records if r["false_pixels_tau50"] > 0),
            "studies_with_max_prob_ge_50": sum(1 for r in records if r["max_prob"] >= 0.50),
        },
        "records": records,
    }

    os.makedirs(os.path.dirname(output_manifest), exist_ok=True)
    with open(output_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Saved hard-negatives manifest to {output_manifest}", flush=True)

    # 7. Render visual inspection panels for top N
    out_overlay_path = Path(overlays_dir)
    out_overlay_path.mkdir(parents=True, exist_ok=True)
    print(f"\nRendering visual inspection panels for top {top_n_overlays} worst false-positive studies in {out_overlay_path}...", flush=True)

    for r in records[:top_n_overlays]:
        rank = r["rank"]
        sid = r["study_id"]
        img_path = images_dir / f"{sid}.jpg"
        im = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        lb_img, meta = transform.apply_image(im)

        inp_t = torch.from_numpy(lb_img).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
        inp_norm = (inp_t.repeat(1, 3, 1, 1) - mean) / std
        with torch.no_grad(), autocast("cuda", enabled=torch.cuda.is_available()):
            out = model(inp_norm)
            probs = torch.sigmoid(out["seg_logits"].float()).squeeze().cpu().numpy()

        # RGB canvas
        bgr = cv2.cvtColor(lb_img, cv2.COLOR_GRAY2BGR)

        # Crop to unpadded region for visual inspection
        crop_bgr = bgr[meta.pad_top : meta.pad_top + meta.new_h, meta.pad_left : meta.pad_left + meta.new_w].copy()
        crop_probs = probs[meta.pad_top : meta.pad_top + meta.new_h, meta.pad_left : meta.pad_left + meta.new_w]

        # False foreground mask at tau=0.50 (red overlay)
        mask50 = crop_probs >= 0.50
        # Heatmap color for probabilities >= 0.30 (yellow-orange-red)
        heat_overlay = crop_bgr.copy()
        heat_overlay[mask50, 2] = 255  # Red channel
        heat_overlay[mask50, 1] = 50   # Green channel
        heat_overlay[mask50, 0] = 50   # Blue channel

        # Blend
        blended = cv2.addWeighted(crop_bgr, 0.65, heat_overlay, 0.35, 0)

        # Draw annotation banner
        other_lines_str = "OtherLines: YES" if r["has_other_lines"] else "OtherLines: NO"
        line_types = []
        if r["has_ett"]: line_types.append("ETT")
        if r["has_ngt"]: line_types.append("NGT")
        if r["has_swan"]: line_types.append("SwanGanz")
        lines_detail = f" ({','.join(line_types)})" if line_types else ""

        banner_text = f"Rank #{rank} | Study: {sid[:12]}... | MaxP: {r['max_prob']:.2f} | FalsePx@0.5: {r['false_pixels_tau50']} | {other_lines_str}{lines_detail}"
        cv2.rectangle(blended, (0, 0), (blended.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(blended, banner_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        out_fn = out_overlay_path / f"rank_{rank:03d}_false_{r['false_pixels_tau50']}px_maxp_{int(r['max_prob']*100)}_{sid[:16]}.png"
        cv2.imwrite(str(out_fn), blended)

    print(f"Top {top_n_overlays} inspection panels generated successfully.", flush=True)
    print("\nSummary of Training Negatives Hardness:")
    for k, v in manifest_data["summary"].items():
        print(f"  - {k}: {v}")

    return manifest_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score and rank training confirmed negatives.")
    parser.add_argument("--checkpoint", default=r"C:\Ankshit\CVC-MAGNet-V2\runs\s2_repaired_768\checkpoints\final_segmentation_v3.pt")
    parser.add_argument("--img_size", type=int, default=768)
    args = parser.parse_args()

    score_training_negatives(
        checkpoint_path=args.checkpoint,
        img_size=args.img_size,
    )
