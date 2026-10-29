"""Decoder Comparison: D0 (Heatmap Cell Center) vs D1 (DARK CVPR 2020 Subpixel Refinement).

Implements Section 6 of the Experiment Specification:
- Using saved heatmaps from the trained tip model, compare:
  D0: Local maximum at heatmap cell centre
  D1: Local subpixel refinement around each peak via Taylor expansion (DARK CVPR 2020)
- Applies local refinement strictly to each detected peak independently.
- Evaluates on identical candidates using Hungarian bipartite matching.
- Compares Success@5, @10, @15, @20, @30, median error, and p95 error.
- Saves result report to reports/decoder_comparison_d0_d1.md and experiments/D0_D1_decoder_comparison/
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms.functional as TF

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.tip_metrics import compute_tip_metrics, compute_patient_clustered_bootstrap
from src.infer.decoders import extract_peaks_from_heatmap, decode_d0_cell_center, decode_d1_subpixel_dark
from src.models.tip_heatmap import CVCTipHeatmapNet, CVCJointSegTipNet

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "D0_D1_decoder_comparison"


def compare_decoders(
    checkpoint_path: str,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    img_size: int = 768,
    threshold: float = 0.20,
    suppression_radius: float = 3.0,
    is_joint: bool = False,
) -> Dict[str, Any]:
    print(f"=== Starting Decoder Comparison (D0 vs D1) on {checkpoint_path} ===")
    device = torch.device(device_str)

    # 1. Load Model
    if is_joint:
        model = CVCJointSegTipNet(pretrained=False)
    else:
        model = CVCTipHeatmapNet(pretrained=False)
    
    st = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = st.get("model_state_dict", st.get("state_dict", st))
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # 2. Load Validation Cohort & Contracts
    with open(DATA_DIR / "splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)
    val_study_ids = splits["partitions"]["dev_pilot_val"]["study_ids"]

    catheter_df = pd.read_parquet(DATA_DIR / "cvc_catheter_contracts.parquet")
    val_catheters = catheter_df[catheter_df["study_id"].isin(val_study_ids)]
    val_grouped = val_catheters.groupby("study_id")

    study_inv = pd.read_parquet(DATA_DIR / "study_inventory.parquet")
    study_to_neg = dict(zip(study_inv["StudyInstanceUID"], ~study_inv["has_cvc"]))
    study_to_pid = dict(zip(study_inv["StudyInstanceUID"], study_inv["PatientID"]))

    images_dir = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")
    
    # 3. Cache Model Heatmaps
    print(f"Extracting heatmaps for {len(val_study_ids)} validation studies at {img_size}px...", flush=True)
    cached_studies: List[Dict[str, Any]] = []

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    with torch.no_grad():
        for sid in val_study_ids:
            fpath = images_dir / f"{sid}.jpg"
            if not fpath.exists():
                fpath = images_dir / f"{sid}.png"
            if not fpath.exists():
                continue

            pil_img = Image.open(fpath).convert("RGB")
            native_w, native_h = pil_img.size

            resized = pil_img.resize((img_size, img_size), Image.Resampling.BILINEAR)
            t_img = TF.to_tensor(resized).to(device)
            t_img = (t_img - mean) / std
            t_img = t_img.unsqueeze(0)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out = model(t_img)
                heatmap = out["tip_probs"][0, 0].float().cpu().numpy()

            gt_tips = []
            if sid in val_grouped.groups:
                gt_rows = val_grouped.get_group(sid)
                for _, r in gt_rows.iterrows():
                    if r["tip_state"] == "assessable" and pd.notna(r["tip_x"]) and pd.notna(r["tip_y"]):
                        gt_tips.append({
                            "x": float(r["tip_x"]),
                            "y": float(r["tip_y"]),
                            "tip_state": str(r["tip_state"]),
                            "instance_id": int(r["instance_id"]),
                            "is_border": False,
                        })

            pid = str(study_to_pid.get(sid, sid))
            is_pos = not study_to_neg.get(sid, False)

            cached_studies.append({
                "study_id": sid,
                "patient_id": pid,
                "is_cvc_positive": is_pos,
                "native_w": native_w,
                "native_h": native_h,
                "heatmap": heatmap,
                "gt_tips": gt_tips,
            })

    # 4. Decode with D0 and D1 on identical candidates
    records_d0 = []
    records_d1 = []

    total_raw_peaks = 0
    total_nms_peaks = 0

    for item in cached_studies:
        hm = item["heatmap"]
        w = item["native_w"]
        h = item["native_h"]

        raw_peaks, nms_peaks = extract_peaks_from_heatmap(
            hm,
            threshold=threshold,
            nms_kernel_size=3,
            suppression_radius_px=suppression_radius,
        )
        total_raw_peaks += len(raw_peaks)
        total_nms_peaks += len(nms_peaks)

        tips_d0 = []
        tips_d1 = []

        for p in nms_peaks:
            # D0
            x0, y0 = decode_d0_cell_center(p, stride=4, img_size=img_size, native_w=w, native_h=h)
            tips_d0.append({"x": x0, "y": y0, "conf": p["conf"]})

            # D1
            x1, y1, _ = decode_d1_subpixel_dark(hm, p, stride=4, img_size=img_size, native_w=w, native_h=h)
            tips_d1.append({"x": x1, "y": y1, "conf": p["conf"]})

        records_d0.append({
            "study_id": item["study_id"],
            "patient_id": item["patient_id"],
            "is_cvc_positive": item["is_cvc_positive"],
            "gt_tips": item["gt_tips"],
            "pred_tips": tips_d0,
            "raw_candidates": tips_d0,
        })

        records_d1.append({
            "study_id": item["study_id"],
            "patient_id": item["patient_id"],
            "is_cvc_positive": item["is_cvc_positive"],
            "gt_tips": item["gt_tips"],
            "pred_tips": tips_d1,
            "raw_candidates": tips_d1,
        })

    # 5. Hungarian Evaluation
    metrics_d0 = compute_tip_metrics(pd.DataFrame(records_d0), confidence_threshold=threshold)
    metrics_d1 = compute_tip_metrics(pd.DataFrame(records_d1), confidence_threshold=threshold)

    sum_d0 = metrics_d0["summary"]
    sum_d1 = metrics_d1["summary"]

    # 6. Save Comparison Artifacts
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS_DIR / "summary_d0.json", "w", encoding="utf-8") as f:
        json.dump(sum_d0, f, indent=2)
    with open(EXPERIMENTS_DIR / "summary_d1.json", "w", encoding="utf-8") as f:
        json.dump(sum_d1, f, indent=2)

    # Markdown Comparison Table
    tol_keys = ["Success@5", "Success@10", "Success@15", "Success@20", "Success@30"]
    rows = []
    for tk in tol_keys:
        v0 = sum_d0["tolerance_metrics"][tk]["success_rate"] * 100
        v1 = sum_d1["tolerance_metrics"][tk]["success_rate"] * 100
        diff = v1 - v0
        rows.append({
            "Metric": tk,
            "D0 (Cell Center)": f"{v0:.2f}%",
            "D1 (DARK Taylor)": f"{v1:.2f}%",
            "Delta (D1 - D0)": f"{diff:+.2f}%",
        })

    # Error distribution
    e0_med = sum_d0["error_distribution_matched_15px"]["median_error_px"]
    e1_med = sum_d1["error_distribution_matched_15px"]["median_error_px"]
    e0_p95 = sum_d0["error_distribution_matched_15px"]["p95_error_px"]
    e1_p95 = sum_d1["error_distribution_matched_15px"]["p95_error_px"]

    rows.append({
        "Metric": "Matched Error Median (px)",
        "D0 (Cell Center)": f"{e0_med:.2f} px" if e0_med else "N/A",
        "D1 (DARK Taylor)": f"{e1_med:.2f} px" if e1_med else "N/A",
        "Delta (D1 - D0)": f"{(e1_med - e0_med):+.2f} px" if (e0_med and e1_med) else "N/A",
    })
    rows.append({
        "Metric": "Matched Error P95 (px)",
        "D0 (Cell Center)": f"{e0_p95:.2f} px" if e0_p95 else "N/A",
        "D1 (DARK Taylor)": f"{e1_p95:.2f} px" if e1_p95 else "N/A",
        "Delta (D1 - D0)": f"{(e1_p95 - e0_p95):+.2f} px" if (e0_p95 and e1_p95) else "N/A",
    })

    df_comp = pd.DataFrame(rows)

    report_md = f"""# Coordinate Decoder Comparison: D0 (Cell Center) vs D1 (DARK CVPR 2020)

**Date**: 2026-10-17  
**Model Checkpoint**: `{checkpoint_path}`  
**Threshold**: {threshold} | **NMS Suppression Radius**: {suppression_radius} px  
**Diagnostics**:
- Total raw peaks above threshold: {total_raw_peaks}
- Post-NMS peaks retained: {total_nms_peaks} (mean {total_nms_peaks/len(val_study_ids):.2f} / image)

---

## Benchmark Comparison (Hungarian Matching on Ground Truth Tips)

{df_comp.to_markdown(index=False)}

---

## Conclusion
- **D0 (Cell Center)**: Direct scaling of integer grid coordinates $(x + 0.5) \times 4$.
- **D1 (DARK Subpixel Refinement)**: Taylor expansion around local Gaussian peak surface using gradient and Hessian.
- Benefit of subpixel refinement is strictly measured on native radiograph pixel tolerances.
"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "decoder_comparison_d0_d1.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\nSaved Decoder Comparison Report to: {report_path}")
    print(df_comp.to_string(index=False))

    return {
        "summary_d0": sum_d0,
        "summary_d1": sum_d1,
        "comparison_table": df_comp.to_dict(orient="records"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--is_joint", action="store_true")
    args = parser.parse_args()
    compare_decoders(checkpoint_path=args.checkpoint, threshold=args.threshold, is_joint=args.is_joint)
