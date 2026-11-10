"""
Comprehensive FROC Trade-off Analysis (Task 4) and Decoding Stage Failure Attribution (Task 5).

Task 4:
- Evaluates sensitivity (Success@15) vs false tips per image across thresholds tau in [0.05, 0.90].
- Compares T0 baseline, T1 Joint HRNet Seed 42, and T1 Joint HRNet Seed 43.
- Plots FROC curve to reports/froc_tradeoff.png and saves reports/sensitivity_vs_false_tips_curves.json.

Task 5:
- Traces every missed target through the 5 decoding stages:
  1. Raw heatmap activation (continuous value at target & max within 15px)
  2. Local 3x3 peak formation within 15px
  3. Threshold filter (tau = 0.20)
  4. 30px NMS suppression (checking collisions in nearby multi-tip cases)
  5. 10-candidate cap
  6. Matching tolerance (15-25px near-miss vs >25px gross error)
- Saves attribution to reports/decoding_pipeline_failure_attribution.csv.
"""
import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import scipy.ndimage as ndimage
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXP_ROOT = PROJECT_ROOT / "experiments" / "hrnet_tip_pilot_v1"
BENCHMARK_V2_PATH = EXP_ROOT / "manifests" / "development_benchmark_manifest_v2_reviewed.json"
IMAGES_DIR = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")

from experiments.hrnet_tip_pilot_v1.src.geometry import (
    CoordinateTransform,
    heatmap_point_to_native,
    native_point_to_heatmap,
)
from experiments.hrnet_tip_pilot_v1.src.models.hrnet_joint_tip import HRNetJointTipNet
from experiments.hrnet_tip_pilot_v1.src.decoder import TransparentTipDecoder


def match_study_tips_constrained(
    targets: List[Tuple[float, float]],
    preds: List[Dict[str, Any]],
    tolerance_px: float = 15.0,
) -> Dict[str, Any]:
    num_targets = len(targets)
    num_preds = len(preds)
    if num_targets == 0 or num_preds == 0:
        return {
            "num_targets": num_targets,
            "num_preds": num_preds,
            "matches": [],
            "matched_distances": [],
            "unmatched_target_indices": list(range(num_targets)),
            "unmatched_pred_indices": list(range(num_preds)),
        }

    cost_matrix = np.full((num_targets, num_preds), 1e6, dtype=np.float64)
    dist_matrix = np.zeros((num_targets, num_preds), dtype=np.float64)

    for i, t in enumerate(targets):
        for j, p in enumerate(preds):
            d = math.hypot(t[0] - p["native_x"], t[1] - p["native_y"])
            dist_matrix[i, j] = d
            if d <= tolerance_px:
                cost_matrix[i, j] = d

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    matched_dists = []
    matched_targets = set()
    matched_preds = set()

    for r, c in zip(row_ind, col_ind):
        d = dist_matrix[r, c]
        if d <= tolerance_px:
            matches.append((int(r), int(c), float(d)))
            matched_dists.append(float(d))
            matched_targets.add(r)
            matched_preds.add(c)

    return {
        "num_targets": num_targets,
        "num_preds": num_preds,
        "matches": matches,
        "matched_distances": matched_dists,
        "unmatched_target_indices": [i for i in range(num_targets) if i not in matched_targets],
        "unmatched_pred_indices": [j for j in range(num_preds) if j not in matched_preds],
    }


def extract_benchmark_heatmaps(model: nn.Module, manifest: Dict[str, Any], device: torch.device):
    """Run inference on 33 benchmark images and return raw heatmaps and transforms matching exact training evaluation."""
    from src.data.transforms import LetterboxTransform
    transform = LetterboxTransform(target_size=(768, 768), clahe_enabled=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    model.eval()
    results = {}
    with torch.no_grad():
        for s in manifest["studies"]:
            sid = s["study_id"]
            img_path = IMAGES_DIR / f"{sid}.jpg"
            pil_im = Image.open(img_path).convert("L")
            im_np = np.array(pil_im)
            orig_h, orig_w = im_np.shape
            coord_tf = CoordinateTransform.from_native_dims(orig_w, orig_h, target_size=768, stride=4)

            lb_img, meta = transform.apply_image(im_np)
            raw_t = torch.from_numpy(lb_img).unsqueeze(0).unsqueeze(0).to(device).float() / 255.0
            inp = (raw_t.repeat(1, 3, 1, 1) - mean) / std

            out = model(inp)
            tip_probs = out["tip_probs"].squeeze().cpu().numpy().astype(np.float32)

            results[sid] = {
                "heatmap": tip_probs,
                "transform": coord_tf,
                "orig_size": (orig_w, orig_h),
            }
    return results


def run_froc_sweep(
    heatmaps_by_study: Dict[str, Any],
    manifest: Dict[str, Any],
    thresholds: List[float],
    nms_radius: float = 30.0,
    max_cand: int = 10,
) -> List[Dict[str, Any]]:
    """Sweep threshold tau across heatmaps using standard decoding."""
    curve = []
    studies = manifest["studies"]
    total_studies = len(studies)
    total_targets = manifest["summary"]["total_assessable_targets"]
    neg_studies = [s for s in studies if s.get("is_negative", False)]

    for tau in thresholds:
        decoder = TransparentTipDecoder(
            conf_threshold=tau,
            nms_radius_native_px=nms_radius,
            max_candidates=max_cand,
            kernel_size_hm=3,
        )
        total_matches = 0
        total_fp = 0
        neg_fp = 0

        for s in studies:
            sid = s["study_id"]
            is_neg = s.get("is_negative", False)
            hm_data = heatmaps_by_study[sid]
            dec_res = decoder.decode(hm_data["heatmap"], hm_data["transform"])
            preds = dec_res["final_predictions"]

            targets = [(float(t["x"]), float(t["y"])) for t in s.get("assessable_tips", [])]
            m_res = match_study_tips_constrained(targets, preds, tolerance_px=15.0)

            total_matches += len(m_res["matches"])
            fp = len(m_res["unmatched_pred_indices"])
            total_fp += fp
            if is_neg:
                neg_fp += fp

        curve.append({
            "threshold": tau,
            "sensitivity": total_matches / total_targets,
            "matches_count": total_matches,
            "total_targets": total_targets,
            "mean_false_tips_per_image": total_fp / total_studies,
            "mean_false_tips_negative_image": neg_fp / len(neg_studies),
            "total_false_tips": total_fp,
        })
    return curve


def run_t0_froc_sweep(
    raw_t0_preds: Dict[str, List[Dict[str, Any]]],
    manifest: Dict[str, Any],
    thresholds: List[float],
) -> List[Dict[str, Any]]:
    """Sweep confidence threshold on saved T0 baseline predictions."""
    curve = []
    studies = manifest["studies"]
    total_studies = len(studies)
    total_targets = manifest["summary"]["total_assessable_targets"]
    neg_studies = [s for s in studies if s.get("is_negative", False)]

    for tau in thresholds:
        total_matches = 0
        total_fp = 0
        neg_fp = 0

        for s in studies:
            sid = s["study_id"]
            is_neg = s.get("is_negative", False)
            all_p = raw_t0_preds.get(sid, [])
            filtered_p = [p for p in all_p if p.get("conf", 1.0) >= tau]

            targets = [(float(t["x"]), float(t["y"])) for t in s.get("assessable_tips", [])]
            m_res = match_study_tips_constrained(targets, filtered_p, tolerance_px=15.0)

            total_matches += len(m_res["matches"])
            fp = len(m_res["unmatched_pred_indices"])
            total_fp += fp
            if is_neg:
                neg_fp += fp

        curve.append({
            "threshold": tau,
            "sensitivity": total_matches / total_targets,
            "matches_count": total_matches,
            "total_targets": total_targets,
            "mean_false_tips_per_image": total_fp / total_studies,
            "mean_false_tips_negative_image": neg_fp / len(neg_studies),
            "total_false_tips": total_fp,
        })
    return curve


def trace_missed_targets(
    heatmaps_by_study: Dict[str, Any],
    manifest: Dict[str, Any],
    seed_name: str,
) -> List[Dict[str, Any]]:
    """Trace every target through decoding stages to attribute failure causes."""
    decoder = TransparentTipDecoder(
        conf_threshold=0.20,
        nms_radius_native_px=30.0,
        max_candidates=10,
        kernel_size_hm=3,
    )

    records = []

    for s in manifest["studies"]:
        sid = s["study_id"]
        tips = s.get("assessable_tips", [])
        if not tips:
            continue

        hm_data = heatmaps_by_study[sid]
        hm = hm_data["heatmap"]  # [192, 192]
        transform = hm_data["transform"]

        # Run full transparent decode
        dec_res = decoder.decode(hm, transform)
        final_preds = dec_res["final_predictions"]
        nms_cands = dec_res["nms_candidates"]

        # Also extract unthresholded peaks (tau=0.01) to find all local maxima
        unconstrained_decoder = TransparentTipDecoder(
            conf_threshold=0.01,
            nms_radius_native_px=30.0,
            max_candidates=100,
            kernel_size_hm=3,
        )
        unconstrained_res = unconstrained_decoder.decode(hm, transform)
        all_peaks = unconstrained_res["raw_candidates"]  # all points >=0.01

        targets = [(float(t["x"]), float(t["y"])) for t in tips]
        m15 = match_study_tips_constrained(targets, final_preds, tolerance_px=15.0)
        matched_target_indices = {m[0] for m in m15["matches"]}

        # Trace each target
        for t_idx, tip in enumerate(tips):
            tx, ty = float(tip["x"]), float(tip["y"])
            inst_id = tip["catheter_instance_id"]
            is_matched = (t_idx in matched_target_indices)

            # Map target to continuous heatmap coordinates
            hx, hy = native_point_to_heatmap(tx, ty, transform)

            # Stage 1: Continuous Heatmap Activation
            # Sample continuous value using bilinear interpolation or clamp
            hx_c = max(0.0, min(191.0, hx))
            hy_c = max(0.0, min(191.0, hy))
            ix, iy = int(round(hx_c)), int(round(hy_c))
            raw_hm_at_target = float(hm[iy, ix])

            # Max activation within 15 native pixels on heatmap
            # Radius in heatmap cells:
            r_hm = 15.0 / (transform.native_width / transform.resized_width * transform.stride)
            x_min = max(0, int(math.floor(hx - r_hm)))
            x_max = min(191, int(math.ceil(hx + r_hm)))
            y_min = max(0, int(math.floor(hy - r_hm)))
            y_max = min(191, int(math.ceil(hy + r_hm)))
            sub_hm = hm[y_min : y_max + 1, x_min : x_max + 1]
            max_activation_15px = float(np.max(sub_hm)) if sub_hm.size > 0 else raw_hm_at_target

            # Stage 2: Peak Formation within 15px native
            peaks_15px = []
            for p in unconstrained_res["nms_candidates"]:
                d_nat = math.hypot(tx - p["native_x"], ty - p["native_y"])
                if d_nat <= 15.0:
                    peaks_15px.append((d_nat, p))

            peaks_15px.sort(key=lambda x: x[0])  # sort by distance
            has_peak_15px = len(peaks_15px) > 0
            closest_peak = peaks_15px[0][1] if has_peak_15px else None
            closest_peak_dist = peaks_15px[0][0] if has_peak_15px else None
            closest_peak_conf = float(closest_peak["conf"]) if closest_peak else 0.0

            # Nearest prediction in final predictions
            if final_preds:
                final_dists = [math.hypot(tx - p["native_x"], ty - p["native_y"]) for p in final_preds]
                nearest_final_dist = min(final_dists)
                nearest_final_p = final_preds[np.argmin(final_dists)]
                nearest_final_conf = nearest_final_p["conf"]
            else:
                nearest_final_dist = float("inf")
                nearest_final_conf = 0.0

            # Stage 3-5: Attribution
            if is_matched:
                attribution = "Success@15 Match"
            else:
                if max_activation_15px < 0.10:
                    attribution = "Omission: No Heatmap Activation (<0.10)"
                elif max_activation_15px < 0.20:
                    attribution = "Subthreshold Activation (0.10-0.20): Dropped by Tau=0.20"
                elif has_peak_15px and closest_peak_conf < 0.20:
                    attribution = "Subthreshold Peak (<0.20): Local Peak Existed But Filtered"
                else:
                    # Check if a peak existed in raw candidates above 0.20 but was suppressed by 30px NMS
                    # Check suppressed peaks in decoder
                    # Find all peaks >= 0.20 before NMS
                    raw_peaks_above_tau = [
                        p for p in unconstrained_res["raw_candidates"]
                        if p["conf"] >= 0.20 and math.hypot(tx - p["native_x"], ty - p["native_y"]) <= 15.0
                    ]
                    # Check if suppressed by NMS
                    was_suppressed_by_nms = False
                    nms_suppressor = None
                    if raw_peaks_above_tau:
                        # Check if any survived to nms_cands
                        survived = any(
                            math.hypot(p["native_x"] - p_nms["native_x"], p["native_y"] - p_nms["native_y"]) < 1.0
                            for p in raw_peaks_above_tau for p_nms in nms_cands
                        )
                        if not survived:
                            was_suppressed_by_nms = True
                            attribution = "NMS Collision: Suppressed by 30px NMS Radius"

                    if not was_suppressed_by_nms:
                        if nearest_final_dist <= 25.0:
                            attribution = f"Precision Near-Miss (15-25px): Nearest at {nearest_final_dist:.1f}px"
                        elif nearest_final_dist <= 70.0:
                            attribution = f"Moderate Localization Error (25-70px): Nearest at {nearest_final_dist:.1f}px"
                        else:
                            attribution = f"Gross Mislocalization / Wrong Branch (>70px): Nearest at {nearest_final_dist:.1f}px"

            records.append({
                "seed": seed_name,
                "study_id": sid,
                "catheter_instance_id": inst_id,
                "review_status": tip.get("review_status", "reviewed"),
                "is_multi_cvc": len(tips) > 1,
                "target_x": tx,
                "target_y": ty,
                "is_matched_15px": is_matched,
                "raw_hm_at_target": round(raw_hm_at_target, 4),
                "max_activation_15px": round(max_activation_15px, 4),
                "has_peak_15px": has_peak_15px,
                "closest_peak_conf": round(closest_peak_conf, 4) if closest_peak else 0.0,
                "closest_peak_dist": round(closest_peak_dist, 2) if closest_peak_dist is not None else -1.0,
                "nearest_final_pred_dist": round(nearest_final_dist, 2) if not math.isinf(nearest_final_dist) else 999.0,
                "nearest_final_pred_conf": round(nearest_final_conf, 4),
                "failure_attribution": attribution,
            })

    return records


def run_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running FROC and Failure Attribution on {device}...")

    with open(BENCHMARK_V2_PATH, "r") as f:
        manifest_v2 = json.load(f)

    # 1. Load Seed 42 Model and Extract Heatmaps
    s42_ckpt_path = EXP_ROOT / "runs" / "seed_42" / "t1_joint_hrnet" / "checkpoints" / "final_model.pt"
    s42_ckpt = torch.load(s42_ckpt_path, map_location=device, weights_only=False)
    model_s42 = HRNetJointTipNet(pretrained=False)
    model_s42.load_state_dict(s42_ckpt["model_state_dict"])
    model_s42.to(device)
    print("Extracting heatmaps for Seed 42...")
    hm_s42 = extract_benchmark_heatmaps(model_s42, manifest_v2, device)

    # 2. Load Seed 43 Model and Extract Heatmaps
    s43_ckpt_path = EXP_ROOT / "runs" / "seed_43" / "t1_joint_hrnet" / "checkpoints" / "final_model.pt"
    s43_ckpt = torch.load(s43_ckpt_path, map_location=device, weights_only=False)
    model_s43 = HRNetJointTipNet(pretrained=False)
    model_s43.load_state_dict(s43_ckpt["model_state_dict"])
    model_s43.to(device)
    print("Extracting heatmaps for Seed 43...")
    hm_s43 = extract_benchmark_heatmaps(model_s43, manifest_v2, device)

    # 3. Load T0 Baseline Predictions
    with open(EXP_ROOT / "predictions" / "t0_baseline_predictions.json", "r") as f:
        raw_t0 = json.load(f)
    t0_by_study = {}
    for sid, preds in raw_t0.items():
        t0_by_study[sid] = [
            {"native_x": float(p["native_x"]), "native_y": float(p["native_y"]), "conf": float(p.get("conf", 1.0))}
            for p in preds
        ]

    # 4. Run FROC Sweep
    thresholds_t1 = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]
    thresholds_t0 = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    print("Sweeping FROC curves...")
    froc_s42 = run_froc_sweep(hm_s42, manifest_v2, thresholds_t1)
    froc_s43 = run_froc_sweep(hm_s43, manifest_v2, thresholds_t1)
    froc_t0 = run_t0_froc_sweep(t0_by_study, manifest_v2, thresholds_t0)

    froc_data = {
        "benchmark_version": "v2_audited",
        "total_targets": manifest_v2["summary"]["total_assessable_targets"],
        "total_studies": manifest_v2["summary"]["total_studies"],
        "T0_baseline": froc_t0,
        "T1_seed_42": froc_s42,
        "T1_seed_43": froc_s43,
    }

    out_froc_json = EXP_ROOT / "reports" / "sensitivity_vs_false_tips_curves.json"
    with open(out_froc_json, "w") as f:
        json.dump(froc_data, f, indent=2)
    print(f"Saved FROC curve data to: {out_froc_json}")

    # Plot FROC Curve
    plt.figure(figsize=(9, 6), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot T0
    t0_x = [pt["mean_false_tips_per_image"] for pt in froc_t0]
    t0_y = [pt["sensitivity"] * 100 for pt in froc_t0]
    plt.plot(t0_x, t0_y, "o-", color="#e74c3c", linewidth=2.5, markersize=7, label="T0 Baseline (Skeleton/Heuristic)")

    # Plot T1 Seed 42
    s42_x = [pt["mean_false_tips_per_image"] for pt in froc_s42]
    s42_y = [pt["sensitivity"] * 100 for pt in froc_s42]
    plt.plot(s42_x, s42_y, "s-", color="#2980b9", linewidth=2.5, markersize=7, label="T1 Joint HRNet (Seed 42)")

    # Plot T1 Seed 43
    s43_x = [pt["mean_false_tips_per_image"] for pt in froc_s43]
    s43_y = [pt["sensitivity"] * 100 for pt in froc_s43]
    plt.plot(s43_x, s43_y, "^-", color="#27ae60", linewidth=2.5, markersize=7, label="T1 Joint HRNet (Seed 43)")

    # Annotate operating point tau = 0.20
    pt_s42_20 = [pt for pt in froc_s42 if abs(pt["threshold"] - 0.20) < 1e-3][0]
    plt.scatter([pt_s42_20["mean_false_tips_per_image"]], [pt_s42_20["sensitivity"] * 100], color="#1f618d", s=180, zorder=5)
    plt.annotate(
        f"Tau=0.20: S@15={pt_s42_20['sensitivity']*100:.1f}%\nFP/img={pt_s42_20['mean_false_tips_per_image']:.2f}",
        (pt_s42_20["mean_false_tips_per_image"], pt_s42_20["sensitivity"] * 100),
        textcoords="offset points", xytext=(-20, 15),
        bbox=dict(boxstyle="round,pad=0.3", fc="#ebf5fb", ec="#2980b9", lw=1),
        fontsize=9, weight="bold"
    )

    plt.xlabel("Mean False Tips per Image", fontsize=12, weight="bold")
    plt.ylabel("Sensitivity / Success@15 (%)", fontsize=12, weight="bold")
    plt.title("Free-Response ROC (FROC) Curve: Sensitivity vs False Tips per Image", fontsize=13, weight="bold")
    plt.xlim(0, 15)
    plt.ylim(0, 45)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()

    out_plot = EXP_ROOT / "reports" / "froc_tradeoff.png"
    plt.savefig(out_plot)
    plt.close()
    print(f"Saved FROC trade-off plot to: {out_plot}")

    # 5. Trace Missed Targets (Task 5)
    print("\nTracing missed targets through decoding pipeline...")
    trace_s42 = trace_missed_targets(hm_s42, manifest_v2, "Seed_42")
    trace_s43 = trace_missed_targets(hm_s43, manifest_v2, "Seed_43")

    df_trace = pd.DataFrame(trace_s42 + trace_s43)
    out_trace_csv = EXP_ROOT / "reports" / "decoding_pipeline_failure_attribution.csv"
    df_trace.to_csv(out_trace_csv, index=False)
    print(f"Saved failure attribution CSV to: {out_trace_csv}")

    # Print Breakdown of Failure Attribution for Seed 42
    df_s42_misses = df_trace[(df_trace["seed"] == "Seed_42") & (~df_trace["is_matched_15px"])]
    print("\n=== FAILURE ATTRIBUTION BREAKDOWN FOR SEED 42 (27 Missed Targets) ===")
    print(df_s42_misses["failure_attribution"].value_counts())

    # Multi-CVC collision check
    print("\n=== MULTI-CVC COLLISION & NMS ANALYSIS ===")
    multi_misses = df_s42_misses[df_s42_misses["is_multi_cvc"]]
    for idx, r in multi_misses.iterrows():
        print(f"Study: {r['study_id'][:16]}... Inst: {r['catheter_instance_id']} | Attrib: {r['failure_attribution']} | Raw HM: {r['raw_hm_at_target']:.3f} | Max 15px: {r['max_activation_15px']:.3f} | Nearest Pred Dist: {r['nearest_final_pred_dist']:.1f}px")


if __name__ == "__main__":
    run_pipeline()
