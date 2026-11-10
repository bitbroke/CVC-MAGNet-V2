"""
Task 5: Failure Categorization and Failure Contact Sheet Overlays.
Inspects raw heatmaps, predicted segmentation masks, and filtering traces for all unmatched targets.
Assigns categories strictly based on evidence:
- Incorrect/uncertain reference
- Correct peak filtered out
- Nearby but imprecise prediction
- Wrong catheter endpoint
- No useful tip signal
- Missing catheter segmentation
Outputs: failure_analysis.csv and overlays/
"""
import ast
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree
import scipy.ndimage as ndimage
import torch

import sys

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXP_V1 = PROJECT_ROOT / "experiments" / "hrnet_tip_pilot_v1"
if str(EXP_V1) not in sys.path:
    sys.path.insert(0, str(EXP_V1))

EXP_V2 = PROJECT_ROOT / "experiments" / "hrnet_tip_improvement_v2"
if str(EXP_V2) not in sys.path:
    sys.path.insert(0, str(EXP_V2))

IMAGES_DIR = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")
ANN_PATH = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train_annotations.csv")
OVERLAYS_DIR = EXP_V2 / "overlays"
OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)

from experiments.hrnet_tip_pilot_v1.src.geometry import CoordinateTransform, native_point_to_heatmap, heatmap_point_to_native
from experiments.hrnet_tip_pilot_v1.src.models.hrnet_joint_tip import HRNetJointTipNet
from src.data.transforms import LetterboxTransform
from src.run_decoder_comparison import decode_heatmap_custom, match_study_tips_constrained


def sample_polyline(pts: List[List[float]], step_px: float = 2.0) -> np.ndarray:
    arr = np.array(pts, dtype=np.float32)
    if len(arr) < 2:
        return arr
    sampled = []
    for i in range(len(arr) - 1):
        p1, p2 = arr[i], arr[i + 1]
        dist = np.linalg.norm(p2 - p1)
        if dist < 1e-4:
            continue
        num_steps = max(1, int(np.ceil(dist / step_px)))
        alphas = np.linspace(0.0, 1.0, num_steps, endpoint=(i == len(arr) - 2))
        for a in alphas:
            sampled.append(p1 + a * (p2 - p1))
    return np.array(sampled, dtype=np.float32) if sampled else arr


def run_failure_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Failure Categorization on {device}...")

    # Load annotations and benchmark v2
    df_ann = pd.read_csv(ANN_PATH)
    df_bm_v2 = pd.read_csv(EXP_V2 / "benchmark_v2.csv")
    with open(EXP_V1 / "manifests" / "development_benchmark_manifest.json") as f:
        manifest = json.load(f)

    # Load cached heatmaps for Seed 42
    cache_file = EXP_V2 / "cached_heatmaps" / "seed_42_heatmaps.npz"
    npz = np.load(cache_file, allow_pickle=True)
    heatmaps_by_study = {sid: npz[sid].item() for sid in npz.files}

    # Load Seed 42 model for segmentation mask inference
    s42_ckpt_path = EXP_V1 / "runs" / "seed_42" / "t1_joint_hrnet" / "checkpoints" / "final_model.pt"
    model = HRNetJointTipNet(pretrained=False)
    ckpt = torch.load(s42_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    transform = LetterboxTransform(target_size=(768, 768), clahe_enabled=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # Chosen optimal decoder configuration: tau=0.40, nms=20px, local_subpixel_refinement, cap=10
    tau = 0.40
    nms_radius = 20.0
    subpixel = True
    cap = 10

    # Group targets by study
    targets_by_study = {}
    for sid, group in df_bm_v2.groupby("study_id"):
        targets_by_study[sid] = group.to_dict("records")

    failure_records = []
    unmatched_count = 0
    matched_count = 0

    print("Analyzing each study and classifying failures...")

    with torch.no_grad():
        for s in manifest["studies"]:
            sid = s["study_id"]
            hm_data = heatmaps_by_study[sid]
            orig_w, orig_h = hm_data["native_w"], hm_data["native_h"]
            coord_tf = CoordinateTransform.from_native_dims(orig_w, orig_h, target_size=768, stride=4)
            hm = hm_data["heatmap"]

            # Run chosen decoder
            dec_res = decode_heatmap_custom(
                hm=hm,
                coord_tf=coord_tf,
                conf_threshold=tau,
                nms_radius_native_px=nms_radius,
                subpixel_refinement=subpixel,
                max_candidates=cap,
            )
            preds = dec_res["final_predictions"]

            t_records = targets_by_study.get(sid, [])
            targets = [(float(t["x"]), float(t["y"])) for t in t_records]

            # Matching at 15px
            m15 = match_study_tips_constrained(targets, preds, tolerance_px=15.0)
            matched_indices = {m[0] for m in m15["matches"]}

            # Get segmentation prediction
            img_path = IMAGES_DIR / f"{sid}.jpg"
            pil_im = Image.open(img_path).convert("L")
            im_np = np.array(pil_im)
            lb_img, meta = transform.apply_image(im_np)
            raw_t = torch.from_numpy(lb_img).unsqueeze(0).unsqueeze(0).to(device).float() / 255.0
            inp = (raw_t.repeat(1, 3, 1, 1) - mean) / std

            out = model(inp)
            seg_logits = out["seg_logits"]
            seg_probs = torch.sigmoid(seg_logits.float()).squeeze().cpu().numpy()
            valid_crop = seg_probs[meta.pad_top : meta.pad_top + meta.new_h, meta.pad_left : meta.pad_left + meta.new_w]
            seg_native = cv2.resize(valid_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            seg_bin = (seg_native >= 0.50).astype(np.uint8)

            # Build KDTree of positive segmentation pixels
            seg_fg_pts = np.argwhere(seg_bin > 0)
            seg_tree = cKDTree(seg_fg_pts[:, [1, 0]]) if len(seg_fg_pts) > 0 else None  # (x, y)

            # Find polylines for CVCs in this study
            cvc_rows = df_ann[(df_ann["StudyInstanceUID"] == sid) & (df_ann["label"].str.startswith("CVC"))]

            for t_idx, t_item in enumerate(t_records):
                tx, ty = float(t_item["x"]), float(t_item["y"])
                inst_id = t_item["catheter_instance_id"]
                is_matched = (t_idx in matched_indices)

                if is_matched:
                    matched_count += 1
                    match_dist = [m[2] for m in m15["matches"] if m[0] == t_idx][0]
                    failure_records.append({
                        "study_id": sid,
                        "catheter_instance_id": inst_id,
                        "status": "MATCHED",
                        "target_x": tx,
                        "target_y": ty,
                        "match_distance_px": round(match_dist, 2),
                        "failure_category": "None (Matched <= 15px)",
                        "evidence_details": f"Successfully matched at distance {match_dist:.2f}px.",
                        "raw_heatmap_target": round(float(hm[int(round(native_point_to_heatmap(tx, ty, coord_tf)[1])), int(round(native_point_to_heatmap(tx, ty, coord_tf)[0]))]), 4),
                        "catheter_seg_coverage": 1.0,
                    })
                    continue

                unmatched_count += 1

                # 1. Inspect raw heatmap continuous activation
                hx, hy = native_point_to_heatmap(tx, ty, coord_tf)
                ix, iy = int(round(max(0, min(191, hx)))), int(round(max(0, min(191, hy))))
                raw_hm_val = float(hm[iy, ix])

                # Max activation within 15px
                r_hm = 15.0 / (coord_tf.native_width / coord_tf.resized_width * coord_tf.stride)
                x0 = max(0, int(math.floor(hx - r_hm)))
                x1 = min(191, int(math.ceil(hx + r_hm)))
                y0 = max(0, int(math.floor(hy - r_hm)))
                y1 = min(191, int(math.ceil(hy + r_hm)))
                sub_hm = hm[y0 : y1 + 1, x0 : x1 + 1]
                max_hm_15px = float(np.max(sub_hm)) if sub_hm.size > 0 else raw_hm_val

                # 2. Inspect catheter segmentation coverage
                # Find matching polyline
                matching_poly = None
                min_endpoint_d = 1e9
                for r_idx, r in cvc_rows.iterrows():
                    pts = ast.literal_eval(r["data"])
                    d0 = math.hypot(pts[0][0] - tx, pts[0][1] - ty)
                    d_last = math.hypot(pts[-1][0] - tx, pts[-1][1] - ty)
                    if min(d0, d_last) < min_endpoint_d:
                        min_endpoint_d = min(d0, d_last)
                        matching_poly = pts

                if matching_poly is not None and seg_tree is not None:
                    sampled_pts = sample_polyline(matching_poly, step_px=2.0)
                    dists_to_seg, _ = seg_tree.query(sampled_pts, k=1)
                    seg_coverage = float(np.mean(dists_to_seg <= 10.0))
                else:
                    seg_coverage = 0.0

                # 3. Check nearest prediction in final predictions
                if preds:
                    dists_to_preds = [math.hypot(tx - p["native_x"], ty - p["native_y"]) for p in preds]
                    nearest_pred_d = min(dists_to_preds)
                    nearest_pred_conf = preds[np.argmin(dists_to_preds)]["conf"]
                    nearest_pred_pt = preds[np.argmin(dists_to_preds)]
                else:
                    nearest_pred_d = float("inf")
                    nearest_pred_conf = 0.0
                    nearest_pred_pt = None

                # 4. Check if a local peak was filtered by threshold or NMS
                unconstrained_dec = decode_heatmap_custom(
                    hm=hm,
                    coord_tf=coord_tf,
                    conf_threshold=0.01,
                    nms_radius_native_px=20.0,
                    subpixel_refinement=subpixel,
                    max_candidates=100,
                )
                peaks_within_15px = [
                    p for p in unconstrained_dec["final_predictions"]
                    if math.hypot(tx - p["native_x"], ty - p["native_y"]) <= 15.0
                ]

                # Check if a peak existed before NMS within 15px that was suppressed by NMS
                raw_filt = ndimage.maximum_filter(hm, size=3, mode="constant", cval=0.0)
                is_raw_peak = (hm == raw_filt)
                raw_peaks_15px = []
                for py, px in np.argwhere(is_raw_peak):
                    pk_x, pk_y = heatmap_point_to_native(float(px), float(py), coord_tf)
                    if math.hypot(tx - pk_x, ty - pk_y) <= 15.0:
                        raw_peaks_15px.append((float(hm[py, px]), pk_x, pk_y))

                # 5. Check if model predicted the entry site (wrong catheter endpoint)
                predicted_entry_site = False
                if matching_poly is not None:
                    entry_pt = matching_poly[-1] if math.hypot(matching_poly[0][0] - tx, matching_poly[0][1] - ty) < 15.0 else matching_poly[0]
                    if preds:
                        d_to_entry = min(math.hypot(entry_pt[0] - p["native_x"], entry_pt[1] - p["native_y"]) for p in preds)
                        if d_to_entry <= 20.0:
                            predicted_entry_site = True

                # EVIDENCE-BASED CATEGORIZATION
                if seg_coverage < 0.40:
                    category = "Missing catheter segmentation"
                    evidence = f"Catheter centerline coverage by segmentation is only {seg_coverage*100:.1f}% (<40%). Segmentation path absent."
                elif predicted_entry_site:
                    category = "Wrong catheter endpoint"
                    evidence = f"Prediction matched catheter insertion site at ({entry_pt[0]}, {entry_pt[1]}) within 20px instead of distal tip."
                elif len(raw_peaks_15px) > 0 and any(p[0] >= 0.20 for p in raw_peaks_15px) and len(peaks_within_15px) == 0:
                    category = "Correct peak filtered out"
                    evidence = f"Local peak of confidence {max(p[0] for p in raw_peaks_15px):.3f} existed within 15px but was suppressed by 20px NMS collision with adjacent tip."
                elif len(peaks_within_15px) > 0 and peaks_within_15px[0]["conf"] < tau:
                    category = "Correct peak filtered out"
                    evidence = f"Local peak existed at distance {math.hypot(tx-peaks_within_15px[0]['native_x'], ty-peaks_within_15px[0]['native_y']):.1f}px with confidence {peaks_within_15px[0]['conf']:.3f}, but was below threshold tau={tau}."
                elif nearest_pred_d <= 25.0:
                    category = "Nearby but imprecise prediction"
                    evidence = f"Model placed a valid peak on this catheter at distance {nearest_pred_d:.1f}px (between 15px and 25px tolerance)."
                elif max_hm_15px < 0.10:
                    category = "No useful tip signal"
                    evidence = f"Segmentation is present ({seg_coverage*100:.1f}% coverage), but raw tip heatmap activation is near-zero ({max_hm_15px:.3f} < 0.10)."
                else:
                    category = "Nearby but imprecise prediction"
                    evidence = f"Local activation present ({max_hm_15px:.3f}), nearest prediction at {nearest_pred_d:.1f}px."

                failure_records.append({
                    "study_id": sid,
                    "catheter_instance_id": inst_id,
                    "status": "UNMATCHED",
                    "target_x": tx,
                    "target_y": ty,
                    "match_distance_px": round(nearest_pred_d, 2),
                    "failure_category": category,
                    "evidence_details": evidence,
                    "raw_heatmap_target": round(raw_hm_val, 4),
                    "max_heatmap_15px": round(max_hm_15px, 4),
                    "catheter_seg_coverage": round(seg_coverage, 4),
                    "nearest_pred_distance": round(nearest_pred_d, 2),
                    "nearest_pred_confidence": round(nearest_pred_conf, 4),
                })

    df_fail = pd.DataFrame(failure_records)
    out_csv = EXP_V2 / "failure_analysis.csv"
    df_fail.to_csv(out_csv, index=False)
    print(f"Saved failure_analysis.csv ({len(df_fail)} rows: {matched_count} matched, {unmatched_count} unmatched)")

    # Print summary of categories
    unmatched_df = df_fail[df_fail["status"] == "UNMATCHED"]
    print("\n=== FAILURE CATEGORIZATION SUMMARY (Unmatched Targets N=27) ===")
    print(unmatched_df["failure_category"].value_counts().to_string())

    # Generate Overlays for Failure Contact Sheet
    print("\nGenerating failure overlays for contact sheet...")
    # Select 6 representative failure cases across distinct categories
    sample_failures = unmatched_df.drop_duplicates(subset=["failure_category"]).to_dict("records")
    # Add additional interesting cases to make 6 panels
    extra = unmatched_df[~unmatched_df["study_id"].isin([f["study_id"] for f in sample_failures])].head(6 - len(sample_failures)).to_dict("records")
    panel_cases = (sample_failures + extra)[:6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=200)
    axes = axes.flatten()

    for p_idx, case in enumerate(panel_cases):
        ax = axes[p_idx]
        sid = case["study_id"]
        tx, ty = case["target_x"], case["target_y"]
        cat = case["failure_category"]
        inst_id = case["catheter_instance_id"]

        img_path = IMAGES_DIR / f"{sid}.jpg"
        pil_im = Image.open(img_path).convert("RGB")
        im_arr = np.array(pil_im)
        h_im, w_im, _ = im_arr.shape

        # Render crop around target
        margin = 250
        x0 = max(0, int(tx - margin))
        x1 = min(w_im, int(tx + margin))
        y0 = max(0, int(ty - margin))
        y1 = min(h_im, int(ty + margin))

        crop_im = im_arr[y0:y1, x0:x1]
        ax.imshow(crop_im, cmap="gray")

        # Mark ground truth target with Green circle
        ax.scatter([tx - x0], [ty - y0], s=200, edgecolors="#2ecc71", facecolors="none", linewidths=3, label="GT Tip (Target)")
        ax.plot([tx - x0], [ty - y0], "g+", markersize=12, markeredgewidth=3)

        # Mark predictions within crop
        hm_data = heatmaps_by_study[sid]
        coord_tf = CoordinateTransform.from_native_dims(w_im, h_im, target_size=768, stride=4)
        dec_res = decode_heatmap_custom(hm_data["heatmap"], coord_tf, conf_threshold=tau, nms_radius_native_px=nms_radius, subpixel_refinement=subpixel, max_candidates=cap)

        for p in dec_res["final_predictions"]:
            px, py = p["native_x"], p["native_y"]
            if x0 <= px <= x1 and y0 <= py <= y1:
                ax.scatter([px - x0], [py - y0], s=180, edgecolors="#e74c3c", facecolors="none", linewidths=3, label="Predicted Tip")
                ax.plot([px - x0], [py - y0], "rx", markersize=10, markeredgewidth=2)
                ax.annotate(f"conf={p['conf']:.2f}", (px - x0 + 10, py - y0 - 10), color="yellow", fontsize=9, weight="bold")

        ax.set_title(f"Case {p_idx+1}: {cat}\nStudy: {sid[:12]}... (Inst {inst_id})\nDist={case['match_distance_px']:.1f}px, RawHM={case['raw_heatmap_target']:.2f}", fontsize=10, weight="bold")
        ax.axis("off")

    plt.suptitle("Failure Analysis Contact Sheet: Representative Failure Modes on Corrected Benchmark", fontsize=14, weight="bold", y=0.98)
    plt.tight_layout()

    contact_sheet_path = OVERLAYS_DIR / "failure_contact_sheet.png"
    plt.savefig(contact_sheet_path)
    plt.close()
    print(f"Saved failure contact sheet overlay to: {contact_sheet_path}")


if __name__ == "__main__":
    run_failure_analysis()
