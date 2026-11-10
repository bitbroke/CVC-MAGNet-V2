"""
Inference-free decoder grid comparison:
1. Performs inference once per checkpoint (Seed 42 and Seed 43) and caches heatmaps.
2. Evaluates the 24-configuration grid:
   - Confidence threshold: 0.10, 0.20, 0.30, 0.40
   - NMS radius native px: 10, 20, 30
   - Coordinate decoding: Grid maximum vs Local subpixel refinement
3. Tests both capped (max 10) and uncapped diagnostics.
4. Outputs: decoder_comparison.csv and candidate_filter_traces.csv.
"""
import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import scipy.ndimage as ndimage
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn

import sys

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXP_V1 = PROJECT_ROOT / "experiments" / "hrnet_tip_pilot_v1"
if str(EXP_V1) not in sys.path:
    sys.path.insert(0, str(EXP_V1))

EXP_V2 = PROJECT_ROOT / "experiments" / "hrnet_tip_improvement_v2"
IMAGES_DIR = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")
HEATMAP_CACHE_DIR = EXP_V2 / "cached_heatmaps"
HEATMAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

from experiments.hrnet_tip_pilot_v1.src.geometry import CoordinateTransform, heatmap_point_to_native
from experiments.hrnet_tip_pilot_v1.src.models.hrnet_joint_tip import HRNetJointTipNet
from src.data.transforms import LetterboxTransform


def cache_checkpoint_heatmaps(ckpt_path: Path, seed_name: str, device: torch.device, manifest: Dict[str, Any]):
    cache_file = HEATMAP_CACHE_DIR / f"{seed_name}_heatmaps.npz"
    if cache_file.exists():
        print(f"Loading cached heatmaps for {seed_name} from {cache_file}...")
        npz = np.load(cache_file, allow_pickle=True)
        results = {}
        for sid in npz.files:
            item = npz[sid].item()
            results[sid] = item
        return results

    print(f"Extracting and caching heatmaps for {seed_name} from {ckpt_path}...")
    model = HRNetJointTipNet(pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    transform = LetterboxTransform(target_size=(768, 768), clahe_enabled=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    results = {}
    save_dict = {}

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

            data_item = {
                "heatmap": tip_probs,
                "native_w": orig_w,
                "native_h": orig_h,
                "pad_left": coord_tf.padding_left,
                "pad_top": coord_tf.padding_top,
                "resized_w": coord_tf.resized_width,
                "resized_h": coord_tf.resized_height,
                "stride": 4,
            }
            results[sid] = data_item
            save_dict[sid] = data_item

    np.savez_compressed(cache_file, **save_dict)
    print(f"Cached {len(save_dict)} heatmaps to {cache_file}")
    return results


def subpixel_refine_peak(hm: np.ndarray, u: int, v: int) -> Tuple[float, float]:
    """Refine integer peak (u, v) using 1D quadratic interpolation on 3x3 neighborhood.
    Returns (v_sub, u_sub) in heatmap coordinates.
    Never averages distinct peaks together.
    """
    H, W = hm.shape
    # If on boundary, return integer coordinates
    if u <= 0 or u >= H - 1 or v <= 0 or v >= W - 1:
        return float(v), float(u)

    # 1D Quadratic fitting along x (columns)
    val_c = float(hm[u, v])
    val_l = float(hm[u, v - 1])
    val_r = float(hm[u, v + 1])
    denom_x = 2.0 * (2.0 * val_c - val_l - val_r)
    if abs(denom_x) > 1e-6:
        dx = (val_r - val_l) / denom_x
        dx = max(-0.5, min(0.5, dx))
    else:
        dx = 0.0

    # 1D Quadratic fitting along y (rows)
    val_u = float(hm[u - 1, v])
    val_d = float(hm[u + 1, v])
    denom_y = 2.0 * (2.0 * val_c - val_u - val_d)
    if abs(denom_y) > 1e-6:
        dy = (val_d - val_u) / denom_y
        dy = max(-0.5, min(0.5, dy))
    else:
        dy = 0.0

    return float(v + dx), float(u + dy)


def decode_heatmap_custom(
    hm: np.ndarray,
    coord_tf: CoordinateTransform,
    conf_threshold: float,
    nms_radius_native_px: float,
    subpixel_refinement: bool = False,
    max_candidates: Optional[int] = 10,
    kernel_size: int = 3,
) -> Dict[str, Any]:
    h, w = hm.shape

    pad_l_hm = coord_tf.padding_left / coord_tf.stride
    pad_t_hm = coord_tf.padding_top / coord_tf.stride
    res_w_hm = coord_tf.resized_width / coord_tf.stride
    res_h_hm = coord_tf.resized_height / coord_tf.stride

    # Local maximum filter
    max_filt = ndimage.maximum_filter(hm, size=kernel_size, mode="constant", cval=0.0)
    is_peak = (hm == max_filt)

    # All local peaks before thresholding
    all_peak_coords = np.argwhere(is_peak)
    valid_peaks_before_thresh = 0
    for u, v in all_peak_coords:
        if pad_t_hm <= u < (pad_t_hm + res_h_hm) and pad_l_hm <= v < (pad_l_hm + res_w_hm):
            valid_peaks_before_thresh += 1

    # Filter by threshold
    is_thresh_peak = is_peak & (hm >= conf_threshold)
    peak_coords = np.argwhere(is_thresh_peak)

    peak_candidates = []
    for u, v in peak_coords:
        if u < pad_t_hm or u >= (pad_t_hm + res_h_hm) or v < pad_l_hm or v >= (pad_l_hm + res_w_hm):
            continue

        conf_val = float(hm[u, v])
        if subpixel_refinement:
            v_sub, u_sub = subpixel_refine_peak(hm, int(u), int(v))
            nx, ny = heatmap_point_to_native(v_sub, u_sub, coord_tf)
            grid_v, grid_u = v_sub, u_sub
        else:
            nx, ny = heatmap_point_to_native(float(v), float(u), coord_tf)
            grid_v, grid_u = float(v), float(u)

        peak_candidates.append({
            "conf": conf_val,
            "native_x": nx,
            "native_y": ny,
            "grid_x": grid_v,
            "grid_y": grid_u,
            "raw_u": int(u),
            "raw_v": int(v),
        })

    # Sort descending by confidence
    peak_candidates.sort(key=lambda c: c["conf"], reverse=True)
    count_after_thresh = len(peak_candidates)

    # Native distance NMS
    nms_candidates = []
    suppressed = np.zeros(len(peak_candidates), dtype=bool)

    for i, cand in enumerate(peak_candidates):
        if suppressed[i]:
            continue
        nms_candidates.append(cand)
        cx, cy = cand["native_x"], cand["native_y"]

        for j in range(i + 1, len(peak_candidates)):
            if not suppressed[j]:
                ox, oy = peak_candidates[j]["native_x"], peak_candidates[j]["native_y"]
                d = math.hypot(cx - ox, cy - oy)
                if d <= nms_radius_native_px:
                    suppressed[j] = True

    count_after_nms = len(nms_candidates)

    # Cap to max_candidates
    if max_candidates is not None and max_candidates > 0:
        final_candidates = nms_candidates[:max_candidates]
    else:
        final_candidates = nms_candidates

    count_final = len(final_candidates)

    return {
        "final_predictions": final_candidates,
        "counts": {
            "peaks_before_thresh": valid_peaks_before_thresh,
            "after_thresh": count_after_thresh,
            "after_nms": count_after_nms,
            "final_count": count_final,
            "suppressed_by_thresh": valid_peaks_before_thresh - count_after_thresh,
            "suppressed_by_nms": count_after_thresh - count_after_nms,
            "suppressed_by_cap": count_after_nms - count_final,
        }
    }


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


def evaluate_grid(
    heatmaps_by_study: Dict[str, Any],
    bm_v2_df: pd.DataFrame,
    seed_name: str,
    thresholds: List[float],
    nms_radii: List[float],
    decoding_modes: List[str],
    cap_values: List[Optional[int]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # Group targets by study
    targets_by_study = {}
    for sid, group in bm_v2_df.groupby("study_id"):
        targets_by_study[sid] = group.to_dict("records")

    studies_list = list(heatmaps_by_study.keys())
    total_studies = len(studies_list)
    total_targets = len(bm_v2_df)
    neg_studies = [sid for sid in studies_list if len(targets_by_study.get(sid, [])) == 0]

    comparison_rows = []
    filter_trace_rows = []

    for tau in thresholds:
        for r_nms in nms_radii:
            for mode in decoding_modes:
                for cap in cap_values:
                    subpixel = (mode == "local_subpixel_refinement")
                    cap_label = "uncapped" if cap is None else f"cap_{cap}"
                    config_id = f"tau{tau:.2f}_nms{int(r_nms)}px_{mode}_{cap_label}"

                    total_m5, total_m10, total_m15, total_m25 = 0, 0, 0, 0
                    total_fp = 0
                    neg_fp = 0
                    single_t, single_m15 = 0, 0
                    multi_t, multi_m15 = 0, 0
                    rev_t, rev_m15 = 0, 0
                    weak_t, weak_m15 = 0, 0

                    study_preds_map = {}

                    for sid in studies_list:
                        hm_data = heatmaps_by_study[sid]
                        orig_w, orig_h = hm_data["native_w"], hm_data["native_h"]
                        coord_tf = CoordinateTransform.from_native_dims(orig_w, orig_h, target_size=768, stride=4)

                        dec_res = decode_heatmap_custom(
                            hm=hm_data["heatmap"],
                            coord_tf=coord_tf,
                            conf_threshold=tau,
                            nms_radius_native_px=r_nms,
                            subpixel_refinement=subpixel,
                            max_candidates=cap,
                        )
                        preds = dec_res["final_predictions"]
                        study_preds_map[sid] = preds

                        counts = dec_res["counts"]
                        filter_trace_rows.append({
                            "seed": seed_name,
                            "config_id": config_id,
                            "threshold": tau,
                            "nms_radius": r_nms,
                            "mode": mode,
                            "cap": cap_label,
                            "study_id": sid,
                            "peaks_before_thresh": counts["peaks_before_thresh"],
                            "after_thresh": counts["after_thresh"],
                            "after_nms": counts["after_nms"],
                            "final_count": counts["final_count"],
                            "suppressed_by_thresh": counts["suppressed_by_thresh"],
                            "suppressed_by_nms": counts["suppressed_by_nms"],
                            "suppressed_by_cap": counts["suppressed_by_cap"],
                        })

                        # Target matching
                        t_records = targets_by_study.get(sid, [])
                        targets = [(float(t["x"]), float(t["y"])) for t in t_records]
                        num_t = len(targets)

                        m5 = match_study_tips_constrained(targets, preds, tolerance_px=5.0)
                        m10 = match_study_tips_constrained(targets, preds, tolerance_px=10.0)
                        m15 = match_study_tips_constrained(targets, preds, tolerance_px=15.0)
                        m25 = match_study_tips_constrained(targets, preds, tolerance_px=25.0)

                        total_m5 += len(m5["matches"])
                        total_m10 += len(m10["matches"])
                        total_m15 += len(m15["matches"])
                        total_m25 += len(m25["matches"])

                        fp = len(m15["unmatched_pred_indices"])
                        total_fp += fp
                        if len(targets) == 0:
                            neg_fp += fp

                        # Single vs multi breakdown
                        if num_t == 1:
                            single_t += 1
                            single_m15 += len(m15["matches"])
                        elif num_t > 1:
                            multi_t += num_t
                            multi_m15 += len(m15["matches"])

                        # Provenance breakdown
                        for t_idx, t_item in enumerate(t_records):
                            stat = t_item.get("review_status", "")
                            is_matched = any(m[0] == t_idx for m in m15["matches"])
                            if stat == "provisional_engineering_review_v3":
                                rev_t += 1
                                if is_matched:
                                    rev_m15 += 1
                            else:
                                weak_t += 1
                                if is_matched:
                                    weak_m15 += 1

                    row = {
                        "seed": seed_name,
                        "config_id": config_id,
                        "threshold": tau,
                        "nms_radius": r_nms,
                        "coordinate_decoding": mode,
                        "candidate_cap": cap_label,
                        "success_at_5px": total_m5 / total_targets,
                        "success_at_10px": total_m10 / total_targets,
                        "success_at_15px": total_m15 / total_targets,
                        "success_at_25px": total_m25 / total_targets,
                        "matches_15px": total_m15,
                        "total_targets": total_targets,
                        "missed_targets": total_targets - total_m15,
                        "mean_false_tips_per_image": total_fp / total_studies,
                        "mean_false_tips_negative_image": neg_fp / max(len(neg_studies), 1),
                        "total_false_tips": total_fp,
                        "single_cvc_success_15px": single_m15 / max(single_t, 1),
                        "single_cvc_matches": single_m15,
                        "single_cvc_targets": single_t,
                        "multi_cvc_success_15px": multi_m15 / max(multi_t, 1),
                        "multi_cvc_matches": multi_m15,
                        "multi_cvc_targets": multi_t,
                        "reviewed_targets_success_15px": rev_m15 / max(rev_t, 1),
                        "reviewed_matches": rev_m15,
                        "reviewed_targets": rev_t,
                        "upgraded_v2_success_15px": weak_m15 / max(weak_t, 1),
                        "upgraded_v2_matches": weak_m15,
                        "upgraded_v2_targets": weak_t,
                    }
                    comparison_rows.append(row)

    return comparison_rows, filter_trace_rows


def run_comparison():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Decoder Grid Comparison on {device}...")

    # Load benchmark v2
    bm_v2_df = pd.read_csv(EXP_V2 / "benchmark_v2.csv")
    with open(EXP_V1 / "manifests" / "development_benchmark_manifest.json") as f:
        manifest = json.load(f)

    # 1. Cache heatmaps for Seed 42 and Seed 43
    s42_ckpt = EXP_V1 / "runs" / "seed_42" / "t1_joint_hrnet" / "checkpoints" / "final_model.pt"
    s43_ckpt = EXP_V1 / "runs" / "seed_43" / "t1_joint_hrnet" / "checkpoints" / "final_model.pt"

    hm_s42 = cache_checkpoint_heatmaps(s42_ckpt, "seed_42", device, manifest)
    hm_s43 = cache_checkpoint_heatmaps(s43_ckpt, "seed_43", device, manifest)

    # 2. Grid parameters
    thresholds = [0.10, 0.20, 0.30, 0.40]
    nms_radii = [10.0, 20.0, 30.0]
    modes = ["grid_maximum", "local_subpixel_refinement"]
    caps = [10, None]  # Include uncapped diagnostic

    print(f"Evaluating grid on Seed 42 ({len(thresholds)*len(nms_radii)*len(modes)*len(caps)} configurations)...")
    comp_s42, trace_s42 = evaluate_grid(
        hm_s42, bm_v2_df, "Seed_42", thresholds, nms_radii, modes, caps
    )

    print(f"Evaluating grid on Seed 43 ({len(thresholds)*len(nms_radii)*len(modes)*len(caps)} configurations)...")
    comp_s43, trace_s43 = evaluate_grid(
        hm_s43, bm_v2_df, "Seed_43", thresholds, nms_radii, modes, caps
    )

    df_comp = pd.DataFrame(comp_s42 + comp_s43)
    df_comp.to_csv(EXP_V2 / "decoder_comparison.csv", index=False)
    print(f"Saved decoder_comparison.csv ({len(df_comp)} rows)")

    df_trace = pd.DataFrame(trace_s42 + trace_s43)
    df_trace.to_csv(EXP_V2 / "candidate_filter_traces.csv", index=False)
    print(f"Saved candidate_filter_traces.csv ({len(df_trace)} rows)")

    # Print top Seed 42 configurations (capped at 10)
    df_s42_capped = df_comp[(df_comp["seed"] == "Seed_42") & (df_comp["candidate_cap"] == "cap_10")]
    print("\n=== TOP 10 CONFIGURATIONS ON SEED 42 (Capped at 10) ===")
    top_s42 = df_s42_capped.sort_values(by=["success_at_15px", "mean_false_tips_per_image"], ascending=[False, True])
    print(top_s42[["config_id", "threshold", "nms_radius", "coordinate_decoding", "success_at_15px", "matches_15px", "mean_false_tips_per_image", "mean_false_tips_negative_image", "multi_cvc_matches"]].head(10).to_string())


if __name__ == "__main__":
    run_comparison()
