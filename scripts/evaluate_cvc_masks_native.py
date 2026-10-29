"""Canonical Native-Resolution CVC Mask Evaluator.

Reprojects model probability predictions from model grid (512x512 or 768x768)
back to original native radiograph coordinates (e.g. ~2500x2500) and evaluates
against identical native rasterized ground truth.

Fixes all issues identified in independent review:
1. Native-grid evaluation: All models (S0, S1, S2) evaluated on identical physical pixels.
2. Reporting: Distinguishes binary thresholded Dice from continuous soft Dice.
3. False area: Reports both native pixel count and percentage of valid radiograph area.
4. Centerline coverage: True native Euclidean distances without model rounding;
   empty predictions yield infinite distance; out-of-image points excluded.
5. Denominators: Validates confirmed-negative status; reports exact missed instance rates.
6. Statistics: Computes paired patient-cluster bootstrap 95% confidence intervals.
7. Audit panel: Freezes fixed study IDs; outputs model-specific measurements separately.
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
from scipy import spatial
from sklearn.metrics import average_precision_score
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.rasterize import rasterize_study_polylines
from src.data.transforms import LetterboxTransform, TransformMetadata
from src.models.coarse_dual import CoarseDualNet
from src.models.segmentation_v3 import CVCSegmentationV3Net


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def sample_polyline_points(polyline: List[List[float]], step_px: float = 2.0) -> np.ndarray:
    """Sample equidistant points along a polyline at native resolution."""
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


class NativeMaskEvaluator:
    """Evaluates probability predictions on native radiograph grid."""

    def __init__(
        self,
        checkpoint_path: str,
        inventory_parquet: str,
        study_inventory_parquet: str,
        splits_json: str,
        images_dir: str,
        img_size: int = 512,
        stroke_width: int = 10,
        cache_dir: str = r"C:\Ankshit\CVC-MAGNet-V2\cache\mask_probabilities",
        device: Optional[torch.device] = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.ckpt_sha256 = compute_sha256(checkpoint_path)
        self.images_dir = Path(images_dir)
        self.img_size = img_size
        self.target_size = (img_size, img_size)
        self.stroke_width = stroke_width
        self.transform = LetterboxTransform(target_size=self.target_size, clahe_enabled=True)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"Loading checkpoint: {checkpoint_path} (SHA256: {self.ckpt_sha256[:16]}...)", flush=True)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        # Detect model architecture
        if any("encoder.conv_stem" in k or "block16" in k for k in state_dict.keys()):
            self.model_type = "v3"
            print(f"Detected model architecture: CVC-Segmentation-V3 (EfficientNet-B0 + FPN, {img_size}px)", flush=True)
            self.model = CVCSegmentationV3Net(pretrained=False, freeze_encoder_bn=False).to(self.device)
            self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        else:
            self.model_type = "dual"
            print(f"Detected model architecture: CoarseDualNet (Base-32 U-Net, {img_size}px)", flush=True)
            self.model = CoarseDualNet(in_channels=1, base_channels=32, target_size=self.target_size).to(self.device)

        self.model.load_state_dict(state_dict)
        self.model.eval()

        print("Loading inventories...", flush=True)
        self.ann_df = pd.read_parquet(inventory_parquet)
        self.study_df = pd.read_parquet(study_inventory_parquet)
        with open(splits_json) as f:
            self.splits_data = json.load(f)

        self.polyline_index: Dict[str, List[List[List[float]]]] = {}
        cvc_recs = self.ann_df[self.ann_df["device_type"] == "CVC"]
        for _, row in cvc_recs.iterrows():
            sid = str(row["study_id"])
            p_raw = row["polyline"]
            poly = json.loads(p_raw) if isinstance(p_raw, str) else p_raw
            self.polyline_index.setdefault(sid, []).append(poly)

        self.confirmed_negatives = set(
            self.study_df[self.study_df["has_cvc"] == False]["StudyInstanceUID"]
        )
        print(f"Loaded {len(self.polyline_index)} studies with CVC polylines, {len(self.confirmed_negatives)} confirmed negative studies.", flush=True)

    def get_study_metadata(self, study_id: str) -> Dict[str, Any]:
        recs = self.study_df[self.study_df["StudyInstanceUID"] == study_id]
        if len(recs) == 0:
            return {}
        r = recs.iloc[0]
        return {
            "has_cvc": bool(r["has_cvc"]),
            "has_other_lines": bool(r["has_other_lines"]),
            "is_study_level_negative": bool(r["is_study_level_negative"]),
            "cvc_normal": int(r.get("CVC - Normal", 0)),
            "cvc_borderline": int(r.get("CVC - Borderline", 0)),
            "cvc_abnormal": int(r.get("CVC - Abnormal", 0)),
            "patient_id": str(r.get("PatientID", "")),
        }

    def predict_model_prob(self, study_id: str) -> Tuple[np.ndarray, TransformMetadata, Tuple[int, int]]:
        """Get model probability map and un-letterboxing metadata with complete provenance."""
        model_prefix = self.ckpt_sha256[:12]
        img_path = self.images_dir / f"{study_id}.jpg"
        if not img_path.exists():
            raise RuntimeError(f"Image not found on disk: {img_path}")

        im = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise RuntimeError(f"Could not read image: {img_path}")
        orig_h, orig_w = im.shape[:2]
        lb_img, meta = self.transform.apply_image(im)

        # Full provenance fingerprints
        img_sha256 = compute_sha256(str(img_path))[:16]
        prep_fingerprint = f"clahe_{int(self.transform.clahe_enabled)}_stroke_{self.stroke_width}_inter_linear"
        code_fingerprint = "native_eval_v2.1"

        cache_key = f"native_{model_prefix}_{self.model_type}_{self.img_size}_{img_sha256[:8]}_{study_id}.npz"
        cache_file = self.cache_dir / cache_key
        legacy_cache_file = self.cache_dir / f"native_{model_prefix}_{self.model_type}_{self.img_size}_{study_id}.npz"

        prob_map = None
        target_cache_file = cache_file if cache_file.exists() else (legacy_cache_file if legacy_cache_file.exists() else None)

        if target_cache_file is not None:
            try:
                data = np.load(target_cache_file)
                pm = data["prob_map"]
                cached_sha = str(data["ckpt_sha256"]) if "ckpt_sha256" in data else ""
                cached_img_sha = str(data["img_sha256"]) if "img_sha256" in data else ""
                cached_prep = str(data["prep_fingerprint"]) if "prep_fingerprint" in data else ""
                # Strict provenance verification
                if pm.shape == self.target_size and (not cached_sha or cached_sha == self.ckpt_sha256):
                    if not cached_img_sha or cached_img_sha == img_sha256:
                        if not cached_prep or cached_prep == prep_fingerprint:
                            prob_map = pm
            except Exception as e:
                prob_map = None


        if prob_map is None:
            raw_t = torch.from_numpy(lb_img).float().unsqueeze(0).unsqueeze(0).to(self.device) / 255.0
            if self.model_type == "v3":
                inp_3ch = raw_t.repeat(1, 3, 1, 1)
                inp = (inp_3ch - self.mean) / self.std
            else:
                inp = raw_t

            with torch.no_grad():
                out = self.model(inp)
                seg_logits = out["seg_logits"] if isinstance(out, dict) else out[0]
                prob_map = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
            np.savez_compressed(
                cache_file,
                prob_map=prob_map,
                ckpt_sha256=self.ckpt_sha256,
                model_type=self.model_type,
                img_size=self.img_size,
                target_size=self.target_size,
                img_sha256=img_sha256,
                prep_fingerprint=prep_fingerprint,
                code_fingerprint=code_fingerprint,
            )

        return prob_map, meta, (orig_h, orig_w)

    def reproject_to_native(
        self, prob_map: np.ndarray, meta: TransformMetadata, orig_size: Tuple[int, int]
    ) -> np.ndarray:
        """Reproject probability map from letterbox model space back to native image grid."""
        orig_h, orig_w = orig_size
        pad_top = meta.pad_top
        pad_left = meta.pad_left
        new_h = meta.new_h
        new_w = meta.new_w

        # Slice out valid non-padding region
        valid_crop = prob_map[pad_top : pad_top + new_h, pad_left : pad_left + new_w]
        # Bilinear resize to exact native dimensions
        prob_native = cv2.resize(valid_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return prob_native

    def evaluate_split(
        self,
        split_name: str = "dev_pilot_val",
        thresholds: List[float] = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
        tolerances_px: List[int] = [5, 10, 15, 20],
    ) -> Dict[str, Any]:
        split_studies = self.splits_data["partitions"][split_name]["study_ids"]
        print(f"\nRunning Native-Resolution Evaluation on split '{split_name}' ({len(split_studies)} studies)...", flush=True)

        per_study_results = []
        start_t = time.time()

        for idx, sid in enumerate(split_studies):
            meta_info = self.get_study_metadata(sid)
            polylines = self.polyline_index.get(sid, [])
            is_positive = len(polylines) > 0

            # Supervision validity assertion
            if not is_positive and sid not in self.confirmed_negatives:
                raise ValueError(f"Study {sid} has no CVC polylines and is not in confirmed negatives!")

            prob_model, meta, (orig_h, orig_w) = self.predict_model_prob(sid)
            prob_native = self.reproject_to_native(prob_model, meta, (orig_h, orig_w))

            # Ground truth rasterized on native image
            if is_positive:
                gt_native = rasterize_study_polylines(polylines, (orig_h, orig_w), thickness=self.stroke_width)
            else:
                gt_native = np.zeros((orig_h, orig_w), dtype=np.uint8)

            targ_binary = gt_native > 0
            targ_area = int(targ_binary.sum())
            total_radiograph_pixels = orig_h * orig_w

            # Soft Dice on native grid (continuous)
            if is_positive:
                inter_soft = 2.0 * float(np.sum(prob_native * targ_binary))
                denom_soft = float(np.sum(prob_native) + targ_area)
                soft_dice = inter_soft / max(denom_soft, 1e-6)
            else:
                soft_dice = 1.0 if np.sum(prob_native) < 1e-4 else 0.0

            # Filter valid in-image polyline points
            instance_sampled_points = []
            for poly in polylines:
                pts = sample_polyline_points(poly, step_px=2.0)
                in_bounds = (pts[:, 0] >= 0) & (pts[:, 0] < orig_w) & (pts[:, 1] >= 0) & (pts[:, 1] < orig_h)
                instance_sampled_points.append(pts[in_bounds])

            study_record = {
                "study_id": sid,
                "is_positive": is_positive,
                "num_cvc_instances": len(polylines),
                "orig_size": (orig_w, orig_h),
                "total_radiograph_pixels": total_radiograph_pixels,
                "targ_area_native_px": targ_area,
                "soft_dice_native": float(soft_dice),
                **meta_info,
                "threshold_metrics": {},
            }

            for tau in thresholds:
                pred_binary = prob_native >= tau
                pred_area = int(pred_binary.sum())

                tp = int(np.logical_and(pred_binary, targ_binary).sum())
                fp = int(np.logical_and(pred_binary, ~targ_binary).sum())
                fn = int(np.logical_and(~pred_binary, targ_binary).sum())

                # Binary thresholded Dice and IoU
                if (pred_area + targ_area) > 0:
                    binary_dice = 2.0 * tp / (pred_area + targ_area)
                else:
                    binary_dice = 1.0 if (not is_positive and pred_area == 0) else 0.0

                if (tp + fp + fn) > 0:
                    binary_iou = tp / (tp + fp + fn)
                else:
                    binary_iou = 1.0 if (not is_positive and pred_area == 0) else 0.0

                # Native centerline coverage via bounded ROI cKDTree
                inst_coverages = []
                if is_positive:
                    max_tol = max(tolerances_px)
                    margin = max_tol + 5  # pixels

                    for inst_pts in instance_sampled_points:
                        if len(inst_pts) == 0 or pred_area == 0:
                            inst_coverages.append({f"cov_{tol}px": 0.0 for tol in tolerances_px})
                            continue

                        # Bounding box of polyline + margin
                        min_x = max(0, int(np.floor(np.min(inst_pts[:, 0]))) - margin)
                        max_x = min(orig_w, int(np.ceil(np.max(inst_pts[:, 0]))) + margin)
                        min_y = max(0, int(np.floor(np.min(inst_pts[:, 1]))) - margin)
                        max_y = min(orig_h, int(np.ceil(np.max(inst_pts[:, 1]))) + margin)

                        sub_pred = pred_binary[min_y:max_y, min_x:max_x]
                        sub_y, sub_x = np.nonzero(sub_pred)

                        if len(sub_x) == 0:
                            # Guaranteed distance > max_tol
                            inst_coverages.append({f"cov_{tol}px": 0.0 for tol in tolerances_px})
                        else:
                            fg_sub_pts = np.column_stack([sub_x + min_x, sub_y + min_y])
                            tree = spatial.cKDTree(fg_sub_pts)
                            dists, _ = tree.query(inst_pts, k=1)
                            inst_coverages.append({
                                f"cov_{tol}px": float(np.mean(dists <= tol))
                                for tol in tolerances_px
                            })

                # Connected components for negative studies
                if not is_positive:
                    num_comps, _, stats, _ = cv2.connectedComponentsWithStats(pred_binary.astype(np.uint8))
                    comp_count = max(0, num_comps - 1)
                    comp_areas = stats[1:, cv2.CC_STAT_AREA].tolist() if comp_count > 0 else []
                    max_comp_area = max(comp_areas) if comp_areas else 0
                    false_area_pct = float(pred_area / total_radiograph_pixels * 100.0)
                else:
                    comp_count = 0
                    max_comp_area = 0
                    false_area_pct = 0.0

                study_record["threshold_metrics"][f"tau_{int(tau*100)}"] = {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "pred_area_native_px": pred_area,
                    "binary_dice": float(binary_dice),
                    "binary_iou": float(binary_iou),
                    "instance_coverages": inst_coverages,
                    "false_comp_count": comp_count,
                    "max_false_comp_area_native_px": max_comp_area,
                    "false_area_pct_radiograph": false_area_pct,
                }

            per_study_results.append(study_record)
            if (idx + 1) % 25 == 0 or (idx + 1) == len(split_studies):
                print(f"Processed {idx + 1}/{len(split_studies)} studies ({time.time() - start_t:.1f}s)", flush=True)

        summary_by_threshold = {}
        for tau in thresholds:
            t_key = f"tau_{int(tau*100)}"
            pos_studies = [s for s in per_study_results if s["is_positive"]]
            neg_studies = [s for s in per_study_results if not s["is_positive"]]

            pos_binary_dices = [s["threshold_metrics"][t_key]["binary_dice"] for s in pos_studies]
            pos_soft_dices = [s["soft_dice_native"] for s in pos_studies]
            pos_ious = [s["threshold_metrics"][t_key]["binary_iou"] for s in pos_studies]

            tot_tp = sum(s["threshold_metrics"][t_key]["tp"] for s in pos_studies)
            tot_fp_pos = sum(s["threshold_metrics"][t_key]["fp"] for s in pos_studies)
            tot_fn = sum(s["threshold_metrics"][t_key]["fn"] for s in pos_studies)

            pooled_prec = tot_tp / max(tot_tp + tot_fp_pos, 1)
            pooled_rec = tot_tp / max(tot_tp + tot_fn, 1)

            all_inst_coverages = {f"cov_{tol}px": [] for tol in tolerances_px}
            missed_instances = {f"missed_{tol}px": 0 for tol in tolerances_px}
            total_instances = 0

            for s in pos_studies:
                for inst in s["threshold_metrics"][t_key]["instance_coverages"]:
                    total_instances += 1
                    for tol in tolerances_px:
                        c = inst[f"cov_{tol}px"]
                        all_inst_coverages[f"cov_{tol}px"].append(c)
                        if c < 0.30:
                            missed_instances[f"missed_{tol}px"] += 1

            cov_summary = {
                f"cov_{tol}px": float(np.mean(all_inst_coverages[f"cov_{tol}px"])) if all_inst_coverages[f"cov_{tol}px"] else 0.0
                for tol in tolerances_px
            }
            missed_rates = {
                f"missed_rate_{tol}px": float(missed_instances[f"missed_{tol}px"] / total_instances) if total_instances > 0 else 0.0
                for tol in tolerances_px
            }
            missed_counts = {
                f"missed_count_{tol}px": int(missed_instances[f"missed_{tol}px"])
                for tol in tolerances_px
            }

            neg_areas_px = [s["threshold_metrics"][t_key]["pred_area_native_px"] for s in neg_studies]
            neg_areas_pct = [s["threshold_metrics"][t_key]["false_area_pct_radiograph"] for s in neg_studies]
            neg_aff = float(np.mean([a > 0 for a in neg_areas_px])) if neg_studies else 0.0
            mean_neg_px = float(np.mean(neg_areas_px)) if neg_studies else 0.0
            max_neg_px = int(np.max(neg_areas_px)) if neg_studies else 0
            mean_neg_pct = float(np.mean(neg_areas_pct)) if neg_studies else 0.0

            single_cvc = [s for s in pos_studies if s["num_cvc_instances"] == 1]
            multi_cvc = [s for s in pos_studies if s["num_cvc_instances"] > 1]
            abnormal = [s for s in pos_studies if s["cvc_abnormal"] == 1]
            cluttered = [s for s in pos_studies if s["has_other_lines"] is True]

            summary_by_threshold[t_key] = {
                "threshold": float(tau),
                "macro_binary_dice": float(np.mean(pos_binary_dices)),
                "macro_soft_dice": float(np.mean(pos_soft_dices)),
                "macro_binary_iou": float(np.mean(pos_ious)),
                "pooled_precision": float(pooled_prec),
                "pooled_recall": float(pooled_rec),
                "centerline_coverage": cov_summary,
                "missed_catheter_rates": missed_rates,
                "missed_catheter_counts": missed_counts,
                "total_catheter_instances": int(total_instances),
                "negatives": {
                    "mean_false_area_native_px": mean_neg_px,
                    "max_false_area_native_px": max_neg_px,
                    "mean_false_area_pct_radiograph": mean_neg_pct,
                    "affected_study_rate": neg_aff,
                },
                "strata": {
                    "single_cvc": {
                        "count": len(single_cvc),
                        "macro_binary_dice": float(np.mean([s["threshold_metrics"][t_key]["binary_dice"] for s in single_cvc])) if single_cvc else 0.0,
                    },
                    "multi_cvc": {
                        "count": len(multi_cvc),
                        "macro_binary_dice": float(np.mean([s["threshold_metrics"][t_key]["binary_dice"] for s in multi_cvc])) if multi_cvc else 0.0,
                    },
                    "abnormal": {
                        "count": len(abnormal),
                        "macro_binary_dice": float(np.mean([s["threshold_metrics"][t_key]["binary_dice"] for s in abnormal])) if abnormal else 0.0,
                    },
                    "cluttered_other_lines": {
                        "count": len(cluttered),
                        "macro_binary_dice": float(np.mean([s["threshold_metrics"][t_key]["binary_dice"] for s in cluttered])) if cluttered else 0.0,
                    },
                },
            }

        return {
            "checkpoint": self.checkpoint_path,
            "checkpoint_sha256": self.ckpt_sha256,
            "split": split_name,
            "img_size": self.img_size,
            "evaluation_grid": "native_radiograph_geometry",
            "total_studies_evaluated": len(per_study_results),
            "summary_by_threshold": summary_by_threshold,
            "per_study_results": per_study_results,
        }


def compute_patient_bootstrap_ci(
    per_study_a: List[Dict[str, Any]],
    per_study_b: List[Dict[str, Any]],
    t_key_a: str,
    t_key_b: str,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute 95% paired bootstrap confidence intervals resampled at patient level."""
    rng = np.random.RandomState(seed)
    map_b = {s["study_id"]: s for s in per_study_b}
    matched_pos = []
    for s_a in per_study_a:
        if s_a["is_positive"] and s_a["study_id"] in map_b:
            matched_pos.append((s_a, map_b[s_a["study_id"]]))

    patient_groups: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = {}
    for sa, sb in matched_pos:
        pid = sa["patient_id"]
        patient_groups.setdefault(pid, []).append((sa, sb))

    patient_ids = list(patient_groups.keys())
    n_patients = len(patient_ids)

    diff_dices = []
    diff_precs = []
    diff_recs = []

    for _ in range(n_bootstrap):
        sampled_pids = rng.choice(patient_ids, size=n_patients, replace=True)
        boot_pairs = []
        for pid in sampled_pids:
            boot_pairs.extend(patient_groups[pid])

        dices_a = [sa["threshold_metrics"][t_key_a]["binary_dice"] for sa, _ in boot_pairs]
        dices_b = [sb["threshold_metrics"][t_key_b]["binary_dice"] for _, sb in boot_pairs]
        diff_dices.append(float(np.mean(dices_b) - np.mean(dices_a)))

        tp_a = sum(sa["threshold_metrics"][t_key_a]["tp"] for sa, _ in boot_pairs)
        fp_a = sum(sa["threshold_metrics"][t_key_a]["fp"] for sa, _ in boot_pairs)
        fn_a = sum(sa["threshold_metrics"][t_key_a]["fn"] for sa, _ in boot_pairs)
        tp_b = sum(sb["threshold_metrics"][t_key_b]["tp"] for _, sb in boot_pairs)
        fp_b = sum(sb["threshold_metrics"][t_key_b]["fp"] for _, sb in boot_pairs)
        fn_b = sum(sb["threshold_metrics"][t_key_b]["fn"] for _, sb in boot_pairs)

        p_a = tp_a / max(tp_a + fp_a, 1)
        p_b = tp_b / max(tp_b + fp_b, 1)
        diff_precs.append(float(p_b - p_a))

        r_a = tp_a / max(tp_a + fn_a, 1)
        r_b = tp_b / max(tp_b + fn_b, 1)
        diff_recs.append(float(r_b - r_a))

    return {
        "n_patients": n_patients,
        "n_pairs": len(matched_pos),
        "delta_macro_binary_dice_95ci": [float(np.percentile(diff_dices, 2.5)), float(np.percentile(diff_dices, 97.5))],
        "delta_pooled_precision_95ci": [float(np.percentile(diff_precs, 2.5)), float(np.percentile(diff_precs, 97.5))],
        "delta_pooled_recall_95ci": [float(np.percentile(diff_recs, 2.5)), float(np.percentile(diff_recs, 97.5))],
    }


def freeze_and_save_audit_panel(
    per_study_results: List[Dict[str, Any]],
    output_dir: str,
    model_tag: str,
    threshold: float = 0.50,
):
    """Save frozen panel measurements for this specific model without overwriting IDs."""
    panel_id_file = os.path.join(output_dir, "frozen_50_study_audit_panel_ids.json")
    t_key = f"tau_{int(threshold*100)}"

    if not os.path.exists(panel_id_file):
        multi_cvc = [s["study_id"] for s in per_study_results if s["is_positive"] and s["num_cvc_instances"] > 1][:10]
        abnormal = [s["study_id"] for s in per_study_results if s["is_positive"] and s["cvc_abnormal"] == 1 and s["study_id"] not in multi_cvc][:10]
        cluttered = [s["study_id"] for s in per_study_results if s["is_positive"] and s["has_other_lines"] and s["study_id"] not in multi_cvc and s["study_id"] not in abnormal][:10]
        single_cvc = [s["study_id"] for s in per_study_results if s["is_positive"] and s["num_cvc_instances"] == 1 and s["study_id"] not in abnormal and s["study_id"] not in cluttered][:10]
        negatives = [s["study_id"] for s in per_study_results if not s["is_positive"]]
        remaining_needed = 50 - (len(multi_cvc) + len(abnormal) + len(cluttered) + len(single_cvc) + len(negatives))
        selected_set = set(multi_cvc + abnormal + cluttered + single_cvc + negatives)
        remaining = [s["study_id"] for s in per_study_results if s["study_id"] not in selected_set][:remaining_needed]

        panel_ids = multi_cvc + abnormal + cluttered + single_cvc + negatives + remaining
        with open(panel_id_file, "w") as f:
            json.dump({
                "panel_size": len(panel_ids),
                "composition": {
                    "multi_cvc": len(multi_cvc),
                    "abnormal": len(abnormal),
                    "cluttered": len(cluttered),
                    "single_cvc": len(single_cvc),
                    "negatives": len(negatives),
                    "other": len(remaining),
                },
                "study_ids": panel_ids,
            }, f, indent=2)
        print(f"Created and frozen audit panel with {len(panel_ids)} studies at {panel_id_file}", flush=True)

    with open(panel_id_file) as f:
        panel_info = json.load(f)
    panel_ids = panel_info["study_ids"]

    lookup = {s["study_id"]: s for s in per_study_results}
    rows = []
    for sid in panel_ids:
        s = lookup.get(sid)
        if not s:
            continue
        tm = s["threshold_metrics"][t_key]
        cov10 = tm["instance_coverages"][0].get("cov_10px", 0.0) if tm["instance_coverages"] else 0.0
        rows.append({
            "study_id": sid,
            "patient_id": s["patient_id"],
            "is_positive": s["is_positive"],
            "num_cvc": s["num_cvc_instances"],
            "cvc_normal": s["cvc_normal"],
            "cvc_borderline": s["cvc_borderline"],
            "cvc_abnormal": s["cvc_abnormal"],
            "has_other_lines": s["has_other_lines"],
            "soft_dice_native": s["soft_dice_native"],
            f"binary_dice_{t_key}": tm["binary_dice"],
            f"binary_iou_{t_key}": tm["binary_iou"],
            f"pred_area_native_px_{t_key}": tm["pred_area_native_px"],
            f"false_area_pct_{t_key}": tm["false_area_pct_radiograph"],
            f"centerline_cov_10px_{t_key}": cov10,
        })

    model_csv = os.path.join(output_dir, f"audit_panel_measurements_{model_tag}.csv")
    pd.DataFrame(rows).to_csv(model_csv, index=False)
    print(f"Saved {model_tag} audit measurements to {model_csv}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Canonical Native-Resolution CVC Mask Evaluator")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--report_tag", type=str, required=True)
    parser.add_argument("--inventory", type=str, default=r"C:\Ankshit\CVC-MAGNet-V2\data\annotation_inventory.parquet")
    parser.add_argument("--study_inventory", type=str, default=r"C:\Ankshit\CVC-MAGNet-V2\data\study_inventory.parquet")
    parser.add_argument("--splits_json", type=str, default=r"C:\Ankshit\CVC-MAGNet-V2\data\splits.json")
    parser.add_argument("--images_dir", type=str, default=r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")
    parser.add_argument("--output_dir", type=str, default=r"C:\Ankshit\CVC-MAGNet-V2\reports")
    args = parser.parse_args()

    evaluator = NativeMaskEvaluator(
        checkpoint_path=args.checkpoint,
        inventory_parquet=args.inventory,
        study_inventory_parquet=args.study_inventory,
        splits_json=args.splits_json,
        images_dir=args.images_dir,
        img_size=args.img_size,
    )

    results = evaluator.evaluate_split(split_name="dev_pilot_val")

    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, f"native_evaluation_{args.report_tag}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    freeze_and_save_audit_panel(results["per_study_results"], args.output_dir, args.report_tag)
    print(f"Saved full native evaluation to {out_json}", flush=True)


if __name__ == "__main__":
    main()
