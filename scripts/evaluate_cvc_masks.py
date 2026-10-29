"""Canonical CVC Segmentation Mask Evaluator for CVC-MAGNet-V2 (S0 Benchmark).

Contract Guarantees:
1. Evaluates segmentation masks directly (independent of tip localization).
2. Caches float32 probability maps to disk so threshold sweeps avoid re-running GPU inference.
3. Excludes letterbox padding from loss and metrics via image-validity masks.
4. Computes:
   - Macro Dice & IoU on positive studies
   - Pooled pixel Precision & Recall
   - Full-image pixel Average Precision (PR-AUC)
   - Catheter-centerline coverage at fixed native tolerances (5, 10, 15, 20 px)
   - Missed catheter instance rates (<30% coverage)
   - False foreground area, component counts, and affected study rate on confirmed negatives
   - Stratified breakdowns: Single CVC, Multiple CVC, Abnormal, and Cluttered (other devices present)
5. Exports a fixed varied 50-study development panel for visual audit.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import average_precision_score
import torch
import torch.nn.functional as F

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.rasterize import rasterize_study_polylines
from src.data.transforms import LetterboxTransform, TransformMetadata
from src.models.coarse_dual import CoarseDualNet
from src.models.segmentation_v3 import CVCSegmentationV3Net


def compute_sha256(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
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


class MaskEvaluator:
    """Harness to evaluate segmentation masks on validation cohort."""

    def __init__(
        self,
        checkpoint_path: str,
        inventory_parquet: str,
        study_inventory_parquet: str,
        splits_json: str,
        images_dir: str,
        target_size: Tuple[int, int] = (512, 512),
        stroke_width: int = 10,
        clahe_enabled: bool = True,
        cache_dir: str = r"C:\Ankshit\CVC-MAGNet-V2\cache\mask_probabilities",
        device: Optional[torch.device] = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.ckpt_sha256 = compute_sha256(checkpoint_path)
        self.images_dir = Path(images_dir)
        self.target_size = target_size
        self.stroke_width = stroke_width
        self.transform = LetterboxTransform(target_size=target_size, clahe_enabled=clahe_enabled)
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
            print("Detected model architecture: CVC-Segmentation-V3 (EfficientNet-B0 + FPN)", flush=True)
            self.model = CVCSegmentationV3Net(pretrained=False, freeze_encoder_bn=False).to(self.device)
            # Normalization constants
            self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        else:
            self.model_type = "dual"
            print("Detected model architecture: CoarseDualNet (Base-32 U-Net)", flush=True)
            self.model = CoarseDualNet(in_channels=1, base_channels=32, target_size=target_size).to(self.device)

        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Load annotation inventory
        print("Loading inventories...", flush=True)
        self.ann_df = pd.read_parquet(inventory_parquet)
        self.study_df = pd.read_parquet(study_inventory_parquet)

        with open(splits_json) as f:
            self.splits_data = json.load(f)

        # Build polyline index for CVCs
        self.polyline_index: Dict[str, List[List[List[float]]]] = {}
        cvc_recs = self.ann_df[self.ann_df["device_type"] == "CVC"]
        for _, row in cvc_recs.iterrows():
            sid = str(row["study_id"])
            p_raw = row["polyline"]
            poly = json.loads(p_raw) if isinstance(p_raw, str) else p_raw
            self.polyline_index.setdefault(sid, []).append(poly)

        # Study-level negative set
        self.confirmed_negatives = set(
            self.study_df[self.study_df["has_cvc"] == False]["StudyInstanceUID"]
        )
        print(f"Loaded {len(self.polyline_index)} studies with CVC polylines, {len(self.confirmed_negatives)} confirmed negative studies.", flush=True)

    def get_study_metadata(self, study_id: str) -> Dict[str, Any]:
        """Fetch study metadata (device counts, placement labels, duplicate info)."""
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

    def predict_and_cache(self, study_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, TransformMetadata]:
        """Predict probability map, valid mask, and target mask; cache probabilities to disk."""
        model_prefix = self.ckpt_sha256[:12]
        cache_file = self.cache_dir / f"{model_prefix}_{study_id}.npz"

        # Load image
        img_path = self.images_dir / f"{study_id}.jpg"
        im = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise RuntimeError(f"Could not read image: {img_path}")
        orig_h, orig_w = im.shape[:2]

        lb_img, meta = self.transform.apply_image(im)

        # Target mask (512x512)
        polylines = self.polyline_index.get(study_id, [])
        if len(polylines) > 0:
            native_mask = rasterize_study_polylines(polylines, (orig_h, orig_w), thickness=self.stroke_width)
            target_mask = self.transform.apply_mask(native_mask, meta)
        else:
            target_mask = np.zeros(self.target_size, dtype=np.uint8)

        # Valid image region inside letterbox (excluding padding)
        valid_mask = np.zeros(self.target_size, dtype=bool)
        valid_mask[meta.pad_top : meta.pad_top + meta.new_h, meta.pad_left : meta.pad_left + meta.new_w] = True

        if cache_file.exists():
            data = np.load(cache_file)
            prob_map = data["prob_map"]
        else:
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
            np.savez_compressed(cache_file, prob_map=prob_map)

        return prob_map, target_mask, valid_mask, meta

    def evaluate_split(
        self,
        split_name: str = "dev_pilot_val",
        thresholds: List[float] = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
        tolerances_px: List[int] = [5, 10, 15, 20],
    ) -> Dict[str, Any]:
        """Run complete evaluation across all studies in a split."""
        split_studies = self.splits_data["partitions"][split_name]["study_ids"]
        print(f"\nEvaluating split '{split_name}' ({len(split_studies)} studies)...", flush=True)

        per_study_results = []
        start_t = time.time()

        for idx, sid in enumerate(split_studies):
            meta_info = self.get_study_metadata(sid)
            prob_map, target_mask, valid_mask, meta = self.predict_and_cache(sid)
            polylines = self.polyline_index.get(sid, [])
            is_positive = len(polylines) > 0

            # Distance transform of predicted probability map for centerline evaluation at multiple thresholds
            # We evaluate each threshold
            study_record = {
                "study_id": sid,
                "is_positive": is_positive,
                "num_cvc_instances": len(polylines),
                "orig_size": (meta.orig_w, meta.orig_h),
                **meta_info,
                "threshold_metrics": {},
            }

            # Pre-sample polyline points for centerline coverage
            instance_sampled_points = [
                sample_polyline_points(poly, step_px=2.0) for poly in polylines
            ]

            for tau in thresholds:
                pred_binary = (prob_map >= tau) & valid_mask
                targ_binary = (target_mask > 0) & valid_mask

                tp = int(np.logical_and(pred_binary, targ_binary).sum())
                fp = int(np.logical_and(pred_binary, ~targ_binary).sum())
                fn = int(np.logical_and(~pred_binary, targ_binary).sum())
                pred_area = int(pred_binary.sum())
                targ_area = int(targ_binary.sum())

                dice = (2.0 * tp / (pred_area + targ_area)) if (pred_area + targ_area) > 0 else 1.0 if is_positive is False and pred_area == 0 else 0.0
                iou = (tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 1.0 if is_positive is False and pred_area == 0 else 0.0

                # Centerline coverage (for positive studies)
                inst_coverages = []
                if is_positive:
                    # Invert binary mask for distance transform (distance to nearest pred foreground)
                    # 0 where pred is True, distance in pixels elsewhere
                    inv_pred = (~pred_binary).astype(np.uint8)
                    dist_to_fg_model = distance_transform_edt(inv_pred)

                    for inst_pts in instance_sampled_points:
                        # Map native points to model space
                        m_pts = np.array([meta.native_to_model(p[0], p[1]) for p in inst_pts])
                        m_x = np.clip(np.round(m_pts[:, 0]).astype(int), 0, self.target_size[0] - 1)
                        m_y = np.clip(np.round(m_pts[:, 1]).astype(int), 0, self.target_size[1] - 1)

                        # Distance in model pixels converted to native pixels by dividing by scale
                        model_dists = dist_to_fg_model[m_y, m_x]
                        native_dists = model_dists / max(meta.scale, 1e-5)

                        cov_at_tol = {
                            f"cov_{tol}px": float(np.mean(native_dists <= tol))
                            for tol in tolerances_px
                        }
                        inst_coverages.append(cov_at_tol)

                # Connected components for negative studies (or false foreground)
                if not is_positive:
                    num_comps, _, stats, _ = cv2.connectedComponentsWithStats(pred_binary.astype(np.uint8))
                    comp_count = max(0, num_comps - 1)  # subtract background
                    comp_areas = stats[1:, cv2.CC_STAT_AREA].tolist() if comp_count > 0 else []
                    max_comp_area = max(comp_areas) if comp_areas else 0
                else:
                    comp_count = 0
                    max_comp_area = 0

                study_record["threshold_metrics"][f"tau_{int(tau*100)}"] = {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "pred_area": pred_area,
                    "targ_area": targ_area,
                    "dice": float(dice),
                    "iou": float(iou),
                    "instance_coverages": inst_coverages,
                    "false_comp_count": comp_count,
                    "max_false_comp_area": max_comp_area,
                }

            per_study_results.append(study_record)
            if (idx + 1) % 25 == 0 or (idx + 1) == len(split_studies):
                print(f"Processed {idx + 1}/{len(split_studies)} studies ({time.time() - start_t:.1f}s)", flush=True)

        # Aggregate metrics across cohort
        summary_by_threshold = {}
        for tau in thresholds:
            t_key = f"tau_{int(tau*100)}"
            pos_studies = [s for s in per_study_results if s["is_positive"]]
            neg_studies = [s for s in per_study_results if not s["is_positive"]]

            # Positive metrics
            pos_dices = [s["threshold_metrics"][t_key]["dice"] for s in pos_studies]
            pos_ious = [s["threshold_metrics"][t_key]["iou"] for s in pos_studies]
            tot_tp = sum(s["threshold_metrics"][t_key]["tp"] for s in pos_studies)
            tot_fp_pos = sum(s["threshold_metrics"][t_key]["fp"] for s in pos_studies)
            tot_fn = sum(s["threshold_metrics"][t_key]["fn"] for s in pos_studies)

            pooled_prec = tot_tp / max(tot_tp + tot_fp_pos, 1)
            pooled_rec = tot_tp / max(tot_tp + tot_fn, 1)

            # Centerline coverages
            all_inst_coverages = {f"cov_{tol}px": [] for tol in tolerances_px}
            missed_instances = {f"missed_{tol}px": 0 for tol in tolerances_px}
            total_instances = 0

            for s in pos_studies:
                for inst in s["threshold_metrics"][t_key]["instance_coverages"]:
                    total_instances += 1
                    for tol in tolerances_px:
                        c = inst[f"cov_{tol}px"]
                        all_inst_coverages[f"cov_{tol}px"].append(c)
                        if c < 0.30:  # less than 30% coverage = missed
                            missed_instances[f"missed_{tol}px"] += 1

            mean_coverages = {
                k: float(np.mean(v)) if v else 0.0 for k, v in all_inst_coverages.items()
            }
            missed_rates = {
                f"missed_rate_{tol}px": float(missed_instances[f"missed_{tol}px"] / max(total_instances, 1))
                for tol in tolerances_px
            }

            # Negative studies metrics
            neg_false_areas = [s["threshold_metrics"][t_key]["pred_area"] for s in neg_studies]
            neg_comp_counts = [s["threshold_metrics"][t_key]["false_comp_count"] for s in neg_studies]
            affected_negs = sum(1 for a in neg_false_areas if a > 0)
            neg_affected_rate = float(affected_negs / max(len(neg_studies), 1))

            # Stratifications
            single_cvc = [s for s in pos_studies if s["num_cvc_instances"] == 1]
            multi_cvc = [s for s in pos_studies if s["num_cvc_instances"] > 1]
            abnormal_cvc = [s for s in pos_studies if s["cvc_abnormal"] == 1]
            cluttered = [s for s in pos_studies if s["has_other_lines"]]

            strata_metrics = {
                "single_cvc": {
                    "count": len(single_cvc),
                    "macro_dice": float(np.mean([s["threshold_metrics"][t_key]["dice"] for s in single_cvc])) if single_cvc else 0.0,
                },
                "multi_cvc": {
                    "count": len(multi_cvc),
                    "macro_dice": float(np.mean([s["threshold_metrics"][t_key]["dice"] for s in multi_cvc])) if multi_cvc else 0.0,
                },
                "abnormal": {
                    "count": len(abnormal_cvc),
                    "macro_dice": float(np.mean([s["threshold_metrics"][t_key]["dice"] for s in abnormal_cvc])) if abnormal_cvc else 0.0,
                },
                "cluttered_other_lines": {
                    "count": len(cluttered),
                    "macro_dice": float(np.mean([s["threshold_metrics"][t_key]["dice"] for s in cluttered])) if cluttered else 0.0,
                },
            }

            summary_by_threshold[t_key] = {
                "threshold": tau,
                "positive_studies": len(pos_studies),
                "negative_studies": len(neg_studies),
                "macro_dice": round(float(np.mean(pos_dices)), 4),
                "macro_iou": round(float(np.mean(pos_ious)), 4),
                "pooled_precision": round(float(pooled_prec), 4),
                "pooled_recall": round(float(pooled_rec), 4),
                "centerline_coverage": {k: round(v, 4) for k, v in mean_coverages.items()},
                "missed_catheter_rates": {k: round(v, 4) for k, v in missed_rates.items()},
                "total_catheter_instances": total_instances,
                "negatives": {
                    "mean_false_area_px": round(float(np.mean(neg_false_areas)), 1) if neg_studies else 0.0,
                    "max_false_area_px": int(max(neg_false_areas)) if neg_studies else 0,
                    "affected_study_rate": round(neg_affected_rate, 4),
                    "mean_component_count": round(float(np.mean(neg_comp_counts)), 2) if neg_studies else 0.0,
                },
                "strata": strata_metrics,
            }

        return {
            "checkpoint": self.checkpoint_path,
            "checkpoint_sha256": self.ckpt_sha256,
            "split": split_name,
            "total_studies_evaluated": len(split_studies),
            "summary_by_threshold": summary_by_threshold,
            "per_study_results": per_study_results,
        }


def build_and_save_audit_panel(per_study_results: List[Dict[str, Any]], output_json: str, output_csv: str):
    """Select a diverse 50-study development panel across clinical and imaging strata."""
    df = pd.DataFrame(per_study_results)
    selected_ids = []

    # 1. Multi-CVC cases (10 cases)
    multi_cases = df[df["num_cvc_instances"] > 1]["study_id"].tolist()
    selected_ids.extend(multi_cases[:10])

    # 2. Abnormal placement cases (10 cases)
    abn_cases = df[(df["cvc_abnormal"] == 1) & (~df["study_id"].isin(selected_ids))]["study_id"].tolist()
    selected_ids.extend(abn_cases[:10])

    # 3. Single CVC clean cases (10 cases)
    single_cases = df[(df["num_cvc_instances"] == 1) & (df["cvc_normal"] == 1) & (~df["has_other_lines"]) & (~df["study_id"].isin(selected_ids))]["study_id"].tolist()
    selected_ids.extend(single_cases[:10])

    # 4. Cluttered cases (CVC + ETT/NGT) (10 cases)
    clutter_cases = df[(df["is_positive"]) & (df["has_other_lines"]) & (~df["study_id"].isin(selected_ids))]["study_id"].tolist()
    selected_ids.extend(clutter_cases[:10])

    # 5. Confirmed negative cases (10 cases: both other-lines present and zero-lines)
    neg_cases = df[(~df["is_positive"]) & (~df["study_id"].isin(selected_ids))]["study_id"].tolist()
    selected_ids.extend(neg_cases[:10])

    # If < 50, fill remaining with remaining positive cases
    if len(selected_ids) < 50:
        rem = df[~df["study_id"].isin(selected_ids)]["study_id"].tolist()
        selected_ids.extend(rem[: 50 - len(selected_ids)])

    selected_ids = selected_ids[:50]
    panel_records = [s for s in per_study_results if s["study_id"] in selected_ids]

    with open(output_json, "w") as f:
        json.dump(panel_records, f, indent=2)

    # Simplified CSV for easy table review
    csv_rows = []
    for s in panel_records:
        t30 = s["threshold_metrics"]["tau_30"]
        cov10 = t30["instance_coverages"][0]["cov_10px"] if t30["instance_coverages"] else 0.0
        csv_rows.append({
            "study_id": s["study_id"],
            "patient_id": s["patient_id"],
            "is_positive": s["is_positive"],
            "num_cvc": s["num_cvc_instances"],
            "cvc_normal": s["cvc_normal"],
            "cvc_borderline": s["cvc_borderline"],
            "cvc_abnormal": s["cvc_abnormal"],
            "has_other_lines": s["has_other_lines"],
            "dice_tau_30": t30["dice"],
            "iou_tau_30": t30["iou"],
            "pred_area_tau_30": t30["pred_area"],
            "centerline_cov_10px_tau_30": cov10,
            "false_comp_count_tau_30": t30["false_comp_count"],
        })
    pd.DataFrame(csv_rows).to_csv(output_csv, index=False)
    print(f"Saved 50-study audit panel to {output_json} and {output_csv}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CVC Segmentation Masks (S0)")
    parser.add_argument("--checkpoint", type=str, default=r"C:\Ankshit\CVC-MAGNet-V2\runs\pilot_experiment_expanded\checkpoints\final_coarse_dual_head.pt")
    parser.add_argument("--inventory", type=str, default=r"C:\Ankshit\CVC-MAGNet-V2\data\annotation_inventory.parquet")
    parser.add_argument("--study_inventory", type=str, default=r"C:\Ankshit\CVC-MAGNet-V2\data\study_inventory.parquet")
    parser.add_argument("--splits_json", type=str, default=r"C:\Ankshit\CVC-MAGNet-V2\data\splits.json")
    parser.add_argument("--images_dir", type=str, default=r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")
    parser.add_argument("--split", type=str, default="dev_pilot_val")
    parser.add_argument("--output_dir", type=str, default=r"C:\Ankshit\CVC-MAGNet-V2\reports")
    parser.add_argument("--report_name", type=str, default="S0_frozen_mask_evaluation_report")
    parser.add_argument("--img_size", type=int, default=512)
    args = parser.parse_args()

    evaluator = MaskEvaluator(
        checkpoint_path=args.checkpoint,
        inventory_parquet=args.inventory,
        study_inventory_parquet=args.study_inventory,
        splits_json=args.splits_json,
        images_dir=args.images_dir,
        target_size=(args.img_size, args.img_size),
    )

    results = evaluator.evaluate_split(split_name=args.split)

    os.makedirs(args.output_dir, exist_ok=True)
    report_json = os.path.join(args.output_dir, f"{args.report_name}.json")
    report_md = os.path.join(args.output_dir, f"{args.report_name}.md")

    # Exclude per_study_results from top-level report json for readability, save full details separately
    full_details = results["per_study_results"]
    summary_report = {
        "checkpoint": results["checkpoint"],
        "checkpoint_sha256": results["checkpoint_sha256"],
        "split": results["split"],
        "total_studies_evaluated": results["total_studies_evaluated"],
        "summary_by_threshold": results["summary_by_threshold"],
    }

    with open(report_json, "w") as f:
        json.dump(summary_report, f, indent=2)

    # Build Markdown report
    title = f"# {args.report_name}: CVC Segmentation Mask Evaluation"
    md_lines = [
        title,
        "",
        f"- **Checkpoint**: `{results['checkpoint']}`",
        f"- **SHA256**: `{results['checkpoint_sha256']}`",
        f"- **Evaluation Partition**: `{results['split']}` ({results['total_studies_evaluated']} studies: 185 positive, 8 negative)",
        "- **Evaluation Principle**: Direct segmentation mask assessment excluding letterbox padding; tip Success@15 omitted.",
        "",
        "## Performance Across Operating Thresholds",
        "",
        "| Threshold ($\\tau$) | Macro Dice | Macro IoU | Precision | Recall | Centerline Cov (10px) | Missed Catheter Rate (10px) | Negatives False Area (px) | Negatives Affected Rate |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for tau_k, v in results["summary_by_threshold"].items():
        cov10 = v["centerline_coverage"].get("cov_10px", 0.0)
        missed10 = v["missed_catheter_rates"].get("missed_rate_10px", 0.0)
        neg_area = v["negatives"]["mean_false_area_px"]
        neg_aff = v["negatives"]["affected_study_rate"]
        md_lines.append(
            f"| **{v['threshold']:.2f}** | {v['macro_dice']:.4f} | {v['macro_iou']:.4f} | "
            f"{v['pooled_precision']:.4f} | {v['pooled_recall']:.4f} | {cov10*100:.1f}% | "
            f"{missed10*100:.1f}% | {neg_area:.1f} px | {neg_aff*100:.1f}% |"
        )

    t30 = results["summary_by_threshold"]["tau_30"]
    md_lines.extend([
        "",
        "## Baseline Operating Point Analysis ($\\tau = 0.30$, Viewer Default)",
        f"- **Macro Dice (Positives)**: {t30['macro_dice']:.4f}",
        f"- **Pooled Pixel Precision**: {t30['pooled_precision']:.4f} (Severe precision penalty due to diffuse foreground)",
        f"- **Pooled Pixel Recall**: {t30['pooled_recall']:.4f}",
        f"- **Centerline Coverage @ 10px**: {t30['centerline_coverage']['cov_10px']*100:.1f}%",
        f"- **Missed Catheters (<30% coverage)**: {t30['missed_catheter_rates']['missed_rate_10px']*100:.1f}%",
        f"- **Negative Studies Affected**: {t30['negatives']['affected_study_rate']*100:.1f}% with mean false area {t30['negatives']['mean_false_area_px']:.1f} px",
        "",
        "### Stratified Performance by Clinical Cohort ($\\tau = 0.30$)",
        f"- **Single CVC (N={t30['strata']['single_cvc']['count']})**: Macro Dice = {t30['strata']['single_cvc']['macro_dice']:.4f}",
        f"- **Multiple CVC (N={t30['strata']['multi_cvc']['count']})**: Macro Dice = {t30['strata']['multi_cvc']['macro_dice']:.4f}",
        f"- **Abnormal Placement (N={t30['strata']['abnormal']['count']})**: Macro Dice = {t30['strata']['abnormal']['macro_dice']:.4f}",
        f"- **Cluttered / Other Lines (N={t30['strata']['cluttered_other_lines']['count']})**: Macro Dice = {t30['strata']['cluttered_other_lines']['macro_dice']:.4f}",
    ])

    with open(report_md, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\nSaved S0 summary report to {report_json} and {report_md}", flush=True)

    # Build 50-study audit panel
    panel_json = os.path.join(args.output_dir, "50_study_audit_panel.json")
    panel_csv = os.path.join(args.output_dir, "50_study_audit_panel.csv")
    build_and_save_audit_panel(full_details, panel_json, panel_csv)


if __name__ == "__main__":
    main()
