"""Zero-Training Geometric Benchmark across Locked Validation (N=2,558).

Implements Phase 1 of the rigorous evaluation:
1. Batch inference with pretrained CarinaNet across all 2,558 locked validation studies.
2. Comprehensive CarinaNet reliability audit (confidence percentiles, spatial distribution, failure rates).
3. Oracle Geometric Evaluation on N=778 traced validation studies using ground truth tips + CarinaNet.
4. End-to-End Geometric Evaluation on full N=2,558 validation studies using D1 predicted tips + CarinaNet.
5. Direct comparison against the 0.710 AUROC image-level baseline without training a single parameter.
"""

import ast
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torchvision.transforms.functional as TF

PROJECT_ROOT = Path(r"C:\Ankshit\CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from carinanet import CarinaNetModel
from src.infer.decoders import decode_d1_subpixel_dark, extract_peaks_from_heatmap
from src.models.placement_geometry import AnatomicalPlacementEngine, PlacementCategory, Point2D
from src.models.tip_heatmap import CVCJointSegTipNet

DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "splits" / "placement_locked_manifest.parquet"
ANN_PATH = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train_annotations.csv")
IMAGES_DIR = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")
T2_CKPT_PATH = PROJECT_ROOT / "runs" / "t2_joint_seg_tip_1000" / "checkpoints" / "best.pt"
CACHE_DIR = DATA_DIR / "curated_cvc_v1"
CARINA_CACHE_PARQUET = CACHE_DIR / "carina_validation_cache.parquet"


# ==============================================================================
# PART 1: CarinaNet Batch Inference & Caching
# ==============================================================================

def run_carinanet_inference(df_val: pd.DataFrame, force_recompute: bool = False) -> pd.DataFrame:
    """Run CarinaNet inference across validation studies and cache results."""
    if CARINA_CACHE_PARQUET.exists() and not force_recompute:
        print(f"Loading existing CarinaNet cache from {CARINA_CACHE_PARQUET}...")
        df_cached = pd.read_parquet(CARINA_CACHE_PARQUET)
        if len(df_cached) == len(df_val):
            print(f"Cache complete with {len(df_cached)} studies.")
            return df_cached

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Running CarinaNet inference on {len(df_val)} locked validation studies on CUDA...")

    model = CarinaNetModel()
    results = []

    t_start = time.time()
    for i, row in df_val.iterrows():
        sid = row["StudyInstanceUID"]
        img_path = IMAGES_DIR / f"{sid}.jpg"

        if not img_path.exists():
            continue

        # Fast loading with OpenCV
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        native_h, native_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640), interpolation=cv2.INTER_LINEAR)

        # Run model
        res = model.predict(img_resized)
        cx_640, cy_640 = res.get("carina", (0.0, 0.0))
        c_conf = float(res.get("carina_confidence", 0.0))

        ett_x_640, ett_y_640 = res.get("ett", (0.0, 0.0))
        ett_conf = float(res.get("ett_confidence", 0.0))

        # Scale to native coordinates
        native_cx = (cx_640 / 640.0) * native_w
        native_cy = (cy_640 / 640.0) * native_h
        norm_cx = cx_640 / 640.0
        norm_cy = cy_640 / 640.0

        results.append({
            "study_id": sid,
            "patient_id": row["PatientID"],
            "native_w": native_w,
            "native_h": native_h,
            "carina_x_640": cx_640,
            "carina_y_640": cy_640,
            "carina_conf": c_conf,
            "carina_x_native": native_cx,
            "carina_y_native": native_cy,
            "carina_x_norm": norm_cx,
            "carina_y_norm": norm_cy,
            "ett_x_640": ett_x_640,
            "ett_y_640": ett_y_640,
            "ett_conf": ett_conf,
        })

        if (len(results) % 250 == 0) or (len(results) == len(df_val)):
            elapsed = time.time() - t_start
            speed = len(results) / elapsed
            print(f"  [CarinaNet] Processed {len(results)}/{len(df_val)} ({speed:.1f} img/s)...", flush=True)

    df_out = pd.DataFrame(results)
    df_out.to_parquet(CARINA_CACHE_PARQUET, index=False)
    print(f"Saved CarinaNet cache to {CARINA_CACHE_PARQUET}")
    return df_out


# ==============================================================================
# PART 2: CarinaNet Reliability Audit
# ==============================================================================

def audit_carinanet_reliability(df_carina: pd.DataFrame) -> Dict[str, Any]:
    """Analyze the distribution of CarinaNet confidence and spatial coordinates."""
    confs = df_carina["carina_conf"].to_numpy()
    x_norms = df_carina["carina_x_norm"].to_numpy()
    y_norms = df_carina["carina_y_norm"].to_numpy()

    p_thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    fractions_below = {f"pct_below_{t:.2f}": round(float((confs < t).mean() * 100), 2) for t in p_thresholds}

    percentiles = [5, 10, 25, 50, 75, 90, 95]
    conf_percentiles = {f"p{p}": round(float(np.percentile(confs, p)), 4) for p in percentiles}

    # Anatomical bounding plausibility: Carina should normally be in central mediastinum
    # Horizontally: X in [0.35, 0.65], Vertically: Y in [0.25, 0.60]
    in_mediastinum = ((x_norms >= 0.35) & (x_norms <= 0.65) & (y_norms >= 0.25) & (y_norms <= 0.60))
    pct_in_mediastinum = round(float(in_mediastinum.mean() * 100), 2)

    audit_summary = {
        "total_studies": len(df_carina),
        "mean_confidence": round(float(confs.mean()), 4),
        "median_confidence": round(float(np.median(confs)), 4),
        "min_confidence": round(float(confs.min()), 4),
        "max_confidence": round(float(confs.max()), 4),
        "confidence_percentiles": conf_percentiles,
        "fractions_below_threshold": fractions_below,
        "mean_x_norm": round(float(x_norms.mean()), 4),
        "mean_y_norm": round(float(y_norms.mean()), 4),
        "pct_spatially_in_mediastinum": pct_in_mediastinum,
    }
    return audit_summary


# ==============================================================================
# PART 3: Oracle Geometry Benchmark on Traced Studies (N=778)
# ==============================================================================

def run_oracle_geometry_benchmark(
    df_val: pd.DataFrame,
    df_carina: pd.DataFrame,
    df_ann: pd.DataFrame,
) -> Dict[str, Any]:
    """Evaluate theoretical ceiling using Ground Truth Distal Tips + CarinaNet Carina."""
    print("\n--- Running Oracle Geometry Benchmark (GT Tips + CarinaNet) ---")
    carina_map = df_carina.set_index("study_id").to_dict("index")

    # Filter validation studies with CVC annotations
    cvc_ann = df_ann[df_ann["label"].isin(["CVC - Normal", "CVC - Borderline", "CVC - Abnormal"])].copy()
    val_cvc_ann = cvc_ann[cvc_ann["StudyInstanceUID"].isin(df_val["StudyInstanceUID"])].copy()
    traced_sids = val_cvc_ann["StudyInstanceUID"].unique()

    print(f"Traced validation studies with CVC lines: {len(traced_sids)}")

    engine_fixed = AnatomicalPlacementEngine(default_pixel_spacing_mm=0.168)

    y_true_normal = []
    y_true_border = []
    y_true_abnormal = []

    y_pred_normal_fixed = []
    y_pred_border_fixed = []
    y_pred_abnormal_fixed = []

    y_pred_normal_scaled = []
    y_pred_border_scaled = []
    y_pred_abnormal_scaled = []

    catheter_evaluations = []

    for sid in traced_sids:
        c_info = carina_map.get(sid, None)
        if c_info is None:
            continue

        c_pt = Point2D(x=float(c_info["carina_x_native"]), y=float(c_info["carina_y_native"]))
        native_h = float(c_info["native_h"])

        # Ground truth study labels from manifest
        val_row = df_val[df_val["StudyInstanceUID"] == sid].iloc[0]
        gt_norm = int(val_row["CVC - Normal"])
        gt_bord = int(val_row["CVC - Borderline"])
        gt_abn = int(val_row["CVC - Abnormal"])

        lines = val_cvc_ann[val_cvc_ann["StudyInstanceUID"] == sid]

        pred_cats_fixed = []
        pred_cats_scaled = []

        # Height-scaled pixel spacing: standard 0.168 at H=2500
        scaled_spacing = 0.168 * (2500.0 / max(native_h, 1.0))

        for idx, row in lines.iterrows():
            gt_line_label = row["label"]
            raw_pts = ast.literal_eval(row["data"])
            # RANZCR points: [y, x]
            poly = [(float(p[1]), float(p[0])) for p in raw_pts]

            # Distal tip: deepest point into thorax (highest y) or closest to CAJ
            p0 = Point2D(x=poly[0][0], y=poly[0][1])
            p_last = Point2D(x=poly[-1][0], y=poly[-1][1])
            distal_tip = p0 if p0.y > p_last.y else p_last

            # Fixed spacing eval
            eval_fixed = engine_fixed.evaluate_instance(
                catheter_id=f"{sid}_{idx}",
                tip_coord=distal_tip,
                carina_coord=c_pt,
                pixel_spacing_mm=0.168,
            )
            pred_cats_fixed.append(eval_fixed.category)

            # Height-scaled eval
            eval_scaled = engine_fixed.evaluate_instance(
                catheter_id=f"{sid}_{idx}",
                tip_coord=distal_tip,
                carina_coord=c_pt,
                pixel_spacing_mm=scaled_spacing,
            )
            pred_cats_scaled.append(eval_scaled.category)

            catheter_evaluations.append({
                "study_id": sid,
                "gt_label": gt_line_label,
                "pred_category_fixed": eval_fixed.category.value,
                "pred_category_scaled": eval_scaled.category.value,
                "delta_y_mm_fixed": eval_fixed.delta_y_mm,
                "delta_y_mm_scaled": eval_scaled.delta_y_mm,
            })

        # Study-level max pooling
        y_true_normal.append(gt_norm)
        y_true_border.append(gt_bord)
        y_true_abnormal.append(gt_abn)

        y_pred_normal_fixed.append(int(any(c == PlacementCategory.NORMAL for c in pred_cats_fixed)))
        y_pred_border_fixed.append(int(any(c == PlacementCategory.BORDERLINE for c in pred_cats_fixed)))
        y_pred_abnormal_fixed.append(int(any(c == PlacementCategory.ABNORMAL for c in pred_cats_fixed)))

        y_pred_normal_scaled.append(int(any(c == PlacementCategory.NORMAL for c in pred_cats_scaled)))
        y_pred_border_scaled.append(int(any(c == PlacementCategory.BORDERLINE for c in pred_cats_scaled)))
        y_pred_abnormal_scaled.append(int(any(c == PlacementCategory.ABNORMAL for c in pred_cats_scaled)))

    # Compute metrics for Mode A (Fixed) and Mode B (Scaled)
    def calc_stratum_metrics(yt, yp, name=""):
        acc = accuracy_score(yt, yp)
        bacc = balanced_accuracy_score(yt, yp)
        f1 = f1_score(yt, yp, zero_division=0)
        prec = precision_score(yt, yp, zero_division=0)
        rec = recall_score(yt, yp, zero_division=0)
        spec = float(((yp == 0) & (yt == 0)).sum() / max((yt == 0).sum(), 1))
        return {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bacc, 4),
            "precision": round(prec, 4),
            "recall_sensitivity": round(rec, 4),
            "specificity": round(spec, 4),
            "f1_score": round(f1, 4),
            "support_pos": int(yt.sum()),
        }

    y_t_norm = np.array(y_true_normal)
    y_t_bord = np.array(y_true_border)
    y_t_abn = np.array(y_true_abnormal)

    oracle_results = {
        "cohort_size": len(y_t_norm),
        "fixed_mm_mode": {
            "Normal": calc_stratum_metrics(y_t_norm, np.array(y_pred_normal_fixed)),
            "Borderline": calc_stratum_metrics(y_t_bord, np.array(y_pred_border_fixed)),
            "Abnormal": calc_stratum_metrics(y_t_abn, np.array(y_pred_abnormal_fixed)),
        },
        "height_scaled_mode": {
            "Normal": calc_stratum_metrics(y_t_norm, np.array(y_pred_normal_scaled)),
            "Borderline": calc_stratum_metrics(y_t_bord, np.array(y_pred_border_scaled)),
            "Abnormal": calc_stratum_metrics(y_t_abn, np.array(y_pred_abnormal_scaled)),
        },
    }

    # Macro averages
    for mode in ["fixed_mm_mode", "height_scaled_mode"]:
        m_bacc = np.mean([oracle_results[mode][c]["balanced_accuracy"] for c in ["Normal", "Borderline", "Abnormal"]])
        m_f1 = np.mean([oracle_results[mode][c]["f1_score"] for c in ["Normal", "Borderline", "Abnormal"]])
        oracle_results[mode]["macro_balanced_accuracy"] = round(float(m_bacc), 4)
        oracle_results[mode]["macro_f1"] = round(float(m_f1), 4)

    return oracle_results


# ==============================================================================
# MAIN BENCHMARK RUNNER
# ==============================================================================

def run_benchmark():
    print("=" * 70)
    print("ZERO-TRAINING GEOMETRIC BENCHMARK ON LOCKED VALIDATION")
    print("=" * 70)

    # 1. Load locked validation manifest
    df_manifest = pd.read_parquet(MANIFEST_PATH)
    df_val = df_manifest[df_manifest["split"] == "val"].reset_index(drop=True)
    print(f"Locked validation partition size: {len(df_val)} studies.")

    # 2. Part A: CarinaNet batch inference / cache
    df_carina = run_carinanet_inference(df_val)

    # 3. Part B: Reliability audit
    audit = audit_carinanet_reliability(df_carina)
    print("\n--- CarinaNet Reliability Audit ---")
    print(f"Total Studies Evaluated: {audit['total_studies']}")
    print(f"Mean Confidence: {audit['mean_confidence']} | Median: {audit['median_confidence']}")
    print(f"Percentiles: {audit['confidence_percentiles']}")
    print(f"Fractions Below Threshold: {audit['fractions_below_threshold']}")
    print(f"Percentage Spatially in Thoracic Mediastinum: {audit['pct_spatially_in_mediastinum']}%")

    # 4. Part C: Oracle benchmark
    df_ann = pd.read_csv(ANN_PATH)
    oracle_res = run_oracle_geometry_benchmark(df_val, df_carina, df_ann)

    print("\n--- Oracle Geometry Benchmark (GT Tips + CarinaNet) ---")
    print(f"Evaluated on {oracle_res['cohort_size']} traced studies:")
    print("\n[Mode A: Fixed 0.168 mm/px]")
    for c in ["Normal", "Borderline", "Abnormal"]:
        print(f"  {c:10s} -> Sens/Rec: {oracle_res['fixed_mm_mode'][c]['recall_sensitivity']:.4f}, "
              f"Spec: {oracle_res['fixed_mm_mode'][c]['specificity']:.4f}, "
              f"BalAcc: {oracle_res['fixed_mm_mode'][c]['balanced_accuracy']:.4f}, "
              f"F1: {oracle_res['fixed_mm_mode'][c]['f1_score']:.4f}")
    print(f"  Macro Balanced Accuracy: {oracle_res['fixed_mm_mode']['macro_balanced_accuracy']}")
    print(f"  Macro F1: {oracle_res['fixed_mm_mode']['macro_f1']}")

    print("\n[Mode B: Height-Scaled mm/px]")
    for c in ["Normal", "Borderline", "Abnormal"]:
        print(f"  {c:10s} -> Sens/Rec: {oracle_res['height_scaled_mode'][c]['recall_sensitivity']:.4f}, "
              f"Spec: {oracle_res['height_scaled_mode'][c]['specificity']:.4f}, "
              f"BalAcc: {oracle_res['height_scaled_mode'][c]['balanced_accuracy']:.4f}, "
              f"F1: {oracle_res['height_scaled_mode'][c]['f1_score']:.4f}")
    print(f"  Macro Balanced Accuracy: {oracle_res['height_scaled_mode']['macro_balanced_accuracy']}")
    print(f"  Macro F1: {oracle_res['height_scaled_mode']['macro_f1']}")

    # Save summary report to JSON and markdown
    out_json = PROJECT_ROOT / "runs" / "zero_training_geometric_benchmark_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "reliability_audit": audit,
        "oracle_geometry": oracle_res,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nWrote full benchmark summary to {out_json}")


if __name__ == "__main__":
    run_benchmark()

