"""Train and Evaluate Out-of-Sample Geometric GBDT (Val-Oracle vs Val-Predicted).

Implements Step 2 of the CVC Placement Architecture:
1. Feature vector (11 dimensions):
   g_i = [dx_norm, dy_norm, |dx_norm|, dy_norm^2, r_norm, cos_theta, sin_theta, length_norm, conf_tip, conf_carina, is_carina_fallback]
   where (cos_theta, sin_theta) is the distal trajectory tangent approaching the tip.
2. Train GBDT on training split (N=8,599 CVC instances across 6,600 traced studies).
3. Evaluate out-of-sample:
   - Val-Oracle (N=778 traced studies, 1,060 instances): Ground truth polylines -> true out-of-sample geometric ceiling.
   - Val-Predicted (N=2,558 full locked validation cohort): D1 predicted tips + masks from fine-tuned T2 checkpoint.
4. Compares Macro AUROC, per-class AUROC, Balanced Accuracy, and quantifies the D1 localization gap.
"""

import ast
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torchvision.transforms.functional as TF

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.infer.decoders import decode_d1_subpixel_dark, extract_peaks_from_heatmap
from src.models.tip_heatmap import CVCJointSegTipNet

DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "splits" / "placement_locked_manifest.parquet"
INV_PATH = DATA_DIR / "annotation_inventory.parquet"
CARINA_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "carina_validation_cache.parquet"
CARINA_TRAIN_PATH = DATA_DIR / "curated_cvc_v1" / "carina_train_cache.parquet"
T2_VAL_CACHE_PATH = DATA_DIR / "curated_cvc_v1" / "t2_val_predictions.parquet"
IMAGES_DIR = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")
T2_CKPT_PATH = PROJECT_ROOT / "runs" / "t2_repaired_finetune_400" / "checkpoints" / "best.pt"

LABEL_MAP = {"CVC - Normal": 0, "CVC - Borderline": 1, "CVC - Abnormal": 2}
CLASSES = ["CVC - Normal", "CVC - Borderline", "CVC - Abnormal"]


def compute_distal_tangent(pts: np.ndarray, tip_idx: int = -1, target_arclength_px: float = 30.0) -> Tuple[float, float]:
    """Compute (cos_theta, sin_theta) pointing toward the distal tip."""
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 2:
        return 0.0, 1.0  # Default downward

    if tip_idx == -1:
        tip = pts[-1]
        accum = 0.0
        ref_pt = pts[0]
        for i in range(len(pts) - 1, 0, -1):
            seg_len = float(np.linalg.norm(pts[i] - pts[i - 1]))
            accum += seg_len
            if accum >= target_arclength_px:
                overshoot = accum - target_arclength_px
                frac = overshoot / max(seg_len, 1e-6)
                ref_pt = pts[i - 1] * (1 - frac) + pts[i] * frac
                break
        vec = tip - ref_pt
    else:
        tip = pts[0]
        accum = 0.0
        ref_pt = pts[-1]
        for i in range(0, len(pts) - 1):
            seg_len = float(np.linalg.norm(pts[i + 1] - pts[i]))
            accum += seg_len
            if accum >= target_arclength_px:
                overshoot = accum - target_arclength_px
                frac = overshoot / max(seg_len, 1e-6)
                ref_pt = pts[i] * frac + pts[i + 1] * (1 - frac)
                break
        vec = tip - ref_pt

    norm = float(np.linalg.norm(vec))
    if norm < 1e-4:
        return 0.0, 1.0
    return float(vec[0] / norm), float(vec[1] / norm)


def compute_seg_tangent(
    seg_prob: np.ndarray,
    tip_model_x: float,
    tip_model_y: float,
    radius: int = 25,
) -> Tuple[float, float]:
    """Extract distal trajectory vector from local segmentation probability map."""
    h_map, w_map = seg_prob.shape
    u, v = int(round(tip_model_y)), int(round(tip_model_x))

    y_min = max(0, u - radius)
    y_max = min(h_map, u + radius + 1)
    x_min = max(0, v - radius)
    x_max = min(w_map, v + radius + 1)

    patch = seg_prob[y_min:y_max, x_min:x_max]
    patch_clean = np.where(patch >= 0.20, patch, 0.0)
    total_mass = float(patch_clean.sum())

    if total_mass < 1e-4:
        return 0.0, 1.0  # Default downward

    grid_y, grid_x = np.indices(patch.shape)
    patch_com_y = float((grid_y * patch_clean).sum() / total_mass)
    patch_com_x = float((grid_x * patch_clean).sum() / total_mass)

    com_y = y_min + patch_com_y
    com_x = x_min + patch_com_x

    # Vector pointing from catheter body (CoM) toward distal tip
    vec_x = tip_model_x - com_x
    vec_y = tip_model_y - com_y
    norm = math.hypot(vec_x, vec_y)

    if norm < 1e-4:
        return 0.0, 1.0
    return float(vec_x / norm), float(vec_y / norm)


def extract_11d_features(
    tip_x: float,
    tip_y: float,
    w: float,
    h: float,
    carina_x: float,
    carina_y: float,
    cos_theta: float,
    sin_theta: float,
    length_px: float,
    conf_tip: float,
    conf_carina: float,
    is_carina_fallback: int,
) -> List[float]:
    """Construct 11-D geometric vector."""
    dx_norm = (tip_x - carina_x) / w
    dy_norm = (tip_y - carina_y) / h
    r_norm = math.sqrt(dx_norm ** 2 + dy_norm ** 2)
    length_norm = length_px / max(w, h)

    return [
        dx_norm,
        dy_norm,
        abs(dx_norm),
        dy_norm ** 2,
        r_norm,
        cos_theta,
        sin_theta,
        length_norm,
        conf_tip,
        conf_carina,
        float(is_carina_fallback),
    ]


def build_training_matrix(df_inv: pd.DataFrame, df_carina_train: pd.DataFrame, manifest: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Build X_train, y_train on training split."""
    train_sids = set(manifest[manifest["split"] == "train"]["StudyInstanceUID"])
    train_cvc = df_inv[
        (df_inv["study_id"].isin(train_sids))
        & (df_inv["device_type"] == "CVC")
        & (df_inv["is_assessable"])
        & (df_inv["tip_x"].notna())
        & (df_inv["tip_y"].notna())
    ].copy()

    carina_dict = df_carina_train.set_index("study_id").to_dict("index")

    X_train = []
    y_train = []

    for _, r in train_cvc.iterrows():
        sid = r["study_id"]
        lbl = r["label"]
        if lbl not in LABEL_MAP:
            continue

        w = float(r["image_width"])
        h = float(r["image_height"])

        c_info = carina_dict.get(sid, {
            "carina_x_native": 0.4805 * w,
            "carina_y_native": 0.4000 * h,
            "carina_conf": 0.25,
            "is_carina_fallback": 1,
        })
        cx = float(c_info["carina_x_native"])
        cy = float(c_info["carina_y_native"])
        c_conf = float(c_info.get("carina_conf", 0.25))
        is_fb = int(c_info.get("is_carina_fallback", 1))

        raw_pts = json.loads(r["polyline"]) if isinstance(r["polyline"], str) else r["polyline"]
        tip_idx = int(r.get("tip_endpoint_index", -1))
        cos_th, sin_th = compute_distal_tangent(raw_pts, tip_idx=tip_idx)

        tip_x_val = float(r["tip_x"])
        tip_y_val = float(r["tip_y"])
        if math.isnan(tip_x_val) or math.isnan(tip_y_val):
            continue

        feat = extract_11d_features(
            tip_x=tip_x_val,
            tip_y=tip_y_val,
            w=w,
            h=h,
            carina_x=cx,
            carina_y=cy,
            cos_theta=cos_th,
            sin_theta=sin_th,
            length_px=float(r.get("polyline_length_px", 200.0)),
            conf_tip=1.0,
            conf_carina=c_conf,
            is_carina_fallback=is_fb,
        )
        if any(math.isnan(v) for v in feat):
            continue

        X_train.append(feat)
        y_train.append(LABEL_MAP[lbl])

    X_arr = np.array(X_train, dtype=np.float32)
    y_arr = np.array(y_train, dtype=np.int64)
    # Final sanity check to guarantee zero NaNs
    assert not np.isnan(X_arr).any(), f"X_arr still contains NaNs: {np.isnan(X_arr).sum()} entries"
    return X_arr, y_arr


def evaluate_val_oracle(
    clf: GradientBoostingClassifier,
    df_inv: pd.DataFrame,
    df_carina_val: pd.DataFrame,
    manifest: pd.DataFrame,
) -> Dict[str, Any]:
    """Evaluate out-of-sample on N=778 traced validation studies using ground truth polylines."""
    val_sids = set(manifest[manifest["split"] == "val"]["StudyInstanceUID"])
    val_cvc = df_inv[
        (df_inv["study_id"].isin(val_sids))
        & (df_inv["device_type"] == "CVC")
        & (df_inv["is_assessable"])
        & (df_inv["tip_x"].notna())
        & (df_inv["tip_y"].notna())
    ].copy()

    val_meta = manifest[manifest["split"] == "val"].set_index("StudyInstanceUID")
    carina_dict = df_carina_val.set_index("study_id").to_dict("index")

    study_preds = {}
    study_gts = {}

    for sid, group in val_cvc.groupby("study_id"):
        w = float(group.iloc[0]["image_width"])
        h = float(group.iloc[0]["image_height"])

        c_info = carina_dict.get(sid, {
            "carina_x_native": 0.4805 * w,
            "carina_y_native": 0.4000 * h,
            "carina_conf": 0.25,
            "is_carina_fallback": 1,
        })
        cx = float(c_info["carina_x_native"])
        cy = float(c_info["carina_y_native"])
        c_conf = float(c_info.get("carina_conf", 0.25))
        is_fb = int(c_info.get("is_carina_fallback", 1))

        inst_probs = []
        for _, r in group.iterrows():
            raw_pts = json.loads(r["polyline"]) if isinstance(r["polyline"], str) else r["polyline"]
            tip_idx = int(r.get("tip_endpoint_index", -1))
            cos_th, sin_th = compute_distal_tangent(raw_pts, tip_idx=tip_idx)

            feat = extract_11d_features(
                tip_x=float(r["tip_x"]),
                tip_y=float(r["tip_y"]),
                w=w,
                h=h,
                carina_x=cx,
                carina_y=cy,
                cos_theta=cos_th,
                sin_theta=sin_th,
                length_px=float(r.get("polyline_length_px", 200.0)),
                conf_tip=1.0,
                conf_carina=c_conf,
                is_carina_fallback=is_fb,
            )
            prob = clf.predict_proba([feat])[0]  # [P(Norm), P(Bord), P(Abn)]
            inst_probs.append(prob)

        # Study-level max pooling
        study_prob = np.max(inst_probs, axis=0)
        study_preds[sid] = study_prob

        # GT labels from manifest
        row_m = val_meta.loc[sid]
        study_gts[sid] = [int(row_m["CVC - Normal"]), int(row_m["CVC - Borderline"]), int(row_m["CVC - Abnormal"])]

    sids_eval = list(study_preds.keys())
    Y_pred = np.array([study_preds[s] for s in sids_eval])
    Y_true = np.array([study_gts[s] for s in sids_eval])

    results = {}
    aurocs = []
    for c_idx, c_name in enumerate(CLASSES):
        auc = roc_auc_score(Y_true[:, c_idx], Y_pred[:, c_idx])
        aurocs.append(auc)
        results[c_name] = {"auroc": round(float(auc), 4), "pos_support": int(Y_true[:, c_idx].sum())}

    macro_auc = float(np.mean(aurocs))
    results["macro_auroc"] = round(macro_auc, 4)
    results["total_studies"] = len(sids_eval)

    return results


def extract_or_load_t2_val_predictions(
    val_manifest: pd.DataFrame,
    images_dir: Path,
    t2_ckpt_path: Path,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    img_size: int = 512,
    tip_threshold: float = 0.15,
) -> pd.DataFrame:
    """Run T2 joint segmentation + tip detector across all 2,558 validation images and cache results."""
    if T2_VAL_CACHE_PATH.exists():
        print(f"Loading cached T2 predictions from {T2_VAL_CACHE_PATH}...")
        df_cached = pd.read_parquet(T2_VAL_CACHE_PATH)
        if len(df_cached) == len(val_manifest):
            print(f"Loaded {len(df_cached)} cached T2 predictions.")
            return df_cached

    device = torch.device(device_str)
    print(f"Running fine-tuned T2 checkpoint ({t2_ckpt_path.name}) across {len(val_manifest)} validation images...")

    model = CVCJointSegTipNet(pretrained=False)
    ckpt = torch.load(t2_ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    results = []
    t0 = time.time()

    for idx, row in val_manifest.iterrows():
        sid = row["StudyInstanceUID"]
        img_path = images_dir / f"{sid}.jpg"

        if not img_path.exists():
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        native_h, native_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

        # Tensor conversion & standard ImageNet normalization
        img_t = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = ((img_t - mean) / std).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img_t)
            tip_probs = out["tip_probs"][0, 0].cpu().numpy()  # Stride 4 -> [128, 128]
            seg_probs = out["seg_probs"][0, 0].cpu().numpy()  # Stride 1 -> [512, 512]

        # Extract peaks
        raw_peaks, nms_peaks = extract_peaks_from_heatmap(
            tip_probs,
            threshold=tip_threshold,
            nms_kernel_size=3,
            max_candidates=5,
            suppression_radius_px=3.0,
        )

        detected_tips = []
        if len(nms_peaks) == 0:
            # Fallback to argmax peak to preserve continuous gradient
            argmax_y, argmax_x = np.unravel_index(np.argmax(tip_probs), tip_probs.shape)
            max_conf = float(tip_probs[argmax_y, argmax_x])
            nms_peaks = [{"grid_y": int(argmax_y), "grid_x": int(argmax_x), "conf": max_conf}]

        for p in nms_peaks:
            nx, ny, _ = decode_d1_subpixel_dark(
                heatmap=tip_probs,
                peak=p,
                stride=4,
                img_size=img_size,
                native_w=native_w,
                native_h=native_h,
            )
            # Model coords for tangent extraction
            mx = (nx / native_w) * img_size
            my = (ny / native_h) * img_size
            cos_th, sin_th = compute_seg_tangent(seg_probs, mx, my, radius=25)

            detected_tips.append({
                "native_x": float(nx),
                "native_y": float(ny),
                "conf": float(p["conf"]),
                "cos_theta": float(cos_th),
                "sin_theta": float(sin_th),
                "length_px": 200.0,
            })

        results.append({
            "study_id": sid,
            "image_width": native_w,
            "image_height": native_h,
            "tips_json": json.dumps(detected_tips),
            "max_tip_conf": max(p["conf"] for p in nms_peaks),
            "num_tips": len(detected_tips),
        })

        if (idx + 1) % 500 == 0 or (idx + 1) == len(val_manifest):
            elapsed = time.time() - t0
            rate = (idx + 1) / max(elapsed, 1e-4)
            print(f"  [T2 Val Inference] Processed {idx + 1}/{len(val_manifest)} ({rate:.1f} img/s)", flush=True)

    df_out = pd.DataFrame(results)
    T2_VAL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(T2_VAL_CACHE_PATH, index=False)
    print(f"Saved {len(df_out)} T2 predictions to {T2_VAL_CACHE_PATH}")
    return df_out


def evaluate_val_predicted(
    clf: GradientBoostingClassifier,
    df_t2_preds: pd.DataFrame,
    df_carina_val: pd.DataFrame,
    manifest: pd.DataFrame,
) -> Dict[str, Any]:
    """Evaluate end-to-end on full N=2,558 locked validation cohort using D1 predicted tips + masks."""
    val_meta = manifest[manifest["split"] == "val"].set_index("StudyInstanceUID")
    carina_dict = df_carina_val.set_index("study_id").to_dict("index")
    t2_dict = df_t2_preds.set_index("study_id").to_dict("index")

    study_preds = {}
    study_gts = {}

    for sid, row_m in val_meta.iterrows():
        # GT study-level labels
        gt_vec = [int(row_m["CVC - Normal"]), int(row_m["CVC - Borderline"]), int(row_m["CVC - Abnormal"])]
        study_gts[sid] = gt_vec

        pred_info = t2_dict.get(sid)
        if pred_info is None:
            # Fallback prior: [P(Normal)=0.68, P(Borderline)=0.19, P(Abnormal)=0.13]
            study_preds[sid] = np.array([0.68, 0.19, 0.13], dtype=np.float32)
            continue

        w = float(pred_info["image_width"])
        h = float(pred_info["image_height"])

        c_info = carina_dict.get(sid, {
            "carina_x_native": 0.4805 * w,
            "carina_y_native": 0.4000 * h,
            "carina_conf": 0.25,
            "is_carina_fallback": 1,
        })
        cx = float(c_info["carina_x_native"])
        cy = float(c_info["carina_y_native"])
        c_conf = float(c_info.get("carina_conf", 0.25))
        is_fb = int(c_info.get("is_carina_fallback", 1))

        tips = json.loads(pred_info["tips_json"])
        inst_probs = []

        for tip in tips:
            feat = extract_11d_features(
                tip_x=float(tip["native_x"]),
                tip_y=float(tip["native_y"]),
                w=w,
                h=h,
                carina_x=cx,
                carina_y=cy,
                cos_theta=float(tip["cos_theta"]),
                sin_theta=float(tip["sin_theta"]),
                length_px=float(tip.get("length_px", 200.0)),
                conf_tip=float(tip["conf"]),
                conf_carina=c_conf,
                is_carina_fallback=is_fb,
            )
            prob = clf.predict_proba([feat])[0]  # [P(Norm), P(Bord), P(Abn)]
            inst_probs.append(prob)

        # Multi-label max pooling across all detected catheter tips
        study_prob = np.max(inst_probs, axis=0)
        study_preds[sid] = study_prob

    sids_eval = list(study_preds.keys())
    Y_pred = np.array([study_preds[s] for s in sids_eval])
    Y_true = np.array([study_gts[s] for s in sids_eval])

    results = {}
    aurocs = []
    for c_idx, c_name in enumerate(CLASSES):
        auc = roc_auc_score(Y_true[:, c_idx], Y_pred[:, c_idx])
        aurocs.append(auc)
        results[c_name] = {"auroc": round(float(auc), 4), "pos_support": int(Y_true[:, c_idx].sum())}

    macro_auc = float(np.mean(aurocs))
    results["macro_auroc"] = round(macro_auc, 4)
    results["total_studies"] = len(sids_eval)

    return results


def main():
    print("=" * 70)
    print("STEP 2: OUT-OF-SAMPLE GEOMETRIC GBDT BENCHMARK")
    print("=" * 70)

    # 1. Load data
    manifest = pd.read_parquet(MANIFEST_PATH)
    inv = pd.read_parquet(INV_PATH)

    if not CARINA_TRAIN_PATH.exists():
        print(f"ERROR: {CARINA_TRAIN_PATH} does not exist yet. Please wait for CarinaNet training cache to finish.")
        return

    df_carina_train = pd.read_parquet(CARINA_TRAIN_PATH)
    df_carina_val = pd.read_parquet(CARINA_VAL_PATH)

    print(f"Training carina cache: {len(df_carina_train):,} studies.")
    print(f"Validation carina cache: {len(df_carina_val):,} studies.")

    # 2. Build Training Matrix on 6,600 training split studies
    print("\n[1/4] Building 11-D training matrix on training split (N=6,600)...")
    X_train, y_train = build_training_matrix(inv, df_carina_train, manifest)
    print(f"X_train shape: {X_train.shape}, y_train distribution: {np.bincount(y_train)}")

    # 3. Fit Model
    print("\n[2/4] Fitting GradientBoostingClassifier(150 trees, max_depth=4)...")
    clf = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42)
    clf.fit(X_train, y_train)

    feature_names = ["dx_norm", "dy_norm", "|dx_norm|", "dy_norm^2", "r_norm", "cos_theta", "sin_theta", "length_norm", "conf_tip", "conf_carina", "is_carina_fallback"]
    importances = dict(zip(feature_names, [round(float(v), 4) for v in clf.feature_importances_]))
    print(f"Feature Importances: {importances}")

    # 4. Out-of-sample Val-Oracle Evaluation (N=778 traced validation studies)
    print("\n[3/4] Evaluating out-of-sample Val-Oracle on N=778 traced validation studies...")
    val_oracle_res = evaluate_val_oracle(clf, inv, df_carina_val, manifest)
    print(f"--> Val-Oracle Macro AUROC: {val_oracle_res['macro_auroc']:.4f}")
    for c in CLASSES:
        print(f"    {c}: AUROC = {val_oracle_res[c]['auroc']:.4f} (pos={val_oracle_res[c]['pos_support']})")

    # 5. Out-of-sample Val-Predicted Evaluation (N=2,558 full locked validation cohort)
    print("\n[4/4] Evaluating out-of-sample Val-Predicted across full N=2,558 locked validation cohort...")
    val_manifest = manifest[manifest["split"] == "val"].reset_index(drop=True)
    df_t2_preds = extract_or_load_t2_val_predictions(
        val_manifest=val_manifest,
        images_dir=IMAGES_DIR,
        t2_ckpt_path=T2_CKPT_PATH,
    )
    val_pred_res = evaluate_val_predicted(clf, df_t2_preds, df_carina_val, manifest)
    print(f"--> Val-Predicted Macro AUROC: {val_pred_res['macro_auroc']:.4f}")
    for c in CLASSES:
        print(f"    {c}: AUROC = {val_pred_res[c]['auroc']:.4f} (pos={val_pred_res[c]['pos_support']})")

    # 6. Localization Gap Summary
    gap = val_oracle_res['macro_auroc'] - val_pred_res['macro_auroc']
    print("\n" + "=" * 70)
    print("FINAL STEP 2 BENCHMARK COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Method / Experiment':<35} | {'Cohort':<12} | {'Normal':<8} | {'Border':<8} | {'Abnorm':<8} | {'Macro AUROC':<10}")
    print("-" * 90)
    print(f"{'Image-Level Traced EffNet Baseline':<35} | {'N=2,558':<12} | {'0.7610':<8} | {'0.6280':<8} | {'0.7405':<8} | {'0.7098':<10}")
    print(f"{'Geometric GBDT (Val-Oracle)':<35} | {'N=778':<12} | {val_oracle_res['CVC - Normal']['auroc']:<8.4f} | {val_oracle_res['CVC - Borderline']['auroc']:<8.4f} | {val_oracle_res['CVC - Abnormal']['auroc']:<8.4f} | {val_oracle_res['macro_auroc']:<10.4f}")
    print(f"{'Geometric GBDT (Val-Predicted)':<35} | {'N=2,558':<12} | {val_pred_res['CVC - Normal']['auroc']:<8.4f} | {val_pred_res['CVC - Borderline']['auroc']:<8.4f} | {val_pred_res['CVC - Abnormal']['auroc']:<8.4f} | {val_pred_res['macro_auroc']:<10.4f}")
    print("-" * 90)
    print(f"Localization Penalty Gap (Oracle -> Predicted): {gap:+.4f} Macro AUROC")
    print("=" * 70)

    # Save summary report to JSON
    out_json = PROJECT_ROOT / "runs" / "geometric_gbdt_evaluation_report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "feature_importances": importances,
        "val_oracle": val_oracle_res,
        "val_predicted": val_pred_res,
        "localization_gap": round(float(gap), 4),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"Wrote full evaluation report to {out_json}")


if __name__ == "__main__":
    main()
