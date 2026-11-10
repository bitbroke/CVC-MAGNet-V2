"""Stage 1: Anatomical Origin & Spatial Reference Sweeps (E01-E12).

Tests 12 ways to ground the coordinate origin, handle fallbacks, normalize thoracic scales,
and extract geometric discriminability on the full locked N=2,558 validation cohort.

Results are appended to reports/master_ablation_ledger.parquet/.csv and predictions saved to
runs/val_preds/val_preds_{exp_id}.npy.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "splits" / "placement_locked_manifest.parquet"
CARINA_TR_PATH = DATA_DIR / "curated_cvc_v1" / "carina_train_cache.parquet"
CARINA_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "carina_validation_cache.parquet"
ZONAL_TR_PATH = DATA_DIR / "curated_cvc_v1" / "zonal_train_features.parquet"
ZONAL_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "zonal_val_features.parquet"

REPORTS_DIR = PROJECT_ROOT / "reports"
VAL_PREDS_DIR = PROJECT_ROOT / "runs" / "val_preds"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
VAL_PREDS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["CVC - Normal", "CVC - Borderline", "CVC - Abnormal"]

BASE_ZONAL_COLS = [
    "m_neck", "m_upper_svc", "m_borderline", "m_caj_normal", "m_atrium_deep", "m_lateral", "m_total_med",
    "frac_neck", "frac_upper_svc", "frac_borderline", "frac_caj_normal", "frac_atrium_deep",
    "deepest_y_norm", "p95_y_norm", "highest_y_norm", "ribbon_pixels"
]


def calc_cohort_auroc(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray = None) -> float:
    if mask is not None:
        y_t = y_true[mask]
        y_p = y_pred[mask]
    else:
        y_t = y_true
        y_p = y_pred

    aucs = []
    for c in range(3):
        if len(np.unique(y_t[:, c])) > 1:
            auc = roc_auc_score(y_t[:, c], y_p[:, c])
            aucs.append(auc)
    return round(float(np.mean(aucs)), 4) if len(aucs) > 0 else 0.0


def fit_and_eval_gbdt(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    is_traced: np.ndarray,
    is_untraced: np.ndarray,
    is_multicvc: np.ndarray,
    exp_id: str,
    stage: int = 1,
    tip_recall_35px: float = 55.7,
) -> Tuple[Dict, np.ndarray]:
    t0 = time.time()
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(np.nan_to_num(X_train, nan=0.0))
    X_val_s = scaler.transform(np.nan_to_num(X_val, nan=0.0))

    p_val = np.zeros((len(y_val), 3), dtype=np.float32)
    for c in range(3):
        clf = LGBMClassifier(
            n_estimators=160,
            learning_rate=0.04,
            max_depth=4,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(X_tr_s, y_train[:, c])
        p_val[:, c] = clf.predict_proba(X_val_s)[:, 1]

    elapsed = round(time.time() - t0, 1)
    all_auc = calc_cohort_auroc(y_val, p_val)
    norm_auc = round(float(roc_auc_score(y_val[:, 0], p_val[:, 0])), 4)
    bord_auc = round(float(roc_auc_score(y_val[:, 1], p_val[:, 1])), 4)
    abno_auc = round(float(roc_auc_score(y_val[:, 2], p_val[:, 2])), 4)

    traced_auc = calc_cohort_auroc(y_val, p_val, is_traced)
    untraced_auc = calc_cohort_auroc(y_val, p_val, is_untraced)
    multicvc_auc = calc_cohort_auroc(y_val, p_val, is_multicvc)

    rec = {
        "exp_id": exp_id,
        "stage": stage,
        "Runtime_s": elapsed,
        "Val_Macro_AUROC_2558": all_auc,
        "Normal_AUROC": norm_auc,
        "Borderline_AUROC": bord_auc,
        "Abnormal_AUROC": abno_auc,
        "Traced_778_AUROC": traced_auc,
        "Untraced_1780_AUROC": untraced_auc,
        "MultiCVC_332_AUROC": multicvc_auc,
        "Tip_Recall_35px": tip_recall_35px,
    }
    return rec, p_val


def main():
    print("=" * 80)
    print("STAGE 1: ANATOMICAL ORIGIN & SPATIAL REFERENCE SWEEPS (E01 - E12)")
    print("=" * 80)

    # Load manifests and caches
    manifest = pd.read_parquet(MANIFEST_PATH)
    val_df = manifest[manifest["split"] == "val"].reset_index(drop=True)
    y_val = val_df[CLASSES].values.astype(int)

    is_traced = (val_df["annotated_cvc_count"] > 0).values
    is_untraced = ~is_traced
    is_multicvc = (val_df["annotated_cvc_count"] > 1) | (y_val.sum(axis=1) > 1)

    df_z_tr = pd.read_parquet(ZONAL_TR_PATH)
    df_z_val = pd.read_parquet(ZONAL_VAL_PATH)
    y_tr = df_z_tr[CLASSES].values.astype(int)

    df_c_tr = pd.read_parquet(CARINA_TR_PATH).set_index("study_id")
    df_c_val = pd.read_parquet(CARINA_VAL_PATH).set_index("study_id")

    # Match carina rows
    sid_tr = df_z_tr["study_id"].tolist()
    sid_val = df_z_val["study_id"].tolist()
    c_tr = df_c_tr.loc[sid_tr].reset_index()
    c_val = df_c_val.loc[sid_val].reset_index()

    # Base features
    X_base_tr = df_z_tr[BASE_ZONAL_COLS].values
    X_base_val = df_z_val[BASE_ZONAL_COLS].values

    # Common tip info
    has_tip_tr = df_z_tr["has_valid_tip"].values[:, None]
    has_tip_val = df_z_val["has_valid_tip"].values[:, None]
    conf_tip_tr = df_z_tr["conf_tip"].values[:, None]
    conf_tip_val = df_z_val["conf_tip"].values[:, None]

    # Reconstruct raw tip coordinates in normalized frame
    # tip_x_norm = xc_norm + dx_norm; tip_y_norm = yc_norm + dy_norm
    xc_tr = c_tr["carina_x_norm"].values
    yc_tr = c_tr["carina_y_norm"].values
    xc_val = c_val["carina_x_norm"].values
    yc_val = c_val["carina_y_norm"].values

    raw_xc_tr = c_tr["raw_carina_x_norm"].values
    raw_yc_tr = c_tr["raw_carina_y_norm"].values
    raw_xc_val = c_val["raw_carina_x_norm"].values
    raw_yc_val = c_val["raw_carina_y_norm"].values

    tip_x_tr = xc_tr + df_z_tr["dx_norm"].values
    tip_y_tr = yc_tr + df_z_tr["dy_norm"].values
    tip_x_val = xc_val + df_z_val["dx_norm"].values
    tip_y_val = yc_val + df_z_val["dy_norm"].values

    conf_carina_tr = c_tr["carina_conf"].values[:, None]
    conf_carina_val = c_val["carina_conf"].values[:, None]
    is_fallback_tr = c_tr["is_carina_fallback"].values[:, None]
    is_fallback_val = c_val["is_carina_fallback"].values[:, None]

    # ETT coordinates
    ett_x_tr = (c_tr["ett_x_640"].values / 640.0)[:, None]
    ett_y_tr = (c_tr["ett_y_640"].values / 640.0)[:, None]
    ett_conf_tr = c_tr["ett_conf"].values[:, None]
    ett_x_val = (c_val["ett_x_640"].values / 640.0)[:, None]
    ett_y_val = (c_val["ett_y_640"].values / 640.0)[:, None]
    ett_conf_val = c_val["ett_conf"].values[:, None]

    aspect_tr = (c_tr["native_h"].values / np.maximum(c_tr["native_w"].values, 1.0))[:, None]
    aspect_val = (c_val["native_h"].values / np.maximum(c_val["native_w"].values, 1.0))[:, None]

    new_records = []

    # -------------------------------------------------------------
    # E01: Raw CarinaNet Point (no median fallback)
    # -------------------------------------------------------------
    dx_raw_tr = (tip_x_tr - raw_xc_tr)[:, None]
    dy_raw_tr = (tip_y_tr - raw_yc_tr)[:, None]
    r_raw_tr = np.sqrt(dx_raw_tr**2 + dy_raw_tr**2)
    cos_raw_tr = dx_raw_tr / np.maximum(r_raw_tr, 1e-6)
    sin_raw_tr = dy_raw_tr / np.maximum(r_raw_tr, 1e-6)

    dx_raw_val = (tip_x_val - raw_xc_val)[:, None]
    dy_raw_val = (tip_y_val - raw_yc_val)[:, None]
    r_raw_val = np.sqrt(dx_raw_val**2 + dy_raw_val**2)
    cos_raw_val = dx_raw_val / np.maximum(r_raw_val, 1e-6)
    sin_raw_val = dy_raw_val / np.maximum(r_raw_val, 1e-6)

    X_e01_tr = np.hstack([X_base_tr, dx_raw_tr, dy_raw_tr, r_raw_tr, cos_raw_tr, sin_raw_tr, has_tip_tr, conf_tip_tr, conf_carina_tr])
    X_e01_val = np.hstack([X_base_val, dx_raw_val, dy_raw_val, r_raw_val, cos_raw_val, sin_raw_val, has_tip_val, conf_tip_val, conf_carina_val])
    rec01, p01 = fit_and_eval_gbdt(X_e01_tr, y_tr, X_e01_val, y_val, is_traced, is_untraced, is_multicvc, "E01_raw_carinanet_point")
    new_records.append((rec01, p01))

    # -------------------------------------------------------------
    # E02: CarinaNet + Median Fallback Flag (Baseline GBDT feature set)
    # -------------------------------------------------------------
    dx_fb_tr = (tip_x_tr - xc_tr)[:, None]
    dy_fb_tr = (tip_y_tr - yc_tr)[:, None]
    r_fb_tr = np.sqrt(dx_fb_tr**2 + dy_fb_tr**2)
    cos_fb_tr = dx_fb_tr / np.maximum(r_fb_tr, 1e-6)
    sin_fb_tr = dy_fb_tr / np.maximum(r_fb_tr, 1e-6)

    dx_fb_val = (tip_x_val - xc_val)[:, None]
    dy_fb_val = (tip_y_val - yc_val)[:, None]
    r_fb_val = np.sqrt(dx_fb_val**2 + dy_fb_val**2)
    cos_fb_val = dx_fb_val / np.maximum(r_fb_val, 1e-6)
    sin_fb_val = dy_fb_val / np.maximum(r_fb_val, 1e-6)

    X_e02_tr = np.hstack([X_base_tr, dx_fb_tr, dy_fb_tr, r_fb_tr, cos_fb_tr, sin_fb_tr, has_tip_tr, conf_tip_tr, conf_carina_tr, is_fallback_tr])
    X_e02_val = np.hstack([X_base_val, dx_fb_val, dy_fb_val, r_fb_val, cos_fb_val, sin_fb_val, has_tip_val, conf_tip_val, conf_carina_val, is_fallback_val])
    rec02, p02 = fit_and_eval_gbdt(X_e02_tr, y_tr, X_e02_val, y_val, is_traced, is_untraced, is_multicvc, "E02_carina_median_fallback")
    new_records.append((rec02, p02))

    # -------------------------------------------------------------
    # E03: Pure Fixed Anatomical Median Origin (xc=0.480, yc=0.400)
    # -------------------------------------------------------------
    dx_m_tr = (tip_x_tr - 0.4805)[:, None]
    dy_m_tr = (tip_y_tr - 0.4000)[:, None]
    r_m_tr = np.sqrt(dx_m_tr**2 + dy_m_tr**2)
    dx_m_val = (tip_x_val - 0.4805)[:, None]
    dy_m_val = (tip_y_val - 0.4000)[:, None]
    r_m_val = np.sqrt(dx_m_val**2 + dy_m_val**2)

    X_e03_tr = np.hstack([X_base_tr, dx_m_tr, dy_m_tr, r_m_tr, has_tip_tr, conf_tip_tr])
    X_e03_val = np.hstack([X_base_val, dx_m_val, dy_m_val, r_m_val, has_tip_val, conf_tip_val])
    rec03, p03 = fit_and_eval_gbdt(X_e03_tr, y_tr, X_e03_val, y_val, is_traced, is_untraced, is_multicvc, "E03_fixed_anatomical_median_origin")
    new_records.append((rec03, p03))

    # -------------------------------------------------------------
    # E04: ETT-Anchored Fallback (Use ETT tip + 0.08H offset if carina fallback)
    # -------------------------------------------------------------
    # In adult radiographs, carina is ~3-5 cm below ETT tip (~0.08 normalized image height)
    ett_carina_y_tr = np.where((is_fallback_tr[:, 0] > 0.5) & (ett_conf_tr[:, 0] > 0.3), ett_y_tr[:, 0] + 0.08, yc_tr)
    ett_carina_y_val = np.where((is_fallback_val[:, 0] > 0.5) & (ett_conf_val[:, 0] > 0.3), ett_y_val[:, 0] + 0.08, yc_val)

    dy_ett_tr = (tip_y_tr - ett_carina_y_tr)[:, None]
    dy_ett_val = (tip_y_val - ett_carina_y_val)[:, None]

    X_e04_tr = np.hstack([X_e02_tr, dy_ett_tr, ett_conf_tr])
    X_e04_val = np.hstack([X_e02_val, dy_ett_val, ett_conf_val])
    rec04, p04 = fit_and_eval_gbdt(X_e04_tr, y_tr, X_e04_val, y_val, is_traced, is_untraced, is_multicvc, "E04_ett_anchored_carina_fallback")
    new_records.append((rec04, p04))

    # -------------------------------------------------------------
    # E05: Confidence-Gated Carina (Carina only if conf >= 0.50)
    # -------------------------------------------------------------
    gate_tr = (conf_carina_tr > 0.50).astype(float)
    gate_val = (conf_carina_val > 0.50).astype(float)
    X_e05_tr = np.hstack([X_e02_tr, gate_tr, gate_tr * dx_fb_tr, gate_tr * dy_fb_tr])
    X_e05_val = np.hstack([X_e02_val, gate_val, gate_val * dx_fb_val, gate_val * dy_fb_val])
    rec05, p05 = fit_and_eval_gbdt(X_e05_tr, y_tr, X_e05_val, y_val, is_traced, is_untraced, is_multicvc, "E05_confidence_gated_carina_origin")
    new_records.append((rec05, p05))

    # -------------------------------------------------------------
    # E06: Precision-Weighted Origin Fusion
    # -------------------------------------------------------------
    # Weighted average of raw carina and median prior based on model confidence
    w_carina_tr = np.clip(conf_carina_tr, 0.0, 1.0)
    fused_yc_tr = w_carina_tr * raw_yc_tr[:, None] + (1.0 - w_carina_tr) * 0.4000
    w_carina_val = np.clip(conf_carina_val, 0.0, 1.0)
    fused_yc_val = w_carina_val * raw_yc_val[:, None] + (1.0 - w_carina_val) * 0.4000

    dy_fused_tr = (tip_y_tr[:, None] - fused_yc_tr)
    dy_fused_val = (tip_y_val[:, None] - fused_yc_val)

    X_e06_tr = np.hstack([X_e02_tr, dy_fused_tr, w_carina_tr])
    X_e06_val = np.hstack([X_e02_val, dy_fused_val, w_carina_val])
    rec06, p06 = fit_and_eval_gbdt(X_e06_tr, y_tr, X_e06_val, y_val, is_traced, is_untraced, is_multicvc, "E06_precision_weighted_carina_fusion")
    new_records.append((rec06, p06))

    # -------------------------------------------------------------
    # E07: Thoracic Aspect-Ratio Scaling
    # -------------------------------------------------------------
    dy_scaled_tr = dy_fb_tr * aspect_tr
    dy_scaled_val = dy_fb_val * aspect_val
    X_e07_tr = np.hstack([X_e02_tr, dy_scaled_tr, aspect_tr])
    X_e07_val = np.hstack([X_e02_val, dy_scaled_val, aspect_val])
    rec07, p07 = fit_and_eval_gbdt(X_e07_tr, y_tr, X_e07_val, y_val, is_traced, is_untraced, is_multicvc, "E07_thoracic_aspect_ratio_scaling")
    new_records.append((rec07, p07))

    # -------------------------------------------------------------
    # E08: Relative Displacement Normalization (Anatomical mm approximation)
    # -------------------------------------------------------------
    # Assuming average chest width ~350mm, height ~400mm
    dx_mm_tr = dx_fb_tr * 350.0
    dy_mm_tr = dy_fb_tr * 400.0
    r_mm_tr = np.sqrt(dx_mm_tr**2 + dy_mm_tr**2)
    dx_mm_val = dx_fb_val * 350.0
    dy_mm_val = dy_fb_val * 400.0
    r_mm_val = np.sqrt(dx_mm_val**2 + dy_mm_val**2)

    X_e08_tr = np.hstack([X_e02_tr, dx_mm_tr, dy_mm_tr, r_mm_tr])
    X_e08_val = np.hstack([X_e02_val, dx_mm_val, dy_mm_val, r_mm_val])
    rec08, p08 = fit_and_eval_gbdt(X_e08_tr, y_tr, X_e08_val, y_val, is_traced, is_untraced, is_multicvc, "E08_anatomical_mm_displacement")
    new_records.append((rec08, p08))

    # -------------------------------------------------------------
    # E09: Non-Linear Radial & Log Displacement
    # -------------------------------------------------------------
    log_r_tr = np.log1p(np.maximum(0.0, r_fb_tr * 100.0))
    log_r_val = np.log1p(np.maximum(0.0, r_fb_val * 100.0))
    signed_sqrt_dy_tr = np.sign(dy_fb_tr) * np.sqrt(np.abs(dy_fb_tr))
    signed_sqrt_dy_val = np.sign(dy_fb_val) * np.sqrt(np.abs(dy_fb_val))

    X_e09_tr = np.hstack([X_e02_tr, log_r_tr, signed_sqrt_dy_tr])
    X_e09_val = np.hstack([X_e02_val, log_r_val, signed_sqrt_dy_val])
    rec09, p09 = fit_and_eval_gbdt(X_e09_tr, y_tr, X_e09_val, y_val, is_traced, is_untraced, is_multicvc, "E09_nonlinear_radial_log_displacement")
    new_records.append((rec09, p09))

    # -------------------------------------------------------------
    # E10: Midline Lateral Rotation Correction
    # -------------------------------------------------------------
    # Compensates for slight patient tilt (lateral skew)
    dx_corr_tr = dx_fb_tr - 0.15 * dy_fb_tr
    dx_corr_val = dx_fb_val - 0.15 * dy_fb_val
    X_e10_tr = np.hstack([X_e02_tr, dx_corr_tr])
    X_e10_val = np.hstack([X_e02_val, dx_corr_val])
    rec10, p10 = fit_and_eval_gbdt(X_e10_tr, y_tr, X_e10_val, y_val, is_traced, is_untraced, is_multicvc, "E10_midline_lateral_rotation_correction")
    new_records.append((rec10, p10))

    # -------------------------------------------------------------
    # E11: Non-Linear Zonal Density Integrals (sqrt and quadratic fractions)
    # -------------------------------------------------------------
    fracs_tr = df_z_tr[["frac_neck", "frac_upper_svc", "frac_borderline", "frac_caj_normal", "frac_atrium_deep"]].values
    fracs_val = df_z_val[["frac_neck", "frac_upper_svc", "frac_borderline", "frac_caj_normal", "frac_atrium_deep"]].values
    sqrt_fracs_tr = np.sqrt(np.maximum(0.0, fracs_tr))
    sqrt_fracs_val = np.sqrt(np.maximum(0.0, fracs_val))

    X_e11_tr = np.hstack([X_e02_tr, sqrt_fracs_tr])
    X_e11_val = np.hstack([X_e02_val, sqrt_fracs_val])
    rec11, p11 = fit_and_eval_gbdt(X_e11_tr, y_tr, X_e11_val, y_val, is_traced, is_untraced, is_multicvc, "E11_nonlinear_zonal_density_integrals")
    new_records.append((rec11, p11))

    # -------------------------------------------------------------
    # E12: Combined Optimal Geometric Feature Set (Stage 1 Grand Champion)
    # -------------------------------------------------------------
    X_e12_tr = np.hstack([
        X_e02_tr,
        dy_fused_tr, w_carina_tr,
        dy_ett_tr, ett_conf_tr,
        dy_scaled_tr, aspect_tr,
        log_r_tr, signed_sqrt_dy_tr,
        sqrt_fracs_tr
    ])
    X_e12_val = np.hstack([
        X_e02_val,
        dy_fused_val, w_carina_val,
        dy_ett_val, ett_conf_val,
        dy_scaled_val, aspect_val,
        log_r_val, signed_sqrt_dy_val,
        sqrt_fracs_val
    ])
    rec12, p12 = fit_and_eval_gbdt(X_e12_tr, y_tr, X_e12_val, y_val, is_traced, is_untraced, is_multicvc, "E12_stage1_optimal_geometry_grand_champion")
    new_records.append((rec12, p12))

    # Save to Master Ledger & runs/val_preds/
    ledger_path = REPORTS_DIR / "master_ablation_ledger.csv"
    existing_df = pd.read_csv(ledger_path)

    for rec, preds in new_records:
        eid = rec["exp_id"]
        np.save(str(VAL_PREDS_DIR / f"val_preds_{eid}.npy"), preds)
        print(f"[{eid}] All: {rec['Val_Macro_AUROC_2558']:.4f} (N:{rec['Normal_AUROC']:.4f}, B:{rec['Borderline_AUROC']:.4f}, A:{rec['Abnormal_AUROC']:.4f}) | Traced: {rec['Traced_778_AUROC']:.4f} | Untraced: {rec['Untraced_1780_AUROC']:.4f} | MultiCVC: {rec['MultiCVC_332_AUROC']:.4f}")

    new_df = pd.DataFrame([r[0] for r in new_records])
    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    updated_df.drop_duplicates(subset=["exp_id"], keep="last", inplace=True)

    updated_df.to_csv(REPORTS_DIR / "master_ablation_ledger.csv", index=False)
    updated_df.to_parquet(REPORTS_DIR / "master_ablation_ledger.parquet", index=False)
    print("\nMaster Ablation Ledger updated successfully!")


if __name__ == "__main__":
    main()

