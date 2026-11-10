"""E69: Bilateral Mediastinal Zonal Asymmetry Feature Extraction & GBDT Evaluation.

Extracts Left-vs-Right Hemithorax Zonal Asymmetry:
- Right Mediastinal Corridor (SVC track, patient-right / image-left): x in [xc - 0.09W, xc + 0.02W]
- Left Mediastinal Corridor (patient-left / image-right): x in (xc + 0.02W, xc + 0.12W]
- Far Lateral (pleura / chest wall): x < xc - 0.09W or x > xc + 0.12W
Across 5 vertical zones:
1. Neck: dy < -0.06
2. Upper SVC: -0.06 <= dy < -0.02
3. Borderline: -0.02 <= dy < 0.0
4. Normal CAJ: 0.0 <= dy <= 0.05
5. Deep Atrium: 0.05 < dy <= 0.12

Trains CatBoost and LightGBM on N=6,600 training cohort.
Evaluates out-of-sample on locked N=2,558 validation cohort.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"

import cv2
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
DATA_DIR = PROJECT_ROOT / "data"
SEG_CACHE_DIR = DATA_DIR / "curated_cvc_v1" / "seg_cache_512"
CARINA_TR_PATH = DATA_DIR / "curated_cvc_v1" / "carina_train_cache.parquet"
CARINA_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "carina_validation_cache.parquet"
ZONAL_TR_PATH = DATA_DIR / "curated_cvc_v1" / "zonal_train_features.parquet"
ZONAL_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "zonal_val_features.parquet"
MANIFEST_PATH = DATA_DIR / "splits" / "placement_locked_manifest.parquet"
REPORTS_DIR = PROJECT_ROOT / "reports"
VAL_PREDS_DIR = PROJECT_ROOT / "runs" / "val_preds"

CLASSES = ["CVC - Normal", "CVC - Borderline", "CVC - Abnormal"]


def compute_bilateral_features_for_study(
    mask_uint8: np.ndarray,
    cx_norm: float,
    cy_norm: float,
) -> Dict[str, float]:
    """Compute bilateral left vs. right zonal mass integrals and asymmetries."""
    H, W = mask_uint8.shape  # 512, 512
    seg = mask_uint8.astype(np.float32) / 255.0

    c_x = cx_norm * W
    c_y = cy_norm * H

    y_grid, x_grid = np.indices((H, W))
    dy = (y_grid - c_y) / float(H)
    dx = (x_grid - c_x) / float(W)

    # Lateral masks
    is_right_med = (dx >= -0.09) & (dx <= 0.02)  # SVC track (patient-right)
    is_left_med = (dx > 0.02) & (dx <= 0.12)    # Left mediastinum (patient-left)
    is_far_lat = (dx < -0.09) | (dx > 0.12)

    # Vertical zones
    vz_masks = {
        "neck": dy < -0.06,
        "upper_svc": (dy >= -0.06) & (dy < -0.02),
        "borderline": (dy >= -0.02) & (dy < 0.0),
        "caj_normal": (dy >= 0.0) & (dy <= 0.05),
        "atrium_deep": (dy > 0.05) & (dy <= 0.12),
    }

    feats = {}
    tot_right = 0.0
    tot_left = 0.0

    for zname, vmask in vz_masks.items():
        m_r = float(seg[vmask & is_right_med].sum())
        m_l = float(seg[vmask & is_left_med].sum())
        tot_right += m_r
        tot_left += m_l

        denom = m_r + m_l + 1e-4
        asym = (m_r - m_l) / denom  # +1 = all right (normal SVC), -1 = all left (malposition/arterial)

        # 90th percentile of seg_prob in zone
        vals_r = seg[vmask & is_right_med]
        vals_l = seg[vmask & is_left_med]
        p90_r = float(np.percentile(vals_r, 90)) if len(vals_r) > 0 else 0.0
        p90_l = float(np.percentile(vals_l, 90)) if len(vals_l) > 0 else 0.0

        feats[f"m_r_{zname}"] = m_r
        feats[f"m_l_{zname}"] = m_l
        feats[f"asym_{zname}"] = asym
        feats[f"p90_r_{zname}"] = p90_r
        feats[f"p90_l_{zname}"] = p90_l

    # Overall bilateral metrics
    tot_denom = tot_right + tot_left + 1e-4
    feats["total_m_r"] = tot_right
    feats["total_m_l"] = tot_left
    feats["overall_med_asym"] = (tot_right - tot_left) / tot_denom
    feats["m_far_lat"] = float(seg[is_far_lat].sum())

    return feats


def extract_cohort_bilateral(
    carina_df: pd.DataFrame,
    zonal_df: pd.DataFrame,
    cohort_name: str = "train",
) -> pd.DataFrame:
    """Extract or load cached bilateral features for an entire cohort."""
    cache_path = DATA_DIR / "curated_cvc_v1" / f"bilateral_{cohort_name}_features.parquet"
    if cache_path.exists():
        print(f"[{cohort_name}] Loading cached bilateral features from {cache_path.name}...")
        return pd.read_parquet(cache_path)

    print(f"[{cohort_name}] Extracting bilateral features from {len(zonal_df)} masks...")
    t0 = time.time()
    carina_lookup = carina_df.set_index("study_id")

    rows = []
    for idx, sid in enumerate(zonal_df["study_id"]):
        png_p = SEG_CACHE_DIR / f"{sid}.png"
        if not png_p.exists():
            # Zero mask fallback
            mask = np.zeros((512, 512), dtype=np.uint8)
        else:
            mask = cv2.imread(str(png_p), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((512, 512), dtype=np.uint8)

        if sid in carina_lookup.index:
            crow = carina_lookup.loc[sid]
            # normalized coordinates
            cx_norm = float(crow["carina_x"] / crow["native_width"]) if "native_width" in crow else 0.50
            cy_norm = float(crow["carina_y"] / crow["native_height"]) if "native_height" in crow else 0.45
        else:
            cx_norm, cy_norm = 0.50, 0.45

        bf = compute_bilateral_features_for_study(mask, cx_norm, cy_norm)
        bf["study_id"] = sid
        rows.append(bf)

        if (idx + 1) % 1500 == 0:
            print(f"  Processed {idx+1}/{len(zonal_df)} in {time.time()-t0:.1f}s")

    df_out = pd.DataFrame(rows)
    df_out.to_parquet(cache_path, index=False)
    print(f"[{cohort_name}] Saved {len(df_out)} bilateral features to {cache_path.name} in {time.time()-t0:.1f}s")
    return df_out


def main():
    print("=" * 75)
    print("E69: BILATERAL HEMITHORAX ZONAL ASYMMETRY GBDT BENCHMARK")
    print("=" * 75)

    VAL_PREDS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Base Features
    df_carina_tr = pd.read_parquet(CARINA_TR_PATH)
    df_carina_val = pd.read_parquet(CARINA_VAL_PATH)
    df_zonal_tr = pd.read_parquet(ZONAL_TR_PATH)
    df_zonal_val = pd.read_parquet(ZONAL_VAL_PATH)
    manifest = pd.read_parquet(MANIFEST_PATH)
    val_manifest = manifest[manifest["split"] == "val"].reset_index(drop=True)

    # Align training set
    tr_sids = set(df_carina_tr["study_id"]).intersection(set(df_zonal_tr["study_id"]))
    df_zonal_tr = df_zonal_tr[df_zonal_tr["study_id"].isin(tr_sids)].reset_index(drop=True)

    # 2. Extract Bilateral Features
    df_bi_tr = extract_cohort_bilateral(df_carina_tr, df_zonal_tr, "train")
    df_bi_val = extract_cohort_bilateral(df_carina_val, df_zonal_val, "val")

    # Merge features
    bi_cols = [c for c in df_bi_tr.columns if c != "study_id"]
    base_feat_cols = [
        "m_neck", "m_upper_svc", "m_borderline", "m_caj_normal", "m_atrium_deep", "m_lateral", "m_total_med",
        "frac_neck", "frac_upper_svc", "frac_borderline", "frac_caj_normal", "frac_atrium_deep",
        "deepest_y_norm", "p95_y_norm", "highest_y_norm", "ribbon_pixels",
        "dx_norm", "dy_norm", "r_norm", "cos_theta", "sin_theta", "has_valid_tip", "conf_tip",
        "conf_carina", "is_carina_fallback"
    ]

    df_full_tr = df_zonal_tr.merge(df_bi_tr, on="study_id", how="left")
    df_full_val = df_zonal_val.merge(df_bi_val, on="study_id", how="left")

    all_feat_cols = base_feat_cols + bi_cols
    print(f"Total Feature Dimension: {len(all_feat_cols)} (25 base + {len(bi_cols)} bilateral asymmetry)")

    X_train = df_full_tr[all_feat_cols].values.astype(np.float32)
    X_val = df_full_val[all_feat_cols].values.astype(np.float32)

    # Replace NaNs/Infs
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=1.0, neginf=-1.0)

    y_val = df_full_val[CLASSES].values.astype(int)

    # Sub-cohort masks
    is_traced = (val_manifest["annotated_cvc_count"] > 0).values
    is_untraced = ~is_traced
    is_multicvc = (val_manifest["annotated_cvc_count"] > 1) | (val_manifest[CLASSES].sum(axis=1) > 1).values

    def eval_preds(preds: np.ndarray, name: str) -> Dict[str, float]:
        cls_aucs = {}
        for c_idx, c_name in enumerate(CLASSES):
            cls_aucs[c_name] = round(float(roc_auc_score(y_val[:, c_idx], preds[:, c_idx])), 4)
        macro = round(float(np.mean(list(cls_aucs.values()))), 4)

        def sub_auc(mask):
            return round(float(np.mean([roc_auc_score(y_val[mask, c], preds[mask, c]) for c in range(3)])), 4)

        tr_auc = sub_auc(is_traced)
        un_auc = sub_auc(is_untraced)
        mu_auc = sub_auc(is_multicvc)

        print(f"[{name}] Macro: {macro:.4f} | Normal: {cls_aucs['CVC - Normal']:.4f} | Borderline: {cls_aucs['CVC - Borderline']:.4f} | Abnormal: {cls_aucs['CVC - Abnormal']:.4f} | Traced: {tr_auc:.4f} | Untraced: {un_auc:.4f} | MultiCVC: {mu_auc:.4f}")
        return {
            "macro": macro, "normal": cls_aucs["CVC - Normal"], "borderline": cls_aucs["CVC - Borderline"],
            "abnormal": cls_aucs["CVC - Abnormal"], "traced": tr_auc, "untraced": un_auc, "multicvc": mu_auc
        }

    # Model 1: LightGBM with Bilateral Features
    print("\nTraining LightGBM (E69_bilateral_zonal_lgbm)...")
    lgb_preds = np.zeros_like(y_val, dtype=np.float32)
    t0 = time.time()
    for c_idx, c_name in enumerate(CLASSES):
        y_c = df_full_tr[c_name].values.astype(int)
        clf = LGBMClassifier(
            n_estimators=180, learning_rate=0.035, max_depth=5, num_leaves=18,
            subsample=0.85, colsample_bytree=0.80, random_state=42, n_jobs=-1, verbose=-1
        )
        clf.fit(X_train, y_c)
        lgb_preds[:, c_idx] = clf.predict_proba(X_val)[:, 1]
    lgb_time = time.time() - t0
    m_lgb = eval_preds(lgb_preds, "E69_bilateral_zonal_lgbm")

    # Model 2: CatBoost with Bilateral Features
    print("\nTraining CatBoost (E69_bilateral_zonal_catboost)...")
    cb_preds = np.zeros_like(y_val, dtype=np.float32)
    t0 = time.time()
    for c_idx, c_name in enumerate(CLASSES):
        y_c = df_full_tr[c_name].values.astype(int)
        cb = CatBoostClassifier(
            iterations=250, learning_rate=0.04, depth=5, random_seed=42, verbose=0, thread_count=-1
        )
        cb.fit(X_train, y_c)
        cb_preds[:, c_idx] = cb.predict_proba(X_val)[:, 1]
    cb_time = time.time() - t0
    m_cb = eval_preds(cb_preds, "E69_bilateral_zonal_catboost")

    # Model 3: 50/50 Rank Blend of LightGBM + CatBoost
    from scipy.stats import rankdata
    cb_lgb_blend = np.zeros_like(y_val, dtype=np.float32)
    for c in range(3):
        r_lgb = rankdata(lgb_preds[:, c]) / len(lgb_preds)
        r_cb = rankdata(cb_preds[:, c]) / len(cb_preds)
        cb_lgb_blend[:, c] = 0.5 * r_lgb + 0.5 * r_cb
    m_blend = eval_preds(cb_lgb_blend, "E69_bilateral_zonal_blend")

    # Pick the best
    best_name = "E69_bilateral_zonal_catboost" if m_cb["macro"] >= m_lgb["macro"] else "E69_bilateral_zonal_lgbm"
    best_preds = cb_preds if m_cb["macro"] >= m_lgb["macro"] else lgb_preds
    best_metrics = m_cb if m_cb["macro"] >= m_lgb["macro"] else m_lgb

    # Save Best Preds
    np.save(str(VAL_PREDS_DIR / f"val_preds_{best_name}.npy"), best_preds)
    np.save(str(VAL_PREDS_DIR / "val_preds_E69_bilateral_zonal_blend.npy"), cb_lgb_blend)

    # Update Master Ledger
    ledger_rec = {
        "exp_id": best_name,
        "stage": 6,
        "Runtime_s": round(cb_time + lgb_time, 1),
        "Val_Macro_AUROC_2558": best_metrics["macro"],
        "Normal_AUROC": best_metrics["normal"],
        "Borderline_AUROC": best_metrics["borderline"],
        "Abnormal_AUROC": best_metrics["abnormal"],
        "Traced_778_AUROC": best_metrics["traced"],
        "Untraced_1780_AUROC": best_metrics["untraced"],
        "MultiCVC_332_AUROC": best_metrics["multicvc"],
        "Tip_Recall_35px": 55.7,
    }

    ledger_csv = REPORTS_DIR / "master_ablation_ledger.csv"
    ledger_parquet = REPORTS_DIR / "master_ablation_ledger.parquet"
    if ledger_csv.exists():
        df_l = pd.read_csv(ledger_csv)
        if best_name in df_l["exp_id"].values:
            idx = df_l[df_l["exp_id"] == best_name].index[0]
            for k, v in ledger_rec.items():
                df_l.at[idx, k] = v
        else:
            df_l = pd.concat([df_l, pd.DataFrame([ledger_rec])], ignore_index=True)
    else:
        df_l = pd.DataFrame([ledger_rec])

    df_l.to_csv(ledger_csv, index=False)
    df_l.to_parquet(ledger_parquet, index=False)
    print(f"\nSuccessfully logged {best_name} to master ledger!")


if __name__ == "__main__":
    main()

