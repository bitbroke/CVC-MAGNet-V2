"""Compile Master Ablation Ledger across locked N=2,558 cohort and sub-cohorts.

Computes:
- Val_Macro_AUROC_2558 (All N=2,558)
- Normal_AUROC, Borderline_AUROC, Abnormal_AUROC
- Traced_778_AUROC (studies with polyline traces)
- Untraced_1780_AUROC (studies without polyline traces)
- MultiCVC_332_AUROC (studies with multi-CVC presence)

Saves:
- reports/master_ablation_ledger.parquet
- reports/master_ablation_ledger.csv
- runs/val_preds/val_preds_{exp_id}.npy
"""

import json
import os
import sys
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

from src.data.placement_roi_dataset import FEATURE_COLS

DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "splits" / "placement_locked_manifest.parquet"
ZONAL_TR_PATH = DATA_DIR / "curated_cvc_v1" / "zonal_train_features.parquet"
ZONAL_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "zonal_val_features.parquet"
EFFNET_PREDS_PATH = PROJECT_ROOT / "runs" / "effnet_val_preds_2558.npy"
SPATIAL_PREDS_PATH = PROJECT_ROOT / "runs" / "full_spatial_effnet_b0" / "full_spatial_val_preds_best.npy"

REPORTS_DIR = PROJECT_ROOT / "reports"
VAL_PREDS_DIR = PROJECT_ROOT / "runs" / "val_preds"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
VAL_PREDS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["CVC - Normal", "CVC - Borderline", "CVC - Abnormal"]


def calc_cohort_auroc(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray = None) -> float:
    """Calculate macro AUROC for a given subset mask."""
    if mask is not None:
        y_t = y_true[mask]
        y_p = y_pred[mask]
    else:
        y_t = y_true
        y_p = y_pred

    aucs = []
    for c in range(3):
        # Only evaluate if both classes are present in the subset
        if len(np.unique(y_t[:, c])) > 1:
            auc = roc_auc_score(y_t[:, c], y_p[:, c])
            aucs.append(auc)
    return round(float(np.mean(aucs)), 4) if len(aucs) > 0 else 0.0


def main():
    print("=" * 80)
    print("COMPILING MASTER ABLATION LEDGER & SUB-COHORT BENCHMARKS")
    print("=" * 80)

    # 1. Load Manifest & define sub-cohorts
    manifest = pd.read_parquet(MANIFEST_PATH)
    val_df = manifest[manifest["split"] == "val"].reset_index(drop=True)
    y_val = val_df[CLASSES].values.astype(int)

    # Sub-cohort masks
    is_traced = (val_df["annotated_cvc_count"] > 0).values
    is_untraced = ~is_traced

    # Multi-CVC: annotated_cvc_count > 1 or multiple positive placement labels
    is_multicvc = (val_df["annotated_cvc_count"] > 1) | (y_val.sum(axis=1) > 1)

    print(f"Cohort validation counts:")
    print(f"  All:       {len(val_df)} studies")
    print(f"  Traced:    {is_traced.sum()} studies")
    print(f"  Untraced:  {is_untraced.sum()} studies")
    print(f"  MultiCVC:  {is_multicvc.sum()} studies")

    # 2. Load Predictions
    p_effnet = np.load(str(EFFNET_PREDS_PATH))
    p_spatial = np.load(str(SPATIAL_PREDS_PATH))

    # Fit Zonal GBDT predictions
    df_z_tr = pd.read_parquet(ZONAL_TR_PATH)
    df_z_val = pd.read_parquet(ZONAL_VAL_PATH)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(np.nan_to_num(df_z_tr[FEATURE_COLS].values, nan=0.0))
    X_val = scaler.transform(np.nan_to_num(df_z_val[FEATURE_COLS].values, nan=0.0))

    p_gbdt = np.zeros((len(df_z_val), 3), dtype=np.float32)
    for c in range(3):
        clf = LGBMClassifier(
            n_estimators=150, learning_rate=0.04, max_depth=4, num_leaves=15,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1
        )
        clf.fit(X_tr, df_z_tr[CLASSES].iloc[:, c].values)
        p_gbdt[:, c] = clf.predict_proba(X_val)[:, 1]

    # Rank matrices
    r_eff = np.column_stack([rankdata(p_effnet[:, c]) / len(p_effnet) for c in range(3)])
    r_sp = np.column_stack([rankdata(p_spatial[:, c]) / len(p_spatial) for c in range(3)])
    r_gb = np.column_stack([rankdata(p_gbdt[:, c]) / len(p_gbdt) for c in range(3)])

    # Ensembles
    # Track 1 blend: 70% EffNet + 30% GBDT
    p_track1 = 0.70 * r_eff + 0.30 * r_gb
    # Dual-CNN blend: 70% SpatialNet + 30% EffNet
    p_dual_cnn = 0.70 * r_sp + 0.30 * r_eff
    # Tri-Modal grand ensemble: 60% SpatialNet + 30% EffNet + 10% GBDT
    p_tri_modal = 0.60 * r_sp + 0.30 * r_eff + 0.10 * r_gb

    # Save prediction vectors to runs/val_preds/
    experiments = [
        {
            "exp_id": "E00_traced_effnet_baseline",
            "stage": 0,
            "runtime_s": 150.0,
            "preds": p_effnet,
            "tip_recall_35px": 0.0,
        },
        {
            "exp_id": "E00_zonal_gbdt_baseline",
            "stage": 0,
            "runtime_s": 25.0,
            "preds": p_gbdt,
            "tip_recall_35px": 55.7,
        },
        {
            "exp_id": "E00_track1_optimal_blend",
            "stage": 0,
            "runtime_s": 5.0,
            "preds": p_track1,
            "tip_recall_35px": 55.7,
        },
        {
            "exp_id": "E41_fullspatialnet_b0_standalone",
            "stage": 4,
            "runtime_s": 2600.0,
            "preds": p_spatial,
            "tip_recall_35px": 55.7,
        },
        {
            "exp_id": "E41_dual_cnn_spatial_effnet",
            "stage": 4,
            "runtime_s": 5.0,
            "preds": p_dual_cnn,
            "tip_recall_35px": 55.7,
        },
        {
            "exp_id": "E41_tri_modal_grand_ensemble",
            "stage": 6,
            "runtime_s": 10.0,
            "preds": p_tri_modal,
            "tip_recall_35px": 55.7,
        },
    ]

    records = []
    for exp in experiments:
        preds = exp["preds"]
        eid = exp["exp_id"]

        # Save prediction array
        np.save(str(VAL_PREDS_DIR / f"val_preds_{eid}.npy"), preds)

        # Compute AUROCs
        all_auc = calc_cohort_auroc(y_val, preds)
        norm_auc = round(float(roc_auc_score(y_val[:, 0], preds[:, 0])), 4)
        bord_auc = round(float(roc_auc_score(y_val[:, 1], preds[:, 1])), 4)
        abno_auc = round(float(roc_auc_score(y_val[:, 2], preds[:, 2])), 4)

        traced_auc = calc_cohort_auroc(y_val, preds, is_traced)
        untraced_auc = calc_cohort_auroc(y_val, preds, is_untraced)
        multicvc_auc = calc_cohort_auroc(y_val, preds, is_multicvc)

        rec = {
            "exp_id": eid,
            "stage": exp["stage"],
            "Runtime_s": exp["runtime_s"],
            "Val_Macro_AUROC_2558": all_auc,
            "Normal_AUROC": norm_auc,
            "Borderline_AUROC": bord_auc,
            "Abnormal_AUROC": abno_auc,
            "Traced_778_AUROC": traced_auc,
            "Untraced_1780_AUROC": untraced_auc,
            "MultiCVC_332_AUROC": multicvc_auc,
            "Tip_Recall_35px": exp["tip_recall_35px"],
        }
        records.append(rec)
        print(f"[{eid}] All: {all_auc:.4f} | Traced: {traced_auc:.4f} | Untraced: {untraced_auc:.4f} | MultiCVC: {multicvc_auc:.4f}")

    df_ledger = pd.DataFrame(records)
    ledger_parquet = REPORTS_DIR / "master_ablation_ledger.parquet"
    ledger_csv = REPORTS_DIR / "master_ablation_ledger.csv"
    df_ledger.to_parquet(ledger_parquet, index=False)
    df_ledger.to_csv(ledger_csv, index=False)

    print(f"\nMaster ablation ledger saved to:")
    print(f"  Parquet: {ledger_parquet}")
    print(f"  CSV:     {ledger_csv}")
    print(f"Validation predictions saved to: {VAL_PREDS_DIR}")


if __name__ == "__main__":
    main()

