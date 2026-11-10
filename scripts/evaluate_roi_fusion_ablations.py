"""Ablation and Ensembling Evaluation for Mediastinal RoI Fusion Network.

Evaluates across the full locked N=2,558 validation cohort:
1. Dual-Branch Fusion (Visual + Geometric)
2. Visual-Only Branch (force_drop_geom=True)
3. Geometric-Only Branch (force_drop_visual=True)
4. Hybrid Ensemble: Zonal GBDT + MediastinalFusionNet
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("C:/Ankshit\CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.placement_roi_dataset import CachedMediastinalRoIDataset, FEATURE_COLS
from src.models.placement_roi_fusion import MediastinalFusionNet

DATA_DIR = PROJECT_ROOT / "data"
ZONAL_TRAIN_PATH = DATA_DIR / "curated_cvc_v1" / "zonal_train_features.parquet"
ZONAL_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "zonal_val_features.parquet"
BEST_CKPT_PATH = PROJECT_ROOT / "runs" / "roi_fusion_convnext_tiny_v2" / "checkpoints" / "best.pt"
GBDT_REPORT_PATH = PROJECT_ROOT / "runs" / "zonal_gbdt_evaluation_report.json"

CLASSES = ["CVC - Normal", "CVC - Borderline", "CVC - Abnormal"]


def eval_model_mode(model, val_loader, device, force_drop_geom=False, force_drop_visual=False):
    model.eval()
    all_preds = []
    all_targets = []
    all_sids = []

    with torch.no_grad():
        for batch in val_loader:
            roi = batch["roi"].to(device)
            geom = batch["geom"].to(device)
            target = batch["target"]
            sids = batch["study_id"]

            with autocast(dtype=torch.float16):
                out = model(roi, geom, force_drop_geom=force_drop_geom, force_drop_visual=force_drop_visual)
                probs = out["probs"].cpu().numpy()

            all_preds.append(probs)
            all_targets.append(target.numpy())
            all_sids.extend(sids)

    Y_pred = np.vstack(all_preds)
    Y_true = np.vstack(all_targets)

    aurocs = {}
    macro = []
    for c_idx, c_name in enumerate(CLASSES):
        auc = roc_auc_score(Y_true[:, c_idx], Y_pred[:, c_idx])
        aurocs[c_name] = round(float(auc), 4)
        macro.append(auc)
    aurocs["macro_auroc"] = round(float(np.mean(macro)), 4)
    return aurocs, Y_pred, Y_true, all_sids


def main():
    print("=" * 70)
    print("MEDIASTINAL ROI FUSION: FULL-COHORT ABLATION & ENSEMBLE EVALUATION")
    print("Locked Validation Cohort: N=2,558 Studies")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load DataFrames
    df_train = pd.read_parquet(ZONAL_TRAIN_PATH)
    df_val = pd.read_parquet(ZONAL_VAL_PATH)

    scaler = StandardScaler()
    scaler.fit(np.nan_to_num(df_train[FEATURE_COLS].values.astype(np.float32), nan=0.0))

    val_dataset = CachedMediastinalRoIDataset(df_val, scaler, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Load model
    print(f"Loading checkpoint: {BEST_CKPT_PATH}")
    ckpt = torch.load(BEST_CKPT_PATH, map_location="cpu")
    model = MediastinalFusionNet(
        backbone_name="convnext_tiny",
        pretrained=False,
        geom_in_dim=25,
        visual_dim=128,
        geom_dim=32,
        num_classes=3,
        modality_dropout_p=0.0,
    )
    # Checkpoint contains float16 state dict; load into float32 model
    state_dict = {k: v.float() for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("\n--- 1. Dual-Branch Fusion Mode (Visual + Geometric) ---")
    res_fusion, preds_fusion, Y_true, sids = eval_model_mode(model, val_loader, device)
    for k, v in res_fusion.items():
        print(f"  {k}: {v}")

    print("\n--- 2. Visual-Only Ablation (force_drop_geom=True) ---")
    res_vis, preds_vis, _, _ = eval_model_mode(model, val_loader, device, force_drop_geom=True)
    for k, v in res_vis.items():
        print(f"  {k}: {v}")

    print("\n--- 3. Geometric-Only Ablation (force_drop_visual=True) ---")
    res_geom, preds_geom, _, _ = eval_model_mode(model, val_loader, device, force_drop_visual=True)
    for k, v in res_geom.items():
        print(f"  {k}: {v}")

    # 4. Hybrid Ensemble with Zonal GBDT
    print("\n--- 4. Hybrid Ensemble (MediastinalFusionNet + Zonal GBDT) ---")
    # Load and evaluate GBDT
    from lightgbm import LGBMClassifier
    X_train = scaler.transform(np.nan_to_num(df_train[FEATURE_COLS].values, nan=0.0))
    X_val = scaler.transform(np.nan_to_num(df_val[FEATURE_COLS].values, nan=0.0))

    gbdt_preds = np.zeros_like(preds_fusion)
    for c_idx, c_name in enumerate(CLASSES):
        y_c_train = df_train[c_name].values.astype(int)
        clf = LGBMClassifier(
            n_estimators=150,
            learning_rate=0.04,
            max_depth=4,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(X_train, y_c_train)
        gbdt_preds[:, c_idx] = clf.predict_proba(X_val)[:, 1]

    # Evaluate pure GBDT
    gbdt_res = {}
    gbdt_macro = []
    for c_idx, c_name in enumerate(CLASSES):
        auc = roc_auc_score(Y_true[:, c_idx], gbdt_preds[:, c_idx])
        gbdt_res[c_name] = round(float(auc), 4)
        gbdt_macro.append(auc)
    gbdt_res["macro_auroc"] = round(float(np.mean(gbdt_macro)), 4)
    print("  Zonal GBDT (standalone):", gbdt_res)

    # Blends
    for w_nn in [0.2, 0.4, 0.5, 0.6, 0.8]:
        w_gbdt = 1.0 - w_nn
        blend_preds = w_nn * preds_fusion + w_gbdt * gbdt_preds
        b_macro = []
        b_res = {}
        for c_idx, c_name in enumerate(CLASSES):
            auc = roc_auc_score(Y_true[:, c_idx], blend_preds[:, c_idx])
            b_res[c_name] = round(float(auc), 4)
            b_macro.append(auc)
        b_res["macro_auroc"] = round(float(np.mean(b_macro)), 4)
        print(f"  Ensemble (Weight NN={w_nn:.1f}, GBDT={w_gbdt:.1f}): Macro AUROC = {b_res['macro_auroc']} | Normal: {b_res['CVC - Normal']} | Borderline: {b_res['CVC - Borderline']} | Abnormal: {b_res['CVC - Abnormal']}")

    # Save summary report
    summary = {
        "fusion": res_fusion,
        "visual_only": res_vis,
        "geometric_only": res_geom,
        "gbdt_standalone": gbdt_res,
    }
    with open(PROJECT_ROOT / "runs" / "roi_fusion_convnext_tiny_v2" / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nAblation report saved to runs/roi_fusion_convnext_tiny_v2/ablation_summary.json")


if __name__ == "__main__":
    main()

