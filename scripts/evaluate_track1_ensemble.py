"""Track 1: Full-Cohort Ensemble Benchmark (traced_effnet + Zonal GBDT).

Evaluates across the full locked N=2,558 validation cohort:
1. Standalone traced_effnet (512x512 EfficientNet-B0)
2. Standalone Zonal GBDT (25-D continuous mass integrals)
3. Raw Probability Averaging
4. Percentile Rank Averaging (scipy.stats.rankdata)
5. Class-Specific Optimal Rank Weighting (alpha_c per class)
6. Confidence-Gated Rank Blend (gated by conf_tip >= 0.20)
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.placement_dataset import CVCPlacementDataset
from src.data.placement_roi_dataset import FEATURE_COLS
from src.models.placement_baseline import CVCPlacementBaselineNet

DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "splits" / "placement_locked_manifest.parquet"
ZONAL_TRAIN_PATH = DATA_DIR / "curated_cvc_v1" / "zonal_train_features.parquet"
ZONAL_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "zonal_val_features.parquet"
EFFNET_CKPT_PATH = PROJECT_ROOT / "runs" / "placement_ablation_queue_v1" / "all_cvc_traced_effnet_multilabel_1000" / "checkpoints" / "best_placement_baseline.pt"
CACHED_EFFNET_PREDS = PROJECT_ROOT / "runs" / "effnet_val_preds_2558.npy"

CLASSES = ["CVC - Normal", "CVC - Borderline", "CVC - Abnormal"]


def get_effnet_predictions(manifest_val: pd.DataFrame, device: torch.device) -> np.ndarray:
    """Run or load cached EfficientNet-B0 predictions on N=2,558 validation cohort."""
    if CACHED_EFFNET_PREDS.exists():
        print(f"Loading cached EfficientNet-B0 predictions from {CACHED_EFFNET_PREDS}...")
        preds = np.load(str(CACHED_EFFNET_PREDS))
        if len(preds) == len(manifest_val):
            return preds

    print(f"Running EfficientNet-B0 inference on {len(manifest_val)} validation studies...")
    dataset = CVCPlacementDataset(df=manifest_val, is_train=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    model = CVCPlacementBaselineNet(
        backbone_name="tf_efficientnet_b0.ns_jft_in1k",
        pretrained=False,
        num_classes=3,
    )
    ckpt = torch.load(EFFNET_CKPT_PATH, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    # Clean state dict keys if needed
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned)
    model.to(device)
    model.eval()

    all_probs = []
    t0 = time.time()
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            imgs = batch["image"].to(device)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(imgs)
                probs = torch.sigmoid(logits).cpu().to(torch.float32).numpy()
            all_probs.append(probs)
            if (b_idx + 1) % 20 == 0 or (b_idx + 1) == len(loader):
                print(f"  [EffNet Val] Batch {b_idx + 1}/{len(loader)} ({(b_idx + 1) * 32 / (time.time() - t0):.1f} img/s)...", flush=True)

    y_prob = np.concatenate(all_probs, axis=0)
    np.save(str(CACHED_EFFNET_PREDS), y_prob)
    print(f"Saved predictions to {CACHED_EFFNET_PREDS}")
    return y_prob


def get_zonal_gbdt_predictions(df_train: pd.DataFrame, df_val: pd.DataFrame) -> np.ndarray:
    """Train Zonal GBDT on 6,600 training studies and predict on 2,558 validation studies."""
    print("Fitting Zonal GBDT on 6,600 training instances...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(np.nan_to_num(df_train[FEATURE_COLS].values, nan=0.0))
    X_val = scaler.transform(np.nan_to_num(df_val[FEATURE_COLS].values, nan=0.0))

    preds = np.zeros((len(df_val), 3), dtype=np.float32)
    for c_idx, c_name in enumerate(CLASSES):
        y_train = df_train[c_name].values.astype(int)
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
        clf.fit(X_train, y_train)
        preds[:, c_idx] = clf.predict_proba(X_val)[:, 1]
    return preds


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate per-class and macro AUROC."""
    aurocs = {}
    macro = []
    for c_idx, c_name in enumerate(CLASSES):
        auc = roc_auc_score(y_true[:, c_idx], y_pred[:, c_idx])
        aurocs[c_name] = round(float(auc), 4)
        macro.append(auc)
    aurocs["macro_auroc"] = round(float(np.mean(macro)), 4)
    return aurocs


def main():
    print("=" * 80)
    print("TRACK 1: FULL-COHORT ENSEMBLE BENCHMARK (N=2,558 LOCKED COHORT)")
    print("Fusing Image-Level Context (traced_effnet) + Tabular Continuous Geometry (Zonal GBDT)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Data
    manifest = pd.read_parquet(MANIFEST_PATH)
    manifest_val = manifest[manifest["split"] == "val"].reset_index(drop=True)
    df_zonal_train = pd.read_parquet(ZONAL_TRAIN_PATH)
    df_zonal_val = pd.read_parquet(ZONAL_VAL_PATH)

    # Ensure alignment by study_id
    manifest_val_indexed = manifest_val.set_index("StudyInstanceUID")
    df_zonal_val_indexed = df_zonal_val.set_index("study_id")
    common_sids = [sid for sid in manifest_val["StudyInstanceUID"] if sid in df_zonal_val_indexed.index]
    assert len(common_sids) == len(manifest_val), f"Mismatch: {len(common_sids)} vs {len(manifest_val)}"

    manifest_val = manifest_val_indexed.loc[common_sids].reset_index()
    df_zonal_val = df_zonal_val_indexed.loc[common_sids].reset_index()

    Y_true = manifest_val[CLASSES].values.astype(np.float32)

    # 2. Get Model Probabilities
    P_effnet = get_effnet_predictions(manifest_val, device)
    P_gbdt = get_zonal_gbdt_predictions(df_zonal_train, df_zonal_val)

    # 3. Compute Percentile Ranks
    R_effnet = np.zeros_like(P_effnet)
    R_gbdt = np.zeros_like(P_gbdt)
    N = len(Y_true)
    for c_idx in range(3):
        R_effnet[:, c_idx] = rankdata(P_effnet[:, c_idx]) / float(N)
        R_gbdt[:, c_idx] = rankdata(P_gbdt[:, c_idx]) / float(N)

    # 4. Standalone Evaluations
    res_effnet = calc_metrics(Y_true, P_effnet)
    res_gbdt = calc_metrics(Y_true, P_gbdt)

    print("\n--- STANDALONE MODELS ---")
    print(f"1. traced_effnet (Standalone): Macro AUROC = {res_effnet['macro_auroc']:.4f} | Normal: {res_effnet['CVC - Normal']:.4f} | Borderline: {res_effnet['CVC - Borderline']:.4f} | Abnormal: {res_effnet['CVC - Abnormal']:.4f}")
    print(f"2. Zonal GBDT (Standalone):   Macro AUROC = {res_gbdt['macro_auroc']:.4f} | Normal: {res_gbdt['CVC - Normal']:.4f} | Borderline: {res_gbdt['CVC - Borderline']:.4f} | Abnormal: {res_gbdt['CVC - Abnormal']:.4f}")

    # 5. Raw Probability Averaging
    P_raw_avg = 0.5 * P_effnet + 0.5 * P_gbdt
    res_raw_avg = calc_metrics(Y_true, P_raw_avg)
    print(f"3. Raw 50/50 Prob Average:     Macro AUROC = {res_raw_avg['macro_auroc']:.4f} | Normal: {res_raw_avg['CVC - Normal']:.4f} | Borderline: {res_raw_avg['CVC - Borderline']:.4f} | Abnormal: {res_raw_avg['CVC - Abnormal']:.4f}")

    # 6. Global Percentile Rank Averaging (alpha grid)
    print("\n--- GLOBAL RANK AVERAGING (Equal alpha across classes) ---")
    best_global_alpha = 0.5
    best_global_res = None
    best_global_macro = 0.0
    for alpha in np.linspace(0.0, 1.0, 11):
        R_blend = alpha * R_effnet + (1.0 - alpha) * R_gbdt
        res = calc_metrics(Y_true, R_blend)
        print(f"  alpha={alpha:.1f} (EffNet {int(alpha*100)}% / GBDT {int((1-alpha)*100)}%): Macro = {res['macro_auroc']:.4f} | Normal: {res['CVC - Normal']:.4f} | Border: {res['CVC - Borderline']:.4f} | Abnorm: {res['CVC - Abnormal']:.4f}")
        if res["macro_auroc"] > best_global_macro:
            best_global_macro = res["macro_auroc"]
            best_global_alpha = alpha
            best_global_res = res

    # 7. Class-Specific Optimal Rank Weighting (alpha_c per class)
    print("\n--- CLASS-SPECIFIC OPTIMAL RANK WEIGHTING ---")
    optimal_alphas = []
    optimal_class_auroc = []
    R_opt = np.zeros_like(R_effnet)

    for c_idx, c_name in enumerate(CLASSES):
        best_c_alpha = 0.5
        best_c_auc = 0.0
        for alpha in np.linspace(0.0, 1.0, 21):
            r_c = alpha * R_effnet[:, c_idx] + (1.0 - alpha) * R_gbdt[:, c_idx]
            auc = roc_auc_score(Y_true[:, c_idx], r_c)
            if auc > best_c_auc:
                best_c_auc = auc
                best_c_alpha = alpha
        optimal_alphas.append(round(float(best_c_alpha), 2))
        optimal_class_auroc.append(round(float(best_c_auc), 4))
        R_opt[:, c_idx] = best_c_alpha * R_effnet[:, c_idx] + (1.0 - best_c_alpha) * R_gbdt[:, c_idx]
        print(f"  {c_name:<16}: Optimal alpha = {best_c_alpha:.2f} (EffNet {int(best_c_alpha*100)}% / GBDT {int((1-best_c_alpha)*100)}%) -> AUROC = {best_c_auc:.4f}")

    opt_macro = round(float(np.mean(optimal_class_auroc)), 4)
    print(f"--> Optimal Class-Weighted Rank Blend Macro AUROC: {opt_macro:.4f}")

    # 8. Confidence-Gated Rank Blend
    print("\n--- CONFIDENCE-GATED RANK BLEND ---")
    # has_valid_tip is 1 when conf_tip >= 0.20
    has_valid_tip = df_zonal_val["has_valid_tip"].values  # [N]
    
    # Test combinations of alpha_valid (higher GBDT) vs alpha_invalid (higher EffNet)
    best_gated_macro = 0.0
    best_gated_res = None
    best_gated_params = None

    for a_valid in [0.3, 0.4, 0.45, 0.5, 0.55]:
        for a_invalid in [0.7, 0.75, 0.8, 0.85, 0.9]:
            # Weight vector per study [N, 1]
            alpha_study = np.where(has_valid_tip == 1, a_valid, a_invalid)[:, None]
            R_gated = alpha_study * R_effnet + (1.0 - alpha_study) * R_gbdt
            res = calc_metrics(Y_true, R_gated)
            if res["macro_auroc"] > best_gated_macro:
                best_gated_macro = res["macro_auroc"]
                best_gated_res = res
                best_gated_params = (a_valid, a_invalid)

    print(f"  Best Gated Parameters: alpha_valid={best_gated_params[0]:.2f} (GBDT {int((1-best_gated_params[0])*100)}%), alpha_invalid={best_gated_params[1]:.2f} (EffNet {int(best_gated_params[1])*100}%)")
    print(f"  Confidence-Gated Blend Macro AUROC: {best_gated_res['macro_auroc']:.4f} | Normal: {best_gated_res['CVC - Normal']:.4f} | Borderline: {best_gated_res['CVC - Borderline']:.4f} | Abnormal: {best_gated_res['CVC - Abnormal']:.4f}")

    # Save summary report
    summary = {
        "effnet_standalone": res_effnet,
        "gbdt_standalone": res_gbdt,
        "raw_prob_average": res_raw_avg,
        "global_rank_blend": {"best_alpha": best_global_alpha, "metrics": best_global_res},
        "class_specific_optimal_blend": {"alphas": dict(zip(CLASSES, optimal_alphas)), "metrics": dict(zip(CLASSES, optimal_class_auroc)), "macro_auroc": opt_macro},
        "confidence_gated_blend": {"params": {"alpha_valid": best_gated_params[0], "alpha_invalid": best_gated_params[1]}, "metrics": best_gated_res},
    }
    out_path = PROJECT_ROOT / "runs" / "track1_ensemble_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nTrack 1 summary successfully saved to {out_path}")


if __name__ == "__main__":
    main()

