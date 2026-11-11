"""E101: Hierarchical Bayes & Continuum-Ordinal Decomposition
Level-1 Clinical Safety Gate (Safe vs. Unsafe) + Level-2 Conditional Severity Specialist (Abnormal | Unsafe)
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix
from scipy.stats import rankdata

PROJECT_ROOT = Path(r"C:\Ankshit\CVC-MAGNet-V2")
DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "splits" / "placement_locked_manifest.parquet"
VAL_PREDS_DIR = PROJECT_ROOT / "runs" / "val_preds"
REPORTS_DIR = PROJECT_ROOT / "reports"
LEDGER_CSV = REPORTS_DIR / "master_ablation_ledger.csv"

CVC_CLASSES = ["CVC - Normal", "CVC - Borderline", "CVC - Abnormal"]

def to_rank(p: np.ndarray) -> np.ndarray:
    out = np.zeros_like(p, dtype=np.float32)
    for c in range(p.shape[1]):
        out[:, c] = rankdata(p[:, c]) / float(len(p))
    return out

def compute_10_metrics(y_true, y_pred, y_unsafe_true, p_unsafe_pred, name="Model"):
    # 1. Binary Unsafe Gate metrics
    unsafe_roc = roc_auc_score(y_unsafe_true, p_unsafe_pred)
    unsafe_pr = average_precision_score(y_unsafe_true, p_unsafe_pred)
    
    # 2. 3-class AUROC
    roc_norm = roc_auc_score(y_true[:, 0], y_pred[:, 0])
    roc_bord = roc_auc_score(y_true[:, 1], y_pred[:, 1])
    roc_abn  = roc_auc_score(y_true[:, 2], y_pred[:, 2])
    macro_roc = (roc_norm + roc_bord + roc_abn) / 3.0
    
    # 3. 3-class AUPRC
    pr_norm = average_precision_score(y_true[:, 0], y_pred[:, 0])
    pr_bord = average_precision_score(y_true[:, 1], y_pred[:, 1])
    pr_abn  = average_precision_score(y_true[:, 2], y_pred[:, 2])
    macro_pr = (pr_norm + pr_bord + pr_abn) / 3.0
    
    # 4. Abnormal High-Sensitivity Operating Points
    yt_abn = y_true[:, 2]
    yp_abn = y_pred[:, 2]
    
    # Find operating point for target recall = 0.90
    prec, rec, thresholds = precision_recall_curve(yt_abn, yp_abn)
    idx90 = np.where(rec >= 0.90)[0]
    th90 = thresholds[idx90[-1]] if len(idx90) > 0 and idx90[-1] < len(thresholds) else thresholds[0]
    yp_bin90 = (yp_abn >= th90).astype(int)
    tn90, fp90, fn90, tp90 = confusion_matrix(yt_abn, yp_bin90).ravel()
    spec_at_90_sens = tn90 / (tn90 + fp90)
    
    # Find sensitivity at 90% specificity (FPR = 0.10)
    from sklearn.metrics import roc_curve
    fpr, tpr, roc_ths = roc_curve(yt_abn, yp_abn)
    idx_spec90 = np.where(fpr <= 0.10)[0]
    sens_at_90_spec = tpr[idx_spec90[-1]] if len(idx_spec90) > 0 else tpr[0]
    
    # F2 score on abnormal
    f2_scores = (5 * prec * rec) / (4 * prec + rec + 1e-8)
    best_f2 = np.max(f2_scores)
    
    # Conformal 98% Recall check (FN count)
    idx98 = np.where(rec >= 0.98)[0]
    th98 = thresholds[idx98[-1]] if len(idx98) > 0 and idx98[-1] < len(thresholds) else thresholds[0]
    yp_bin98 = (yp_abn >= th98).astype(int)
    tn98, fp98, fn98, tp98 = confusion_matrix(yt_abn, yp_bin98).ravel()
    
    metrics = {
        "Model": name,
        "Stage1_Unsafe_AUROC": unsafe_roc,
        "Stage1_Unsafe_AUPRC": unsafe_pr,
        "Macro_AUROC": macro_roc,
        "Normal_AUROC": roc_norm,
        "Borderline_AUROC": roc_bord,
        "Abnormal_AUROC": roc_abn,
        "Macro_AUPRC": macro_pr,
        "Abnormal_AUPRC": pr_abn,
        "Abnormal_F2": best_f2,
        "Sens_at_90_Spec": sens_at_90_spec,
        "Spec_at_90_Sens": spec_at_90_sens,
        "FN_at_98_Sens": fn98,
        "Total_Abnormal_Positives": yt_abn.sum()
    }
    return metrics

def main():
    print("=" * 80)
    print("E101: HIERARCHICAL BAYES SAFE-VS-UNSAFE TRIAGE PIPELINE")
    print("=" * 80)
    start_time = time.time()
    
    manifest = pd.read_parquet(MANIFEST_PATH)
    val_manifest = manifest[manifest["split"] == "val"].reset_index(drop=True)
    y_val = val_manifest[CVC_CLASSES].values.astype(int)
    patient_ids = val_manifest["PatientID"].values
    N_val = len(val_manifest)
    
    is_multicvc = ((val_manifest["annotated_cvc_count"] > 1) | (y_val.sum(axis=1) > 1)).values
    
    # Define Ground Truths
    # Level-1 Unsafe Target: Borderline OR Abnormal
    y_unsafe = np.maximum(y_val[:, 1], y_val[:, 2]).astype(int)
    # Level-2 Conditional Target: Abnormal within Unsafe cohort
    y_abn_given_unsafe = y_val[:, 2].astype(int)
    
    print(f"Validation Cohort: N={N_val:,} | Unsafe (Actionable): {y_unsafe.sum()} ({y_unsafe.sum()/N_val:.1%}) | Safe (Normal): {(y_unsafe==0).sum()}")
    print(f"Inside Unsafe Cohort: Abnormal = {y_val[:, 2].sum()} vs Borderline = {y_val[:, 1].sum()} (Balanced ~1:2.7 ratio)")

    # Gather feature streams
    stream_files = [
        ("E89_mgap_dense", VAL_PREDS_DIR / "val_preds_E89_mgap_densenet121_512.npy"),
        ("E90_mgap_srx", VAL_PREDS_DIR / "val_preds_E90_mgap_seresnext50_512.npy"),
        ("E91_mgap_768", VAL_PREDS_DIR / "val_preds_E91_mgap_densenet121_768_highres.npy"),
        ("E95_multiscale", VAL_PREDS_DIR / "val_preds_E95_multiscale_global_plus_corridor_blend.npy"),
        ("E96_distill_dense", VAL_PREDS_DIR / "val_preds_E96_noisy_student_densenet121_512.npy"),
        ("E97_distill_srx", VAL_PREDS_DIR / "val_preds_E97_noisy_student_seresnext50_512.npy"),
        ("E98_110d", VAL_PREDS_DIR / "val_preds_E98_110d_gbdt.npy"),
        ("E85_apex", VAL_PREDS_DIR / "val_preds_E85_multiplicity_gated_t3_grand_apex.npy"),
        ("E99_apex", VAL_PREDS_DIR / "val_preds_E99_grand_meta_apex.npy"),
    ]
    
    feat_cols = []
    for tag, p in stream_files:
        if p.exists():
            arr = np.load(p)
            feat_cols.append(to_rank(arr))
    X = np.hstack(feat_cols)
    print(f"Feature Space Matrix shape: {X.shape}")

    # 5-Fold GroupKFold by PatientID
    gkf = GroupKFold(n_splits=5)
    oof_p_unsafe = np.zeros(N_val, dtype=np.float32)
    oof_p_crit_given_unsafe = np.zeros(N_val, dtype=np.float32)
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_unsafe, groups=patient_ids)):
        # ----------------------------------------------------------------------
        # STAGE 1: Level-1 Clinical Safety Gate (Safe vs Unsafe)
        # ----------------------------------------------------------------------
        X_tr, y_tr_unsafe = X[train_idx], y_unsafe[train_idx]
        X_va, y_va_unsafe = X[val_idx], y_unsafe[val_idx]
        
        clf_unsafe = LGBMClassifier(
            n_estimators=180, learning_rate=0.035, max_depth=4, num_leaves=15,
            subsample=0.85, colsample_bytree=0.80, random_state=42 + fold, verbose=-1
        )
        clf_unsafe.fit(X_tr, y_tr_unsafe)
        oof_p_unsafe[val_idx] = clf_unsafe.predict_proba(X_va)[:, 1]
        
        # ----------------------------------------------------------------------
        # STAGE 2: Level-2 Conditional Specialist (Abnormal vs Borderline | Unsafe)
        # ----------------------------------------------------------------------
        unsafe_tr_mask = (y_tr_unsafe == 1)
        X_tr_unsafe = X_tr[unsafe_tr_mask]
        y_tr_crit = y_abn_given_unsafe[train_idx][unsafe_tr_mask]
        
        # Train cost-sensitive specialist on balanced unsafe cohort
        clf_crit = CatBoostClassifier(
            iterations=250, learning_rate=0.035, depth=4, l2_leaf_reg=4.0,
            auto_class_weights="Balanced", random_seed=42 + fold, verbose=0
        )
        clf_crit.fit(X_tr_unsafe, y_tr_crit)
        oof_p_crit_given_unsafe[val_idx] = clf_crit.predict_proba(X_va)[:, 1]

    # Exact Bayes Marginal Reconstruction
    p_hier_norm = 1.0 - oof_p_unsafe
    p_hier_abn  = oof_p_unsafe * oof_p_crit_given_unsafe
    p_hier_bord = oof_p_unsafe * (1.0 - oof_p_crit_given_unsafe)
    
    oof_hier = np.stack([p_hier_norm, p_hier_bord, p_hier_abn], axis=1)
    
    # Multi-CVC Bypass: where multiplicity detected, fuse with direct E99 multi-label apex
    e99_preds = np.load(VAL_PREDS_DIR / "val_preds_E99_grand_meta_apex.npy")
    for i in range(N_val):
        if is_multicvc[i]:
            oof_hier[i] = 0.50 * oof_hier[i] + 0.50 * e99_preds[i]
            
    # Save predictions
    save_path = VAL_PREDS_DIR / "val_preds_E101_hierarchical_bayes.npy"
    np.save(save_path, oof_hier)
    print(f"\n✅ Saved E101 predictions to: {save_path}")

    # Compute and display 10-metric clinical evaluation
    metrics = compute_10_metrics(y_val, oof_hier, y_unsafe, oof_p_unsafe, name="E101_Hierarchical_Bayes")
    
    print("\n" + "=" * 80)
    print("E101 CLINICAL SAFETY METRICS BREAKDOWN:")
    print("=" * 80)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<28}: {v:.4f}")
        else:
            print(f"  {k:<28}: {v}")
    print("=" * 80)
    print(f"Stage 1 Binary Safe vs Unsafe AUROC : {metrics['Stage1_Unsafe_AUROC']:.4f}")
    print(f"3-Class Macro AUROC                 : {metrics['Macro_AUROC']:.4f}")
    print(f"Runtime                             : {time.time() - start_time:.1f} seconds")

if __name__ == "__main__":
    main()

