"""E103: Cost-Sensitive Clinical Safety Apex Fusion
Optimizes convex ensemble weights between E99-B (BCE Meta-Apex), E101 (Hierarchical Bayes), and E102 (pAUROC Specialist)
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix, roc_curve

PROJECT_ROOT = Path(r"C:\Ankshit\CVC-MAGNet-V2")
DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "splits" / "placement_locked_manifest.parquet"
VAL_PREDS_DIR = PROJECT_ROOT / "runs" / "val_preds"
REPORTS_DIR = PROJECT_ROOT / "reports"
LEDGER_CSV = REPORTS_DIR / "master_ablation_ledger.csv"

CVC_CLASSES = ["CVC - Normal", "CVC - Borderline", "CVC - Abnormal"]

def compute_10_metrics(y_true, y_pred, y_unsafe_true, p_unsafe_pred, name="Model"):
    unsafe_roc = roc_auc_score(y_unsafe_true, p_unsafe_pred)
    unsafe_pr = average_precision_score(y_unsafe_true, p_unsafe_pred)
    
    roc_norm = roc_auc_score(y_true[:, 0], y_pred[:, 0])
    roc_bord = roc_auc_score(y_true[:, 1], y_pred[:, 1])
    roc_abn  = roc_auc_score(y_true[:, 2], y_pred[:, 2])
    macro_roc = (roc_norm + roc_bord + roc_abn) / 3.0
    
    pr_norm = average_precision_score(y_true[:, 0], y_pred[:, 0])
    pr_bord = average_precision_score(y_true[:, 1], y_pred[:, 1])
    pr_abn  = average_precision_score(y_true[:, 2], y_pred[:, 2])
    macro_pr = (pr_norm + pr_bord + pr_abn) / 3.0
    
    yt_abn = y_true[:, 2]
    yp_abn = y_pred[:, 2]
    
    prec, rec, thresholds = precision_recall_curve(yt_abn, yp_abn)
    idx90 = np.where(rec >= 0.90)[0]
    th90 = thresholds[idx90[-1]] if len(idx90) > 0 and idx90[-1] < len(thresholds) else thresholds[0]
    yp_bin90 = (yp_abn >= th90).astype(int)
    tn90, fp90, fn90, tp90 = confusion_matrix(yt_abn, yp_bin90).ravel()
    spec_at_90_sens = tn90 / (tn90 + fp90)
    
    fpr, tpr, roc_ths = roc_curve(yt_abn, yp_abn)
    idx_spec90 = np.where(fpr <= 0.10)[0]
    sens_at_90_spec = tpr[idx_spec90[-1]] if len(idx_spec90) > 0 else tpr[0]
    
    f2_scores = (5 * prec * rec) / (4 * prec + rec + 1e-8)
    best_f2 = np.max(f2_scores)
    
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
        "Total_Abnormal_Positives": int(yt_abn.sum())
    }
    return metrics

def main():
    print("=" * 80)
    print("E103: COST-SENSITIVE CLINICAL SAFETY APEX FUSION")
    print("=" * 80)
    start_time = time.time()
    
    manifest = pd.read_parquet(MANIFEST_PATH)
    val_manifest = manifest[manifest["split"] == "val"].reset_index(drop=True)
    y_val = val_manifest[CVC_CLASSES].values.astype(int)
    N_val = len(val_manifest)
    
    y_unsafe = np.maximum(y_val[:, 1], y_val[:, 2]).astype(int)
    
    p_e99  = np.load(VAL_PREDS_DIR / "val_preds_E99_grand_meta_apex.npy")
    p_e101 = np.load(VAL_PREDS_DIR / "val_preds_E101_hierarchical_bayes.npy")
    p_e102 = np.load(VAL_PREDS_DIR / "val_preds_E102_pauroc_specialist.npy")
    
    # Class-wise optimal blending grid search
    best_weights = []
    fused_cols = []
    
    for c in range(3):
        yt = y_val[:, c]
        best_auc = -1.0
        best_w = None
        best_pred = None
        
        # Grid search over simplex weights (w1, w2, w3)
        for w1 in np.linspace(0.0, 1.0, 11):
            for w2 in np.linspace(0.0, 1.0 - w1, 11):
                w3 = round(1.0 - w1 - w2, 4)
                if w3 < 0:
                    continue
                cand = w1 * p_e99[:, c] + w2 * p_e101[:, c] + w3 * p_e102[:, c]
                auc = roc_auc_score(yt, cand)
                if auc > best_auc:
                    best_auc = auc
                    best_w = (round(w1, 2), round(w2, 2), round(w3, 2))
                    best_pred = cand
                    
        print(f"Class {CVC_CLASSES[c]:<16}: Best weights (E99={best_w[0]}, E101={best_w[1]}, E102={best_w[2]}) -> AUROC: {best_auc:.4f}")
        best_weights.append(best_w)
        fused_cols.append(best_pred)
        
    p_e103 = np.stack(fused_cols, axis=1)
    
    # Save predictions
    save_path = VAL_PREDS_DIR / "val_preds_E103_clinical_safety_apex.npy"
    np.save(save_path, p_e103)
    print(f"\n✅ Saved E103 predictions to: {save_path}")
    
    # Evaluate
    p_unsafe = np.maximum(p_e103[:, 1], p_e103[:, 2])
    metrics = compute_10_metrics(y_val, p_e103, y_unsafe, p_unsafe, name="E103_Clinical_Safety_Apex")
    
    print("\n" + "=" * 80)
    print("E103 CLINICAL SAFETY METRICS BREAKDOWN:")
    print("=" * 80)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<28}: {v:.4f}")
        else:
            print(f"  {k:<28}: {v}")
    print("=" * 80)
    print(f"Macro AUROC                         : {metrics['Macro_AUROC']:.4f}")
    print(f"Abnormal AUROC                      : {metrics['Abnormal_AUROC']:.4f}")
    print(f"Sens @ 90% Spec (High-Recall Regime): {metrics['Sens_at_90_Spec']:.4f}")
    print(f"Abnormal F2 (Recall-Biased F-score) : {metrics['Abnormal_F2']:.4f}")
    print(f"FN @ 98% Sens (Missed Positives)    : {metrics['FN_at_98_Sens']}")
    print(f"Runtime                             : {time.time() - start_time:.1f} seconds")

if __name__ == "__main__":
    main()
