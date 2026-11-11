"""E103b: Cost-Sensitive Clinical Safety Apex Fusion
Blends E99-B + E102b (Two-Tier pAUROC) for Abnormal (dropping E101 from Abnormal)
Uses E101 for Stage-1 Binary Unsafe Gate (0.8768) and Borderline.
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

CVC_CLASSES = ["CVC - Normal", "CVC - Borderline", "CVC - Abnormal"]

def evaluate_metrics(y_true, y_pred, name="Model"):
    y_unsafe_true = np.maximum(y_true[:, 1], y_true[:, 2]).astype(int)
    p_unsafe_pred = np.maximum(y_pred[:, 1], y_pred[:, 2])
    
    unsafe_roc = roc_auc_score(y_unsafe_true, p_unsafe_pred)
    roc_norm = roc_auc_score(y_true[:, 0], y_pred[:, 0])
    roc_bord = roc_auc_score(y_true[:, 1], y_pred[:, 1])
    roc_abn  = roc_auc_score(y_true[:, 2], y_pred[:, 2])
    macro_roc = (roc_norm + roc_bord + roc_abn) / 3.0
    
    pr_abn  = average_precision_score(y_true[:, 2], y_pred[:, 2])
    macro_pr = np.mean([average_precision_score(y_true[:, c], y_pred[:, c]) for c in range(3)])
    
    yt_abn = y_true[:, 2]
    yp_abn = y_pred[:, 2]
    
    prec, rec, _ = precision_recall_curve(yt_abn, yp_abn)
    f2_scores = (5 * prec * rec) / (4 * prec + rec + 1e-8)
    best_f2 = float(np.max(f2_scores))
    
    fpr, tpr, roc_ths = roc_curve(yt_abn, yp_abn)
    idx_sp90 = np.where(fpr <= 0.10)[0]
    sens_at_90_spec = float(tpr[idx_sp90[-1]] if len(idx_sp90) > 0 else tpr[0])
    th90 = roc_ths[idx_sp90[-1]]
    fn_at_90_spec = int(((yp_abn < th90) & (yt_abn == 1)).sum())
    
    idx_sp80 = np.where(fpr <= 0.20)[0]
    sens_at_80_spec = float(tpr[idx_sp80[-1]] if len(idx_sp80) > 0 else tpr[0])
    th80 = roc_ths[idx_sp80[-1]]
    fn_at_80_spec = int(((yp_abn < th80) & (yt_abn == 1)).sum())
    
    pos_scores = np.sort(yp_abn[yt_abn == 1])
    k_miss = int(np.floor(0.02 * (len(pos_scores) + 1)))
    tau_crc = pos_scores[k_miss]
    neg_cleared = int(((yp_abn < tau_crc) & (yt_abn == 0)).sum())
    rule_out_spec_crc = float(neg_cleared / (len(yt_abn) - sum(yt_abn)))
    green_npv = float(neg_cleared / (neg_cleared + k_miss))
    
    return {
        "Model": name,
        "Macro_AUROC": round(macro_roc, 4),
        "Binary_Unsafe_AUROC": round(unsafe_roc, 4),
        "Abnormal_AUROC": round(roc_abn, 4),
        "Abnormal_AUPRC": round(pr_abn, 4),
        "Macro_AUPRC": round(macro_pr, 4),
        "Abnormal_F2": round(best_f2, 4),
        "Sens_at_90_Spec": round(sens_at_90_spec * 100.0, 2),
        "Missed_Abn_at_90_Spec": fn_at_90_spec,
        "Sens_at_80_Spec": round(sens_at_80_spec * 100.0, 2),
        "Missed_Abn_at_80_Spec": fn_at_80_spec,
        "RuleOut_Spec_at_Conformal_98": round(rule_out_spec_crc * 100.0, 2),
        "Green_Zone_NPV": round(green_npv * 100.0, 2)
    }

def main():
    print("=" * 80)
    print("E103b: COST-SENSITIVE CLINICAL SAFETY APEX FUSION")
    print("=" * 80)
    
    manifest = pd.read_parquet(MANIFEST_PATH)
    val_manifest = manifest[manifest["split"] == "val"].reset_index(drop=True)
    y_val = val_manifest[CVC_CLASSES].values.astype(int)
    
    p_e99  = np.load(VAL_PREDS_DIR / "val_preds_E99_grand_meta_apex.npy")
    p_e101 = np.load(VAL_PREDS_DIR / "val_preds_E101_hierarchical_bayes.npy")
    p_e102 = np.load(VAL_PREDS_DIR / "val_preds_E102_pauroc_specialist.npy")
    
    # Optimal class routing:
    # Normal: Pure E99 (0.8546)
    # Borderline: Pure E99 (0.8427)
    # Abnormal: 0.65 E102b (Two-Tier pAUROC) + 0.35 E99 (Meta-Apex) -> 0.8574 AUROC, 0.5035 AUPRC, 57.14% Sens@90Sp
    norm_col = p_e99[:, 0]
    bord_col = p_e99[:, 1]
    abn_col  = 0.65 * p_e102[:, 2] + 0.35 * p_e99[:, 2]
    
    p_e103 = np.stack([norm_col, bord_col, abn_col], axis=1)
    save_path = VAL_PREDS_DIR / "val_preds_E103_clinical_safety_apex.npy"
    np.save(save_path, p_e103)
    print(f"✅ Saved E103b predictions to: {save_path}")
    
    m = evaluate_metrics(y_val, p_e103, name="E103b_Clinical_Safety_Apex")
    print("\n" + "=" * 80)
    for k, v in m.items():
        print(f"  {k:<30}: {v}")
    print("=" * 80)

if __name__ == "__main__":
    main()
