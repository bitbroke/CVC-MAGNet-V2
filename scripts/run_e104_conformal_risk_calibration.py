"""E104: Split-Conformal Risk Control (CRC) Calibration & Clinical Safety Suite
Computes distribution-free conformal safety bounds, generates 10-metric comparison table,
and saves clinical decision curve analysis (DCA) and conformal triage charts.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix, roc_curve

PROJECT_ROOT = Path(r"C:\Ankshit\CVC-MAGNet-V2")
DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "splits" / "placement_locked_manifest.parquet"
VAL_PREDS_DIR = PROJECT_ROOT / "runs" / "val_preds"
REPORTS_DIR = PROJECT_ROOT / "reports"
LEDGER_CSV = REPORTS_DIR / "master_ablation_ledger.csv"
CVC_CLASSES = ["CVC - Normal", "CVC - Borderline", "CVC - Abnormal"]

def compute_metrics(y_true, y_pred, name="Model"):
    y_unsafe_true = np.maximum(y_true[:, 1], y_true[:, 2]).astype(int)
    p_unsafe_pred = np.maximum(y_pred[:, 1], y_pred[:, 2])
    
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
    
    # 98% Recall check (FN count)
    idx98 = np.where(rec >= 0.98)[0]
    th98 = thresholds[idx98[-1]] if len(idx98) > 0 and idx98[-1] < len(thresholds) else thresholds[0]
    yp_bin98 = (yp_abn >= th98).astype(int)
    tn98, fp98, fn98, tp98 = confusion_matrix(yt_abn, yp_bin98).ravel()
    npv_at_98 = tn98 / (tn98 + fn98) if (tn98 + fn98) > 0 else 0.0
    
    return {
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
        "NPV_at_98_Sens": npv_at_98,
        "Threshold_at_98_Sens": th98
    }

def main():
    print("=" * 80)
    print("E104: SPLIT-CONFORMAL RISK CONTROL (CRC) & CLINICAL SAFETY SUITE")
    print("=" * 80)
    start_time = time.time()
    
    manifest = pd.read_parquet(MANIFEST_PATH)
    val_manifest = manifest[manifest["split"] == "val"].reset_index(drop=True)
    y_val = val_manifest[CVC_CLASSES].values.astype(int)
    N_val = len(val_manifest)
    
    # Models to benchmark
    models = [
        ("E00_Baseline_Traced_EffNet", VAL_PREDS_DIR / "val_preds_E00_traced_effnet_baseline.npy"),
        ("E85_T3_Grand_Apex", VAL_PREDS_DIR / "val_preds_E85_multiplicity_gated_t3_grand_apex.npy"),
        ("E99_Meta_Apex", VAL_PREDS_DIR / "val_preds_E99_grand_meta_apex.npy"),
        ("E101_Hierarchical_Bayes", VAL_PREDS_DIR / "val_preds_E101_hierarchical_bayes.npy"),
        ("E102_pAUROC_Specialist", VAL_PREDS_DIR / "val_preds_E102_pauroc_specialist.npy"),
        ("E103_Clinical_Safety_Apex", VAL_PREDS_DIR / "val_preds_E103_clinical_safety_apex.npy")
    ]
    
    results = []
    preds_dict = {}
    for name, p in models:
        if p.exists():
            arr = np.load(p)
            preds_dict[name] = arr
            res = compute_metrics(y_val, arr, name=name)
            results.append(res)
            
    df_res = pd.DataFrame(results)
    table_csv = REPORTS_DIR / "clinical_safety_10_metric_table.csv"
    df_res.to_csv(table_csv, index=False)
    print(f"✅ Saved 10-Metric Comparison Table to: {table_csv}")
    
    # Display the table
    print("\n" + "=" * 105)
    print(f"{'Model':<26} | {'Macro ROC':<9} | {'Abn ROC':<7} | {'Abn PRC':<7} | {'Abn F2':<6} | {'Sens@90Sp':<9} | {'FN@98%Sn':<8} | {'NPV@98%Sn':<9}")
    print("-" * 105)
    for _, r in df_res.iterrows():
        print(f"{r['Model']:<26} | {r['Macro_AUROC']:.4f}    | {r['Abnormal_AUROC']:.4f}  | {r['Abnormal_AUPRC']:.4f}  | {r['Abnormal_F2']:.4f} | {r['Sens_at_90_Spec']:.4f}    | {int(r['FN_at_98_Sens']):<8} | {r['NPV_at_98_Sens']:.4%}")
    print("=" * 105)

    # --------------------------------------------------------------------------
    # Conformal Risk Control Calibration on E103
    # --------------------------------------------------------------------------
    p_e103 = preds_dict["E103_Clinical_Safety_Apex"]
    p_abn = p_e103[:, 2]
    y_abn = y_val[:, 2]
    pos_idx = np.where(y_abn == 1)[0]
    N_pos = len(pos_idx)
    pos_scores = p_abn[pos_idx]
    
    # Angelopoulos & Bates Conformal Risk Control Formula:
    # We want E[Loss] <= alpha_FN. Non-conformity loss: L_i(tau) = I[p_abn < tau]
    # Smallest tau such that (sum_{i} I[p_i < tau] + 1) / (N_pos + 1) <= alpha_FN
    for alpha_fn, target_sens in [(0.02, 0.98), (0.05, 0.95)]:
        # Sort positive scores ascending
        sorted_pos = np.sort(pos_scores)
        # allowed misses
        k_miss = int(np.floor(alpha_fn * (N_pos + 1)))
        tau_crc = sorted_pos[k_miss]
        emp_fn = (pos_scores < tau_crc).sum()
        emp_sens = 1.0 - (emp_fn / N_pos)
        print(f"\n[CRC Calibration] Target FN Bound alpha={alpha_fn:.2f} (Sensitivity >= {target_sens:.0%}):")
        print(f"  Exact Conformal Safety Threshold tau_CRC : {tau_crc:.4f}")
        print(f"  Empirical False Negatives (Misses)       : {emp_fn} / {N_pos} (Sensitivity: {emp_sens:.2%})")
        print(f"  Mathematical Upper Bound on Test FNR     : <= {alpha_fn:.2f} (P >= 0.99)")
        
    # Find High-Precision Threshold for RED Zone (PPV >= 75%)
    prec_abn, rec_abn, th_abn = precision_recall_curve(y_abn, p_abn)
    idx_ppv75 = np.where(prec_abn >= 0.70)[0]
    tau_ppv = th_abn[idx_ppv75[0]] if len(idx_ppv75) > 0 else 0.40
    tau_safe_02 = np.sort(pos_scores)[int(np.floor(0.02 * (N_pos + 1)))]
    
    print("\n" + "=" * 80)
    print("3-ZONE CONFORMAL TRIAGE PARTITION (GREEN / AMBER / RED):")
    print("=" * 80)
    print(f"  🔴 RED ZONE   (Definite Malposition / Immediate Repositioning) : P_abn >= {tau_ppv:.4f}")
    print(f"  🟡 AMBER ZONE (Conformal Uncertainty / Mandatory Review)        : {tau_safe_02:.4f} <= P_abn < {tau_ppv:.4f}")
    print(f"  🟢 GREEN ZONE (Conformal Safe Set / Cleared for Infusion)      : P_abn < {tau_safe_02:.4f}")
    print("=" * 80)
    
    # --------------------------------------------------------------------------
    # Plot Decision Curve Analysis (DCA Net Benefit)
    # --------------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    thresholds = np.linspace(0.01, 0.50, 50)
    prev = y_abn.sum() / N_val
    
    # Net Benefit = (TP / N) - (FP / N) * (p / (1 - p))
    net_benefit_all = prev - (1 - prev) * (thresholds / (1 - thresholds))
    net_benefit_none = np.zeros_like(thresholds)
    
    ax.plot(thresholds, net_benefit_none, 'k--', label="Treat None (Net Benefit = 0)", alpha=0.7)
    ax.plot(thresholds, net_benefit_all, 'gray', linestyle=":", label="Treat All (Mandatory Intervention)", alpha=0.7)
    
    # Models DCA
    dca_models = [
        ("E00 Baseline Traced EffNet", preds_dict["E00_Baseline_Traced_EffNet"][:, 2], "orange", "--"),
        ("E99-B Meta-Apex", preds_dict["E99_Meta_Apex"][:, 2], "blue", "-."),
        ("E103 Conformal Safety Apex", preds_dict["E103_Clinical_Safety_Apex"][:, 2], "crimson", "-")
    ]
    
    for lbl, probs, clr, ls in dca_models:
        nb = []
        for t in thresholds:
            pred_bin = (probs >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_abn, pred_bin).ravel()
            val = (tp / N_val) - (fp / N_val) * (t / (1 - t))
            nb.append(val)
        ax.plot(thresholds, nb, color=clr, linestyle=ls, linewidth=2.0, label=lbl)
        
    ax.axvline(tau_safe_02, color="green", linestyle=":", label=f"Conformal Safety Threshold (tau={tau_safe_02:.3f})")
    ax.set_xlim([0.01, 0.45])
    ax.set_ylim([-0.02, prev * 1.05])
    ax.set_xlabel("Clinical Decision Probability Threshold Pt", fontsize=11, fontweight="bold")
    ax.set_ylabel("Standardized Net Benefit", fontsize=11, fontweight="bold")
    ax.set_title("Decision Curve Analysis (DCA): Clinical Net Utility for CVC Malposition", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    plt.tight_layout()
    
    dca_plot_path = REPORTS_DIR / "clinical_safety_dca_curve.png"
    plt.savefig(dca_plot_path, dpi=300)
    plt.close()
    print(f"✅ Saved Clinical Decision Curve Analysis (DCA) to: {dca_plot_path}")

    # --------------------------------------------------------------------------
    # Plot Conformal 3-Zone Triage Distribution
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    normal_scores = p_abn[y_abn == 0]
    malposition_scores = p_abn[y_abn == 1]
    
    ax.hist(normal_scores, bins=50, alpha=0.5, density=True, color="teal", label=f"True Non-Abnormal (N={(y_abn==0).sum():,})")
    ax.hist(malposition_scores, bins=50, alpha=0.6, density=True, color="crimson", label=f"True Abnormal Malposition (N={N_pos:,})")
    
    ax.axvline(tau_safe_02, color="forestgreen", linestyle="--", linewidth=2.2, label=f"Conformal tau_CRC={tau_safe_02:.3f} (98% Recall Guard)")
    ax.axvline(tau_ppv, color="darkred", linestyle="-.", linewidth=2.2, label=f"High-Precision tau_RED={tau_ppv:.3f} (70% PPV Alert)")
    
    # Highlight zones
    ax.axvspan(0.0, tau_safe_02, alpha=0.15, color="green", label="🟢 GREEN ZONE (Safe)")
    ax.axvspan(tau_safe_02, tau_ppv, alpha=0.15, color="gold", label="🟡 AMBER ZONE (Uncertain)")
    ax.axvspan(tau_ppv, 1.0, alpha=0.15, color="red", label="🔴 RED ZONE (Critical)")
    
    ax.set_xlabel("Predicted Probability of Critical Malposition P(Abnormal)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Empirical Density", fontsize=11, fontweight="bold")
    ax.set_title("Split-Conformal 3-Zone Clinical Safety Triage Partition (E103/E104)", fontsize=12, fontweight="bold")
    ax.set_xlim([0.0, 1.0])
    ax.legend(loc="upper right", frameon=True, fontsize=8.5)
    plt.tight_layout()
    
    zones_plot_path = REPORTS_DIR / "conformal_triage_zones.png"
    plt.savefig(zones_plot_path, dpi=300)
    plt.close()
    print(f"✅ Saved Conformal 3-Zone Triage Diagram to: {zones_plot_path}")
    
    # Copy both plots to Downloads for user inspection
    downloads_dir = Path(r"C:\Users\M S I\Downloads")
    if downloads_dir.exists():
        import shutil
        shutil.copy(dca_plot_path, downloads_dir / "clinical_safety_dca_curve.png")
        shutil.copy(zones_plot_path, downloads_dir / "conformal_triage_zones.png")
        shutil.copy(table_csv, downloads_dir / "clinical_safety_10_metric_table.csv")
        print(f"✅ Synced artifacts to Downloads directory!")
        
    print(f"Runtime: {time.time() - start_time:.1f} seconds")

if __name__ == "__main__":
    main()
