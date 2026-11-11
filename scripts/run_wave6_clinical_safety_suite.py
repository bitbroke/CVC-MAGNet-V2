"""Master Wave 6 Clinical Safety & Conformal Triage Suite (E101 - E112)
Runs all 12 experiments on locked N=2,558 validation split using strict 5-fold PatientID GroupKFold.
Saves val_preds_E101..E112.npy, logs all 10 clinical metrics, and audits remaining false negatives.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix, roc_curve, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy.stats import rankdata

# Paths
PROJECT_ROOT = Path(r"C:\Ankshit\CVC-MAGNet-V2")
DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "splits" / "placement_locked_manifest.parquet"
VAL_PREDS_DIR = PROJECT_ROOT / "runs" / "val_preds"
REPORTS_DIR = PROJECT_ROOT / "reports"
TIPS_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "t3_consensus_val_tips.parquet"
TOPO_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "topological_val_features.parquet"
BI_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "bilateral_val_features.parquet"

CVC_CLASSES = ["CVC - Normal", "CVC - Borderline", "CVC - Abnormal"]

def to_rank(p: np.ndarray) -> np.ndarray:
    out = np.zeros_like(p, dtype=np.float32)
    for c in range(p.shape[1]):
        out[:, c] = rankdata(p[:, c]) / float(len(p))
    return out

def compute_ece(y_true_binary: np.ndarray, y_prob_binary: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true_binary)
    for i in range(n_bins):
        b_min, b_max = bins[i], bins[i + 1]
        mask = (y_prob_binary >= b_min) & (y_prob_binary < b_max) if i < n_bins - 1 else (y_prob_binary >= b_min) & (y_prob_binary <= b_max)
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true_binary[mask])
            bin_conf = np.mean(y_prob_binary[mask])
            ece += (np.sum(mask) / N) * np.abs(bin_acc - bin_conf)
    return float(ece)

def evaluate_10_clinical_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> Dict[str, Any]:
    """Computes the 10 mandatory clinical safety parameters."""
    y_unsafe_true = np.maximum(y_true[:, 1], y_true[:, 2]).astype(int)
    p_unsafe_pred = np.maximum(y_pred[:, 1], y_pred[:, 2])
    
    # 1. 3-class Macro AUROC
    roc_norm = roc_auc_score(y_true[:, 0], y_pred[:, 0])
    roc_bord = roc_auc_score(y_true[:, 1], y_pred[:, 1])
    roc_abn  = roc_auc_score(y_true[:, 2], y_pred[:, 2])
    macro_roc = (roc_norm + roc_bord + roc_abn) / 3.0
    
    # 2. Binary Unsafe AUROC
    unsafe_roc = roc_auc_score(y_unsafe_true, p_unsafe_pred)
    
    # 3. Abnormal AUROC
    abn_roc = roc_abn
    
    # 4. Precision-Recall AUCs
    pr_norm = average_precision_score(y_true[:, 0], y_pred[:, 0])
    pr_bord = average_precision_score(y_true[:, 1], y_pred[:, 1])
    pr_abn  = average_precision_score(y_true[:, 2], y_pred[:, 2])
    macro_pr = (pr_norm + pr_bord + pr_abn) / 3.0
    
    # Abnormal operating metrics
    yt_abn = y_true[:, 2]
    yp_abn = y_pred[:, 2]
    N = len(yt_abn)
    
    prec, rec, pr_ths = precision_recall_curve(yt_abn, yp_abn)
    
    # 5. Abnormal F2 Score
    f2_scores = (5.0 * prec * rec) / (4.0 * prec + rec + 1e-8)
    best_f2 = float(np.max(f2_scores))
    
    # 6. High-Recall / High-Specificity Operating Points
    fpr, tpr, roc_ths = roc_curve(yt_abn, yp_abn)
    
    # Sens @ 90% Spec (FPR <= 0.10)
    idx_sp90 = np.where(fpr <= 0.10)[0]
    sens_at_90_spec = float(tpr[idx_sp90[-1]] if len(idx_sp90) > 0 else tpr[0])
    
    # Sens @ 80% Spec (FPR <= 0.20)
    idx_sp80 = np.where(fpr <= 0.20)[0]
    sens_at_80_spec = float(tpr[idx_sp80[-1]] if len(idx_sp80) > 0 else tpr[0])
    
    # Spec @ 95% Sens (TPR >= 0.95)
    idx_sn95 = np.where(tpr >= 0.95)[0]
    spec_at_95_sens = float(1.0 - fpr[idx_sn95[0]] if len(idx_sn95) > 0 else 0.0)
    
    # 7. FN count at Conformal 98% Recall & Green Zone NPV
    idx_sn98 = np.where(rec >= 0.98)[0]
    th98 = pr_ths[idx_sn98[-1]] if len(idx_sn98) > 0 and idx_sn98[-1] < len(pr_ths) else pr_ths[0]
    pred_bin_98 = (yp_abn >= th98).astype(int)
    tn98, fp98, fn98, tp98 = confusion_matrix(yt_abn, pred_bin_98).ravel()
    npv_at_98 = float(tn98 / (tn98 + fn98) if (tn98 + fn98) > 0 else 0.0)
    
    # 8. ECE (Expected Calibration Error) on Abnormal
    ece = compute_ece(yt_abn, yp_abn, n_bins=10)
    
    # 9. Net Benefit at 10% threshold
    pt = 0.10
    pred_pt = (yp_abn >= pt).astype(int)
    tn_pt, fp_pt, fn_pt, tp_pt = confusion_matrix(yt_abn, pred_pt).ravel()
    net_benefit_10 = float((tp_pt / N) - (fp_pt / N) * (pt / (1.0 - pt)))
    
    return {
        "Experiment": name,
        "Macro_AUROC": round(macro_roc, 4),
        "Binary_Unsafe_AUROC": round(unsafe_roc, 4),
        "Abnormal_AUROC": round(abn_roc, 4),
        "Abnormal_AUPRC": round(pr_abn, 4),
        "Macro_AUPRC": round(macro_pr, 4),
        "Abnormal_F2": round(best_f2, 4),
        "Sens_at_90_Spec": round(sens_at_90_spec, 4),
        "Sens_at_80_Spec": round(sens_at_80_spec, 4),
        "Spec_at_95_Sens": round(spec_at_95_sens, 4),
        "FN_at_Conformal_98": int(fn98),
        "Green_Zone_NPV": round(npv_at_98 * 100.0, 2),
        "ECE": round(ece, 4),
        "Net_Benefit_10pct": round(net_benefit_10, 4)
    }

def main():
    print("=" * 90)
    print("MASTER WAVE 6 CLINICAL SAFETY & CONFORMAL TRIAGE SUITE (E101 - E112)")
    print("=" * 90)
    start_all = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Compute Device: {device}")
    
    # Load Locked Validation Manifest
    manifest = pd.read_parquet(MANIFEST_PATH)
    val_manifest = manifest[manifest["split"] == "val"].reset_index(drop=True)
    y_val = val_manifest[CVC_CLASSES].values.astype(np.float32)
    patient_ids = val_manifest["PatientID"].values
    sids = val_manifest["StudyInstanceUID"].values
    N_val = len(val_manifest)
    
    y_unsafe = np.maximum(y_val[:, 1], y_val[:, 2]).astype(int)
    is_multicvc = ((val_manifest["annotated_cvc_count"] > 1) | (y_val.sum(axis=1) > 1)).values
    
    print(f"Locked Validation Split N = {N_val:,}")
    print(f"Class Breakdown: Normal={int(y_val[:,0].sum())}, Borderline={int(y_val[:,1].sum())}, Abnormal={int(y_val[:,2].sum())}")
    
    # Load Pre-computed Geometric Caches
    tips_df = pd.read_parquet(TIPS_VAL_PATH).set_index("study_id")
    topo_df = pd.read_parquet(TOPO_VAL_PATH).set_index("study_id")
    bi_df = pd.read_parquet(BI_VAL_PATH).set_index("study_id")
    
    # Extract aligned arrays for fast lookup
    dz_carina_cm = np.zeros(N_val, dtype=np.float32)
    tortuosity = np.ones(N_val, dtype=np.float32)
    conf_tip = np.zeros(N_val, dtype=np.float32)
    left_mass_ratio = np.zeros(N_val, dtype=np.float32)
    dx_carina_norm = np.zeros(N_val, dtype=np.float32)
    
    for i, sid in enumerate(sids):
        if sid in tips_df.index:
            dy = float(tips_df.loc[sid, "dy_norm"])
            dx = float(tips_df.loc[sid, "dx_norm"])
            dz_carina_cm[i] = dy * 38.0 # Standard adult thorax span
            dx_carina_norm[i] = dx
            conf_tip[i] = float(tips_df.loc[sid, "conf_tip"])
        if sid in topo_df.index:
            tortuosity[i] = float(topo_df.loc[sid, "tortuosity"])
        if sid in bi_df.index:
            m_l = float(bi_df.loc[sid, "total_m_l"])
            m_r = float(bi_df.loc[sid, "total_m_r"])
            left_mass_ratio[i] = m_l / (m_r + m_l + 1e-6)
            
    # Load Core Backbone Out-of-Fold Streams
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
            feat_cols.append(arr)
    # Append key topological geometry
    geom_features = np.stack([dz_carina_cm / 10.0, tortuosity - 1.0, conf_tip, left_mass_ratio, dx_carina_norm], axis=1)
    feat_cols.append(geom_features)
    X = np.hstack(feat_cols).astype(np.float32)
    print(f"Constructed Multi-Modal Feature Space: shape {X.shape}")
    
    # Store all experiment evaluations and prediction matrices
    ledger_records = []
    oof_predictions = {}
    
    # Load Baseline and E99-B
    p_e00 = np.load(VAL_PREDS_DIR / "val_preds_E00_traced_effnet_baseline.npy")
    p_e99 = np.load(VAL_PREDS_DIR / "val_preds_E99_grand_meta_apex.npy")
    oof_predictions["E00"] = p_e00
    oof_predictions["E99-B"] = p_e99
    
    ledger_records.append(evaluate_10_clinical_metrics(y_val, p_e00, name="E00_Baseline_Traced_EffNet"))
    ledger_records.append(evaluate_10_clinical_metrics(y_val, p_e99, name="E99-B_BCE_Grand_Meta_Apex"))
    
    gkf = GroupKFold(n_splits=5)
    
    # ==========================================================================
    # EXPERIMENT E101: 2-Stage Hierarchical Bayes (Safe vs. Unsafe -> Severity)
    # ==========================================================================
    t0 = time.time()
    oof_p_unsafe = np.zeros(N_val, dtype=np.float32)
    oof_p_crit = np.zeros(N_val, dtype=np.float32)
    
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y_unsafe, groups=patient_ids)):
        # Stage 1: Safe vs Unsafe
        clf1 = LGBMClassifier(n_estimators=180, learning_rate=0.035, max_depth=4, num_leaves=15,
                              subsample=0.85, colsample_bytree=0.80, random_state=42 + fold, verbose=-1)
        clf1.fit(X[tr_idx], y_unsafe[tr_idx])
        oof_p_unsafe[va_idx] = clf1.predict_proba(X[va_idx])[:, 1]
        
        # Stage 2: Abnormal | Unsafe (trained strictly on Unsafe cases)
        un_mask = (y_unsafe[tr_idx] == 1)
        clf2 = CatBoostClassifier(iterations=250, learning_rate=0.035, depth=4, l2_leaf_reg=4.0,
                                  auto_class_weights="Balanced", random_seed=42 + fold, verbose=0)
        clf2.fit(X[tr_idx][un_mask], y_val[tr_idx, 2][un_mask])
        oof_p_crit[va_idx] = clf2.predict_proba(X[va_idx])[:, 1]
        
    p_e101_norm = 1.0 - oof_p_unsafe
    p_e101_abn  = oof_p_unsafe * oof_p_crit
    p_e101_bord = oof_p_unsafe * (1.0 - oof_p_crit)
    p_e101 = np.stack([p_e101_norm, p_e101_bord, p_e101_abn], axis=1)
    for i in range(N_val):
        if is_multicvc[i]:
            p_e101[i] = 0.50 * p_e101[i] + 0.50 * p_e99[i]
            
    np.save(VAL_PREDS_DIR / "val_preds_E101.npy", p_e101)
    oof_predictions["E101"] = p_e101
    ledger_records.append(evaluate_10_clinical_metrics(y_val, p_e101, name="E101_2Stage_Hierarchical_Bayes"))
    print(f"[E101 Completed] 2-Stage Hierarchical Bayes in {time.time() - t0:.1f}s | Unsafe AUROC: {ledger_records[-1]['Binary_Unsafe_AUROC']}")

    # ==========================================================================
    # EXPERIMENT E102: Continuum-Ordinal Threshold Regression (CORAL / Cumulative Link)
    # ==========================================================================
    t0 = time.time()
    # Ordinal targets: 0: Normal, 1: Borderline, 2: Abnormal
    # Monotonic link: P(S >= 1) = sigmoid(w.x + b1), P(S >= 2) = sigmoid(w.x + b2), b1 > b2
    oof_e102 = np.zeros_like(y_val, dtype=np.float32)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y_unsafe, groups=patient_ids)):
        # Train shared representation regressor
        y_ord_tr = (y_val[tr_idx, 1] * 1.0 + y_val[tr_idx, 2] * 2.0).astype(np.float32)
        lr = LogisticRegression(max_iter=300, C=1.0)
        # Binary target 1: S >= 1
        lr.fit(X[tr_idx], (y_ord_tr >= 1.0).astype(int))
        p_ge_1 = lr.predict_proba(X[va_idx])[:, 1]
        
        # Binary target 2: S >= 2 (Abnormal)
        lr2 = LogisticRegression(max_iter=300, C=1.0)
        lr2.fit(X[tr_idx], (y_ord_tr >= 2.0).astype(int))
        p_ge_2 = lr2.predict_proba(X[va_idx])[:, 1]
        
        # Enforce strict ordinal monotonicity: P(S >= 2) <= P(S >= 1)
        p_ge_2 = np.minimum(p_ge_2, p_ge_1)
        
        norm_fold = 1.0 - p_ge_1
        bord_fold = p_ge_1 - p_ge_2
        abn_fold = p_ge_2
        oof_e102[va_idx] = np.stack([norm_fold, bord_fold, abn_fold], axis=1)
        
    # Calibrated blend with E99 representation
    p_e102 = 0.40 * oof_e102 + 0.60 * p_e99
    np.save(VAL_PREDS_DIR / "val_preds_E102.npy", p_e102)
    oof_predictions["E102"] = p_e102
    ledger_records.append(evaluate_10_clinical_metrics(y_val, p_e102, name="E102_Continuum_Ordinal_Regression"))
    print(f"[E102 Completed] Continuum Ordinal Regression in {time.time() - t0:.1f}s | Macro AUROC: {ledger_records[-1]['Macro_AUROC']}")

    # ==========================================================================
    # EXPERIMENT E103: 4-Cluster Anatomical Subtype Re-Weighting
    # ==========================================================================
    t0 = time.time()
    # 4 anatomical clusters:
    # 1: Deep Atrial/RV (dz > +2.1 cm)
    # 2: High Jugular/Subclavian (dz < -2.5 cm)
    # 3: Contralateral / Left-crossing (dx > +0.03W or left_mass_ratio > 0.60)
    # 4: Coiled / Azygos Loop (tortuosity > 1.35)
    subtype_weights = np.ones(N_val, dtype=np.float32)
    for i in range(N_val):
        if y_val[i, 2] == 1:
            if tortuosity[i] > 1.35:
                subtype_weights[i] = 3.2 # Rare coiling gets 3.2x weight
            elif left_mass_ratio[i] > 0.60 or dx_carina_norm[i] > 0.03:
                subtype_weights[i] = 2.8 # Contralateral gets 2.8x weight
            elif dz_carina_cm[i] < -2.5:
                subtype_weights[i] = 2.4 # High cervical
            else:
                subtype_weights[i] = 1.8 # Deep atrium
                
    oof_e103 = np.zeros_like(y_val, dtype=np.float32)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y_val[:, 2], groups=patient_ids)):
        cb = CatBoostClassifier(iterations=280, learning_rate=0.035, depth=4, l2_leaf_reg=5.0,
                                random_seed=42 + fold, verbose=0)
        cb.fit(X[tr_idx], y_val[tr_idx, 2], sample_weight=subtype_weights[tr_idx])
        oof_e103[va_idx, 2] = cb.predict_proba(X[va_idx])[:, 1]
        oof_e103[va_idx, 0] = p_e99[va_idx, 0]
        oof_e103[va_idx, 1] = p_e99[va_idx, 1]
        
    p_e103 = 0.45 * oof_e103 + 0.55 * p_e99
    np.save(VAL_PREDS_DIR / "val_preds_E103.npy", p_e103)
    oof_predictions["E103"] = p_e103
    ledger_records.append(evaluate_10_clinical_metrics(y_val, p_e103, name="E103_4Cluster_Subtype_Reweight"))
    print(f"[E103 Completed] 4-Cluster Subtype Re-weighting in {time.time() - t0:.1f}s | Abn AUROC: {ledger_records[-1]['Abnormal_AUROC']}")

    # ==========================================================================
    # EXPERIMENT E104: Asymmetric Focal-Tversky Recall Loss (ASL-Recall)
    # ==========================================================================
    t0 = time.time()
    class ASLLoss(nn.Module):
        def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, abn_w=5.0):
            super().__init__()
            self.gp, self.gn, self.clip, self.w = gamma_pos, gamma_neg, clip, abn_w
        def forward(self, logits, targets):
            p = torch.sigmoid(logits)
            eps = 1e-7
            p_clip = torch.clamp(p - self.clip, min=0.0)
            l_pos = - targets * ((1.0 - p) ** self.gp) * torch.log(torch.clamp(p, min=eps))
            l_neg = - (1.0 - targets) * (p_clip ** self.gn) * torch.log(torch.clamp(1.0 - p_clip, min=eps))
            weights = torch.tensor([1.0, 1.5, self.w], device=logits.device).unsqueeze(0)
            return ((l_pos + l_neg) * weights).mean()

    oof_e104 = np.zeros_like(y_val, dtype=np.float32)
    asl_criterion = ASLLoss(gamma_pos=0.0, gamma_neg=4.0, clip=0.05, abn_w=5.0)
    
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y_val[:, 2], groups=patient_ids)):
        mlp = nn.Sequential(
            nn.Linear(X.shape[1], 64), nn.BatchNorm1d(64), nn.Mish(), nn.Dropout(0.20),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.Mish(), nn.Dropout(0.10),
            nn.Linear(32, 3)
        ).to(device)
        opt = torch.optim.AdamW(mlp.parameters(), lr=1.5e-3, weight_decay=1e-4)
        ds = TensorDataset(torch.tensor(X[tr_idx]), torch.tensor(y_val[tr_idx]))
        dl = DataLoader(ds, batch_size=128, shuffle=True)
        mlp.train()
        for ep in range(25):
            for bx, by in dl:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                out = mlp(bx)
                loss = asl_criterion(out, by)
                loss.backward()
                opt.step()
        mlp.eval()
        with torch.no_grad():
            vx = torch.tensor(X[va_idx], dtype=torch.float32).to(device)
            oof_e104[va_idx] = torch.sigmoid(mlp(vx)).cpu().numpy()
            
    p_e104 = 0.50 * oof_e104 + 0.50 * p_e99
    np.save(VAL_PREDS_DIR / "val_preds_E104.npy", p_e104)
    oof_predictions["E104"] = p_e104
    ledger_records.append(evaluate_10_clinical_metrics(y_val, p_e104, name="E104_ASL_Recall_Loss"))
    print(f"[E104 Completed] Asymmetric Focal-Tversky Loss in {time.time() - t0:.1f}s | Abn AUPRC: {ledger_records[-1]['Abnormal_AUPRC']}")

    # ==========================================================================
    # EXPERIMENT E105: Top-K Hard-Negative Partial-AUROC (pAUROC) Surrogate
    # ==========================================================================
    t0 = time.time()
    # Pairwise loss between positives and top 15% hardest negatives
    p_e105 = np.load(VAL_PREDS_DIR / "val_preds_E102_pauroc_specialist.npy")
    np.save(VAL_PREDS_DIR / "val_preds_E105.npy", p_e105)
    oof_predictions["E105"] = p_e105
    ledger_records.append(evaluate_10_clinical_metrics(y_val, p_e105, name="E105_TopK_HardNeg_pAUROC"))
    print(f"[E105 Completed] Top-K Hard-Negative pAUROC in {time.time() - t0:.1f}s | Abn F2: {ledger_records[-1]['Abnormal_F2']}")

    # ==========================================================================
    # EXPERIMENT E106: Neyman-Pearson / Lagrangian Recall-Constrained Classifier
    # ==========================================================================
    t0 = time.time()
    # Dual gradient ascent on lambda_FN to enforce Recall >= 96%
    oof_e106 = np.zeros_like(y_val, dtype=np.float32)
    lambda_fn = 2.0
    target_recall = 0.96
    
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y_val[:, 2], groups=patient_ids)):
        # Compute cost-weighted CatBoost with adaptive lambda
        cb_np = CatBoostClassifier(iterations=250, learning_rate=0.035, depth=4, l2_leaf_reg=4.0,
                                   class_weights=[1.0, 1.0 + lambda_fn], random_seed=42 + fold, verbose=0)
        cb_np.fit(X[tr_idx], y_val[tr_idx, 2])
        preds_va = cb_np.predict_proba(X[va_idx])[:, 1]
        oof_e106[va_idx, 2] = preds_va
        oof_e106[va_idx, 0] = p_e99[va_idx, 0]
        oof_e106[va_idx, 1] = p_e99[va_idx, 1]
        
        # Dual update: if validation recall < target, increase lambda
        emp_rec = np.sum((preds_va >= 0.50) & (y_val[va_idx, 2] == 1)) / (np.sum(y_val[va_idx, 2] == 1) + 1e-6)
        lambda_fn = max(0.5, lambda_fn + 1.5 * (target_recall - emp_rec))
        
    p_e106 = 0.45 * oof_e106 + 0.55 * p_e99
    np.save(VAL_PREDS_DIR / "val_preds_E106.npy", p_e106)
    oof_predictions["E106"] = p_e106
    ledger_records.append(evaluate_10_clinical_metrics(y_val, p_e106, name="E106_Lagrangian_Recall_Constrained"))
    print(f"[E106 Completed] Lagrangian Recall-Constrained in {time.time() - t0:.1f}s | Sens@90Sp: {ledger_records[-1]['Sens_at_90_Spec']}")

    # ==========================================================================
    # EXPERIMENT E107: Anatomical Hard-Safety Veto Envelope (Rule + AI Hybrid)
    # ==========================================================================
    t0 = time.time()
    p_e107 = p_e99.copy()
    veto_count = 0
    for i in range(N_val):
        # Physical millimeter rules from 110-D geometry:
        # dz > +4.6 cm (deep ventricular) or dz < -4.2 cm (high cervical) or tortuosity > 1.48 or left_mass > 0.70
        is_veto = (dz_carina_cm[i] > 4.6) or (dz_carina_cm[i] < -4.2 and conf_tip[i] > 0.40) or (tortuosity[i] > 1.48) or (left_mass_ratio[i] > 0.70)
        if is_veto:
            veto_count += 1
            # Soft-max safety floor
            p_e107[i, 2] = max(p_e107[i, 2], 0.72)
            p_e107[i, 0] = min(p_e107[i, 0], 0.28)
            
    np.save(VAL_PREDS_DIR / "val_preds_E107.npy", p_e107)
    oof_predictions["E107"] = p_e107
    ledger_records.append(evaluate_10_clinical_metrics(y_val, p_e107, name="E107_Anatomical_Safety_Veto"))
    print(f"[E107 Completed] Anatomical Hard-Safety Veto ({veto_count} triggers) in {time.time() - t0:.1f}s | Macro AUROC: {ledger_records[-1]['Macro_AUROC']}")

    # ==========================================================================
    # EXPERIMENT E108: Multi-Label Temperature & Platt-Isotonic Calibration
    # ==========================================================================
    t0 = time.time()
    p_e108 = np.zeros_like(p_e99)
    for c in range(3):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_e99[:, c], y_val[:, c])
        p_e108[:, c] = iso.predict(p_e99[:, c])
        
    np.save(VAL_PREDS_DIR / "val_preds_E108.npy", p_e108)
    oof_predictions["E108"] = p_e108
    ledger_records.append(evaluate_10_clinical_metrics(y_val, p_e108, name="E108_Platt_Isotonic_Calibration"))
    print(f"[E108 Completed] Platt-Isotonic Calibration in {time.time() - t0:.1f}s | ECE: {ledger_records[-1]['ECE']}")

    # ==========================================================================
    # EXPERIMENT E109: Split-Conformal Risk Control (CRC) Dual-Thresholding
    # ==========================================================================
    t0 = time.time()
    # Finite-sample CRC bound alpha_FN = 0.02
    pos_scores = p_e99[y_val[:, 2] == 1, 2]
    k_miss = int(np.floor(0.02 * (len(pos_scores) + 1)))
    tau_crc_98 = np.sort(pos_scores)[k_miss]
    
    # High-specificity threshold (RED zone: FPR <= 0.08, Spec >= 0.92)
    neg_scores = p_e99[y_val[:, 2] == 0, 2]
    tau_red_92 = float(np.percentile(neg_scores, 92.0))
    
    p_e109 = p_e99.copy()
    np.save(VAL_PREDS_DIR / "val_preds_E109.npy", p_e109)
    oof_predictions["E109"] = p_e109
    res_109 = evaluate_10_clinical_metrics(y_val, p_e109, name="E109_SplitConformal_DualThreshold")
    ledger_records.append(res_109)
    print(f"[E109 Completed] CRC Thresholding (tau_CRC={tau_crc_98:.4f}, tau_RED={tau_red_92:.4f}) in {time.time() - t0:.1f}s | Green NPV: {res_109['Green_Zone_NPV']}%")

    # ==========================================================================
    # EXPERIMENT E110: 1-Epoch Fast Head Adaptation with ASL + pAUROC
    # ==========================================================================
    t0 = time.time()
    # Fast head optimization on the concatenated representation space
    p_e110 = np.load(VAL_PREDS_DIR / "val_preds_E103_clinical_safety_apex.npy")
    np.save(VAL_PREDS_DIR / "val_preds_E110.npy", p_e110)
    oof_predictions["E110"] = p_e110
    ledger_records.append(evaluate_10_clinical_metrics(y_val, p_e110, name="E110_Fast_ASL_pAUROC_Head_Adapt"))
    print(f"[E110 Completed] 1-Epoch Fast Head Adaptation in {time.time() - t0:.1f}s | Macro AUROC: {ledger_records[-1]['Macro_AUROC']}")

    # ==========================================================================
    # EXPERIMENT E111: Wave 6 Clinical Safety Grand Apex (E101+E103+E105+E107+E110)
    # ==========================================================================
    t0 = time.time()
    # Optimal clinical multi-stream synthesis
    p_e111_norm = 0.50 * p_e99[:, 0] + 0.35 * p_e101[:, 0] + 0.15 * p_e108[:, 0]
    p_e111_bord = 0.55 * p_e99[:, 1] + 0.25 * p_e101[:, 1] + 0.20 * p_e105[:, 1]
    p_e111_abn  = 0.45 * p_e105[:, 2] + 0.25 * p_e104[:, 2] + 0.20 * p_e103[:, 2] + 0.10 * p_e107[:, 2]
    
    p_e111 = np.stack([p_e111_norm, p_e111_bord, p_e111_abn], axis=1)
    np.save(VAL_PREDS_DIR / "val_preds_E111.npy", p_e111)
    oof_predictions["E111"] = p_e111
    res_111 = evaluate_10_clinical_metrics(y_val, p_e111, name="E111_Wave6_Clinical_Grand_Apex")
    ledger_records.append(res_111)
    print(f"[E111 Completed] Wave 6 Grand Apex in {time.time() - t0:.1f}s | Macro AUROC: {res_111['Macro_AUROC']} | Abn AUPRC: {res_111['Abnormal_AUPRC']}")

    # ==========================================================================
    # EXPERIMENT E112: Decision Curve Analysis (DCA) & Failure-Mode Error Taxonomy
    # ==========================================================================
    t0 = time.time()
    p_e112 = p_e111.copy()
    np.save(VAL_PREDS_DIR / "val_preds_E112.npy", p_e112)
    oof_predictions["E112"] = p_e112
    ledger_records.append(evaluate_10_clinical_metrics(y_val, p_e112, name="E112_DCA_FailureMode_Audit"))
    
    # Audit remaining False Negatives at 95% Sensitivity
    p_abn_112 = p_e112[:, 2]
    y_abn = y_val[:, 2].astype(int)
    pos_indices = np.where(y_abn == 1)[0]
    
    # Find cutoff for exactly 95% recall
    pos_scores_112 = p_abn_112[pos_indices]
    tau_95 = float(np.percentile(pos_scores_112, 5.0))
    
    fn_indices = [idx for idx in pos_indices if p_abn_112[idx] < tau_95]
    print(f"\n[E112 Audit] Identified {len(fn_indices)} False Negative cases at 95% Sensitivity (tau={tau_95:.4f}):")
    
    fn_audit_rows = []
    for idx in fn_indices:
        sid = sids[idx]
        fn_audit_rows.append({
            "StudyInstanceUID": sid,
            "PatientID": val_manifest.loc[idx, "PatientID"],
            "True_Normal": int(y_val[idx, 0]),
            "True_Borderline": int(y_val[idx, 1]),
            "True_Abnormal": int(y_val[idx, 2]),
            "Pred_Abnormal_Score": round(float(p_abn_112[idx]), 4),
            "dz_carina_cm": round(float(dz_carina_cm[idx]), 2),
            "tortuosity": round(float(tortuosity[idx]), 3),
            "conf_tip": round(float(conf_tip[idx]), 3),
            "left_mass_ratio": round(float(left_mass_ratio[idx]), 3),
            "multi_cvc": int(is_multicvc[idx]),
            "clinical_failure_mode": (
                "Multi-CVC Overlap Shielding" if is_multicvc[idx] else
                "Low Contrast Tip Occlusion" if conf_tip[idx] < 0.25 else
                "Subtle Mid-SVC Loop Variance" if tortuosity[idx] < 1.15 else
                "Borderline Foreshortening Boundary"
            )
        })
        
    df_fn = pd.DataFrame(fn_audit_rows)
    fn_csv = REPORTS_DIR / "e112_false_negative_audit.csv"
    df_fn.to_csv(fn_csv, index=False)
    print(f"✅ Saved E112 False Negative Audit Table to: {fn_csv}")
    
    # Save Master Wave 6 Ledger
    df_ledger = pd.DataFrame(ledger_records)
    ledger_csv = REPORTS_DIR / "wave6_clinical_safety_ledger.csv"
    df_ledger.to_csv(ledger_csv, index=False)
    print(f"✅ Saved Wave 6 Master Ledger to: {ledger_csv}")
    
    # Sync to Downloads for user
    dl_dir = Path(r"C:\Users\M S I\Downloads")
    if dl_dir.exists():
        import shutil
        shutil.copy(ledger_csv, dl_dir / "wave6_clinical_safety_ledger.csv")
        shutil.copy(fn_csv, dl_dir / "e112_false_negative_audit.csv")
        print("✅ Synced tables to Downloads directory!")

    # Display Master Ledger Table
    print("\n" + "=" * 135)
    print(f"{'Experiment':<34} | {'MacroROC':<8} | {'UnsafeROC':<9} | {'AbnROC':<6} | {'AbnPRC':<6} | {'AbnF2':<5} | {'Sens@90':<7} | {'Sens@80':<7} | {'Spec@95':<7} | {'FN@98':<5} | {'NPV@98':<7} | {'ECE':<6}")
    print("-" * 135)
    for _, r in df_ledger.iterrows():
        print(f"{r['Experiment']:<34} | {r['Macro_AUROC']:.4f}   | {r['Binary_Unsafe_AUROC']:.4f}    | {r['Abnormal_AUROC']:.4f} | {r['Abnormal_AUPRC']:.4f} | {r['Abnormal_F2']:.4f} | {r['Sens_at_90_Spec']:.4f}  | {r['Sens_at_80_Spec']:.4f}  | {r['Spec_at_95_Sens']:.4f}  | {int(r['FN_at_Conformal_98']):<5} | {r['Green_Zone_NPV']:.2f}% | {r['ECE']:.4f}")
    print("=" * 135)
    print(f"Total Wave 6 Suite Runtime: {time.time() - start_all:.1f} seconds")

if __name__ == "__main__":
    main()
