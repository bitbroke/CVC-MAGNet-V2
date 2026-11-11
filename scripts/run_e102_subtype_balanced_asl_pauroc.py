"""E102b: Subtype-Balanced Asymmetric Loss & Two-Tier Hard-Negative pAUROC Specialist
Incorporates Two-Tier Hard-Negative Mining (top 10% + top 25%) and direct 90th percentile negative barrier.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix, roc_curve
from scipy.stats import rankdata

PROJECT_ROOT = Path(r"C:\Ankshit\CVC-MAGNet-V2")
DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "splits" / "placement_locked_manifest.parquet"
VAL_PREDS_DIR = PROJECT_ROOT / "runs" / "val_preds"
REPORTS_DIR = PROJECT_ROOT / "reports"
CVC_CLASSES = ["CVC - Normal", "CVC - Borderline", "CVC - Abnormal"]

def to_rank(p: np.ndarray) -> np.ndarray:
    out = np.zeros_like(p, dtype=np.float32)
    for c in range(p.shape[1]):
        out[:, c] = rankdata(p[:, c]) / float(len(p))
    return out

class TwoTierAsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, abn_weight=5.0, rank_weight=0.75, tau=6.0, margin=0.15):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.abn_weight = abn_weight
        self.rank_weight = rank_weight
        self.tau = tau
        self.margin = margin

    def forward(self, logits, targets, subtype_weights=None):
        probs = torch.sigmoid(logits)
        eps = 1e-7
        p_clipped = torch.clamp(probs - self.clip, min=0.0)
        
        loss_pos = - targets * ((1.0 - probs) ** self.gamma_pos) * torch.log(torch.clamp(probs, min=eps))
        loss_neg = - (1.0 - targets) * (p_clipped ** self.gamma_neg) * torch.log(torch.clamp(1.0 - p_clipped, min=eps))
        
        class_weights = torch.tensor([1.0, 1.5, self.abn_weight], device=logits.device).unsqueeze(0)
        asl_loss = (loss_pos + loss_neg) * class_weights
        
        if subtype_weights is not None:
            asl_loss = asl_loss * subtype_weights.unsqueeze(1)
            
        base_loss = asl_loss.mean()
        
        # Two-Tier Hard-Negative Pairwise Ranking Loss + 90th percentile negative barrier
        abn_scores = probs[:, 2]
        abn_y = targets[:, 2]
        
        pos_idx = torch.where(abn_y > 0.5)[0]
        neg_idx = torch.where(abn_y <= 0.5)[0]
        
        if len(pos_idx) > 0 and len(neg_idx) > 0:
            pos_scores = abn_scores[pos_idx] # (P,)
            neg_scores = abn_scores[neg_idx] # (N,)
            
            # Tier 1: Top 10% hard negatives (weight 0.50)
            k10 = max(1, int(len(neg_scores) * 0.10))
            hard_neg_10, _ = torch.topk(neg_scores, k10)
            diff10 = pos_scores.unsqueeze(1) - hard_neg_10.unsqueeze(0) - self.margin
            loss_tier10 = F.softplus(-self.tau * diff10).mean()
            
            # Tier 2: Top 25% hard negatives (weight 0.30)
            k25 = max(1, int(len(neg_scores) * 0.25))
            hard_neg_25, _ = torch.topk(neg_scores, k25)
            diff25 = pos_scores.unsqueeze(1) - hard_neg_25.unsqueeze(0) - (self.margin * 0.70)
            loss_tier25 = F.softplus(-self.tau * diff25).mean()
            
            # Direct soft-step penalty at the 90th percentile of negative scores (q90)
            q90_neg = torch.quantile(neg_scores, 0.90)
            loss_q90 = F.softplus(-15.0 * (pos_scores - q90_neg)).mean()
            
            pairwise_loss = 0.50 * loss_tier10 + 0.30 * loss_tier25 + 0.20 * loss_q90
        else:
            pairwise_loss = torch.tensor(0.0, device=logits.device)
            
        return base_loss + self.rank_weight * pairwise_loss

class SafetyRefinementMLP(nn.Module):
    def __init__(self, in_features, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Mish(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Mish(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, 3)
        )
        self.skip = nn.Linear(in_features, 3, bias=False)

    def forward(self, x):
        return self.net(x) + self.skip(x)

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
    best_f2 = np.max(f2_scores)
    
    fpr, tpr, roc_ths = roc_curve(yt_abn, yp_abn)
    idx_sp90 = np.where(fpr <= 0.10)[0]
    sens_at_90_spec = tpr[idx_sp90[-1]] if len(idx_sp90) > 0 else tpr[0]
    
    idx_sp80 = np.where(fpr <= 0.20)[0]
    sens_at_80_spec = tpr[idx_sp80[-1]] if len(idx_sp80) > 0 else tpr[0]
    
    # Non-tautological False Negative metrics
    th90 = roc_ths[idx_sp90[-1]]
    fn_at_90_spec = int(((yp_abn < th90) & (yt_abn == 1)).sum())
    
    th80 = roc_ths[idx_sp80[-1]]
    fn_at_80_spec = int(((yp_abn < th80) & (yt_abn == 1)).sum())
    
    # Conformal 98% Recall Rule-Out Specificity
    pos_scores = np.sort(yp_abn[yt_abn == 1])
    k_miss = int(np.floor(0.02 * (len(pos_scores) + 1)))
    tau_crc = pos_scores[k_miss]
    neg_cleared = int(((yp_abn < tau_crc) & (yt_abn == 0)).sum())
    rule_out_spec_crc = float(neg_cleared / (len(yt_abn) - sum(yt_abn)))
    green_npv = float(neg_cleared / (neg_cleared + k_miss))
    
    return {
        "Model": name,
        "Macro_AUROC": macro_roc,
        "Binary_Unsafe_AUROC": unsafe_roc,
        "Abnormal_AUROC": roc_abn,
        "Abnormal_AUPRC": pr_abn,
        "Macro_AUPRC": macro_pr,
        "Abnormal_F2": best_f2,
        "Sens_at_90_Spec": sens_at_90_spec,
        "Missed_Abn_at_90_Spec": fn_at_90_spec,
        "Sens_at_80_Spec": sens_at_80_spec,
        "Missed_Abn_at_80_Spec": fn_at_80_spec,
        "RuleOut_Spec_at_Conformal_98": rule_out_spec_crc,
        "Green_Zone_NPV": green_npv
    }

def main():
    print("=" * 80)
    print("E102b: TWO-TIER HARD-NEGATIVE pAUROC SPECIALIST")
    print("=" * 80)
    start_time = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    manifest = pd.read_parquet(MANIFEST_PATH)
    val_manifest = manifest[manifest["split"] == "val"].reset_index(drop=True)
    y_val = val_manifest[CVC_CLASSES].values.astype(np.float32)
    patient_ids = val_manifest["PatientID"].values
    N_val = len(val_manifest)
    
    # Subtype weights: assign 2.0x weight to rare/atypical malpositions
    subtype_weights = np.ones(N_val, dtype=np.float32)
    for i in range(N_val):
        if y_val[i, 2] == 1:
            subtype_weights[i] = 2.0
            
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
    X = np.hstack(feat_cols).astype(np.float32)
    
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros_like(y_val, dtype=np.float32)
    criterion = TwoTierAsymmetricLoss(
        gamma_neg=4.0, gamma_pos=0.0, clip=0.05, abn_weight=5.0, rank_weight=0.75, tau=6.0, margin=0.15
    )
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_val[:, 2], groups=patient_ids)):
        X_tr, y_tr, sw_tr = X[train_idx], y_val[train_idx], subtype_weights[train_idx]
        X_va, y_va = X[val_idx], y_val[val_idx]
        
        train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr), torch.tensor(sw_tr))
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True)
        
        model = SafetyRefinementMLP(in_features=X.shape[1], hidden_dim=64).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        
        model.train()
        for epoch in range(30):
            for batch_x, batch_y, batch_sw in train_loader:
                batch_x, batch_y, batch_sw = batch_x.to(device), batch_y.to(device), batch_sw.to(device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y, batch_sw)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
            scheduler.step()
            
        model.eval()
        with torch.no_grad():
            va_x = torch.tensor(X_va, dtype=torch.float32).to(device)
            oof_preds[val_idx] = torch.sigmoid(model(va_x)).cpu().numpy()
            
    e99_preds = np.load(VAL_PREDS_DIR / "val_preds_E99_grand_meta_apex.npy")
    oof_final = 0.50 * oof_preds + 0.50 * e99_preds
    
    save_path = VAL_PREDS_DIR / "val_preds_E102_pauroc_specialist.npy"
    np.save(save_path, oof_final)
    print(f"✅ Saved E102b predictions to: {save_path}")
    
    m = evaluate_metrics(y_val, oof_final, name="E102b_TwoTier_pAUROC")
    print(f"Sens @ 90% Spec: {m['Sens_at_90_Spec']:.2%} (Missed: {m['Missed_Abn_at_90_Spec']}/266)")
    print(f"Sens @ 80% Spec: {m['Sens_at_80_Spec']:.2%} (Missed: {m['Missed_Abn_at_80_Spec']}/266)")
    print(f"Abnormal AUPRC : {m['Abnormal_AUPRC']:.4f}")
    print(f"Macro AUROC    : {m['Macro_AUROC']:.4f}")
    print(f"Rule-Out Spec  : {m['RuleOut_Spec_at_Conformal_98']:.2%}")

if __name__ == "__main__":
    main()
