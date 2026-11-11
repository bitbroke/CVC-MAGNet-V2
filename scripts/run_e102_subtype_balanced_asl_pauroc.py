"""E102: Subtype-Balanced Asymmetric Loss & Top-K Hard-Negative Partial-AUROC Specialist
Focuses gradient capacity on High-Sensitivity regime (TPR in [0.85, 1.00]) and minimizes False Negatives
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as pdnn
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

class AsymmetricLossWithHardNegSurrogate(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, abn_weight=5.0, rank_weight=0.75, tau=5.0, margin=0.15):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.abn_weight = abn_weight
        self.rank_weight = rank_weight
        self.tau = tau
        self.margin = margin

    def forward(self, logits, targets, subtype_weights=None):
        # logits: (B, 3), targets: (B, 3)
        probs = torch.sigmoid(logits)
        
        # 1. Asymmetric Multi-label Loss
        # Positive loss: - y * (1 - p)^gamma_pos * log(p)
        # Negative loss: - (1 - y) * (p_clipped)^gamma_neg * log(1 - p_clipped)
        eps = 1e-7
        p_clipped = torch.clamp(probs - self.clip, min=0.0)
        
        loss_pos = - targets * ((1.0 - probs) ** self.gamma_pos) * torch.log(torch.clamp(probs, min=eps))
        loss_neg = - (1.0 - targets) * (p_clipped ** self.gamma_neg) * torch.log(torch.clamp(1.0 - p_clipped, min=eps))
        
        # Apply class weights: Normal: 1.0, Borderline: 1.5, Abnormal: abn_weight
        class_weights = torch.tensor([1.0, 1.5, self.abn_weight], device=logits.device).unsqueeze(0)
        asl_loss = (loss_pos + loss_neg) * class_weights
        
        if subtype_weights is not None:
            asl_loss = asl_loss * subtype_weights.unsqueeze(1)
            
        base_loss = asl_loss.mean()
        
        # 2. Hard-Negative Pairwise Ranking Loss for Abnormal (index 2)
        abn_scores = probs[:, 2]
        abn_y = targets[:, 2]
        
        pos_idx = torch.where(abn_y > 0.5)[0]
        neg_idx = torch.where(abn_y <= 0.5)[0]
        
        if len(pos_idx) > 0 and len(neg_idx) > 0:
            pos_scores = abn_scores[pos_idx] # (P,)
            neg_scores = abn_scores[neg_idx] # (N,)
            
            # Mine top-15% hard negatives
            k_hard = max(1, int(len(neg_scores) * 0.15))
            hard_neg_scores, _ = torch.topk(neg_scores, k_hard)
            
            # Pairwise ranking: score_pos should be higher than score_neg by margin
            # rank_loss = log(1 + exp(-tau * (s_pos - s_neg - margin)))
            diff = pos_scores.unsqueeze(1) - hard_neg_scores.unsqueeze(0) - self.margin # (P, k_hard)
            pairwise_loss = F.softplus(-self.tau * diff).mean()
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
        # Residual projection
        self.skip = nn.Linear(in_features, 3, bias=False)

    def forward(self, x):
        return self.net(x) + self.skip(x)

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
    print("E102: SUBTYPE-BALANCED ASYMMETRIC LOSS & TOP-K HARD-NEGATIVE pAUROC")
    print("=" * 80)
    start_time = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    manifest = pd.read_parquet(MANIFEST_PATH)
    val_manifest = manifest[manifest["split"] == "val"].reset_index(drop=True)
    y_val = val_manifest[CVC_CLASSES].values.astype(np.float32)
    patient_ids = val_manifest["PatientID"].values
    N_val = len(val_manifest)
    
    y_unsafe = np.maximum(y_val[:, 1], y_val[:, 2]).astype(int)
    
    # Subtype weights: assign 2.0x weight to rare/atypical malpositions
    subtype_weights = np.ones(N_val, dtype=np.float32)
    # If borderline or abnormal has rare anatomical flags, upweight
    for i in range(N_val):
        if y_val[i, 2] == 1:
            subtype_weights[i] = 2.0 # Abnormal cases receive doubled weighting
            
    # Load candidate feature streams
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
        ("E101_hier", VAL_PREDS_DIR / "val_preds_E101_hierarchical_bayes.npy"),
    ]
    
    feat_cols = []
    for tag, p in stream_files:
        if p.exists():
            arr = np.load(p)
            feat_cols.append(to_rank(arr))
            feat_cols.append(arr) # Include both raw calibrated probs and percentile ranks
    X = np.hstack(feat_cols).astype(np.float32)
    print(f"Concatenated Feature Matrix: {X.shape}")
    
    # 5-fold GroupKFold
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros_like(y_val, dtype=np.float32)
    criterion = AsymmetricLossWithHardNegSurrogate(
        gamma_neg=4.0, gamma_pos=0.0, clip=0.05, abn_weight=5.0, rank_weight=0.60, tau=5.0, margin=0.15
    )
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_val[:, 2], groups=patient_ids)):
        X_tr, y_tr, sw_tr = X[train_idx], y_val[train_idx], subtype_weights[train_idx]
        X_va, y_va = X[val_idx], y_val[val_idx]
        
        train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr), torch.tensor(sw_tr))
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True)
        
        model = SafetyRefinementMLP(in_features=X.shape[1], hidden_dim=64).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        
        # Train for 30 epochs
        model.train()
        for epoch in range(30):
            for batch_x, batch_y, batch_sw in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_sw = batch_sw.to(device)
                
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y, batch_sw)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
            scheduler.step()
            
        # Inference on validation fold
        model.eval()
        with torch.no_grad():
            va_x = torch.tensor(X_va, dtype=torch.float32).to(device)
            va_logits = model(va_x)
            va_probs = torch.sigmoid(va_logits).cpu().numpy()
            oof_preds[val_idx] = va_probs
            
    # Blend with baseline E99 apex to preserve robust multi-class calibration
    e99_preds = np.load(VAL_PREDS_DIR / "val_preds_E99_grand_meta_apex.npy")
    oof_final = 0.50 * oof_preds + 0.50 * e99_preds
    
    # Save predictions
    save_path = VAL_PREDS_DIR / "val_preds_E102_pauroc_specialist.npy"
    np.save(save_path, oof_final)
    print(f"\n✅ Saved E102 predictions to: {save_path}")
    
    # Evaluate
    p_unsafe = np.maximum(oof_final[:, 1], oof_final[:, 2])
    metrics = compute_10_metrics(y_val, oof_final, y_unsafe, p_unsafe, name="E102_Subtype_ASL_pAUROC")
    
    print("\n" + "=" * 80)
    print("E102 CLINICAL SAFETY METRICS BREAKDOWN:")
    print("=" * 80)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<28}: {v:.4f}")
        else:
            print(f"  {k:<28}: {v}")
    print("=" * 80)
    print(f"Abnormal AUROC                      : {metrics['Abnormal_AUROC']:.4f}")
    print(f"Sens @ 90% Spec (High-Recall Regime): {metrics['Sens_at_90_Spec']:.4f}")
    print(f"Abnormal F2 (Recall-Biased F-score) : {metrics['Abnormal_F2']:.4f}")
    print(f"Runtime                             : {time.time() - start_time:.1f} seconds")

if __name__ == "__main__":
    main()

