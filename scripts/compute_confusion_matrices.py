import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

ROOT = Path(r"C:\Ankshit\CVC-MAGNet-V2")
MANIFEST_PATH = ROOT / "data" / "splits" / "placement_locked_manifest.parquet"
VAL_PREDS_E99 = ROOT / "runs" / "val_preds" / "val_preds_E99_grand_meta_apex.npy"
VAL_PREDS_E96 = ROOT / "runs" / "val_preds" / "val_preds_E96_noisy_student_densenet121_512.npy"
VAL_PREDS_E00 = ROOT / "runs" / "val_preds" / "val_preds_E00_traced_effnet_baseline.npy"

df = pd.read_parquet(MANIFEST_PATH)
val_df = df[df["split"] == "val"].reset_index(drop=True)
print(f"Locked Validation Cohort Size: N={len(val_df):,}")

CVC_CLASSES = ["CVC - Normal", "CVC - Borderline", "CVC - Abnormal"]
y_true = val_df[CVC_CLASSES].values.astype(int)

# Multi-class true label (argmax assignment: 0=Normal, 1=Borderline, 2=Abnormal)
# Notice: Normal is dominant (1848), Borderline (729), Abnormal (266)
y_true_mc = np.argmax(y_true, axis=1)

def evaluate_predictions(preds_path, model_name):
    print("\n" + "=" * 85)
    print(f"CONFUSION MATRIX & CLINICAL METRICS: {model_name}")
    print("=" * 85)
    if not preds_path.exists():
        print(f"File not found: {preds_path}")
        return
    preds = np.load(preds_path)
    if preds.shape != y_true.shape:
        print(f"Shape mismatch: {preds.shape} vs {y_true.shape}")
        return
        
    y_pred_mc = np.argmax(preds, axis=1)
    
    # 1. 3x3 Multi-Class Confusion Matrix
    cm_3x3 = confusion_matrix(y_true_mc, y_pred_mc, labels=[0, 1, 2])
    cm_df = pd.DataFrame(
        cm_3x3,
        index=["True Normal (0)", "True Borderline (1)", "True Abnormal (2)"],
        columns=["Pred Normal (0)", "Pred Borderline (1)", "Pred Abnormal (2)"]
    )
    print("\n1. Multi-Class (3x3) Confusion Matrix (Raw Patient Counts):")
    print(cm_df)
    
    # Normalized row-wise (Recall / Sensitivity per class)
    cm_norm = cm_3x3.astype(float) / cm_3x3.sum(axis=1, keepdims=True)
    cm_norm_df = pd.DataFrame(
        cm_norm * 100,
        index=["True Normal (0)", "True Borderline (1)", "True Abnormal (2)"],
        columns=["Pred Normal (0)", "Pred Borderline (1)", "Pred Abnormal (2)"]
    )
    print("\nNormalized Confusion Matrix (Recall / Sensitivity % per row):")
    print(cm_norm_df.round(1).astype(str) + " %")
    
    acc = np.mean(y_true_mc == y_pred_mc)
    print(f"\n3-Class Overall Accuracy: {acc:.2%}")

    # 2. Binary Breakdown per class (Multi-label evaluation)
    print("\n2. Binary Classification Breakdown per Clinical Category (Optimal Youden Threshold):")
    print(f"{'Class':<18} | {'Threshold':<9} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'TN':<5} | {'Sensitivity':<11} | {'Specificity':<11} | {'Precision':<10} | {'F1-Score':<8}")
    print("-" * 115)
    for idx, cls_name in enumerate(CVC_CLASSES):
        yt = y_true[:, idx]
        yp_prob = preds[:, idx]
        
        # Calculate optimal Youden index threshold on validation set
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(yt, yp_prob)
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        th = thresholds[best_idx]
        
        yp = (yp_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        
        print(f"{cls_name:<18} | {th:.4f}    | {tp:<5} | {fp:<5} | {fn:<5} | {tn:<5} | {sens:<11.2%} | {spec:<11.2%} | {ppv:<10.2%} | {f1:.4f}")

evaluate_predictions(VAL_PREDS_E99, "E99-B CVC-MAGNet-V2 Grand Meta-Stacker Apex (0.8499 Macro AUROC)")
evaluate_predictions(VAL_PREDS_E96, "E96 Distilled DenseNet-121 Standalone (0.8115 Macro AUROC)")
evaluate_predictions(VAL_PREDS_E00, "E00 Unanchored Baseline GAP CNN Floor (0.7108 Macro AUROC)")
