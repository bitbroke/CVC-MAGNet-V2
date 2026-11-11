import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve
from pathlib import Path

ROOT = Path(r"C:\Ankshit\CVC-MAGNet-V2")
MANIFEST_PATH = ROOT / "data" / "splits" / "placement_locked_manifest.parquet"
VAL_PREDS_E99 = ROOT / "runs" / "val_preds" / "val_preds_E99_grand_meta_apex.npy"
OUT_IMG = ROOT / "reports" / "confusion_matrix_final_model.png"
DL_IMG = Path(r"C:\Users\M S I\Downloads\confusion_matrix_final_model.png")

df = pd.read_parquet(MANIFEST_PATH)
val_df = df[df["split"] == "val"].reset_index(drop=True)
CVC_CLASSES = ["CVC - Normal", "CVC - Borderline", "CVC - Abnormal"]
y_true = val_df[CVC_CLASSES].values.astype(int)
preds = np.load(VAL_PREDS_E99)

y_true_mc = np.argmax(y_true, axis=1)
y_pred_mc = np.argmax(preds, axis=1)

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 1. Raw Counts Heatmap
cm_3x3 = confusion_matrix(y_true_mc, y_pred_mc, labels=[0, 1, 2])
class_labels = ["Normal", "Borderline", "Abnormal"]

sns.heatmap(
    cm_3x3,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=True,
    xticklabels=class_labels,
    yticklabels=class_labels,
    ax=axes[0],
    annot_kws={"size": 14, "weight": "bold"}
)
axes[0].set_title("CVC-MAGNet-V2 Apex (E99-B): Raw Patient Counts\n(Locked Validation Cohort, N=2,558)", fontsize=13, fontweight="bold", pad=12)
axes[0].set_xlabel("Predicted Clinical Category", fontsize=12, fontweight="bold")
axes[0].set_ylabel("True Clinical Ground Truth", fontsize=12, fontweight="bold")

# 2. Normalized Recall % Heatmap
cm_norm = (cm_3x3.astype(float) / cm_3x3.sum(axis=1, keepdims=True)) * 100

sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".1f",
    cmap="Greens",
    cbar=True,
    xticklabels=class_labels,
    yticklabels=class_labels,
    ax=axes[1],
    annot_kws={"size": 14, "weight": "bold"}
)
# Add % sign to annotations in axes[1]
for t in axes[1].texts:
    t.set_text(t.get_text() + " %")

axes[1].set_title("CVC-MAGNet-V2 Apex (E99-B): Recall / Sensitivity (%)\n(Row-Normalized: Percentage of True Class Correctly Identified)", fontsize=13, fontweight="bold", pad=12)
axes[1].set_xlabel("Predicted Clinical Category", fontsize=12, fontweight="bold")
axes[1].set_ylabel("True Clinical Ground Truth", fontsize=12, fontweight="bold")

plt.suptitle("Official Confusion Matrix — CVC-MAGNet-V2 Final Apex (0.8499 Macro AUROC)", fontsize=16, fontweight="bold", y=0.98)
plt.tight_layout()

OUT_IMG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_IMG, dpi=300, bbox_inches="tight")
fig.savefig(DL_IMG, dpi=300, bbox_inches="tight")
print(f"✅ Successfully saved publication-ready Confusion Matrix figure to:\n  - {OUT_IMG}\n  - {DL_IMG}")

