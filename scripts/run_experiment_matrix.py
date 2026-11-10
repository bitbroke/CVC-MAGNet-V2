"""Master 80-Experiment Tournament Runner for CVC Placement Benchmark.

Implements the 4-Stage Funnel:
- Stage 1: Zero-Training Anatomical Origin & Spatial Reference Sweeps (E01-E12) [Completed]
- Stage 2: Segmenter & Tip Detector (T3 Upgrades) (E13-E28)
- Stage 3: Multi-CVC Instance Disentanglement & Noisy-OR Grouping (E29-E40)
- Stage 4: Spatial Input Channel Engineering (CoordConv / Multi-Zone) (E41-E56)
- Stage 5: Loss Functions & Semi-Supervised Pseudo-Mask Expansion (E57-E68)
- Stage 6: Rich Tabular Geometry (60-D) & Stepwise Meta-Ensembling (E69-E80)

Logs all runs to reports/master_ablation_ledger.parquet/.csv and saves
prediction matrices to runs/val_preds/val_preds_{exp_id}.npy.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORTS_DIR = PROJECT_ROOT / "reports"
VAL_PREDS_DIR = PROJECT_ROOT / "runs" / "val_preds"
LEDGER_CSV = REPORTS_DIR / "master_ablation_ledger.csv"
LEDGER_PARQUET = REPORTS_DIR / "master_ablation_ledger.parquet"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
VAL_PREDS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["CVC - Normal", "CVC - Borderline", "CVC - Abnormal"]


def log_experiment_result(
    exp_id: str,
    stage: int,
    runtime_s: float,
    val_preds: np.ndarray,
    y_val: np.ndarray,
    is_traced: np.ndarray,
    is_untraced: np.ndarray,
    is_multicvc: np.ndarray,
    tip_recall_35px: float = 0.0,
) -> Dict:
    """Save prediction matrix and append/update experiment record in master ledger."""
    # 1. Save out-of-sample predictions
    pred_path = VAL_PREDS_DIR / f"val_preds_{exp_id}.npy"
    np.save(str(pred_path), val_preds.astype(np.float32))

    # 2. Compute metrics
    def calc_auc(y_t, y_p):
        aucs = []
        for c in range(3):
            if len(np.unique(y_t[:, c])) > 1:
                aucs.append(roc_auc_score(y_t[:, c], y_p[:, c]))
        return round(float(np.mean(aucs)), 4) if len(aucs) > 0 else 0.0

    all_auc = calc_auc(y_val, val_preds)
    norm_auc = round(float(roc_auc_score(y_val[:, 0], val_preds[:, 0])), 4)
    bord_auc = round(float(roc_auc_score(y_val[:, 1], val_preds[:, 1])), 4)
    abno_auc = round(float(roc_auc_score(y_val[:, 2], val_preds[:, 2])), 4)

    traced_auc = calc_auc(y_val[is_traced], val_preds[is_traced])
    untraced_auc = calc_auc(y_val[is_untraced], val_preds[is_untraced])
    multicvc_auc = calc_auc(y_val[is_multicvc], val_preds[is_multicvc])

    record = {
        "exp_id": exp_id,
        "stage": stage,
        "Runtime_s": round(runtime_s, 1),
        "Val_Macro_AUROC_2558": all_auc,
        "Normal_AUROC": norm_auc,
        "Borderline_AUROC": bord_auc,
        "Abnormal_AUROC": abno_auc,
        "Traced_778_AUROC": traced_auc,
        "Untraced_1780_AUROC": untraced_auc,
        "MultiCVC_332_AUROC": multicvc_auc,
        "Tip_Recall_35px": round(tip_recall_35px, 1),
    }

    # 3. Update persistent ledger
    if LEDGER_CSV.exists():
        df = pd.read_csv(LEDGER_CSV)
        # Update or append
        if exp_id in df["exp_id"].values:
            idx = df[df["exp_id"] == exp_id].index[0]
            for k, v in record.items():
                df.at[idx, k] = v
        else:
            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])

    df.to_csv(LEDGER_CSV, index=False)
    df.to_parquet(LEDGER_PARQUET, index=False)

    print(f"[{exp_id}] All: {all_auc:.4f} (N:{norm_auc:.4f}, B:{bord_auc:.4f}, A:{abno_auc:.4f}) | Traced: {traced_auc:.4f} | Untraced: {untraced_auc:.4f} | MultiCVC: {multicvc_auc:.4f}")
    return record


def main():
    parser = argparse.ArgumentParser(description="Master Tournament Runner")
    parser.add_argument("--stage", type=str, default="status", choices=["status", "stage1", "stage3", "all"])
    args = parser.parse_args()

    if args.stage == "status":
        if LEDGER_CSV.exists():
            df = pd.read_csv(LEDGER_CSV)
            print("=" * 80)
            print("CURRENT MASTER ABLATION LEDGER STATUS")
            print("=" * 80)
            print(df[["exp_id", "stage", "Val_Macro_AUROC_2558", "Normal_AUROC", "Borderline_AUROC", "Abnormal_AUROC", "Traced_778_AUROC", "Untraced_1780_AUROC", "MultiCVC_332_AUROC"]].to_string(index=False))
            print(f"\nTotal completed experiments: {len(df)}")
            print(f"Ledger file: {LEDGER_CSV}")
        else:
            print("No master ledger found yet.")


if __name__ == "__main__":
    main()

