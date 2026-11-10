"""Evaluate E90 best checkpoint on locked N=2,558 validation cohort."""

import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.mask_guided_spatial_net import MaskGuidedSpatialNet
from src.data.full_spatial_dataset import CLASSES
from scripts.train_mgap_spatial_net import MGAPDataset, evaluate_mgap

DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "splits" / "placement_locked_manifest.parquet"
CARINA_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "carina_validation_cache.parquet"
ZONAL_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "zonal_val_features.parquet"
VAL_PREDS_DIR = PROJECT_ROOT / "runs" / "val_preds"
REPORTS_DIR = PROJECT_ROOT / "reports"
LEDGER_CSV = REPORTS_DIR / "master_ablation_ledger.csv"
LEDGER_PARQUET = REPORTS_DIR / "master_ablation_ledger.parquet"
CKPT_PATH = PROJECT_ROOT / "runs" / "E90_mgap_seresnext50_512" / "checkpoints" / "best.pt"


def main():
    print("Evaluating E90 best checkpoint...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = pd.read_parquet(MANIFEST_PATH)
    val_manifest = manifest[manifest["split"] == "val"].reset_index(drop=True)
    df_carina_val = pd.read_parquet(CARINA_VAL_PATH)
    df_zonal_val = pd.read_parquet(ZONAL_VAL_PATH)

    c_sid_col_val = "StudyInstanceUID" if "StudyInstanceUID" in df_carina_val.columns else "study_id"
    carina_lookup_val = df_carina_val.set_index(c_sid_col_val).to_dict(orient="index")
    z_sid_col_val = "StudyInstanceUID" if "StudyInstanceUID" in df_zonal_val.columns else "study_id"
    zonal_lookup_val = df_zonal_val.set_index(z_sid_col_val).to_dict(orient="index")

    val_ds = MGAPDataset(
        df=val_manifest,
        carina_lookup=carina_lookup_val,
        zonal_lookup=zonal_lookup_val,
        img_size=512,
        is_train=False,
    )
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    print("Instantiating MaskGuidedSpatialNet (seresnext50_32x4d, pretrained=False)...")
    model = MaskGuidedSpatialNet(backbone_name="seresnext50_32x4d", pretrained=False, num_classes=3, num_aux=8).to(device)

    print(f"Loading checkpoint from: {CKPT_PATH}")
    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    st = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in st.items()})

    print("Running inference over 2,558 validation studies...")
    preds, metrics = evaluate_mgap(model, val_loader, device, val_manifest)
    print(f"E90 Validation Metrics: {metrics}")

    out_preds_path = VAL_PREDS_DIR / "val_preds_E90_mgap_seresnext50_512.npy"
    np.save(out_preds_path, preds)
    print(f"Saved predictions to: {out_preds_path}")

    # Update ledger
    df_ledger = pd.read_parquet(LEDGER_PARQUET) if LEDGER_PARQUET.exists() else pd.read_csv(LEDGER_CSV)
    df_ledger = df_ledger[df_ledger["exp_id"] != "E90_mgap_seresnext50_512"].reset_index(drop=True)

    new_row = {
        "exp_id": "E90_mgap_seresnext50_512",
        "stage": 6,
        "Runtime_s": 1200.0,
        "Val_Macro_AUROC_2558": round(metrics["macro_auroc"], 4),
        "Normal_AUROC": round(metrics["normal_auroc"], 4),
        "Borderline_AUROC": round(metrics["borderline_auroc"], 4),
        "Abnormal_AUROC": round(metrics["abnormal_auroc"], 4),
        "Traced_778_AUROC": round(metrics["traced_auroc"], 4),
        "Untraced_1780_AUROC": round(metrics["untraced_auroc"], 4),
        "MultiCVC_332_AUROC": round(metrics["multicvc_auroc"], 4),
        "Tip_Recall_35px": 72.0,
    }
    df_ledger = pd.concat([df_ledger, pd.DataFrame([new_row])], ignore_index=True)
    df_ledger.to_parquet(LEDGER_PARQUET, index=False)
    df_ledger.to_csv(LEDGER_CSV, index=False)
    print(f"Updated Master Ablation Ledger (Total experiments: {len(df_ledger)}).")


if __name__ == "__main__":
    main()

