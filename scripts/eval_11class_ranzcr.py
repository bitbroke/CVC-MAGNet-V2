"""Evaluates the full 11-Class RANZCR-CLiP Macro AUROC using E89 MGAP-DenseNet-121."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

ROOT_DIR = Path(r"C:\Ankshit\CVC-MAGNet-V2")
sys.path.insert(0, str(ROOT_DIR))

from src.models.mask_guided_spatial_net import MaskGuidedSpatialNet
from scripts.train_mgap_spatial_net import (
    MGAPDataset,
    MANIFEST_PATH,
    CARINA_VAL_PATH,
    ZONAL_VAL_PATH,
    CONSENSUS_SEG_DIR,
    IMAGES_DIR,
    AUX_CLASSES,
)
from src.data.full_spatial_dataset import CLASSES as CVC_CLASSES

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running 11-Class RANZCR evaluation on device: {device}")
    
    # Load manifest
    manifest_df = pd.read_parquet(MANIFEST_PATH)
    val_df = manifest_df[manifest_df["split"] == "val"].reset_index(drop=True)
    
    # Load lookups
    carina_val_df = pd.read_parquet(CARINA_VAL_PATH)
    c_sid_col = "StudyInstanceUID" if "StudyInstanceUID" in carina_val_df.columns else "study_id"
    carina_lookup = carina_val_df.set_index(c_sid_col).to_dict(orient="index")

    zonal_val_df = pd.read_parquet(ZONAL_VAL_PATH)
    z_sid_col = "StudyInstanceUID" if "StudyInstanceUID" in zonal_val_df.columns else "study_id"
    zonal_lookup = zonal_val_df.set_index(z_sid_col).to_dict(orient="index")
    
    val_dataset = MGAPDataset(
        df=val_df,
        carina_lookup=carina_lookup,
        zonal_lookup=zonal_lookup,
        images_dir=IMAGES_DIR,
        seg_cache_dir=CONSENSUS_SEG_DIR,
        img_size=512,
        is_train=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    # Initialize Model
    model = MaskGuidedSpatialNet(
        backbone_name="densenet121",
        num_classes=3,
        num_aux=8,
        pretrained=False,
    )
    
    ckpt_path = ROOT_DIR / "runs" / "E89_mgap_densenet121_512" / "checkpoints" / "best.pt"
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    state_dict_f32 = {k: v.float() if v.is_floating_point() else v for k, v in state_dict.items()}
    model.load_state_dict(state_dict_f32)
    model.to(device)
    model.eval()
    
    all_cvc_preds = []
    all_aux_preds = []
    
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device, non_blocking=True)
            with autocast("cuda", dtype=torch.float16):
                logits_cvc, logits_aux = model(imgs)
                p_cvc = torch.sigmoid(logits_cvc)
                p_aux = torch.sigmoid(logits_aux)
            all_cvc_preds.append(p_cvc.cpu().float().numpy())
            all_aux_preds.append(p_aux.cpu().float().numpy())
            
    cvc_preds = np.concatenate(all_cvc_preds, axis=0)
    aux_preds = np.concatenate(all_aux_preds, axis=0)
    
    y_cvc_true = val_df[CVC_CLASSES].values.astype(int)
    y_aux_true = val_df[AUX_CLASSES].values.astype(int)
    
    all_class_names = CVC_CLASSES + AUX_CLASSES
    all_preds = np.concatenate([cvc_preds, aux_preds], axis=1)
    all_trues = np.concatenate([y_cvc_true, y_aux_true], axis=1)
    
    class_aucs = {}
    print("\n" + "=" * 80)
    print("INDIVIDUAL CLASS AUROCS ON VALIDATION SET (N=2,558):")
    print("=" * 80)
    for c_idx, name in enumerate(all_class_names):
        try:
            auc = roc_auc_score(all_trues[:, c_idx], all_preds[:, c_idx])
            class_aucs[name] = auc
            tag = "[CVC CORE]" if name in CVC_CLASSES else "[AUX TUBE]"
            print(f"  {tag:<11} {name:<30} : {auc:.4f}")
        except Exception as e:
            print(f"  {name:<30} : Error {e}")
            
    cvc_macro = np.mean([class_aucs[c] for c in CVC_CLASSES])
    aux_macro = np.mean([class_aucs[c] for c in AUX_CLASSES])
    total_11_macro = np.mean(list(class_aucs.values()))
    
    print("-" * 80)
    print(f"Strict 3-Class CVC Macro AUROC       : {cvc_macro:.4f}")
    print(f"Auxiliary 8-Class Non-CVC Macro AUROC: {aux_macro:.4f}")
    print(f"OVERALL 11-CLASS RANZCR MACRO AUROC  : {total_11_macro:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
