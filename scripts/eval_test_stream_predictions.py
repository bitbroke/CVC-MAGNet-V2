"""Inference of New MGAP & Dynamic Corridor Zoom Networks on Sealed Test Split (N=2,284).

Evaluates:
- E89: runs/E89_mgap_densenet121_512/checkpoints/best.pt
- E90: runs/E90_mgap_seresnext50_512/checkpoints/best.pt
- E93: runs/E93_dynamic_corridor_zoom_densenet121/checkpoints/best.pt
- E94: runs/E94_dynamic_corridor_zoom_seresnext50/checkpoints/best.pt

Saves predictions to runs/test_preds/test_preds_{exp_id}.npy.
"""

import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.mask_guided_spatial_net import MaskGuidedSpatialNet
from scripts.train_mgap_spatial_net import MGAPDataset
from scripts.train_corridor_zoom_net import DynamicCorridorZoomDataset

DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "splits" / "placement_locked_manifest.parquet"
CARINA_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "carina_validation_cache.parquet"
ZONAL_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "zonal_val_features.parquet"
CONSENSUS_TEST_DIR = DATA_DIR / "curated_cvc_v1" / "consensus_seg_test_cache_512"
CONSENSUS_DIR = DATA_DIR / "curated_cvc_v1" / "consensus_seg_cache_512"
TEST_PREDS_DIR = PROJECT_ROOT / "runs" / "test_preds"
TEST_PREDS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Evaluating new models on test set (N=2,284)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = pd.read_parquet(MANIFEST_PATH)
    test_manifest = manifest[manifest["split"] == "test"].reset_index(drop=True)
    print(f"Test cohort count: {len(test_manifest)}")

    # Carina & Zonal lookups
    df_carina_val = pd.read_parquet(CARINA_VAL_PATH)
    df_zonal_val = pd.read_parquet(ZONAL_VAL_PATH)

    c_sid = "StudyInstanceUID" if "StudyInstanceUID" in df_carina_val.columns else "study_id"
    carina_dict = df_carina_val.set_index(c_sid).to_dict(orient="index")
    z_sid = "StudyInstanceUID" if "StudyInstanceUID" in df_zonal_val.columns else "study_id"
    zonal_dict = df_zonal_val.set_index(z_sid).to_dict(orient="index")

    # Use consensus_seg_test_cache_512 if available, else consensus_seg_cache_512
    seg_dir = CONSENSUS_TEST_DIR if CONSENSUS_TEST_DIR.exists() and len(list(CONSENSUS_TEST_DIR.glob("*.png"))) > 100 else CONSENSUS_DIR

    # 1. Full-Thorax MGAP Test Dataset
    test_ds_full = MGAPDataset(
        df=test_manifest,
        carina_lookup=carina_dict,
        zonal_lookup=zonal_dict,
        seg_cache_dir=seg_dir,
        img_size=512,
        is_train=False,
    )
    test_loader_full = DataLoader(test_ds_full, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    # 2. Dynamic Corridor Zoom Test Dataset
    test_ds_zoom = DynamicCorridorZoomDataset(
        df=test_manifest,
        carina_lookup=carina_dict,
        zonal_lookup=zonal_dict,
        seg_cache_dir=seg_dir,
        is_train=False,
    )
    test_loader_zoom = DataLoader(test_ds_zoom, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    models_to_run = [
        ("E89_mgap_dense", "densenet121", PROJECT_ROOT / "runs" / "E89_mgap_densenet121_512" / "checkpoints" / "best.pt", test_loader_full),
        ("E90_mgap_srx", "seresnext50_32x4d", PROJECT_ROOT / "runs" / "E90_mgap_seresnext50_512" / "checkpoints" / "best.pt", test_loader_full),
        ("E93_zoom_dense", "densenet121", PROJECT_ROOT / "runs" / "E93_dynamic_corridor_zoom_densenet121" / "checkpoints" / "best.pt", test_loader_zoom),
        ("E94_zoom_srx", "seresnext50_32x4d", PROJECT_ROOT / "runs" / "E94_dynamic_corridor_zoom_seresnext50" / "checkpoints" / "best.pt", test_loader_zoom),
    ]

    for tag, backbone, ckpt_p, loader in models_to_run:
        out_p = TEST_PREDS_DIR / f"test_preds_{tag}.npy"
        if out_p.exists():
            print(f"[{tag}] already exists at {out_p}, skipping.")
            continue

        if not ckpt_p.exists():
            print(f"[{tag}] checkpoint {ckpt_p} not found, skipping.")
            continue

        print(f"Running inference for {tag} ({backbone})...", flush=True)
        m = MaskGuidedSpatialNet(backbone_name=backbone, pretrained=False, num_classes=3, num_aux=8).to(device)
        st = torch.load(str(ckpt_p), map_location=device, weights_only=False)
        m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in st.get("model_state_dict", st).items()})
        m.eval()

        preds = []
        with torch.no_grad():
            for batch in loader:
                imgs = batch["image"].to(device, non_blocking=True)
                with autocast("cuda", dtype=torch.float16):
                    lc, _ = m(imgs)
                    p = torch.sigmoid(lc)
                preds.append(p.cpu().float().numpy())

        p_arr = np.concatenate(preds, axis=0)
        np.save(out_p, p_arr)
        print(f"Saved {tag} test predictions -> {out_p} (shape: {p_arr.shape})", flush=True)


if __name__ == "__main__":
    main()
