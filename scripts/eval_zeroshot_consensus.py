"""Evaluate Zero-Shot Downstream CNNs on Dual-Segmenter Consensus Masks & T3 Tips.

Tests whether E56 (DenseNet-121), E62 (SEResNeXt-50), and E61b (EffNet-B0)
benefit from the cleaner consensus ribbons right out of the box without any retraining.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.full_spatial_dataset import CLASSES, generate_anatomical_channel
from src.models.full_spatial_net import FullSpatialNet

DATA_DIR = PROJECT_ROOT / "data"
CONSENSUS_DIR = DATA_DIR / "curated_cvc_v1" / "consensus_seg_cache_512"
CARINA_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "carina_validation_cache.parquet"
T3_TIPS_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "t3_consensus_val_tips.parquet"
MANIFEST_PATH = DATA_DIR / "splits" / "placement_locked_manifest.parquet"
IMAGES_DIR = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")
VAL_PREDS_DIR = PROJECT_ROOT / "runs" / "val_preds"


class ConsensusSpatialDataset(Dataset):
    def __init__(self, df, carina_lookup, tip_lookup, ch2_mode="hybrid_ch2"):
        self.df = df.reset_index(drop=True)
        self.carina_lookup = carina_lookup
        self.tip_lookup = tip_lookup
        self.ch2_mode = ch2_mode
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = row["study_id"]
        
        img_path = IMAGES_DIR / f"{sid}.jpg"
        img = cv2.imread(str(img_path), cv2.IMREAD_REDUCED_GRAYSCALE_4)
        if img is None:
            ch0 = np.zeros((512, 512), dtype=np.float32)
        else:
            ch0 = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

        mask_p = CONSENSUS_DIR / f"{sid}.png"
        if mask_p.exists():
            mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
            ch1 = mask.astype(np.float32) / 255.0 if mask is not None else np.zeros((512, 512), dtype=np.float32)
        else:
            ch1 = np.zeros((512, 512), dtype=np.float32)

        c_info = self.carina_lookup.get(sid, {})
        cx = float(c_info.get("carina_x", 256.0))
        cy = float(c_info.get("carina_y", 230.0))

        t_info = self.tip_lookup.get(sid, {})
        xtip = float(t_info.get("xtip_512", cx))
        ytip = float(t_info.get("ytip_512", cy + 40.0))
        conf_tip = float(t_info.get("conf_tip", 0.0))

        ch2 = generate_anatomical_channel(cx, cy, xtip=xtip, ytip=ytip, conf_tip=conf_tip, mode=self.ch2_mode)

        tensor_img = torch.from_numpy(np.stack([ch0, ch1, ch2], axis=0)).float()
        tensor_img = (tensor_img - self.mean) / self.std
        labels = torch.tensor([float(row[c]) for c in CLASSES], dtype=torch.float32)
        return {"image": tensor_img, "label": labels}


def eval_model(model, loader, device, y_val, is_traced, is_untraced, is_multicvc, label):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(imgs)
                probs = torch.sigmoid(logits)
            preds.append(probs.cpu().float().numpy())
    p = np.concatenate(preds, axis=0)

    aucs = [roc_auc_score(y_val[:, c], p[:, c]) for c in range(3)]
    macro = float(np.mean(aucs))
    tr = np.mean([roc_auc_score(y_val[is_traced, c], p[is_traced, c]) for c in range(3)])
    un = np.mean([roc_auc_score(y_val[is_untraced, c], p[is_untraced, c]) for c in range(3)])
    mu = np.mean([roc_auc_score(y_val[is_multicvc, c], p[is_multicvc, c]) for c in range(3)])
    print(f"[{label:<35}] Macro: {macro:.4f} | N: {aucs[0]:.4f} | B: {aucs[1]:.4f} | A: {aucs[2]:.4f} | Tr: {tr:.4f} | Un: {un:.4f} | Mu: {mu:.4f}")
    return p, macro


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("ZERO-SHOT EVALUATION OF CONVERGED CNNS ON CONSENSUS MASKS & T3 TIPS")
    print("=" * 80)

    manifest = pd.read_parquet(MANIFEST_PATH)
    val_m = manifest[manifest["split"] == "val"].reset_index(drop=True)
    val_df = val_m.rename(columns={"StudyInstanceUID": "study_id"})
    y_val = val_df[CLASSES].values.astype(int)

    is_traced = (val_m["annotated_cvc_count"] > 0).values
    is_untraced = ~is_traced
    is_multicvc = (val_m["annotated_cvc_count"] > 1) | (y_val.sum(axis=1) > 1)

    df_carina_val = pd.read_parquet(CARINA_VAL_PATH)
    carina_lookup = {}
    for _, r in df_carina_val.iterrows():
        cx = float(r.get("carina_x_640", 307.5)) * (512.0 / 640.0)
        cy = float(r.get("carina_y_640", 256.0)) * (512.0 / 640.0)
        carina_lookup[r["study_id"]] = {"carina_x": cx, "carina_y": cy}

    df_t3_tips = pd.read_parquet(T3_TIPS_VAL_PATH).set_index("study_id")
    tip_lookup = df_t3_tips.to_dict(orient="index")

    # 1. Evaluate DenseNet-121 (hybrid_ch2)
    val_ds_hybrid = ConsensusSpatialDataset(val_df, carina_lookup, tip_lookup, ch2_mode="hybrid_ch2")
    loader_hybrid = DataLoader(val_ds_hybrid, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    dense_ckpt_path = PROJECT_ROOT / "runs" / "E56_densenet121_semi_supervised" / "checkpoints" / "best.pt"
    if dense_ckpt_path.exists():
        model_dense = FullSpatialNet(backbone_name="densenet121", pretrained=False, num_classes=3, dropout_rate=0.25).to(device)
        st = torch.load(dense_ckpt_path, map_location=device, weights_only=False)
        model_dense.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in st.get("model_state_dict", st).items()})
        p_dense, m_dense = eval_model(model_dense, loader_hybrid, device, y_val, is_traced, is_untraced, is_multicvc, "E56_DenseNet121_ZeroShot_T3")
        np.save(VAL_PREDS_DIR / "val_preds_E83_densenet121_zeroshot_consensus.npy", p_dense)

    # 2. Evaluate SEResNeXt-50 (hybrid_ch2)
    srx_ckpt_path = PROJECT_ROOT / "runs" / "E62_seresnext50_semi_supervised" / "checkpoints" / "best.pt"
    if srx_ckpt_path.exists():
        model_srx = FullSpatialNet(backbone_name="seresnext50_32x4d", pretrained=False, num_classes=3, dropout_rate=0.25).to(device)
        st = torch.load(srx_ckpt_path, map_location=device, weights_only=False)
        model_srx.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in st.get("model_state_dict", st).items()})
        p_srx, m_srx = eval_model(model_srx, loader_hybrid, device, y_val, is_traced, is_untraced, is_multicvc, "E62_SEResNeXt50_ZeroShot_T3")
        np.save(VAL_PREDS_DIR / "val_preds_E84_seresnext50_zeroshot_consensus.npy", p_srx)

    # 3. Evaluate EffNet-B0 (signed_carina_coord)
    val_ds_coord = ConsensusSpatialDataset(val_df, carina_lookup, tip_lookup, ch2_mode="signed_carina_coord")
    loader_coord = DataLoader(val_ds_coord, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    eff_ckpt_path = PROJECT_ROOT / "runs" / "E61b_effnet_b0_native_semi" / "checkpoints" / "best.pt"
    if eff_ckpt_path.exists():
        model_eff = FullSpatialNet(backbone_name="tf_efficientnet_b0.ns_jft_in1k", pretrained=False, num_classes=3, dropout_rate=0.25).to(device)
        st = torch.load(eff_ckpt_path, map_location=device, weights_only=False)
        model_eff.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in st.get("model_state_dict", st).items()})
        p_eff, m_eff = eval_model(model_eff, loader_coord, device, y_val, is_traced, is_untraced, is_multicvc, "E61b_EffNetB0_ZeroShot_T3")
        np.save(VAL_PREDS_DIR / "val_preds_E61b_effnet_b0_zeroshot_consensus.npy", p_eff)


if __name__ == "__main__":
    main()

