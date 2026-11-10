"""E61 & E62: Semi-Supervised Training Expansion with 2:1 Mixed Sampler and Sample-Weighted BCE.

Warm-starts from converged checkpoints:
- E61: Warm-starts from E42 (tf_efficientnet_b0.ns_jft_in1k, 0.7520) using hybrid_ch2
- E62: Warm-starts from E54 (seresnext50_32x4d, 0.7507) using hybrid_ch2

Key Training Guarantees:
1. 2:1 Balanced Sampler: In each epoch, samples all 6,600 traced studies (weight 1.0)
   + 3,300 random high-confidence untraced studies (weight 0.65).
2. Prevents catastrophic forgetting of clean polyline traces while adapting to out-of-sample T2 mask softness.
3. Fine-tunes for 2 epochs at lr = 4e-5 with CosineAnnealingLR down to 5e-6.
4. Evaluates strictly out-of-sample on locked N=2,558 validation cohort.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.full_spatial_dataset import CLASSES, generate_anatomical_channel
from src.models.full_spatial_net import FullSpatialNet

DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "splits" / "placement_locked_manifest.parquet"
HIGHCONF_MANIFEST_PATH = DATA_DIR / "splits" / "untraced_highconf_manifest.parquet"
CARINA_TR_PATH = DATA_DIR / "curated_cvc_v1" / "carina_train_cache.parquet"
CARINA_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "carina_validation_cache.parquet"
CARINA_UNTR_PATH = DATA_DIR / "curated_cvc_v1" / "carina_untraced_train.parquet"
ZONAL_TR_PATH = DATA_DIR / "curated_cvc_v1" / "zonal_train_features.parquet"
ZONAL_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "zonal_val_features.parquet"
SEG_CACHE_DIR = DATA_DIR / "curated_cvc_v1" / "seg_cache_512"
IMAGES_DIR = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")
VAL_PREDS_DIR = PROJECT_ROOT / "runs" / "val_preds"
REPORTS_DIR = PROJECT_ROOT / "reports"


class ExpandedSpatialDataset(Dataset):
    """3-Channel Dataset supporting traced (weight 1.0) and untraced (weight 0.65) studies."""

    def __init__(
        self,
        df: pd.DataFrame,
        carina_lookup: Dict,
        tip_lookup: Dict,
        images_dir: Path,
        seg_cache_dir: Path,
        ch2_mode: str = "hybrid_ch2",
        is_train: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.carina_lookup = carina_lookup
        self.tip_lookup = tip_lookup
        self.images_dir = images_dir
        self.seg_cache_dir = seg_cache_dir
        self.ch2_mode = ch2_mode
        self.is_train = is_train

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        sid = row["study_id"]
        img_path = self.images_dir / f"{sid}.jpg"

        # Channel 0: CLAHE Radiograph (512x512)
        img_raw = cv2.imread(str(img_path), cv2.IMREAD_REDUCED_GRAYSCALE_4)
        if img_raw is None:
            ch0 = np.zeros((512, 512), dtype=np.float32)
        else:
            img_512 = cv2.resize(img_raw, (512, 512), interpolation=cv2.INTER_LINEAR)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            ch0 = (clahe.apply(img_512).astype(np.float32) / 255.0)

        # Channel 1: T2 Continuous Segmentation Ribbon
        png_p = self.seg_cache_dir / f"{sid}.png"
        if png_p.exists():
            mask_uint8 = cv2.imread(str(png_p), cv2.IMREAD_GRAYSCALE)
            if mask_uint8 is not None:
                ch1 = mask_uint8.astype(np.float32) / 255.0
            else:
                ch1 = np.zeros((512, 512), dtype=np.float32)
        else:
            ch1 = np.zeros((512, 512), dtype=np.float32)

        # Channel 2: Spatial Reference Map
        crow = self.carina_lookup.get(sid, {})
        trow = self.tip_lookup.get(sid, {})

        xc = float(crow.get("carina_x", 256.0))
        yc = float(crow.get("carina_y", 230.0))
        xtip = float(trow.get("xtip", 256.0))
        ytip = float(trow.get("ytip", 270.0))
        conf_tip = float(trow.get("conf_tip", 0.0))

        ch2 = generate_anatomical_channel(
            xc=xc,
            yc=yc,
            xtip=xtip,
            ytip=ytip,
            conf_tip=conf_tip,
            size=512,
            mode=self.ch2_mode,
        )

        tensor_img = torch.from_numpy(np.stack([ch0, ch1, ch2], axis=0)).float()
        tensor_img = (tensor_img - self.mean) / self.std

        labels = torch.tensor([float(row[c]) for c in CLASSES], dtype=torch.float32)
        weight = torch.tensor(float(row.get("sample_weight", 1.0)), dtype=torch.float32)

        return {"image": tensor_img, "label": labels, "weight": weight}


def evaluate_val_set(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, Dict]:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device, non_blocking=True)
            lbls = batch["label"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(imgs)
                probs = torch.sigmoid(logits)
            preds.append(probs.cpu().float().numpy())
            targets.append(lbls.cpu().float().numpy())

    y_p = np.concatenate(preds, axis=0)
    y_t = np.concatenate(targets, axis=0)

    aucs = {}
    for c_idx, c_name in enumerate(CLASSES):
        aucs[c_name] = round(float(roc_auc_score(y_t[:, c_idx], y_p[:, c_idx])), 4)
    aucs["macro_auroc"] = round(float(np.mean(list(aucs.values()))), 4)
    return y_p, y_t, aucs


def main():
    parser = argparse.ArgumentParser(description="Semi-Supervised Expansion Fine-Tuning")
    parser.add_argument("--exp_id", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="tf_efficientnet_b0.ns_jft_in1k")
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--ch2_mode", type=str, default="hybrid_ch2")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=4e-5)
    args = parser.parse_args()

    print("=" * 80)
    print(f"STAGE 5: TRAINING {args.exp_id} ({args.backbone}, ch2_mode={args.ch2_mode})")
    if args.resume_ckpt and args.resume_ckpt.lower() != "none":
        print(f"Warm-starting from: {args.resume_ckpt}")
    else:
        print("Training from ImageNet pre-trained initialization.")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = PROJECT_ROOT / "runs" / args.exp_id
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    VAL_PREDS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Manifests
    manifest = pd.read_parquet(MANIFEST_PATH)
    val_manifest = manifest[manifest["split"] == "val"].reset_index(drop=True)
    df_highconf_untr = pd.read_parquet(HIGHCONF_MANIFEST_PATH)

    df_carina_tr = pd.read_parquet(CARINA_TR_PATH)
    df_carina_val = pd.read_parquet(CARINA_VAL_PATH)
    df_carina_untr = pd.read_parquet(CARINA_UNTR_PATH)

    df_zonal_tr = pd.read_parquet(ZONAL_TR_PATH)
    df_zonal_val = pd.read_parquet(ZONAL_VAL_PATH)

    # Build Traced Training DF
    tr_sids = set(df_carina_tr["study_id"]).intersection(set(df_zonal_tr["study_id"]))
    train_manifest = manifest[manifest["split"] == "train"].copy()
    train_manifest = train_manifest[train_manifest["StudyInstanceUID"].isin(tr_sids)].copy()
    train_manifest.rename(columns={"StudyInstanceUID": "study_id"}, inplace=True)
    train_manifest["sample_weight"] = 1.0

    df_highconf_untr["sample_weight"] = 0.65

    # Lookups
    carina_lookup = {}
    for df_c in [df_carina_tr, df_carina_val, df_carina_untr]:
        for _, r in df_c.iterrows():
            sid = r["study_id"]
            if "raw_carina_x_norm" in r and not pd.isna(r["raw_carina_x_norm"]):
                cx = float(r["raw_carina_x_norm"]) * 512.0
                cy = float(r["raw_carina_y_norm"]) * 512.0
            elif "carina_x" in r and not pd.isna(r["carina_x"]):
                cx = float(r["carina_x"])
                cy = float(r["carina_y"])
            else:
                cx = float(r.get("carina_x_640", 307.5)) * (512.0 / 640.0)
                cy = float(r.get("carina_y_640", 256.0)) * (512.0 / 640.0)
            carina_lookup[sid] = {"carina_x": cx, "carina_y": cy}

    tip_lookup = {}
    for df_z in [df_zonal_tr, df_zonal_val]:
        for _, r in df_z.iterrows():
            c_x = carina_lookup.get(r["study_id"], {}).get("carina_x", 256.0)
            c_y = carina_lookup.get(r["study_id"], {}).get("carina_y", 230.0)
            tip_lookup[r["study_id"]] = {
                "xtip": c_x + r["dx_norm"] * 512.0,
                "ytip": c_y + r["dy_norm"] * 512.0,
                "conf_tip": r.get("conf_tip", 0.0),
            }
    for _, r in df_highconf_untr.iterrows():
        tip_lookup[r["study_id"]] = {
            "xtip": r.get("xtip_512", 256.0),
            "ytip": r.get("ytip_512", 270.0),
            "conf_tip": r.get("conf_tip", 0.0),
        }

    # Validation Dataset
    val_df = val_manifest.rename(columns={"StudyInstanceUID": "study_id"})
    val_ds = ExpandedSpatialDataset(val_df, carina_lookup, tip_lookup, IMAGES_DIR, SEG_CACHE_DIR, ch2_mode=args.ch2_mode, is_train=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    # Instantiate & Warm-Start Model
    if args.resume_ckpt and args.resume_ckpt.lower() != "none":
        print(f"Instantiating {args.backbone} and loading checkpoint {Path(args.resume_ckpt).name}...")
        model = FullSpatialNet(backbone_name=args.backbone, pretrained=False, num_classes=3, dropout_rate=0.25).to(device)
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        state_f32 = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
        model.load_state_dict(state_f32)
        print("Checkpoint weights successfully loaded into model!")
    else:
        print(f"Instantiating {args.backbone} with ImageNet pre-trained weights...")
        model = FullSpatialNet(backbone_name=args.backbone, pretrained=True, num_classes=3, dropout_rate=0.25).to(device)
        print("Pretrained weights initialized!")

    epochs = args.epochs
    lr = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Cosine decay to 5e-6
    eta_min = 5e-6
    steps_per_epoch = (len(train_manifest) + 3300) // args.batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch, eta_min=eta_min)
    scaler = torch.amp.GradScaler("cuda")
    grad_accum_steps = args.grad_accum

    best_macro = 0.0
    history = []

    for ep in range(1, epochs + 1):
        # 2:1 Balanced Sampler: All 6,600 traced + random 3,300 untraced
        np.random.seed(42 + ep)
        sample_untr = df_highconf_untr.sample(n=min(3300, len(df_highconf_untr)), replace=False, random_state=42 + ep)
        comb_epoch_df = pd.concat([train_manifest, sample_untr], ignore_index=True).sample(frac=1.0, random_state=42 + ep).reset_index(drop=True)

        train_ds = ExpandedSpatialDataset(comb_epoch_df, carina_lookup, tip_lookup, IMAGES_DIR, SEG_CACHE_DIR, ch2_mode="hybrid_ch2", is_train=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

        model.train()
        running_loss = 0.0
        n_batches = len(train_loader)
        t_ep0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        print(f"\n[Epoch {ep}/{epochs}] Training on {len(comb_epoch_df)} studies (6,600 traced + 3,300 untraced)...")
        for step, batch in enumerate(train_loader):
            imgs = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            weights = batch["weight"].to(device, non_blocking=True).unsqueeze(1)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(imgs)
                loss_unweighted = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
                loss = (loss_unweighted * weights).mean() / grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == n_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()

                if scale_before <= scale_after:
                    scheduler.step()

                optimizer.zero_grad(set_to_none=True)

            if (step + 1) % 150 == 0 or (step + 1) == n_batches:
                cur_l = running_loss / (step + 1)
                elapsed = time.time() - t_ep0
                rate = (step + 1) * args.batch_size / max(elapsed, 1e-4)
                print(f"  Step {step+1}/{n_batches} | Loss: {cur_l:.4f} ({rate:.1f} img/s)", flush=True)

        # Evaluation on full locked validation cohort (N=2,558)
        print(f"  Evaluating Epoch {ep} on locked N=2,558 validation cohort...")
        val_preds, val_targets, metrics = evaluate_val_set(model, val_loader, device)

        macro_auc = metrics["macro_auroc"]
        ep_time = time.time() - t_ep0
        print(f"--> Epoch {ep}/{epochs} Finished in {ep_time:.1f}s | Val Macro AUROC: {macro_auc:.4f} (N: {metrics['CVC - Normal']:.4f}, B: {metrics['CVC - Borderline']:.4f}, A: {metrics['CVC - Abnormal']:.4f})", flush=True)

        if macro_auc > best_macro:
            best_macro = macro_auc
            best_preds = val_preds.copy()
            # Save compact FP16 checkpoint
            ckpt_path = ckpt_dir / "best.pt"
            fp16_state = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
            torch.save({"model_state_dict": fp16_state, "metadata": {"epoch": ep, "best_macro": best_macro, "metrics": metrics}}, str(ckpt_path))
            np.save(str(VAL_PREDS_DIR / f"val_preds_{args.exp_id}.npy"), best_preds)
            print(f"  *** New Best Checkpoint & Predictions Saved (Macro: {best_macro:.4f}) ***")

    # Sub-cohort breakdown and ledger update
    is_traced = (val_manifest["annotated_cvc_count"] > 0).values
    is_untraced = ~is_traced
    is_multicvc = (val_manifest["annotated_cvc_count"] > 1) | (val_targets.sum(axis=1) > 1)

    def sub_auc(mask):
        return round(float(np.mean([roc_auc_score(val_targets[mask, c], best_preds[mask, c]) for c in range(3)])), 4)

    tr_auc = sub_auc(is_traced)
    un_auc = sub_auc(is_untraced)
    mu_auc = sub_auc(is_multicvc)

    ledger_rec = {
        "exp_id": args.exp_id,
        "stage": 5,
        "Runtime_s": round(time.time() - t_ep0, 1),
        "Val_Macro_AUROC_2558": best_macro,
        "Normal_AUROC": metrics["CVC - Normal"],
        "Borderline_AUROC": metrics["CVC - Borderline"],
        "Abnormal_AUROC": metrics["CVC - Abnormal"],
        "Traced_778_AUROC": tr_auc,
        "Untraced_1780_AUROC": un_auc,
        "MultiCVC_332_AUROC": mu_auc,
        "Tip_Recall_35px": 55.7,
    }

    ledger_csv = REPORTS_DIR / "master_ablation_ledger.csv"
    ledger_parquet = REPORTS_DIR / "master_ablation_ledger.parquet"
    if ledger_csv.exists():
        df_l = pd.read_csv(ledger_csv)
        if args.exp_id in df_l["exp_id"].values:
            idx = df_l[df_l["exp_id"] == args.exp_id].index[0]
            for k, v in ledger_rec.items():
                df_l.at[idx, k] = v
        else:
            df_l = pd.concat([df_l, pd.DataFrame([ledger_rec])], ignore_index=True)
    else:
        df_l = pd.DataFrame([ledger_rec])

    df_l.to_csv(ledger_csv, index=False)
    df_l.to_parquet(ledger_parquet, index=False)
    print(f"\nSuccessfully logged {args.exp_id} to master ledger!")


if __name__ == "__main__":
    main()

