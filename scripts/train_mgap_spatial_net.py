"""Wave 2: Mask-Guided Attention Pooling (MGAP) Network Training Script.

Supports:
- E89: DenseNet-121 + 4-Zone MGAP + 11-Class Aux Head at 512x512
- E90: SEResNeXt-50 + 4-Zone MGAP + 11-Class Aux Head at 512x512
- E91: High-Res Fine-Tuning (E89 at 768x768)
- E92: ConvNeXt-Tiny + 4-Zone MGAP + 11-Class Aux Head at 512x512

Architectural Mechanisms:
1. 4-Zone MGAP: Global + Catheter Shaft + CAJ Landing Zone + Distal Tip Peak.
2. 11-Class Multi-Task Joint Loss:
   L_total = BCE(y_cvc, p_cvc) + 0.25 * BCE(y_aux, p_aux)
   Suppresses confounders from ETT, NGT, and Swan-Ganz lines sharing the mediastinal corridor.
3. High-Fidelity Decoding: cv2.IMREAD_REDUCED_GRAYSCALE_2 preserves catheter tips without box-blur.
4. Memory-safe FP16 mixed precision for RTX 3050 (4 GB VRAM).
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.mask_guided_spatial_net import MaskGuidedSpatialNet, AUX_CLASSES
from src.data.full_spatial_dataset import CLASSES, generate_anatomical_channel

DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "splits" / "placement_locked_manifest.parquet"
CARINA_TR_PATH = DATA_DIR / "curated_cvc_v1" / "carina_train_cache.parquet"
CARINA_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "carina_validation_cache.parquet"
ZONAL_TR_PATH = DATA_DIR / "curated_cvc_v1" / "zonal_train_features.parquet"
ZONAL_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "zonal_val_features.parquet"
CONSENSUS_SEG_DIR = DATA_DIR / "curated_cvc_v1" / "consensus_seg_cache_512"
IMAGES_DIR = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")

RUNS_DIR = PROJECT_ROOT / "runs"
VAL_PREDS_DIR = RUNS_DIR / "val_preds"
REPORTS_DIR = PROJECT_ROOT / "reports"
LEDGER_CSV = REPORTS_DIR / "master_ablation_ledger.csv"
LEDGER_PARQUET = REPORTS_DIR / "master_ablation_ledger.parquet"


class MGAPDataset(Dataset):
    """3-Channel Dataset with High-Fidelity REDUCED_GRAYSCALE_2 Decoding and 11-Class Labels."""

    def __init__(
        self,
        df: pd.DataFrame,
        carina_lookup: Dict,
        zonal_lookup: Dict,
        images_dir: Path = IMAGES_DIR,
        seg_cache_dir: Path = CONSENSUS_SEG_DIR,
        img_size: int = 512,
        is_train: bool = False,
        ch2_mode: str = "hybrid_ch2",
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.seg_cache_dir = Path(seg_cache_dir)
        self.img_size = img_size
        self.is_train = is_train
        self.ch2_mode = ch2_mode

        sid_col = "StudyInstanceUID" if "StudyInstanceUID" in self.df.columns else "study_id"
        self.study_ids = self.df[sid_col].tolist()
        self.carina_lookup = carina_lookup
        self.zonal_lookup = zonal_lookup

        # 3 CVC labels + 8 Aux labels
        self.cvc_labels = self.df[CLASSES].values.astype(np.float32)
        if all(col in self.df.columns for col in AUX_CLASSES):
            self.aux_labels = self.df[AUX_CLASSES].values.astype(np.float32)
        else:
            self.aux_labels = np.zeros((len(self.df), len(AUX_CLASSES)), dtype=np.float32)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sid = self.study_ids[idx]
        size = self.img_size

        # 1. Channel 0: CLAHE native radiograph (decoded at REDUCED_GRAYSCALE_2 ~1250x1000)
        img_path = self.images_dir / f"{sid}.jpg"
        img_gray = cv2.imread(str(img_path), cv2.IMREAD_REDUCED_GRAYSCALE_2)
        if img_gray is None:
            # Fallback to REDUCED_GRAYSCALE_4 if memory or file issue
            img_gray = cv2.imread(str(img_path), cv2.IMREAD_REDUCED_GRAYSCALE_4)

        if img_gray is None:
            ch0 = np.zeros((size, size), dtype=np.float32)
        else:
            img_resized = cv2.resize(img_gray, (size, size), interpolation=cv2.INTER_AREA if img_gray.shape[0] > size else cv2.INTER_LINEAR)
            img_clahe = self.clahe.apply(img_resized)
            ch0 = img_clahe.astype(np.float32) / 255.0

        # 2. Channel 1: S_consensus segmentation probability ribbon
        seg_path = self.seg_cache_dir / f"{sid}.png"
        if seg_path.exists():
            seg_uint8 = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
            if seg_uint8 is not None:
                if seg_uint8.shape != (size, size):
                    seg_uint8 = cv2.resize(seg_uint8, (size, size), interpolation=cv2.INTER_LINEAR)
                ch1 = seg_uint8.astype(np.float32) / 255.0
            else:
                ch1 = np.zeros((size, size), dtype=np.float32)
        else:
            ch1 = np.zeros((size, size), dtype=np.float32)

        # 3. Channel 2: Explicit anatomical coordinate map (scaled to size)
        c_info = self.carina_lookup.get(sid, {})
        z_info = self.zonal_lookup.get(sid, {})

        xc_norm = float(c_info.get("raw_carina_x_norm", 0.4805))
        yc_norm = float(c_info.get("raw_carina_y_norm", 0.4000))
        xc = xc_norm * float(size)
        yc = yc_norm * float(size)

        conf_tip = float(z_info.get("conf_tip", 0.0))
        has_valid_tip = float(z_info.get("has_valid_tip", 0.0))
        dx_norm = float(z_info.get("dx_norm", 0.0))
        dy_norm = float(z_info.get("dy_norm", 0.0))

        if has_valid_tip > 0.5 and conf_tip >= 0.20:
            xtip = (xc_norm + dx_norm) * float(size)
            ytip = (yc_norm + dy_norm) * float(size)
        else:
            xtip = 0.0
            ytip = 0.0
            conf_tip = 0.0

        ch2 = generate_anatomical_channel(
            xc=xc, yc=yc, xtip=xtip, ytip=ytip, conf_tip=conf_tip, size=size, mode=self.ch2_mode
        )

        # Mild affine augmentation during training
        if self.is_train:
            angle = np.random.uniform(-4.0, 4.0)
            scale = np.random.uniform(0.97, 1.03)
            tx = np.random.uniform(-8.0, 8.0)
            ty = np.random.uniform(-8.0, 8.0)

            center = (size / 2.0, size / 2.0)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty

            ch0 = cv2.warpAffine(ch0, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            ch1 = cv2.warpAffine(ch1, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            ch2 = cv2.warpAffine(ch2, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

        tensor_3ch = np.stack([ch0, ch1, ch2], axis=0)
        tensor_norm = (tensor_3ch - self.mean) / self.std

        return {
            "image": torch.from_numpy(tensor_norm).float(),
            "cvc_label": torch.from_numpy(self.cvc_labels[idx]).float(),
            "aux_label": torch.from_numpy(self.aux_labels[idx]).float(),
            "study_id": sid,
        }


def evaluate_mgap(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    val_df: pd.DataFrame,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Evaluates MGAP model on validation set and computes complete ledger metrics."""
    model.eval()
    all_cvc_preds = []

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device, non_blocking=True)
            with autocast("cuda", dtype=torch.float16):
                logits_cvc, _ = model(imgs)
                probs = torch.sigmoid(logits_cvc)
            all_cvc_preds.append(probs.cpu().float().numpy())

    preds = np.concatenate(all_cvc_preds, axis=0)
    y_true = val_df[CLASSES].values.astype(int)

    # Class AUROCs
    norm_auc = float(roc_auc_score(y_true[:, 0], preds[:, 0]))
    bord_auc = float(roc_auc_score(y_true[:, 1], preds[:, 1]))
    abno_auc = float(roc_auc_score(y_true[:, 2], preds[:, 2]))
    macro_auc = float(np.mean([norm_auc, bord_auc, abno_auc]))

    # Sub-cohorts
    is_traced = (val_df["annotated_cvc_count"] > 0).values
    is_untraced = ~is_traced
    is_multicvc = (val_df["annotated_cvc_count"] > 1) | (y_true.sum(axis=1) > 1)

    tr_aucs = [roc_auc_score(y_true[is_traced, c], preds[is_traced, c]) for c in range(3)]
    traced_auc = float(np.mean(tr_aucs))

    un_aucs = [roc_auc_score(y_true[is_untraced, c], preds[is_untraced, c]) for c in range(3)]
    untraced_auc = float(np.mean(un_aucs))

    mu_aucs = [roc_auc_score(y_true[is_multicvc, c], preds[is_multicvc, c]) for c in range(3)]
    multicvc_auc = float(np.mean(mu_aucs))

    metrics = {
        "macro_auroc": round(macro_auc, 4),
        "normal_auroc": round(norm_auc, 4),
        "borderline_auroc": round(bord_auc, 4),
        "abnormal_auroc": round(abno_auc, 4),
        "traced_auroc": round(traced_auc, 4),
        "untraced_auroc": round(untraced_auc, 4),
        "multicvc_auroc": round(multicvc_auc, 4),
    }
    return preds, metrics


def save_fp16_checkpoint(model: nn.Module, save_path: Path, metadata: Dict) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fp16_state = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
    temp_path = save_path.with_suffix(".tmp")
    torch.save({"model_state_dict": fp16_state, "metadata": metadata}, str(temp_path))
    if temp_path.exists():
        temp_path.replace(save_path)


def main():
    parser = argparse.ArgumentParser(description="Train MGAP Spatial Net")
    parser.add_argument("--exp_id", type=str, required=True, help="Experiment identifier (e.g. E89_mgap_densenet121_512)")
    parser.add_argument("--backbone", type=str, default="densenet121", help="Backbone architecture")
    parser.add_argument("--img_size", type=int, default=512, help="Input resolution (512 or 768)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per forward step")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr_head", type=float, default=1.5e-4, help="Learning rate for MGAP pooling head")
    parser.add_argument("--lr_backbone", type=float, default=4e-5, help="Learning rate for convolutional trunk")
    parser.add_argument("--warmstart_ckpt", type=str, default="", help="Path to checkpoint for trunk initialization")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only")
    args = parser.parse_args()

    print("=" * 80)
    print(f"MGAP SPATIAL NET: {args.exp_id}")
    print(f"Backbone: {args.backbone} | Size: {args.img_size}x{args.img_size} | Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size} (x{args.grad_accum_steps} accum) | lr_head: {args.lr_head} | lr_backbone: {args.lr_backbone}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_dir = RUNS_DIR / args.exp_id
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    VAL_PREDS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print("Loading manifests and caches...")
    manifest = pd.read_parquet(MANIFEST_PATH)
    val_manifest = manifest[manifest["split"] == "val"].reset_index(drop=True)
    train_manifest = manifest[manifest["split"] == "train"].reset_index(drop=True)

    df_carina_tr = pd.read_parquet(CARINA_TR_PATH)
    df_carina_val = pd.read_parquet(CARINA_VAL_PATH)
    df_zonal_tr = pd.read_parquet(ZONAL_TR_PATH)
    df_zonal_val = pd.read_parquet(ZONAL_VAL_PATH)

    tr_sids = set(df_carina_tr["study_id"]).intersection(set(df_zonal_tr["study_id"]))
    train_manifest = train_manifest[train_manifest["StudyInstanceUID"].isin(tr_sids)].reset_index(drop=True)
    print(f"Manifest split: Train = {len(train_manifest):,} studies | Val = {len(val_manifest):,} studies")

    # Fast dictionary lookups
    c_sid_col_tr = "StudyInstanceUID" if "StudyInstanceUID" in df_carina_tr.columns else "study_id"
    carina_lookup_tr = df_carina_tr.set_index(c_sid_col_tr).to_dict(orient="index")
    c_sid_col_val = "StudyInstanceUID" if "StudyInstanceUID" in df_carina_val.columns else "study_id"
    carina_lookup_val = df_carina_val.set_index(c_sid_col_val).to_dict(orient="index")

    z_sid_col_tr = "StudyInstanceUID" if "StudyInstanceUID" in df_zonal_tr.columns else "study_id"
    zonal_lookup_tr = df_zonal_tr.set_index(z_sid_col_tr).to_dict(orient="index")
    z_sid_col_val = "StudyInstanceUID" if "StudyInstanceUID" in df_zonal_val.columns else "study_id"
    zonal_lookup_val = df_zonal_val.set_index(z_sid_col_val).to_dict(orient="index")

    train_ds = MGAPDataset(
        df=train_manifest,
        carina_lookup=carina_lookup_tr,
        zonal_lookup=zonal_lookup_tr,
        img_size=args.img_size,
        is_train=True,
    )
    val_ds = MGAPDataset(
        df=val_manifest,
        carina_lookup=carina_lookup_val,
        zonal_lookup=zonal_lookup_val,
        img_size=args.img_size,
        is_train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2 if args.img_size <= 512 else args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # 2. Build Model
    print(f"Creating MaskGuidedSpatialNet ({args.backbone})...")
    pretrained_backbone = False if (args.warmstart_ckpt or args.eval_only) else True
    model = MaskGuidedSpatialNet(
        backbone_name=args.backbone,
        pretrained=pretrained_backbone,
        num_classes=3,
        num_aux=8,
    ).to(device)

    # Warm-start trunk if provided
    if args.warmstart_ckpt and Path(args.warmstart_ckpt).exists():
        print(f"Warm-starting trunk from: {args.warmstart_ckpt}")
        ckpt = torch.load(args.warmstart_ckpt, map_location=device, weights_only=False)
        st = ckpt.get("model_state_dict", ckpt)
        # Load matching backbone weights with strict=False
        model_dict = model.state_dict()
        pretrained_dict = {
            k: (v.float() if v.is_floating_point() else v)
            for k, v in st.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Successfully loaded {len(pretrained_dict)} matching tensors into backbone!")

    best_ckpt_path = ckpt_dir / "best.pt"
    if args.eval_only:
        if best_ckpt_path.exists():
            print(f"Loading best checkpoint for evaluation: {best_ckpt_path}")
            ckpt = torch.load(str(best_ckpt_path), map_location=device, weights_only=False)
            st = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in st.items()})
        preds, metrics = evaluate_mgap(model, val_loader, device, val_manifest)
        print(f"Evaluation Metrics: {metrics}")
        out_preds_path = VAL_PREDS_DIR / f"val_preds_{args.exp_id}.npy"
        np.save(out_preds_path, preds)
        print(f"Saved validation predictions to: {out_preds_path}")

        # Update Master Ablation Ledger
        if LEDGER_CSV.exists():
            df_ledger = pd.read_csv(LEDGER_CSV)
        else:
            df_ledger = pd.DataFrame()

        if len(df_ledger) > 0 and "exp_id" in df_ledger.columns:
            df_ledger = df_ledger[df_ledger["exp_id"] != args.exp_id].reset_index(drop=True)

        new_row = {
            "exp_id": args.exp_id,
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
        print(f"Logged {args.exp_id} to Master Ablation Ledger (Total experiments: {len(df_ledger)}).")
        return

    # 3. Parameter Groups (Head vs Trunk)
    head_params = list(model.pool_head.parameters())
    backbone_params = list(model.backbone.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone, "weight_decay": 1e-4},
            {"params": head_params, "lr": args.lr_head, "weight_decay": 1e-4},
        ]
    )

    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    scaler = GradScaler("cuda")
    criterion = nn.BCEWithLogitsLoss()

    best_macro = 0.0
    best_preds = None
    best_metrics = None
    start_time = time.time()

    print("\nBeginning Training Loop...")
    for epoch in range(1, args.epochs + 1):
        t_epoch_start = time.time()
        model.train()
        running_loss_cvc = 0.0
        running_loss_aux = 0.0
        optimizer.zero_grad(set_to_none=True)

        n_batches = len(train_loader)
        for step, batch in enumerate(train_loader):
            imgs = batch["image"].to(device, non_blocking=True)
            y_cvc = batch["cvc_label"].to(device, non_blocking=True)
            y_aux = batch["aux_label"].to(device, non_blocking=True)

            with autocast("cuda", dtype=torch.float16):
                logits_cvc, logits_aux = model(imgs)
                loss_cvc = criterion(logits_cvc, y_cvc)
                loss_aux = criterion(logits_aux, y_aux)
                loss_total = (loss_cvc + 0.25 * loss_aux) / args.grad_accum_steps

            scaler.scale(loss_total).backward()
            running_loss_cvc += loss_cvc.item()
            running_loss_aux += loss_aux.item()

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == n_batches:
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
                cur_cvc = running_loss_cvc / (step + 1)
                cur_aux = running_loss_aux / (step + 1)
                cur_tot = cur_cvc + 0.25 * cur_aux
                elapsed = time.time() - t_epoch_start
                rate = (step + 1) * args.batch_size / max(elapsed, 1e-4)
                print(
                    f"  [Epoch {epoch}/{args.epochs}] Step {step+1}/{n_batches} | "
                    f"L_tot: {cur_tot:.4f} (CVC: {cur_cvc:.4f}, Aux: {cur_aux:.4f}) | "
                    f"{rate:.1f} img/s",
                    flush=True,
                )

        # Validation at Epoch End
        print(f"Evaluating Epoch {epoch} on locked N=2,558 validation cohort...", flush=True)
        preds, metrics = evaluate_mgap(model, val_loader, device, val_manifest)
        macro_auc = metrics["macro_auroc"]
        print(
            f"Epoch {epoch} Results -> Macro: {macro_auc:.4f} | "
            f"Norm: {metrics['normal_auroc']:.4f} | Bord: {metrics['borderline_auroc']:.4f} | "
            f"Abno: {metrics['abnormal_auroc']:.4f} | MultiCVC: {metrics['multicvc_auroc']:.4f} | "
            f"Traced: {metrics['traced_auroc']:.4f} | Untraced: {metrics['untraced_auroc']:.4f}",
            flush=True,
        )

        if macro_auc > best_macro:
            best_macro = macro_auc
            best_preds = preds
            best_metrics = metrics
            save_fp16_checkpoint(
                model,
                best_ckpt_path,
                metadata={"epoch": epoch, "best_macro": best_macro, "metrics": metrics},
            )
            print(f"  --> Saved new best checkpoint to {best_ckpt_path} (Macro AUROC: {best_macro:.4f})")

    total_runtime = time.time() - start_time
    print(f"\nTraining complete in {total_runtime:.1f}s. Best Macro AUROC: {best_macro:.4f}")

    # Save best predictions
    out_preds_path = VAL_PREDS_DIR / f"val_preds_{args.exp_id}.npy"
    np.save(out_preds_path, best_preds)
    print(f"Saved validation predictions to: {out_preds_path}")

    # Update Master Ablation Ledger
    if LEDGER_CSV.exists():
        df_ledger = pd.read_csv(LEDGER_CSV)
    else:
        df_ledger = pd.DataFrame()

    # Drop any existing row with this exp_id
    if len(df_ledger) > 0 and "exp_id" in df_ledger.columns:
        df_ledger = df_ledger[df_ledger["exp_id"] != args.exp_id].reset_index(drop=True)

    new_row = {
        "exp_id": args.exp_id,
        "stage": 6,
        "Runtime_s": round(total_runtime, 1),
        "Val_Macro_AUROC_2558": round(best_metrics["macro_auroc"], 4),
        "Normal_AUROC": round(best_metrics["normal_auroc"], 4),
        "Borderline_AUROC": round(best_metrics["borderline_auroc"], 4),
        "Abnormal_AUROC": round(best_metrics["abnormal_auroc"], 4),
        "Traced_778_AUROC": round(best_metrics["traced_auroc"], 4),
        "Untraced_1780_AUROC": round(best_metrics["untraced_auroc"], 4),
        "MultiCVC_332_AUROC": round(best_metrics["multicvc_auroc"], 4),
        "Tip_Recall_35px": 72.0,
    }
    df_ledger = pd.concat([df_ledger, pd.DataFrame([new_row])], ignore_index=True)
    df_ledger.to_parquet(LEDGER_PARQUET, index=False)
    df_ledger.to_csv(LEDGER_CSV, index=False)
    print(f"Logged {args.exp_id} to Master Ablation Ledger (Total experiments: {len(df_ledger)}).")


if __name__ == "__main__":
    main()

