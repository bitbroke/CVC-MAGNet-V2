"""Training Script for Mediastinal RoI Dual-Branch Fusion Model (Zero-Domain-Shift).

Implements Step 4 of the CVC Placement Architecture:
- Backbone: ConvNeXt-Tiny (or EfficientNet-B0) + Geometric MLP
- 384x384 Mediastinal crops with tight 0.32W x 0.32H anatomical window
- 3-Channel Input: CLAHE crop + T2 seg_prob + spatial Carina/Tip reference map
- 25-D Zonal & Gated Geometric Vector: Scaled with StandardScaler and clipped to [-5, 5]
- Modality Dropout (p=0.25) to prevent visual branch collapse
- Training Augmentation: Uniform(0.5, 1.2) intensity scaling + ChannelDropout(p=0.2) on seg_prob
- Multi-label BCEWithLogitsLoss with rank-preserving mild pos_weight [1.0, 1.5, 2.0]
- Evaluates out-of-sample across the FULL N=2,558 locked validation cohort after every epoch
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.placement_roi_dataset import CachedMediastinalRoIDataset, FEATURE_COLS
from src.models.placement_roi_fusion import MediastinalFusionNet

DATA_DIR = PROJECT_ROOT / "data"
ZONAL_TRAIN_PATH = DATA_DIR / "curated_cvc_v1" / "zonal_train_features.parquet"
ZONAL_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "zonal_val_features.parquet"
ROI_CACHE_DIR = DATA_DIR / "curated_cvc_v1" / "roi_cache"
RUNS_DIR = PROJECT_ROOT / "runs"

CLASSES = ["CVC - Normal", "CVC - Borderline", "CVC - Abnormal"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mediastinal RoI Fusion Network")
    parser.add_argument("--backbone", type=str, default="convnext_tiny", choices=["convnext_tiny", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per forward pass")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--modality_dropout_p", type=float, default=0.25, help="Geometric branch dropout probability")
    parser.add_argument("--run_name", type=str, default="roi_fusion_convnext_tiny_v2")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader workers (0 for Windows)")
    return parser.parse_args()


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate dual-branch model across the full N=2,558 validation cohort."""
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            roi = batch["roi"].to(device)
            geom = batch["geom"].to(device)
            target = batch["target"]

            with autocast(dtype=torch.float16):
                out = model(roi, geom)
                probs = out["probs"].cpu().numpy()

            all_preds.append(probs)
            all_targets.append(target.numpy())

    Y_pred = np.vstack(all_preds)
    Y_true = np.vstack(all_targets)

    results = {}
    aurocs = []
    for c_idx, c_name in enumerate(CLASSES):
        try:
            auc = roc_auc_score(Y_true[:, c_idx], Y_pred[:, c_idx])
        except Exception:
            auc = 0.5
        aurocs.append(auc)
        results[c_name] = round(float(auc), 4)

    macro_auc = float(np.mean(aurocs))
    results["macro_auroc"] = round(macro_auc, 4)
    results["total_validation_studies"] = len(Y_pred)

    return results


def main():
    args = parse_args()
    print("=" * 70)
    print("STEP 4: MEDIASTINAL ROI DUAL-BRANCH FUSION TRAINING (V2)")
    print(f"Backbone: {args.backbone} | Modality Dropout: {args.modality_dropout_p}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load DataFrames
    df_zonal_train = pd.read_parquet(ZONAL_TRAIN_PATH)
    df_zonal_val = pd.read_parquet(ZONAL_VAL_PATH)

    print(f"Loaded {len(df_zonal_train):,} training studies and {len(df_zonal_val):,} validation studies.")

    # 2. Fit StandardScaler on Training Features
    raw_train_feats = df_zonal_train[FEATURE_COLS].values.astype(np.float32)
    raw_train_feats = np.nan_to_num(raw_train_feats, nan=0.0)
    scaler = StandardScaler()
    scaler.fit(raw_train_feats)

    # 3. Datasets & Loaders
    print("Initializing CachedMediastinalRoIDataset (Train & Val)...")
    train_dataset = CachedMediastinalRoIDataset(
        zonal_df=df_zonal_train,
        scaler=scaler,
        is_train=True,
        channel_dropout_p=0.20,
    )

    val_dataset = CachedMediastinalRoIDataset(
        zonal_df=df_zonal_val,
        scaler=scaler,
        is_train=False,
    )

    print(f"Train samples: {len(train_dataset):,} | Val samples: {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 4. Multi-Label BCEWithLogitsLoss with Mild Clamped Weights
    # Normal is ~72%, Borderline ~28%, Abnormal ~10%
    # Mild weights [1.0, 1.5, 2.0] stabilize ranking without exploding minority gradients
    pos_weights = torch.tensor([1.0, 1.5, 2.0], dtype=torch.float32).to(device)
    print(f"BCE pos_weight: {pos_weights.cpu().numpy().tolist()}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # 5. Model, Optimizer, Scaler
    model = MediastinalFusionNet(
        backbone_name=args.backbone,
        pretrained=True,
        geom_in_dim=25,
        visual_dim=128,
        geom_dim=32,
        num_classes=3,
        modality_dropout_p=args.modality_dropout_p,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * (len(train_loader) // args.grad_accum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.05)
    scaler_amp = GradScaler()

    run_dir = RUNS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_macro_auc = 0.0
    history = []

    print(f"\nStarting training for {args.epochs} epochs (grad_accum={args.grad_accum}, effective batch={args.batch_size * args.grad_accum})...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            roi = batch["roi"].to(device)
            geom = batch["geom"].to(device)
            target = batch["target"].to(device)

            with autocast(dtype=torch.float16):
                out = model(roi, geom)
                loss = criterion(out["logits"], target)
                loss = loss / args.grad_accum

            scaler_amp.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scale_before = scaler_amp.get_scale()
                scaler_amp.step(optimizer)
                scaler_amp.update()
                scale_after = scaler_amp.get_scale()
                if scale_before <= scale_after:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * args.grad_accum
            n_batches += 1

            if (step + 1) % 100 == 0:
                cur_loss = epoch_loss / n_batches
                print(f"  [Epoch {epoch}/{args.epochs}] Step {step+1}/{len(train_loader)} | Loss: {cur_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}", flush=True)

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        # Evaluate across the full N=2,558 locked validation cohort
        val_res = evaluate_model(model, val_loader, device)
        macro_auc = val_res["macro_auroc"]
        print(f"\n--> Epoch {epoch} Complete ({elapsed:.1f}s) | Train Loss: {avg_loss:.4f}")
        print(f"    Val Macro AUROC (N=2,558): {macro_auc:.4f} | Normal: {val_res['CVC - Normal']:.4f} | Borderline: {val_res['CVC - Borderline']:.4f} | Abnormal: {val_res['CVC - Abnormal']:.4f}", flush=True)

        record = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "val_macro_auroc": macro_auc,
            "val_normal_auroc": val_res["CVC - Normal"],
            "val_borderline_auroc": val_res["CVC - Borderline"],
            "val_abnormal_auroc": val_res["CVC - Abnormal"],
        }
        history.append(record)

        # Save best checkpoint (lightweight float16, ~54MB, no optimizer state)
        if macro_auc > best_macro_auc:
            best_macro_auc = macro_auc
            tmp_ckpt = ckpt_dir / "best_temp.pt"
            best_ckpt = ckpt_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": {k: v.cpu().half() for k, v in model.state_dict().items()},
                "val_macro_auroc": macro_auc,
                "val_metrics": val_res,
                "args": vars(args),
            }, tmp_ckpt)
            if best_ckpt.exists():
                best_ckpt.unlink()
            tmp_ckpt.rename(best_ckpt)
            print(f"    *** NEW BEST VAL MACRO AUROC: {best_macro_auc:.4f} -> Saved checkpoint to {best_ckpt}", flush=True)

    # Save training history
    with open(run_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE. Best Val Macro AUROC: {best_macro_auc:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
