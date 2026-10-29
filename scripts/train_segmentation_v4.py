"""Trainer for CVC-Segmentation-V4 with Attention Gates, Deep Supervision, and Advanced Medical Augmentations.

Supports:
- Backbones: efficientnet_b0, efficientnet_b2, efficientnet_b4, resnet34d via timm
- Attention-Gated skip connections
- Multi-scale deep supervision with aux loss balancing
- Full deterministic resume and exposure ledger
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.segmentation_v4_dataset import (
    BalancedCohortSampler,
    create_segmentation_v4_dataloader,
)
from src.models.segmentation_v4 import CVCSegmentationV4Net


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_per_image_dice(
    pred_prob: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred_valid = pred_prob * valid_mask
    targ_valid = target * valid_mask
    intersection = (pred_valid * targ_valid).sum(dim=(1, 2, 3))
    cardinality = (pred_valid + targ_valid).sum(dim=(1, 2, 3))
    return (2.0 * intersection + eps) / (cardinality + eps)


def train_segmentation_v4(
    output_dir: str,
    cohort_manifest: str,
    backbone: str = "efficientnet_b0",
    use_attention: bool = True,
    deep_supervision: bool = True,
    aux_weight: float = 0.3,
    target_updates: int = 10000,
    effective_batch_size: int = 4,
    micro_batch_size: int = 2,
    img_size: int = 768,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    bce_pos_weight: float = 15.0,
    dice_weight: float = 1.5,
    seed: int = 42,
    aug_seed: int = 42,
    resume_checkpoint: Optional[str] = None,
    overwrite: bool = False,
):
    out_path = Path(output_dir)
    ckpt_dir = out_path / "checkpoints"
    final_path = ckpt_dir / "final_segmentation_v4.pt"

    if out_path.exists() and (out_path / "train_summary.json").exists() and not resume_checkpoint and not overwrite:
        raise FileExistsError(f"Output directory {output_dir} already contains run artifacts. Specify --overwrite or new path.")

    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    data_dir = PROJECT_ROOT / "data"
    inv_df = pd.read_parquet(data_dir / "annotation_inventory.parquet")
    study_inv_df = pd.read_parquet(data_dir / "study_inventory.parquet")
    with open(data_dir / "splits.json") as f:
        splits = json.load(f)

    if not os.path.exists(cohort_manifest):
        raise FileNotFoundError(f"Cohort manifest not found: {cohort_manifest}")

    c_df = pd.read_parquet(cohort_manifest)
    train_study_ids = c_df["study_id"].astype(str).tolist()
    pos_study_ids = c_df[c_df["has_cvc"] == True]["study_id"].astype(str).tolist()
    neg_study_ids = c_df[c_df["has_cvc"] == False]["study_id"].astype(str).tolist()

    val_study_ids = splits["partitions"]["dev_pilot_val"]["study_ids"]
    val_leak = set(train_study_ids).intersection(set(val_study_ids))
    assert len(val_leak) == 0, f"LEAKAGE: {len(val_leak)} studies overlap with val!"

    images_dir = r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train"

    train_loader, train_sampler = create_segmentation_v4_dataloader(
        study_ids=train_study_ids,
        inventory_df=inv_df,
        study_inventory_df=study_inv_df,
        images_dir=images_dir,
        batch_size=effective_batch_size,
        img_size=img_size,
        is_train=True,
        num_workers=0,
        total_samples_per_epoch=len(train_study_ids),
        seed=seed,
        aug_seed=aug_seed,
    )

    val_loader, _ = create_segmentation_v4_dataloader(
        study_ids=val_study_ids,
        inventory_df=inv_df,
        study_inventory_df=study_inv_df,
        images_dir=images_dir,
        batch_size=micro_batch_size,
        img_size=img_size,
        is_train=False,
        num_workers=0,
    )

    print(f"Instantiating CVC-Segmentation-V4 ({backbone}, Attention={use_attention}, DeepSupervision={deep_supervision})...", flush=True)
    model = CVCSegmentationV4Net(
        backbone=backbone,
        pretrained=True,
        freeze_encoder_bn=True,
        deep_supervision=deep_supervision,
        use_attention=use_attention,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_steps = min(100, max(1, target_updates // 10))
    warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, target_updates - warmup_steps), eta_min=1e-6)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = GradScaler("cuda", enabled=(torch.cuda.is_available() and not use_bf16))
    print(f"Precision Mode: {'bfloat16 (native Ampere, zero-overflow)' if use_bf16 else 'float16 with GradScaler'}", flush=True)

    pos_weight = torch.tensor([bce_pos_weight], device=device, dtype=torch.float32)

    successful_updates = 0
    total_attempts = 0
    skipped_updates = 0
    best_val_dice = -1.0
    start_time = time.time()

    ledger_path = out_path / "exposure_ledger.jsonl"
    ledger_file = open(ledger_path, "a" if resume_checkpoint else "w", encoding="utf-8")

    print(f"\nStarting V4 Training for {target_updates} updates...", flush=True)

    while successful_updates < target_updates:
        for eff_batch in train_loader:
            model.train()
            eff_images = eff_batch["image"]
            eff_masks = eff_batch["mask"]
            eff_valids = eff_batch["valid_mask"]
            eff_losses = eff_batch.get("loss_mask", eff_valids)
            eff_is_pos = eff_batch["is_positive"]
            eff_study_ids = eff_batch["study_id"]
            current_eff_size = len(eff_study_ids)

            eff_supervision = eff_valids * eff_losses
            total_supervised_pixels = float(eff_supervision.sum().item())
            total_positives = int(eff_is_pos.sum().item())

            total_attempts += 1
            optimizer.zero_grad(set_to_none=True)
            eff_total_loss_accum = 0.0
            eff_bce_accum = 0.0
            eff_dice_accum = 0.0

            num_mbs = (current_eff_size + micro_batch_size - 1) // micro_batch_size
            for m in range(num_mbs):
                start_i = m * micro_batch_size
                end_i = min((m + 1) * micro_batch_size, current_eff_size)

                mb_images = eff_images[start_i:end_i].to(device, non_blocking=True)
                mb_masks = eff_masks[start_i:end_i].to(device, non_blocking=True)
                mb_supervision = eff_supervision[start_i:end_i].to(device, non_blocking=True)
                mb_is_pos = eff_is_pos[start_i:end_i].to(device)

                with autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                    out = model(mb_images)
                    logits_fp32 = out["seg_logits"].float()
                    targets_fp32 = mb_masks.float()
                    supervision_fp32 = mb_supervision.float()

                    bce_raw = nn.functional.binary_cross_entropy_with_logits(
                        logits_fp32, targets_fp32, pos_weight=pos_weight, reduction="none"
                    )
                    mb_bce_loss = (bce_raw * supervision_fp32).sum() / max(total_supervised_pixels, 1.0)

                    probs_fp32 = torch.sigmoid(logits_fp32)
                    pos_mask = mb_is_pos.bool()
                    if pos_mask.any() and total_positives > 0:
                        mb_dices = compute_per_image_dice(
                            probs_fp32[pos_mask], targets_fp32[pos_mask], supervision_fp32[pos_mask]
                        )
                        mb_dice_loss = (1.0 - mb_dices).sum() / float(total_positives)
                    else:
                        mb_dice_loss = torch.tensor(0.0, device=device)

                    mb_total_loss = mb_bce_loss + dice_weight * mb_dice_loss

                    # Auxiliary deep supervision losses
                    if deep_supervision and "aux_logits" in out:
                        aux_loss_sum = torch.tensor(0.0, device=device)
                        for aux_log in out["aux_logits"]:
                            aux_bce = (
                                nn.functional.binary_cross_entropy_with_logits(
                                    aux_log.float(), targets_fp32, pos_weight=pos_weight, reduction="none"
                                ) * supervision_fp32
                            ).sum() / max(total_supervised_pixels, 1.0)
                            aux_loss_sum = aux_loss_sum + aux_bce
                        mb_total_loss = mb_total_loss + (aux_weight * aux_loss_sum / len(out["aux_logits"]))

                scaler.scale(mb_total_loss).backward()
                eff_total_loss_accum += mb_total_loss.item()
                eff_bce_accum += mb_bce_loss.item()
                eff_dice_accum += (dice_weight * mb_dice_loss).item()

            scaler.unscale_(optimizer)
            has_inf_or_nan = any(
                p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()
            )

            if has_inf_or_nan:
                skipped_updates += 1
                if scaler.is_enabled():
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            successful_updates += 1

            if successful_updates % 50 == 0 or successful_updates == target_updates:
                elapsed = time.time() - start_time
                vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
                cur_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Update {successful_updates:4d}/{target_updates} (Attempt {total_attempts:4d}) | "
                    f"Loss: {eff_total_loss_accum:.4f} (BCE: {eff_bce_accum:.4f}, Dice: {eff_dice_accum:.4f}) | "
                    f"LR: {cur_lr:.2e} | VRAM: {vram_mb:.0f} MB | Elapsed: {elapsed:.1f}s",
                    flush=True,
                )

            if successful_updates % 250 == 0 or successful_updates == target_updates:
                model.eval()
                val_dices = []
                with torch.no_grad():
                    for vbatch in val_loader:
                        vimgs = vbatch["image"].to(device)
                        vmasks = vbatch["mask"].to(device)
                        vvalids = vbatch["valid_mask"].to(device)
                        vis_pos = vbatch["is_positive"].to(device)
                        with autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                            vout = model(vimgs)
                        pos_mask = vis_pos.bool()
                        if pos_mask.any():
                            d = compute_per_image_dice(
                                vout["seg_probs"][pos_mask], vmasks[pos_mask], vvalids[pos_mask]
                            )
                            val_dices.extend(d.cpu().tolist())

                mean_val_dice = float(np.mean(val_dices)) if val_dices else 0.0
                print(f"  -> Validation Macro Soft Dice: {mean_val_dice:.4f}", flush=True)

                if mean_val_dice > best_val_dice:
                    best_val_dice = mean_val_dice
                    best_path = ckpt_dir / "best_segmentation_v4.pt"
                    tmp_best = best_path.with_suffix(".tmp")
                    torch.save({"model_state_dict": model.state_dict(), "best_val_dice": best_val_dice}, tmp_best)
                    os.replace(tmp_best, best_path)
                    print(f"  -> Saved new best V4 model to {best_path}", flush=True)

            if successful_updates >= target_updates:
                break

    ledger_file.close()
    summary_path = out_path / "train_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "output_dir": str(output_dir),
                "backbone": backbone,
                "use_attention": use_attention,
                "deep_supervision": deep_supervision,
                "best_val_dice": best_val_dice,
                "successful_updates": successful_updates,
                "total_attempts": total_attempts,
                "img_size": img_size,
                "elapsed_seconds": time.time() - start_time,
            },
            f,
            indent=2,
        )
    print(f"\nV4 Training Complete. Best Val Dice: {best_val_dice:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CVC-Segmentation-V4")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cohort_manifest", type=str, default="data/curated_cvc_v1/gold_clean_manifest.parquet")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0")
    parser.add_argument("--use_attention", action="store_true", default=True)
    parser.add_argument("--deep_supervision", action="store_true", default=True)
    parser.add_argument("--aux_weight", type=float, default=0.3)
    parser.add_argument("--target_updates", type=int, default=10000)
    parser.add_argument("--effective_batch_size", type=int, default=4)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--img_size", type=int, default=768)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--bce_pos_weight", type=float, default=15.0)
    parser.add_argument("--dice_weight", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aug_seed", type=int, default=42)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    train_segmentation_v4(
        output_dir=args.output_dir,
        cohort_manifest=args.cohort_manifest,
        backbone=args.backbone,
        use_attention=args.use_attention,
        deep_supervision=args.deep_supervision,
        aux_weight=args.aux_weight,
        target_updates=args.target_updates,
        effective_batch_size=args.effective_batch_size,
        micro_batch_size=args.micro_batch_size,
        img_size=args.img_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        bce_pos_weight=args.bce_pos_weight,
        dice_weight=args.dice_weight,
        seed=args.seed,
        aug_seed=args.aug_seed,
        resume_checkpoint=args.resume_checkpoint,
        overwrite=args.overwrite,
    )
