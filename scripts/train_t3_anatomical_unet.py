"""Trainer for T3 Anatomical U-Net at 768x768 with Soft-clDice & 5x Tip Weighting.

Path 2 - Step 1:
- Input resolution: 768x768, Output: 768x768 segmentation mask + 192x192 tip heatmap (stride 4)
- Loss: TipWeighted-BCE+Dice + 0.35 * Soft-clDice + 2.0 * TipFocalLoss
  where pixels within 35px of distal tip are boosted by 1.0 + 4.0 * exp(-d^2 / (2*sigma^2))
- Warm-starts directly from T2 checkpoint: runs/t2_joint_seg_tip_1000/checkpoints/best.pt
- Budget: 1,200 updates (effective batch 4, micro-batch 1, accum 4)
- Mixed precision AMP (FP16)
- Evaluates tip recall: Success@15px, Success@35px (~5mm), Success@65px (~10mm) via Hungarian matching + D1 DARK
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tip_dataset import CVCTipDataset
from src.evaluation.tip_metrics import compute_tip_metrics
from src.infer.decoders import extract_peaks_from_heatmap, decode_d0_cell_center, decode_d1_subpixel_dark
from src.models.tip_heatmap import CVCJointSegTipNet, TipFocalLoss

DATA_DIR = PROJECT_ROOT / "data"
T2_CKPT_PATH = PROJECT_ROOT / "runs" / "t2_joint_seg_tip_1000" / "checkpoints" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "T3_anatomical_unet"


def seed_everything(seed: int = 43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iters: int = 2) -> torch.Tensor:
    img = torch.clamp(img, 0.0, 1.0)
    img_skel = F.relu(img - soft_open(img))
    for _ in range(iters):
        img = soft_erode(img)
        img_skel = img_skel + F.relu(img - soft_open(img))
    return torch.clamp(img_skel, 0.0, 1.0)


def soft_cldice_loss(v_pred: torch.Tensor, v_true: torch.Tensor, iters: int = 2, smooth: float = 1.0) -> torch.Tensor:
    """Centerline Dice loss for topology preservation along catheter shafts."""
    # Downsample to 384x384 for fast execution & numerical stability
    v_pred_ds = F.interpolate(v_pred, size=(384, 384), mode="bilinear", align_corners=False)
    v_true_ds = F.interpolate(v_true, size=(384, 384), mode="nearest")
    
    s_pred = soft_skeletonize(v_pred_ds, iters=iters)
    s_true = soft_skeletonize(v_true_ds, iters=iters)
    
    tprec = (torch.sum(s_pred * v_true_ds, dim=(-2, -1)) + smooth) / (torch.sum(s_pred, dim=(-2, -1)) + smooth)
    tsens = (torch.sum(s_true * v_pred_ds, dim=(-2, -1)) + smooth) / (torch.sum(s_true, dim=(-2, -1)) + smooth)
    cl_dice = 2.0 * (tprec * tsens) / (tprec + tsens + 1e-6)
    return 1.0 - cl_dice.mean()


def compute_soft_dice(pred_prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    intersection = (pred_prob * target).sum(dim=(1, 2, 3))
    cardinality = (pred_prob + target).sum(dim=(1, 2, 3))
    return (2.0 * intersection + eps) / (cardinality + eps)


class BalancedTipSampler:
    """Samples 3 positive studies and 1 negative study (50% selective hard negatives)."""

    def __init__(self, positive_studies: List[str], uniform_negative_studies: List[str], hard_negative_studies: List[str], seed: int = 43):
        self.positive_studies = list(positive_studies)
        self.uniform_negative_studies = list(uniform_negative_studies)
        self.hard_negative_studies = list(hard_negative_studies)
        self.rng = random.Random(seed)
        self.pos_cursor = 0
        self.rng.shuffle(self.positive_studies)
        self.rng.shuffle(self.uniform_negative_studies)
        self.rng.shuffle(self.hard_negative_studies)

    def sample_batch(self, batch_size: int = 4) -> List[str]:
        batch = []
        for _ in range(3):
            if self.pos_cursor >= len(self.positive_studies):
                self.rng.shuffle(self.positive_studies)
                self.pos_cursor = 0
            batch.append(self.positive_studies[self.pos_cursor])
            self.pos_cursor += 1

        use_hard = self.rng.random() < 0.5 and len(self.hard_negative_studies) > 0
        if use_hard:
            sid = self.rng.choice(self.hard_negative_studies)
        else:
            sid = self.rng.choice(self.uniform_negative_studies)
        batch.append(sid)
        return batch


def evaluate_tip_recall(model: nn.Module, val_dataset: CVCTipDataset, device: torch.device, max_studies: int = 250) -> Dict[str, float]:
    """Evaluate tip localization using D1 DARK subpixel decoding."""
    model.eval()
    val_records = []
    eval_ids = val_dataset.study_ids[:max_studies]

    with torch.no_grad():
        for i, sid in enumerate(eval_ids):
            sample = val_dataset[i]
            img = sample["image"].unsqueeze(0).to(device)
            native_w = sample["native_w"]
            native_h = sample["native_h"]
            is_pos = sample["is_cvc_positive"]
            pid = sample.get("patient_id", sid)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out = model(img)
                tip_map = out["tip_probs"][0, 0].cpu().float().numpy()

            _, nms_peaks = extract_peaks_from_heatmap(tip_map, threshold=0.15, nms_kernel_size=3, suppression_radius_px=3.0)
            pred_tips = []
            for p in nms_peaks:
                nx, ny, _ = decode_d1_subpixel_dark(tip_map, p, stride=4, img_size=768, native_w=native_w, native_h=native_h)
                pred_tips.append({"x": nx, "y": ny, "conf": p["conf"]})

            gt_tips = []
            if sid in val_dataset.study_tips:
                for tx, ty in val_dataset.study_tips[sid]:
                    gt_tips.append({"x": float(tx), "y": float(ty), "tip_state": "assessable", "instance_id": 1, "is_border": False})

            val_records.append({
                "study_id": sid,
                "patient_id": pid,
                "is_cvc_positive": is_pos,
                "gt_tips": gt_tips,
                "pred_tips": pred_tips,
                "raw_candidates": pred_tips,
            })

    val_df = pd.DataFrame(val_records)
    metrics_bundle = compute_tip_metrics(val_df, confidence_threshold=0.15)
    model.train()
    return metrics_bundle


def main():
    parser = argparse.ArgumentParser(description="T3 Anatomical U-Net Training")
    parser.add_argument("--target_updates", type=int, default=1200)
    parser.add_argument("--effective_batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=768)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--eval_interval", type=int, default=300)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"PATH 2 - STEP 1: TRAINING T3 ANATOMICAL U-NET (768x768, Soft-clDice, 5x Tip Weight)")
    print(f"Device: {device} | Budget: {args.target_updates} updates | Warm-start: {T2_CKPT_PATH.name}")
    print("=" * 80)

    ckpt_dir = OUTPUT_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data Manifests
    catheter_df = pd.read_parquet(DATA_DIR / "cvc_catheter_contracts.parquet")
    study_sup_df = pd.read_parquet(DATA_DIR / "cvc_study_supervision.parquet")
    
    with open(DATA_DIR / "expanded_training_cohort_manifest.json", "r") as f:
        expanded_manifest = json.load(f)
    train_positive_ids = expanded_manifest["positive_study_ids"]
    all_neg_ids = expanded_manifest["negative_study_ids"]

    with open(DATA_DIR / "selective_hard_negatives_manifest.json", "r") as f:
        hn_manifest = json.load(f)
    hard_neg_ids = hn_manifest["study_ids"]
    uniform_neg_ids = [s for s in all_neg_ids if s not in hard_neg_ids]

    with open(DATA_DIR / "splits.json", "r") as f:
        splits = json.load(f)
    val_study_ids = splits["partitions"]["dev_pilot_val"]["study_ids"]

    images_dir = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")
    print(f"Cohort: {len(train_positive_ids):,} positive traces, {len(uniform_neg_ids):,} uniform negs, {len(hard_neg_ids):,} hard negs.")

    # 2. Datasets
    train_all_ids = list(set(train_positive_ids + all_neg_ids))
    train_dataset = CVCTipDataset(
        study_ids=train_all_ids,
        catheter_contracts_df=catheter_df,
        study_supervision_df=study_sup_df,
        images_dir=images_dir,
        img_size=args.img_size,
        is_train=True,
        include_segmentation=True,
        aug_seed=args.seed,
    )
    study_id_to_idx = {sid: idx for idx, sid in enumerate(train_all_ids)}

    val_dataset = CVCTipDataset(
        study_ids=val_study_ids,
        catheter_contracts_df=catheter_df,
        study_supervision_df=study_sup_df,
        images_dir=images_dir,
        img_size=args.img_size,
        is_train=False,
        include_segmentation=False,
    )

    sampler = BalancedTipSampler(
        positive_studies=train_positive_ids,
        uniform_negative_studies=uniform_neg_ids,
        hard_negative_studies=hard_neg_ids,
        seed=args.seed,
    )

    # 3. Model & Warm-Start from T2
    model = CVCJointSegTipNet(pretrained=False, freeze_encoder_bn=True)
    if T2_CKPT_PATH.exists():
        print(f"Warm-starting from T2 checkpoint: {T2_CKPT_PATH}...")
        st = torch.load(T2_CKPT_PATH, map_location="cpu", weights_only=False)
        src_state = st.get("model_state_dict", st.get("state_dict", st))
        tgt_state = model.state_dict()
        transferred = 0
        for k, v in src_state.items():
            if k in tgt_state and tgt_state[k].shape == v.shape:
                tgt_state[k] = v
                transferred += 1
        model.load_state_dict(tgt_state)
        print(f"Successfully transferred {transferred} weights from converged T2 checkpoint!")
    else:
        print("Warning: T2 checkpoint not found! Initializing with pretrained backbone...")
        model = CVCJointSegTipNet(pretrained=True, freeze_encoder_bn=True)

    model.to(device)

    # 4. Optimizer, Scheduler & Losses
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.target_updates, eta_min=2e-6)
    scaler = GradScaler(enabled=(device.type == "cuda"), init_scale=1024.0)
    tip_criterion = TipFocalLoss(alpha=2.0, beta=4.0)

    # 5. Training Loop
    updates = 0
    accum_steps = args.effective_batch_size  # micro-batch 1, accum 4
    best_success35 = -1.0
    best_metrics = {}
    t_start = time.time()

    print(f"\n--- Launching {args.target_updates} updates ---")

    while updates < args.target_updates:
        batch_study_ids = sampler.sample_batch(batch_size=args.effective_batch_size)
        micro_batches = [train_dataset[study_id_to_idx[sid]] for sid in batch_study_ids]

        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_cldice = 0.0

        for mb in micro_batches:
            img = mb["image"].unsqueeze(0).to(device)
            targ_tip = mb["tip_heatmap"].unsqueeze(0).to(device)
            tip_mask = mb["tip_mask"].view(1, 1, 1, 1).to(device)
            targ_seg = mb["seg_mask"].unsqueeze(0).to(device)
            seg_mask_valid = mb["seg_mask_valid"].to(device)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out = model(img)
                pred_tip = out["tip_probs"]
                pred_seg_logits = out["seg_logits"]
                pred_seg_probs = out["seg_probs"]

                # 1. Tip Focal Loss (weight 2.0)
                loss_tip = tip_criterion(pred_tip.float(), targ_tip.float(), mask=tip_mask.float())

                # 2. Tip-Weighted BCE + Soft Dice
                if seg_mask_valid > 0.5:
                    # Upsample tip Gaussian to 768x768 to form 5x tip distance weight
                    tip_weight_map = 1.0 + 4.0 * F.interpolate(targ_tip.float(), size=(768, 768), mode="bilinear", align_corners=False)
                    bce_raw = F.binary_cross_entropy_with_logits(pred_seg_logits.float(), targ_seg.float(), reduction="none")
                    loss_bce_weighted = (bce_raw * tip_weight_map).mean()

                    if mb["is_cvc_positive"]:
                        dice_loss = 1.0 - compute_soft_dice(pred_seg_probs.float(), targ_seg.float()).mean()
                        # 3. Soft-clDice loss (weight 0.35)
                        loss_cldice = soft_cldice_loss(pred_seg_probs.float(), targ_seg.float(), iters=2)
                    else:
                        dice_loss = torch.tensor(0.0, device=device)
                        loss_cldice = torch.tensor(0.0, device=device)

                    loss_seg = loss_bce_weighted + dice_loss
                else:
                    loss_seg = torch.tensor(0.0, device=device)
                    loss_cldice = torch.tensor(0.0, device=device)

                # Total loss per micro-batch
                total_loss = (2.0 * loss_tip + 1.0 * loss_seg + 0.35 * loss_cldice) / accum_steps

            scaler.scale(total_loss).backward()
            running_loss += total_loss.item() * accum_steps
            running_cldice += loss_cldice.item()

        # Step optimizer
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        updates += 1

        if updates % 50 == 0 or updates == args.target_updates:
            elapsed = time.time() - t_start
            rate = updates * args.effective_batch_size / max(elapsed, 1e-4)
            print(f"  Update {updates:4d}/{args.target_updates} | Loss: {running_loss:.4f} (clDice: {running_cldice/accum_steps:.4f}) | {rate:.1f} img/s (LR: {scheduler.get_last_lr()[0]:.2e})", flush=True)

        # Evaluation Gate
        if updates % args.eval_interval == 0 or updates == args.target_updates:
            print(f"\nEvaluating Tip Recall at Update {updates} via D1 DARK on Validation Set...")
            metrics = evaluate_tip_recall(model, val_dataset, device, max_studies=200)
            s15 = metrics.get("primary", {}).get("success_rate", 0.0)
            s35 = metrics.get("secondary_tolerances", {}).get("tolerance_35px", {}).get("success_rate", 0.0)
            s65 = metrics.get("secondary_tolerances", {}).get("tolerance_65px", {}).get("success_rate", 0.0)
            print(f"--> [Update {updates}] Success@15px: {s15*100:.1f}% | Success@35px (~5mm): {s35*100:.1f}% | Success@65px (~10mm): {s65*100:.1f}% (T2 Baseline: 33.8% / 55.7% / 65.7%)", flush=True)

            if s35 > best_success35:
                best_success35 = s35
                best_metrics = {"update": updates, "s15": s15, "s35": s35, "s65": s65}
                ckpt_path = ckpt_dir / "best.pt"
                # Save compact FP16 checkpoint
                fp16_state = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
                torch.save({"model_state_dict": fp16_state, "metadata": best_metrics}, str(ckpt_path))
                print(f"  *** New Best T3 Checkpoint Saved! (Success@35px: {s35*100:.1f}%) ***\n", flush=True)

    # Save final checkpoint
    final_fp16 = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
    torch.save({"model_state_dict": final_fp16, "metadata": best_metrics}, str(ckpt_dir / "final.pt"))
    print("=" * 80)
    print(f"T3 Training Completed in {(time.time()-t_start)/60:.1f} min! Best Success@35px: {best_success35*100:.1f}%")
    print(f"Saved best checkpoint to: {ckpt_dir / 'best.pt'}")
    print("=" * 80)


if __name__ == "__main__":
    main()

