"""Trainer for CVC-Segmentation-V3 with Strict Equivalence, Telemetry, and Resume Determinism.

Supervision and Objective Guarantees:
1. Segmentation-only: no tip, anatomy, or classification heads.
2. Padding exclusion: valid_mask explicitly excludes letterbox padding from loss and metrics.
3. Strict Effective-Batch Loss Equivalence:
   - Positive-pixel BCE normalized by effective-batch valid pixels (V_eff).
   - Per-positive-image soft Dice loss normalized by total effective-batch positives (P_eff).
   - Equivalence invariant to micro-batch size and gradient accumulation steps.
4. Real Optimizer Step Counting & Bounded Attempts:
   - Tracks update attempts, real updates, and AMP skips due to non-finite gradients.
   - Schedulers and step counters never advance on non-finite gradient attempts.
   - Fails fast if consecutive non-finite attempts exceed threshold.
5. Deterministic Reproducibility & State Persistence:
   - Dedicated augmentation RNG decoupled from model forward randomness.
   - Initial weight file loading fails fast if missing; refuses silent fallback.
   - Refuses accidental reuse of completed output directories.
   - Ordered study exposure and per-sample augmentation parameters recorded to ledger.
   - Exact resume restores CPU/CUDA RNG tensors, Python/NumPy states, sampler cursor, and exposure counts.
   - Checkpoints saved atomically via temporary files.
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

from src.data.segmentation_v3_dataset import (
    BalancedCohortSampler,
    create_segmentation_v3_dataloader,
)
from src.models.segmentation_v3 import CVCSegmentationV3Net


def seed_everything(seed: int = 42):
    """Seed Python, NumPy, and PyTorch for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_sha256(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_per_image_dice(
    pred_prob: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute per-image soft Dice for batch of positive images."""
    pred_valid = pred_prob * valid_mask
    targ_valid = target * valid_mask

    intersection = (pred_valid * targ_valid).sum(dim=(1, 2, 3))
    cardinality = (pred_valid + targ_valid).sum(dim=(1, 2, 3))

    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return dice


def train_segmentation(
    output_dir: str = r"C:\Ankshit\CVC-MAGNet-V2\runs\s1_pretrained_segmentation",
    target_updates: int = 1000,
    effective_batch_size: int = 4,
    micro_batch_size: int = 2,
    img_size: int = 512,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    bce_pos_weight: float = 10.0,
    dice_weight: float = 1.0,
    seed: int = 42,
    aug_seed: int = 42,
    save_initial_weights: Optional[str] = None,
    load_initial_weights: Optional[str] = None,
    resume_checkpoint: Optional[str] = None,
    hard_negative_manifest: Optional[str] = None,
    cohort_manifest: Optional[str] = None,
    overwrite: bool = False,
):
    out_path = Path(output_dir)
    ckpt_dir = out_path / "checkpoints"
    final_path = ckpt_dir / "final_segmentation_v3.pt"

    # Refuse accidental reuse of completed output directories
    completed_markers = [
        final_path,
        ckpt_dir / "best_segmentation_v3.pt",
        ckpt_dir / "last_segmentation_v3.pt",
        out_path / "train_summary.json",
        out_path / "exposure_ledger.jsonl",
    ]

    if any(p.exists() for p in completed_markers) and not resume_checkpoint and not overwrite:
        existing = [p.name for p in completed_markers if p.exists()]
        raise FileExistsError(
            f"Output directory {output_dir} already contains run artifacts ({existing}). "
            "Refusing to overwrite completed or active run. Specify a new directory, --resume_checkpoint, or --overwrite."
        )

    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    accum_steps = effective_batch_size // micro_batch_size
    if effective_batch_size % micro_batch_size != 0:
        raise ValueError(
            f"effective_batch_size ({effective_batch_size}) must be divisible by micro_batch_size ({micro_batch_size})"
        )

    print(
        f"Optimization: Effective Batch = {effective_batch_size} (Micro-batch = {micro_batch_size}, Accumulation Steps = {accum_steps})",
        flush=True,
    )

    # Fail fast on missing requested initialization weights
    if load_initial_weights:
        if not os.path.exists(load_initial_weights):
            raise FileNotFoundError(f"Requested initial weights file not found: {load_initial_weights}")
        if os.path.getsize(load_initial_weights) == 0:
            raise ValueError(f"Requested initial weights file is empty (0 bytes): {load_initial_weights}")

    # Fail fast on missing requested resume checkpoint
    if resume_checkpoint:
        if not os.path.exists(resume_checkpoint):
            raise FileNotFoundError(f"Requested resume checkpoint not found: {resume_checkpoint}")
        if os.path.getsize(resume_checkpoint) == 0:
            raise ValueError(f"Requested resume checkpoint is empty (0 bytes): {resume_checkpoint}")

    # Load inventories and splits
    data_dir = PROJECT_ROOT / "data"
    inv_df = pd.read_parquet(data_dir / "annotation_inventory.parquet")
    study_inv_df = pd.read_parquet(data_dir / "study_inventory.parquet")
    with open(data_dir / "splits.json") as f:
        splits = json.load(f)

    # Load cohort: from custom/expanded manifest or default splits
    if cohort_manifest:
        if not os.path.exists(cohort_manifest):
            raise FileNotFoundError(f"Requested cohort manifest not found: {cohort_manifest}")
        if cohort_manifest.endswith(".parquet"):
            c_df = pd.read_parquet(cohort_manifest)
            train_study_ids = c_df["study_id"].astype(str).tolist()
            pos_study_ids = c_df[c_df["has_cvc"] == True]["study_id"].astype(str).tolist()
            neg_study_ids = c_df[c_df["has_cvc"] == False]["study_id"].astype(str).tolist()
            cohort_type = f"Parquet Cohort ({Path(cohort_manifest).stem})"
        else:
            with open(cohort_manifest, encoding="utf-8") as f:
                cdata = json.load(f)
            train_study_ids = cdata["all_study_ids"]
            pos_study_ids = cdata["positive_study_ids"]
            neg_study_ids = cdata["negative_study_ids"]
            cohort_type = "Expanded Cohort" if len(train_study_ids) == 1196 else "Custom Cohort"
    else:
        train_study_ids = splits["partitions"]["dev_pilot_train"]["study_ids"]
        pos_set = set(inv_df[inv_df["device_type"] == "CVC"]["study_id"])
        pos_study_ids = [sid for sid in train_study_ids if sid in pos_set]
        neg_study_ids = [sid for sid in train_study_ids if sid not in pos_set]
        cohort_type = "Duration Control Cohort"

    val_study_ids = splits["partitions"]["dev_pilot_val"]["study_ids"]

    # Strict cohort assertions before launch
    if len(train_study_ids) == 796:
        assert len(pos_study_ids) == 400, f"Control cohort must have exactly 400 positives, got {len(pos_study_ids)}"
        assert len(neg_study_ids) == 396, f"Control cohort must have exactly 396 negatives, got {len(neg_study_ids)}"
        print(f"[{cohort_type} ASSERTION PASSED]: Loaded 796 studies (400 annotated positives, 396 confirmed negatives)", flush=True)
    elif len(train_study_ids) == 1196:
        assert len(pos_study_ids) == 800, f"Expanded cohort must have exactly 800 positives, got {len(pos_study_ids)}"
        assert len(neg_study_ids) == 396, f"Expanded cohort must have exactly 396 negatives, got {len(neg_study_ids)}"
        print(f"[{cohort_type} ASSERTION PASSED]: Loaded 1,196 studies (800 annotated positives, 396 confirmed negatives)", flush=True)
    else:
        assert len(pos_study_ids) > 0, "Cohort must have at least one positive study!"
        assert len(neg_study_ids) > 0, "Cohort must have at least one negative study!"
        assert len(pos_study_ids) + len(neg_study_ids) == len(train_study_ids), "Positives and negatives must sum to total studies!"
        print(f"[{cohort_type} ASSERTION PASSED]: Loaded {len(train_study_ids)} studies ({len(pos_study_ids)} positives, {len(neg_study_ids)} negatives)", flush=True)

    val_leak = set(train_study_ids).intersection(set(val_study_ids))
    assert len(val_leak) == 0, f"CRITICAL LEAKAGE: {len(val_leak)} validation studies found in training cohort!"

    images_dir = r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train"

    # Optional hard-negative mining set
    hard_negative_set = None
    if hard_negative_manifest:
        if not os.path.exists(hard_negative_manifest):
            raise FileNotFoundError(f"Requested hard negative manifest not found: {hard_negative_manifest}")
        with open(hard_negative_manifest, encoding="utf-8") as f:
            hn_data = json.load(f)
        records = hn_data.get("records") or hn_data.get("studies") or []
        if records:
            hard_negative_set = {
                r["study_id"]
                for r in records
                if not r.get("excluded_from_hard_negatives", False)
            }
        elif "study_ids" in hn_data:
            hard_negative_set = set(hn_data["study_ids"])
        else:
            hard_negative_set = set()
        print(f"Loaded {len(hard_negative_set)} active hard-negative studies from {hard_negative_manifest}.", flush=True)

    # Training dataloader with dedicated augmentation seed and resumable sampler
    train_loader, train_sampler = create_segmentation_v3_dataloader(
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
        hard_negative_set=hard_negative_set,
    )

    val_loader, _ = create_segmentation_v3_dataloader(
        study_ids=val_study_ids,
        inventory_df=inv_df,
        study_inventory_df=study_inv_df,
        images_dir=images_dir,
        batch_size=micro_batch_size,
        img_size=img_size,
        is_train=False,
        num_workers=0,
    )

    print("Instantiating CVC-Segmentation-V3 Net...", flush=True)
    model = CVCSegmentationV3Net(pretrained=True, freeze_encoder_bn=True).to(device)

    # Initial weights matching across resolution arms
    initial_weights_sha256 = ""
    if load_initial_weights:
        print(f"Loading certified initial weights from {load_initial_weights}...", flush=True)
        init_state = torch.load(load_initial_weights, map_location=device, weights_only=False)
        model.load_state_dict(init_state["model_state_dict"] if "model_state_dict" in init_state else init_state)
        initial_weights_sha256 = compute_sha256(load_initial_weights)
        print(f"Loaded initial weights SHA256: {initial_weights_sha256[:16]}...", flush=True)
    elif save_initial_weights:
        print(f"Saving initialized model weights to {save_initial_weights}...", flush=True)
        tmp_init = Path(save_initial_weights).with_suffix(".tmp")
        torch.save({"model_state_dict": model.state_dict(), "seed": seed}, tmp_init)
        os.replace(tmp_init, save_initial_weights)
        initial_weights_sha256 = compute_sha256(save_initial_weights)
        print(f"Saved initial weights SHA256: {initial_weights_sha256[:16]}...", flush=True)

    # Save resolved run configuration with full provenance
    resolved_config = {
        "output_dir": str(output_dir),
        "target_updates": target_updates,
        "effective_batch_size": effective_batch_size,
        "micro_batch_size": micro_batch_size,
        "accum_steps": accum_steps,
        "img_size": img_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "bce_pos_weight": bce_pos_weight,
        "dice_weight": dice_weight,
        "seed": seed,
        "aug_seed": aug_seed,
        "load_initial_weights": str(load_initial_weights) if load_initial_weights else None,
        "initial_weights_sha256": initial_weights_sha256,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "hard_negative_manifest": str(hard_negative_manifest) if hard_negative_manifest else None,
        "hard_negatives_count": len(hard_negative_set) if hard_negative_set else 0,
        "device": str(device),
        "train_samples_count": len(train_study_ids),
        "val_samples_count": len(val_study_ids),
        "splits_sha256": compute_sha256(str(data_dir / "splits.json")),
        "annotation_inv_sha256": compute_sha256(str(data_dir / "annotation_inventory.parquet")),
        "study_inv_sha256": compute_sha256(str(data_dir / "study_inventory.parquet")),
    }
    config_path = out_path / "resolved_config.json"
    tmp_config = config_path.with_suffix(".tmp")
    with open(tmp_config, "w") as f:
        json.dump(resolved_config, f, indent=2)
    os.replace(tmp_config, config_path)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_steps = min(100, max(1, target_updates // 10))
    warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, target_updates - warmup_steps), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = GradScaler("cuda", enabled=(torch.cuda.is_available() and not use_bf16))
    print(f"Precision Mode: {'bfloat16 (native Ampere, zero-overflow)' if use_bf16 else 'float16 with GradScaler'}", flush=True)

    successful_updates = 0
    skipped_updates = 0
    total_attempts = 0
    consecutive_skips = 0
    max_consecutive_skips = 20
    max_total_attempts = target_updates * 3
    best_val_dice = -1.0
    start_time = time.time()

    # Exact resume capability
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}...", flush=True)
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if not use_bf16 and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        successful_updates = ckpt.get("successful_updates", ckpt.get("update", 0))
        skipped_updates = ckpt.get("skipped_updates", 0)
        total_attempts = ckpt.get("total_attempts", successful_updates)
        best_val_dice = ckpt.get("best_val_dice", -1.0)

        # Restore CPU RNG tensors on CPU
        if "torch_rng_state" in ckpt:
            rng_t = ckpt["torch_rng_state"]
            if isinstance(rng_t, torch.Tensor):
                rng_t = rng_t.cpu()
            torch.set_rng_state(rng_t)

        if "cuda_rng_state" in ckpt and ckpt["cuda_rng_state"] is not None and torch.cuda.is_available():
            cuda_t = ckpt["cuda_rng_state"]
            if isinstance(cuda_t, torch.Tensor):
                cuda_t = cuda_t.cpu()
            torch.cuda.set_rng_state(cuda_t)

        if "numpy_rng_state" in ckpt:
            np.random.set_state(ckpt["numpy_rng_state"])
        if "python_rng_state" in ckpt:
            random.setstate(ckpt["python_rng_state"])

        if "sampler_state_dict" in ckpt and train_sampler is not None:
            train_sampler.load_state_dict(ckpt["sampler_state_dict"])

        if "study_exposure_counts" in ckpt and hasattr(train_loader.dataset, "study_exposure_counts"):
            train_loader.dataset.study_exposure_counts = dict(ckpt["study_exposure_counts"])

        print(
            f"Resumed successfully at update {successful_updates}/{target_updates} "
            f"(Total attempts: {total_attempts}, Skips: {skipped_updates}, Best Val Soft Dice: {best_val_dice:.4f})",
            flush=True,
        )

    pos_weight = torch.tensor([bce_pos_weight], device=device, dtype=torch.float32)

    ledger_path = out_path / "exposure_ledger.jsonl"
    ledger_file = open(ledger_path, "a" if resume_checkpoint else "w", encoding="utf-8")

    print(f"\nStarting Training for {target_updates} successful updates...", flush=True)
    optimizer.zero_grad()

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

            # Global effective-batch denominators for exact mathematical equivalence
            eff_supervision = eff_valids * eff_losses
            total_supervised_pixels = float(eff_supervision.sum().item())
            total_valid_pixels = float(eff_valids.sum().item())
            total_positives = int(eff_is_pos.sum().item())

            # Format per-sample augmentation parameters for ledger
            per_sample_aug = []
            if "aug_params" in eff_batch and eff_batch["aug_params"]:
                ap = eff_batch["aug_params"]
                for i in range(current_eff_size):
                    sample_dict = {}
                    for k in ap:
                        val = ap[k][i]
                        sample_dict[k] = val.item() if isinstance(val, torch.Tensor) else val
                    per_sample_aug.append(sample_dict)

            # Retry loop: if gradients overflow, clear grads, reduce scale, and retry the SAME batch
            batch_retries = 0
            max_batch_retries = 5

            while True:
                total_attempts += 1
                optimizer.zero_grad(set_to_none=True)
                eff_total_loss_accum = 0.0
                eff_bce_accum = 0.0
                eff_dice_accum = 0.0

                # Forward and backward across micro-batches
                num_mbs = (current_eff_size + micro_batch_size - 1) // micro_batch_size
                for m in range(num_mbs):
                    start_i = m * micro_batch_size
                    end_i = min((m + 1) * micro_batch_size, current_eff_size)

                    mb_images = eff_images[start_i:end_i].to(device, non_blocking=True)
                    mb_masks = eff_masks[start_i:end_i].to(device, non_blocking=True)
                    mb_valids = eff_valids[start_i:end_i].to(device, non_blocking=True)
                    mb_supervision = eff_supervision[start_i:end_i].to(device, non_blocking=True)
                    mb_is_pos = eff_is_pos[start_i:end_i].to(device)

                    with autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                        out = model(mb_images)
                        logits_fp32 = out["seg_logits"].float()
                        targets_fp32 = mb_masks.float()
                        supervision_fp32 = mb_supervision.float()

                        # Exact effective BCE term (weighted by supervision mask)
                        bce_raw = nn.functional.binary_cross_entropy_with_logits(
                            logits_fp32, targets_fp32, pos_weight=pos_weight, reduction="none"
                        )
                        mb_bce_loss = (bce_raw * supervision_fp32).sum() / max(total_supervised_pixels, 1.0)

                        # Exact effective Dice term (computed with boundary-aware supervision)
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

                    scaler.scale(mb_total_loss).backward()

                    eff_total_loss_accum += mb_total_loss.item()
                    eff_bce_accum += mb_bce_loss.item()
                    eff_dice_accum += (dice_weight * mb_dice_loss).item()

                # Check gradients for inf / nan
                scaler.unscale_(optimizer)

                has_inf_or_nan = False
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        has_inf_or_nan = True
                        break

                if has_inf_or_nan:
                    skipped_updates += 1
                    consecutive_skips += 1
                    if scaler.is_enabled():
                        scaler.update()         # Backs off scale factor without calling optimizer.step
                    optimizer.zero_grad(set_to_none=True)
                    print(
                        f"Warning: Update attempt {total_attempts} encountered non-finite gradients. "
                        f"Safely skipping batch and advancing to next batch...",
                        flush=True,
                    )
                    if consecutive_skips >= max_consecutive_skips:
                        raise RuntimeError(
                            f"Encountered {max_consecutive_skips} consecutive non-finite gradient attempts. Halting to avoid endless loop."
                        )
                    break

                consecutive_skips = 0

                # Successful update
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                successful_updates += 1

                # Log successful update to exposure ledger
                ledger_entry = {
                    "update": successful_updates,
                    "attempt": total_attempts,
                    "study_ids": eff_study_ids,
                    "is_positive": [int(x) for x in eff_is_pos.tolist()],
                    "total_valid_pixels": total_valid_pixels,
                    "total_positives": total_positives,
                    "aug_params": per_sample_aug,
                }
                ledger_file.write(json.dumps(ledger_entry) + "\n")
                ledger_file.flush()

                if successful_updates % 50 == 0 or successful_updates == target_updates:
                    elapsed = time.time() - start_time
                    vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
                    cur_lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"Update {successful_updates:4d}/{target_updates} (Attempt {total_attempts:4d}, Skips: {skipped_updates:2d}) | "
                        f"Loss: {eff_total_loss_accum:.4f} (BCE: {eff_bce_accum:.4f}, Dice: {eff_dice_accum:.4f}) | "
                        f"LR: {cur_lr:.2e} | VRAM: {vram_mb:.0f} MB | Elapsed: {elapsed:.1f}s",
                        flush=True,
                    )

                    last_payload = {
                        "successful_updates": successful_updates,
                        "skipped_updates": skipped_updates,
                        "total_attempts": total_attempts,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                        "numpy_rng_state": np.random.get_state(),
                        "python_rng_state": random.getstate(),
                        "sampler_state_dict": train_sampler.state_dict() if train_sampler else None,
                        "study_exposure_counts": dict(train_loader.dataset.study_exposure_counts),
                        "best_val_dice": best_val_dice,
                        "initial_weights_sha256": initial_weights_sha256,
                    }
                    last_path = ckpt_dir / "last_segmentation_v3.pt"
                    tmp_last = last_path.with_suffix(".tmp")
                    torch.save(last_payload, tmp_last)
                    os.replace(tmp_last, last_path)

                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if successful_updates % 250 == 0 or successful_updates == target_updates:
                    # Run validation macro soft dice
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

                    ckpt_payload = {
                        "successful_updates": successful_updates,
                        "skipped_updates": skipped_updates,
                        "total_attempts": total_attempts,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                        "numpy_rng_state": np.random.get_state(),
                        "python_rng_state": random.getstate(),
                        "sampler_state_dict": train_sampler.state_dict() if train_sampler else None,
                        "study_exposure_counts": dict(train_loader.dataset.study_exposure_counts),
                        "val_soft_dice": mean_val_dice,
                        "best_val_dice": max(best_val_dice, mean_val_dice),
                        "initial_weights_sha256": initial_weights_sha256,
                    }

                    if mean_val_dice > best_val_dice:
                        best_val_dice = mean_val_dice
                        ckpt_payload["best_val_dice"] = best_val_dice
                        best_path = ckpt_dir / "best_segmentation_v3.pt"
                        tmp_best = best_path.with_suffix(".tmp")
                        torch.save(ckpt_payload, tmp_best)
                        os.replace(tmp_best, best_path)
                        print(f"  -> Saved new best model atomically to {best_path}", flush=True)

                    import gc
                    gc.collect()

                # Successful update completed: break from retry loop to advance to next effective batch
                break

            # Explicit batch cleanup to prevent memory fragmentation on long runs
            del eff_images, eff_masks, eff_valids, eff_is_pos, eff_study_ids, eff_batch

            if successful_updates >= target_updates:
                break

            if total_attempts >= max_total_attempts and successful_updates < target_updates:
                raise RuntimeError(
                    f"Exceeded max total attempts ({total_attempts} >= {max_total_attempts}). Halting training."
                )

    ledger_file.close()

    # Save final checkpoint atomically
    final_payload = {
        "successful_updates": successful_updates,
        "skipped_updates": skipped_updates,
        "total_attempts": total_attempts,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "sampler_state_dict": train_sampler.state_dict() if train_sampler else None,
        "study_exposure_counts": dict(train_loader.dataset.study_exposure_counts),
        "final_val_dice": mean_val_dice if "mean_val_dice" in locals() else -1.0,
        "best_val_dice": best_val_dice,
        "initial_weights_sha256": initial_weights_sha256,
    }
    tmp_final = final_path.with_suffix(".tmp")
    torch.save(final_payload, tmp_final)
    os.replace(tmp_final, final_path)
    print(
        f"\nTraining Complete ({successful_updates} successful updates in {total_attempts} attempts, {skipped_updates} skips). "
        f"Saved final model atomically to {final_path}",
        flush=True,
    )

    # Save run summary
    summary_path = out_path / "train_summary.json"
    tmp_sum = summary_path.with_suffix(".tmp")
    with open(tmp_sum, "w") as f:
        json.dump(
            {
                "output_dir": str(output_dir),
                "cohort_type": cohort_type,
                "cohort_manifest": str(cohort_manifest) if cohort_manifest else None,
                "total_train_studies": len(train_study_ids),
                "positive_studies_count": len(pos_study_ids),
                "negative_studies_count": len(neg_study_ids),
                "hard_negative_count": len(hard_negative_set) if hard_negative_set else 0,
                "successful_updates": successful_updates,
                "skipped_updates": skipped_updates,
                "total_attempts": total_attempts,
                "best_val_dice": best_val_dice,
                "initial_weights_sha256": initial_weights_sha256,
                "final_checkpoint": str(final_path),
                "elapsed_seconds": time.time() - start_time,
                "img_size": img_size,
                "effective_batch_size": effective_batch_size,
                "micro_batch_size": micro_batch_size,
                "seed": seed,
                "aug_seed": aug_seed,
            },
            f,
            indent=2,
        )
    os.replace(tmp_sum, summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CVC-Segmentation-V3 with Strict Equivalence")
    parser.add_argument("--output_dir", type=str, default=r"C:\Ankshit\CVC-MAGNet-V2\runs\s1_pretrained_segmentation")
    parser.add_argument("--target_updates", type=int, default=1000)
    parser.add_argument("--effective_batch_size", type=int, default=4)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=None, help="Alias for micro_batch_size")
    parser.add_argument("--accum_steps", type=int, default=None, help="Accumulation steps multiplier")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--bce_pos_weight", type=float, default=10.0)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aug_seed", type=int, default=42)
    parser.add_argument("--save_initial_weights", type=str, default=None)
    parser.add_argument("--load_initial_weights", type=str, default=None)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--hard_negative_manifest", type=str, default=None, help="Path to hard negatives manifest JSON")
    parser.add_argument("--cohort_manifest", type=str, default=None, help="Path to training cohort manifest JSON")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting completed output directory")
    args = parser.parse_args()

    micro_bs = args.batch_size if args.batch_size is not None else args.micro_batch_size
    eff_bs = args.effective_batch_size
    if args.accum_steps is not None and args.batch_size is not None:
        eff_bs = args.batch_size * args.accum_steps

    train_segmentation(
        output_dir=args.output_dir,
        target_updates=args.target_updates,
        effective_batch_size=eff_bs,
        micro_batch_size=micro_bs,
        img_size=args.img_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        bce_pos_weight=args.bce_pos_weight,
        dice_weight=args.dice_weight,
        seed=args.seed,
        aug_seed=args.aug_seed,
        save_initial_weights=args.save_initial_weights,
        load_initial_weights=args.load_initial_weights,
        resume_checkpoint=args.resume_checkpoint,
        hard_negative_manifest=args.hard_negative_manifest,
        cohort_manifest=args.cohort_manifest,
        overwrite=args.overwrite,
    )
