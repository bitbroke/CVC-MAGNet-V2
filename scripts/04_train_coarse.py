"""Script 04: Stage 1 Coarse Localization Pilot Training (Experiment A).

Contract:
- Single-channel 512x512 letterbox radiograph input with CLAHE.
- Target: binary segmentation mask (polyline dilated to 10px empirical stroke width).
- Model: CoarseSegNet (base-32 U-Net encoder-decoder + Pyramid Pooling Module).
- Loss: BCE + Dice with pos_weight=10.0 to counteract extreme background pixel imbalance.
- Optimizer: AdamW (lr=1e-4, wd=1e-4), CosineAnnealingLR (T_max=1000, eta_min=1e-6).
- Budget: Exactly 1,000 optimizer steps at batch size 4 (4,000 training examples exposed).
- Hardened checkpointing with atomic save, full RNG/scheduler/sampler state.
- Validation at steps 200, 400, 600, 800, 1000 on the frozen dev_pilot_val cohort (193 studies, 248 targets).
- Standardized evaluation with threshold-constrained Hungarian assignment against frozen V1 baseline.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import CoarseDataset
from src.infer.pipeline import MAGNetInferencePipeline
from src.models.checkpoint import save_checkpoint
from src.models.coarse import CoarseSegNet
from src.models.losses import CoarseLoss
import importlib.util

spec = importlib.util.spec_from_file_location("eval_mod", PROJECT_ROOT / "scripts" / "06_evaluate.py")
eval_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_mod)
evaluate_predictions = eval_mod.evaluate_predictions


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file for provenance."""
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]


def set_seed(seed: int = 42):
    """Set deterministic seeds across PyTorch, NumPy, and Python."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_coarse_validation(
    coarse_model: CoarseSegNet,
    val_dataset: CoarseDataset,
    device: torch.device,
    criterion: CoarseLoss,
    targets_path: Path,
    study_inventory_path: Path,
    step: int,
    reports_dir: Path,
    coarse_threshold: float = 0.3,
) -> Dict[str, Any]:
    """Run validation pass on the frozen 193-study validation cohort."""
    coarse_model.eval()
    val_losses = []
    val_bce_losses = []
    val_dice_losses = []
    val_mask_dices = []

    pipeline = MAGNetInferencePipeline(
        coarse_model=coarse_model,
        fine_model=None,
        device=device,
        coarse_threshold=coarse_threshold,
        fine_threshold=0.5,
    )

    predictions_list = []
    sample_overlays = []

    print(f"\n[Validation Step {step}] Evaluating {len(val_dataset)} validation studies...")
    t_val_start = time.time()

    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    with torch.no_grad():
        for batch in val_loader:
            x = batch["image"].to(device)
            y = batch["mask"].to(device)
            with torch.amp.autocast("cuda"):
                logits = coarse_model(x)
                loss, loss_dict = criterion(logits, y)

            val_losses.append(loss.item())
            val_bce_losses.append(loss_dict["loss_bce"])
            val_dice_losses.append(loss_dict["loss_dice"])

            # Compute pixel-level mask Dice on batch
            probs = torch.sigmoid(logits)
            pred_fg = (probs > coarse_threshold).float()
            gt_fg = (y > 0.5).float()
            for b in range(x.shape[0]):
                p_b = pred_fg[b].view(-1)
                g_b = gt_fg[b].view(-1)
                inter = (p_b * g_b).sum().item()
                denom = p_b.sum().item() + g_b.sum().item()
                if denom > 0:
                    d = (2.0 * inter) / denom
                else:
                    d = 1.0 if inter == 0 else 0.0
                val_mask_dices.append(d)

    # Run distal-tip candidate extraction and coordinate reconstruction via pipeline
    for idx in range(len(val_dataset)):
        item = val_dataset[idx]
        sid = item["study_id"]
        # Pass native image file path to pipeline
        img_path = val_dataset.images_dir / f"{sid}.jpg"
        pred_res = pipeline.predict_image(img_path, study_id=sid)
        predictions_list.append(pred_res)

        # Collect 4 sample overlays for diagnostic report
        if len(sample_overlays) < 4 and item["mask"].sum().item() > 0:
            lb_img = item["image"][0].numpy()
            gt_mask = item["mask"][0].numpy()
            with torch.no_grad():
                prob_map = torch.sigmoid(coarse_model(item["image"].unsqueeze(0).to(device)))[0, 0].cpu().numpy()
            sample_overlays.append((sid, lb_img, gt_mask, prob_map))

    val_time = time.time() - t_val_start
    mean_val_loss = float(np.mean(val_losses))
    mean_val_bce = float(np.mean(val_bce_losses))
    mean_val_dice_loss = float(np.mean(val_dice_losses))
    mean_mask_dice = float(np.mean(val_mask_dices))

    # Save temporary predictions JSON and evaluate
    temp_preds_path = reports_dir / f"temp_val_preds_step_{step}.json"
    with open(temp_preds_path, "w") as f:
        json.dump({"predictions": predictions_list}, f)

    eval_results = evaluate_predictions(
        predictions_path=temp_preds_path,
        targets_path=targets_path,
        study_inventory_path=study_inventory_path,
        threshold=0.50,
    )

    # Save overlay diagnostic figure
    if len(sample_overlays) > 0:
        fig, axes = plt.subplots(len(sample_overlays), 4, figsize=(16, 4 * len(sample_overlays)))
        if len(sample_overlays) == 1:
            axes = np.expand_dims(axes, 0)
        for i, (sid, img_np, gt_np, pr_np) in enumerate(sample_overlays):
            axes[i, 0].imshow(img_np, cmap="gray")
            axes[i, 0].set_title(f"Val Radiograph: {sid[:10]}...")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(img_np, cmap="gray")
            axes[i, 1].imshow(gt_np, cmap="Reds", alpha=0.55)
            axes[i, 1].set_title(f"GT Mask ({int(gt_np.sum())} px)")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(pr_np, cmap="magma")
            axes[i, 2].set_title(f"Coarse Prob (Max={pr_np.max():.2f})")
            axes[i, 2].axis("off")

            axes[i, 3].imshow(img_np, cmap="gray")
            axes[i, 3].imshow(pr_np > coarse_threshold, cmap="Greens", alpha=0.55)
            axes[i, 3].set_title(f"Coarse Pred (tau={coarse_threshold})")
            axes[i, 3].axis("off")

        plt.tight_layout()
        overlay_fig_path = reports_dir / f"pilot_coarse_repaired_val_step_{step}.png"
        plt.savefig(overlay_fig_path, dpi=120)
        plt.close()

    metrics = {
        "step": step,
        "val_loss": mean_val_loss,
        "val_bce_loss": mean_val_bce,
        "val_dice_loss": mean_val_dice_loss,
        "val_mask_dice": mean_mask_dice,
        "val_time_s": val_time,
        "all_target_success_at_15": eval_results["primary_metric"]["matched"],
        "all_target_success_at_15_rate": eval_results["primary_metric"]["rate_pct"] / 100.0,
        "proposal_coverage": eval_results["proposal_diagnostics"]["tip_in_crop_coverage"]["covered_targets"],
        "proposal_coverage_rate": eval_results["proposal_diagnostics"]["tip_in_crop_coverage"]["rate_pct"] / 100.0,
        "ungated_candidate_success_at_15": eval_results["proposal_diagnostics"]["ungated_candidate_success_at_15"]["matched_targets"],
        "ungated_candidate_success_at_15_rate": eval_results["proposal_diagnostics"]["ungated_candidate_success_at_15"]["rate_pct"] / 100.0,
        "median_localization_error_px": eval_results["localization_errors_px"]["matched_median_error_px"],
        "p95_localization_error_px": eval_results["localization_errors_px"]["matched_p95_error_px"],
        "all_target_nearest_error_p95_px": eval_results["localization_errors_px"]["all_target_nearest_error_p95_px"],
        "negative_acceptance_rate": eval_results["negative_acceptance"]["rate_pct"],
    }

    print(
        f"  [Validation Step {step}] Mask Dice: {mean_mask_dice:.4f} | "
        f"All-Target Success@15: {metrics['all_target_success_at_15']}/248 ({metrics['all_target_success_at_15_rate']*100:.2f}%) | "
        f"Crop Coverage: {metrics['proposal_coverage']}/248 ({metrics['proposal_coverage_rate']*100:.2f}%) | "
        f"Val Time: {val_time:.1f}s"
    )

    return metrics, predictions_list


def main():
    parser = argparse.ArgumentParser(description="Train Stage 1 Coarse Localization")
    parser.add_argument("--pos_weight", type=float, default=None, help="BCE positive pixel loss weight")
    parser.add_argument("--run_name", type=str, default="pilot_coarse_repaired", help="Run output subdirectory name")
    args = parser.parse_args()

    run_name = args.run_name
    print("=" * 75)
    print(f"EXPERIMENT A: CVC-MAGNet-V2 STAGE 1 COARSE PILOT - RUN: {run_name}")
    print("=" * 75)
    t_pilot_start = time.time()

    # 1. Load configuration and check integrity
    config_path = PROJECT_ROOT / "configs" / "pilot.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    pilot_cfg = cfg["training"]["pilot"]["coarse"]
    losses_cfg = cfg["losses"]["coarse"]

    seed = cfg["training"]["seed"]
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"CUDA Device: {torch.cuda.get_device_name(device)}")

    # Paths
    data_dir = PROJECT_ROOT / "data"
    splits_path = data_dir / "splits.json"
    inv_path = data_dir / "annotation_inventory.parquet"
    study_inv_path = data_dir / "study_inventory.parquet"
    val_targets_path = data_dir / "splits" / "dev_pilot_val_targets.csv"
    raw_images_dir = Path(cfg["data"]["raw_dir"]) / "train"

    # Provenance tracking
    input_hashes = {
        "config_hash": compute_file_hash(config_path),
        "splits_hash": compute_file_hash(splits_path),
        "inventory_hash": compute_file_hash(inv_path),
    }
    print(f"Input Hashes: {input_hashes}")

    # Output directories
    runs_dir = PROJECT_ROOT / "runs" / run_name
    ckpt_dir = runs_dir / "checkpoints"
    reports_dir = PROJECT_ROOT / "reports" / "overlays"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 2. Build Datasets & Dataloader
    with open(splits_path, "r") as f:
        splits = json.load(f)

    train_study_ids = splits["partitions"]["dev_pilot_train"]["study_ids"]
    val_study_ids = splits["partitions"]["dev_pilot_val"]["study_ids"]
    inv_df = pd.read_parquet(inv_path)
    study_df = pd.read_parquet(study_inv_path)

    confirmed_neg_ids = set(
        study_df[study_df["is_study_level_negative"] & (~study_df["has_cvc"])]["StudyInstanceUID"]
    )
    n_train_pos = len([s for s in train_study_ids if s not in confirmed_neg_ids])
    n_train_neg = len([s for s in train_study_ids if s in confirmed_neg_ids])

    print(f"\nTraining set:   {len(train_study_ids)} studies ({n_train_pos} annotation-supported pos + {n_train_neg} label-supported neg)")
    print(f"Validation set: {len(val_study_ids)} studies (248 assessable CVC targets)")

    train_dataset = CoarseDataset(
        study_ids=train_study_ids,
        inventory_df=inv_df,
        images_dir=raw_images_dir,
        target_size=(512, 512),
        mask_stroke_width=int(cfg["preprocessing"]["coarse"]["mask_stroke_width_native"]),
        clahe_enabled=True,
        confirmed_negative_study_ids=confirmed_neg_ids,
    )
    val_dataset = CoarseDataset(
        study_ids=val_study_ids,
        inventory_df=inv_df,
        images_dir=raw_images_dir,
        target_size=(512, 512),
        mask_stroke_width=int(cfg["preprocessing"]["coarse"]["mask_stroke_width_native"]),
        clahe_enabled=True,
        confirmed_negative_study_ids=confirmed_neg_ids,
    )

    batch_size = pilot_cfg["batch_size"]
    max_steps = pilot_cfg["max_steps"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    # 3. Model, Loss, Optimizer & Scheduler
    model = CoarseSegNet(in_channels=1, out_channels=1, base_channels=32).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CoarseSegNet initialized: {total_params:,} trainable parameters.")

    if args.pos_weight is not None:
        pos_weight_val = float(args.pos_weight)
    else:
        pos_weight_val = float(losses_cfg.get("pos_weight", 10.0))
    cfg["losses"]["coarse"]["pos_weight"] = pos_weight_val
    print(f"Coarse Loss pos_weight (resolved): {pos_weight_val:.1f}")
    pos_weight = torch.tensor([pos_weight_val], device=device)
    criterion = CoarseLoss(
        bce_weight=float(losses_cfg["bce_weight"]),
        dice_weight=float(losses_cfg["dice_weight"]),
        pos_weight=pos_weight,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(pilot_cfg["lr"]),
        weight_decay=float(pilot_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_steps,
        eta_min=float(pilot_cfg["min_lr"]),
    )
    scaler = torch.amp.GradScaler("cuda")

    # 4. Training Loop (Step-controlled)
    print("\n" + "-" * 75)
    print(f"Beginning Step-Controlled Training: {max_steps} optimizer updates (Batch {batch_size})")
    print("-" * 75)

    history: List[Dict[str, Any]] = []
    val_history: List[Dict[str, Any]] = []
    best_metric = -1.0
    best_step = -1
    best_preds_data = None

    step = 0
    epoch = 0
    running_loss = 0.0
    running_bce = 0.0
    running_dice = 0.0
    t_step_start = time.time()

    val_interval = 200  # Validate at steps 200, 400, 600, 800, 1000

    while step < max_steps:
        epoch += 1
        model.train()
        for batch in train_loader:
            if step >= max_steps:
                break

            step += 1
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss, loss_dict = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            running_bce += loss_dict["loss_bce"]
            running_dice += loss_dict["loss_dice"]

            if step % 25 == 0 or step == 1:
                cur_lr = scheduler.get_last_lr()[0]
                n_avg = 25 if step > 1 else 1
                avg_l = running_loss / n_avg
                avg_bce = running_bce / n_avg
                avg_dice = running_dice / n_avg
                running_loss, running_bce, running_dice = 0.0, 0.0, 0.0
                step_ms = (time.time() - t_step_start) / n_avg * 1000.0
                t_step_start = time.time()
                vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

                print(
                    f"Step {step:04d}/{max_steps} (Epoch {epoch}) | "
                    f"Loss: {avg_l:.4f} (BCE: {avg_bce:.4f}, Dice: {avg_dice:.4f}) | "
                    f"LR: {cur_lr:.2e} | VRAM: {vram_mb:.0f} MB | {step_ms:.0f} ms/step"
                )

                history.append({
                    "step": step,
                    "epoch": epoch,
                    "loss": avg_l,
                    "loss_bce": avg_bce,
                    "loss_dice": avg_dice,
                    "lr": cur_lr,
                    "vram_mb": vram_mb,
                })

            # Validation checkpoint
            if step % val_interval == 0 or step == max_steps:
                val_metrics, val_preds = run_coarse_validation(
                    coarse_model=model,
                    val_dataset=val_dataset,
                    device=device,
                    criterion=criterion,
                    targets_path=val_targets_path,
                    study_inventory_path=study_inv_path,
                    step=step,
                    reports_dir=reports_dir,
                    coarse_threshold=0.3,
                )
                val_history.append(val_metrics)

                target_metric = val_metrics["all_target_success_at_15"]
                if len(val_history) == 1:
                    is_best = True
                elif target_metric > best_metric:
                    is_best = True
                elif target_metric == best_metric and val_metrics["val_mask_dice"] > val_history[-2]["val_mask_dice"]:
                    is_best = True
                else:
                    is_best = False

                if is_best:
                    best_metric = target_metric
                    best_step = step
                    best_preds_data = val_preds
                    print(f"  >>> New Best Model at Step {step}: Success@15 = {best_metric}/248 ({val_metrics['all_target_success_at_15_rate']*100:.2f}%)!")

                    save_checkpoint(
                        path=ckpt_dir / "best_coarse.pt",
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        scheduler=scheduler,
                        step=step,
                        config=cfg,
                        input_hashes=input_hashes,
                        metrics=val_metrics,
                        is_best=True,
                    )
                    # Save best predictions JSON
                    with open(runs_dir / "val_predictions_best.json", "w") as f:
                        json.dump({"predictions": val_preds}, f, indent=2)

                model.train()

    # 5. Final Checkpoint & History Persistence
    print("\n" + "=" * 75)
    print("TRAINING COMPLETE: Saving Final Checkpoint & History")
    print("=" * 75)
    save_checkpoint(
        path=ckpt_dir / "final_coarse.pt",
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        step=max_steps,
        config=cfg,
        input_hashes=input_hashes,
        metrics=val_history[-1] if val_history else {},
        is_best=False,
    )

    history_payload = {
        "config": cfg,
        "input_hashes": input_hashes,
        "train_steps": history,
        "val_evaluations": val_history,
        "best_step": best_step,
        "best_metric": best_metric,
        "total_time_seconds": time.time() - t_pilot_start,
    }
    with open(runs_dir / "history.json", "w") as f:
        json.dump(history_payload, f, indent=2)

    print(f"Saved complete training history to {runs_dir / 'history.json'}")
    print(f"Best checkpoint at Step {best_step} with Success@15 = {best_metric}/248.")
    print(f"Total Experiment Time: {(time.time() - t_pilot_start) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
