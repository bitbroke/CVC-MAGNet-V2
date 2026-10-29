"""Train Image-Only CVC Placement Baseline (EfficientNet-B0).

Guarantees & Constraints:
1. Multi-label Formulation: Independent sigmoid logits for Normal, Borderline, Abnormal using BCEWithLogitsLoss.
2. Patient Isolation: Evaluates strictly on 100% unexposed validation partition (2,558 studies).
3. Safety: Native bfloat16 AMP, zero optimizer steps on nonfinite gradients.
4. Hardware Optimization: Resolution 512x512, micro-batch 8, 4 accumulation steps (effective batch 32).
5. Comprehensive Reporting: AUROC, AUPRC, sensitivity, specificity, calibration, single-vs-multi CVC breakdown.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.placement_dataset import create_placement_dataloader
from src.models.placement_baseline import CVCPlacementBaselineNet


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_sha256(filepath: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Find threshold maximizing F1 score on validation set."""
    best_thresh = 0.5
    best_f1 = 0.0
    for t in np.linspace(0.1, 0.9, 81):
        y_pred = (y_prob >= t).astype(int)
        f = f1_score(y_true, y_pred, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thresh = float(t)
    return best_thresh, best_f1


@torch.no_grad()
def evaluate_placement_model(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    thresholds: Optional[List[float]] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate placement model on validation set."""
    model.eval()
    all_probs = []
    all_targets = []
    all_study_ids = []
    all_is_multi_cvc = []

    total_batches = len(val_loader) if max_batches is None else min(len(val_loader), max_batches)
    for batch_idx, batch in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["labels"].numpy()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32):
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().to(torch.float32).numpy()

        all_probs.append(probs)
        all_targets.append(targets)
        all_study_ids.extend(batch["study_id"])
        all_is_multi_cvc.extend(batch["is_multi_cvc"].numpy())

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
            print(f"  [Val Eval] Processed batch {batch_idx + 1}/{total_batches}...", flush=True)

    y_prob = np.concatenate(all_probs, axis=0)  # [N, 3]
    y_true = np.concatenate(all_targets, axis=0)  # [N, 3]
    is_multi_cvc = np.array(all_is_multi_cvc, dtype=bool)

    class_names = ["Normal", "Borderline", "Abnormal"]
    metrics: Dict[str, Any] = {"per_class": {}, "macro": {}, "subgroups": {}}

    # Per-class AUROC, AUPRC, and threshold selection
    optimal_thresholds = []
    aurocs = []
    auprcs = []
    brier_scores = []

    for i, cname in enumerate(class_names):
        yt = y_true[:, i]
        yp = y_prob[:, i]

        # AUROC & AUPRC
        auc = float(roc_auc_score(yt, yp)) if len(np.unique(yt)) > 1 else 0.5
        prc = float(average_precision_score(yt, yp)) if len(np.unique(yt)) > 1 else 0.0
        brier = float(brier_score_loss(yt, yp))

        aurocs.append(auc)
        auprcs.append(prc)
        brier_scores.append(brier)

        # Threshold selection
        if thresholds is None:
            opt_thresh, opt_f1 = find_optimal_threshold(yt, yp)
        else:
            opt_thresh = thresholds[i]
            opt_f1 = f1_score(yt, (yp >= opt_thresh).astype(int), zero_division=0)
        optimal_thresholds.append(opt_thresh)

        y_pred = (yp >= opt_thresh).astype(int)
        sens = float(recall_score(yt, y_pred, zero_division=0))
        spec = float(((y_pred == 0) & (yt == 0)).sum() / max((yt == 0).sum(), 1))
        prec = float(precision_score(yt, y_pred, zero_division=0))

        metrics["per_class"][cname] = {
            "auroc": round(auc, 4),
            "auprc": round(prc, 4),
            "brier_score": round(brier, 4),
            "optimal_threshold": round(opt_thresh, 3),
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "precision": round(prec, 4),
            "f1_score": round(opt_f1, 4),
            "support_positive": int(yt.sum()),
            "support_total": len(yt),
        }

    metrics["macro"] = {
        "macro_auroc": round(float(np.mean(aurocs)), 4),
        "macro_auprc": round(float(np.mean(auprcs)), 4),
        "macro_brier": round(float(np.mean(brier_scores)), 4),
        "selected_thresholds": [round(t, 3) for t in optimal_thresholds],
    }

    # Subgroups: Single-CVC vs Multi-CVC breakdown
    for sub_name, mask in [("single_cvc", ~is_multi_cvc), ("multi_cvc", is_multi_cvc)]:
        if mask.sum() > 0:
            sub_aurocs = []
            sub_class_metrics = {}
            for i, cname in enumerate(class_names):
                yt_s = y_true[mask, i]
                yp_s = y_prob[mask, i]
                auc_s = float(roc_auc_score(yt_s, yp_s)) if len(np.unique(yt_s)) > 1 else 0.5
                sub_aurocs.append(auc_s)
                sub_class_metrics[cname] = {
                    "auroc": round(auc_s, 4),
                    "positive_count": int(yt_s.sum()),
                }
            metrics["subgroups"][sub_name] = {
                "study_count": int(mask.sum()),
                "macro_auroc": round(float(np.mean(sub_aurocs)), 4),
                "per_class": sub_class_metrics,
            }

    return metrics


def train_placement(
    output_dir: str = r"C:\Ankshit\CVC-MAGNet-V2\runs\placement_image_baseline_512",
    target_updates: int = 2500,
    eval_interval: int = 250,
    print_interval: int = 10,
    val_max_batches: Optional[int] = None,
    micro_batch_size: int = 8,
    effective_batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
    img_size: int = 512,
    num_workers: int = 0,
    manifest_path: Optional[str] = None,
    backbone_name: str = "tf_efficientnet_b0.ns_jft_in1k",
):
    out_path = Path(output_dir)
    ckpt_dir = out_path / "checkpoints"
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # 1. Load manifest
    manifest_path = Path(manifest_path) if manifest_path else (PROJECT_ROOT / "data" / "splits" / "placement_locked_manifest.parquet")
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing locked manifest at {manifest_path}")

    df_manifest = pd.read_parquet(manifest_path)
    df_train = df_manifest[df_manifest["split"] == "train"].copy()
    df_val = df_manifest[df_manifest["split"] == "val"].copy()

    print(f"Dataset split: Train = {len(df_train):,} studies, Val = {len(df_val):,} studies (100% UNEXPOSED).", flush=True)

    # 2. Build DataLoaders
    train_loader = create_placement_dataloader(
        df=df_train,
        target_size=(img_size, img_size),
        batch_size=micro_batch_size,
        is_train=True,
        aug_seed=seed,
        num_workers=num_workers,
    )
    val_loader = create_placement_dataloader(
        df=df_val,
        target_size=(img_size, img_size),
        batch_size=micro_batch_size * 2,
        is_train=False,
        num_workers=num_workers,
    )

    # 3. Model, Loss, Optimizer
    model = CVCPlacementBaselineNet(backbone_name=backbone_name, pretrained=True, num_classes=3, dropout_rate=0.25).to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=target_updates, eta_min=1e-6)

    accum_steps = effective_batch_size // micro_batch_size
    print(f"Training Config: Effective Batch = {effective_batch_size} (Micro = {micro_batch_size}, Accum = {accum_steps})", flush=True)

    # Checkpoint tracking
    best_macro_auc = -1.0
    best_metrics = {}
    train_start_time = time.time()
    update_count = 0
    running_loss = 0.0
    loss_history = []
    eval_history = []

    # Infinite data iterator
    def train_generator():
        while True:
            for b in train_loader:
                yield b

    train_iter = train_generator()
    optimizer.zero_grad(set_to_none=True)

    print(f"Starting training for {target_updates} updates. Validation every {eval_interval} updates...", flush=True)

    while update_count < target_updates:
        # Micro-batch accumulation loop
        accum_loss = 0.0
        for _ in range(accum_steps):
            batch = next(train_iter)
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32):
                logits = model(images)
                loss = loss_fn(logits, labels)
                loss_scaled = loss / accum_steps

            loss_scaled.backward()
            accum_loss += loss.item() / accum_steps

        # Check for non-finite gradients
        has_nonfinite = False
        for param in model.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                has_nonfinite = True
                break

        if has_nonfinite:
            print(f"[WARNING] Non-finite gradient detected at update {update_count+1}. Zeroing grad and skipping step.", flush=True)
            optimizer.zero_grad(set_to_none=True)
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        update_count += 1
        running_loss += accum_loss
        loss_history.append({"update": update_count, "loss": round(accum_loss, 4)})

        if update_count % print_interval == 0 or update_count == 1:
            avg_loss = running_loss / max(update_count % print_interval if update_count % print_interval != 0 else print_interval, 1)
            cur_lr = scheduler.get_last_lr()[0]
            print(f"Update {update_count:4d}/{target_updates} | Loss: {accum_loss:.4f} (avg: {avg_loss:.4f}) | LR: {cur_lr:.2e}", flush=True)
            running_loss = 0.0

        # Periodic validation
        if update_count % eval_interval == 0 or update_count == target_updates:
            val_start = time.time()
            print(f"\n--- Starting Validation @ Update {update_count} ---", flush=True)
            metrics = evaluate_placement_model(model, val_loader, device, max_batches=val_max_batches)
            val_dur = time.time() - val_start

            macro_auc = metrics["macro"]["macro_auroc"]
            macro_prc = metrics["macro"]["macro_auprc"]
            norm_auc = metrics["per_class"]["Normal"]["auroc"]
            bord_auc = metrics["per_class"]["Borderline"]["auroc"]
            abn_auc = metrics["per_class"]["Abnormal"]["auroc"]

            print(
                f"\n--- Validation @ Update {update_count} (Took {val_dur:.1f}s) ---\n"
                f"Macro AUROC: {macro_auc:.4f} | Macro AUPRC: {macro_prc:.4f}\n"
                f"Normal AUROC: {norm_auc:.4f} | Borderline AUROC: {bord_auc:.4f} | Abnormal AUROC: {abn_auc:.4f}\n"
                f"Selected Thresholds: {metrics['macro']['selected_thresholds']}\n",
                flush=True,
            )

            eval_history.append({"update": update_count, "metrics": metrics})

            # Save best checkpoint
            if macro_auc > best_macro_auc:
                best_macro_auc = macro_auc
                best_metrics = metrics
                best_ckpt_path = ckpt_dir / "best_placement_baseline.pt"
                torch.save(
                    {
                        "update": update_count,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_macro_auc": best_macro_auc,
                        "metrics": metrics,
                    },
                    best_ckpt_path,
                )
                print(f"--> Saved NEW BEST checkpoint to {best_ckpt_path} (Macro AUROC: {best_macro_auc:.4f})", flush=True)

    # Save final checkpoint
    final_ckpt_path = ckpt_dir / "final_placement_baseline.pt"
    torch.save(
        {
            "update": update_count,
            "model_state_dict": model.state_dict(),
            "final_metrics": metrics,
            "best_macro_auc": best_macro_auc,
        },
        final_ckpt_path,
    )

    total_time = time.time() - train_start_time
    summary = {
        "run_name": "placement_image_baseline_512",
        "model": backbone_name,
        "manifest_path": str(manifest_path),
        "resolution": img_size,
        "target_updates": target_updates,
        "effective_batch_size": effective_batch_size,
        "micro_batch_size": micro_batch_size,
        "total_train_time_sec": round(total_time, 1),
        "manifest_sha256": compute_sha256(manifest_path),
        "best_checkpoint": {
            "path": str(ckpt_dir / "best_placement_baseline.pt"),
            "sha256": compute_sha256(ckpt_dir / "best_placement_baseline.pt"),
            "macro_auroc": best_macro_auc,
            "metrics": best_metrics,
        },
        "eval_history": eval_history,
    }

    summary_path = out_path / "placement_baseline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote placement baseline summary to {summary_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=r"C:\Ankshit\CVC-MAGNet-V2\runs\placement_image_baseline_512")
    parser.add_argument("--updates", type=int, default=2500)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--print_interval", type=int, default=10)
    parser.add_argument("--val_max_batches", type=int, default=None)
    parser.add_argument("--micro_batch", type=int, default=8)
    parser.add_argument("--effective_batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--manifest", type=str, default=None, help="Optional cohort manifest; defaults to the locked full manifest.")
    parser.add_argument("--backbone", type=str, default="tf_efficientnet_b0.ns_jft_in1k")
    args = parser.parse_args()

    train_placement(
        output_dir=args.output_dir,
        target_updates=args.updates,
        eval_interval=args.eval_interval,
        print_interval=args.print_interval,
        val_max_batches=args.val_max_batches,
        micro_batch_size=args.micro_batch,
        effective_batch_size=args.effective_batch,
        lr=args.lr,
        num_workers=args.num_workers,
        manifest_path=args.manifest,
        backbone_name=args.backbone,
    )
