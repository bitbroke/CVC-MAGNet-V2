"""Paper-inspired, single-annotated-CVC placement classification pilot.

Uses only RANZCR image-level labels that agree with the single CVC trace label.
The locked test rows are never loaded by this trainer. This is an image-level
Normal/Borderline/Abnormal task, not a replication of a carina-distance study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.placement_dataset import create_placement_dataloader
from src.models.placement_baseline import CVCPlacementBaselineNet

LABELS = ["CVC - Normal", "CVC - Borderline", "CVC - Abnormal"]
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


@torch.no_grad()
def evaluate(model: nn.Module, loader: Any, device: torch.device) -> dict[str, Any]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["labels"].argmax(dim=1).numpy()
        ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if device.type == "cuda" else nullcontext()
        with ctx:
            logits = model(x)
        all_logits.append(logits.float().cpu().numpy())
        all_y.append(y)

    logits = np.concatenate(all_logits)
    y_true = np.concatenate(all_y).astype(np.int64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    y_pred = probs.argmax(axis=1)
    per_class: dict[str, Any] = {}
    aucs: list[float] = []
    aps: list[float] = []
    for i, name in enumerate(LABELS):
        binary = (y_true == i).astype(np.uint8)
        auc = float(roc_auc_score(binary, probs[:, i])) if np.unique(binary).size == 2 else None
        ap = float(average_precision_score(binary, probs[:, i])) if binary.sum() else None
        if auc is not None:
            aucs.append(auc)
        if ap is not None:
            aps.append(ap)
        per_class[name] = {
            "support": int(binary.sum()),
            "auroc_ovr": auc,
            "auprc_ovr": ap,
            "precision": float(precision_score(binary, y_pred == i, zero_division=0)),
            "sensitivity": float(recall_score(binary, y_pred == i, zero_division=0)),
        }
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_auroc_ovr": float(np.mean(aucs)) if aucs else None,
        "macro_auprc_ovr": float(np.mean(aps)) if aps else None,
        "confusion_matrix_true_rows_pred_cols": confusion_matrix(y_true, y_pred, labels=range(3)).tolist(),
        "per_class": per_class,
    }


def atomic_torch_save(data: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="experiments/paper_matched_single_cvc_v1/single_annotated_cvc_manifest.parquet")
    parser.add_argument("--output_dir", default="runs/paper_matched_single_cvc_v1")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--micro_batch", type=int, default=8)
    parser.add_argument("--effective_batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_path = (PROJECT_ROOT / args.manifest).resolve()
    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {out_dir}")
    if args.effective_batch < args.micro_batch or args.effective_batch % args.micro_batch:
        raise ValueError("effective_batch must be a positive multiple of micro_batch")
    if args.updates <= 0 or args.eval_interval <= 0:
        raise ValueError("updates and eval_interval must be positive")

    df = pd.read_parquet(manifest_path)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("Manifest must contain non-empty train and val partitions")
    if set(train_df["PatientID"]) & set(val_df["PatientID"]):
        raise AssertionError("Patient leakage between training and validation")
    if set(train_df["duplicate_group_id"]) & set(val_df["duplicate_group_id"]):
        raise AssertionError("Duplicate-group leakage between training and validation")
    if val_df["is_exposed"].astype(bool).any():
        raise AssertionError("Validation includes previously exposed studies")

    for part, frame in [("train", train_df), ("val", val_df)]:
        if not (frame[LABELS].sum(axis=1) == 1).all():
            raise AssertionError(f"{part} contains non-exclusive placement labels")
        if not (frame[LABELS].idxmax(axis=1) == frame["trace_label"]).all():
            raise AssertionError(f"{part} contains trace/image-label disagreement")

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This planned pilot requires CUDA; refusing a slow CPU training run")
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir()

    train_loader = create_placement_dataloader(
        train_df, batch_size=args.micro_batch, is_train=True, num_workers=0, aug_seed=args.seed
    )
    val_loader = create_placement_dataloader(
        val_df, batch_size=args.micro_batch * 2, is_train=False, num_workers=0
    )
    model = CVCPlacementBaselineNet(pretrained=True, num_classes=3, dropout_rate=0.25).to(device)
    counts = train_df[LABELS].sum(axis=0).to_numpy(dtype=np.float64)
    if np.any(counts <= 0):
        raise ValueError(f"Training partition missing a class: {counts.tolist()}")
    # Effective-number inverse-frequency weighting, normalized to mean 1.
    class_weights = len(train_df) / (3.0 * counts)
    class_weights = class_weights / class_weights.mean()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.updates, eta_min=1e-6)
    accum_steps = args.effective_batch // args.micro_batch
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    metadata = {
        "run_name": "paper_matched_single_cvc_v1",
        "task": "single annotated CVC, exclusive Normal/Borderline/Abnormal classification",
        "method_note": "RANZCR image labels + one matching CVC polyline; no carina-distance labels are inferred.",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "updates": args.updates,
        "eval_interval": args.eval_interval,
        "micro_batch": args.micro_batch,
        "effective_batch": args.effective_batch,
        "learning_rate": args.lr,
        "seed": args.seed,
        "class_names": LABELS,
        "train_studies": int(len(train_df)),
        "validation_studies": int(len(val_df)),
        "train_class_counts": {label: int(train_df[label].sum()) for label in LABELS},
        "validation_class_counts": {label: int(val_df[label].sum()) for label in LABELS},
        "class_weights": class_weights.tolist(),
        "test_policy": "Test partition is never loaded by this training program.",
    }
    (out_dir / "resolved_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"device": str(device), "amp_dtype": str(amp_dtype), **{k: metadata[k] for k in ["train_studies", "validation_studies", "train_class_counts", "validation_class_counts"]}}, indent=2), flush=True)

    best_auc = -math.inf
    history: list[dict[str, Any]] = []
    start = time.time()
    update = 0
    train_iter = iter(train_loader)
    while update < args.updates:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            x = batch["image"].to(device, non_blocking=True)
            y = batch["labels"].argmax(dim=1).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(x)
                loss = loss_fn(logits.float(), y) / accum_steps
            loss.backward()
            loss_sum += float(loss.detach().item()) * accum_steps / accum_steps

        finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        if not finite:
            optimizer.zero_grad(set_to_none=True)
            print(f"WARNING non-finite gradient; skipped batch at update {update + 1}", flush=True)
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        update += 1
        if update % 10 == 0 or update == 1:
            print(f"Update {update}/{args.updates} loss={loss_sum:.4f} lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

        if update % args.eval_interval == 0 or update == args.updates:
            metrics = evaluate(model, val_loader, device)
            record = {"update": update, "train_loss": loss_sum, "validation": metrics}
            history.append(record)
            save_json(out_dir / "history.json", history)
            print(f"VALIDATION update={update} macro_AUROC={metrics['macro_auroc_ovr']} macro_AUPRC={metrics['macro_auprc_ovr']} balanced_accuracy={metrics['balanced_accuracy']} macro_F1={metrics['macro_f1']}", flush=True)
            auc = metrics["macro_auroc_ovr"]
            if auc is not None and auc > best_auc:
                best_auc = auc
                atomic_torch_save(
                    {
                        "update": update,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "validation_metrics": metrics,
                        "resolved_config": metadata,
                    },
                    checkpoint_dir / "best_single_cvc_placement.pt",
                )

    final_metrics = history[-1]["validation"] if history else None
    atomic_torch_save(
        {"update": update, "model_state_dict": model.state_dict(), "final_validation_metrics": final_metrics},
        checkpoint_dir / "final_single_cvc_placement.pt",
    )
    save_json(out_dir / "run_summary.json", {
        "status": "complete",
        "elapsed_seconds": round(time.time() - start, 2),
        "best_validation_macro_auroc": best_auc if math.isfinite(best_auc) else None,
        "best_checkpoint": str(checkpoint_dir / "best_single_cvc_placement.pt"),
        "history": history,
        "resolved_config": metadata,
    })
    print(f"COMPLETE elapsed_seconds={time.time() - start:.1f} best_macro_AUROC={best_auc:.4f}", flush=True)


if __name__ == "__main__":
    main()
