"""Trainer for T1 Direct CVC Tip Heatmap Model.

Implements Section 5 (T1) of the Experiment Specification:
- Model: CVCTipHeatmapNet (Image -> Multi-tip Heatmap at stride 4, retaining deep context)
- Input: 768px, Output: 192x192 Gaussian tip heatmap (sigma=2.0)
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4) with CosineAnnealingLR
- Budget: 1,000 successful updates
- Effective batch size: 4 (micro-batch 1, grad accumulation 4)
- Mixed Precision (AMP) with FP32 loss reduction
- In-place same-batch retry on AMP gradient overflow
- Balanced sampling: 3 positives : 1 negative (50% uniform, 50% selective hard negative)
- Exact tip supervision masking: loss applied only on completely labelled positives and confirmed negatives
- Saves complete result package in experiments/T1_direct_tip_heatmap/
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms.functional as TF

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tip_dataset import CVCTipDataset
from src.evaluation.tip_metrics import compute_tip_metrics, compute_patient_clustered_bootstrap
from src.experiments.registry import ExperimentContract, compute_file_sha256, update_master_table
from src.infer.decoders import extract_peaks_from_heatmap, decode_d0_cell_center
from src.models.tip_heatmap import CVCTipHeatmapNet, TipFocalLoss

DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENT_ID = "T1_direct_tip_heatmap"


def seed_everything(seed: int = 43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class BalancedTipSampler:
    """Samples 3 positive studies and 1 negative study (50% selective hard negatives)."""

    def __init__(
        self,
        positive_studies: List[str],
        uniform_negative_studies: List[str],
        hard_negative_studies: List[str],
        seed: int = 43,
    ):
        self.positive_studies = list(positive_studies)
        self.uniform_negative_studies = list(uniform_negative_studies)
        self.hard_negative_studies = list(hard_negative_studies)
        self.rng = random.Random(seed)
        self.pos_cursor = 0
        self.neg_cursor = 0
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


def evaluate_tip_validation(
    model: nn.Module,
    val_dataset: CVCTipDataset,
    device: torch.device,
    conf_thresh: float = 0.20,
    nms_radius: float = 3.0,
) -> Dict[str, Any]:
    model.eval()
    val_records: List[Dict[str, Any]] = []

    with torch.no_grad():
        for i in range(len(val_dataset)):
            if (i + 1) % 50 == 0 or (i + 1) == len(val_dataset):
                print(f"   [Val] Processed {i+1}/{len(val_dataset)} studies...", flush=True)
            item = val_dataset[i]
            sid = item["study_id"]
            pid = item["patient_id"]
            is_pos = item["is_cvc_positive"]
            native_w = item["native_w"]
            native_h = item["native_h"]

            x = item["image"].unsqueeze(0).to(device)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out = model(x)
                heatmap = out["tip_probs"][0, 0].float().cpu().numpy()

            # Decode peaks using D0 (cell center)
            raw_peaks, nms_peaks = extract_peaks_from_heatmap(
                heatmap,
                threshold=conf_thresh,
                nms_kernel_size=3,
                suppression_radius_px=nms_radius,
            )
            pred_tips = []
            for p in nms_peaks:
                nx, ny = decode_d0_cell_center(
                    p,
                    stride=4,
                    img_size=768,
                    native_w=native_w,
                    native_h=native_h,
                )
                pred_tips.append({"x": nx, "y": ny, "conf": p["conf"]})

            # Ground truth assessable tips
            gt_tips = []
            if sid in val_dataset.study_tips:
                for tx, ty in val_dataset.study_tips[sid]:
                    gt_tips.append({
                        "x": float(tx),
                        "y": float(ty),
                        "tip_state": "assessable",
                        "instance_id": 1,
                        "is_border": False,
                    })

            val_records.append({
                "study_id": sid,
                "patient_id": pid,
                "is_cvc_positive": is_pos,
                "gt_tips": gt_tips,
                "pred_tips": pred_tips,
                "raw_candidates": pred_tips,
            })

    val_df = pd.DataFrame(val_records)
    metrics_bundle = compute_tip_metrics(val_df, confidence_threshold=conf_thresh)
    return metrics_bundle


def train_t1():
    parser = argparse.ArgumentParser(description="T1 Direct CVC Tip Heatmap Trainer")
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "runs" / "t1_direct_tip_heatmap_1000"))
    parser.add_argument("--target_updates", type=int, default=1000)
    parser.add_argument("--effective_batch_size", type=int, default=4)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=768)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--aug_seed", type=int, default=43)
    parser.add_argument("--load_initial_weights", type=str, default=str(PROJECT_ROOT / "runs" / "s5_duration_control_2500" / "checkpoints" / "final_segmentation_v3.pt"))
    parser.add_argument("--max_retry_attempts", type=int, default=5)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting T1 Direct Tip Heatmap Training on {device} ===")

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data Contracts & Inventories
    catheter_df = pd.read_parquet(DATA_DIR / "cvc_catheter_contracts.parquet")
    study_sup_df = pd.read_parquet(DATA_DIR / "cvc_study_supervision.parquet")
    
    with open(DATA_DIR / "expanded_training_cohort_manifest.json", "r", encoding="utf-8") as f:
        expanded_manifest = json.load(f)
    train_positive_ids = expanded_manifest["positive_study_ids"]
    all_neg_ids = expanded_manifest["negative_study_ids"]

    with open(DATA_DIR / "selective_hard_negatives_manifest.json", "r", encoding="utf-8") as f:
        hn_manifest = json.load(f)
    hard_neg_ids = hn_manifest["study_ids"]
    uniform_neg_ids = [s for s in all_neg_ids if s not in hard_neg_ids]

    with open(DATA_DIR / "splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)
    val_study_ids = splits["partitions"]["dev_pilot_val"]["study_ids"]

    images_dir = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")

    print(f"Cohort: {len(train_positive_ids)} positives, {len(uniform_neg_ids)} uniform negatives, {len(hard_neg_ids)} hard negatives.")
    print(f"Validation: {len(val_study_ids)} studies.")

    # 2. Datasets
    train_all_ids = list(set(train_positive_ids + all_neg_ids))
    train_dataset = CVCTipDataset(
        study_ids=train_all_ids,
        catheter_contracts_df=catheter_df,
        study_supervision_df=study_sup_df,
        images_dir=images_dir,
        img_size=args.img_size,
        is_train=True,
        include_segmentation=False,
        aug_seed=args.aug_seed,
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

    # 3. Model Architecture
    model = CVCTipHeatmapNet(pretrained=False, freeze_encoder_bn=True)
    if args.load_initial_weights and os.path.exists(args.load_initial_weights):
        print(f"Loading initial shared backbone/decoder weights from: {args.load_initial_weights}")
        st = torch.load(args.load_initial_weights, map_location="cpu", weights_only=False)
        src_state = st.get("model_state_dict", st.get("state_dict", st))
        # Filter matching keys
        tgt_state = model.state_dict()
        transferred = 0
        for k, v in src_state.items():
            if k in tgt_state and tgt_state[k].shape == v.shape:
                tgt_state[k] = v
                transferred += 1
        model.load_state_dict(tgt_state)
        print(f"Transferred {transferred} weight tensors successfully.")

    model.to(device)
    torch.save({"model_state_dict": model.state_dict()}, ckpt_dir / "initial.pt")

    # 4. Optimizer & Loss
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.target_updates, eta_min=1e-6)
    scaler = GradScaler(enabled=(device.type == "cuda"), init_scale=1024.0)
    criterion = TipFocalLoss(alpha=2.0, beta=4.0)

    # 5. Training Loop with Same-Batch Retry
    updates = 0
    attempts = 0
    skips = 0
    best_success15 = -1.0
    accum_steps = args.effective_batch_size // args.micro_batch_size

    history_records = []
    t_start = time.time()

    print(f"\n--- Launching 1,000 updates (Effective batch {args.effective_batch_size}, Micro-batch {args.micro_batch_size}) ---")

    while updates < args.target_updates:
        batch_study_ids = sampler.sample_batch(batch_size=args.effective_batch_size)
        
        # Micro-batch preparation
        micro_batches = []
        for sid in batch_study_ids:
            idx = study_id_to_idx[sid]
            sample = train_dataset[idx]
            micro_batches.append(sample)

        # Retry loop for gradient overflow
        retry = 0
        success = False

        while retry <= args.max_retry_attempts:
            attempts += 1
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for mb in micro_batches:
                img = mb["image"].unsqueeze(0).to(device)
                targ_map = mb["tip_heatmap"].unsqueeze(0).to(device)
                tip_mask = mb["tip_mask"].view(1, 1, 1, 1).to(device)

                with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    out = model(img)
                    pred_map = out["tip_probs"]
                    # Mixed precision forward with FP32 loss reduction
                    loss = criterion(pred_map.float(), targ_map.float(), mask=tip_mask.float()) / accum_steps

                scaler.scale(loss).backward()
                accum_loss += loss.item()

            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if torch.isfinite(grad_norm):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                updates += 1
                success = True
                break
            else:
                skips += 1
                retry += 1
                scaler.update()
                print(f"Warning: Non-finite gradients at update {updates} (retry {retry}/{args.max_retry_attempts}). Same-batch retry...")

        if not success:
            raise RuntimeError(f"FATAL: Exceeded maximum gradient overflow retries ({args.max_retry_attempts}) at update {updates}.")

        if updates % 50 == 0 or updates == args.target_updates:
            elapsed = time.time() - t_start
            lr_curr = optimizer.param_groups[0]["lr"]
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
            print(f"Update {updates:4d}/{args.target_updates} | Loss: {accum_loss:.4f} | LR: {lr_curr:.2e} | VRAM: {vram_mb:.0f} MB | Elapsed: {elapsed/60:.1f}m", flush=True)
            history_records.append({
                "update": updates,
                "loss": accum_loss,
                "lr": lr_curr,
                "elapsed_sec": elapsed,
            })

        # Periodic Validation
        if updates % 250 == 0 or updates == args.target_updates:
            print(f"--> Running Validation Evaluation at update {updates}...", flush=True)
            val_res = evaluate_tip_validation(model, val_dataset, device)
            s15 = val_res["summary"]["primary_metric"]["value"]
            prec = val_res["summary"]["tolerance_metrics"]["Success@15"]["precision"]
            rec = val_res["summary"]["tolerance_metrics"]["Success@15"]["recall"]
            print(f"--> Validation Success@15: {s15:.4f} (Prec: {prec:.4f}, Rec: {rec:.4f})", flush=True)
            
            # Save checkpoints
            torch.save({"model_state_dict": model.state_dict(), "update": updates}, ckpt_dir / "last.pt")
            if s15 > best_success15:
                best_success15 = s15
                torch.save({"model_state_dict": model.state_dict(), "update": updates}, ckpt_dir / "best.pt")
                print(f"--> Saved new BEST checkpoint (Success@15: {s15:.4f})", flush=True)

    # Save Final Checkpoint
    final_ckpt_path = ckpt_dir / "final.pt"
    torch.save({"model_state_dict": model.state_dict(), "update": updates}, final_ckpt_path)
    pd.DataFrame(history_records).to_csv(output_dir / "history.csv", index=False)
    print(f"\nTraining Complete in {(time.time() - t_start)/60:.1f} mins. Saved {final_ckpt_path}")

    # 6. Final Benchmark Evaluation on Frozen Final Checkpoint
    print(f"\n=== Running Final Hungarian Tip Benchmark on {final_ckpt_path} ===")
    contract = ExperimentContract(EXPERIMENT_ID)
    final_eval = evaluate_tip_validation(model, val_dataset, device)
    summary = final_eval["summary"]
    per_study_df = final_eval["per_study_df"]
    matches_df = final_eval["matches_df"]

    boot_ci = compute_patient_clustered_bootstrap(per_study_df, n_bootstrap=1000, seed=43)

    # Save result package
    eval_dir = contract.evaluation_dir
    with open(eval_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    per_study_df.to_csv(eval_dir / "per_study_metrics.csv", index=False)
    matches_df.to_parquet(eval_dir / "matches.parquet", index=False)
    with open(eval_dir / "bootstrap_intervals.json", "w", encoding="utf-8") as f:
        json.dump(boot_ci, f, indent=2)

    # Copy checkpoints to experiment package
    import shutil
    shutil.copy2(ckpt_dir / "final.pt", contract.checkpoints_dir / "final.pt")
    if (ckpt_dir / "best.pt").exists():
        shutil.copy2(ckpt_dir / "best.pt", contract.checkpoints_dir / "best.pt")
    shutil.copy2(ckpt_dir / "initial.pt", contract.checkpoints_dir / "initial.pt")
    shutil.copy2(ckpt_dir / "last.pt", contract.checkpoints_dir / "last.pt")

    # Write Hypothesis and Command
    with open(contract.exp_dir / "hypothesis.txt", "w", encoding="utf-8") as f:
        f.write("Hypothesis T1: Direct landmark learning via multi-tip heatmap outperforms skeleton endpoints from segmentation.\n")
    with open(contract.exp_dir / "command.txt", "w", encoding="utf-8") as f:
        f.write(f"python scripts/train_t1_heatmap.py --target_updates 1000 --effective_batch_size 4 --micro_batch_size 1 --img_size 768 --lr 1e-4 --seed 43\n")

    # Update Master Metrics Table
    update_master_table(
        experiment_id=EXPERIMENT_ID,
        seed=args.seed,
        checkpoint_sha=compute_file_sha256(str(final_ckpt_path)),
        cohort_hash=compute_file_sha256(str(DATA_DIR / "expanded_training_cohort_manifest.json")),
        successful_updates=1000,
        primary_metric=summary["primary_metric"]["value"],
        precision=summary["tolerance_metrics"]["Success@15"]["precision"],
        recall=summary["tolerance_metrics"]["Success@15"]["recall"],
        false_predictions_per_image=summary["candidate_diagnostics"]["false_tips_per_image"],
        median_error=summary["error_distribution_matched_15px"]["median_error_px"] or 0.0,
        p95_error=summary["error_distribution_matched_15px"]["p95_error_px"] or 0.0,
        runtime_seconds=time.time() - t_start,
        peak_vram_mb=torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0,
        status="completed",
        decision="evaluated",
    )
    print(f"Master metrics table updated for {EXPERIMENT_ID}!")


if __name__ == "__main__":
    train_t1()
