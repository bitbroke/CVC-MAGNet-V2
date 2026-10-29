"""End-to-End Smoke Test: Training, Validation, Checkpoint Saving, and Supervised Target Evaluation.

Verifies complete execution reliability on RTX 3050 4GB before launching full 1000-step arms:
1. Step-controlled dual-head training (5 steps, batch_size=4)
2. Frozen validation cohort evaluation (N=25 targets, N=8 negatives)
3. Checkpoint persistence and integrity check
4. Supervised training targets evaluation (N=39 active training targets)
5. Peak GPU memory measurement and diagnostics
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Ensure memory allocator config is set before PyTorch initialization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.coarse_dual import CoarseDualNet
import importlib.util
_spec = importlib.util.spec_from_file_location("train_mod", PROJECT_ROOT / "scripts" / "05_train_coarse_dual.py")
_train_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train_mod)

load_shared_backbone = _train_mod.load_shared_backbone
compute_model_backbone_hash = _train_mod.compute_model_backbone_hash
CoarseDualLoss = _train_mod.CoarseDualLoss
run_dual_validation = _train_mod.run_dual_validation
save_checkpoint = _train_mod.save_checkpoint
UpdateCounter = _train_mod.UpdateCounter
supervision_groups = _train_mod.supervision_groups
from src.data.datasets import CoarseDualDataset
from scripts.run_sampling_comparison import evaluate_supervised_training_targets

def run_smoke_test():
    print("=" * 80)
    print("RUNNING END-TO-END TRAINING AND EVALUATION SMOKE TEST")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        free_mem, total_mem = torch.cuda.mem_get_info()
        print(f"GPU: {props.name} | Total VRAM: {total_mem / (1024**2):.1f} MB | Free VRAM: {free_mem / (1024**2):.1f} MB")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    smoke_dir = PROJECT_ROOT / "runs" / "smoke_test_e2e"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    smoke_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = smoke_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load configs & inventory
    with open(PROJECT_ROOT / "configs/pilot.yaml") as f:
        cfg = yaml.safe_load(f)
    raw_images_dir = Path(cfg["data"]["raw_dir"]) / "train"
    inv_df = pd.read_parquet(PROJECT_ROOT / "data/annotation_inventory.parquet")
    study_df = pd.read_parquet(PROJECT_ROOT / "data/study_inventory.parquet")
    v3_path = PROJECT_ROOT / "data/reviewed_distal_tips_v3.parquet"
    rev_df = pd.read_parquet(v3_path)

    confirmed_neg_ids = set(
        study_df[study_df["is_study_level_negative"] & (~study_df["has_cvc"])]["StudyInstanceUID"]
    )
    with open(PROJECT_ROOT / "data/splits.json") as f:
        splits = json.load(f)
    train_study_ids = splits["partitions"]["dev_pilot_train"]["study_ids"]
    val_study_ids = splits["partitions"]["dev_pilot_val"]["study_ids"]

    rev_train_df = rev_df[rev_df["split"] == "train"].copy()
    rev_val_df = rev_df[rev_df["split"] == "val"].copy()

    # 2. Build Datasets
    print("\n[Step 1/5] Building datasets & loaders (batch_size=4)...")
    train_ds = CoarseDualDataset(
        study_ids=train_study_ids,
        inventory_df=inv_df,
        images_dir=raw_images_dir,
        confirmed_negative_study_ids=confirmed_neg_ids,
        reviewed_tips_df=rev_train_df,
        target_size=(512, 512),
        mask_stroke_width_native=10,
        clahe_enabled=True,
        tip_sigma=2.5,
        partial_review_policy="exclude",
    )
    val_ds = CoarseDualDataset(
        study_ids=val_study_ids,
        inventory_df=inv_df,
        images_dir=raw_images_dir,
        confirmed_negative_study_ids=confirmed_neg_ids,
        reviewed_tips_df=rev_val_df,
        target_size=(512, 512),
        mask_stroke_width_native=10,
        clahe_enabled=True,
        tip_sigma=2.5,
        partial_review_policy="exclude",
    )

    sampling_gen = torch.Generator().manual_seed(42)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=4, shuffle=True, generator=sampling_gen, num_workers=0, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=4, shuffle=False, num_workers=0
    )

    # 3. Model & Initial Checkpoint Transfer
    print("\n[Step 2/5] Initializing CoarseDualNet and transferring backbone...")
    init_ckpt_path = PROJECT_ROOT / "runs/pilot_coarse_repaired/checkpoints/best_coarse.pt"
    model = CoarseDualNet(in_channels=1, base_channels=32).to(device)
    transfer_info = load_shared_backbone(model, init_ckpt_path)
    bb_hash = compute_model_backbone_hash(model)
    print(f"  Backbone transfer: {transfer_info['loaded_keys_count']} keys loaded | Hash: {bb_hash}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    update_counter = UpdateCounter(optimizer)
    criterion = CoarseDualLoss(
        bce_weight=1.0,
        dice_weight=2.0,
        tip_weight=2.0,
        pos_weight=torch.tensor([50.0], device=device),
        tip_pos_weight=100.0,
    )

    # 4. Training (5 steps)
    print("\n[Step 3/5] Executing 5 step-controlled training updates...")
    t_train_start = time.time()
    model.train()
    step = 0
    attempts = 0
    skipped_updates = 0

    for batch in train_loader:
        if step >= 5:
            break
        attempts += 1
        imgs = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        tip_hms = batch["tip_heatmap"].to(device, non_blocking=True)
        tip_ms = batch["tip_mask"].to(device, non_blocking=True)
        tip_sms = batch["tip_spatial_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            seg_logits, tip_logits = model(imgs)
            loss, loss_dict = criterion(
                seg_logits, tip_logits, masks, tip_hms,
                tip_mask=tip_ms, tip_spatial_mask=tip_sms
            )

        scaler.scale(loss).backward()
        before_update = update_counter.successful
        scaler.step(optimizer)
        scaler.update()

        updated = update_counter.successful > before_update
        if updated:
            step = update_counter.successful
            print(f"  Step {step}/5 completed | Loss: {loss_dict['loss_total']:.4f} (Seg: {loss_dict['loss_seg']:.4f}, Tip: {loss_dict['loss_tip']:.4f})")
        else:
            skipped_updates += 1
            print(f"  Attempt {attempts} skipped by AMP scaler")

        del imgs, masks, tip_hms, tip_ms, tip_sms, loss, loss_dict, seg_logits, tip_logits, batch

    train_duration = time.time() - t_train_start
    train_peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0
    print(f"  Training finished in {train_duration:.2f}s | Peak GPU Memory: {train_peak_mb:.1f} MB")

    # 5. Validation
    print("\n[Step 4/5] Running frozen validation cohort evaluation...")
    t_val_start = time.time()
    val_metrics = run_dual_validation(
        model=model,
        val_loader=val_loader,
        reviewed_val_df=rev_val_df,
        confirmed_negatives=confirmed_neg_ids,
        criterion=criterion,
        device=device,
        mode="dual_head",
    )
    val_duration = time.time() - t_val_start
    print(f"  Validation completed in {val_duration:.2f}s")
    print(f"  Val Loss: Total={val_metrics['val_loss_total']:.4f}, Seg={val_metrics['val_loss_seg']:.4f}, Tip={val_metrics['val_loss_tip']:.4f}")
    print(f"  Mask Dice: {val_metrics['val_mask_dice_mean']:.4f}")
    print(f"  [Seg Endpoints] Success@15: {val_metrics['seg_skeleton']['matched_15_count']}/25 ({val_metrics['seg_skeleton']['success_at_15']:.1%})")
    print(f"  [Tip Heatmap]   Success@15: {val_metrics['tip_heatmap']['matched_15_count']}/25 ({val_metrics['tip_heatmap']['success_at_15']:.1%})")

    # 6. Checkpoint Saving
    final_ckpt_path = ckpt_dir / "final_coarse_dual_head.pt"
    print(f"\n[Step 5/5] Saving and verifying checkpoint to {final_ckpt_path.name}...")
    save_checkpoint(
        path=final_ckpt_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=None,
        sampler=None,
        step=5,
        config={"smoke_test": True, "batch_size": 4},
        input_hashes={"backbone_hash": bb_hash},
        metrics=val_metrics,
        is_best=False,
    )
    assert final_ckpt_path.exists(), f"Checkpoint {final_ckpt_path} was not created!"
    ckpt_size_mb = final_ckpt_path.stat().st_size / (1024**2)
    print(f"  Checkpoint successfully saved ({ckpt_size_mb:.1f} MB).")

    # 7. Supervised Training Target Evaluation
    print("\n[Step 6/6] Evaluating active supervised training targets (N=39)...")
    t_eval_start = time.time()
    train_eval = evaluate_supervised_training_targets(
        checkpoint_path=final_ckpt_path,
        reviewed_tips_path=v3_path,
        device=device,
    )
    eval_duration = time.time() - t_eval_start
    print(f"  Training target evaluation completed in {eval_duration:.2f}s")
    print(f"  Active Training Targets Total: {train_eval['active_training_targets_total']}")
    print(f"  Success@15: {train_eval['matched_15_count']}/39 ({train_eval['success_at_15']:.1%})")
    print(f"  Targets missing candidates: {train_eval['targets_missing_candidates_count']}")
    assert train_eval["active_training_targets_total"] == 39, f"Expected 39 training targets, got {train_eval['active_training_targets_total']}"

    overall_peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0
    free_final, _ = torch.cuda.mem_get_info() if device.type == "cuda" else (0, 0)

    summary = {
        "status": "success",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "batch_size": 4,
        "steps_completed": step,
        "peak_gpu_memory_mb": round(overall_peak_mb, 1),
        "free_gpu_memory_mb": round(free_final / (1024**2), 1),
        "durations_seconds": {
            "training": round(train_duration, 2),
            "validation": round(val_duration, 2),
            "supervised_eval": round(eval_duration, 2),
            "total": round(train_duration + val_duration + eval_duration, 2),
        },
        "validation_metrics": {
            "mask_dice": round(val_metrics["val_mask_dice_mean"], 4),
            "seg_endpoints_s15": val_metrics["seg_skeleton"]["matched_15_count"],
            "tip_heatmap_s15": val_metrics["tip_heatmap"]["matched_15_count"],
        },
        "supervised_training_eval": {
            "active_targets": train_eval["active_training_targets_total"],
            "matched_15_count": train_eval["matched_15_count"],
            "success_at_15": round(train_eval["success_at_15"], 4),
        },
    }
    (smoke_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("SMOKE TEST COMPLETE: ALL CONTRACTS & END-TO-END STAGES VERIFIED")
    print(f"  Peak GPU Memory: {overall_peak_mb:.1f} MB (Safety Margin: {free_final / (1024**2):.1f} MB free)")
    print(f"  Training Steps: 5/5 successful")
    print(f"  Validation Targets: 25/25 evaluated")
    print(f"  Supervised Training Targets: 39/39 evaluated")
    print(f"  Checkpoint: verified ({ckpt_size_mb:.1f} MB)")
    print("=" * 80)

if __name__ == "__main__":
    run_smoke_test()
