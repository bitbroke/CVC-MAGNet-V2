"""Shared Production Training and Evaluation Runner for HRNet Tip Pilot (T1).

Model:
- HRNetJointTipNet initialized from frozen Seed-42 HRNet segmentation weights.
- Multi-tip heatmap head at stride 4 (192x192 at 768px input).

Supervision Invariants:
1. Exact binary segmentation masks preserved as 0/1 float.
2. Positive tip supervision enabled only on fully accounted studies.
3. Confirmed negatives supervised with empty heatmaps (valid image area active, padding excluded).
4. Incomplete positive studies have whole-image tip loss masked (zero loss, zero gradient from tip head).
5. Padding receives strictly zero loss and zero gradient.
6. Effective-batch normalization across microbatches (BCE by valid pixels, Dice by positives).
7. Full state checkpointing and bitwise-exact resume.
8. Directory overwrite protection.
"""

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
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

EXP_ROOT = PROJECT_ROOT / "experiments" / "hrnet_tip_pilot_v1"
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from src.data.rasterize import rasterize_study_polylines
from src.data.transforms import LetterboxTransform
from experiments.hrnet_tip_pilot_v1.src.geometry import (
    CoordinateTransform,
    native_point_to_heatmap,
    heatmap_point_to_native,
    render_multitip_gaussian_heatmap,
)
from experiments.hrnet_tip_pilot_v1.src.models.hrnet_joint_tip import HRNetJointTipNet
from experiments.hrnet_tip_pilot_v1.src.decoder import TransparentTipDecoder
from experiments.hrnet_tip_pilot_v1.src.evaluation import evaluate_cohort_tips, match_study_tips

DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = EXP_ROOT / "frozen_reference" / "configs"
CKPT_REF_DIR = EXP_ROOT / "frozen_reference" / "checkpoints"
IMAGES_DIR = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")
FROZEN_HRNET_CKPT = CKPT_REF_DIR / "hrnet_seed42_best.pt"


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def hash_file(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class TipWeightedBCELoss(nn.Module):
    """Numerically stable weighted BCE loss for spatial tip heatmaps."""

    def __init__(self, pos_weight: float = 20.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_hm: torch.Tensor,
        valid_mask: torch.Tensor,
        has_positive_tips: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute imbalance-aware loss on valid heatmap regions."""
        logits = pred_logits.float()
        targets = target_hm.float()
        valids = valid_mask.float()

        # BCE elementwise with built-in numerical stability
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        pos_mask = (targets >= 0.50).float() * valids
        neg_mask = (targets < 0.50).float() * valids

        weighted_bce = (bce * pos_mask * self.pos_weight) + (bce * neg_mask)
        total_loss = (weighted_bce * valids).sum()

        num_pos_peaks = pos_mask.sum().item()
        num_valid_pixels = valids.sum().item()

        if has_positive_tips and num_pos_peaks > 0:
            norm_loss = total_loss / (num_pos_peaks * self.pos_weight + num_valid_pixels * 0.05)
        else:
            norm_loss = total_loss / max(num_valid_pixels, 1.0)

        loss_info = {
            "pos_loss": float((bce * pos_mask).sum().item()),
            "neg_loss": float((bce * neg_mask).sum().item()),
            "num_pos_peaks": float(num_pos_peaks),
            "num_valid_pixels": float(num_valid_pixels),
        }

        return norm_loss, loss_info


class BenchmarkValidator:
    """Evaluates Joint Model on locked development benchmark."""

    def __init__(self, benchmark_path: Path, device: torch.device):
        self.device = device
        with open(benchmark_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        self.studies = self.manifest["studies"]
        self.transform = LetterboxTransform(target_size=(768, 768), clahe_enabled=True)
        self.decoder = TransparentTipDecoder(conf_threshold=0.20, nms_radius_native_px=30.0, max_candidates=10)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def evaluate(self, model: nn.Module) -> Dict[str, Any]:
        model.eval()
        cohort_results = []
        all_preds = {}

        with torch.no_grad():
            for s in self.studies:
                sid = s["study_id"]
                img_path = IMAGES_DIR / f"{sid}.jpg"
                if not img_path.exists():
                    continue

                pil_im = Image.open(img_path).convert("L")
                im_np = np.array(pil_im)
                orig_h, orig_w = im_np.shape

                coord_tf = CoordinateTransform.from_native_dims(orig_w, orig_h, target_size=768, stride=4)

                lb_img, meta = self.transform.apply_image(im_np)
                raw_t = torch.from_numpy(lb_img).unsqueeze(0).unsqueeze(0).to(self.device).float() / 255.0
                inp = (raw_t.repeat(1, 3, 1, 1) - self.mean) / self.std

                out = model(inp)
                tip_probs = out["tip_probs"].squeeze().cpu().numpy().astype(np.float32)

                dec_res = self.decoder.decode(tip_probs, coord_tf)
                final_preds = dec_res["final_predictions"]
                all_preds[sid] = final_preds

                targets = [(float(t["x"]), float(t["y"])) for t in s.get("assessable_tips", [])]

                cohort_results.append({
                    "study_id": sid,
                    "is_negative": s.get("is_negative", False),
                    "num_cvcs": s.get("num_cvcs", len(targets)),
                    "targets": targets,
                    "predictions": final_preds,
                })

        model.train()
        metrics = evaluate_cohort_tips(cohort_results)
        metrics["predictions"] = all_preds
        metrics["cohort_results"] = cohort_results
        return metrics


def save_tip_checkpoint(
    ckpt_path: Path,
    model: nn.Module,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    update: int,
    best_success15: float,
    best_step: int,
    seed: int,
    peak_vram_mb: float = 0.0,
):
    torch.save(
        {
            "update": update,
            "best_success15": best_success15,
            "best_step": best_step,
            "seed": seed,
            "peak_vram_mb": peak_vram_mb,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        },
        ckpt_path,
    )


def run_training_pilot(
    target_updates: int = 1000,
    eval_interval: int = 250,
    seed: int = 42,
    output_dir: Optional[str] = None,
    lambda_tip: float = 1.0,
    resume_from: Optional[str] = None,
    overwrite: bool = False,
    skip_initial_eval: bool = False,
    run_eval_on_final: bool = True,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    print("\n" + "=" * 75)
    print(f"=== LAUNCHING HRNET JOINT TIP PILOT (T1, Seed {seed}, {target_updates} Updates) ===")
    print("=" * 75)

    device = torch.device(device_str)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        print(f"Device: {device.type} ({torch.cuda.get_device_name(device)})")
    else:
        print(f"Device: {device.type}")

    seed_everything(seed)

    if output_dir is not None:
        run_dir = Path(output_dir)
    else:
        run_dir = EXP_ROOT / "runs" / f"seed_{seed}" / "t1_joint_hrnet"

    ckpt_dir = run_dir / "checkpoints"

    # Directory Overwrite Protection
    if run_dir.exists() and resume_from is None and not overwrite:
        existing_artifacts = list(run_dir.glob("*.jsonl")) + list(ckpt_dir.glob("*.pt"))
        if existing_artifacts:
            raise FileExistsError(
                f"Run directory '{run_dir}' already exists with prior training artifacts! "
                f"Refusing to overwrite without explicit --overwrite or --resume_from flag."
            )

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Section 14 Run Metadata & Hashes
    resolved_config = {
        "model_name": "HRNetJointTipNet",
        "initialization": "frozen_hrnet_seed42",
        "seed": seed,
        "target_updates": target_updates,
        "eval_interval": eval_interval,
        "input_resolution": [768, 768],
        "heatmap_resolution": [192, 192],
        "stride": 4,
        "microbatch": 1,
        "effective_batch": 4,
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "min_learning_rate": 1e-6,
        "weight_decay": 1e-4,
        "scheduler": "CosineAnnealingLR",
        "lambda_tip": lambda_tip,
        "tip_loss": "TipWeightedBCELoss",
        "tip_pos_weight": 20.0,
        "seg_loss": "BCE(pos_weight=10) + SoftDice",
        "mixed_precision": "torch.amp.autocast(fp16)",
        "gradient_clipping": 10.0,
    }
    with open(run_dir / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(resolved_config, f, indent=2)

    with open(run_dir / "environment.txt", "w", encoding="utf-8") as f:
        f.write(f"Python: {sys.version}\n")
        f.write(f"PyTorch: {torch.__version__}\n")
        f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"Device: {torch.cuda.get_device_name(0)}\n")
            f.write(f"CUDA Version: {torch.version.cuda}\n")
            f.write(f"cuDNN Version: {torch.backends.cudnn.version()}\n")

    with open(run_dir / "initial_checkpoint_hash.json", "w", encoding="utf-8") as f:
        json.dump({
            "initial_checkpoint": str(FROZEN_HRNET_CKPT),
            "sha256": hash_file(FROZEN_HRNET_CKPT),
        }, f, indent=2)

    if (EXP_ROOT / "protocol" / "source_hashes.json").exists():
        shutil.copy(EXP_ROOT / "protocol" / "source_hashes.json", run_dir / "source_hashes.json")

    with open(run_dir / "status.json", "w", encoding="utf-8") as f:
        json.dump({
            "status": "RUNNING",
            "model": "HRNetJointTipNet",
            "seed": seed,
            "target_updates": target_updates,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }, f, indent=2)

    # 1. Instantiate Joint Model and load from frozen seed-42 HRNet
    model = HRNetJointTipNet(pretrained=False).to(device)
    model.load_from_frozen_hrnet(FROZEN_HRNET_CKPT)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[HRNetJointTipNet] Total Parameters: {total_params:,} ({trainable_params:,} trainable)")

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    tip_criterion = TipWeightedBCELoss()

    # 2. Ingest Immutable Batch Ledger
    ledger_path = CONFIGS_DIR / "batch_ledger.jsonl"
    if not ledger_path.exists():
        ledger_path = PROJECT_ROOT / "experiments/cvc_mamba_hybrid_20261017/configs/batch_ledger.jsonl"

    ledger: Dict[int, Dict[str, Any]] = {}
    with open(ledger_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                update_key = item.get("update_id", item.get("update"))
                ledger[update_key] = item
    print(f"Loaded immutable batch ledger with {len(ledger)} updates.")

    # 3. Ingest Target Registry & Polylines
    reg_path = EXP_ROOT / "manifests" / "instance_target_registry.parquet"
    reg_df = pd.read_parquet(reg_path)

    # Index tips by study_id
    study_tips_lookup: Dict[str, List[Tuple[float, float]]] = {}
    study_supervision_status: Dict[str, bool] = {}

    for sid, group in reg_df.groupby("study_id"):
        # Check if study is fully accounted
        all_assessable = all(r["tip_state"] == "Assessable distal tip" for _, r in group.iterrows())
        has_audited = any(r["review_status"] == "provisional_engineering_review_v3" for _, r in group.iterrows())
        # For positive supervision in pilot: require all CVCs to have assessable coordinates
        study_supervision_status[sid] = all_assessable
        tips = [(float(r["tip_x_native"]), float(r["tip_y_native"])) for _, r in group.iterrows() if r["tip_state"] == "Assessable distal tip"]
        study_tips_lookup[sid] = tips

    inv_df = pd.read_parquet(DATA_DIR / "annotation_inventory.parquet")
    train_polylines: Dict[str, List[List[List[float]]]] = {}
    for _, row in inv_df[inv_df["device_type"] == "CVC"].iterrows():
        sid = str(row["study_id"])
        p_raw = row["polyline"]
        poly = json.loads(p_raw) if isinstance(p_raw, str) else p_raw
        train_polylines.setdefault(sid, []).append(poly)

    # 4. Ingest Benchmark
    benchmark_path = EXP_ROOT / "manifests" / "development_benchmark_manifest.json"
    validator = BenchmarkValidator(benchmark_path, device=device)

    # Normalization constants
    transform = LetterboxTransform(target_size=(768, 768), clahe_enabled=True)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    current_update = 1
    best_success15 = -1.0
    best_step = -1
    history_entries = []
    history_file = run_dir / "history.jsonl"

    if resume_from:
        print(f"\n[Resume] Restoring checkpoint from {resume_from}...")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])

        if "torch_rng_state" in ckpt and ckpt["torch_rng_state"] is not None:
            torch.set_rng_state(ckpt["torch_rng_state"].cpu())
        if device.type == "cuda" and ckpt.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(ckpt["cuda_rng_state"].cpu())
        if "numpy_rng_state" in ckpt:
            np.random.set_state(ckpt["numpy_rng_state"])
        if "python_rng_state" in ckpt:
            random.setstate(ckpt["python_rng_state"])

        current_update = ckpt["update"] + 1
        best_success15 = ckpt.get("best_success15", -1.0)
        best_step = ckpt.get("best_step", -1)

        if history_file.exists():
            with open(history_file, "r", encoding="utf-8") as f:
                history_entries = [json.loads(line) for line in f if line.strip()]
        print(f"[Resume] Successfully restored states. Resuming from update {current_update} to {target_updates}.\n", flush=True)
    else:
        # Step 0 Initial Benchmark Evaluation
        sum_0 = None
        if not skip_initial_eval:
            print("\n--- Running Step 0 Initial Evaluation for T1 ---", flush=True)
            eval_0 = validator.evaluate(model)
            sum_0 = {k: v for k, v in eval_0.items() if k != "predictions"}
            print(f"[T1 Step 0] Success@15px: {sum_0['primary_success_at_15px']*100:.2f}% | Mean FP/img: {sum_0['mean_false_tips_per_image']:.3f} | Neg FP/img: {sum_0['mean_false_tips_per_negative_image']:.3f}", flush=True)

        entry_0 = {
            "update": 0,
            "seg_loss": None,
            "tip_loss": None,
            "total_loss": None,
            "lr": 1e-4,
            "val_metrics": sum_0,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        history_entries.append(entry_0)
        with open(history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry_0) + "\n")

        save_tip_checkpoint(
            ckpt_dir / "update_0000.pt", model, optimizer, scheduler, scaler,
            0, best_success15, best_step, seed
        )

    # 5. Training Loop
    t_start = time.time()
    last_eval_res = None
    model.train()

    while current_update <= target_updates:
        entry = ledger[current_update]
        study_ids = entry["study_ids"]
        aug_list = entry["augmentation_parameters"]

        micro_data = []
        for sid, aug_params in zip(study_ids, aug_list):
            img_path = IMAGES_DIR / f"{sid}.jpg"
            pil_im = Image.open(img_path).convert("L")
            im_np = np.array(pil_im)
            orig_h, orig_w = im_np.shape

            coord_tf = CoordinateTransform.from_native_dims(orig_w, orig_h, target_size=768, stride=4)

            # Polylines and catheter target
            polylines = train_polylines.get(sid, [])
            is_pos = len(polylines) > 0

            mask_np = rasterize_study_polylines(polylines, (orig_h, orig_w), thickness=10) if is_pos else np.zeros((orig_h, orig_w), dtype=np.uint8)

            # Apply letterbox transform
            lb_img, meta = transform.apply_image(im_np)
            lb_mask = transform.apply_mask(mask_np, meta) if is_pos else np.zeros((768, 768), dtype=np.uint8)

            # Valid pixel mask
            valid_np = np.zeros((768, 768), dtype=np.uint8)
            valid_np[meta.pad_top : meta.pad_top + meta.new_h, meta.pad_left : meta.pad_left + meta.new_w] = 1

            pil_img = Image.fromarray(lb_img)
            pil_mask = Image.fromarray(lb_mask)
            pil_valid = Image.fromarray(valid_np)

            # Tip Supervision setup
            tips_native = study_tips_lookup.get(sid, [])
            is_fully_accounted = study_supervision_status.get(sid, False)

            if not is_pos:
                has_tip_sup = True
                hm_tensor, hm_valid = render_multitip_gaussian_heatmap([], coord_tf, sigma_hm=1.5)
            elif is_fully_accounted and tips_native:
                has_tip_sup = True
                hm_tensor, hm_valid = render_multitip_gaussian_heatmap(tips_native, coord_tf, sigma_hm=1.5)
            else:
                has_tip_sup = False
                hm_tensor, hm_valid = render_multitip_gaussian_heatmap([], coord_tf, sigma_hm=1.5)

            # Augmentations from immutable ledger
            if aug_params["hflip"]:
                pil_img = TF.hflip(pil_img)
                pil_mask = TF.hflip(pil_mask)
                pil_valid = TF.hflip(pil_valid)
                hm_tensor = TF.hflip(hm_tensor)
                hm_valid = TF.hflip(hm_valid)

            translate_768 = [aug_params["translate_x_pct"] * 768, aug_params["translate_y_pct"] * 768]
            translate_192 = [aug_params["translate_x_pct"] * 192, aug_params["translate_y_pct"] * 192]

            pil_img = TF.affine(pil_img, angle=aug_params["angle"], translate=translate_768, scale=aug_params["scale"], shear=0, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            pil_mask = TF.affine(pil_mask, angle=aug_params["angle"], translate=translate_768, scale=aug_params["scale"], shear=0, interpolation=TF.InterpolationMode.NEAREST, fill=0)
            pil_valid = TF.affine(pil_valid, angle=aug_params["angle"], translate=translate_768, scale=aug_params["scale"], shear=0, interpolation=TF.InterpolationMode.NEAREST, fill=0)

            hm_tensor = TF.affine(hm_tensor, angle=aug_params["angle"], translate=translate_192, scale=aug_params["scale"], shear=0, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            hm_valid = TF.affine(hm_valid, angle=aug_params["angle"], translate=translate_192, scale=aug_params["scale"], shear=0, interpolation=TF.InterpolationMode.NEAREST, fill=0)

            if aug_params["brightness"] != 1.0:
                pil_img = TF.adjust_brightness(pil_img, aug_params["brightness"])
            if aug_params["contrast"] != 1.0:
                pil_img = TF.adjust_contrast(pil_img, aug_params["contrast"])

            img_tensor = TF.to_tensor(pil_img).repeat(3, 1, 1)
            img_tensor = (img_tensor - mean) / std

            mask_tensor = torch.from_numpy((np.array(pil_mask) > 0).astype(np.float32)).unsqueeze(0)
            valid_tensor = torch.from_numpy((np.array(pil_valid) > 0).astype(np.float32)).unsqueeze(0)

            micro_data.append((img_tensor, mask_tensor, valid_tensor, is_pos, hm_tensor, hm_valid, has_tip_sup))

        # Effective-batch loss accumulation
        optimizer.zero_grad(set_to_none=True)
        accum_seg_loss = 0.0
        accum_tip_loss = 0.0
        accum_total_loss = 0.0

        total_valid_pixels = sum(float(vt.sum().item()) for _, _, vt, _, _, _, _ in micro_data)
        total_positives = sum(1 for _, _, _, p, _, _, _ in micro_data if p)
        total_tip_sup = sum(1 for _, _, _, _, _, _, sup in micro_data if sup)

        for img_t, mask_t, valid_t, is_pos, hm_t, hm_v, has_tip_sup in micro_data:
            img_b = img_t.unsqueeze(0).to(device)
            mask_b = mask_t.unsqueeze(0).to(device)
            valid_b = valid_t.unsqueeze(0).to(device)
            hm_b = hm_t.unsqueeze(0).to(device)
            hm_vb = hm_v.unsqueeze(0).to(device)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out = model(img_b)
                seg_logits = out["seg_logits"].float()
                tip_logits = out["tip_logits"].float()

                # 1. Segmentation BCE
                bce_raw = nn.functional.binary_cross_entropy_with_logits(
                    seg_logits, mask_b.float(), pos_weight=torch.tensor([10.0], device=device), reduction="none"
                )
                mb_bce_loss = (bce_raw * valid_b.float()).sum() / max(total_valid_pixels, 1.0)

                # 2. Segmentation Soft Dice
                probs_fp32 = torch.sigmoid(seg_logits)
                if is_pos and total_positives > 0:
                    inter = (probs_fp32 * mask_b.float() * valid_b.float()).sum()
                    card = (probs_fp32 * valid_b.float() + mask_b.float() * valid_b.float()).sum()
                    dice_score = (2.0 * inter + 1e-6) / (card + 1e-6)
                    mb_dice_loss = (1.0 - dice_score) / float(total_positives)
                else:
                    mb_dice_loss = torch.tensor(0.0, device=device)

                seg_sample_loss = mb_bce_loss + mb_dice_loss

                # 3. Tip Focal Loss (masked if not supervised)
                if has_tip_sup and total_tip_sup > 0:
                    tip_loss_sample, _ = tip_criterion(
                        tip_logits, hm_b, hm_vb, has_positive_tips=is_pos
                    )
                    tip_loss_eff = (tip_loss_sample * lambda_tip) / float(total_tip_sup)
                else:
                    tip_loss_eff = torch.tensor(0.0, device=device)

                total_sample_loss = seg_sample_loss + tip_loss_eff

            accum_seg_loss += seg_sample_loss.item()
            accum_tip_loss += tip_loss_eff.item()
            accum_total_loss += total_sample_loss.item()

            scaler.scale(total_sample_loss).backward()

        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        new_scale = scaler.get_scale()

        if new_scale >= old_scale:
            scheduler.step()

        gn_str = f"{grad_norm.item():.3f}" if (not torch.isnan(grad_norm) and not torch.isinf(grad_norm)) else "skipped"
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0

        if current_update % 5 == 0 or current_update == 1 or current_update == target_updates:
            elapsed = time.time() - t_start
            rate = current_update / max(elapsed, 0.1)
            cur_lr = scheduler.get_last_lr()[0]
            print(f"[T1 Joint] Upd {current_update:4d}/{target_updates:4d} | TotLoss: {accum_total_loss:.4f} (Seg: {accum_seg_loss:.4f}, Tip: {accum_tip_loss:.4f}) | GN: {gn_str} | LR: {cur_lr:.2e} | VRAM: {peak_vram:.0f}MB | {rate:.2f} upd/s", flush=True)

        # Periodic Evaluation
        is_interval = (current_update % eval_interval == 0)
        is_final = (current_update == target_updates)
        should_eval = is_interval or (is_final and run_eval_on_final)

        if should_eval:
            eval_res = validator.evaluate(model)
            clean_metrics = {k: v for k, v in eval_res.items() if k != "predictions"}
            last_eval_res = clean_metrics
            succ15 = clean_metrics["primary_success_at_15px"]

            print(f"\n>>> [T1 Upd {current_update}] Val Success@15px: {succ15*100:.2f}% | S@10px: {clean_metrics['success_at_10px']*100:.2f}% | Mean FP/img: {clean_metrics['mean_false_tips_per_image']:.3f} <<<\n", flush=True)

            entry_eval = {
                "update": current_update,
                "seg_loss": accum_seg_loss,
                "tip_loss": accum_tip_loss,
                "total_loss": accum_total_loss,
                "lr": scheduler.get_last_lr()[0],
                "val_metrics": clean_metrics,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            history_entries.append(entry_eval)
            with open(history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry_eval) + "\n")

            if succ15 > best_success15:
                best_success15 = succ15
                best_step = current_update
                save_tip_checkpoint(
                    ckpt_dir / "best_model.pt", model, optimizer, scheduler, scaler,
                    current_update, best_success15, best_step, seed, peak_vram
                )
                print(f"--> Saved NEW BEST model (Success@15: {best_success15*100:.2f}% at update {best_step})", flush=True)

            save_tip_checkpoint(
                ckpt_dir / f"update_{current_update:04d}.pt",
                model, optimizer, scheduler, scaler,
                current_update, best_success15, best_step, seed, peak_vram
            )

        current_update += 1

    # Save final model
    save_tip_checkpoint(
        ckpt_dir / "final_model.pt",
        model, optimizer, scheduler, scaler,
        target_updates, best_success15, best_step, seed, peak_vram
    )

    t_total = time.time() - t_start
    final_sha = hash_file(ckpt_dir / "final_model.pt")

    # Section 14 Completion Exports
    if last_eval_res:
        clean_sum = {k: v for k, v in last_eval_res.items() if k not in ["cohort_results", "predictions"]}
        clean_sum["total_training_time_minutes"] = t_total / 60.0
        clean_sum["peak_vram_mb"] = peak_vram
        clean_sum["checkpoint_final_sha256"] = final_sha

        with open(run_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
            json.dump(clean_sum, f, indent=2)
        with open(EXP_ROOT / "reports" / f"t1_joint_hrnet_seed_{seed}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(clean_sum, f, indent=2)

        if "predictions" in last_eval_res:
            with open(EXP_ROOT / "predictions" / f"t1_joint_hrnet_seed_{seed}_predictions.json", "w", encoding="utf-8") as f:
                json.dump(last_eval_res["predictions"], f, indent=2)

        per_study_rows = []
        decoded_cands_rows = []
        target_matches_rows = []
        failures_rows = []

        for cr in last_eval_res.get("cohort_results", []):
            sid = cr["study_id"]
            is_neg = cr["is_negative"]
            targets = cr["targets"]
            preds = cr["predictions"]

            m_res = match_study_tips(targets, preds, tolerance_px=15.0)

            per_study_rows.append({
                "study_id": sid,
                "is_negative": is_neg,
                "num_targets": len(targets),
                "num_predictions": len(preds),
                "num_matches_15px": m_res["num_matches"],
                "num_false_positives": m_res["num_false_positives"],
                "all_matched": (m_res["num_matches"] == len(targets)) if len(targets) > 0 else (len(preds) == 0),
            })

            for p_idx, p in enumerate(preds):
                decoded_cands_rows.append({
                    "study_id": sid,
                    "pred_idx": p_idx,
                    "native_x": p["native_x"],
                    "native_y": p["native_y"],
                    "conf": p["conf"],
                    "grid_x": p["grid_x"],
                    "grid_y": p["grid_y"],
                })

            for m in m_res["matches"]:
                t_idx, p_idx, d_px = m
                target_matches_rows.append({
                    "study_id": sid,
                    "target_idx": t_idx,
                    "pred_idx": p_idx,
                    "distance_native_px": d_px,
                    "target_x": targets[t_idx][0],
                    "target_y": targets[t_idx][1],
                    "pred_x": preds[p_idx]["native_x"],
                    "pred_y": preds[p_idx]["native_y"],
                    "conf": preds[p_idx]["conf"],
                })

            matched_targets = {m[0] for m in m_res["matches"]}
            for t_idx, t in enumerate(targets):
                if t_idx not in matched_targets:
                    min_dist = float("inf")
                    closest_conf = 0.0
                    for p in preds:
                        d = math.hypot(p["native_x"] - t[0], p["native_y"] - t[1])
                        if d < min_dist:
                            min_dist = d
                            closest_conf = p["conf"]
                    failures_rows.append({
                        "study_id": sid,
                        "failure_type": "missed_target",
                        "target_x": t[0],
                        "target_y": t[1],
                        "closest_pred_dist_px": min_dist if min_dist != float("inf") else -1.0,
                        "closest_pred_conf": closest_conf,
                    })

            matched_preds = {m[1] for m in m_res["matches"]}
            for p_idx, p in enumerate(preds):
                if p_idx not in matched_preds:
                    failures_rows.append({
                        "study_id": sid,
                        "failure_type": "false_positive_tip" if not is_neg else "false_positive_negative_study",
                        "target_x": -1.0,
                        "target_y": -1.0,
                        "closest_pred_dist_px": 0.0,
                        "closest_pred_conf": p["conf"],
                    })

        pd.DataFrame(per_study_rows).to_csv(run_dir / "per_study_metrics.csv", index=False)
        pd.DataFrame(decoded_cands_rows).to_csv(run_dir / "decoded_candidates.csv", index=False)
        pd.DataFrame(target_matches_rows).to_csv(run_dir / "target_matches.csv", index=False)
        pd.DataFrame(failures_rows).to_csv(run_dir / "failure_analysis.csv", index=False)

    with open(run_dir / "status.json", "w", encoding="utf-8") as f:
        json.dump({
            "status": "COMPLETED",
            "model": "HRNetJointTipNet",
            "seed": seed,
            "target_updates": target_updates,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "best_update": best_step,
            "best_success15": best_success15,
            "peak_vram_mb": peak_vram,
            "total_training_time_minutes": t_total / 60.0,
        }, f, indent=2)

    print(f"\n{'='*65}")
    print("=== COMPLETED T1 PILOT: HRNET JOINT SEGMENTATION & TIP HEATMAP ===")
    if last_eval_res:
        print(f"Final Success@15px: {last_eval_res['primary_success_at_15px']*100:.2f}% (Best: {best_success15*100:.2f}% @ step {best_step})")
        print(f"Success@5px:  {last_eval_res['success_at_5px']*100:.2f}%")
        print(f"Success@10px: {last_eval_res['success_at_10px']*100:.2f}%")
        print(f"Success@25px: {last_eval_res['success_at_25px']*100:.2f}%")
        print(f"Mean False Tips / Image: {last_eval_res['mean_false_tips_per_image']:.3f}")
        print(f"Mean False Tips / Negative Image: {last_eval_res['mean_false_tips_per_negative_image']:.3f}")
        print(f"Single-CVC Success@15: {last_eval_res['subgroup_single_cvc_success_15px']*100:.2f}%")
        print(f"Multi-CVC Success@15:  {last_eval_res['subgroup_multi_cvc_success_15px']*100:.2f}%")
    print(f"Peak VRAM: {peak_vram:.1f} MB")
    print(f"Total Run Time: {t_total/60:.1f} minutes")
    print(f"{'='*65}\n")

    return {
        "model": "HRNetJointTipNet",
        "seed": seed,
        "total_updates": target_updates,
        "total_training_time_sec": t_total,
        "peak_vram_mb": peak_vram,
        "best_update": best_step,
        "best_success15": best_success15,
        "final_metrics": last_eval_res,
        "checkpoint_final_sha256": final_sha,
        "history": history_entries,
    }


def main():
    parser = argparse.ArgumentParser(description="HRNet Joint Tip Pilot Runner")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--lambda_tip", type=float, default=1.0)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_training_pilot(
        target_updates=args.updates,
        eval_interval=args.eval_interval,
        seed=args.seed,
        output_dir=args.output_dir,
        lambda_tip=args.lambda_tip,
        resume_from=args.resume_from,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
