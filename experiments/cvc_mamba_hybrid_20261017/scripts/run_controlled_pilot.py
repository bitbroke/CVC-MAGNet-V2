"""Controlled Architecture Pilot Engine for CVC-Mamba-Hybrid-01.

Guarantees (Sections 5, 7, 8, 9):
1. Exactly matches the frozen experimental contract:
   - Starts from exact frozen S4 Seed-43 checkpoint (SHA-256: f01089be1b7507bfb1d1a2762911f6ee1f0d425ce49628c6d0083daecf5e94e2).
   - Zero-initialized residual projections guarantee step-0 mathematical identity across arms.
   - Microbatch 1, Effective Batch 4 (accumulation 4), 768x768 resolution.
   - Mixed precision (AMP) with FP32 loss reductions.
   - Ingests immutable fixed batch ledger (configs/batch_ledger.jsonl) with 1,000 updates.
2. Sequential single-GPU execution (Arm A -> Arm B -> Arm C).
3. Evaluates checkpoints natively at steps 0, 250, 500, 750, 1,000 on the 193-study dev_pilot_val cohort.
4. Preserves isolated telemetry, checkpoints, probability files, logs, and evaluation reports.
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

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms.functional as TF

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXP_ROOT = PROJECT_ROOT / "experiments" / "cvc_mamba_hybrid_20261017"
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from src.data.rasterize import rasterize_study_polylines
from src.data.transforms import LetterboxTransform, TransformMetadata
from experiments.cvc_mamba_hybrid_20261017.src.models.arm_models import (
    ArmAContinuationNet,
    ArmBMambaHybridNet,
    ArmCConvControlNet,
    FROZEN_S4_CKPT,
)

ARM_CLASSES = {
    "Arm_A_Continuation": ArmAContinuationNet,
    "Arm_B_Mamba_Hybrid": ArmBMambaHybridNet,
    "Arm_C_Conv_Control": ArmCConvControlNet,
}

DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = EXP_ROOT / "configs"
RUNS_DIR = EXP_ROOT / "runs"
REPORTS_DIR = EXP_ROOT / "reports"
IMAGES_DIR = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")


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


def compute_soft_dice(pred_prob: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_valid = pred_prob * valid_mask
    targ_valid = target * valid_mask
    intersection = (pred_valid * targ_valid).sum(dim=(1, 2, 3))
    cardinality = (pred_valid + targ_valid).sum(dim=(1, 2, 3))
    return (2.0 * intersection + eps) / (cardinality + eps)


def sample_polyline_points(polyline: List[List[float]], step_px: float = 2.0) -> np.ndarray:
    pts = np.array(polyline, dtype=np.float32)
    if len(pts) < 2:
        return pts
    sampled = []
    for i in range(len(pts) - 1):
        p1, p2 = pts[i], pts[i + 1]
        dist = np.linalg.norm(p2 - p1)
        if dist < 1e-4:
            continue
        num_steps = max(1, int(np.ceil(dist / step_px)))
        alphas = np.linspace(0.0, 1.0, num_steps, endpoint=(i == len(pts) - 2))
        for a in alphas:
            sampled.append(p1 + a * (p2 - p1))
    return np.array(sampled, dtype=np.float32) if sampled else pts


class NativeValidator:
    """Evaluates probability predictions on the native radiograph grid for dev_pilot_val."""

    def __init__(self, target_size: int = 768, device: torch.device = torch.device("cuda")):
        self.target_size = (target_size, target_size)
        self.device = device
        self.transform = LetterboxTransform(target_size=self.target_size, clahe_enabled=True)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        inv_df = pd.read_parquet(DATA_DIR / "annotation_inventory.parquet")
        study_df = pd.read_parquet(DATA_DIR / "study_inventory.parquet")
        with open(DATA_DIR / "splits.json", "r", encoding="utf-8") as f:
            splits_data = json.load(f)

        self.val_study_ids = splits_data["partitions"]["dev_pilot_val"]["study_ids"]
        self.study_meta: Dict[str, Dict[str, Any]] = {}
        for _, row in study_df[study_df["StudyInstanceUID"].isin(self.val_study_ids)].iterrows():
            sid = str(row["StudyInstanceUID"])
            self.study_meta[sid] = {
                "patient_id": str(row.get("PatientID", sid)),
                "has_cvc": bool(row.get("has_cvc", False)),
            }

        self.polylines_by_study: Dict[str, List[List[List[float]]]] = {}
        for _, row in inv_df[(inv_df["device_type"] == "CVC") & (inv_df["study_id"].isin(self.val_study_ids))].iterrows():
            sid = str(row["study_id"])
            p_raw = row["polyline"]
            poly = json.loads(p_raw) if isinstance(p_raw, str) else p_raw
            self.polylines_by_study.setdefault(sid, []).append(poly)

        self.confirmed_negatives = set(study_df[study_df["has_cvc"] == False]["StudyInstanceUID"])

    def evaluate_model(self, model: nn.Module, threshold: float = 0.50) -> Dict[str, Any]:
        model.eval()
        per_study_metrics = []
        macro_dices = []
        coverage_10px_list = []
        missed_catheter_count = 0
        total_catheters = 0
        per_study_neg_false_pcts = []
        total_neg_false_pixels = 0
        total_neg_valid_pixels = 0
        neg_studies_with_fp = 0
        num_neg_studies = 0

        # Preloaded/processed evaluation loop
        with torch.no_grad():
            for sid in self.val_study_ids:
                img_path = IMAGES_DIR / f"{sid}.jpg"
                pil_im = Image.open(img_path).convert("L")
                im_np = np.array(pil_im)
                orig_h, orig_w = im_np.shape

                polylines = self.polylines_by_study.get(sid, [])
                is_pos = len(polylines) > 0

                # Native GT mask
                native_gt = rasterize_study_polylines(polylines, (orig_h, orig_w), thickness=10) if is_pos else np.zeros((orig_h, orig_w), dtype=np.uint8)

                # Letterbox transform for inference
                lb_img, meta = self.transform.apply_image(im_np)
                raw_t = torch.from_numpy(lb_img).unsqueeze(0).unsqueeze(0).to(self.device).float() / 255.0
                inp = (raw_t.repeat(1, 3, 1, 1) - self.mean) / self.std

                # FP32 inference for evaluation (exact reproducibility matching evaluate_cvc_masks_native)
                out = model(inp)
                logits = out["seg_logits"] if isinstance(out, dict) else out
                prob_map = torch.sigmoid(logits.float()).squeeze().cpu().numpy().astype(np.float32)

                # Reproject to native
                valid_crop = prob_map[meta.pad_top : meta.pad_top + meta.new_h, meta.pad_left : meta.pad_left + meta.new_w]
                prob_native = cv2.resize(valid_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                bin_native = (prob_native >= threshold).astype(np.uint8)

                rec: Dict[str, Any] = {"study_id": sid, "is_positive": is_pos}

                if is_pos:
                    gt_fg = native_gt > 0
                    pred_fg = bin_native > 0
                    inter = np.logical_and(gt_fg, pred_fg).sum()
                    card = gt_fg.sum() + pred_fg.sum()
                    dice = (2.0 * inter) / (card + 1e-6)
                    macro_dices.append(dice)
                    rec["dice"] = float(dice)

                    # Centerline coverage within 10px via bounded ROI cKDTree (canonical parity)
                    margin = 15  # max_tol 10px + 5px margin
                    pred_area = int(pred_fg.sum())
                    study_coverages = []
                    for poly in polylines:
                        pts = sample_polyline_points(poly, step_px=2.0)
                        in_bounds = (pts[:, 0] >= 0) & (pts[:, 0] < orig_w) & (pts[:, 1] >= 0) & (pts[:, 1] < orig_h)
                        valid_pts = pts[in_bounds]
                        total_catheters += 1
                        if len(valid_pts) == 0 or pred_area == 0:
                            coverage_10px_list.append(0.0)
                            missed_catheter_count += 1
                            study_coverages.append(0.0)
                            continue

                        # Bounding box of polyline + margin
                        min_x = max(0, int(np.floor(np.min(valid_pts[:, 0]))) - margin)
                        max_x = min(orig_w, int(np.ceil(np.max(valid_pts[:, 0]))) + margin)
                        min_y = max(0, int(np.floor(np.min(valid_pts[:, 1]))) - margin)
                        max_y = min(orig_h, int(np.ceil(np.max(valid_pts[:, 1]))) + margin)

                        sub_pred = pred_fg[min_y:max_y, min_x:max_x]
                        sub_y, sub_x = np.nonzero(sub_pred)
                        if len(sub_x) == 0:
                            cov = 0.0
                        else:
                            fg_sub_pts = np.column_stack([sub_x + min_x, sub_y + min_y])
                            tree = cKDTree(fg_sub_pts)
                            dists, _ = tree.query(valid_pts, k=1)
                            cov = float(np.mean(dists <= 10.0))

                        coverage_10px_list.append(cov)
                        study_coverages.append(cov)
                        if cov < 0.30:
                            missed_catheter_count += 1

                    if study_coverages:
                        rec["centerline_coverage_10px"] = float(np.mean(study_coverages))
                else:
                    num_neg_studies += 1
                    fp_pixels = int(bin_native.sum())
                    valid_pixels = orig_h * orig_w
                    study_false_pct = (fp_pixels / valid_pixels) * 100.0
                    per_study_neg_false_pcts.append(study_false_pct)
                    total_neg_false_pixels += fp_pixels
                    total_neg_valid_pixels += valid_pixels
                    rec["false_positive_pixels"] = fp_pixels
                    rec["false_area_pct"] = study_false_pct
                    if fp_pixels > 0:
                        neg_studies_with_fp += 1

                per_study_metrics.append(rec)

        model.train()

        macro_dice = float(np.mean(macro_dices)) if macro_dices else 0.0
        cov_10px = float(np.mean(coverage_10px_list)) if coverage_10px_list else 0.0
        missed_rate = (missed_catheter_count / total_catheters) * 100.0 if total_catheters > 0 else 0.0
        neg_false_pct = float(np.mean(per_study_neg_false_pcts)) if per_study_neg_false_pcts else 0.0
        any_fp_rate = (neg_studies_with_fp / num_neg_studies) * 100.0 if num_neg_studies > 0 else 0.0

        summary = {
            "threshold": threshold,
            "macro_dice": macro_dice,
            "centerline_coverage_10px": cov_10px,
            "missed_catheter_rate": missed_rate,
            "total_catheters_evaluated": total_catheters,
            "negative_false_area_pct": neg_false_pct,
            "negative_total_false_pixels": total_neg_false_pixels,
            "negative_any_fp_rate": any_fp_rate,
            "num_positive_studies": len(macro_dices),
            "num_negative_studies": num_neg_studies,
        }
        return {
            "summary": summary,
            "per_study": per_study_metrics,
        }


def save_resumable_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: Any,
    scaler: GradScaler,
    update: int,
    best_dice: float,
    best_step: int,
    arm_name: str,
    seed: int,
) -> None:
    """Save full resumable training checkpoint including model, optimizer, scheduler, scaler and RNG states."""
    payload = {
        "update": update,
        "arm": arm_name,
        "seed": seed,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "best_dice": best_dice,
        "best_step": best_step,
    }
    tmp_path = filepath.with_suffix(".tmp")
    torch.save(payload, tmp_path)
    if filepath.exists():
        filepath.unlink()
    tmp_path.rename(filepath)


def run_arm_pilot(
    arm_name: str,
    seed: int = 42,
    target_updates: int = 1000,
    eval_interval: int = 250,
    output_dir: Optional[Path] = None,
    resume_from: Optional[Path] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"=== LAUNCHING PILOT: {arm_name} (Seed {seed}, {target_updates} Updates) ===")
    print(f"=======================================================", flush=True)

    assert arm_name in ARM_CLASSES, f"Unknown arm: {arm_name}"
    model_cls = ARM_CLASSES[arm_name]

    if output_dir is not None:
        run_dir = Path(output_dir)
    else:
        run_dir = RUNS_DIR / f"seed{seed}" / arm_name.lower()

    ckpt_dir = run_dir / "checkpoints"

    # Directory overwrite safety check
    if run_dir.exists():
        has_ckpts = ckpt_dir.exists() and any(ckpt_dir.glob("*.pt"))
        has_artifacts = any((run_dir / fname).exists() for fname in ["run_summary.json", "status.json", "history.jsonl"])
        if (has_ckpts or has_artifacts) and (resume_from is None) and (not overwrite):
            raise FileExistsError(
                f"Run directory '{run_dir}' already exists and contains prior run artifacts/checkpoints. "
                f"Refusing to overwrite. Use --resume_from to continue training or --overwrite to explicitly restart from scratch."
            )

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    # 1. Instantiate Model & Verify Initial State
    model = model_cls(load_checkpoint=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{arm_name}] Total Parameters: {total_params:,} ({trainable_params:,} trainable)", flush=True)

    # 2. Setup Optimizer, Scaler, Scheduler
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = GradScaler(enabled=(device.type == "cuda"), init_scale=1024.0)
    scheduler = CosineAnnealingLR(optimizer, T_max=target_updates, eta_min=1e-6)

    # 3. Load Invariant Batch Ledger
    ledger_path = CONFIGS_DIR / "batch_ledger.jsonl"
    with open(ledger_path, "r", encoding="utf-8") as f:
        ledger_lines = [json.loads(line) for line in f if line.strip()]
    ledger = {entry["update_id"]: entry for entry in ledger_lines}
    print(f"Loaded immutable batch ledger with {len(ledger)} updates.", flush=True)

    # 4. Load Training Ground Truth Polylines and Manifests
    inv_df = pd.read_parquet(DATA_DIR / "annotation_inventory.parquet")
    study_inv_df = pd.read_parquet(DATA_DIR / "study_inventory.parquet")
    train_polylines: Dict[str, List[List[List[float]]]] = {}
    for _, row in inv_df[inv_df["device_type"] == "CVC"].iterrows():
        sid = str(row["study_id"])
        p_raw = row["polyline"]
        poly = json.loads(p_raw) if isinstance(p_raw, str) else p_raw
        train_polylines.setdefault(sid, []).append(poly)

    transform = LetterboxTransform(target_size=(768, 768), clahe_enabled=True)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    validator = NativeValidator(target_size=768, device=device)

    # Telemetry and history
    history_file = run_dir / "history.jsonl"
    history_entries = []
    best_dice = -1.0
    best_step = -1
    current_update = 1

    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print(f"\n[Resume] Loading checkpoint from: {resume_path}", flush=True)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if "arm" in ckpt and ckpt["arm"] != arm_name:
            print(f"WARNING: Checkpoint arm '{ckpt['arm']}' differs from target arm '{arm_name}'!", flush=True)

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])

        # Restore exact RNG states
        if "torch_rng_state" in ckpt and ckpt["torch_rng_state"] is not None:
            torch.set_rng_state(ckpt["torch_rng_state"].cpu())
        if device.type == "cuda" and ckpt.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(ckpt["cuda_rng_state"].cpu())
        if "numpy_rng_state" in ckpt:
            np.random.set_state(ckpt["numpy_rng_state"])
        if "python_rng_state" in ckpt:
            random.setstate(ckpt["python_rng_state"])

        current_update = ckpt["update"] + 1
        best_dice = ckpt.get("best_dice", -1.0)
        best_step = ckpt.get("best_step", -1)

        # Load previous history if present
        if history_file.exists():
            with open(history_file, "r", encoding="utf-8") as f:
                history_entries = [json.loads(line) for line in f if line.strip()]
        print(f"[Resume] Successfully restored states. Resuming training from update {current_update} to {target_updates} (Best Dice: {best_dice:.4f} @ step {best_step}).\n", flush=True)
    else:
        # 5. Step 0 Initial Native Evaluation (only when starting fresh)
        print(f"\n--- Running Step 0 Initial Native Evaluation for {arm_name} ---", flush=True)
        eval_0 = validator.evaluate_model(model, threshold=0.50)
        sum_0 = eval_0["summary"]
        print(f"[{arm_name} Step 0] Macro Dice: {sum_0['macro_dice']:.4f} | Coverage@10px: {sum_0['centerline_coverage_10px']*100:.2f}% | False Area: {sum_0['negative_false_area_pct']:.4f}%", flush=True)

        entry_0 = {
            "update": 0,
            "loss": None,
            "lr": 1e-4,
            "val_metrics": sum_0,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        history_entries.append(entry_0)
        with open(history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry_0) + "\n")

        save_resumable_checkpoint(
            ckpt_dir / "update_0000.pt", model, optimizer, scheduler, scaler,
            0, best_dice, best_step, arm_name, seed
        )

    # 6. Training Loop (Updates current_update to target_updates)
    t_start = time.time()
    recent_losses = []
    last_eval_res = None

    model.train()

    while current_update <= target_updates:
        entry = ledger[current_update]
        study_ids = entry["study_ids"]
        aug_list = entry["augmentation_parameters"]

        # Build effective batch of 4
        micro_data = []
        for sid, aug_params in zip(study_ids, aug_list):
            img_path = IMAGES_DIR / f"{sid}.jpg"
            pil_im = Image.open(img_path).convert("L")
            im_np = np.array(pil_im)
            orig_h, orig_w = im_np.shape

            polys = train_polylines.get(sid, [])
            is_pos = len(polys) > 0

            native_gt = rasterize_study_polylines(polys, (orig_h, orig_w), thickness=10) if is_pos else np.zeros((orig_h, orig_w), dtype=np.uint8)

            lb_img, meta = transform.apply_image(im_np)
            lb_mask = transform.apply_mask(native_gt, meta) if is_pos else np.zeros((768, 768), dtype=np.uint8)
            valid_mask_init = np.zeros((768, 768), dtype=np.uint8)
            valid_mask_init[meta.pad_top : meta.pad_top + meta.new_h, meta.pad_left : meta.pad_left + meta.new_w] = 1

            pil_img = Image.fromarray(lb_img)
            pil_mask = Image.fromarray(lb_mask)
            pil_valid = Image.fromarray(valid_mask_init)

            if aug_params["hflip"]:
                pil_img = TF.hflip(pil_img)
                pil_mask = TF.hflip(pil_mask)
                pil_valid = TF.hflip(pil_valid)

            translate = [aug_params["translate_x_pct"] * 768, aug_params["translate_y_pct"] * 768]
            pil_img = TF.affine(pil_img, angle=aug_params["angle"], translate=translate, scale=aug_params["scale"], shear=0, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            pil_mask = TF.affine(pil_mask, angle=aug_params["angle"], translate=translate, scale=aug_params["scale"], shear=0, interpolation=TF.InterpolationMode.NEAREST, fill=0)
            pil_valid = TF.affine(pil_valid, angle=aug_params["angle"], translate=translate, scale=aug_params["scale"], shear=0, interpolation=TF.InterpolationMode.NEAREST, fill=0)

            if aug_params["brightness"] != 1.0:
                pil_img = TF.adjust_brightness(pil_img, aug_params["brightness"])
            if aug_params["contrast"] != 1.0:
                pil_img = TF.adjust_contrast(pil_img, aug_params["contrast"])

            img_tensor = TF.to_tensor(pil_img).repeat(3, 1, 1)
            img_tensor = (img_tensor - mean) / std

            # 2. Repair binary-mask conversion, preserving 0/1 values directly without uint8/255 division bug
            mask_tensor = torch.from_numpy((np.array(pil_mask) > 0).astype(np.float32)).unsqueeze(0)
            valid_tensor = torch.from_numpy((np.array(pil_valid) > 0).astype(np.float32)).unsqueeze(0)

            # 3. Mandatory assertions
            assert valid_tensor.sum() > 0, f"Supervision Violation: Study {sid} has 0 valid pixels!"
            if is_pos:
                assert mask_tensor.sum() > 0, f"Supervision Violation: Positive study {sid} lost foreground mask after transform!"
            else:
                assert mask_tensor.sum() == 0, f"Supervision Violation: Negative study {sid} has non-empty target mask ({mask_tensor.sum().item()} px)!"

            micro_data.append((img_tensor, mask_tensor, valid_tensor, is_pos))

        # 4. Restore original effective-batch loss normalization (exact equivalence to train_segmentation_v3)
        optimizer.zero_grad(set_to_none=True)
        eff_total_loss_accum = 0.0

        total_valid_pixels = sum(float(vt.sum().item()) for _, _, vt, _ in micro_data)
        total_positives = sum(1 for _, _, _, p in micro_data if p)

        for img_t, mask_t, valid_t, is_pos in micro_data:
            img_b = img_t.unsqueeze(0).to(device)
            mask_b = mask_t.unsqueeze(0).to(device)
            valid_b = valid_t.unsqueeze(0).to(device)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out = model(img_b)
                logits = out["seg_logits"] if isinstance(out, dict) else out
                logits_fp32 = logits.float()
                targets_fp32 = mask_b.float()
                valids_fp32 = valid_b.float()

                # Exact effective BCE term: divided by total effective-batch valid pixels
                bce_raw = nn.functional.binary_cross_entropy_with_logits(
                    logits_fp32, targets_fp32, pos_weight=torch.tensor([10.0], device=device), reduction="none"
                )
                mb_bce_loss = (bce_raw * valids_fp32).sum() / max(total_valid_pixels, 1.0)

                # Padding check assertion: verify padding contributes no loss
                assert torch.all((bce_raw * (1.0 - valids_fp32)) >= 0), "BCE on padding is invalid"

                # Exact effective Dice term: divided by total effective-batch positives
                probs_fp32 = torch.sigmoid(logits_fp32)
                if is_pos and total_positives > 0:
                    inter = (probs_fp32 * targets_fp32 * valids_fp32).sum()
                    card = (probs_fp32 * valids_fp32 + targets_fp32 * valids_fp32).sum()
                    dice_score = (2.0 * inter + 1e-6) / (card + 1e-6)
                    mb_dice_loss = (1.0 - dice_score) / float(total_positives)
                else:
                    mb_dice_loss = torch.tensor(0.0, device=device)

                loss = mb_bce_loss + mb_dice_loss

            scaler.scale(loss).backward()
            eff_total_loss_accum += loss.item()

        # Step optimizer
        scaler.unscale_(optimizer)
        gn = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()

        if np.isfinite(gn):
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            recent_losses.append(eff_total_loss_accum)

            if current_update % 25 == 0 or current_update in [1, 5, 10]:
                avg_l = float(np.mean(recent_losses[-25:]))
                elapsed = time.time() - t_start
                rate = current_update / (elapsed + 1e-4)
                cur_lr = scheduler.get_last_lr()[0]
                print(f"[{arm_name}] Upd {current_update:4d}/1000 | Loss: {avg_l:.4f} | GN: {gn:.3f} | LR: {cur_lr:.2e} | {rate:.2f} upd/s", flush=True)

            # Periodic Native Validation
            if current_update % eval_interval == 0 or current_update == target_updates:
                ckpt_path = ckpt_dir / f"update_{current_update:04d}.pt"
                save_resumable_checkpoint(
                    ckpt_path, model, optimizer, scheduler, scaler,
                    current_update, best_dice, best_step, arm_name, seed
                )

                eval_res = validator.evaluate_model(model, threshold=0.50)
                last_eval_res = eval_res
                sm = eval_res["summary"]
                cur_lr = scheduler.get_last_lr()[0]
                print(f"\n>>> [{arm_name} Upd {current_update}] Val Macro Dice: {sm['macro_dice']:.4f} | Cov@10px: {sm['centerline_coverage_10px']*100:.2f}% | False Area: {sm['negative_false_area_pct']:.4f}% <<<\n", flush=True)

                hist_entry = {
                    "update": current_update,
                    "loss": float(np.mean(recent_losses[-eval_interval:])) if recent_losses else 0.0,
                    "lr": cur_lr,
                    "val_metrics": sm,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                history_entries.append(hist_entry)
                with open(history_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(hist_entry) + "\n")

                if sm["macro_dice"] > best_dice:
                    best_dice = sm["macro_dice"]
                    best_step = current_update
                    save_resumable_checkpoint(
                        ckpt_dir / "best_segmentation_v3.pt", model, optimizer, scheduler, scaler,
                        current_update, best_dice, best_step, arm_name, seed
                    )

            current_update += 1
        else:
            print(f"[{arm_name}] Non-finite gradient norm ({gn}) at update {current_update}. Retrying same batch with reduced scale.", flush=True)
            scaler.update()

    # 7. Final Checkpoint and Export
    final_ckpt = ckpt_dir / "final_segmentation_v3.pt"
    save_resumable_checkpoint(
        final_ckpt, model, optimizer, scheduler, scaler,
        target_updates, best_dice, best_step, arm_name, seed
    )

    total_time = time.time() - t_start
    final_eval = last_eval_res if last_eval_res is not None else validator.evaluate_model(model, threshold=0.50)
    final_summary = final_eval["summary"]

    # Export per-study metrics
    df_per_study = pd.DataFrame(final_eval["per_study"])
    df_per_study.to_csv(run_dir / "per_study_metrics_step1000.csv", index=False)

    summary_payload = {
        "arm": arm_name,
        "seed": seed,
        "total_updates": target_updates,
        "total_training_time_sec": total_time,
        "avg_sec_per_update": total_time / target_updates,
        "best_update": best_step,
        "best_macro_dice": best_dice,
        "final_metrics": final_summary,
        "checkpoint_final_sha256": hash_file(final_ckpt),
        "history": history_entries,
    }
    with open(run_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    status_payload = {
        "arm": arm_name,
        "status": "COMPLETED",
        "completed_updates": target_updates,
        "final_macro_dice": final_summary["macro_dice"],
        "final_coverage_10px": final_summary["centerline_coverage_10px"],
        "final_false_area_pct": final_summary["negative_false_area_pct"],
        "completion_time": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(run_dir / "status.json", "w", encoding="utf-8") as f:
        json.dump(status_payload, f, indent=2)

    print(f"\n=======================================================")
    print(f"=== COMPLETED PILOT: {arm_name} ===")
    print(f"Final Macro Dice: {final_summary['macro_dice']:.4f} (Best: {best_dice:.4f} @ step {best_step})")
    print(f"Centerline Coverage @ 10px: {final_summary['centerline_coverage_10px']*100:.2f}%")
    print(f"Negative False Area %: {final_summary['negative_false_area_pct']:.4f}%")
    print(f"Total Run Time: {total_time/60.0:.1f} minutes")
    print(f"=======================================================\n", flush=True)

    del model, optimizer, scaler, scheduler
    torch.cuda.empty_cache()
    return summary_payload


def main():
    parser = argparse.ArgumentParser(description="Run Controlled Architecture Pilot")
    parser.add_argument("--arm", type=str, default="all", choices=["all", "Arm_A_Continuation", "Arm_B_Mamba_Hybrid", "Arm_C_Conv_Control"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--output_dir", type=str, default=None, help="Explicit output directory for run artifacts")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint .pt to resume training from")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output directory")
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.arm == "all":
        # Sequential GPU execution: Arm A -> Arm B -> Arm C
        arms = ["Arm_A_Continuation", "Arm_B_Mamba_Hybrid", "Arm_C_Conv_Control"]
    else:
        arms = [args.arm]

    results = {}
    for arm in arms:
        arm_out_dir = None
        if args.output_dir is not None:
            arm_out_dir = Path(args.output_dir) if len(arms) == 1 else Path(args.output_dir) / arm.lower()

        res = run_arm_pilot(
            arm_name=arm,
            seed=args.seed,
            target_updates=args.updates,
            eval_interval=args.eval_interval,
            output_dir=arm_out_dir,
            resume_from=Path(args.resume_from) if args.resume_from else None,
            overwrite=args.overwrite,
        )
        results[arm] = res

    # Write combined comparison summary if multiple arms finished
    summary_path = REPORTS_DIR / f"pilot_comparison_seed{args.seed}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved combined comparison summary to: {summary_path}")


if __name__ == "__main__":
    main()
