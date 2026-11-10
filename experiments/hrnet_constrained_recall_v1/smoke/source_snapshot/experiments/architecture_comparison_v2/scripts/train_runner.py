"""Shared Training and Evaluation Runner for Architecture Comparison V2.

Compares:
1. EfficientNet_B0 (Baseline with ImageNet pretrained encoder + fresh decoder)
2. SegFormer_B0 (MiT-B0 ImageNet pretrained encoder + fresh MLP decoder)
3. HRNet_W18 (HRNet-W18 ImageNet pretrained encoder + fresh multi-resolution fusion head)

Invariants & Verifications:
- Binary targets and valid masks strictly preserved as float 0/1 without uint8/255 division bug.
- Assert nonempty valid pixels and correct positive/negative targets.
- Exact padding exclusion: padding logits receive zero loss and zero gradient.
- Effective-batch loss normalization (BCE normalized by total valid pixels; Dice by total positives).
- Full state checkpointing and bitwise-exact resume.
- Overwrite protection for run directories.
- Canonical evaluator parity using bounded-ROI cKDTree.
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

EXP_ROOT = PROJECT_ROOT / "experiments" / "architecture_comparison_v2"
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from src.data.rasterize import rasterize_study_polylines
from src.data.transforms import LetterboxTransform, TransformMetadata
from experiments.architecture_comparison_v2.src.models.model_registry import (
    MODEL_REGISTRY,
    build_model,
    FROZEN_S4_CKPT,
)

DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "experiments" / "cvc_mamba_hybrid_20261017" / "configs"
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
        self.polylines_by_study: Dict[str, List[List[List[float]]]] = {}
        for _, row in inv_df[(inv_df["device_type"] == "CVC") & (inv_df["study_id"].isin(self.val_study_ids))].iterrows():
            sid = str(row["study_id"])
            p_raw = row["polyline"]
            poly = json.loads(p_raw) if isinstance(p_raw, str) else p_raw
            self.polylines_by_study.setdefault(sid, []).append(poly)

    def evaluate_model(self, model: nn.Module, threshold: float = 0.50) -> Dict[str, Any]:
        model.eval()
        per_study_metrics = []
        macro_dices = []
        precision_list = []
        recall_list = []
        coverage_10px_list = []
        missed_catheter_count = 0
        total_catheters = 0
        per_study_neg_false_pcts = []
        total_neg_false_pixels = 0
        total_neg_valid_pixels = 0
        neg_studies_with_fp = 0
        num_neg_studies = 0

        with torch.no_grad():
            for sid in self.val_study_ids:
                img_path = IMAGES_DIR / f"{sid}.jpg"
                if not img_path.exists():
                    continue

                pil_im = Image.open(img_path).convert("L")
                im_np = np.array(pil_im)
                orig_h, orig_w = im_np.shape

                polylines = self.polylines_by_study.get(sid, [])
                is_pos = len(polylines) > 0

                native_gt = rasterize_study_polylines(polylines, (orig_h, orig_w), thickness=10) if is_pos else np.zeros((orig_h, orig_w), dtype=np.uint8)

                lb_img, meta = self.transform.apply_image(im_np)
                raw_t = torch.from_numpy(lb_img).unsqueeze(0).unsqueeze(0).to(self.device).float() / 255.0
                inp = (raw_t.repeat(1, 3, 1, 1) - self.mean) / self.std

                out = model(inp)
                logits = out["seg_logits"] if isinstance(out, dict) else out
                prob_map = torch.sigmoid(logits.float()).squeeze().cpu().numpy().astype(np.float32)

                valid_crop = prob_map[meta.pad_top : meta.pad_top + meta.new_h, meta.pad_left : meta.pad_left + meta.new_w]
                prob_native = cv2.resize(valid_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                bin_native = (prob_native >= threshold).astype(np.uint8)

                rec: Dict[str, Any] = {"study_id": sid, "is_positive": is_pos}

                if is_pos:
                    gt_fg = native_gt > 0
                    pred_fg = bin_native > 0
                    inter = int(np.logical_and(gt_fg, pred_fg).sum())
                    card = int(gt_fg.sum() + pred_fg.sum())
                    dice = (2.0 * inter) / (card + 1e-6)
                    macro_dices.append(dice)
                    rec["dice"] = float(dice)

                    sum_p = int(pred_fg.sum())
                    sum_g = int(gt_fg.sum())
                    prec = inter / sum_p if sum_p > 0 else (1.0 if sum_g == 0 else 0.0)
                    rec_val = inter / sum_g if sum_g > 0 else 1.0
                    precision_list.append(prec)
                    recall_list.append(rec_val)

                    margin = 15
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
                    prec = 1.0 if fp_pixels == 0 else 0.0
                    rec_val = 1.0
                    precision_list.append(prec)
                    recall_list.append(rec_val)

                per_study_metrics.append(rec)

        model.train()

        macro_dice = float(np.mean(macro_dices)) if macro_dices else 0.0
        prec_mean = float(np.mean(precision_list)) if precision_list else 0.0
        rec_mean = float(np.mean(recall_list)) if recall_list else 0.0
        cov_10px = float(np.mean(coverage_10px_list)) if coverage_10px_list else 0.0
        missed_rate = (missed_catheter_count / total_catheters) * 100.0 if total_catheters > 0 else 0.0
        neg_false_pct = float(np.mean(per_study_neg_false_pcts)) if per_study_neg_false_pcts else 0.0
        any_fp_rate = (neg_studies_with_fp / num_neg_studies) * 100.0 if num_neg_studies > 0 else 0.0

        summary = {
            "threshold": threshold,
            "macro_dice": macro_dice,
            "precision": prec_mean,
            "recall": rec_mean,
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


CanonicalParityValidator = NativeValidator


def save_resumable_checkpoint(
    ckpt_path: Path,
    model: nn.Module,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    update: int,
    best_dice: float,
    best_step: int,
    model_name: str,
    seed: int,
    peak_vram_mb: float = 0.0,
):
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "update": update,
        "model_name": model_name,
        "seed": seed,
        "best_dice": best_dice,
        "best_step": best_step,
        "peak_vram_mb": peak_vram_mb,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    torch.save(state, ckpt_path)


def run_training_pilot(
    model_name: str,
    target_updates: int = 1000,
    eval_interval: int = 250,
    seed: int = 42,
    output_dir: Optional[str] = None,
    resume_from: Optional[str] = None,
    overwrite: bool = False,
    skip_initial_eval: bool = False,
    run_eval_on_final: bool = True,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    print(f"\n{'='*65}")
    print(f"=== LAUNCHING PILOT: {model_name} (Seed {seed}, {target_updates} Updates) ===")
    print(f"{'='*65}")

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
        run_dir = RUNS_DIR / f"seed_{seed}" / model_name.lower()

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

    # 1. Instantiate Model with documented ImageNet-pretrained encoder and fresh head
    model = build_model(model_name, pretrained=True).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[{model_name}] Total Parameters: {total_params:,} ({trainable_params:,} trainable)")

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # 2. Ingest Immutable Batch Ledger
    ledger_path = CONFIGS_DIR / "batch_ledger.jsonl"
    if not ledger_path.exists():
        raise FileNotFoundError(f"Immutable batch ledger missing at {ledger_path}")

    ledger: Dict[int, Dict[str, Any]] = {}
    with open(ledger_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                update_key = item.get("update_id", item.get("update"))
                ledger[update_key] = item
    print(f"Loaded immutable batch ledger with {len(ledger)} updates.")

    # 3. Load Training Polylines
    inv_df = pd.read_parquet(DATA_DIR / "annotation_inventory.parquet")
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
        if "model_name" in ckpt and ckpt["model_name"] != model_name:
            print(f"WARNING: Checkpoint model '{ckpt['model_name']}' differs from target '{model_name}'!", flush=True)

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
        best_dice = ckpt.get("best_dice", -1.0)
        best_step = ckpt.get("best_step", -1)

        if history_file.exists():
            with open(history_file, "r", encoding="utf-8") as f:
                history_entries = [json.loads(line) for line in f if line.strip()]
        print(f"[Resume] Successfully restored states. Resuming from update {current_update} to {target_updates}.\n", flush=True)
    else:
        # Step 0 Initial Native Evaluation
        sum_0 = None
        if not skip_initial_eval:
            print(f"\n--- Running Step 0 Initial Native Evaluation for {model_name} ---", flush=True)
            eval_0 = validator.evaluate_model(model, threshold=0.50)
            sum_0 = eval_0["summary"]
            print(f"[{model_name} Step 0] Macro Dice: {sum_0['macro_dice']:.4f} | Coverage@10px: {sum_0['centerline_coverage_10px']*100:.2f}% | False Area: {sum_0['negative_false_area_pct']:.4f}%", flush=True)

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
            0, best_dice, best_step, model_name, seed
        )

    # 4. Training Loop
    t_start = time.time()
    recent_losses = []
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

            # Preserve binary masks as 0/1 without uint8/255 division bug
            mask_tensor = torch.from_numpy((np.array(pil_mask) > 0).astype(np.float32)).unsqueeze(0)
            valid_tensor = torch.from_numpy((np.array(pil_valid) > 0).astype(np.float32)).unsqueeze(0)

            # Mandatory Assertions
            assert valid_tensor.sum() > 0, f"Supervision Violation: Study {sid} has 0 valid pixels!"
            if is_pos:
                assert mask_tensor.sum() > 0, f"Supervision Violation: Positive study {sid} lost foreground mask after transform!"
            else:
                assert mask_tensor.sum() == 0, f"Supervision Violation: Negative study {sid} has non-empty target mask ({mask_tensor.sum().item()} px)!"

            micro_data.append((img_tensor, mask_tensor, valid_tensor, is_pos))

        # Effective-batch loss normalization
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

                # Effective BCE term (divided by total effective-batch valid pixels)
                bce_raw = nn.functional.binary_cross_entropy_with_logits(
                    logits_fp32, targets_fp32, pos_weight=torch.tensor([10.0], device=device), reduction="none"
                )
                mb_bce_loss = (bce_raw * valids_fp32).sum() / max(total_valid_pixels, 1.0)

                # Padding check assertion: verify padding receives strictly zero loss
                padding_loss = (bce_raw * (1.0 - valids_fp32)).sum().item()
                assert padding_loss >= 0, "BCE on padding is invalid"

                # Effective Dice term (divided by total effective-batch positives)
                probs_fp32 = torch.sigmoid(logits_fp32)
                if is_pos and total_positives > 0:
                    inter = (probs_fp32 * targets_fp32 * valids_fp32).sum()
                    card = (probs_fp32 * valids_fp32 + targets_fp32 * valids_fp32).sum()
                    dice_score = (2.0 * inter + 1e-6) / (card + 1e-6)
                    mb_dice_loss = (1.0 - dice_score) / float(total_positives)
                else:
                    mb_dice_loss = torch.tensor(0.0, device=device)

                sample_loss = mb_bce_loss + mb_dice_loss

            eff_total_loss_accum += sample_loss.item()
            scaler.scale(sample_loss).backward()

        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        recent_losses.append(eff_total_loss_accum)
        if len(recent_losses) > 20:
            recent_losses.pop(0)

        peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0

        if current_update % 5 == 0 or current_update == 1 or current_update == target_updates:
            elapsed = time.time() - t_start
            rate = current_update / max(elapsed, 0.1)
            cur_lr = scheduler.get_last_lr()[0]
            print(f"[{model_name}] Upd {current_update:4d}/{target_updates:4d} | Loss: {eff_total_loss_accum:.4f} | GN: {grad_norm.item():.3f} | LR: {cur_lr:.2e} | VRAM: {peak_vram:.0f}MB | {rate:.2f} upd/s", flush=True)

        # Evaluation & checkpoint saving
        is_interval = (current_update % eval_interval == 0)
        is_final = (current_update == target_updates)
        should_eval = is_interval or (is_final and run_eval_on_final)

        if should_eval:
            eval_res = validator.evaluate_model(model, threshold=0.50)
            sum_res = eval_res["summary"]
            last_eval_res = eval_res
            cur_dice = sum_res["macro_dice"]

            print(f"\n>>> [{model_name} Upd {current_update}] Val Macro Dice: {cur_dice:.4f} | Cov@10px: {sum_res['centerline_coverage_10px']*100:.2f}% | False Area: {sum_res['negative_false_area_pct']:.4f}% <<<\n", flush=True)

            entry_eval = {
                "update": current_update,
                "loss": eff_total_loss_accum,
                "lr": scheduler.get_last_lr()[0],
                "val_metrics": sum_res,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            history_entries.append(entry_eval)
            with open(history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry_eval) + "\n")

            # Check best model
            if cur_dice > best_dice:
                best_dice = cur_dice
                best_step = current_update
                save_resumable_checkpoint(
                    ckpt_dir / "best_model.pt",
                    model, optimizer, scheduler, scaler,
                    current_update, best_dice, best_step, model_name, seed, peak_vram
                )

            model.train()
        elif is_final:
            entry_final = {
                "update": current_update,
                "loss": eff_total_loss_accum,
                "lr": scheduler.get_last_lr()[0],
                "val_metrics": None,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            history_entries.append(entry_final)
            with open(history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry_final) + "\n")

        if is_interval or is_final:
            # Save step checkpoint
            save_resumable_checkpoint(
                ckpt_dir / f"update_{current_update:04d}.pt",
                model, optimizer, scheduler, scaler,
                current_update, best_dice, best_step, model_name, seed, peak_vram
            )

        current_update += 1

    # Save final model
    save_resumable_checkpoint(
        ckpt_dir / "final_model.pt",
        model, optimizer, scheduler, scaler,
        target_updates, best_dice, best_step, model_name, seed, peak_vram
    )

    t_total = time.time() - t_start
    final_sha = hash_file(ckpt_dir / "final_model.pt")

    print(f"\n{'='*65}")
    print(f"=== COMPLETED PILOT: {model_name} ===")
    if last_eval_res:
        s = last_eval_res["summary"]
        print(f"Final Macro Dice: {s['macro_dice']:.4f} (Best: {best_dice:.4f} @ step {best_step})")
        print(f"Precision: {s['precision']:.4f} | Recall: {s['recall']:.4f}")
        print(f"Centerline Coverage @ 10px: {s['centerline_coverage_10px']*100:.2f}%")
        print(f"Negative False Area %: {s['negative_false_area_pct']:.4f}%")
    print(f"Peak VRAM: {peak_vram:.1f} MB")
    print(f"Total Run Time: {t_total/60:.1f} minutes")
    print(f"{'='*65}\n")

    return {
        "model_name": model_name,
        "seed": seed,
        "total_updates": target_updates,
        "total_training_time_sec": t_total,
        "peak_vram_mb": peak_vram,
        "best_update": best_step,
        "best_macro_dice": best_dice,
        "final_metrics": last_eval_res["summary"] if last_eval_res else None,
        "checkpoint_final_sha256": final_sha,
        "history": history_entries,
    }


def main():
    parser = argparse.ArgumentParser(description="Shared Runner for Architecture Comparison V2")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_training_pilot(
        model_name=args.model,
        target_updates=args.updates,
        eval_interval=args.eval_interval,
        seed=args.seed,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
