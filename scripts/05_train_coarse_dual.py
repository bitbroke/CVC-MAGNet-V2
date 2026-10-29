"""Script 05: Coarse Dual-Head Pilot Training & Controlled Interface.

Dual-Head Contract & Implementation Guarantees:
1. Mask Thickness & Preprocessing:
   - Native CVC polylines are rasterized with width 10 native pixels before resizing.
   - Radiographs undergo CLAHE at native resolution before isotropic letterbox resizing.
2. Coordinate Encoding Symmetry:
   - Heatmap targets use meta.native_to_model(tx, ty).
   - Peak decoding uses meta.model_to_native(px, py).
   - Both strictly obey OpenCV pixel-center convention (+0.5 / -0.5).
3. Partial-Review Exclusion Policy (Defensible First Pilot):
   - To avoid spatial ignore zone approximations and overlap ambiguities, partially reviewed
     multi-CVC studies are completely excluded from tip supervision (tip_mask = 0.0).
   - Tip supervision is strictly active on:
     * Completely reviewed positive studies (all CVCs audited): tip_mask = 1.0.
     * Confirmed negative studies (is_study_level_negative & ~has_cvc): tip_mask = 1.0.
   - All unreviewed or partially reviewed studies receive tip_mask = 0.0 (zero gradient to tip head),
     while receiving 100% standard segmentation mask supervision.
4. Three Rigorous Evaluation Paths on Identical Target Registry:
   - Arm A (Segmentation-only continuation):
     * Prediction: Segmentation -> skeleton endpoints (extract_distal_tips).
     * tip_logits is NEVER evaluated or decoded; checkpoint selection uses segmentation performance.
     * Purpose: Valid control.
   - Arm B (Dual-head continuation):
     * Prediction 1: Segmentation -> skeleton endpoints.
       Purpose: Measures whether auxiliary tip supervision improves segmentation quality.
     * Prediction 2: Tip-heatmap peaks.
       Purpose: Measures direct localization benefit from heatmap head.
   - All three evaluated with identical threshold-constrained Hungarian assignment (radius 15.0px).
   - Primary comparison: Final fixed-step (Step 1000) checkpoints.
   - Secondary comparison: Development-selected best checkpoints.
5. Research Qualifications & Provenance:
   - Audited validation targets (N=25) are labeled as provisional engineering-audit labels
     awaiting clinical attestation, not clinical accuracy.
   - Sensitivity notice: Each single case changes the 25-case score by 4.0 percentage points.
6. Execution Safety:
   - By default, verifies initialization, datasets, models, and preflight passes without
     launching training updates unless --execute_training is explicitly passed.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import CoarseDualDataset
from src.data.tip_sampling import supervision_groups, TipBalancedBatchSampler, UpdateCounter
from src.data.transforms import TransformMetadata
from src.infer.endpoints import extract_distal_tips
from src.models.checkpoint import save_checkpoint
from src.models.coarse_dual import (
    CoarseDualNet,
    decode_tip_peaks,
    generate_target_heatmap,
)
from src.models.losses import CoarseDualLoss, DiceLoss

# Dynamically import evaluation module
spec = importlib.util.spec_from_file_location("eval_mod", PROJECT_ROOT / "scripts" / "06_evaluate.py")
eval_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_mod)
threshold_constrained_matching = eval_mod.threshold_constrained_matching


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file for provenance."""
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]


def compute_model_backbone_hash(model: CoarseDualNet) -> str:
    """Compute SHA-256 hash of shared backbone parameters for arm verification."""
    h = hashlib.sha256()
    backbone_names = ["conv1", "conv2", "conv3", "conv4", "ppm", "up7", "conv7", "up8", "conv8", "up9", "conv9", "seg_head"]
    for name, param in sorted(model.named_parameters()):
        if any(name.startswith(bn) for bn in backbone_names):
            h.update(name.encode("utf-8"))
            h.update(param.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def set_seed(seed: int = 42):
    """Set deterministic seeds across PyTorch, NumPy, and Python."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_shared_backbone(model: CoarseDualNet, ckpt_path: Path) -> Dict[str, Any]:
    """Load pretrained shared backbone weights from CoarseSegNet checkpoint.
    
    Ensures bitwise identical backbone initialization across both comparison arms:
    - Arm A (Segmentation-only continuation)
    - Arm B (Dual-head continuation)
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Initial checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src_sd = ckpt["model_state_dict"]
    tgt_sd = model.state_dict()

    loaded_keys = []
    skipped_keys = []

    # Map head keys: CoarseSegNet 'head.' -> CoarseDualNet 'seg_head.'
    for k, v in src_sd.items():
        target_k = k
        if target_k not in tgt_sd and target_k.startswith("head."):
            target_k = target_k.replace("head.", "seg_head.")

        if target_k in tgt_sd and tgt_sd[target_k].shape == v.shape:
            tgt_sd[target_k].copy_(v)
            loaded_keys.append((k, target_k))
        else:
            skipped_keys.append(k)

    model.load_state_dict(tgt_sd)
    return {
        "loaded_keys_count": len(loaded_keys),
        "skipped_keys_count": len(skipped_keys),
        "total_target_keys": len(tgt_sd),
        "source_step": ckpt.get("step", 0),
        "source_effective_pos_weight": ckpt.get("effective_pos_weight", 50.0),
    }


def run_dual_validation(
    model: CoarseDualNet,
    val_loader: DataLoader,
    reviewed_val_df: pd.DataFrame,
    confirmed_negatives: Set[str],
    criterion: CoarseDualLoss,
    device: torch.device,
    mode: str = "dual_head",
    coarse_threshold: float = 0.30,
    peak_threshold: float = 0.30,
    peak_min_distance: int = 10,
    tolerances: Tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 30.0, 50.0),
) -> Dict[str, Any]:
    """Run standardized validation pass supporting the three designated evaluation paths.
    
    Evaluation Paths:
    - Path 1 (seg_only mode): Evaluates ONLY Segmentation -> skeleton endpoints.
      tip_logits is NEVER decoded or evaluated.
    - Path 2 (dual_head mode, seg_skeleton): Evaluates Segmentation -> skeleton endpoints.
      Measures whether auxiliary tip supervision improves segmentation quality.
    - Path 3 (dual_head mode, tip_heatmap): Evaluates Tip-heatmap peaks.
      Measures direct localization benefit from heatmap head.
      
    All paths evaluate against the same fixed target registry using identical
    threshold-constrained Hungarian matching (radius 15.0px).
    """
    model.eval()
    val_loss_total = 0.0
    val_loss_seg = 0.0
    val_loss_bce = 0.0
    val_loss_dice = 0.0
    val_loss_tip = 0.0
    total_batches = 0
    dice_scores = []

    # Index reviewed validation targets by study_id: sid -> List[{"cid": int, "x": float, "y": float}]
    reviewed_targets_by_study: Dict[str, List[Dict[str, Any]]] = {}
    for _, r in reviewed_val_df.iterrows():
        sid = str(r["study_id"])
        cid = int(r["catheter_instance_id"])
        tx = float(r["reviewed_tip_x"])
        ty = float(r["reviewed_tip_y"])
        reviewed_targets_by_study.setdefault(sid, []).append({"cid": cid, "x": tx, "y": ty})

    total_reviewed_targets = sum(len(v) for v in reviewed_targets_by_study.values())

    # Path 1 & 2: Skeleton endpoint matching trackers
    skel_matched_15_errors: List[float] = []
    skel_tolerance_counts: Dict[float, int] = {tol: 0 for tol in tolerances}
    skel_observed_nearest_dists: List[float] = []
    skel_targets_missing_cands = 0
    skel_neg_false_cands = 0

    # Path 3: Tip heatmap matching trackers (only active in dual_head mode)
    tip_matched_15_errors: List[float] = []
    tip_tolerance_counts: Dict[float, int] = {tol: 0 for tol in tolerances}
    tip_observed_nearest_dists: List[float] = []
    tip_targets_missing_cands = 0
    tip_neg_false_peaks = 0

    neg_studies_count = 0

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            tip_hms = batch["tip_heatmap"].to(device)
            tip_ms = batch["tip_mask"].to(device)
            tip_sms = batch["tip_spatial_mask"].to(device)

            with torch.amp.autocast("cuda"):
                seg_logits, tip_logits = model(imgs)
                loss, loss_dict = criterion(
                    seg_logits, tip_logits, masks, tip_hms,
                    tip_mask=tip_ms, tip_spatial_mask=tip_sms
                )

            val_loss_total += loss_dict["loss_total"]
            val_loss_seg += loss_dict["loss_seg"]
            val_loss_bce += loss_dict["loss_bce"]
            val_loss_dice += loss_dict["loss_dice"]
            val_loss_tip += loss_dict["loss_tip"]
            total_batches += 1

            # Mask Dice calculation
            seg_probs = torch.sigmoid(seg_logits)
            pred_fg = (seg_probs > coarse_threshold).float()
            gt_fg = (masks > 0.5).float()
            for b in range(imgs.shape[0]):
                p_b = pred_fg[b].view(-1)
                g_b = gt_fg[b].view(-1)
                inter = (p_b * g_b).sum().item()
                union = p_b.sum().item() + g_b.sum().item()
                d_val = (2.0 * inter / union) if union > 0 else (1.0 if g_b.sum().item() == 0 else 0.0)
                dice_scores.append(d_val)

            sids = batch["study_id"]
            meta_dict = batch["meta"]
            seg_probs_np = seg_probs.cpu().numpy()[:, 0]

            if mode == "dual_head":
                tip_probs_np = torch.sigmoid(tip_logits).cpu().numpy()[:, 0]
            else:
                tip_probs_np = None

            for b in range(len(sids)):
                sid = sids[b]
                meta_b = TransformMetadata(
                    scale=float(meta_dict["scale"][b]),
                    pad_x=float(meta_dict["pad_x"][b]),
                    pad_y=float(meta_dict["pad_y"][b]),
                    orig_w=int(meta_dict["orig_w"][b]),
                    orig_h=int(meta_dict["orig_h"][b]),
                    target_w=int(meta_dict["target_w"][b]),
                    target_h=int(meta_dict["target_h"][b]),
                    crop_x0=float(meta_dict["crop_x0"][b]),
                    crop_y0=float(meta_dict["crop_y0"][b]),
                    new_w=int(meta_dict["new_w"][b]),
                    new_h=int(meta_dict["new_h"][b]),
                    pad_left=int(meta_dict["pad_left"][b]),
                    pad_top=int(meta_dict["pad_top"][b]),
                )

                targets = reviewed_targets_by_study.get(sid, [])
                t_arr = np.array([[t["x"], t["y"]] for t in targets], dtype=np.float32) if len(targets) > 0 else np.zeros((0, 2), dtype=np.float32)
                is_neg = sid in confirmed_negatives

                if is_neg:
                    neg_studies_count += 1

                # -------------------------------------------------------------
                # Prediction Method 1: Segmentation -> Skeleton Endpoints
                # (Active for BOTH seg_only and dual_head arms)
                # -------------------------------------------------------------
                sp_b = seg_probs_np[b]
                raw_cands, _ = extract_distal_tips(sp_b, threshold=coarse_threshold, min_component_area=40)
                skel_coords = []
                for cand in raw_cands:
                    px_nat, py_nat = meta_b.model_to_native(cand.x, cand.y)
                    skel_coords.append([px_nat, py_nat])
                skel_arr = np.array(skel_coords, dtype=np.float32) if len(skel_coords) > 0 else np.zeros((0, 2), dtype=np.float32)

                if len(targets) > 0:
                    matched_skel_15, _, _ = threshold_constrained_matching(skel_arr, t_arr, radius=15.0)
                    for _, _, dist in matched_skel_15:
                        skel_matched_15_errors.append(dist)
                    for tol in tolerances:
                        m_tol, _, _ = threshold_constrained_matching(skel_arr, t_arr, radius=tol)
                        skel_tolerance_counts[tol] += len(m_tol)
                    if len(skel_arr) > 0:
                        for t in targets:
                            d = float(np.min(np.hypot(skel_arr[:, 0] - t["x"], skel_arr[:, 1] - t["y"])))
                            skel_observed_nearest_dists.append(d)
                    else:
                        skel_targets_missing_cands += len(targets)

                if is_neg:
                    skel_neg_false_cands += len(skel_coords)

                # -------------------------------------------------------------
                # Prediction Method 2: Tip Heatmap Peaks
                # (Active ONLY in dual_head mode; strictly SKIPPED in seg_only)
                # -------------------------------------------------------------
                if mode == "dual_head":
                    hm_b = tip_probs_np[b]
                    peaks = decode_tip_peaks(hm_b, threshold=peak_threshold, min_distance=peak_min_distance)
                    tip_coords = []
                    for p in peaks:
                        px_nat, py_nat = meta_b.model_to_native(p[0], p[1])
                        tip_coords.append([px_nat, py_nat])
                    tip_arr = np.array(tip_coords, dtype=np.float32) if len(tip_coords) > 0 else np.zeros((0, 2), dtype=np.float32)

                    if len(targets) > 0:
                        matched_tip_15, _, _ = threshold_constrained_matching(tip_arr, t_arr, radius=15.0)
                        for _, _, dist in matched_tip_15:
                            tip_matched_15_errors.append(dist)
                        for tol in tolerances:
                            m_tol, _, _ = threshold_constrained_matching(tip_arr, t_arr, radius=tol)
                            tip_tolerance_counts[tol] += len(m_tol)
                        if len(tip_arr) > 0:
                            for t in targets:
                                d = float(np.min(np.hypot(tip_arr[:, 0] - t["x"], tip_arr[:, 1] - t["y"])))
                                tip_observed_nearest_dists.append(d)
                        else:
                            tip_targets_missing_cands += len(targets)

                    if is_neg:
                        tip_neg_false_peaks += len(peaks)

    n_val = max(total_batches, 1)
    rev_denom = max(total_reviewed_targets, 1)

    skel_results = {
        "matched_15_count": len(skel_matched_15_errors),
        "success_at_15": len(skel_matched_15_errors) / rev_denom,
        "median_error_15": float(np.median(skel_matched_15_errors)) if len(skel_matched_15_errors) > 0 else None,
        "tolerance_success_rates": {tol: count / rev_denom for tol, count in skel_tolerance_counts.items()},
        "targets_total": rev_denom,
        "targets_with_candidates_count": len(skel_observed_nearest_dists),
        "targets_missing_candidates_count": skel_targets_missing_cands,
        "diagnostic_observed_nearest_dist": {
            "median": float(np.median(skel_observed_nearest_dists)) if skel_observed_nearest_dists else None,
            "q25": float(np.quantile(skel_observed_nearest_dists, 0.25)) if skel_observed_nearest_dists else None,
            "q75": float(np.quantile(skel_observed_nearest_dists, 0.75)) if skel_observed_nearest_dists else None,
            "note": "Nearest-prediction distance is an unconstrained diagnostic metric, NOT one-to-one matched error.",
        },
        "negative_false_candidates_total": skel_neg_false_cands,
        "negative_false_candidates_per_study": skel_neg_false_cands / max(neg_studies_count, 1),
    }

    if mode == "dual_head":
        tip_results = {
            "matched_15_count": len(tip_matched_15_errors),
            "success_at_15": len(tip_matched_15_errors) / rev_denom,
            "median_error_15": float(np.median(tip_matched_15_errors)) if len(tip_matched_15_errors) > 0 else None,
            "tolerance_success_rates": {tol: count / rev_denom for tol, count in tip_tolerance_counts.items()},
            "targets_total": rev_denom,
            "targets_with_candidates_count": len(tip_observed_nearest_dists),
            "targets_missing_candidates_count": tip_targets_missing_cands,
            "diagnostic_observed_nearest_dist": {
                "median": float(np.median(tip_observed_nearest_dists)) if tip_observed_nearest_dists else None,
                "q25": float(np.quantile(tip_observed_nearest_dists, 0.25)) if tip_observed_nearest_dists else None,
                "q75": float(np.quantile(tip_observed_nearest_dists, 0.75)) if tip_observed_nearest_dists else None,
                "note": "Nearest-prediction distance is an unconstrained diagnostic metric, NOT one-to-one matched error.",
            },
            "negative_false_peaks_total": tip_neg_false_peaks,
            "negative_false_peaks_per_study": tip_neg_false_peaks / max(neg_studies_count, 1),
        }
    else:
        tip_results = None

    return {
        "mode": mode,
        "val_loss_total": val_loss_total / n_val,
        "val_loss_seg": val_loss_seg / n_val,
        "val_loss_bce": val_loss_bce / n_val,
        "val_loss_dice": val_loss_dice / n_val,
        "val_loss_tip": val_loss_tip / n_val,
        "val_mask_dice_mean": float(np.mean(dice_scores)),
        "reviewed_targets_total": total_reviewed_targets,
        "negative_studies_evaluated": neg_studies_count,
        "seg_skeleton": skel_results,
        "tip_heatmap": tip_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Coarse Dual-Head Pilot Training & Controlled Interface")
    parser.add_argument("--config", type=str, default="configs/pilot.yaml", help="Path to base pilot YAML config")
    parser.add_argument("--mode", type=str, choices=["dual_head", "seg_only"], default="dual_head",
                        help="Comparison arm: dual_head (both heads) or seg_only (continuation control)")
    parser.add_argument("--max_steps", type=int, default=1000, help="Optimizer update budget")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (RTX 3050 VRAM constraint)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Cosine schedule min lr")
    parser.add_argument("--tip_weight", type=float, default=2.0, help="Distal tip loss weight")
    parser.add_argument("--tip_pos_weight", type=float, default=100.0, help="Distal tip BCE positive pixel weight")
    parser.add_argument("--tip_shaft_weight", type=float, default=1.0, help="Additional BCE weight multiplier for catheter shaft background (excluding tip neighborhoods)")
    parser.add_argument("--shaft_tip_exclusion_thresh", type=float, default=0.01, help="Heatmap target threshold defining tip exclusion region for shaft weighting")
    parser.add_argument("--seg_pos_weight", type=float, default=50.0, help="Segmentation BCE positive weight (repaired baseline ref)")
    parser.add_argument("--seg_bce_weight", type=float, default=1.0, help="Segmentation BCE loss weight")
    parser.add_argument("--seg_dice_weight", type=float, default=2.0, help="Segmentation Dice loss weight")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--init_ckpt", type=str, default="runs/pilot_coarse_repaired/checkpoints/best_coarse.pt",
                        help="Pretrained CoarseSegNet checkpoint for shared backbone initialization")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Run directory (defaults to runs/pilot_coarse_{mode})")
    parser.add_argument("--val_interval", type=int, default=200, help="Validation step interval")
    parser.add_argument("--partial_review_policy", type=str, choices=["exclude", "spatial_mask"], default="exclude",
                        help="Policy for partially reviewed images (default 'exclude' for first pilot)")
    parser.add_argument("--reviewed_tips_path", type=str, default="data/reviewed_distal_tips_v2.parquet",
                        help="Path to reviewed distal tips parquet table (default data/reviewed_distal_tips_v2.parquet)")
    parser.add_argument("--train_manifest_path", type=str, default=None,
                        help="Path to training expansion manifest parquet or csv. If None, training tips load from reviewed_tips_path split=='train' (Arm A Control). If provided, training tips load from this manifest (Arm B Experiment).")
    parser.add_argument("--execute_training", action="store_true", default=False,
                        help="CRITICAL GUARD: Only runs optimizer loops when explicitly specified.")
    parser.add_argument('--sampling', choices=['uniform', 'tip_balanced'], default='uniform')
    parser.add_argument('--max_attempts', type=int, default=None,
                        help='Abort after this many batches including AMP skips (default 2x budget)')
    args = parser.parse_args()
    if args.max_attempts is None:
        args.max_attempts = 2 * args.max_steps
    if args.max_steps < 1 or args.max_attempts < args.max_steps:
        parser.error('Require positive update budget and max_attempts >= max_steps')
    if args.sampling == 'tip_balanced' and args.batch_size != 4:
        parser.error('tip_balanced uses batch_size=4: one positive, one negative, two unknown')

    if args.output_dir is None:
        args.output_dir = f"runs/pilot_coarse_{args.mode}"

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Paths & Provenance
    config_path = PROJECT_ROOT / args.config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    splits_path = PROJECT_ROOT / "data" / "splits.json"
    inv_path = PROJECT_ROOT / "data" / "annotation_inventory.parquet"
    study_inv_path = PROJECT_ROOT / "data" / "study_inventory.parquet"
    reviewed_tips_path = PROJECT_ROOT / args.reviewed_tips_path
    raw_images_dir = Path(cfg["data"]["raw_dir"]) / "train"

    train_manifest_hash = "none"
    if args.train_manifest_path is not None:
        train_manifest_hash = compute_file_hash(PROJECT_ROOT / args.train_manifest_path)

    input_hashes = {
        "config_hash": compute_file_hash(config_path),
        "splits_hash": compute_file_hash(splits_path),
        "inventory_hash": compute_file_hash(inv_path),
        "study_inventory_hash": compute_file_hash(study_inv_path),
        "reviewed_tips_hash": compute_file_hash(reviewed_tips_path),
        "train_manifest_hash": train_manifest_hash,
        "init_ckpt_hash": compute_file_hash(Path(args.init_ckpt)),
        "trainer_hash": compute_file_hash(Path(__file__)),
        "sampling_code_hash": compute_file_hash(PROJECT_ROOT / 'src/data/tip_sampling.py'),
    }

    # Output directories
    output_dir = PROJECT_ROOT / args.output_dir
    if (output_dir / 'resolved_config.yaml').exists():
        raise FileExistsError(f'Use a fresh run directory; existing run: {output_dir}')
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Metadata and Splits
    with open(splits_path, "r") as f:
        splits = json.load(f)

    train_study_ids = splits["partitions"]["dev_pilot_train"]["study_ids"]
    val_study_ids = splits["partitions"]["dev_pilot_val"]["study_ids"]

    inv_df = pd.read_parquet(inv_path)
    study_df = pd.read_parquet(study_inv_path)
    reviewed_df = pd.read_parquet(reviewed_tips_path)

    # Explicit confirmed negatives from study inventory
    confirmed_neg_ids = set(
        study_df[study_df["is_study_level_negative"] & (~study_df["has_cvc"])]["StudyInstanceUID"]
    )

    # Validation tips: unconditionally loaded from reviewed_tips_path (val split)
    reviewed_val_df = reviewed_df[reviewed_df["split"] == "val"].copy()

    # Training tips: loaded from train_manifest_path (Arm B) or reviewed train split (Arm A Control)
    if args.train_manifest_path is not None:
        train_manifest_p = PROJECT_ROOT / args.train_manifest_path
        if not train_manifest_p.exists():
            raise FileNotFoundError(f"Training manifest not found: {train_manifest_p}")
        if str(train_manifest_p).endswith(".parquet"):
            manifest_raw = pd.read_parquet(train_manifest_p)
        else:
            manifest_raw = pd.read_csv(train_manifest_p)

        # Filter strictly to active tip supervision instances
        if "tip_supervision_active" in manifest_raw.columns:
            manifest_active = manifest_raw[manifest_raw["tip_supervision_active"] == True].copy()
        else:
            manifest_active = manifest_raw.copy()

        # Ensure required columns for CoarseDualDataset
        if "final_tip_x" in manifest_active.columns and "reviewed_tip_x" not in manifest_active.columns:
            manifest_active["reviewed_tip_x"] = manifest_active["final_tip_x"]
            manifest_active["reviewed_tip_y"] = manifest_active["final_tip_y"]

        reviewed_train_df = manifest_active
        train_supervision_source = str(args.train_manifest_path)
        print(f"Loaded separate training manifest from: {args.train_manifest_path}")
        print(f"  Active training tips: {len(reviewed_train_df)} across {reviewed_train_df['study_id'].nunique()} studies")
    else:
        reviewed_train_df = reviewed_df[reviewed_df["split"] == "train"].copy()
        train_supervision_source = f"{args.reviewed_tips_path} (split=='train')"
        print(f"Loaded reviewed training tips from: {train_supervision_source}")
        print(f"  Active training tips: {len(reviewed_train_df)} across {reviewed_train_df['study_id'].nunique()} studies")

    # 3. Construct Datasets
    mask_stroke_native = int(cfg["preprocessing"]["coarse"].get("mask_stroke_width_native", 10))

    train_dataset = CoarseDualDataset(
        study_ids=train_study_ids,
        inventory_df=inv_df,
        images_dir=raw_images_dir,
        confirmed_negative_study_ids=confirmed_neg_ids,
        reviewed_tips_df=reviewed_train_df,
        target_size=(512, 512),
        mask_stroke_width_native=mask_stroke_native,
        clahe_enabled=True,
        tip_sigma=2.5,
        unreviewed_spatial_ignore_radius_native=50.0,
        partial_review_policy=args.partial_review_policy,
    )

    val_dataset = CoarseDualDataset(
        study_ids=val_study_ids,
        inventory_df=inv_df,
        images_dir=raw_images_dir,
        confirmed_negative_study_ids=confirmed_neg_ids,
        reviewed_tips_df=reviewed_val_df,
        target_size=(512, 512),
        mask_stroke_width_native=mask_stroke_native,
        clahe_enabled=True,
        tip_sigma=2.5,
        unreviewed_spatial_ignore_radius_native=50.0,
        partial_review_policy=args.partial_review_policy,
    )

    groups = supervision_groups(train_dataset)
    sampling_generator = torch.Generator().manual_seed(args.seed)
    batch_sampler = None
    if args.sampling == 'tip_balanced':
        batch_sampler = TipBalancedBatchSampler(groups, args.max_attempts, args.seed)
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler,
                                  num_workers=0, pin_memory=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  generator=sampling_generator, num_workers=0,
                                  pin_memory=False, drop_last=True)
    if len(train_loader) == 0:
        raise ValueError('No complete training batches')

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 4. Construct Model & Load Shared Backbone
    model = CoarseDualNet(in_channels=1, base_channels=32).to(device)
    transfer_info = load_shared_backbone(model, Path(args.init_ckpt))
    backbone_hash = compute_model_backbone_hash(model)

    # 5. Construct Loss, Optimizer & Scheduler
    effective_tip_weight = args.tip_weight if args.mode == "dual_head" else 0.0

    pos_weight = torch.tensor([args.seg_pos_weight], device=device)
    criterion = CoarseDualLoss(
        bce_weight=args.seg_bce_weight,
        dice_weight=args.seg_dice_weight,
        tip_weight=effective_tip_weight,
        pos_weight=pos_weight,
        tip_pos_weight=args.tip_pos_weight,
        shaft_weight=args.tip_shaft_weight,
        shaft_tip_exclusion_thresh=args.shaft_tip_exclusion_thresh,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_steps,
        eta_min=args.min_lr,
    )

    scaler = torch.amp.GradScaler("cuda")
    update_counter = UpdateCounter(optimizer)

    # 6. Save Resolved Configuration
    resolved_config = {
        "mode": args.mode,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "sampling": args.sampling,
        "sampling_exposure_policy": (
            "Uniform random sampling ensures bitwise identical study exposure sequence across comparison arms, "
            "strictly isolating the experimental variable to training tip supervision."
            if args.sampling == "uniform" else "Tip balanced sampler with fixed group quotas"
        ),
        "sampling_group_counts": {k: len(v) for k, v in groups.items()},
        "budget_unit": "successful_optimizer_updates",
        "max_attempts": args.max_attempts,
        "optimizer": {
            "type": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "min_lr": args.min_lr,
        },
        "losses": {
            "seg_pos_weight": args.seg_pos_weight,
            "seg_bce_weight": args.seg_bce_weight,
            "seg_dice_weight": args.seg_dice_weight,
            "tip_weight": effective_tip_weight,
            "tip_pos_weight": args.tip_pos_weight,
            "tip_shaft_weight": args.tip_shaft_weight,
            "shaft_tip_exclusion_thresh": args.shaft_tip_exclusion_thresh,
        },
        "training_supervision": {
            "train_manifest_path": str(args.train_manifest_path) if args.train_manifest_path else None,
            "training_supervision_source": train_supervision_source,
            "active_training_tips_count": len(reviewed_train_df),
            "active_training_studies_count": reviewed_train_df['study_id'].nunique(),
        },
        "geometry_and_supervision": {
            "mask_stroke_width_native": mask_stroke_native,
            "clahe_enabled": True,
            "tip_sigma": 2.5,
            "partial_review_policy": args.partial_review_policy,
            "affine_convention": "TransformMetadata OpenCV pixel-center (+0.5 / -0.5)",
            "validation_reviewed_tips_file": str(args.reviewed_tips_path),
            "review_status": "provisional_engineering_review_awaiting_clinical_attestation",
        },
        "evaluation_protocol": {
            "primary_comparison": "Final fixed-step checkpoint (Step 1000)",
            "secondary_reporting": "Development-selected best checkpoint",
            "target_registry": f"{args.reviewed_tips_path} (val split, N={len(reviewed_val_df)})",
            "one_to_one_matching": "threshold_constrained_matching (Hungarian algorithm, radius 15.0px)",
            "provisional_validation_disclaimer": "Provisional engineering-review targets for developmental guidance; does not assert clinical ground truth.",
            "arms": {
                "seg_only": "Segmentation -> skeleton endpoints (valid control)",
                "dual_head_skel": "Segmentation -> skeleton endpoints (measures segmentation change)",
                "dual_head_tip": "Tip-heatmap peaks (measures direct localization benefit)",
            },
        },
        "initialization": {
            "init_ckpt": args.init_ckpt,
            "init_ckpt_hash": input_hashes["init_ckpt_hash"],
            "backbone_hash": backbone_hash,
            "transfer_info": transfer_info,
        },
        "input_hashes": input_hashes,
        "resolved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    resolved_config_path = output_dir / "resolved_config.yaml"
    with open(resolved_config_path, "w") as f:
        yaml.dump(resolved_config, f, default_flow_style=False, sort_keys=False)

    print("=" * 80)
    print("COARSE DUAL-HEAD TRAINING PIPELINE SPECIFICATION (RESOLVED & CONTRACT-COMPLIANT)")
    print("=" * 80)
    print(f"Arm / Mode:                 {args.mode}")
    print(f"Max Steps:                  {args.max_steps}")
    print(f"Batch Size:                 {args.batch_size}")
    print(f"Sampling Policy:            {args.sampling} (uniform exposure matched across arms)")
    print(f"Training Tip Source:        {train_supervision_source} ({len(reviewed_train_df)} active tips, {reviewed_train_df['study_id'].nunique()} studies)")
    print(f"Validation Target Registry: {args.reviewed_tips_path} (val split, N={len(reviewed_val_df)})")
    print(f"Learning Rate:              {args.lr}")
    print(f"Segmentation Pos Weight:    {args.seg_pos_weight} (ref pos_weight=50.0)")
    print(f"Segmentation Loss Weights:  BCE={args.seg_bce_weight}, Dice={args.seg_dice_weight}")
    print(f"Distal Tip Loss Weight:     {effective_tip_weight}")
    print(f"Distal Tip Pos Weight:      {args.tip_pos_weight}")
    print(f"Mask Native Stroke:         {mask_stroke_native} native px (pre-resize)")
    print(f"CLAHE Preprocessing:        Native resolution prior to letterbox")
    print(f"Coordinate Affine:          TransformMetadata (+0.5 / -0.5) bi-directional")
    print(f"Partial Review Policy:      {args.partial_review_policy} (partially reviewed excluded from tip loss)")
    print(f"Backbone Initialization:    {args.init_ckpt} (Hash: {input_hashes['init_ckpt_hash']})")
    print(f"Backbone Parameter Hash:    {backbone_hash} ({transfer_info['loaded_keys_count']} layers transferred)")
    print(f"Evaluation Protocol:        Primary: Step 1000 | Secondary: Dev-Best | Hungarian Radius: 15.0px")
    if args.mode == "seg_only":
        print(f"Prediction Method:          Segmentation -> skeleton endpoints (CONTROL: tip head NEVER evaluated)")
    else:
        print(f"Prediction Methods:         1) Segmentation -> skeleton endpoints | 2) Tip-heatmap peaks")
    print(f"Resolved Config Saved To:   {resolved_config_path}")
    print("=" * 80)

    # 7. Check Execution Flag
    if not args.execute_training:
        print("[STATUS: TRAINING IS PAUSED AS DIRECTED]")
        print("To launch training when authorized, specify --execute_training.")
        print("Pipeline is fully configured, verified, and ready.")
        return

    # 8. Training Loop Execution (Only when explicitly authorized)
    print("\nBeginning step-controlled training...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    t_start = time.time()
    history = []
    val_history = []
    best_primary_metric = -1.0
    best_step = -1

    step = 0
    attempts = 0
    skipped_updates = 0
    exposures = {k: 0 for k in groups}
    positive_batches = 0
    index_group = {train_dataset.study_ids[i]: k for k, indices in groups.items() for i in indices}
    exposure_by_study = {}
    running_count = 0
    running_loss = 0.0
    running_seg = 0.0
    running_tip = 0.0

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    while step < args.max_steps:
        model.train()
        for batch in train_loader:
            if step >= args.max_steps:
                break
            if attempts >= args.max_attempts:
                raise RuntimeError('Attempt limit reached before successful-update budget')
            attempts += 1
            batch_groups = [index_group[sid] for sid in batch['study_id']]
            positive_batches += int('positive' in batch_groups)
            for sid, group in zip(batch['study_id'], batch_groups):
                exposures[group] += 1
                exposure_by_study[sid] = exposure_by_study.get(sid, 0) + 1

            imgs = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            tip_hms = batch["tip_heatmap"].to(device, non_blocking=True)
            tip_ms = batch["tip_mask"].to(device, non_blocking=True)
            tip_sms = batch["tip_spatial_mask"].to(device, non_blocking=True)
            expected_tip_masks = torch.tensor([g != 'unknown' for g in batch_groups], device=device)
            if not torch.equal(tip_ms.bool(), expected_tip_masks):
                raise ValueError('Sampler classification disagrees with actual dataset tip masks')

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                seg_logits, tip_logits = model(imgs)
                loss, loss_dict = criterion(
                    seg_logits, tip_logits, masks, tip_hms,
                    tip_mask=tip_ms, tip_spatial_mask=tip_sms
                )

            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at attempt {attempts}: {loss_dict}')
            scaler.scale(loss).backward()
            before_update = update_counter.successful
            scaler.step(optimizer)
            scaler.update()
            updated = update_counter.successful > before_update
            if updated:
                scheduler.step()
                step = update_counter.successful
            else:
                skipped_updates += 1
            telemetry = {
                'attempted_batches': attempts,
                'successful_updates': step,
                'skipped_updates': skipped_updates,
                'positive_tip_batches': positive_batches,
                'study_exposures': dict(exposures),
                'exposure_by_study': dict(exposure_by_study),
                'amp_scale': scaler.get_scale(),
                'peak_gpu_memory_mb': round(torch.cuda.max_memory_allocated() / (1024**2), 1) if torch.cuda.is_available() else 0.0,
                'cur_gpu_allocated_mb': round(torch.cuda.memory_allocated() / (1024**2), 1) if torch.cuda.is_available() else 0.0,
            }
            try:
                (output_dir / 'progress.tmp').write_text(json.dumps(telemetry, indent=2))
                os.replace(output_dir / 'progress.tmp', output_dir / 'progress.json')
            except OSError:
                pass
            if not updated:
                print(f'AMP skipped attempt {attempts}; successful updates={step}; scale={scaler.get_scale()}')
                del imgs, masks, tip_hms, tip_ms, tip_sms, loss, loss_dict, seg_logits, tip_logits, batch
                continue

            running_loss += loss_dict["loss_total"]
            running_seg += loss_dict["loss_seg"]
            running_tip += loss_dict["loss_tip"]
            running_count += 1
            del imgs, masks, tip_hms, tip_ms, tip_sms, loss, loss_dict, seg_logits, tip_logits, batch

            if step % 25 == 0 or step == 1:
                cur_lr = scheduler.get_last_lr()[0]
                n_avg = running_count
                avg_l = running_loss / n_avg
                avg_seg = running_seg / n_avg
                avg_tip = running_tip / n_avg
                running_loss, running_seg, running_tip = 0.0, 0.0, 0.0
                running_count = 0
                print(f"Step {step:4d}/{args.max_steps} | Loss: {avg_l:.4f} (Seg: {avg_seg:.4f}, Tip: {avg_tip:.4f}) | LR: {cur_lr:.2e}")
                if step % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                history.append({
                    "step": step,
                    "loss_total": avg_l,
                    "loss_seg": avg_seg,
                    "loss_tip": avg_tip,
                    "lr": cur_lr,
                    "attempted_batches": attempts,
                    "skipped_updates": skipped_updates,
                    "study_exposures": dict(exposures),
                })

            # Validation
            if step % args.val_interval == 0 or step == args.max_steps:
                print(f"\n[Validation Step {step}] Evaluating frozen validation cohort (Mode: {args.mode})...")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                val_metrics = run_dual_validation(
                    model=model,
                    val_loader=val_loader,
                    reviewed_val_df=reviewed_val_df,
                    confirmed_negatives=confirmed_neg_ids,
                    criterion=criterion,
                    device=device,
                    mode=args.mode,
                )
                val_metrics["step"] = step
                val_metrics['training_telemetry'] = dict(telemetry)
                val_history.append(val_metrics)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print(f"  Val Loss: Total={val_metrics['val_loss_total']:.4f}, Seg={val_metrics['val_loss_seg']:.4f}, Tip={val_metrics['val_loss_tip']:.4f}")
                print(f"  Mask Dice: {val_metrics['val_mask_dice_mean']:.4f}")
                
                skel_s15 = val_metrics["seg_skeleton"]["success_at_15"]
                skel_cnt = val_metrics["seg_skeleton"]["matched_15_count"]
                total_rev = val_metrics["reviewed_targets_total"]
                print(f"  [Seg -> Skeleton Endpoints] Success@15: {skel_cnt}/{total_rev} ({skel_s15:.2%}) | FP on Negatives: {val_metrics['seg_skeleton']['negative_false_candidates_total']}")

                if args.mode == "dual_head":
                    tip_s15 = val_metrics["tip_heatmap"]["success_at_15"]
                    tip_cnt = val_metrics["tip_heatmap"]["matched_15_count"]
                    print(f"  [Tip Heatmap Peaks]         Success@15: {tip_cnt}/{total_rev} ({tip_s15:.2%}) | FP on Negatives: {val_metrics['tip_heatmap']['negative_false_peaks_total']}")
                    selection_metric = tip_s15
                else:
                    selection_metric = skel_s15

                is_best = selection_metric > best_primary_metric
                if is_best:
                    best_primary_metric = selection_metric
                    best_step = step
                    best_ckpt_name = f"best_coarse_{args.mode}.pt"
                    save_checkpoint(
                        path=ckpt_dir / best_ckpt_name,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        scheduler=scheduler,
                        sampler=batch_sampler,
                        step=step,
                        config=resolved_config,
                        input_hashes=input_hashes,
                        metrics=val_metrics,
                        is_best=True,
                    )
                    print(f"  -> Saved new best checkpoint: {best_ckpt_name} (Metric: {selection_metric:.2%})")

                model.train()

    # Final checkpoint save at Step 1000 (PRIMARY comparison artifact)
    final_ckpt_name = f"final_coarse_{args.mode}.pt"
    save_checkpoint(
        path=ckpt_dir / final_ckpt_name,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        sampler=batch_sampler,
        step=args.max_steps,
        config=resolved_config,
        input_hashes=input_hashes,
        metrics=val_history[-1] if val_history else {},
        is_best=False,
    )

    history_payload = {
        "resolved_config": resolved_config,
        "train_steps": history,
        "val_evaluations": val_history,
        "best_step": best_step,
        "best_metric": best_primary_metric,
        "final_step": args.max_steps,
        "total_time_seconds": time.time() - t_start,
        "training_telemetry": telemetry,
        "comparison_note": "Primary comparison is final_step (Step 1000). best_step is development-selected.",
    }
    with open(output_dir / "history.json", "w") as f:
        json.dump(history_payload, f, indent=2)

    print(f"\nTraining completed successfully for arm '{args.mode}'.")
    print(f"Primary comparison checkpoint: {ckpt_dir / final_ckpt_name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Capture crash diagnostics in output directory if specified
        parser = argparse.ArgumentParser()
        parser.add_argument("--output_dir", type=str, default=None)
        args, _ = parser.parse_known_args()
        if args.output_dir:
            out_p = PROJECT_ROOT / args.output_dir
            if out_p.exists():
                crash_info = {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "cuda_diagnostics": {
                        "allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 2) if torch.cuda.is_available() else 0,
                        "reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 2) if torch.cuda.is_available() else 0,
                        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 2) if torch.cuda.is_available() else 0,
                    } if torch.cuda.is_available() else {},
                }
                try:
                    (out_p / "crash_diagnostics.json").write_text(json.dumps(crash_info, indent=2), encoding="utf-8")
                except Exception:
                    pass
        raise
