"""Matched Continuation Pilot: Control (T1 unchanged) vs Experiment (T1 + Local Offset Head).

Protocol:
- Base Checkpoint: Frozen T1 seed-42 final checkpoint (1000 updates).
- Budget: Exactly 250 updates per arm, executed sequentially on GPU.
- Cohort: Identical positive (400) + negative (396) cohort.
- Microbatch: 1, Effective Batch: 4 (accumulated over 4 microbatches).
- Ledger: Exactly identical immutable batch ledger entries (updates 1 to 250).
- Segmentation Retention: Segmentation loss identical (BCE pos_weight=10 + SoftDice); retention evaluated.
- Output: training_comparison.csv, resolved_configs/, checkpoints/, logs/.
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
import scipy.ndimage as ndimage
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms.functional as TF

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXP_ROOT = PROJECT_ROOT / "experiments" / "hrnet_tip_improvement_v2"
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from src.data.rasterize import rasterize_study_polylines
from src.data.transforms import LetterboxTransform
from experiments.hrnet_tip_pilot_v1.src.geometry import (
    CoordinateTransform,
    heatmap_point_to_native,
    render_multitip_gaussian_heatmap,
)
from experiments.hrnet_tip_improvement_v2.src.models.hrnet_joint_offset_tip import HRNetJointOffsetTipNet

DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "experiments" / "hrnet_tip_pilot_v1" / "frozen_reference" / "configs"
CKPT_REF = PROJECT_ROOT / "experiments" / "hrnet_tip_pilot_v1" / "runs" / "seed_42" / "t1_joint_hrnet" / "checkpoints" / "final_model.pt"
IMAGES_DIR = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")
BENCHMARK_CSV = EXP_ROOT / "benchmark_v2.csv"
BENCHMARK_MANIFEST = PROJECT_ROOT / "experiments" / "hrnet_tip_pilot_v1" / "manifests" / "development_benchmark_manifest.json"


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TipWeightedBCELoss(nn.Module):
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
        logits = pred_logits.float()
        targets = target_hm.float()
        valids = valid_mask.float()

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

        return norm_loss, {"num_pos_peaks": float(num_pos_peaks), "num_valid_pixels": float(num_valid_pixels)}


def extract_peaks_and_offsets(hm_tensor: torch.Tensor, min_val: float = 0.50) -> List[Dict[str, Any]]:
    """Extract integer peaks and sub-stride continuous offsets from ground truth heatmap."""
    hm = hm_tensor.squeeze().cpu().numpy()
    H, W = hm.shape
    max_filt = ndimage.maximum_filter(hm, size=3, mode="constant", cval=0.0)
    is_peak = (hm == max_filt) & (hm >= min_val)
    coords = np.argwhere(is_peak)

    peak_info = []
    for u, v in coords:
        if u <= 0 or u >= H - 1 or v <= 0 or v >= W - 1:
            continue
        val_c = float(hm[u, v])
        val_l = float(hm[u, v - 1])
        val_r = float(hm[u, v + 1])
        denom_x = 2.0 * (2.0 * val_c - val_l - val_r)
        dx = (val_r - val_l) / denom_x if abs(denom_x) > 1e-6 else 0.0
        dx = max(-0.5, min(0.5, dx))

        val_u = float(hm[u - 1, v])
        val_d = float(hm[u + 1, v])
        denom_y = 2.0 * (2.0 * val_c - val_u - val_d)
        dy = (val_d - val_u) / denom_y if abs(denom_y) > 1e-6 else 0.0
        dy = max(-0.5, min(0.5, dy))

        peak_info.append({"gx": int(v), "gy": int(u), "dx": float(dx), "dy": float(dy)})
    return peak_info


def subpixel_refine_quad(hm: np.ndarray, u: int, v: int) -> Tuple[float, float]:
    H, W = hm.shape
    if u <= 0 or u >= H - 1 or v <= 0 or v >= W - 1:
        return float(v), float(u)
    val_c = float(hm[u, v])
    val_l = float(hm[u, v - 1])
    val_r = float(hm[u, v + 1])
    denom_x = 2.0 * (2.0 * val_c - val_l - val_r)
    dx = (val_r - val_l) / denom_x if abs(denom_x) > 1e-6 else 0.0
    dx = max(-0.5, min(0.5, dx))

    val_u = float(hm[u - 1, v])
    val_d = float(hm[u + 1, v])
    denom_y = 2.0 * (2.0 * val_c - val_u - val_d)
    dy = (val_d - val_u) / denom_y if abs(denom_y) > 1e-6 else 0.0
    dy = max(-0.5, min(0.5, dy))

    return float(v + dx), float(u + dy)


def evaluate_model_on_benchmark(
    model: nn.Module,
    device: torch.device,
    bench_df: pd.DataFrame,
    studies_manifest: List[Dict[str, Any]],
    use_offset_head: bool = False,
    conf_thresh: float = 0.40,
    nms_radius: float = 20.0,
    cap: int = 10,
) -> Dict[str, Any]:
    model.eval()
    transform = LetterboxTransform(target_size=(768, 768), clahe_enabled=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    targets_by_study = {}
    for sid, grp in bench_df.groupby("study_id"):
        targets_by_study[sid] = grp.to_dict("records")

    total_m5, total_m10, total_m15, total_m25 = 0, 0, 0, 0
    total_fp = 0
    neg_fp = 0
    single_m15, multi_m15 = 0, 0
    seg_inter_total = 0.0
    seg_card_total = 0.0

    all_preds_dict = {}

    inv_path = DATA_DIR / "annotation_inventory.parquet"
    eval_polylines: Dict[str, List[Any]] = {}
    if inv_path.exists():
        inv_df = pd.read_parquet(inv_path)
        for _, row in inv_df[inv_df["device_type"] == "CVC"].iterrows():
            sid_s = str(row["study_id"])
            p_raw = row["polyline"]
            poly = json.loads(p_raw) if isinstance(p_raw, str) else p_raw
            eval_polylines.setdefault(sid_s, []).append(poly)

    with torch.no_grad():
        for s in studies_manifest:
            sid = s["study_id"]
            targets = [(float(t.get("x", t.get("target_x_native"))), float(t.get("y", t.get("target_y_native")))) for t in targets_by_study.get(sid, [])]
            is_neg = (len(targets) == 0)

            img_path = IMAGES_DIR / f"{sid}.jpg"
            pil_im = Image.open(img_path).convert("L")
            im_np = np.array(pil_im)
            orig_h, orig_w = im_np.shape
            coord_tf = CoordinateTransform.from_native_dims(orig_w, orig_h, target_size=768, stride=4)
            lb_img, meta = transform.apply_image(im_np)
            raw_t = torch.from_numpy(lb_img).unsqueeze(0).unsqueeze(0).to(device).float() / 255.0
            inp = (raw_t.repeat(1, 3, 1, 1) - mean) / std

            out = model(inp)
            tip_probs = out["tip_probs"].squeeze().cpu().numpy().astype(np.float32)

            if use_offset_head and "offset_pred" in out:
                offset_map = out["offset_pred"].squeeze().cpu().numpy().astype(np.float32)
            else:
                offset_map = None

            # Segmentation retention monitoring
            seg_probs = out["seg_probs"].squeeze().cpu().numpy().astype(np.float32)
            if not is_neg:
                polys = eval_polylines.get(sid, [])
                if polys:
                    m_gt = rasterize_study_polylines(polys, (orig_h, orig_w), thickness=10)
                    lb_gt = transform.apply_mask(m_gt, meta)
                    pred_bin = (seg_probs > 0.5).astype(np.float32)
                    gt_bin = (lb_gt > 0).astype(np.float32)
                    inter = float((pred_bin * gt_bin).sum())
                    card = float(pred_bin.sum() + gt_bin.sum())
                    seg_inter_total += inter
                    seg_card_total += card

            # Valid area
            pad_l_hm = coord_tf.padding_left / 4.0
            pad_t_hm = coord_tf.padding_top / 4.0
            res_w_hm = coord_tf.resized_width / 4.0
            res_h_hm = coord_tf.resized_height / 4.0

            # Find peaks
            max_filt = ndimage.maximum_filter(tip_probs, size=3, mode="constant", cval=0.0)
            is_peak = (tip_probs == max_filt) & (tip_probs >= conf_thresh)
            peak_coords = np.argwhere(is_peak)

            cands = []
            for u, v in peak_coords:
                if u < pad_t_hm or u >= (pad_t_hm + res_h_hm) or v < pad_l_hm or v >= (pad_l_hm + res_w_hm):
                    continue
                conf = float(tip_probs[u, v])
                if offset_map is not None:
                    dx = float(offset_map[0, u, v])
                    dy = float(offset_map[1, u, v])
                    dx = max(-0.5, min(0.5, dx))
                    dy = max(-0.5, min(0.5, dy))
                    v_sub, u_sub = v + dx, u + dy
                else:
                    v_sub, u_sub = subpixel_refine_quad(tip_probs, int(u), int(v))

                nx, ny = heatmap_point_to_native(v_sub, u_sub, coord_tf)
                cands.append({"conf": conf, "native_x": nx, "native_y": ny})

            # Sort descending by confidence
            cands.sort(key=lambda c: c["conf"], reverse=True)

            # Native NMS
            kept = []
            for c in cands:
                suppress = False
                for k in kept:
                    if math.hypot(c["native_x"] - k["native_x"], c["native_y"] - k["native_y"]) < nms_radius:
                        suppress = True
                        break
                if not suppress:
                    kept.append(c)

            # Cap
            if cap is not None:
                kept = kept[:cap]

            all_preds_dict[sid] = kept

            # Match against targets
            targets = [(float(t.get("x", t.get("target_x_native"))), float(t.get("y", t.get("target_y_native")))) for t in targets_by_study.get(sid, [])]
            num_t = len(targets)
            num_p = len(kept)
            is_neg = (num_t == 0)

            if is_neg:
                total_fp += num_p
                neg_fp += num_p
            else:
                for tol, target_match_counter in [
                    (5.0, "m5"),
                    (10.0, "m10"),
                    (15.0, "m15"),
                    (25.0, "m25"),
                ]:
                    cost = np.full((num_t, num_p), 1e6, dtype=np.float64)
                    dist = np.zeros((num_t, num_p), dtype=np.float64)
                    for i, (tx, ty) in enumerate(targets):
                        for j, p in enumerate(kept):
                            d = math.hypot(tx - p["native_x"], ty - p["native_y"])
                            dist[i, j] = d
                            if d <= tol:
                                cost[i, j] = d
                    if num_t > 0 and num_p > 0:
                        r_ind, c_ind = linear_sum_assignment(cost)
                        m_count = sum(1 for r, c in zip(r_ind, c_ind) if dist[r, c] <= tol)
                    else:
                        m_count = 0

                    if target_match_counter == "m5":
                        total_m5 += m_count
                    elif target_match_counter == "m10":
                        total_m10 += m_count
                    elif target_match_counter == "m15":
                        total_m15 += m_count
                        total_fp += (num_p - m_count)
                        if num_t == 1:
                            single_m15 += m_count
                        else:
                            multi_m15 += m_count
                    elif target_match_counter == "m25":
                        total_m25 += m_count

    total_targets = len(bench_df)
    total_studies = len(studies_manifest)
    total_neg_studies = sum(1 for s in studies_manifest if len(targets_by_study.get(s["study_id"], [])) == 0)

    res = {
        "success_at_5px": total_m5 / total_targets,
        "matches_5px": total_m5,
        "success_at_10px": total_m10 / total_targets,
        "matches_10px": total_m10,
        "success_at_15px": total_m15 / total_targets,
        "matches_15px": total_m15,
        "success_at_25px": total_m25 / total_targets,
        "matches_25px": total_m25,
        "mean_false_tips_per_image": total_fp / total_studies,
        "mean_false_tips_negative_image": neg_fp / max(total_neg_studies, 1),
        "total_false_tips": total_fp,
        "single_cvc_matches": single_m15,
        "single_cvc_success": single_m15 / 11,
        "multi_cvc_matches": multi_m15,
        "multi_cvc_success": multi_m15 / 30,
        "seg_dice": (2.0 * seg_inter_total) / max(seg_card_total, 1.0),
    }
    model.train()
    return res


def run_arm(
    arm_name: str,  # 'control' or 'experiment'
    updates: int = 250,
    seed: int = 42,
    lr: float = 5e-5,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"=== RUNNING CONTINUATION ARM: {arm_name.upper()} ({updates} UPDATES) ===")
    print("=" * 80)

    device = torch.device(device_str)
    seed_everything(seed)

    enable_offset = (arm_name == "experiment")
    arm_dir = EXP_ROOT / "runs" / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = arm_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    resolved_config = {
        "arm_name": arm_name,
        "model_architecture": "HRNetJointOffsetTipNet" if enable_offset else "HRNetJointTipNet",
        "base_checkpoint": str(CKPT_REF),
        "updates": updates,
        "effective_batch": 4,
        "microbatch": 1,
        "learning_rate": lr,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "enable_offset_head": enable_offset,
        "lambda_seg": 1.0,
        "lambda_tip": 1.0,
        "lambda_offset": 1.0 if enable_offset else 0.0,
        "seg_loss": "BCE(pos_weight=10) + SoftDice",
        "tip_loss": "TipWeightedBCELoss(pos_weight=20)",
        "offset_loss": "SmoothL1Loss(beta=0.1)" if enable_offset else "None",
    }
    res_cfg_dir = EXP_ROOT / "resolved_configs"
    res_cfg_dir.mkdir(parents=True, exist_ok=True)
    with open(res_cfg_dir / f"{arm_name}_config.json", "w") as f:
        json.dump(resolved_config, f, indent=2)

    # 1. Instantiate Model
    model = HRNetJointOffsetTipNet(pretrained=False, enable_offset=enable_offset).to(device)
    model.load_from_t1_checkpoint(CKPT_REF)

    # Ingest benchmark manifest and DataFrame
    bench_df = pd.read_csv(BENCHMARK_CSV)
    with open(BENCHMARK_MANIFEST) as f:
        studies_manifest = json.load(f)["studies"]

    # Step 0 Benchmark Evaluation
    print(f"\n--- Running Step 0 Baseline Evaluation for {arm_name} ---", flush=True)
    eval_0 = evaluate_model_on_benchmark(
        model=model,
        device=device,
        bench_df=bench_df,
        studies_manifest=studies_manifest,
        use_offset_head=enable_offset,
    )
    print(
        f"[{arm_name} Step 0] Success@15px: {eval_0['success_at_15px']*100:.2f}% ({eval_0['matches_15px']}/41) | "
        f"Success@10px: {eval_0['success_at_10px']*100:.2f}% | Success@5px: {eval_0['success_at_5px']*100:.2f}% | "
        f"Mean FP: {eval_0['mean_false_tips_per_image']:.3f} | Neg FP: {eval_0['mean_false_tips_negative_image']:.3f}",
        flush=True,
    )

    # 2. Ingest Immutable Batch Ledger
    ledger_path = CONFIGS_DIR / "batch_ledger.jsonl"
    ledger: Dict[int, Dict[str, Any]] = {}
    with open(ledger_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= updates:
                break
            if line.strip():
                item = json.loads(line)
                update_key = item.get("update_id", item.get("update"))
                ledger[update_key] = item

    # 3. Ingest Target Registry & Polylines
    reg_df = pd.read_parquet(PROJECT_ROOT / "experiments" / "hrnet_tip_pilot_v1" / "manifests" / "instance_target_registry.parquet")
    study_tips_lookup: Dict[str, List[Tuple[float, float]]] = {}
    study_supervision_status: Dict[str, bool] = {}
    for sid, group in reg_df.groupby("study_id"):
        all_assessable = all(r["tip_state"] == "Assessable distal tip" for _, r in group.iterrows())
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

    transform = LetterboxTransform(target_size=(768, 768), clahe_enabled=True)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # 4. Pre-cache unique studies to keep execution ultra-fast
    unique_sids = set()
    for item in ledger.values():
        unique_sids.update(item["study_ids"])
    print(f"Pre-caching deterministic letterbox representations for {len(unique_sids)} unique studies...", flush=True)

    study_cache: Dict[str, Dict[str, Any]] = {}
    t0_cache = time.time()
    for sid in unique_sids:
        img_path = IMAGES_DIR / f"{sid}.jpg"
        pil_im = Image.open(img_path).convert("L")
        im_np = np.array(pil_im)
        orig_h, orig_w = im_np.shape
        coord_tf = CoordinateTransform.from_native_dims(orig_w, orig_h, target_size=768, stride=4)
        polylines = train_polylines.get(sid, [])
        is_pos = len(polylines) > 0
        mask_np = rasterize_study_polylines(polylines, (orig_h, orig_w), thickness=10) if is_pos else np.zeros((orig_h, orig_w), dtype=np.uint8)

        lb_img, meta = transform.apply_image(im_np)
        lb_mask = transform.apply_mask(mask_np, meta) if is_pos else np.zeros((768, 768), dtype=np.uint8)

        valid_np = np.zeros((768, 768), dtype=np.uint8)
        valid_np[meta.pad_top : meta.pad_top + meta.new_h, meta.pad_left : meta.pad_left + meta.new_w] = 1

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

        study_cache[sid] = {
            "lb_img": lb_img,
            "lb_mask": lb_mask,
            "valid_np": valid_np,
            "hm_tensor": hm_tensor,
            "hm_valid": hm_valid,
            "is_pos": is_pos,
            "has_tip_sup": has_tip_sup,
        }
    print(f"Cached {len(study_cache)} studies in {time.time() - t0_cache:.2f}s.", flush=True)

    # 5. Training Setup
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=updates, eta_min=1e-6)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    tip_criterion = TipWeightedBCELoss()
    offset_criterion = nn.SmoothL1Loss(beta=0.1)

    print(f"Starting sequential training for {arm_name} (Updates 1 to {updates})...", flush=True)
    t_start = time.time()
    history = []

    for step in range(1, updates + 1):
        entry = ledger[step]
        study_ids = entry["study_ids"]
        aug_list = entry["augmentation_parameters"]

        micro_data = []
        for sid, aug_params in zip(study_ids, aug_list):
            cached = study_cache[sid]
            pil_img = Image.fromarray(cached["lb_img"])
            pil_mask = Image.fromarray(cached["lb_mask"])
            pil_valid = Image.fromarray(cached["valid_np"])
            hm_t = cached["hm_tensor"].clone()
            hm_v = cached["hm_valid"].clone()

            # Augmentations
            if aug_params["hflip"]:
                pil_img = TF.hflip(pil_img)
                pil_mask = TF.hflip(pil_mask)
                pil_valid = TF.hflip(pil_valid)
                hm_t = TF.hflip(hm_t)
                hm_v = TF.hflip(hm_v)

            translate_768 = [aug_params["translate_x_pct"] * 768, aug_params["translate_y_pct"] * 768]
            translate_192 = [aug_params["translate_x_pct"] * 192, aug_params["translate_y_pct"] * 192]

            pil_img = TF.affine(pil_img, angle=aug_params["angle"], translate=translate_768, scale=aug_params["scale"], shear=0, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            pil_mask = TF.affine(pil_mask, angle=aug_params["angle"], translate=translate_768, scale=aug_params["scale"], shear=0, interpolation=TF.InterpolationMode.NEAREST, fill=0)
            pil_valid = TF.affine(pil_valid, angle=aug_params["angle"], translate=translate_768, scale=aug_params["scale"], shear=0, interpolation=TF.InterpolationMode.NEAREST, fill=0)

            hm_t = TF.affine(hm_t, angle=aug_params["angle"], translate=translate_192, scale=aug_params["scale"], shear=0, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            hm_v = TF.affine(hm_v, angle=aug_params["angle"], translate=translate_192, scale=aug_params["scale"], shear=0, interpolation=TF.InterpolationMode.NEAREST, fill=0)

            if aug_params["brightness"] != 1.0:
                pil_img = TF.adjust_brightness(pil_img, aug_params["brightness"])
            if aug_params["contrast"] != 1.0:
                pil_img = TF.adjust_contrast(pil_img, aug_params["contrast"])

            img_tensor = (TF.to_tensor(pil_img).repeat(3, 1, 1) - mean) / std
            mask_tensor = torch.from_numpy((np.array(pil_mask) > 0).astype(np.float32)).unsqueeze(0)
            valid_tensor = torch.from_numpy((np.array(pil_valid) > 0).astype(np.float32)).unsqueeze(0)

            # Ground truth peaks & offsets
            gt_peaks = []
            if cached["has_tip_sup"] and cached["is_pos"]:
                gt_peaks = extract_peaks_and_offsets(hm_t, min_val=0.50)

            micro_data.append((img_tensor, mask_tensor, valid_tensor, cached["is_pos"], hm_t, hm_v, cached["has_tip_sup"], gt_peaks))

        # Effective-batch gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        accum_seg_loss = 0.0
        accum_tip_loss = 0.0
        accum_offset_loss = 0.0
        accum_total_loss = 0.0

        total_valid_pixels = sum(float(vt.sum().item()) for _, _, vt, _, _, _, _, _ in micro_data)
        total_positives = sum(1 for _, _, _, p, _, _, _, _ in micro_data if p)
        total_tip_sup = sum(1 for _, _, _, _, _, _, sup, _ in micro_data if sup)

        for img_t, mask_t, valid_t, is_pos, hm_t, hm_v, has_tip_sup, gt_peaks in micro_data:
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

                # 3. Tip Heatmap Loss
                if has_tip_sup and total_tip_sup > 0:
                    tip_loss_sample, _ = tip_criterion(tip_logits, hm_b, hm_vb, has_positive_tips=is_pos)
                    tip_loss_eff = tip_loss_sample / float(total_tip_sup)
                else:
                    tip_loss_eff = torch.tensor(0.0, device=device)

                # 4. Continuous Offset Head Loss (Experiment arm only)
                if enable_offset and len(gt_peaks) > 0 and "offset_pred" in out:
                    offset_pred = out["offset_pred"]  # [1, 2, 192, 192]
                    sample_off_loss = torch.tensor(0.0, device=device)
                    for pk in gt_peaks:
                        gx, gy = pk["gx"], pk["gy"]
                        pred_dx = offset_pred[0, 0, gy, gx]
                        pred_dy = offset_pred[0, 1, gy, gx]
                        target_dx = torch.tensor(pk["dx"], device=device)
                        target_dy = torch.tensor(pk["dy"], device=device)
                        sample_off_loss += offset_criterion(pred_dx, target_dx) + offset_criterion(pred_dy, target_dy)
                    offset_loss_eff = (sample_off_loss / len(gt_peaks)) / float(max(total_tip_sup, 1))
                else:
                    offset_loss_eff = torch.tensor(0.0, device=device)

                total_sample_loss = seg_sample_loss + tip_loss_eff + offset_loss_eff

            accum_seg_loss += seg_sample_loss.item()
            accum_tip_loss += tip_loss_eff.item()
            accum_offset_loss += offset_loss_eff.item()
            accum_total_loss += total_sample_loss.item()

            scaler.scale(total_sample_loss).backward()

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % 25 == 0 or step == updates:
            print(
                f"[{arm_name}] Update {step:03d}/{updates} | Seg Loss: {accum_seg_loss:.4f} | "
                f"Tip Loss: {accum_tip_loss:.4f} | Off Loss: {accum_offset_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )

    elapsed = time.time() - t_start
    print(f"\nCompleted {updates} updates for {arm_name} in {elapsed:.1f}s ({elapsed/updates:.2f}s/update).", flush=True)

    # Save final model
    ckpt_path = ckpt_dir / "final_model.pt"
    torch.save(
        {
            "arm": arm_name,
            "updates": updates,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        ckpt_path,
    )
    print(f"Saved {arm_name} checkpoint to {ckpt_path}.")
    top_ckpt_dir = EXP_ROOT / "checkpoints"
    top_ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arm": arm_name,
            "updates": updates,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        top_ckpt_dir / f"{arm_name}_final_model.pt",
    )

    # Step 250 Final Benchmark Evaluation
    print(f"\n--- Running Step {updates} Evaluation for {arm_name} ---", flush=True)
    eval_end = evaluate_model_on_benchmark(
        model=model,
        device=device,
        bench_df=bench_df,
        studies_manifest=studies_manifest,
        use_offset_head=enable_offset,
    )
    print(
        f"[{arm_name} Step {updates}] Success@15px: {eval_end['success_at_15px']*100:.2f}% ({eval_end['matches_15px']}/41) | "
        f"Success@10px: {eval_end['success_at_10px']*100:.2f}% | Success@5px: {eval_end['success_at_5px']*100:.2f}% | "
        f"Mean FP: {eval_end['mean_false_tips_per_image']:.3f} | Neg FP: {eval_end['mean_false_tips_negative_image']:.3f}",
        flush=True,
    )

    return {
        "arm": arm_name,
        "step_0": eval_0,
        "step_end": eval_end,
        "elapsed_sec": elapsed,
        "checkpoint_path": str(ckpt_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["control", "experiment", "both"], default="both")
    parser.add_argument("--updates", type=int, default=250)
    args = parser.parse_args()

    results = []

    if args.arm in ["control", "both"]:
        res_ctrl = run_arm("control", updates=args.updates)
        results.append(res_ctrl)

    if args.arm in ["experiment", "both"]:
        res_exp = run_arm("experiment", updates=args.updates)
        results.append(res_exp)

    # Save training_comparison.csv
    rows = []
    for r in results:
        arm = r["arm"]
        for step_name, m in [("start_step_0", r["step_0"]), ("end_step_250", r["step_end"])]:
            rows.append({
                "arm": arm,
                "evaluation_point": step_name,
                "success_at_5px": m["success_at_5px"],
                "matches_5px": m["matches_5px"],
                "success_at_10px": m["success_at_10px"],
                "matches_10px": m["matches_10px"],
                "success_at_15px": m["success_at_15px"],
                "matches_15px": m["matches_15px"],
                "success_at_25px": m["success_at_25px"],
                "matches_25px": m["matches_25px"],
                "mean_false_tips_per_image": m["mean_false_tips_per_image"],
                "mean_false_tips_negative_image": m["mean_false_tips_negative_image"],
                "total_false_tips": m["total_false_tips"],
                "single_cvc_matches": m["single_cvc_matches"],
                "multi_cvc_matches": m["multi_cvc_matches"],
                "seg_dice": m.get("seg_dice", 0.0),
            })

    comp_df = pd.DataFrame(rows)
    comp_csv = EXP_ROOT / "training_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)
    print(f"\nSaved training comparison to {comp_csv}")
    print(comp_df.to_string(index=False))
