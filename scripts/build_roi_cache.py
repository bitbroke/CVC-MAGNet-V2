"""Build 3-Channel 384x384 Mediastinal RoI Cache (Zero-Domain-Shift).

Generates [3, 384, 384] uint8 tensors:
- Channel 0: CLAHE-enhanced native crop (tight 0.32W x 0.32H mediastinal window)
- Channel 1: T2 continuous sigmoid seg_prob crop (mapped directly from 512x512)
- Channel 2: Spatial Reference Map:
  - Carina Gaussian (+1.0 -> 255)
  - D1 Tip Gaussian (+0.5 -> 128) if conf_tip >= 0.20 and inside crop; 0 otherwise.

Uses T2 and CarinaNet predictions for BOTH training (N=6,600) and validation (N=2,558),
guaranteeing complete train-test domain parity and eliminating synthetic cheat shortcuts.
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.infer.decoders import decode_d1_subpixel_dark, extract_peaks_from_heatmap
from src.models.tip_heatmap import CVCJointSegTipNet

DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "splits" / "placement_locked_manifest.parquet"
CARINA_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "carina_validation_cache.parquet"
CARINA_TRAIN_PATH = DATA_DIR / "curated_cvc_v1" / "carina_train_cache.parquet"
IMAGES_DIR = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")
T2_CKPT_PATH = PROJECT_ROOT / "runs" / "t2_repaired_finetune_400" / "checkpoints" / "best.pt"
ROI_CACHE_DIR = DATA_DIR / "curated_cvc_v1" / "roi_cache"

ROI_SIZE = 384


def generate_gaussian_point(
    h: int,
    w: int,
    center_x: float,
    center_y: float,
    sigma: float = 6.0,
    peak_val: float = 1.0,
) -> np.ndarray:
    """Generate 2D Gaussian on an HxW grid centered at (center_x, center_y)."""
    grid_y, grid_x = np.ogrid[:h, :w]
    dist_sq = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2
    gaussian = np.exp(-dist_sq / (2.0 * sigma ** 2), dtype=np.float32)
    return gaussian * peak_val


def process_study(
    sid: str,
    img_bgr: np.ndarray,
    c_info: Dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    clahe: Any,
) -> np.ndarray:
    """Generate 3-channel [3, 384, 384] uint8 array."""
    native_h, native_w = img_bgr.shape[:2]

    # 1. Run T2 forward pass on 512x512
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
    img_t = torch.from_numpy(img_resized.transpose(2, 0, 1)).float().to(device) / 255.0
    img_t = ((img_t - mean) / std).unsqueeze(0)

    with torch.no_grad():
        out = model(img_t)
        tip_probs = out["tip_probs"][0, 0].cpu().numpy()  # [128, 128]
        seg_probs = out["seg_probs"][0, 0].cpu().numpy()  # [512, 512]

    # 2. Tight Mediastinal Crop Bounds (0.32W x 0.32H centered on Carina)
    cx_nat = float(c_info.get("carina_x_native", 0.4805 * native_w))
    cy_nat = float(c_info.get("carina_y_native", 0.4000 * native_h))

    span_w = 0.32 * native_w
    span_h = 0.32 * native_h

    x_min = int(max(0, min(native_w - span_w, cx_nat - 0.16 * native_w)))
    x_max = int(x_min + span_w)
    y_min = int(max(0, min(native_h - span_h, cy_nat - 0.10 * native_h)))
    y_max = int(y_min + span_h)

    actual_w = max(x_max - x_min, 1)
    actual_h = max(y_max - y_min, 1)

    # Channel 0: Native CLAHE crop
    crop_gray = cv2.cvtColor(img_bgr[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2GRAY)
    crop_clahe = clahe.apply(crop_gray)
    ch0 = cv2.resize(crop_clahe, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_LINEAR)

    # Channel 1: Continuous T2 seg_prob crop
    # Map [x_min, x_max, y_min, y_max] to 512x512 grid
    x1_512 = int(round((x_min / native_w) * 512))
    x2_512 = int(round((x_max / native_w) * 512))
    y1_512 = int(round((y_min / native_h) * 512))
    y2_512 = int(round((y_max / native_h) * 512))

    x1_512 = max(0, min(511, x1_512))
    x2_512 = max(x1_512 + 1, min(512, x2_512))
    y1_512 = max(0, min(511, y1_512))
    y2_512 = max(y1_512 + 1, min(512, y2_512))

    seg_crop = seg_probs[y1_512:y2_512, x1_512:x2_512]
    ch1_float = cv2.resize(seg_crop, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_LINEAR)
    ch1 = np.clip(ch1_float * 255.0, 0, 255).astype(np.uint8)

    # Channel 2: Spatial Reference Map (Carina +1.0, Tip +0.5 if conf >= 0.20)
    ch2_float = np.zeros((ROI_SIZE, ROI_SIZE), dtype=np.float32)
    c_roi_x = ((cx_nat - x_min) / float(actual_w)) * ROI_SIZE
    c_roi_y = ((cy_nat - y_min) / float(actual_h)) * ROI_SIZE

    if 0 <= c_roi_x < ROI_SIZE and 0 <= c_roi_y < ROI_SIZE:
        g_carina = generate_gaussian_point(ROI_SIZE, ROI_SIZE, c_roi_x, c_roi_y, sigma=6.0, peak_val=1.0)
        ch2_float = np.maximum(ch2_float, g_carina)

    # D1 tip peak extraction with confidence gating
    raw_peaks, nms_peaks = extract_peaks_from_heatmap(
        tip_probs,
        threshold=0.15,
        nms_kernel_size=3,
        max_candidates=3,
        suppression_radius_px=3.0,
    )

    if len(nms_peaks) > 0 and nms_peaks[0]["conf"] >= 0.20:
        top_peak = nms_peaks[0]
        nx, ny, _ = decode_d1_subpixel_dark(
            heatmap=tip_probs,
            peak=top_peak,
            stride=4,
            img_size=512,
            native_w=native_w,
            native_h=native_h,
        )
        t_roi_x = ((nx - x_min) / float(actual_w)) * ROI_SIZE
        t_roi_y = ((ny - y_min) / float(actual_h)) * ROI_SIZE

        if 0 <= t_roi_x < ROI_SIZE and 0 <= t_roi_y < ROI_SIZE:
            g_tip = generate_gaussian_point(ROI_SIZE, ROI_SIZE, t_roi_x, t_roi_y, sigma=6.0, peak_val=0.5)
            ch2_float = np.maximum(ch2_float, g_tip)

    ch2 = np.clip(ch2_float * 255.0, 0, 255).astype(np.uint8)

    # Stack into [3, 384, 384] uint8 array
    return np.stack([ch0, ch1, ch2], axis=0)


def build_cache_for_split(
    sids: List[str],
    carina_df: pd.DataFrame,
    images_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    split_name: str = "validation",
):
    print(f"\n[{split_name}] Starting RoI caching for {len(sids):,} studies...")
    carina_dict = carina_df.set_index("study_id").to_dict("index")
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    t0 = time.time()
    skipped = 0
    saved = 0

    for idx, sid in enumerate(sids):
        out_path = ROI_CACHE_DIR / f"{sid}.npz"
        if out_path.exists():
            saved += 1
            continue

        img_path = images_dir / f"{sid}.jpg"
        if not img_path.exists():
            skipped += 1
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            skipped += 1
            continue

        c_info = carina_dict.get(sid, {})
        roi_uint8 = process_study(
            sid=sid,
            img_bgr=img_bgr,
            c_info=c_info,
            model=model,
            device=device,
            mean=mean,
            std=std,
            clahe=clahe,
        )

        np.savez_compressed(out_path, roi=roi_uint8)
        saved += 1

        if (idx + 1) % 500 == 0 or (idx + 1) == len(sids):
            elapsed = time.time() - t0
            rate = (idx + 1) / max(elapsed, 1e-4)
            print(f"  [{split_name}] Cached {idx + 1}/{len(sids)} ({rate:.1f} img/s)", flush=True)

    print(f"[{split_name}] Complete: {saved} saved, {skipped} skipped.")


def main():
    print("=" * 70)
    print("BUILDING 3-CHANNEL 384x384 MEDIASTINAL ROI CACHE")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    ROI_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Model
    print(f"Loading fine-tuned T2 checkpoint: {T2_CKPT_PATH.name}...")
    model = CVCJointSegTipNet(pretrained=False)
    ckpt = torch.load(T2_CKPT_PATH, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # 2. Load Manifest & Carina Caches
    manifest = pd.read_parquet(MANIFEST_PATH)
    df_carina_val = pd.read_parquet(CARINA_VAL_PATH)
    df_carina_train = pd.read_parquet(CARINA_TRAIN_PATH)

    val_sids = list(manifest[manifest["split"] == "val"]["StudyInstanceUID"])
    train_sids = list(df_carina_train["study_id"])

    # 3. Process Validation Split (N=2,558)
    build_cache_for_split(
        sids=val_sids,
        carina_df=df_carina_val,
        images_dir=IMAGES_DIR,
        model=model,
        device=device,
        split_name="Validation (N=2,558)",
    )

    # 4. Process Training Split (N=6,600)
    build_cache_for_split(
        sids=train_sids,
        carina_df=df_carina_train,
        images_dir=IMAGES_DIR,
        model=model,
        device=device,
        split_name="Training (N=6,600)",
    )

    print("\nAll RoI caches generated successfully!")


if __name__ == "__main__":
    main()

