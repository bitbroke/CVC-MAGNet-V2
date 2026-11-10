"""Build Zonal Segmentation Mass & Gated Geometric Feature Caches.

Extracts robust mediastinal features from fine-tuned T2 checkpoint:
1. 6 Anatomical Zonal Mass Features (integrals of continuous seg_prob relative to Carina):
   - Neck / Upper IJV (dy < -0.06)
   - Upper SVC (-0.06 <= dy < -0.02)
   - Borderline Mid-SVC (-0.02 <= dy < 0.0)
   - Normal CAJ corridor (0.0 <= dy <= +0.05)
   - Deep Right Atrium / Ventricle (+0.05 < dy <= +0.12)
   - Lateral thoracic noise (|dx| > 0.12)
2. Normalized Zonal Fractions & Deepest Extent (deepest_y_norm, p95_y_norm).
3. Confidence-Gated D1 Tip Coordinates:
   - If conf_tip >= 0.20: exact D1 subpixel offsets, (cos_theta, sin_theta), has_valid_tip=1.0.
   - If conf_tip < 0.20: coordinates zeroed out, has_valid_tip=0.0.
4. Carina landmark confidence and fallback flag.
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

ZONAL_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "zonal_val_features.parquet"
ZONAL_TRAIN_PATH = DATA_DIR / "curated_cvc_v1" / "zonal_train_features.parquet"


def compute_seg_tangent(
    seg_prob: np.ndarray,
    tip_model_x: float,
    tip_model_y: float,
    radius: int = 25,
) -> Tuple[float, float]:
    """Extract distal trajectory vector from local segmentation probability map."""
    h_map, w_map = seg_prob.shape
    u, v = int(round(tip_model_y)), int(round(tip_model_x))

    y_min = max(0, u - radius)
    y_max = min(h_map, u + radius + 1)
    x_min = max(0, v - radius)
    x_max = min(w_map, v + radius + 1)

    patch = seg_prob[y_min:y_max, x_min:x_max]
    patch_clean = np.where(patch >= 0.20, patch, 0.0)
    total_mass = float(patch_clean.sum())

    if total_mass < 1e-4:
        return 0.0, 1.0  # Default downward

    grid_y, grid_x = np.indices(patch.shape)
    patch_com_y = float((grid_y * patch_clean).sum() / total_mass)
    patch_com_x = float((grid_x * patch_clean).sum() / total_mass)

    com_y = y_min + patch_com_y
    com_x = x_min + patch_com_x

    vec_x = tip_model_x - com_x
    vec_y = tip_model_y - com_y
    norm = math.hypot(vec_x, vec_y)

    if norm < 1e-4:
        return 0.0, 1.0
    return float(vec_x / norm), float(vec_y / norm)


def extract_features_from_outputs(
    seg_probs: np.ndarray,
    tip_probs: np.ndarray,
    native_w: float,
    native_h: float,
    cx_nat: float,
    cy_nat: float,
    carina_conf: float,
    is_carina_fallback: int,
    tip_conf_thresh: float = 0.20,
) -> Dict[str, float]:
    """Extract 20 robust zonal mass and gated geometric features."""
    H, W = seg_probs.shape  # 512, 512
    c_x = (cx_nat / native_w) * W
    c_y = (cy_nat / native_h) * H

    # Coordinate grids relative to Carina (normalized by H and W)
    y_indices, x_indices = np.indices((H, W))
    dy_grid = (y_indices - c_y) / float(H)
    dx_grid = (x_indices - c_x) / float(W)

    # 1. Zonal Segmentation Mass Integrals
    is_central = np.abs(dx_grid) <= 0.12
    mask_neck = (dy_grid < -0.06) & is_central
    mask_upper_svc = (dy_grid >= -0.06) & (dy_grid < -0.02) & is_central
    mask_borderline = (dy_grid >= -0.02) & (dy_grid < 0.0) & is_central
    mask_caj_normal = (dy_grid >= 0.0) & (dy_grid <= 0.05) & is_central
    mask_atrium_deep = (dy_grid > 0.05) & (dy_grid <= 0.12) & is_central
    mask_lateral = np.abs(dx_grid) > 0.12

    m_neck = float(seg_probs[mask_neck].sum())
    m_upper_svc = float(seg_probs[mask_upper_svc].sum())
    m_borderline = float(seg_probs[mask_borderline].sum())
    m_caj_normal = float(seg_probs[mask_caj_normal].sum())
    m_atrium_deep = float(seg_probs[mask_atrium_deep].sum())
    m_lateral = float(seg_probs[mask_lateral].sum())

    m_total_med = m_neck + m_upper_svc + m_borderline + m_caj_normal + m_atrium_deep
    denom = max(m_total_med, 1e-5)

    frac_neck = m_neck / denom
    frac_upper_svc = m_upper_svc / denom
    frac_borderline = m_borderline / denom
    frac_caj_normal = m_caj_normal / denom
    frac_atrium_deep = m_atrium_deep / denom

    # 2. Deepest Ribbon Penetration
    ribbon_pts = np.argwhere((seg_probs >= 0.25) & is_central)
    if len(ribbon_pts) > 0:
        y_pts = ribbon_pts[:, 0]
        deepest_y_norm = float((np.max(y_pts) - c_y) / float(H))
        p95_y_norm = float((np.percentile(y_pts, 95) - c_y) / float(H))
        highest_y_norm = float((np.min(y_pts) - c_y) / float(H))
        total_ribbon_pixels = float(len(ribbon_pts))
    else:
        deepest_y_norm = 0.0
        p95_y_norm = 0.0
        highest_y_norm = 0.0
        total_ribbon_pixels = 0.0

    # 3. D1 Tip Extraction with Confidence Gating
    raw_peaks, nms_peaks = extract_peaks_from_heatmap(
        tip_probs,
        threshold=0.15,
        nms_kernel_size=3,
        max_candidates=5,
        suppression_radius_px=3.0,
    )

    if len(nms_peaks) > 0 and nms_peaks[0]["conf"] >= tip_conf_thresh:
        top_peak = nms_peaks[0]
        conf_tip = float(top_peak["conf"])
        has_valid_tip = 1.0

        nx, ny, _ = decode_d1_subpixel_dark(
            heatmap=tip_probs,
            peak=top_peak,
            stride=4,
            img_size=H,
            native_w=native_w,
            native_h=native_h,
        )
        mx = (nx / native_w) * W
        my = (ny / native_h) * H
        cos_th, sin_th = compute_seg_tangent(seg_probs, mx, my, radius=25)

        dx_norm = (nx - cx_nat) / native_w
        dy_norm = (ny - cy_nat) / native_h
        r_norm = math.hypot(dx_norm, dy_norm)
    else:
        # Confidence gating: completely zero out coordinates to prevent noise injection
        conf_tip = float(nms_peaks[0]["conf"]) if len(nms_peaks) > 0 else 0.0
        has_valid_tip = 0.0
        dx_norm = 0.0
        dy_norm = 0.0
        r_norm = 0.0
        cos_th = 0.0
        sin_th = 0.0

    return {
        # Zonal masses
        "m_neck": m_neck,
        "m_upper_svc": m_upper_svc,
        "m_borderline": m_borderline,
        "m_caj_normal": m_caj_normal,
        "m_atrium_deep": m_atrium_deep,
        "m_lateral": m_lateral,
        "m_total_med": m_total_med,
        # Normalized fractions
        "frac_neck": frac_neck,
        "frac_upper_svc": frac_upper_svc,
        "frac_borderline": frac_borderline,
        "frac_caj_normal": frac_caj_normal,
        "frac_atrium_deep": frac_atrium_deep,
        # Ribbon extent
        "deepest_y_norm": deepest_y_norm,
        "p95_y_norm": p95_y_norm,
        "highest_y_norm": highest_y_norm,
        "ribbon_pixels": total_ribbon_pixels,
        # Gated tip coordinates
        "dx_norm": dx_norm,
        "dy_norm": dy_norm,
        "r_norm": r_norm,
        "cos_theta": cos_th,
        "sin_theta": sin_th,
        "has_valid_tip": has_valid_tip,
        "conf_tip": conf_tip,
        # CarinaNet metadata
        "conf_carina": carina_conf,
        "is_carina_fallback": float(is_carina_fallback),
    }


def process_cohort(
    manifest_df: pd.DataFrame,
    carina_df: pd.DataFrame,
    images_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    save_path: Path,
    cohort_name: str = "val",
    img_size: int = 512,
) -> pd.DataFrame:
    """Run T2 inference and extract zonal features across a cohort."""
    if save_path.exists():
        print(f"[{cohort_name}] Cache already exists at {save_path}. Loading...")
        df_cached = pd.read_parquet(save_path)
        if len(df_cached) == len(manifest_df):
            print(f"[{cohort_name}] Loaded {len(df_cached)} cached features.")
            return df_cached

    carina_dict = carina_df.set_index("study_id").to_dict("index")
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    results = []
    t0 = time.time()
    total_imgs = len(manifest_df)

    print(f"[{cohort_name}] Extracting zonal features for {total_imgs} studies on {device}...")

    for idx, row in manifest_df.reset_index(drop=True).iterrows():
        sid = row["StudyInstanceUID"]
        img_path = images_dir / f"{sid}.jpg"

        if not img_path.exists():
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        native_h, native_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

        img_t = torch.from_numpy(img_resized.transpose(2, 0, 1)).float().to(device) / 255.0
        img_t = ((img_t - mean) / std).unsqueeze(0)

        with torch.no_grad():
            out = model(img_t)
            tip_probs = out["tip_probs"][0, 0].cpu().numpy()
            seg_probs = out["seg_probs"][0, 0].cpu().numpy()

        c_info = carina_dict.get(sid, {
            "carina_x_native": 0.4805 * native_w,
            "carina_y_native": 0.4000 * native_h,
            "carina_conf": 0.25,
            "is_carina_fallback": 1,
        })
        cx = float(c_info.get("carina_x_native", 0.4805 * native_w))
        cy = float(c_info.get("carina_y_native", 0.4000 * native_h))
        c_conf = float(c_info.get("carina_conf", 0.25))
        is_fb = int(c_info.get("is_carina_fallback", 1))

        feat_dict = extract_features_from_outputs(
            seg_probs=seg_probs,
            tip_probs=tip_probs,
            native_w=native_w,
            native_h=native_h,
            cx_nat=cx,
            cy_nat=cy,
            carina_conf=c_conf,
            is_carina_fallback=is_fb,
            tip_conf_thresh=0.20,
        )
        feat_dict["study_id"] = sid
        feat_dict["CVC - Normal"] = int(row.get("CVC - Normal", 0))
        feat_dict["CVC - Borderline"] = int(row.get("CVC - Borderline", 0))
        feat_dict["CVC - Abnormal"] = int(row.get("CVC - Abnormal", 0))
        results.append(feat_dict)

        if (idx + 1) % 500 == 0 or (idx + 1) == total_imgs:
            elapsed = time.time() - t0
            rate = (idx + 1) / max(elapsed, 1e-4)
            print(f"  [{cohort_name}] Processed {idx + 1}/{total_imgs} ({rate:.1f} img/s)", flush=True)

    df_out = pd.DataFrame(results)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(save_path, index=False)
    print(f"[{cohort_name}] Saved {len(df_out)} records to {save_path}")
    return df_out


def main():
    print("=" * 70)
    print("BUILDING ZONAL SEGMENTATION MASS & GATED GEOMETRIC FEATURE CACHES")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading fine-tuned T2 checkpoint: {T2_CKPT_PATH.name}...")
    model = CVCJointSegTipNet(pretrained=False)
    ckpt = torch.load(T2_CKPT_PATH, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    manifest = pd.read_parquet(MANIFEST_PATH)
    df_carina_val = pd.read_parquet(CARINA_VAL_PATH)
    df_carina_train = pd.read_parquet(CARINA_TRAIN_PATH)

    # 1. Validation Cohort (N=2,558)
    val_manifest = manifest[manifest["split"] == "val"].reset_index(drop=True)
    df_zonal_val = process_cohort(
        manifest_df=val_manifest,
        carina_df=df_carina_val,
        images_dir=IMAGES_DIR,
        model=model,
        device=device,
        save_path=ZONAL_VAL_PATH,
        cohort_name="Validation (N=2,558)",
    )

    # 2. Training Traced Cohort (N=6,600)
    train_manifest = manifest[manifest["split"] == "train"].reset_index(drop=True)
    # Filter to traced training studies that exist in carina_train_cache
    traced_sids = set(df_carina_train["study_id"])
    train_manifest_traced = train_manifest[train_manifest["StudyInstanceUID"].isin(traced_sids)].reset_index(drop=True)

    df_zonal_train = process_cohort(
        manifest_df=train_manifest_traced,
        carina_df=df_carina_train,
        images_dir=IMAGES_DIR,
        model=model,
        device=device,
        save_path=ZONAL_TRAIN_PATH,
        cohort_name="Training (N=6,600)",
    )

    print("\nFeature extraction complete for both splits!")


if __name__ == "__main__":
    main()

