"""Heatmap Peak Decoders and Coordinate Transforms.

Implements Section 6 of the Experiment Specification:
- D0: Local maximum at heatmap cell center
- D1: Local subpixel refinement around each peak via Taylor expansion (DARK CVPR 2020)
- NMS peak extraction with thresholding and candidate caps
- Native-coordinate round-trip transforms
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F


def extract_peaks_from_heatmap(
    heatmap: np.ndarray,
    threshold: float = 0.20,
    nms_kernel_size: int = 3,
    max_candidates: int = 10,
    suppression_radius_px: float = 3.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract raw peaks and post-NMS peaks from a single 2D heatmap [H, W].

    Args:
        heatmap: 2D numpy array with values in [0, 1].
        threshold: Minimum peak confidence.
        nms_kernel_size: Neighborhood size for local maximum detection (odd int, typically 3).
        max_candidates: Maximum number of peaks to retain.
        suppression_radius_px: Radius within which duplicate/tied peaks are suppressed.

    Returns:
        raw_peaks: All candidate cells above threshold before NMS.
        nms_peaks: Local maximum peaks above threshold sorted by confidence descending.
    """
    heatmap = np.asarray(heatmap, dtype=np.float32)
    h, w = heatmap.shape
    
    # 1. Raw candidates above threshold
    raw_coords = np.argwhere(heatmap >= threshold)
    raw_peaks: List[Dict[str, Any]] = []
    for u, v in raw_coords:
        raw_peaks.append({
            "grid_y": int(u),
            "grid_x": int(v),
            "conf": float(heatmap[u, v]),
        })
    raw_peaks.sort(key=lambda p: p["conf"], reverse=True)

    # 2. Local maximum detection (NMS)
    max_filtered = ndimage.maximum_filter(heatmap, size=nms_kernel_size, mode="constant", cval=0.0)
    is_peak = (heatmap == max_filtered) & (heatmap >= threshold)
    
    peak_coords = np.argwhere(is_peak)
    candidates: List[Dict[str, Any]] = []
    for u, v in peak_coords:
        candidates.append({
            "grid_y": int(u),
            "grid_x": int(v),
            "conf": float(heatmap[u, v]),
        })

    candidates.sort(key=lambda p: p["conf"], reverse=True)

    # 3. Greedy spatial suppression to eliminate plateaus / grid ties around the same peak
    nms_peaks: List[Dict[str, Any]] = []
    suppressed = np.zeros(len(candidates), dtype=bool)

    for i, cand in enumerate(candidates):
        if suppressed[i]:
            continue
        nms_peaks.append(cand)
        if max_candidates is not None and len(nms_peaks) >= max_candidates:
            break
        # Suppress any subsequent lower or equal candidate within suppression radius
        cy, cx = cand["grid_y"], cand["grid_x"]
        for j in range(i + 1, len(candidates)):
            if not suppressed[j]:
                oy, ox = candidates[j]["grid_y"], candidates[j]["grid_x"]
                if math.hypot(cy - oy, cx - ox) <= suppression_radius_px:
                    suppressed[j] = True

    return raw_peaks, nms_peaks


def decode_d0_cell_center(
    peak: Dict[str, Any],
    stride: int = 4,
    img_size: int = 768,
    native_w: int = 3000,
    native_h: int = 3000,
) -> Tuple[float, float]:
    """D0 Decoder: Heatmap cell center scaled to native coordinates.

    Args:
        peak: Dict with 'grid_x' and 'grid_y'
        stride: Heatmap downsampling stride (e.g. 4)
        img_size: Square model input resolution (e.g. 768)
        native_w: Native image width
        native_h: Native image height

    Returns:
        native_x, native_y in native pixels.
    """
    # Model input coordinates (cell center offset +0.5)
    model_x = (peak["grid_x"] + 0.5) * stride
    model_y = (peak["grid_y"] + 0.5) * stride

    # Transform to native
    native_x = model_x * (native_w / img_size)
    native_y = model_y * (native_h / img_size)
    return float(native_x), float(native_y)


def decode_d1_subpixel_dark(
    heatmap: np.ndarray,
    peak: Dict[str, Any],
    stride: int = 4,
    img_size: int = 768,
    native_w: int = 3000,
    native_h: int = 3000,
    eps: float = 1e-6,
) -> Tuple[float, float, Tuple[float, float]]:
    """D1 Decoder: Local subpixel refinement around peak via Taylor expansion (DARK CVPR 2020).

    Applies local refinement strictly around each peak independently.

    Args:
        heatmap: 2D numpy array [H, W]
        peak: Dict with 'grid_x' (v) and 'grid_y' (u)
        stride: Heatmap stride
        img_size: Square model input resolution
        native_w: Native image width
        native_h: Native image height

    Returns:
        native_x, native_y, (delta_x, delta_y)
    """
    u = peak["grid_y"]
    v = peak["grid_x"]
    H_map, W_map = heatmap.shape

    # If peak is at boundary, fallback to cell center
    if u <= 0 or u >= H_map - 1 or v <= 0 or v >= W_map - 1:
        nx, ny = decode_d0_cell_center(peak, stride, img_size, native_w, native_h)
        return nx, ny, (0.0, 0.0)

    # Use log-transformed surface for Gaussian peak modeling
    patch = np.log(np.maximum(heatmap[u - 1 : u + 2, v - 1 : v + 2], eps))

    # Gradient at center (1, 1)
    g_x = (patch[1, 2] - patch[1, 0]) / 2.0
    g_y = (patch[2, 1] - patch[0, 1]) / 2.0
    g = np.array([g_x, g_y], dtype=np.float64)

    # Hessian matrix
    h_xx = patch[1, 2] - 2.0 * patch[1, 1] + patch[1, 0]
    h_yy = patch[2, 1] - 2.0 * patch[1, 1] + patch[0, 1]
    h_xy = (patch[2, 2] - patch[2, 0] - patch[0, 2] + patch[0, 0]) / 4.0
    Hessian = np.array([[h_xx, h_xy], [h_xy, h_yy]], dtype=np.float64)

    # Solve delta = - H^{-1} g
    try:
        det = Hessian[0, 0] * Hessian[1, 1] - Hessian[0, 1] * Hessian[1, 0]
        if abs(det) > 1e-7:
            inv_H = np.linalg.inv(Hessian)
            delta = -inv_H @ g
            # Clamp offset to [-0.5, 0.5]
            delta_x = float(np.clip(delta[0], -0.5, 0.5))
            delta_y = float(np.clip(delta[1], -0.5, 0.5))
        else:
            delta_x, delta_y = 0.0, 0.0
    except Exception:
        delta_x, delta_y = 0.0, 0.0

    # Subpixel grid coordinates
    sub_x = (v + delta_x + 0.5) * stride
    sub_y = (u + delta_y + 0.5) * stride

    native_x = sub_x * (native_w / img_size)
    native_y = sub_y * (native_h / img_size)

    return float(native_x), float(native_y), (delta_x, delta_y)


def decode_predictions(
    heatmap: np.ndarray,
    decoder_type: str = "D0",
    threshold: float = 0.20,
    nms_kernel_size: int = 3,
    max_candidates: int = 10,
    stride: int = 4,
    img_size: int = 768,
    native_w: int = 3000,
    native_h: int = 3000,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Decode predictions from heatmap using specified decoder (D0 or D1).

    Returns:
        predictions: list of dicts [{'x': float, 'y': float, 'conf': float, 'decoder': str, ...}]
        raw_candidates: list of raw candidate coordinates before NMS
    """
    raw_peaks, nms_peaks = extract_peaks_from_heatmap(
        heatmap,
        threshold=threshold,
        nms_kernel_size=nms_kernel_size,
        max_candidates=max_candidates,
    )

    raw_candidates: List[Dict[str, Any]] = []
    for rp in raw_peaks:
        rx, ry = decode_d0_cell_center(rp, stride, img_size, native_w, native_h)
        raw_candidates.append({
            "x": rx,
            "y": ry,
            "conf": rp["conf"],
            "grid_x": rp["grid_x"],
            "grid_y": rp["grid_y"],
        })

    predictions: List[Dict[str, Any]] = []
    for peak in nms_peaks:
        if decoder_type.upper() == "D1":
            px, py, delta = decode_d1_subpixel_dark(
                heatmap, peak, stride, img_size, native_w, native_h
            )
            predictions.append({
                "x": px,
                "y": py,
                "conf": peak["conf"],
                "decoder": "D1",
                "delta_x": delta[0],
                "delta_y": delta[1],
                "grid_x": peak["grid_x"],
                "grid_y": peak["grid_y"],
            })
        else:  # Default D0
            px, py = decode_d0_cell_center(peak, stride, img_size, native_w, native_h)
            predictions.append({
                "x": px,
                "y": py,
                "conf": peak["conf"],
                "decoder": "D0",
                "grid_x": peak["grid_x"],
                "grid_y": peak["grid_y"],
            })

    return predictions, raw_candidates
