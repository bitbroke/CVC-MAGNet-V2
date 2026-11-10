"""Polyline rasterization, tip heatmap generation, and attention mask supervision.

Method Adaptation Note:
In the original MAG-Net publication (Wang et al., 2022), PICC segmentation used dense
manual pixel boundary masks. In RANZCR CLiP, catheter annotations are unoriented polylines.
We adapt MAG-Net to polyline annotations by rasterizing each polyline with an empirical
stroke thickness hyperparameter (default: 10 native pixels for CVC; uncalibrated to physical millimeters).

For fine-stage mask attention supervision, MAG-Net supervises a learned spatial attention module
against a smoothed mask generated via an 11x11 Gaussian blur kernel (sigma=2.0).
"""
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
import torch


def rasterize_polyline(
    points: Union[List[List[float]], np.ndarray],
    image_shape: Tuple[int, int],
    thickness: int = 10,
) -> np.ndarray:
    """Rasterize a single polyline into a binary mask.
    
    Coordinates extending outside native image bounds are clipped to [0, w-1] x [0, h-1]
    for clean segmentation rasterization. Original annotations are preserved in raw storage.
    Note: Clipped boundary endpoints represent edge-crossing cuts, NOT true distal tips.
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 2:
        return mask
    
    # Clip coordinates to image boundary
    pts_clipped = np.clip(pts, [0.0, 0.0], [float(w - 1), float(h - 1)])
    pts_int = np.round(pts_clipped).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(mask, [pts_int], isClosed=False, color=1, thickness=thickness)
    return mask


def rasterize_study_polylines(
    polylines: List[Union[List[List[float]], np.ndarray]],
    image_shape: Tuple[int, int],
    thickness: int = 10,
) -> np.ndarray:
    """Rasterize multiple catheter polylines directly into a single binary mask in-place.
    
    Coordinates extending outside native image bounds are clipped to [0, w-1] x [0, h-1]
    for clean segmentation rasterization. Original annotations are preserved in raw storage.
    Note: Clipped boundary endpoints represent edge-crossing cuts, NOT true distal tips.
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    pts_list = []
    for poly in polylines:
        pts = np.asarray(poly, dtype=np.float32)
        if len(pts) >= 2:
            pts_clipped = np.clip(pts, [0.0, 0.0], [float(w - 1), float(h - 1)])
            pts_int = np.round(pts_clipped).astype(np.int32).reshape((-1, 1, 2))
            pts_list.append(pts_int)
    if pts_list:
        cv2.polylines(mask, pts_list, isClosed=False, color=1, thickness=thickness)
    return mask


def generate_gaussian_tip_heatmap(
    tip_coords: List[Tuple[float, float]],
    image_shape: Tuple[int, int],
    sigma: float = 3.0,
    radius: Optional[int] = None,
) -> np.ndarray:
    """Generate a continuous Gaussian heatmap centered at catheter distal tips.
    
    Args:
        tip_coords: List of (tip_x, tip_y) coordinates in pixel space.
        image_shape: (height, width) of the output heatmap.
        sigma: Standard deviation of the Gaussian kernel.
        radius: Radius in pixels around each tip (default 3*sigma).
        
    Returns:
        heatmap: (H, W) float32 array in [0.0, 1.0].
    """
    h, w = image_shape
    heatmap = np.zeros((h, w), dtype=np.float32)
    if radius is None:
        radius = int(np.ceil(3 * sigma))

    for tx, ty in tip_coords:
        if np.isnan(tx) or np.isnan(ty):
            continue
        cx, cy = int(round(tx)), int(round(ty))
        x0 = max(0, cx - radius)
        x1 = min(w, cx + radius + 1)
        y0 = max(0, cy - radius)
        y1 = min(h, cy + radius + 1)

        if x1 <= x0 or y1 <= y0:
            continue

        ys, xs = np.ogrid[y0:y1, x0:x1]
        dist_sq = (xs - cx) ** 2 + (ys - cy) ** 2
        g = np.exp(-dist_sq / (2 * sigma ** 2)).astype(np.float32)
        heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], g)

    return heatmap


def create_gaussian_attention_mask(
    binary_mask: np.ndarray,
    kernel_size: int = 11,
    sigma: float = 2.0,
) -> np.ndarray:
    """Generate the Gaussian-smoothed mask for MAG-Net fine stage attention supervision.
    
    Matches the published formulation:
    "G is the ground truth catheter mask smoothed by an 11x11 Gaussian blur kernel."
    
    Args:
        binary_mask: (H, W) array with values in {0, 1} or [0, 1].
        kernel_size: Gaussian kernel dimension (must be odd, default 11).
        sigma: Gaussian standard deviation (default 2.0).
        
    Returns:
        smoothed_mask: (H, W) float32 array in [0.0, 1.0].
    """
    mask_f32 = binary_mask.astype(np.float32)
    if mask_f32.max() > 1.0:
        mask_f32 = mask_f32 / 255.0
        
    if mask_f32.sum() == 0:
        return np.zeros_like(mask_f32)
        
    blurred = cv2.GaussianBlur(mask_f32, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)
    # Normalize to [0, 1] peak if non-empty
    b_max = blurred.max()
    if b_max > 0:
        blurred = blurred / b_max
    return blurred.astype(np.float32)
