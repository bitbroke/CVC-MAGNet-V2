"""Coordinate Geometry and Paired Forward/Inverse Transforms for Tip Heatmaps.

Ensures mathematically exact round-trip mapping between:
1. Native radiograph grid [native_w, native_h]
2. Letterboxed model input [768, 768] with padding [pad_left, pad_top]
3. Stride-4 spatial heatmap grid [192, 192]

Rules:
- Native pixel centers and fractional coordinates preserved.
- Non-square images handled via uniform letterbox scaling.
- Near-border and out-of-frame points handled without wrapping.
- Multiple peaks combined via torch.maximum (not summed > 1).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np
import torch


@dataclass(frozen=True)
class CoordinateTransform:
    """Immutable record of image coordinate transformations."""
    native_width: int
    native_height: int
    resized_width: int
    resized_height: int
    padding_left: int
    padding_top: int
    model_input_width: int = 768
    model_input_height: int = 768
    heatmap_width: int = 192
    heatmap_height: int = 192
    stride: int = 4

    @classmethod
    def from_native_dims(
        cls,
        native_w: int,
        native_h: int,
        target_size: int = 768,
        stride: int = 4,
    ) -> "CoordinateTransform":
        scale = min(target_size / native_w, target_size / native_h)
        scaled_w = int(round(native_w * scale))
        scaled_h = int(round(native_h * scale))
        pad_x = (target_size - scaled_w) // 2
        pad_y = (target_size - scaled_h) // 2
        hm_size = target_size // stride
        return cls(
            native_width=native_w,
            native_height=native_h,
            resized_width=scaled_w,
            resized_height=scaled_h,
            padding_left=pad_x,
            padding_top=pad_y,
            model_input_width=target_size,
            model_input_height=target_size,
            heatmap_width=hm_size,
            heatmap_height=hm_size,
            stride=stride,
        )

    def to_dict(self) -> Dict[str, int]:
        return {
            "native_width": self.native_width,
            "native_height": self.native_height,
            "resized_width": self.resized_width,
            "resized_height": self.resized_height,
            "padding_left": self.padding_left,
            "padding_top": self.padding_top,
            "model_input_width": self.model_input_width,
            "model_input_height": self.model_input_height,
            "heatmap_width": self.heatmap_width,
            "heatmap_height": self.heatmap_height,
            "stride": self.stride,
        }


def native_point_to_heatmap(
    x_native: float,
    y_native: float,
    transform: CoordinateTransform,
) -> Tuple[float, float]:
    """Map continuous native image coordinates (x, y) to continuous heatmap coordinates (hx, hy)."""
    # 1. Scale to resized letterbox inner dimension
    x_res = (x_native / transform.native_width) * transform.resized_width
    y_res = (y_native / transform.native_height) * transform.resized_height
    # 2. Add padding in model input coordinates
    x_model = x_res + transform.padding_left
    y_model = y_res + transform.padding_top
    # 3. Downscale to heatmap coordinates
    hx = x_model / transform.stride
    hy = y_model / transform.stride
    return float(hx), float(hy)


def heatmap_point_to_native(
    hx: float,
    hy: float,
    transform: CoordinateTransform,
) -> Tuple[float, float]:
    """Map continuous heatmap coordinates (hx, hy) back to native radiograph coordinates (x, y)."""
    # 1. Upscale from heatmap to model input coordinates
    x_model = hx * transform.stride
    y_model = hy * transform.stride
    # 2. Subtract padding
    x_res = x_model - transform.padding_left
    y_res = y_model - transform.padding_top
    # 3. Scale back to native image dimensions
    x_native = (x_res / max(transform.resized_width, 1)) * transform.native_width
    y_native = (y_res / max(transform.resized_height, 1)) * transform.native_height
    return float(x_native), float(y_native)


def render_multitip_gaussian_heatmap(
    tips_native: List[Tuple[float, float]],
    transform: CoordinateTransform,
    sigma_hm: float = 1.5,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Render continuous-center Gaussian targets on stride-4 heatmap.

    Args:
        tips_native: List of (x_native, y_native) tuples.
        transform: CoordinateTransform instance.
        sigma_hm: Gaussian standard deviation in heatmap cell units.
        device: Output torch device.

    Returns:
        heatmap: [1, H_map, W_map] float32 tensor with maximum-combined Gaussians in [0, 1].
        valid_mask: [1, H_map, W_map] float32 tensor with 1.0 in active image area, 0.0 in padding.
    """
    H_map, W_map = transform.heatmap_height, transform.heatmap_width
    heatmap = torch.zeros((1, H_map, W_map), dtype=torch.float32, device=device)

    # Compute valid mask in heatmap coordinates (excluding padding)
    pad_l_hm = transform.padding_left / transform.stride
    pad_t_hm = transform.padding_top / transform.stride
    res_w_hm = transform.resized_width / transform.stride
    res_h_hm = transform.resized_height / transform.stride

    y_grid = torch.arange(H_map, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, W_map)
    x_grid = torch.arange(W_map, dtype=torch.float32, device=device).unsqueeze(0).repeat(H_map, 1)

    in_valid = (
        (x_grid >= pad_l_hm)
        & (x_grid < (pad_l_hm + res_w_hm))
        & (y_grid >= pad_t_hm)
        & (y_grid < (pad_t_hm + res_h_hm))
    )
    valid_mask = in_valid.float().unsqueeze(0)

    if not tips_native:
        return heatmap, valid_mask

    radius = int(math.ceil(3.0 * sigma_hm))

    for nx, ny in tips_native:
        hx, hy = native_point_to_heatmap(nx, ny, transform)

        # Skip tips outside model bounds
        if hx < 0 or hx >= W_map or hy < 0 or hy >= H_map:
            continue

        ix = int(round(hx))
        iy = int(round(hy))

        x_min = max(0, ix - radius)
        x_max = min(W_map, ix + radius + 1)
        y_min = max(0, iy - radius)
        y_max = min(H_map, iy + radius + 1)

        patch_x = x_grid[y_min:y_max, x_min:x_max]
        patch_y = y_grid[y_min:y_max, x_min:x_max]

        gaussian = torch.exp(-((patch_x - hx) ** 2 + (patch_y - hy) ** 2) / (2.0 * sigma_hm ** 2))
        if gaussian.numel() > 0:
            # Mask out any portion extending into padding
            gaussian = gaussian * valid_mask[0, y_min:y_max, x_min:x_max]
            heatmap[0, y_min:y_max, x_min:x_max] = torch.maximum(
                heatmap[0, y_min:y_max, x_min:x_max], gaussian
            )

    return heatmap, valid_mask
