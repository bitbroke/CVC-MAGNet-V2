"""Transparent Distal-Tip Heatmap Peak Decoder.

Requirements:
1. Find local heatmap peaks.
2. Apply a declared confidence threshold.
3. Apply Non-Maximum Suppression (NMS) with distance specified explicitly in native pixels.
4. Convert coordinates to native pixels using paired CoordinateTransform.
5. Preserve candidate confidence and all filtering decisions:
   - raw_candidates: All grid points above threshold.
   - nms_candidates: Peaks retained after native-distance NMS.
   - final_candidates: Peaks retained after any candidate cap.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import scipy.ndimage as ndimage
import torch

try:
    from experiments.hrnet_tip_pilot_v1.src.geometry import (
        CoordinateTransform,
        heatmap_point_to_native,
    )
except ModuleNotFoundError:
    from src.geometry import (
        CoordinateTransform,
        heatmap_point_to_native,
    )


class TransparentTipDecoder:
    """Decodes multi-tip heatmap predictions into native radiograph coordinates."""

    def __init__(
        self,
        conf_threshold: float = 0.20,
        nms_radius_native_px: float = 30.0,
        max_candidates: int = 10,
        kernel_size_hm: int = 3,
    ):
        self.conf_threshold = conf_threshold
        self.nms_radius_native_px = nms_radius_native_px
        self.max_candidates = max_candidates
        self.kernel_size_hm = kernel_size_hm

    def decode(
        self,
        heatmap_prob: np.ndarray,
        transform: CoordinateTransform,
    ) -> Dict[str, Any]:
        """Decode single 2D heatmap [192, 192] to native coordinates.

        Args:
            heatmap_prob: [H_map, W_map] float32 array in [0, 1].
            transform: CoordinateTransform instance for native mapping.

        Returns:
            Dictionary containing:
                - 'final_predictions': List of dicts with native_x, native_y, conf, grid_x, grid_y
                - 'raw_candidates': All candidate grid cells >= conf_threshold
                - 'nms_candidates': Candidates passing NMS before cap
                - 'filtering_stats': Counts at each stage
        """
        hm = np.asarray(heatmap_prob, dtype=np.float32)
        h, w = hm.shape

        # 1. Mask out padding region
        pad_l_hm = transform.padding_left / transform.stride
        pad_t_hm = transform.padding_top / transform.stride
        res_w_hm = transform.resized_width / transform.stride
        res_h_hm = transform.resized_height / transform.stride

        # 2. Raw candidates above threshold
        raw_mask = (hm >= self.conf_threshold)
        raw_coords = np.argwhere(raw_mask)
        raw_candidates: List[Dict[str, Any]] = []

        for u, v in raw_coords:
            if u < pad_t_hm or u >= (pad_t_hm + res_h_hm) or v < pad_l_hm or v >= (pad_l_hm + res_w_hm):
                continue
            nx, ny = heatmap_point_to_native(float(v), float(u), transform)
            raw_candidates.append({
                "grid_x": int(v),
                "grid_y": int(u),
                "conf": float(hm[u, v]),
                "native_x": float(nx),
                "native_y": float(ny),
            })

        raw_candidates.sort(key=lambda c: c["conf"], reverse=True)

        # 3. Local maximum peaks
        max_filtered = ndimage.maximum_filter(hm, size=self.kernel_size_hm, mode="constant", cval=0.0)
        is_peak = (hm == max_filtered) & (hm >= self.conf_threshold)
        peak_coords = np.argwhere(is_peak)

        peak_candidates: List[Dict[str, Any]] = []
        for u, v in peak_coords:
            if u < pad_t_hm or u >= (pad_t_hm + res_h_hm) or v < pad_l_hm or v >= (pad_l_hm + res_w_hm):
                continue
            nx, ny = heatmap_point_to_native(float(v), float(u), transform)
            peak_candidates.append({
                "grid_x": int(v),
                "grid_y": int(u),
                "conf": float(hm[u, v]),
                "native_x": float(nx),
                "native_y": float(ny),
            })

        peak_candidates.sort(key=lambda c: c["conf"], reverse=True)

        # 4. Native Distance NMS
        nms_candidates: List[Dict[str, Any]] = []
        suppressed = np.zeros(len(peak_candidates), dtype=bool)

        for i, cand in enumerate(peak_candidates):
            if suppressed[i]:
                continue
            nms_candidates.append(cand)
            cx, cy = cand["native_x"], cand["native_y"]

            for j in range(i + 1, len(peak_candidates)):
                if not suppressed[j]:
                    ox, oy = peak_candidates[j]["native_x"], peak_candidates[j]["native_y"]
                    dist_native = math.hypot(cx - ox, cy - oy)
                    if dist_native <= self.nms_radius_native_px:
                        suppressed[j] = True

        # 5. Cap to max_candidates
        final_candidates = nms_candidates[: self.max_candidates]

        return {
            "final_predictions": final_candidates,
            "raw_candidates": raw_candidates,
            "nms_candidates": nms_candidates,
            "filtering_stats": {
                "raw_count": len(raw_candidates),
                "peak_count": len(peak_candidates),
                "nms_count": len(nms_candidates),
                "final_count": len(final_candidates),
                "suppressed_by_nms": len(peak_candidates) - len(nms_candidates),
                "suppressed_by_cap": max(0, len(nms_candidates) - len(final_candidates)),
            },
        }
