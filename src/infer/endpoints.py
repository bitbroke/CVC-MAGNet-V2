"""Catheter distal endpoint extraction, graph skeletonization, and vendor heuristic auditing.

Requirements:
- Audit the vendor endpoint heuristic; do not assume it identifies every CVC distal tip correctly.
- Separate segmentation quality from endpoint-selection quality.
- Preserve multiple candidates where identity is unresolved (e.g. crossing branches).
- Document endpoint extraction as an explicit method adaptation.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.ndimage import convolve
import skimage.morphology as morph


@dataclass
class TipCandidate:
    """Represents a localized catheter distal tip candidate."""
    x: float
    y: float
    confidence: float
    component_id: int
    border_distance: float
    is_primary: bool
    source_method: str = "skeleton_border_distance"

    def to_dict(self) -> Dict[str, Union[float, int, bool, str]]:
        return {
            "x": self.x,
            "y": self.y,
            "confidence": self.confidence,
            "component_id": self.component_id,
            "border_distance": self.border_distance,
            "is_primary": self.is_primary,
            "source_method": self.source_method,
        }


def vendor_findpoint(mask: np.ndarray) -> Tuple[Optional[float], Optional[float], str]:
    """Vendor endpoint heuristic from MAG-Net config/test_function.py.
    
    Searches for the lowest pixel (maximum y) to the right of the horizontal centroid.
    Audited: Fails on left-sided CVCs, femoral lines, and empty masks.
    """
    if mask.sum() == 0:
        return None, None, "empty_mask"
    
    choice = np.where(mask > 0.5)
    if len(choice[0]) == 0:
        return None, None, "no_foreground"

    maxy = choice[0].max()
    match_indices = np.where(choice[0] == maxy)[0]
    maxx = choice[1][match_indices[-1]]
    cxmax = choice[1].max()
    cxmin = choice[1].min()
    cx = (cxmax + cxmin) / 2.0

    l0 = choice[0].copy()
    loop_count = 0
    while maxx < cx and loop_count < 1000:
        loop_count += 1
        l0[l0 == maxy] = 0
        if l0.max() == 0:
            break
        maxy = l0.max()
        match_indices = np.where(choice[0] == maxy)[0]
        if len(match_indices) == 0:
            break
        maxx = choice[1][match_indices[-1]]
        cxmax = choice[1].max()
        cxmin = choice[1].min()
        cx = (cxmax + cxmin) / 2.0

    return float(maxx), float(maxy), "vendor_heuristic"


def extract_distal_tips(
    prob_map: np.ndarray,
    threshold: float = 0.5,
    min_component_area: int = 50,
) -> Tuple[List[TipCandidate], Dict[str, Union[str, int, float]]]:
    """Extract distal tip candidates using topological skeletonization and mediastinal centrality.
    
    Args:
        prob_map: (H, W) float32 probability map in [0, 1].
        threshold: Binarization cutoff.
        min_component_area: Minimum pixel count to filter noise.
        
    Returns:
        candidates: List of TipCandidate objects sorted by confidence.
        diag_info: Diagnostic status dictionary.
    """
    h, w = prob_map.shape
    binary = (prob_map > threshold).astype(np.uint8)

    if binary.sum() < min_component_area:
        return [], {
            "status": "no_catheter_detected",
            "foreground_pixels": int(binary.sum()),
            "num_candidates": 0,
        }

    # Morphological skeletonization
    skeleton = morph.skeletonize(binary > 0).astype(np.uint8)
    if skeleton.sum() < 5:
        return [], {
            "status": "skeleton_too_small",
            "foreground_pixels": int(binary.sum()),
            "num_candidates": 0,
        }

    # Kernel to count 8-neighbors on skeleton
    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.uint8)

    neighbor_count = convolve(skeleton, kernel, mode="constant", cval=0)
    # Endpoints have exactly 1 neighbor in the skeleton
    endpoint_mask = (skeleton == 1) & (neighbor_count == 1)

    # Connected components on binary mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    candidates: List[TipCandidate] = []

    for comp_idx in range(1, num_labels):
        area = stats[comp_idx, cv2.CC_STAT_AREA]
        if area < min_component_area:
            continue

        comp_mask = (labels == comp_idx)
        comp_endpoints = np.where(endpoint_mask & comp_mask)

        pts_y = comp_endpoints[0]
        pts_x = comp_endpoints[1]

        if len(pts_x) == 0:
            # Fallback: if loop or circular catheter with no degree-1 node, find point farthest from borders
            comp_pts = np.where(comp_mask)
            cy_pts, cx_pts = comp_pts[0], comp_pts[1]
            dist_left = cx_pts
            dist_right = (w - 1) - cx_pts
            dist_top = cy_pts
            dist_bottom = (h - 1) - cy_pts
            border_dists = np.minimum(np.minimum(dist_left, dist_right), np.minimum(dist_top, dist_bottom))
            best_idx = np.argmax(border_dists)
            best_x, best_y = float(cx_pts[best_idx]), float(cy_pts[best_idx])
            b_dist = float(border_dists[best_idx])
            local_conf = float(prob_map[int(best_y), int(best_x)])
            candidates.append(
                TipCandidate(
                    x=best_x,
                    y=best_y,
                    confidence=local_conf,
                    component_id=comp_idx,
                    border_distance=b_dist,
                    is_primary=True,
                    source_method="max_border_distance_fallback",
                )
            )
            continue

        # Evaluate candidate endpoints by distance to outer borders
        # Distal CVC tip terminates centrally (SVC/CAJ), farthest from image edge
        comp_candidates = []
        for x, y in zip(pts_x, pts_y):
            dist_left = x
            dist_right = (w - 1) - x
            dist_top = y
            dist_bottom = (h - 1) - y
            min_border_dist = min(dist_left, dist_right, dist_top, dist_bottom)

            # Local mean probability in 5x5 window
            r0, r1 = max(0, y - 2), min(h, y + 3)
            c0, c1 = max(0, x - 2), min(w, x + 3)
            local_prob = float(prob_map[r0:r1, c0:c1].mean())

            comp_candidates.append({
                "x": float(x),
                "y": float(y),
                "border_distance": float(min_border_dist),
                "confidence": local_prob,
            })

        # Sort by border distance descending (central-most first)
        comp_candidates.sort(key=lambda c: c["border_distance"], reverse=True)

        for i, c in enumerate(comp_candidates):
            candidates.append(
                TipCandidate(
                    x=c["x"],
                    y=c["y"],
                    confidence=c["confidence"],
                    component_id=comp_idx,
                    border_distance=c["border_distance"],
                    is_primary=(i == 0),
                    source_method="skeleton_endpoints",
                )
            )

    # Sort all candidates by primary status, then confidence
    candidates.sort(key=lambda c: (c.is_primary, c.confidence), reverse=True)

    diag_info = {
        "status": "success" if len(candidates) > 0 else "no_endpoints_found",
        "foreground_pixels": int(binary.sum()),
        "skeleton_pixels": int(skeleton.sum()),
        "num_components": int(num_labels - 1),
        "num_candidates": len(candidates),
    }

    return candidates, diag_info
