"""Unit tests for Coordinate Geometry and Forward/Inverse Round-Trip Consistency.

Tests:
1. Fractional coordinates.
2. Non-square images (tall and wide aspect ratios).
3. Letterbox padding offsets.
4. Near-border tips.
5. Multiple tips in one image (max-combination, no sum > 1.0).
6. Exact round-trip consistency (native -> heatmap -> native within 1e-4 pixels).
7. Rasterization and decoding of synthetic peaks.
"""

import math
import sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXP_ROOT = PROJECT_ROOT / "experiments" / "hrnet_tip_pilot_v1"
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from experiments.hrnet_tip_pilot_v1.src.geometry import (
    CoordinateTransform,
    native_point_to_heatmap,
    heatmap_point_to_native,
    render_multitip_gaussian_heatmap,
)


def test_geometry_round_trip():
    print("=== Testing Geometry Round-Trip Consistency ===")

    test_cases = [
        # (native_w, native_h, points)
        (3000, 3000, [(1500.0, 1500.0), (123.45, 678.91), (2999.0, 2999.0), (0.5, 0.5)]),
        (2048, 2500, [(1024.0, 1250.0), (45.6, 2400.1), (2000.0, 100.0)]),
        (3072, 1920, [(1536.0, 960.0), (12.3, 1900.5), (3050.0, 50.0)]),
        (2430, 2430, [(1215.75, 1215.25), (500.123, 1800.789)]),
    ]

    for nw, nh, points in test_cases:
        tf = CoordinateTransform.from_native_dims(nw, nh, target_size=768, stride=4)

        for nx, ny in points:
            hx, hy = native_point_to_heatmap(nx, ny, tf)
            rx, ry = heatmap_point_to_native(hx, hy, tf)

            err_x = abs(rx - nx)
            err_y = abs(ry - ny)

            assert err_x < 1e-4, f"Round-trip X failed: {nx} -> {hx} -> {rx} (err={err_x}) on {nw}x{nh}"
            assert err_y < 1e-4, f"Round-trip Y failed: {ny} -> {hy} -> {ry} (err={err_y}) on {nw}x{nh}"

    print("[PASS] Forward/Inverse coordinate consistency strictly verified across square and non-square images!")


def test_gaussian_multitip_rasterization():
    print("=== Testing Multi-Tip Gaussian Rasterization ===")

    tf = CoordinateTransform.from_native_dims(3000, 2500, target_size=768, stride=4)
    tips = [(1000.0, 1200.0), (1050.0, 1220.0), (2200.0, 1800.0)]

    hm, valid = render_multitip_gaussian_heatmap(tips, tf, sigma_hm=1.5)

    assert hm.shape == (1, 192, 192), f"Unexpected shape {hm.shape}"
    assert valid.shape == (1, 192, 192), f"Unexpected valid shape {valid.shape}"

    # Verify bounds
    assert hm.min() >= 0.0, "Heatmap has negative values"
    assert hm.max() <= 1.000001, f"Heatmap exceeds 1.0: {hm.max()}"

    # Verify padding has strictly zero heatmap and zero valid
    pad_l_hm = int(tf.padding_left / tf.stride)
    pad_t_hm = int(tf.padding_top / tf.stride)
    if pad_l_hm > 0:
        assert (hm[:, :, :pad_l_hm] == 0).all(), "Non-zero heatmap in left padding"
        assert (valid[:, :, :pad_l_hm] == 0).all(), "Non-zero valid in left padding"
    if pad_t_hm > 0:
        assert (hm[:, :pad_t_hm, :] == 0).all(), "Non-zero heatmap in top padding"
        assert (valid[:, :pad_t_hm, :] == 0).all(), "Non-zero valid in top padding"

    # Verify peak decoding
    for nx, ny in tips:
        hx, hy = native_point_to_heatmap(nx, ny, tf)
        ix, iy = int(round(hx)), int(round(hy))
        peak_val = hm[0, iy, ix].item()
        assert peak_val >= 0.90, f"Expected strong peak at ({ix}, {iy}), got {peak_val}"

    print("[PASS] Multi-tip Gaussian rasterization and peak bounds verified!")


if __name__ == "__main__":
    test_geometry_round_trip()
    test_gaussian_multitip_rasterization()
    print("\nALL GEOMETRY TESTS PASSED SUCCESSFULLY!")
