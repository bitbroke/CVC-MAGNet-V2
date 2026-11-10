import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.transforms import FineCropTransform, LetterboxTransform, TransformMetadata


def test_analytical_reversibility_various_aspect_ratios():
    """Verify exact analytical reversibility across diverse image aspect ratios."""
    test_shapes = [
        (2500, 2048),
        (2048, 2500),
        (1987, 2345),  # odd dimensions
        (1024, 1024),
        (3000, 1500),  # extreme 2:1 ratio
        (1500, 3000),  # extreme 1:2 ratio
    ]
    transform = LetterboxTransform(target_size=(512, 512), clahe_enabled=False)

    for orig_h, orig_w in test_shapes:
        meta = transform.compute_params(orig_h, orig_w)
        test_points = [
            (0.0, 0.0),
            (float(orig_w - 1), float(orig_h - 1)),
            (orig_w / 2.0, orig_h / 2.0),
            (123.4, 567.8),
            (orig_w * 0.25, orig_h * 0.75),
        ]
        for tx, ty in test_points:
            mx, my = meta.native_to_model(tx, ty)
            nx, ny = meta.model_to_native(mx, my)
            err = np.hypot(nx - tx, ny - ty)
            assert err < 1e-4, f"Reversibility error {err:.6f} px on shape ({orig_w}, {orig_h})"


def test_opencv_landmark_resampling_alignment():
    """Verify alignment against actual OpenCV bilinear resampled landmarks on rectangular & odd images."""
    test_cases = [
        {"orig_w": 2048, "orig_h": 2500, "lx": 450.0, "ly": 800.0},
        {"orig_w": 2500, "orig_h": 2048, "lx": 1200.0, "ly": 650.0},
        {"orig_w": 1987, "orig_h": 2345, "lx": 789.0, "ly": 1234.0},  # odd dimensions
    ]
    transform = LetterboxTransform(target_size=(512, 512), clahe_enabled=False)

    for case in test_cases:
        orig_w, orig_h = case["orig_w"], case["orig_h"]
        lx, ly = case["lx"], case["ly"]
        
        # Create image with smooth Gaussian landmark
        yy, xx = np.mgrid[:orig_h, :orig_w]
        sigma = 3.0
        native_img = np.exp(-((xx - lx)**2 + (yy - ly)**2) / (2 * sigma**2)).astype(np.float32)

        # Apply letterbox transform
        lb_img, meta = transform.apply_image(native_img)

        # Measure center of mass of resampled landmark in model space
        cyy, cxx = np.mgrid[:512, :512]
        total_mass = np.sum(lb_img)
        assert total_mass > 0, "Resampled landmark has zero mass"
        measured_mx = np.sum(cxx * lb_img) / total_mass
        measured_my = np.sum(cyy * lb_img) / total_mass

        # Predicted model coordinate using TransformMetadata
        pred_mx, pred_my = meta.native_to_model(lx, ly)
        err = np.hypot(measured_mx - pred_mx, measured_my - pred_my)

        # Must agree with continuous resampled landmark center within subpixel accuracy (< 0.05 px)
        assert err < 0.05, f"Disagreement with resampled landmark: {err:.4f} px on shape ({orig_w}, {orig_h})"


def test_coarse_mask_unpadding_spatial_alignment():
    """Verify unpadding coarse prediction removes padding and aligns with native space."""
    orig_w, orig_h = 2000, 2500
    transform = LetterboxTransform(target_size=(512, 512), clahe_enabled=False)
    meta = transform.compute_params(orig_h, orig_w)

    # Synthetic native feature placed at native coordinates (1000, 1500)
    native_mask = np.zeros((orig_h, orig_w), dtype=np.float32)
    native_mask[1490:1510, 990:1010] = 1.0

    # Letterbox the mask
    lb_mask = transform.apply_mask(native_mask, meta, interpolation=cv2.INTER_LINEAR)

    # Now simulate the pipeline repair: unpad and resize valid content back to native
    valid_content = lb_mask[meta.pad_top : meta.pad_top + meta.new_h, meta.pad_left : meta.pad_left + meta.new_w]
    restored_native = cv2.resize(valid_content, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Restored feature center must match original feature center (1000, 1500)
    yy, xx = np.mgrid[:orig_h, :orig_w]
    orig_cx = np.sum(xx * native_mask) / np.sum(native_mask)
    orig_cy = np.sum(yy * native_mask) / np.sum(native_mask)

    restored_cx = np.sum(xx * restored_native) / np.sum(restored_native)
    restored_cy = np.sum(yy * restored_native) / np.sum(restored_native)

    shift = np.hypot(restored_cx - orig_cx, restored_cy - orig_cy)
    # On 5x downsampled-upsampled feature (2500 -> 512 -> 2500), shift is < 1.5 native pixels (< 0.3 model pixels)
    assert shift < 1.5, f"Unpadding continuous spatial shift exceeded tolerance: {shift:.3f} px"


def test_out_of_bounds_coordinate_preservation():
    """Verify out-of-bounds coordinates are not clamped onto image borders."""
    orig_w, orig_h = 2000, 2000
    meta = TransformMetadata(
        scale=0.256,
        pad_x=0.0,
        pad_y=0.0,
        orig_w=orig_w,
        orig_h=orig_h,
        target_w=512,
        target_h=512,
        new_w=512,
        new_h=512,
        pad_left=0,
        pad_top=0,
    )
    # Model prediction outside valid range (e.g., negative or exceeding target_w)
    pred_x, pred_y = -10.0, 525.0
    nat_x, nat_y = meta.model_to_native(pred_x, pred_y)

    # Verify native coordinates are outside image and NOT clamped
    assert nat_x < 0.0, f"Expected negative nat_x, got {nat_x}"
    assert nat_y >= orig_h, f"Expected nat_y >= orig_h, got {nat_y}"

    is_in_bounds = (0.0 <= nat_x < orig_w) and (0.0 <= nat_y < orig_h)
    assert not is_in_bounds, "Expected is_in_bounds to be False"
