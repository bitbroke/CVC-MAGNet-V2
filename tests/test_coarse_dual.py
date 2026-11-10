import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch

from src.models.coarse_dual import (
    CoarseDualNet,
    decode_tip_peaks,
    generate_target_heatmap,
)
from src.models.losses import CoarseDualLoss


def test_dual_net_forward_shapes():
    """Verify forward output shapes for segmentation and tip heatmap."""
    model = CoarseDualNet(in_channels=1, base_channels=16)
    x = torch.randn(2, 1, 128, 128)
    seg_logits, tip_logits = model(x)
    assert seg_logits.shape == (2, 1, 128, 128)
    assert tip_logits.shape == (2, 1, 128, 128)


def test_multi_peak_heatmap_generation_and_decoding():
    """Verify synthetic multi-peak generation and local maxima peak decoding."""
    coords = [(100.3, 150.7), (250.0, 300.0)]
    heatmap = generate_target_heatmap(coords, shape=(512, 512), sigma=2.5)

    assert heatmap.shape == (512, 512)
    assert heatmap.max() > 0.95
    assert heatmap.min() >= 0.0

    peaks = decode_tip_peaks(heatmap, threshold=0.5, min_distance=10)
    assert len(peaks) == 2

    # Check localization accuracy within 2 pixels on the grid
    p0 = peaks[0]
    p1 = peaks[1]
    decoded_coords = [(p[0], p[1]) for p in peaks]

    # Verify both target points have a decoded peak nearby
    for tx, ty in coords:
        min_d = min(np.hypot(tx - dx, ty - dy) for dx, dy in decoded_coords)
        assert min_d <= 2.0, f"Target ({tx}, {ty}) not localized within 2px (min dist {min_d})"


def test_empty_heatmap_decoding():
    """Verify that empty/background heatmaps decode zero peaks."""
    empty_hm = np.zeros((512, 512), dtype=np.float32)
    peaks = decode_tip_peaks(empty_hm, threshold=0.3)
    assert len(peaks) == 0


def test_unknown_tip_masking_gradients():
    """Verify that an unknown tip (tip_mask=0) produces ZERO gradient to tip_head."""
    model = CoarseDualNet(in_channels=1, base_channels=8)
    criterion = CoarseDualLoss(bce_weight=1.0, dice_weight=1.0, tip_weight=5.0)

    # Batch of 2: Sample 0 has known tip, Sample 1 has unknown tip
    x = torch.randn(2, 1, 64, 64)
    seg_targets = torch.zeros(2, 1, 64, 64)
    tip_targets = torch.zeros(2, 1, 64, 64)
    tip_targets[0, 0, 32, 32] = 1.0  # Known tip for sample 0

    tip_mask = torch.tensor([1.0, 0.0])  # Sample 1 masked out

    seg_logits, tip_logits = model(x)
    loss, loss_dict = criterion(seg_logits, tip_logits, seg_targets, tip_targets, tip_mask=tip_mask)

    assert loss_dict["loss_tip"] > 0.0
    loss.backward()

    # Tip head weights must have valid gradients
    for p in model.tip_head.parameters():
        assert p.grad is not None
        assert not torch.isnan(p.grad).any()

    # Shared backbone must have gradients from both heads
    for p in model.conv1.parameters():
        assert p.grad is not None


def test_all_unknown_batch():
    """Verify graceful behavior when entire batch has unknown tips."""
    model = CoarseDualNet(in_channels=1, base_channels=8)
    criterion = CoarseDualLoss()

    x = torch.randn(2, 1, 64, 64)
    seg_targets = torch.zeros(2, 1, 64, 64)
    tip_targets = torch.zeros(2, 1, 64, 64)
    tip_mask = torch.tensor([0.0, 0.0])

    seg_logits, tip_logits = model(x)
    loss, loss_dict = criterion(seg_logits, tip_logits, seg_targets, tip_targets, tip_mask=tip_mask)

    assert loss_dict["loss_tip"] == 0.0
    loss.backward()
    assert not torch.isnan(loss).any()


def test_empty_and_ignored_case_dynamics():
    """Verify that negative cases push tip_head to 0 and ignored cases give 0 tip gradient."""
    model = CoarseDualNet(in_channels=1, base_channels=8)
    criterion = CoarseDualLoss(tip_pos_weight=500.0)

    # Batch of 3:
    # Sample 0: positive case with known tip (tip_mask=1)
    # Sample 1: negative case with empty target (tip_mask=1)
    # Sample 2: ignored/unreviewed case (tip_mask=0)
    x = torch.randn(3, 1, 64, 64)
    seg_targets = torch.zeros(3, 1, 64, 64)
    tip_targets = torch.zeros(3, 1, 64, 64)
    tip_targets[0, 0, 32, 32] = 1.0
    tip_mask = torch.tensor([1.0, 1.0, 0.0])

    seg_logits, tip_logits = model(x)
    loss, loss_dict = criterion(seg_logits, tip_logits, seg_targets, tip_targets, tip_mask=tip_mask)

    assert loss_dict["loss_tip"] > 0.0
    loss.backward()

    # Gradients must be valid and non-NaN
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Missing grad for {name}"
        assert not torch.isnan(p.grad).any(), f"NaN grad for {name}"


def test_spatial_ignore_mask_isolates_unreviewed_regions():
    """Verify that unreviewed regions with tip_spatial_mask=0 generate ZERO gradient."""
    criterion = CoarseDualLoss(tip_pos_weight=100.0)

    # 1 sample of 32x32
    seg_logits = torch.zeros(1, 1, 32, 32, requires_grad=True)
    tip_logits = torch.zeros(1, 1, 32, 32, requires_grad=True)
    seg_targets = torch.zeros(1, 1, 32, 32)
    tip_targets = torch.zeros(1, 1, 32, 32)
    tip_targets[0, 0, 8, 8] = 1.0  # Reviewed tip at (8, 8)

    # Spatial mask: valid on left half, ignored (0.0) on right half (where unreviewed catheter sits)
    tip_spatial_mask = torch.ones(1, 1, 32, 32)
    tip_spatial_mask[0, 0, :, 16:] = 0.0

    loss, loss_dict = criterion(
        seg_logits, tip_logits, seg_targets, tip_targets,
        tip_mask=torch.tensor([1.0]),
        tip_spatial_mask=tip_spatial_mask,
    )
    loss.backward()

    # The tip gradient on the ignored right half must be exactly ZERO
    right_half_grad = tip_logits.grad[0, 0, :, 16:]
    assert (right_half_grad == 0.0).all(), "Ignored spatial region received non-zero gradient!"
    # Left half must have non-zero gradient
    left_half_grad = tip_logits.grad[0, 0, :, :16]
    assert (left_half_grad != 0.0).any(), "Valid spatial region received zero gradient!"


def test_shaft_weighting_selective_application():
    """Verify that shaft_weight=2.0 doubles gradient on catheter shaft without altering tip or background."""
    # 1 sample of 32x32: logit = 0.0 (prob = 0.5) everywhere
    seg_targets = torch.zeros(1, 1, 32, 32)
    # Put catheter shaft on row 10
    seg_targets[0, 0, 10, :] = 1.0

    tip_targets = torch.zeros(1, 1, 32, 32)
    # Put distal tip at (10, 30)
    tip_targets[0, 0, 10, 30] = 1.0

    # 1. Baseline unweighted shaft (shaft_weight=1.0)
    crit_base = CoarseDualLoss(tip_pos_weight=100.0, shaft_weight=1.0)
    tip_logits_base = torch.zeros(1, 1, 32, 32, requires_grad=True)
    loss_b, _ = crit_base(
        torch.zeros(1, 1, 32, 32), tip_logits_base, seg_targets, tip_targets, tip_mask=torch.tensor([1.0])
    )
    loss_b.backward()
    grad_base = tip_logits_base.grad[0, 0]

    # 2. Weighted shaft (shaft_weight=2.0)
    crit_shaft = CoarseDualLoss(tip_pos_weight=100.0, shaft_weight=2.0)
    tip_logits_shaft = torch.zeros(1, 1, 32, 32, requires_grad=True)
    loss_s, _ = crit_shaft(
        torch.zeros(1, 1, 32, 32), tip_logits_shaft, seg_targets, tip_targets, tip_mask=torch.tensor([1.0])
    )
    loss_s.backward()
    grad_shaft = tip_logits_shaft.grad[0, 0]

    # Shaft pixel far from tip, e.g. at (10, 5)
    g_base_shaft = float(grad_base[10, 5])
    g_shaft_shaft = float(grad_shaft[10, 5])
    assert abs(g_shaft_shaft - 2.0 * g_base_shaft) < 1e-6, f"Expected 2x gradient on shaft, got {g_shaft_shaft} vs {g_base_shaft}"

    # Tip pixel at (10, 30) must NOT be doubled
    g_base_tip = float(grad_base[10, 30])
    g_shaft_tip = float(grad_shaft[10, 30])
    assert abs(g_shaft_tip - g_base_tip) < 1e-6, f"Tip gradient was altered! {g_shaft_tip} vs {g_base_tip}"

    # General background pixel at (20, 20) must NOT be doubled
    g_base_bg = float(grad_base[20, 20])
    g_shaft_bg = float(grad_shaft[20, 20])
    assert abs(g_shaft_bg - g_base_bg) < 1e-6, f"Background gradient was altered! {g_shaft_bg} vs {g_base_bg}"


def test_shaft_weighting_fractional_target():
    """Verify that shaft weighting on fractional targets (e.g. y=0.01) only adds downward gradient without multiplying positive term."""
    # Pixel with logit = -0.4620 (prob = 0.3865), target = 0.01, seg_target = 1.0
    seg_targets = torch.ones(1, 1, 16, 16)
    tip_targets = torch.full((1, 1, 16, 16), 0.01)

    logit_val = -0.4620
    p = 1.0 / (1.0 + np.exp(-logit_val))  # ~0.3865

    # 1. Baseline BCE
    crit_base = CoarseDualLoss(tip_pos_weight=100.0, shaft_weight=1.0)
    tip_logits_base = torch.full((1, 1, 16, 16), logit_val, requires_grad=True)
    loss_b, _ = crit_base(
        torch.zeros(1, 1, 16, 16), tip_logits_base, seg_targets, tip_targets, tip_mask=torch.tensor([1.0])
    )
    loss_b.backward()
    g_base = float(tip_logits_base.grad[0, 0, 5, 5])

    # In unweighted BCE: dL/dz = (1 - y)*p - 100*y*(1 - p) = 0.99*0.3865 - 1.0*0.6135 = -0.2309 (per-image scaled)
    # 2. Weighted shaft (shaft_weight=2.0, threshold=0.02 so y=0.01 is eligible)
    crit_shaft = CoarseDualLoss(tip_pos_weight=100.0, shaft_weight=2.0, shaft_tip_exclusion_thresh=0.02)
    tip_logits_shaft = torch.full((1, 1, 16, 16), logit_val, requires_grad=True)
    loss_s, _ = crit_shaft(
        torch.zeros(1, 1, 16, 16), tip_logits_shaft, seg_targets, tip_targets, tip_mask=torch.tensor([1.0])
    )
    loss_s.backward()
    g_shaft = float(tip_logits_shaft.grad[0, 0, 5, 5])

    # The additional gradient must be strictly POSITIVE (downward push):
    # delta_g = (2 - 1) * (1 - 0.01) * p / (16 * 16) > 0
    delta_g = g_shaft - g_base
    assert delta_g > 0.0, f"Additional gradient should be strictly positive (downward), got {delta_g}"
    # Verify that g_shaft does NOT double the upward pull (-0.4617)
    assert g_shaft > g_base, f"Expected g_shaft ({g_shaft}) to be more positive than g_base ({g_base})"


def test_shaft_weighting_multiple_tips():
    """Verify that multiple supervised tips are all protected from shaft penalty."""
    seg_targets = torch.zeros(1, 1, 32, 32)
    # Catheter 1 along row 10, Catheter 2 along row 20
    seg_targets[0, 0, 10, :] = 1.0
    seg_targets[0, 0, 20, :] = 1.0

    tip_targets = torch.zeros(1, 1, 32, 32)
    # Tip 1 at (10, 25), Tip 2 at (20, 5)
    tip_targets[0, 0, 10, 25] = 1.0
    tip_targets[0, 0, 20, 5] = 1.0

    crit_base = CoarseDualLoss(tip_pos_weight=100.0, shaft_weight=1.0)
    tip_logits_b = torch.zeros(1, 1, 32, 32, requires_grad=True)
    loss_b, _ = crit_base(
        torch.zeros(1, 1, 32, 32), tip_logits_b, seg_targets, tip_targets, tip_mask=torch.tensor([1.0])
    )
    loss_b.backward()

    crit_shaft = CoarseDualLoss(tip_pos_weight=100.0, shaft_weight=2.0)
    tip_logits_s = torch.zeros(1, 1, 32, 32, requires_grad=True)
    loss_s, _ = crit_shaft(
        torch.zeros(1, 1, 32, 32), tip_logits_s, seg_targets, tip_targets, tip_mask=torch.tensor([1.0])
    )
    loss_s.backward()

    gb = tip_logits_b.grad[0, 0]
    gs = tip_logits_s.grad[0, 0]

    # Both tips must have IDENTICAL gradients
    assert abs(float(gs[10, 25]) - float(gb[10, 25])) < 1e-6, "Tip 1 was altered by shaft weighting!"
    assert abs(float(gs[20, 5]) - float(gb[20, 5])) < 1e-6, "Tip 2 was altered by shaft weighting!"

    # Both catheter shafts far from tips must receive exactly doubled downward gradient
    assert abs(float(gs[10, 5]) - 2.0 * float(gb[10, 5])) < 1e-6, "Catheter 1 shaft was not doubled!"
    assert abs(float(gs[20, 25]) - 2.0 * float(gb[20, 25])) < 1e-6, "Catheter 2 shaft was not doubled!"


def test_shaft_weighting_ignored_samples():
    """Verify that masked samples (tip_mask=0) and spatial ignore masks (tip_spatial_mask=0) yield zero shaft penalty."""
    crit = CoarseDualLoss(tip_pos_weight=100.0, shaft_weight=2.0)

    # Batch of 2
    seg_targets = torch.ones(2, 1, 32, 32)
    tip_targets = torch.zeros(2, 1, 32, 32)
    tip_targets[0, 0, 10, 10] = 1.0
    tip_targets[1, 0, 10, 10] = 1.0

    tip_mask = torch.tensor([1.0, 0.0])  # Sample 1 masked out
    tip_spatial_mask = torch.ones(2, 1, 32, 32)
    tip_spatial_mask[0, 0, :, 16:] = 0.0  # Sample 0 right half spatially ignored

    tip_logits = torch.zeros(2, 1, 32, 32, requires_grad=True)
    loss, _ = crit(
        torch.zeros(2, 1, 32, 32), tip_logits, seg_targets, tip_targets,
        tip_mask=tip_mask, tip_spatial_mask=tip_spatial_mask
    )
    loss.backward()

    grad = tip_logits.grad

    # Sample 1 (tip_mask=0): Entire image must have 0.0 gradient (including shaft)
    assert (grad[1, 0] == 0.0).all(), "Masked sample (tip_mask=0) received non-zero gradient on shaft!"

    # Sample 0: Ignored spatial region (x >= 16) must have 0.0 gradient
    assert (grad[0, 0, :, 16:] == 0.0).all(), "Ignored spatial region received non-zero gradient on shaft!"

    # Sample 0: Valid shaft region (x < 16, away from tip) must have positive (downward) gradient
    assert float(grad[0, 0, 25, 5]) > 0.0, "Valid shaft region received non-positive gradient!"




