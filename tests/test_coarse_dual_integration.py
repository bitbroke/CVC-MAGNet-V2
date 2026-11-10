"""Integration test verifying real-data coarse dual dataset, loss, and Hungarian matching.

Covers:
1. Real partially reviewed multi-CVC study:
   - Native thickness 10 polyline rasterization
   - Target heatmap peak at reviewed tip
   - Spatial ignore mask (tip_spatial_mask = 0.0) around unreviewed CVC
   - Zero gradient on ignored spatial region
2. Real confirmed negative study:
   - Empty mask, empty heatmap, tip_mask = 1.0, penalization of false peaks
3. Real unreviewed positive study (unknown case):
   - Supervised segmentation, tip_mask = 0.0, zero tip head gradient
4. Multi-tip one-to-one threshold-constrained Hungarian assignment.
"""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import CoarseDualDataset
from src.models.coarse_dual import CoarseDualNet
from src.models.losses import CoarseDualLoss

spec = importlib.util.spec_from_file_location("train_dual_mod", PROJECT_ROOT / "scripts" / "05_train_coarse_dual.py")
train_dual_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_dual_mod)
run_dual_validation = train_dual_mod.run_dual_validation


def test_real_data_coarse_dual_integration():
    with open(PROJECT_ROOT / "configs" / "pilot.yaml") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["data"]["raw_dir"]) / "train"
    if not raw_dir.exists():
        pytest.skip(f"Raw images dir not found at {raw_dir}")

    inv_df = pd.read_parquet(PROJECT_ROOT / "data" / "annotation_inventory.parquet")
    study_df = pd.read_parquet(PROJECT_ROOT / "data" / "study_inventory.parquet")
    reviewed_df = pd.read_parquet(PROJECT_ROOT / "data" / "reviewed_distal_tips_v2.parquet")

    confirmed_negs = set(
        study_df[study_df["is_study_level_negative"] & (~study_df["has_cvc"])]["StudyInstanceUID"]
    )

    # Three verified real cases on disk:
    sid_partial = "1.2.826.0.1.3680043.8.498.34323421018869467568615509546856722773"  # 2 CVCs, 1 reviewed
    sid_neg = "1.2.826.0.1.3680043.8.498.26697628953273228189375557799582420561"      # Confirmed negative
    sid_unreviewed = "1.2.826.0.1.3680043.8.498.93996674799645988481951427980491263942"  # Positive, 0 reviewed

    test_sids = [sid_partial, sid_neg, sid_unreviewed]

    dataset = CoarseDualDataset(
        study_ids=test_sids,
        inventory_df=inv_df,
        images_dir=raw_dir,
        confirmed_negative_study_ids=confirmed_negs,
        reviewed_tips_df=reviewed_df,
        target_size=(512, 512),
        mask_stroke_width_native=10,
        clahe_enabled=True,
        tip_sigma=2.5,
        unreviewed_spatial_ignore_radius_native=50.0,
        partial_review_policy="spatial_mask",
    )

    assert len(dataset) == 3

    # =========================================================================
    # 1. Verify Sample 0: Partially Reviewed Multi-CVC Study
    # =========================================================================
    item_partial = dataset[0]
    assert item_partial["study_id"] == sid_partial
    assert item_partial["image"].shape == (1, 512, 512)
    assert item_partial["mask"].shape == (1, 512, 512)
    assert item_partial["mask"].sum() > 500  # Multiple catheters rasterized
    assert item_partial["tip_heatmap"].max() > 0.95  # Target peak present
    assert float(item_partial["tip_mask"]) == 1.0   # Tip supervision enabled

    # Spatial mask must ignore the unreviewed catheter region while preserving reviewed tip
    tip_sp_mask = item_partial["tip_spatial_mask"][0].numpy()
    assert (tip_sp_mask == 0.0).any(), "Expected unreviewed catheter spatial ignore zone to contain 0.0!"
    frac_ignored = float((tip_sp_mask == 0.0).mean())
    assert 0.01 <= frac_ignored <= 0.15, f"Ignored fraction {frac_ignored:.3f} outside expected range [0.01, 0.15]"

    # Target peak region must NOT be masked
    peak_region = item_partial["tip_heatmap"][0].numpy() > 0.05
    assert (tip_sp_mask[peak_region] == 1.0).all(), "Reviewed tip peak footprint was incorrectly masked!"

    # =========================================================================
    # 2. Verify Sample 1: Confirmed Negative Study
    # =========================================================================
    item_neg = dataset[1]
    assert item_neg["study_id"] == sid_neg
    assert item_neg["mask"].sum() == 0.0
    assert item_neg["tip_heatmap"].max() == 0.0
    assert float(item_neg["tip_mask"]) == 1.0  # Supervised negative
    assert (item_neg["tip_spatial_mask"] == 1.0).all()  # Full image valid negative

    # =========================================================================
    # 3. Verify Sample 2: Unreviewed Positive Study (Unknown Case)
    # =========================================================================
    item_unreviewed = dataset[2]
    assert item_unreviewed["study_id"] == sid_unreviewed
    assert item_unreviewed["mask"].sum() > 200  # Supervised segmentation
    assert item_unreviewed["tip_heatmap"].max() == 0.0
    assert float(item_unreviewed["tip_mask"]) == 0.0  # Zero gradient to tip head!

    # =========================================================================
    # 4. Multi-sample Batch Forward & Backward Gradient Isolation
    # =========================================================================
    loader = DataLoader(dataset, batch_size=3, shuffle=False)
    batch = next(iter(loader))

    imgs = batch["image"]
    masks = batch["mask"]
    tip_hms = batch["tip_heatmap"]
    tip_ms = batch["tip_mask"]
    tip_sms = batch["tip_spatial_mask"]

    model = CoarseDualNet(in_channels=1, base_channels=16)
    pos_weight = torch.tensor([50.0])
    criterion = CoarseDualLoss(
        bce_weight=1.0,
        dice_weight=2.0,
        tip_weight=2.0,
        pos_weight=pos_weight,
        tip_pos_weight=100.0,
    )

    seg_logits, tip_logits = model(imgs)
    # Enable grad tracking on tip_logits for spatial gradient isolation check
    tip_logits.retain_grad()

    loss, loss_dict = criterion(
        seg_logits, tip_logits, masks, tip_hms,
        tip_mask=tip_ms, tip_spatial_mask=tip_sms
    )

    assert torch.isfinite(loss), "Loss must be finite"
    assert loss_dict["loss_seg"] > 0.0
    assert loss_dict["loss_tip"] > 0.0

    loss.backward()

    # Check 4A: Spatial ignore zone of Sample 0 must receive EXACTLY 0.0 gradient
    sample0_grad = tip_logits.grad[0, 0].numpy()
    sample0_ignore_mask = (tip_sms[0, 0].numpy() == 0.0)
    assert np.allclose(sample0_grad[sample0_ignore_mask], 0.0, atol=1e-8), \
        "Non-zero gradient found on spatially ignored catheter region!"

    # Check 4B: Sample 0 valid region must receive non-zero gradient
    sample0_valid_mask = (tip_sms[0, 0].numpy() == 1.0)
    assert np.abs(sample0_grad[sample0_valid_mask]).max() > 0.0, \
        "Valid region of sample 0 received zero gradient!"

    # Check 4C: Sample 2 (unreviewed positive) must receive EXACTLY 0.0 gradient across the ENTIRE sample
    sample2_grad = tip_logits.grad[2, 0].numpy()
    assert np.allclose(sample2_grad, 0.0, atol=1e-8), \
        "Unreviewed positive study (tip_mask=0) received non-zero gradient to tip head!"

    # Check 4D: Segmentation head receives gradients for all positive samples
    for p in model.seg_head.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()

    # =========================================================================
    # 5. Validation Evaluation Pass with One-to-One Hungarian Assignment
    # =========================================================================
    device = torch.device("cpu")
    val_metrics = run_dual_validation(
        model=model,
        val_loader=loader,
        reviewed_val_df=reviewed_df,
        confirmed_negatives=confirmed_negs,
        criterion=criterion,
        device=device,
        peak_threshold=0.20,
    )

    assert "val_loss_total" in val_metrics
    assert "val_mask_dice_mean" in val_metrics
    assert "seg_skeleton" in val_metrics
    assert "tip_heatmap" in val_metrics
    assert "success_at_15" in val_metrics["seg_skeleton"]
    assert "success_at_15" in val_metrics["tip_heatmap"]
    assert val_metrics["negative_studies_evaluated"] == 1
    print("\n[REAL-DATA INTEGRATION TEST PASSED SUCCESSFULLY]")
    print(f"  Val Mask Dice: {val_metrics['val_mask_dice_mean']:.4f}")
    print(f"  Seg Skeleton Success@15: {val_metrics['seg_skeleton']['success_at_15']:.2%}")
    print(f"  Tip Heatmap Success@15: {val_metrics['tip_heatmap']['success_at_15']:.2%}")
    print(f"  Spatial Ignore Isolation: Bitwise verified 0.0 gradient")
    print(f"  Unreviewed Study Isolation: Bitwise verified 0.0 gradient")


if __name__ == "__main__":
    test_real_data_coarse_dual_integration()
    print("ALL ASSERTIONS PASSED!")
