"""Unit tests verifying:
1. That seg_only evaluation NEVER uses tip_logits (outputs are 100% invariant even when tip_logits is corrupted with NaNs/noise).
2. That Arm A (seg_only) and Arm B (dual_head) produce 100% bitwise identical segmentation outputs before training.
3. That dual_head evaluation provides both prediction paths (seg_skeleton and tip_heatmap) on the same target registry.
4. That partially reviewed multi-CVC images are excluded from tip supervision under the default pilot policy.
"""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import CoarseDualDataset
from src.data.transforms import LetterboxTransform
from src.models.coarse_dual import CoarseDualNet
from src.models.losses import CoarseDualLoss

spec = importlib.util.spec_from_file_location("train_dual_mod", PROJECT_ROOT / "scripts" / "05_train_coarse_dual.py")
train_dual_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_dual_mod)
run_dual_validation = train_dual_mod.run_dual_validation
load_shared_backbone = train_dual_mod.load_shared_backbone


def test_pre_training_identical_segmentation_outputs():
    """Verify both arms produce 100% bitwise identical segmentation outputs before training."""
    ckpt_path = PROJECT_ROOT / "runs" / "pilot_coarse_repaired" / "checkpoints" / "best_coarse.pt"
    assert ckpt_path.exists(), f"Reference checkpoint missing at {ckpt_path}"

    model_a = CoarseDualNet(in_channels=1, base_channels=32)
    model_b = CoarseDualNet(in_channels=1, base_channels=32)

    load_shared_backbone(model_a, ckpt_path)
    load_shared_backbone(model_b, ckpt_path)

    model_a.eval()
    model_b.eval()

    # Pass identical random input tensor
    torch.manual_seed(1234)
    x = torch.randn(2, 1, 512, 512)

    with torch.no_grad():
        seg_logits_a, _ = model_a(x)
        seg_logits_b, _ = model_b(x)

    # 1. Assert bitwise identical raw logits
    assert torch.equal(seg_logits_a, seg_logits_b), "Raw segmentation logits differ between Arm A and Arm B before training!"

    # 2. Assert bitwise identical probabilities
    probs_a = torch.sigmoid(seg_logits_a)
    probs_b = torch.sigmoid(seg_logits_b)
    assert torch.equal(probs_a, probs_b), "Segmentation probabilities differ between Arm A and Arm B before training!"

    # 3. Assert bitwise identical binary masks
    mask_a = (probs_a > 0.30).float()
    mask_b = (probs_b > 0.30).float()
    assert torch.equal(mask_a, mask_b), "Predicted binary segmentation masks differ between Arm A and Arm B before training!"
    print("\n[VERIFIED] Both comparison arms produce 100% bitwise identical segmentation outputs before training.")


class TamperedTipModel(nn.Module):
    """Wrapper that wraps CoarseDualNet but injects NaNs/extreme noise into tip_logits."""

    def __init__(self, base_model: CoarseDualNet, tamper_mode: str = "nan"):
        super().__init__()
        self.base_model = base_model
        self.tamper_mode = tamper_mode

    def forward(self, x: torch.Tensor):
        seg_logits, tip_logits = self.base_model(x)
        if self.tamper_mode == "nan":
            corrupted_tip = torch.full_like(tip_logits, float("nan"))
        elif self.tamper_mode == "noise":
            corrupted_tip = torch.randn_like(tip_logits) * 1e6
        else:
            corrupted_tip = tip_logits
        return seg_logits, corrupted_tip


class SyntheticValDataset(Dataset):
    """Small synthetic dataset to test validation evaluation logic deterministically."""

    def __init__(self):
        tf = LetterboxTransform((512, 512))
        img = np.zeros((1000, 1000), dtype=np.uint8)
        _, meta = tf.apply_image(img)
        self.meta_dict = meta.to_dict()

        # Create synthetic binary mask with a clear vertical catheter
        mask = np.zeros((512, 512), dtype=np.float32)
        mask[100:200, 250:254] = 1.0  # Endpoint at (252, 199) and (252, 100)
        self.mask = torch.from_numpy(mask).unsqueeze(0)
        self.image = torch.zeros(1, 512, 512)

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        sid = f"SYNTH_STUDY_{idx}"
        return {
            "image": self.image,
            "mask": self.mask,
            "tip_heatmap": torch.zeros(1, 512, 512),
            "tip_mask": torch.tensor(1.0),
            "tip_spatial_mask": torch.ones(1, 512, 512),
            "study_id": sid,
            "orig_size": (1000, 1000),
            "meta": self.meta_dict,
        }


def test_seg_only_evaluation_never_uses_tip_logits():
    """Verify that seg_only evaluation NEVER touches tip_logits and is invariant to corrupted tip_logits."""
    val_dataset = SyntheticValDataset()
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    # Reviewed validation targets located near the catheter endpoints
    reviewed_val_df = pd.DataFrame([
        {
            "study_id": "SYNTH_STUDY_0",
            "catheter_instance_id": 1,
            "reviewed_tip_x": 492.0,  # Near native mapped x
            "reviewed_tip_y": 390.0,  # Near native mapped y
        },
        {
            "study_id": "SYNTH_STUDY_1",
            "catheter_instance_id": 1,
            "reviewed_tip_x": 492.0,
            "reviewed_tip_y": 390.0,
        },
    ])

    device = torch.device("cpu")
    base_model = CoarseDualNet(in_channels=1, base_channels=8).to(device)
    criterion = CoarseDualLoss(tip_weight=0.0)  # seg_only loss has tip_weight=0

    # 1. Run baseline evaluation with untampered model
    metrics_clean = run_dual_validation(
        model=base_model,
        val_loader=val_loader,
        reviewed_val_df=reviewed_val_df,
        confirmed_negatives=set(),
        criterion=criterion,
        device=device,
        mode="seg_only",
    )

    assert metrics_clean["mode"] == "seg_only"
    assert metrics_clean["tip_heatmap"] is None, "seg_only evaluation must not report tip_heatmap results!"
    assert metrics_clean["seg_skeleton"] is not None

    # 2. Run evaluation with NaNs injected into tip_logits
    model_nan = TamperedTipModel(base_model, tamper_mode="nan")
    metrics_nan = run_dual_validation(
        model=model_nan,
        val_loader=val_loader,
        reviewed_val_df=reviewed_val_df,
        confirmed_negatives=set(),
        criterion=criterion,
        device=device,
        mode="seg_only",
    )

    # 3. Run evaluation with extreme random noise injected into tip_logits
    model_noise = TamperedTipModel(base_model, tamper_mode="noise")
    metrics_noise = run_dual_validation(
        model=model_noise,
        val_loader=val_loader,
        reviewed_val_df=reviewed_val_df,
        confirmed_negatives=set(),
        criterion=criterion,
        device=device,
        mode="seg_only",
    )

    # 4. Assert that seg_only evaluation is 100% invariant to tip_logits corruption
    for key in ["matched_15_count", "success_at_15", "median_error_15"]:
        val_clean = metrics_clean["seg_skeleton"][key]
        val_nan = metrics_nan["seg_skeleton"][key]
        val_noise = metrics_noise["seg_skeleton"][key]
        assert val_clean == val_nan == val_noise, f"Metric {key} varied under tip_logits corruption!"

    assert metrics_clean["val_mask_dice_mean"] == metrics_nan["val_mask_dice_mean"] == metrics_noise["val_mask_dice_mean"]
    print("[VERIFIED] seg_only evaluation is 100% invariant to tip_logits: tip head is never evaluated or decoded.")


def test_dual_head_evaluation_provides_both_prediction_paths():
    """Verify that dual_head evaluation provides both seg_skeleton and tip_heatmap on the same targets."""
    val_dataset = SyntheticValDataset()
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    reviewed_val_df = pd.DataFrame([
        {
            "study_id": "SYNTH_STUDY_0",
            "catheter_instance_id": 1,
            "reviewed_tip_x": 492.0,
            "reviewed_tip_y": 390.0,
        },
    ])

    device = torch.device("cpu")
    model = CoarseDualNet(in_channels=1, base_channels=8).to(device)
    criterion = CoarseDualLoss(tip_weight=2.0)

    metrics = run_dual_validation(
        model=model,
        val_loader=val_loader,
        reviewed_val_df=reviewed_val_df,
        confirmed_negatives=set(),
        criterion=criterion,
        device=device,
        mode="dual_head",
    )

    assert metrics["mode"] == "dual_head"
    assert metrics["seg_skeleton"] is not None, "Missing Path 2: seg_skeleton"
    assert metrics["tip_heatmap"] is not None, "Missing Path 3: tip_heatmap"
    assert "success_at_15" in metrics["seg_skeleton"]
    assert "success_at_15" in metrics["tip_heatmap"]
    print("[VERIFIED] dual_head evaluation simultaneously outputs both seg_skeleton and tip_heatmap.")


def test_partial_review_exclusion_policy_in_dataset():
    """Verify that partially reviewed images receive tip_mask=0.0 under the default pilot policy."""
    inv_df = pd.read_parquet(PROJECT_ROOT / "data" / "annotation_inventory.parquet")
    study_df = pd.read_parquet(PROJECT_ROOT / "data" / "study_inventory.parquet")
    reviewed_df = pd.read_parquet(PROJECT_ROOT / "data" / "reviewed_distal_tips_v2.parquet")

    confirmed_negs = set(
        study_df[study_df["is_study_level_negative"] & (~study_df["has_cvc"])]["StudyInstanceUID"]
    )

    with open(PROJECT_ROOT / "configs" / "pilot.yaml") as f:
        cfg = yaml.safe_load(f)
    raw_dir = Path(cfg["data"]["raw_dir"]) / "train"

    # Known partially reviewed multi-CVC study: 2 CVCs total, 1 reviewed in v2
    sid_partial = "1.2.826.0.1.3680043.8.498.34323421018869467568615509546856722773"

    dataset_exclude = CoarseDualDataset(
        study_ids=[sid_partial],
        inventory_df=inv_df,
        images_dir=raw_dir,
        confirmed_negative_study_ids=confirmed_negs,
        reviewed_tips_df=reviewed_df,
        partial_review_policy="exclude",  # Default defensible policy
    )

    item = dataset_exclude[0]
    # Under 'exclude' policy: partially reviewed images must receive tip_mask = 0.0
    assert float(item["tip_mask"]) == 0.0, "Partially reviewed image must have tip_mask = 0.0 under exclude policy!"
    assert float(item["tip_heatmap"].max()) == 0.0, "Partially reviewed image must have zero heatmap under exclude policy!"
    assert float(item["mask"].sum()) > 500, "Segmentation mask must remain fully supervised!"

    print("[VERIFIED] Default partial review policy excludes partially reviewed images from tip supervision.")


if __name__ == "__main__":
    test_pre_training_identical_segmentation_outputs()
    test_seg_only_evaluation_never_uses_tip_logits()
    test_dual_head_evaluation_provides_both_prediction_paths()
    test_partial_review_exclusion_policy_in_dataset()
    print("\nALL FOUR CRITICAL TESTS PASSED SUCCESSFULLY!")
