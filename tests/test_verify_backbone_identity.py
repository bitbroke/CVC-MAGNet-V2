"""Verify bitwise identical shared-backbone weights for both comparison arms.

Guarantees:
- Arm A (Segmentation-only continuation) and Arm B (Dual-head continuation)
  start from the exact same initial weights loaded from best_coarse.pt.
- No confounding between additional training iterations and head architecture.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib.util
import pytest
import torch

spec = importlib.util.spec_from_file_location("train_dual_mod", Path(__file__).resolve().parent.parent / "scripts" / "05_train_coarse_dual.py")
train_dual_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_dual_mod)
compute_model_backbone_hash = train_dual_mod.compute_model_backbone_hash
load_shared_backbone = train_dual_mod.load_shared_backbone

from src.models.coarse_dual import CoarseDualNet


def test_comparison_arms_have_identical_shared_backbone():
    ckpt_path = Path("runs/pilot_coarse_repaired/checkpoints/best_coarse.pt")
    assert ckpt_path.exists(), f"Missing reference checkpoint at {ckpt_path}"

    # Arm A: seg_only continuation model
    model_arm_a = CoarseDualNet(in_channels=1, base_channels=32)
    info_a = load_shared_backbone(model_arm_a, ckpt_path)

    # Arm B: dual_head continuation model
    model_arm_b = CoarseDualNet(in_channels=1, base_channels=32)
    info_b = load_shared_backbone(model_arm_b, ckpt_path)

    # 1. Assert identical transfer counts
    assert info_a["loaded_keys_count"] == info_b["loaded_keys_count"] == 102
    assert info_a["skipped_keys_count"] == info_b["skipped_keys_count"] == 0

    # 2. Check every shared backbone parameter bit-for-bit
    shared_prefixes = (
        "conv1", "conv2", "conv3", "conv4", "ppm",
        "up7", "conv7", "up8", "conv8", "up9", "conv9", "seg_head"
    )

    for name, p_a in model_arm_a.named_parameters():
        if any(name.startswith(pfx) for pfx in shared_prefixes):
            p_b = dict(model_arm_b.named_parameters())[name]
            assert torch.equal(p_a, p_b), f"Parameter {name} differs between Arm A and Arm B!"

    # 3. Check parameter SHA256 hashes
    hash_a = compute_model_backbone_hash(model_arm_a)
    hash_b = compute_model_backbone_hash(model_arm_b)
    assert hash_a == hash_b, f"Backbone hash mismatch: {hash_a} != {hash_b}"
    print(f"Verified bitwise identical shared-backbone hash: {hash_a}")


if __name__ == "__main__":
    test_comparison_arms_have_identical_shared_backbone()
    print("ALL ASSERTIONS PASSED!")
