"""Unit tests for CVC-Segmentation-V3 Architecture."""

import pytest
import torch
import torch.nn as nn

from src.models.segmentation_v3 import CVCSegmentationV3Net, DecoderBlock


def test_decoder_block():
    block = DecoderBlock(in_channels=64, skip_channels=32, out_channels=32, num_groups=8)
    x = torch.randn(2, 64, 16, 16)
    skip = torch.randn(2, 32, 32, 32)
    out = block(x, skip)
    assert out.shape == (2, 32, 32, 32)


def test_cvc_segmentation_v3_shapes():
    model = CVCSegmentationV3Net(pretrained=False, freeze_encoder_bn=True)
    model.eval()

    # Test single-channel input
    x1 = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        out1 = model(x1)
    assert "seg_logits" in out1
    assert "seg_probs" in out1
    assert out1["seg_logits"].shape == (2, 1, 256, 256)
    assert out1["seg_probs"].shape == (2, 1, 256, 256)
    assert (out1["seg_probs"] >= 0.0).all() and (out1["seg_probs"] <= 1.0).all()

    # Test 3-channel input
    x3 = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out3 = model(x3)
    assert out3["seg_logits"].shape == (1, 1, 512, 512)


def test_encoder_bn_freeze():
    model = CVCSegmentationV3Net(pretrained=False, freeze_encoder_bn=True)
    model.train()

    # All BatchNorm layers in encoder must be in eval mode
    for m in model.encoder.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            assert not m.training, "Encoder BatchNorm layer was not frozen in training mode!"

    # Decoder layers must be in training mode
    assert model.bottleneck.training
    assert model.block16.training


def test_cvc_segmentation_v3_backward():
    model = CVCSegmentationV3Net(pretrained=False, freeze_encoder_bn=True)
    model.train()

    x = torch.randn(2, 1, 128, 128)  # small size for fast test
    target = torch.randint(0, 2, (2, 1, 128, 128)).float()

    out = model(x)
    loss = nn.BCEWithLogitsLoss()(out["seg_logits"], target)
    loss.backward()

    # Verify gradients exist and are finite
    has_encoder_grad = False
    has_decoder_grad = False
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"
            if "encoder" in name:
                has_encoder_grad = True
            elif "block" in name or "bottleneck" in name:
                has_decoder_grad = True

    assert has_encoder_grad
    assert has_decoder_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
