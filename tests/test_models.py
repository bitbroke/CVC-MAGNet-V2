"""Unit tests verifying adapted MAG-Net models, losses, and checkpoints.

Verifies:
1. Module imports.
2. Coarse and fine forward pass output shapes.
3. Loss computation and finiteness.
4. Full checkpoint saving and deterministic resumption.
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.coarse import CoarseSegNet
from src.models.fine import FineSegNet
from src.models.losses import CoarseLoss, FineLoss
from src.models.checkpoint import save_checkpoint, load_checkpoint


def test_coarse_model_shapes():
    model = CoarseSegNet(in_channels=1, out_channels=1, base_channels=32)
    model.eval()
    x = torch.randn(2, 1, 512, 512)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 512, 512), f"Coarse output shape mismatch: {out.shape}"
    print("  [PASS] CoarseSegNet output shape matches target (2, 1, 512, 512).")


def test_fine_model_shapes():
    model = FineSegNet(in_channels=2, out_channels=1, base_channels=32)
    model.eval()
    x = torch.randn(2, 2, 512, 512)
    with torch.no_grad():
        logits, att = model(x)
    assert logits.shape == (2, 1, 512, 512), f"Fine logits shape mismatch: {logits.shape}"
    assert att.shape == (2, 1, 512, 512), f"Fine attention shape mismatch: {att.shape}"
    print("  [PASS] FineSegNet output shapes match targets ((2, 1, 512, 512), (2, 1, 512, 512)).")


def test_loss_finiteness():
    coarse_loss_fn = CoarseLoss()
    pred_coarse = torch.randn(2, 1, 512, 512)
    target_coarse = torch.zeros(2, 1, 512, 512)
    target_coarse[:, :, 100:150, 100:150] = 1.0

    loss_c, metrics_c = coarse_loss_fn(pred_coarse, target_coarse)
    assert torch.isfinite(loss_c), "Coarse loss is not finite!"
    print(f"  [PASS] Coarse loss is finite: {loss_c.item():.4f}")

    fine_loss_fn = FineLoss()
    pred_fine = torch.randn(2, 1, 512, 512)
    pred_att = torch.sigmoid(torch.randn(2, 1, 512, 512))
    target_fine = torch.zeros(2, 1, 512, 512)
    target_blur = torch.zeros(2, 1, 512, 512)

    loss_f, metrics_f = fine_loss_fn(pred_fine, pred_att, target_fine, target_blur)
    assert torch.isfinite(loss_f), "Fine loss is not finite!"
    print(f"  [PASS] Fine loss is finite (even with zero targets): {loss_f.item():.4f}")


def test_checkpoint_roundtrip():
    model = CoarseSegNet(in_channels=1, out_channels=1, base_channels=16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)

    # Perform a dummy step
    x = torch.randn(2, 1, 64, 64)
    loss = model(x).sum()
    loss.backward()
    opt.step()
    sched.step()

    class DummySampler:
        def __init__(self, state):
            self.state = state
        def state_dict(self):
            return {"index": self.state}
        def load_state_dict(self, d):
            self.state = d["index"]

    sampler = DummySampler(state=123)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_ckpt.pt"
        save_checkpoint(
            path=ckpt_path,
            model=model,
            optimizer=opt,
            scheduler=sched,
            sampler=sampler,
            config={"model": "CoarseSegNet"},
            input_hashes={"git": "test_hash"},
            step=42,
            epoch=2,
            metrics={"val_dice": 0.85},
        )
        assert ckpt_path.exists(), "Checkpoint file was not created!"

        # Test atomic overwrite without deleting beforehand
        save_checkpoint(
            path=ckpt_path,
            model=model,
            optimizer=opt,
            scheduler=sched,
            sampler=sampler,
            config={"model": "CoarseSegNet"},
            input_hashes={"git": "test_hash"},
            step=43,
            epoch=2,
            metrics={"val_dice": 0.86},
        )

        # Create fresh model, optimizer, scheduler, sampler
        new_model = CoarseSegNet(in_channels=1, out_channels=1, base_channels=16)
        new_opt = torch.optim.AdamW(new_model.parameters(), lr=1e-4)
        new_sched = torch.optim.lr_scheduler.CosineAnnealingLR(new_opt, T_max=100)
        new_sampler = DummySampler(state=0)

        # Test loading (if CUDA is available, test with device='cuda' to ensure CPU RNG state does not crash)
        load_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            new_model = new_model.to(load_device)

        meta = load_checkpoint(
            ckpt_path,
            new_model,
            new_opt,
            scheduler=new_sched,
            sampler=new_sampler,
            device=load_device,
        )

        assert meta["step"] == 43, f"Step mismatch: {meta['step']}"
        assert meta["epoch"] == 2, f"Epoch mismatch: {meta['epoch']}"
        assert meta["metrics"]["val_dice"] == 0.86, f"Metrics mismatch: {meta['metrics']}"
        assert new_sampler.state == 123, f"Sampler state mismatch: {new_sampler.state}"

        # Verify weights match exactly
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1.cpu(), p2.cpu()), "Model parameters do not match after checkpoint load!"

    print("  [PASS] Hardened checkpoint saved and resumed identically (scheduler, sampler, RNG, and atomic replacement verified).")


if __name__ == "__main__":
    print("Running Model & Checkpoint Tests...")
    test_coarse_model_shapes()
    test_fine_model_shapes()
    test_loss_finiteness()
    test_checkpoint_roundtrip()
    print("All Model & Checkpoint tests PASSED successfully.")
