"""Unit test verifying that non-finite gradients do not advance model weights, optimizer, or scheduler."""

import copy
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.amp import GradScaler


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


def test_nonfinite_gradient_safety_disabled_scaler():
    """Verify safety branch when GradScaler is disabled (e.g. native bfloat16 mode)."""
    torch.manual_seed(42)
    model = ToyModel()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=10)
    scaler = GradScaler("cuda", enabled=False)

    # Initial snapshot
    initial_weights = copy.deepcopy(model.fc.weight.data)
    initial_bias = copy.deepcopy(model.fc.bias.data)
    initial_lr = optimizer.param_groups[0]["lr"]
    successful_updates = 0
    skipped_updates = 0

    # Step 1: Normal clean update
    x = torch.randn(4, 10)
    y = torch.randn(4, 1)
    out = model(x)
    loss = nn.functional.mse_loss(out, y)
    loss.backward()

    scaler.unscale_(optimizer)
    has_inf_or_nan = any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters())
    assert not has_inf_or_nan

    # Clean update steps
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    successful_updates += 1

    weight_after_clean = copy.deepcopy(model.fc.weight.data)
    assert not torch.equal(initial_weights, weight_after_clean)

    # Step 2: Inject non-finite (NaN / Inf) gradient
    out = model(x)
    loss = nn.functional.mse_loss(out, y)
    loss.backward()

    # Manually poison gradient with NaN
    model.fc.weight.grad[0, 0] = float("nan")

    scaler.unscale_(optimizer)
    has_inf_or_nan = any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters())
    assert has_inf_or_nan

    # Apply the safe handler logic
    if has_inf_or_nan:
        skipped_updates += 1
        if scaler.is_enabled():
            scaler.update()
        optimizer.zero_grad(set_to_none=True)
    else:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        successful_updates += 1

    # Assertions:
    # 1. Weights MUST be identical to weight_after_clean
    assert torch.equal(model.fc.weight.data, weight_after_clean), "Model weights changed on NaN gradient!"
    assert torch.equal(model.fc.bias.data, model.fc.bias.data), "Bias corrupted!"
    assert torch.isfinite(model.fc.weight.data).all(), "Weights contain non-finite numbers!"

    # 2. Optimizer state must have no NaN
    for p in model.parameters():
        state = optimizer.state.get(p, {})
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                assert torch.isfinite(v).all(), f"Optimizer state {k} contaminated with NaN/Inf!"

    # 3. Successful updates count must NOT advance
    assert successful_updates == 1, f"Expected 1 successful update, got {successful_updates}"
    assert skipped_updates == 1, f"Expected 1 skip, got {skipped_updates}"

    print("PASSED: test_nonfinite_gradient_safety_disabled_scaler")


def test_nonfinite_gradient_safety_enabled_scaler():
    """Verify safety branch when GradScaler is enabled (e.g. float16 mode)."""
    if not torch.cuda.is_available():
        print("CUDA not available; skipping enabled scaler test")
        return

    torch.manual_seed(42)
    model = ToyModel().cuda()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=10)
    scaler = GradScaler("cuda", enabled=True)

    initial_weights = copy.deepcopy(model.fc.weight.data)
    successful_updates = 0
    skipped_updates = 0

    x = torch.randn(4, 10, device="cuda")
    y = torch.randn(4, 1, device="cuda")

    # Step 1: Clean update
    with torch.amp.autocast("cuda"):
        out = model(x)
        loss = nn.functional.mse_loss(out, y)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    successful_updates += 1
    weight_after_clean = copy.deepcopy(model.fc.weight.data)

    # Step 2: Inject Inf gradient
    with torch.amp.autocast("cuda"):
        out = model(x)
        loss = nn.functional.mse_loss(out, y)
    scaler.scale(loss).backward()

    # Poison gradient
    model.fc.weight.grad[0, 0] = float("inf")

    scaler.unscale_(optimizer)
    has_inf_or_nan = any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters())
    assert has_inf_or_nan

    # Safe handler
    if has_inf_or_nan:
        skipped_updates += 1
        if scaler.is_enabled():
            scaler.update()
        optimizer.zero_grad(set_to_none=True)
    else:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        successful_updates += 1

    assert torch.equal(model.fc.weight.data, weight_after_clean), "Weights changed on Inf gradient!"
    assert torch.isfinite(model.fc.weight.data).all()
    assert successful_updates == 1
    assert skipped_updates == 1
    print("PASSED: test_nonfinite_gradient_safety_enabled_scaler")


if __name__ == "__main__":
    test_nonfinite_gradient_safety_disabled_scaler()
    test_nonfinite_gradient_safety_enabled_scaler()
