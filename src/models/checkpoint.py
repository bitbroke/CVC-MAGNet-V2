"""Full-state checkpoint saving and resumption utility.

Captures all deterministic state:
- Model architecture configuration.
- Input dataset / split hashes.
- Model weights and optimizer state.
- Learning rate scheduler state.
- Data sampler state.
- Gradient scaler state (torch.amp).
- Complete RNG states (Python, NumPy, PyTorch CPU, PyTorch CUDA).

Hardening guarantees:
- Atomic replacement (os.replace) eliminates checkpoint corruption window.
- CPU RNG state is strictly guaranteed to be a CPU ByteTensor, preventing CUDA mapping errors.
- Scheduler and sampler states are preserved for bitwise-exact resume.
"""
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch


def get_all_rng_states() -> Dict[str, Any]:
    """Capture RNG states across Python, NumPy, and PyTorch (CPU and CUDA)."""
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().cpu(),  # Strictly store on CPU
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = [s.cpu() for s in torch.cuda.get_rng_state_all()]
    return states


def set_all_rng_states(states: Dict[str, Any]) -> None:
    """Restore all RNG states for exact training continuation."""
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    if "torch" in states:
        torch_state = states["torch"]
        # Guarantee CPU ByteTensor even if checkpoint was loaded with map_location='cuda'
        if isinstance(torch_state, torch.Tensor):
            torch_state = torch_state.cpu()
        torch.set_rng_state(torch_state)
    if "torch_cuda" in states and torch.cuda.is_available():
        try:
            cuda_states = states["torch_cuda"]
            # Convert any CUDA tensors back to CPU ByteTensors if needed by set_rng_state_all
            clean_cuda_states = [s.cpu() if isinstance(s, torch.Tensor) else s for s in cuda_states]
            torch.cuda.set_rng_state_all(clean_cuda_states)
        except Exception as e:
            print(f"Warning: Could not fully restore CUDA RNG state: {e}")


def save_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    sampler: Optional[Any] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    config: Optional[Dict[str, Any]] = None,
    input_hashes: Optional[Dict[str, str]] = None,
    step: int = 0,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    is_best: bool = False,
) -> None:
    """Save an atomic, complete training checkpoint with zero failure window."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")

    # Capture scheduler state
    sched_state = None
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        sched_state = scheduler.state_dict()

    # Capture sampler state
    samp_state = None
    if sampler is not None:
        if hasattr(sampler, "state_dict"):
            samp_state = sampler.state_dict()
        elif hasattr(sampler, "state"):
            samp_state = sampler.state

    checkpoint_data = {
        "step": step,
        "epoch": epoch,
        "config": config or {},
        "input_hashes": input_hashes or {},
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": sched_state,
        "sampler_state": samp_state,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "rng_states": get_all_rng_states(),
        "metrics": metrics or {},
        "is_best": is_best,
    }

    # Save to temporary path first
    torch.save(checkpoint_data, temp_path)
    # Atomic replace: overwrites path in a single atomic filesystem operation without deletion window
    os.replace(temp_path, path)


def load_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    sampler: Optional[Any] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    restore_rng: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Load a checkpoint and restore model, optimizer, scheduler, sampler, scaler, and RNG states."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    try:
        checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device or "cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        if hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if sampler is not None and checkpoint.get("sampler_state") is not None:
        if hasattr(sampler, "load_state_dict"):
            sampler.load_state_dict(checkpoint["sampler_state"])
        elif hasattr(sampler, "state"):
            sampler.state = checkpoint["sampler_state"]

    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if restore_rng and "rng_states" in checkpoint:
        set_all_rng_states(checkpoint["rng_states"])

    return {
        "step": checkpoint.get("step", 0),
        "epoch": checkpoint.get("epoch", 0),
        "config": checkpoint.get("config", {}),
        "input_hashes": checkpoint.get("input_hashes", {}),
        "metrics": checkpoint.get("metrics", {}),
        "scheduler_state_dict": checkpoint.get("scheduler_state_dict"),
        "sampler_state": checkpoint.get("sampler_state"),
    }
