"""Confirmation of Promising Candidate (HRNet-W18) vs Baseline (EfficientNet-B0) on Seed 43.

Executes:
1. EfficientNet_B0 (1,000 updates, seed 43)
2. HRNet_W18 (1,000 updates, seed 43)
3. Evaluates both models and computes paired consistency metrics (Seed 42 vs Seed 43).
"""

import gc
import json
import os
import sys
import time
from pathlib import Path
import torch

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXP_ROOT = PROJECT_ROOT / "experiments" / "architecture_comparison_v2"
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from experiments.architecture_comparison_v2.scripts.train_runner import (
    run_training_pilot,
    NativeValidator,
    IMAGES_DIR,
)
from experiments.architecture_comparison_v2.src.models.model_registry import build_model


def run_seed43():
    seed = 43
    models_to_run = ["EfficientNet_B0", "HRNet_W18"]

    print("\n" + "=" * 75)
    print(f"=== COMMENCING SEED 43 CONFIRMATION PILOT ===")
    print("=" * 75 + "\n")

    for model_name in models_to_run:
        run_dir = EXP_ROOT / "runs" / f"seed_{seed}" / model_name.lower()
        final_ckpt = run_dir / "checkpoints" / "final_model.pt"

        if final_ckpt.exists():
            print(f"[{model_name}] Found existing completed checkpoint at {final_ckpt}. Skipping training.")
            continue

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        print(f"\n>>> Starting {model_name} (Seed {seed}) <<<")
        run_training_pilot(
            model_name=model_name,
            target_updates=1000,
            eval_interval=250,
            seed=seed,
            output_dir=str(run_dir),
            overwrite=False,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Evaluation
    print("\n=== Evaluating Seed 43 Checkpoints ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validator = NativeValidator(target_size=768, device=device)

    results_43 = {}
    for model_name in models_to_run:
        run_dir = EXP_ROOT / "runs" / f"seed_{seed}" / model_name.lower()
        best_ckpt = run_dir / "checkpoints" / "best_model.pt"
        if not best_ckpt.exists():
            best_ckpt = run_dir / "checkpoints" / "final_model.pt"

        m = build_model(model_name, pretrained=False).to(device)
        ck = torch.load(best_ckpt, map_location=device, weights_only=False)
        sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
        m.load_state_dict(sd)
        m.eval()

        res = validator.evaluate_model(m, threshold=0.50)
        s = res["summary"]
        s["eval_checkpoint"] = best_ckpt.name
        s["best_step"] = ck.get("best_step", ck.get("update", 1000))
        s["peak_vram_mb"] = ck.get("peak_vram_mb", 0.0)
        results_43[model_name] = s

    out_file = EXP_ROOT / "reports" / "seed43_confirmation_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_43, f, indent=2)

    print(f"\nSaved Seed 43 results to {out_file}.")


if __name__ == "__main__":
    run_seed43()
