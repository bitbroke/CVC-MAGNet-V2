"""Master Sequential Runner for Seed-43 Confirmation Experiment.

Executes sequentially with strict single-process GPU discipline:
1. Arm A (Control, Seed 43): 1,000 updates, 768px, uniform negative sampling.
2. Arm B (Experiment, Seed 43): 1,000 updates, 768px, 50% uniform + 50% selective top-99 hard negatives.
3. Native-resolution dual-seed comparative evaluation across 193 studies with bootstrap CIs.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_command(cmd_list, desc):
    print("\n" + "=" * 70)
    print(f"STARTING: {desc}")
    print("Command:", " ".join(cmd_list))
    print("=" * 70, flush=True)

    t0 = time.time()
    result = subprocess.run(cmd_list, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n[ERROR] {desc} failed with return code {result.returncode} after {elapsed:.1f}s!", flush=True)
        sys.exit(result.returncode)

    print(f"\n[SUCCESS] {desc} completed in {elapsed:.1f}s ({elapsed/60:.1f} mins).", flush=True)
    return elapsed


def main():
    print("=" * 70)
    print("Seed-43 Confirmation Experiment (Matched Sequential Pilots)")
    print("=" * 70)

    python_exe = sys.executable
    initial_weights = str(PROJECT_ROOT / "runs" / "s3_experiment_hard_neg" / "checkpoints" / "final_segmentation_v3.pt")
    hard_manifest = str(PROJECT_ROOT / "data" / "selective_hard_negatives_manifest.json")

    if not os.path.exists(initial_weights):
        raise FileNotFoundError(f"Frozen reference weights not found: {initial_weights}")
    if not os.path.exists(hard_manifest):
        raise FileNotFoundError(f"Selective hard negative manifest not found: {hard_manifest}")

    # 1. ARM A: CONTROL (Seed 43, Uniform Negatives)
    arm_a_final = PROJECT_ROOT / "runs" / "s4_control_seed43" / "checkpoints" / "final_segmentation_v3.pt"
    cmd_arm_a = [
        python_exe,
        str(PROJECT_ROOT / "scripts" / "train_segmentation_v3.py"),
        "--output_dir", str(PROJECT_ROOT / "runs" / "s4_control_seed43"),
        "--target_updates", "1000",
        "--effective_batch_size", "4",
        "--micro_batch_size", "1",
        "--img_size", "768",
        "--lr", "1e-4",
        "--weight_decay", "1e-4",
        "--seed", "43",
        "--aug_seed", "43",
        "--load_initial_weights", initial_weights,
    ]

    if arm_a_final.exists():
        print(f"\n[CACHE HIT] Arm A Control (Seed 43) checkpoint already exists at {arm_a_final}. Skipping training.")
        with open(PROJECT_ROOT / "runs" / "s4_control_seed43" / "train_summary.json") as f:
            time_arm_a = json.load(f).get("elapsed_seconds", 800.0)
    else:
        time_arm_a = run_command(cmd_arm_a, "Arm A: Control (Seed 43, 100% Uniform Negatives, 1000 updates @ 768px)")

    # 2. ARM B: EXPERIMENT (Seed 43, 50% Uniform + 50% Selective Top-99 Hard Negatives)
    arm_b_final = PROJECT_ROOT / "runs" / "s4_experiment_seed43" / "checkpoints" / "final_segmentation_v3.pt"
    cmd_arm_b = [
        python_exe,
        str(PROJECT_ROOT / "scripts" / "train_segmentation_v3.py"),
        "--output_dir", str(PROJECT_ROOT / "runs" / "s4_experiment_seed43"),
        "--target_updates", "1000",
        "--effective_batch_size", "4",
        "--micro_batch_size", "1",
        "--img_size", "768",
        "--lr", "1e-4",
        "--weight_decay", "1e-4",
        "--seed", "43",
        "--aug_seed", "43",
        "--load_initial_weights", initial_weights,
        "--hard_negative_manifest", hard_manifest,
    ]

    if arm_b_final.exists():
        print(f"\n[CACHE HIT] Arm B Experiment (Seed 43) checkpoint already exists at {arm_b_final}. Skipping training.")
        with open(PROJECT_ROOT / "runs" / "s4_experiment_seed43" / "train_summary.json") as f:
            time_arm_b = json.load(f).get("elapsed_seconds", 800.0)
    else:
        time_arm_b = run_command(cmd_arm_b, "Arm B: Experiment (Seed 43, 50% Uniform + 50% Top-99 Hard Negatives, 1000 updates @ 768px)")

    # 3. NATIVE-RESOLUTION DUAL-SEED COMPARATIVE EVALUATION
    cmd_eval = [
        python_exe,
        str(PROJECT_ROOT / "scripts" / "evaluate_seed_confirmation.py"),
    ]

    time_eval = run_command(cmd_eval, "Dual-Seed Benchmark Evaluation & Bootstrap CIs")

    total_time = time_arm_a + time_arm_b + time_eval
    print("\n" + "=" * 70)
    print(f"SEED-43 CONFIRMATION PILOTS AND EVALUATIONS COMPLETED!")
    print(f"  Arm A Control Time:    {time_arm_a/60:.1f} mins")
    print(f"  Arm B Experiment Time: {time_arm_b/60:.1f} mins")
    print(f"  Evaluation Time:       {time_eval/60:.1f} mins")
    print(f"  Total Elapsed Time:    {total_time/60:.1f} mins")
    print("=" * 70)


if __name__ == "__main__":
    main()
