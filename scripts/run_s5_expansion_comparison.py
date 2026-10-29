"""Master Runner for S5 Controlled 2,500-Update Expansion Comparison.

Compares:
Arm A: Duration Control (400 pos, 396 neg = 796 studies), 2,500 updates @ 768px, 50% Top-99 Hard Negatives.
Arm B: Expanded Cohort (800 pos, 396 neg = 1,196 studies), 2,500 updates @ 768px, 50% Top-99 Hard Negatives.

Both arms:
- Initialized from identical frozen S3 checkpoint (runs/s3_experiment_hard_neg/checkpoints/final_segmentation_v3.pt).
- Same-batch in-place AMP retry logic on gradient overflow.
- Strict single-process sequential GPU execution on 4GB RTX 3050.
- Evaluated on native-resolution grid on dev_pilot_val (193 studies).
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
PYTHON_EXE = sys.executable


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_command_stream(cmd_args, desc: str):
    print("\n" + "=" * 70)
    print(f"LAUNCHING: {desc}")
    print(f"COMMAND: {' '.join(str(x) for x in cmd_args)}")
    print("=" * 70, flush=True)

    t0 = time.time()
    proc = subprocess.Popen(
        cmd_args,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in iter(proc.stdout.readline, ""):
        print(line, end="", flush=True)

    proc.stdout.close()
    rc = proc.wait()
    elapsed = time.time() - t0

    if rc != 0:
        raise RuntimeError(f"Step '{desc}' failed with exit code {rc} after {elapsed/60:.1f} mins.")

    print(f"\n[COMPLETED]: {desc} in {elapsed/60:.1f} mins (exit code 0).\n", flush=True)
    return elapsed


def main():
    print("=" * 70)
    print("STARTING S5 2,500-UPDATE COHORT EXPANSION COMPARISON")
    print("=" * 70)

    # 1. Preflight File & Hash Verifications
    s3_ckpt = PROJECT_ROOT / "runs" / "s3_experiment_hard_neg" / "checkpoints" / "final_segmentation_v3.pt"
    if not s3_ckpt.exists():
        raise FileNotFoundError(f"Missing S3 baseline checkpoint: {s3_ckpt}")
    s3_hash = compute_sha256(str(s3_ckpt))
    print(f"S3 Baseline Checkpoint verified: {s3_ckpt} (SHA256: {s3_hash[:16]}...)")

    hn_manifest = PROJECT_ROOT / "data" / "selective_hard_negatives_manifest.json"
    if not hn_manifest.exists():
        raise FileNotFoundError(f"Missing hard negative manifest: {hn_manifest}")
    print(f"Selective Hard Negatives Manifest verified: {hn_manifest}")

    exp_manifest = PROJECT_ROOT / "data" / "expanded_training_cohort_manifest.json"
    if not exp_manifest.exists():
        raise FileNotFoundError(f"Missing expanded cohort manifest: {exp_manifest}")
    print(f"Expanded Cohort Manifest verified: {exp_manifest}")

    # 2. Arm A: Duration Control (796 studies, 2,500 updates)
    arm_a_dir = PROJECT_ROOT / "runs" / "s5_duration_control_2500"
    arm_a_final = arm_a_dir / "checkpoints" / "final_segmentation_v3.pt"
    arm_a_last = arm_a_dir / "checkpoints" / "last_segmentation_v3.pt"
    if arm_a_final.exists():
        print(f"Arm A already completed at: {arm_a_final}")
    else:
        cmd_arm_a = [
            PYTHON_EXE,
            str(PROJECT_ROOT / "scripts" / "train_segmentation_v3.py"),
            "--output_dir", str(arm_a_dir),
            "--target_updates", "2500",
            "--effective_batch_size", "4",
            "--micro_batch_size", "2",
            "--img_size", "768",
            "--lr", "1e-4",
            "--seed", "43",
            "--aug_seed", "43",
            "--hard_negative_manifest", str(hn_manifest),
        ]
        if arm_a_last.exists():
            print(f"Resuming Arm A from checkpoint: {arm_a_last}")
            cmd_arm_a.extend(["--resume_checkpoint", str(arm_a_last)])
        else:
            cmd_arm_a.extend(["--load_initial_weights", str(s3_ckpt)])
        run_command_stream(cmd_arm_a, "Arm A: Duration Control (400 pos, 396 neg, 2,500 updates)")

    # 3. Arm B: Expanded Cohort (1,196 studies, 2,500 updates)
    arm_b_dir = PROJECT_ROOT / "runs" / "s5_expanded_cohort_2500"
    arm_b_final = arm_b_dir / "checkpoints" / "final_segmentation_v3.pt"
    arm_b_last = arm_b_dir / "checkpoints" / "last_segmentation_v3.pt"
    if arm_b_final.exists():
        print(f"Arm B already completed at: {arm_b_final}")
    else:
        cmd_arm_b = [
            PYTHON_EXE,
            str(PROJECT_ROOT / "scripts" / "train_segmentation_v3.py"),
            "--output_dir", str(arm_b_dir),
            "--cohort_manifest", str(exp_manifest),
            "--target_updates", "2500",
            "--effective_batch_size", "4",
            "--micro_batch_size", "2",
            "--img_size", "768",
            "--lr", "1e-4",
            "--seed", "43",
            "--aug_seed", "43",
            "--hard_negative_manifest", str(hn_manifest),
        ]
        if arm_b_last.exists():
            print(f"Resuming Arm B from checkpoint: {arm_b_last}")
            cmd_arm_b.extend(["--resume_checkpoint", str(arm_b_last)])
        else:
            cmd_arm_b.extend(["--load_initial_weights", str(s3_ckpt)])
        run_command_stream(cmd_arm_b, "Arm B: Expanded Cohort (800 pos, 396 neg, 2,500 updates)")

    print("=" * 70)
    print("BOTH ARMS COMPLETED SUCCESSFULLY! Launching native-resolution evaluation.")
    print("=" * 70)

    # 4. Evaluation
    cmd_eval = [
        PYTHON_EXE,
        str(PROJECT_ROOT / "scripts" / "evaluate_s5_comparison.py"),
    ]
    run_command_stream(cmd_eval, "Native-Resolution Evaluation & Bootstrap Comparison")


if __name__ == "__main__":
    main()
