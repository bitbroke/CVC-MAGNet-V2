"""Direction B Master Overnight Heavy Training Orchestrator.

Executes:
1. E96: Full-Cohort Noisy-Student Distillation on DenseNet-121 across 21,855 studies (10 epochs).
2. E97: Full-Cohort Noisy-Student Distillation on SEResNeXt-50 across 21,855 studies (10 epochs).
3. Level-2 Meta-Stacker Re-Training & Validation Evaluation on locked validation set (N=2,558).
4. Master Ledger & Artifact updates.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Ankshit\CVC-MAGNet-V2")
PYTHON_EXE = sys.executable

def run_command_logged(cmd: list, log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        print(f"\n[ORCHESTRATOR] Launching: {' '.join(cmd)}")
        print(f"[ORCHESTRATOR] Logging output to: {log_file}")
        f.write(f"=== LAUNCHED AT {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(f"COMMAND: {' '.join(cmd)}\n\n")
        f.flush()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(PROJECT_ROOT)
        )
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()

        process.wait()
        f.write(f"\n=== FINISHED WITH CODE {process.returncode} AT {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        return process.returncode

def main():
    print("=" * 80)
    print("DIRECTION B: AUTONOMOUS OVERNIGHT HEAVY TRAINING SPRINT")
    print("=" * 80)
    print("Target: Closing the gap between 0.8338 and the 0.8923 Human Oracle Ceiling.")
    print("Evaluation Cohort: Locked Validation Split (N=2,558).")
    print("Sealed Test Split: Permanently Frozen (Zero Touching).")
    print("=" * 80)

    # 1. E96: Full-Cohort Noisy-Student DenseNet-121 (10 Epochs)
    preds_e96 = PROJECT_ROOT / "runs" / "val_preds" / "val_preds_E96_noisy_student_densenet121_512.npy"
    if preds_e96.exists():
        print(f"[ORCHESTRATOR] E96 already completed ({preds_e96.name} exists). Skipping to E97!")
    else:
        log_e96 = PROJECT_ROOT / "runs" / "E96_noisy_student_densenet121_512" / "train_e96.log"
        cmd_e96 = [
            PYTHON_EXE, "-u",
            str(PROJECT_ROOT / "scripts" / "train_e96_e97_noisy_student.py"),
            "--model", "e96",
            "--epochs", "10"
        ]
        code_96 = run_command_logged(cmd_e96, log_e96)
        if code_96 != 0:
            print(f"[WARNING] E96 exited with code {code_96}. Check log: {log_e96}")

    # 2. E97: Full-Cohort Noisy-Student SEResNeXt-50 (10 Epochs)
    log_e97 = PROJECT_ROOT / "runs" / "E97_noisy_student_seresnext50_512" / "train_e97.log"
    cmd_e97 = [
        PYTHON_EXE, "-u",
        str(PROJECT_ROOT / "scripts" / "train_e96_e97_noisy_student.py"),
        "--model", "e97",
        "--epochs", "10"
    ]
    code_97 = run_command_logged(cmd_e97, log_e97)
    if code_97 != 0:
        print(f"[WARNING] E97 exited with code {code_97}. Check log: {log_e97}")

    # 3. Level-2 Meta-Stacker Apex Fusion
    print("\n" + "=" * 80)
    print("Running Level-2 Grand Meta-Stacker with New Distilled Students...")
    print("=" * 80)
    cmd_stacker = [
        PYTHON_EXE, "-u",
        str(PROJECT_ROOT / "scripts" / "run_e99_grand_meta_apex.py")
    ]
    log_stacker = PROJECT_ROOT / "runs" / "meta_apex_overnight.log"
    code_stacker = run_command_logged(cmd_stacker, log_stacker)

    print("\n" + "=" * 80)
    print(f"Direction B Sprint Completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == "__main__":
    main()

