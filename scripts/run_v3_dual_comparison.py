"""Sequential, logged execution of the user-authorized V3 reviewed-label comparison.

Runs:
1. Segmentation-only continuation (1,000 steps) -> control
2. Dual-head continuation (1,000 steps) -> tip supervision test

Both initialized from the exact same backbone checkpoint:
runs/pilot_coarse_repaired/checkpoints/best_coarse.pt
using frozen reviewed_distal_tips_v3.parquet (39 positive training tips across 23 studies,
396 confirmed negative studies, partial_review_policy="exclude").
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "v3_comparison_20261012"
RUN.mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1")

def write_json(name, data):
    dest = RUN / name
    temp = dest.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(dest)

def compute_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]

v3_path = ROOT / "data" / "reviewed_distal_tips_v3.parquet"
if not v3_path.exists():
    raise FileNotFoundError(f"V3 label table not found at {v3_path}")
v3_hash = compute_file_hash(v3_path)

status = {
    "status": "running",
    "scope": "V3 engineering-reviewed label feasibility; awaiting clinical attestation",
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "v3_label_file": "data/reviewed_distal_tips_v3.parquet",
    "v3_label_hash": v3_hash,
    "arms": {},
}
write_json("status.json", status)

results = {}
try:
    for mode in ("seg_only", "dual_head"):
        output = RUN / mode
        command = [
            sys.executable, "-u", str(ROOT / "scripts" / "05_train_coarse_dual.py"),
            "--mode", mode,
            "--output_dir", str(output),
            "--max_steps", "1000",
            "--batch_size", "4",
            "--seed", "42",
            "--seg_pos_weight", "50",
            "--seg_dice_weight", "2",
            "--seg_bce_weight", "1",
            "--tip_pos_weight", "100",
            "--tip_weight", "2",
            "--partial_review_policy", "exclude",
            "--reviewed_tips_path", "data/reviewed_distal_tips_v3.parquet",
            "--execute_training",
        ]
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Launching arm: {mode}...")
        status["current_arm"] = mode
        status["arms"][mode] = {
            "status": "running",
            "command": command,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json("status.json", status)

        with (RUN / f"{mode}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)

        if result.returncode != 0:
            raise RuntimeError(f"{mode} exited with code {result.returncode}; see {RUN / f'{mode}.log'}")

        history_path = output / "history.json"
        if not history_path.exists():
            raise FileNotFoundError(f"history.json missing in {output}")
        history = json.loads(history_path.read_text(encoding="utf-8"))

        if history["final_step"] != 1000:
            raise RuntimeError(f"{mode} did not complete the fixed-step budget of 1000 steps")

        results[mode] = {
            "final": history["val_evaluations"][-1],
            "best_step": history["best_step"],
            "best_metric": history["best_metric"],
            "seconds": history["total_time_seconds"],
            "initialization": history["resolved_config"]["initialization"],
            "reviewed_tips_hash": history["resolved_config"]["input_hashes"]["reviewed_tips_hash"],
        }
        status["arms"][mode]["status"] = "complete"
        status["arms"][mode]["completed_utc"] = datetime.now(timezone.utc).isoformat()
        status["arms"][mode]["seconds"] = history["total_time_seconds"]
        write_json("comparison.json", results)
        write_json("status.json", status)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Arm {mode} completed in {history['total_time_seconds']:.1f}s.")

    # Consistency checks
    seg_bb = results["seg_only"]["initialization"]["backbone_hash"]
    dual_bb = results["dual_head"]["initialization"]["backbone_hash"]
    if seg_bb != dual_bb:
        raise RuntimeError(f"Initial backbone hash mismatch: {seg_bb} vs {dual_bb}")

    seg_tip_h = results["seg_only"]["reviewed_tips_hash"]
    dual_tip_h = results["dual_head"]["reviewed_tips_hash"]
    if seg_tip_h != v3_hash or dual_tip_h != v3_hash:
        raise RuntimeError(f"Reviewed tips hash mismatch with V3: {seg_tip_h}, {dual_tip_h} vs {v3_hash}")

    status["status"] = "complete"
    print("\nAll arms completed successfully. Checkpoint backbone hashes matched perfectly.")

except Exception as exc:
    status["status"] = "failed"
    status["error"] = str(exc)
    print(f"\nExecution failed: {exc}", file=sys.stderr)
    raise
finally:
    status["updated_utc"] = datetime.now(timezone.utc).isoformat()
    write_json("status.json", status)
