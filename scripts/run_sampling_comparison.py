"""Controlled Comparison Runner: Uniform Sampling (Control) vs. Tip-Balanced Sampling (Experiment).

Evaluates the isolated effect of controlled batch sampling on direct tip heatmap localization:
- Model Architecture: Dual-head CoarseDualNet (identical)
- Initialization: runs/pilot_coarse_repaired/checkpoints/best_coarse.pt (identical backbone hash)
- Supervision Labels: data/reviewed_distal_tips_v3.parquet (identical v3 table)
- Budget: Exactly 1,000 successful optimizer updates
- Thresholds & Decoder: Unchanged (tau=0.30, max_peaks=10, min_distance=10, NO plateau filtering)
- Comparison Targets:
  1. All-target validation Success@15 (N=25 targets, one-to-one Hungarian matching)
  2. False peaks on negative studies (N=8 validation negatives)
  3. Fully supervised training accuracy (N=39 active training targets across 23 studies)
  4. Observed localization error distributions (only where predictions exist; missing reported separately)
  5. Measured exposure statistics (positive batches, per-study presentations, AMP skips)
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.coarse_dual import CoarseDualNet, decode_tip_peaks
from src.data.datasets import CoarseDualDataset
from src.data.transforms import TransformMetadata
import importlib.util
_spec = importlib.util.spec_from_file_location("eval_mod", PROJECT_ROOT / "scripts" / "06_evaluate.py")
_eval_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_mod)
threshold_constrained_matching = _eval_mod.threshold_constrained_matching

env = os.environ.copy()
env.update(
    OMP_NUM_THREADS="2",
    MKL_NUM_THREADS="2",
    PYTHONUNBUFFERED="1",
    OPENCV_OPENCL_DEVICE="disabled",
    OPENCV_FORBID_OPENCL="1",
    CV_NUM_THREADS="0",
    PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128",
)


def write_json(run_dir: Path, name: str, data: dict):
    dest = run_dir / name
    temp = dest.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(dest)


def compute_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]


def evaluate_supervised_training_targets(
    checkpoint_path: Path,
    reviewed_tips_path: Path,
    device: torch.device,
) -> dict:
    """Evaluate localization accuracy on the 39 active, fully reviewed training targets.
    
    Fix: Missing predictions are counted and reported separately, NOT encoded as artificial 999px errors.
    Observed distance statistics are computed strictly on targets where candidate predictions exist.
    """
    with open(PROJECT_ROOT / "configs/pilot.yaml") as f:
        cfg = yaml.safe_load(f)
    raw_images_dir = Path(cfg["data"]["raw_dir"]) / "train"

    inv_df = pd.read_parquet(PROJECT_ROOT / "data/annotation_inventory.parquet")
    study_df = pd.read_parquet(PROJECT_ROOT / "data/study_inventory.parquet")
    rev_v3_df = pd.read_parquet(reviewed_tips_path)

    confirmed_neg_ids = set(
        study_df[study_df["is_study_level_negative"] & (~study_df["has_cvc"])]["StudyInstanceUID"]
    )
    rev_train_v3 = rev_v3_df[rev_v3_df["split"] == "train"].copy()
    active_sids = rev_train_v3["study_id"].unique().tolist()

    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CoarseDualNet(in_channels=1, base_channels=32).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    train_ds = CoarseDualDataset(
        study_ids=active_sids,
        inventory_df=inv_df,
        images_dir=raw_images_dir,
        confirmed_negative_study_ids=confirmed_neg_ids,
        reviewed_tips_df=rev_train_v3,
        target_size=(512, 512),
        mask_stroke_width_native=10,
        clahe_enabled=True,
        tip_sigma=2.5,
        partial_review_policy="exclude",
    )
    loader = DataLoader(train_ds, batch_size=1, shuffle=False)

    matched_15_errors = []
    observed_nearest_dists = []
    targets_with_no_predictions = 0
    prob_at_true_tips = []
    total_targets = 0

    with torch.no_grad():
        for batch in loader:
            sid = batch["study_id"][0]
            raw_meta = {k: v[0].item() if hasattr(v[0], "item") else v[0] for k, v in batch["meta"].items()}
            meta = TransformMetadata(**raw_meta)
            img = batch["image"].to(device)

            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
                _, tip_logits = model(img)
            tip_probs = torch.sigmoid(tip_logits)[0, 0].cpu().numpy()

            # Decoder unchanged: threshold=0.30, min_distance=10, max_peaks=10
            peaks = decode_tip_peaks(tip_probs, threshold=0.30, min_distance=10, max_peaks=10)
            pred_native = []
            for p in peaks:
                nx, ny = meta.model_to_native(p[0], p[1])
                pred_native.append([nx, ny])
            pred_arr = np.array(pred_native, dtype=np.float32) if pred_native else np.zeros((0, 2), dtype=np.float32)

            gt_rows = rev_train_v3[rev_train_v3["study_id"] == sid]
            gt_native = []
            for _, r in gt_rows.iterrows():
                tx, ty = float(r["reviewed_tip_x"]), float(r["reviewed_tip_y"])
                gt_native.append([tx, ty])
                total_targets += 1

                # Sample model probability directly at true tip
                mx, my = meta.native_to_model(tx, ty)
                mi, mj = int(np.clip(round(mx), 0, 511)), int(np.clip(round(my), 0, 511))
                prob_at_true_tips.append(float(tip_probs[mj, mi]))

                # Nearest-prediction diagnostic distance (only recorded if candidates exist)
                if len(pred_arr) > 0:
                    d = np.min(np.hypot(pred_arr[:, 0] - tx, pred_arr[:, 1] - ty))
                    observed_nearest_dists.append(float(d))
                else:
                    targets_with_no_predictions += 1

            gt_arr = np.array(gt_native, dtype=np.float32)
            matched_15, _, _ = threshold_constrained_matching(pred_arr, gt_arr, radius=15.0)
            for _, _, dist in matched_15:
                matched_15_errors.append(dist)

    return {
        "active_training_targets_total": total_targets,
        "matched_15_count": len(matched_15_errors),
        "success_at_15": len(matched_15_errors) / max(total_targets, 1),
        "median_matched_error_15": float(np.median(matched_15_errors)) if matched_15_errors else None,
        "targets_with_candidates_count": len(observed_nearest_dists),
        "targets_missing_candidates_count": targets_with_no_predictions,
        "diagnostic_observed_nearest_dist": {
            "median": float(np.median(observed_nearest_dists)) if observed_nearest_dists else None,
            "q25": float(np.quantile(observed_nearest_dists, 0.25)) if observed_nearest_dists else None,
            "q75": float(np.quantile(observed_nearest_dists, 0.75)) if observed_nearest_dists else None,
            "note": "Nearest-prediction distance is an unconstrained diagnostic metric, NOT one-to-one matched error.",
        },
        "prob_at_true_tip_mean": float(np.mean(prob_at_true_tips)) if prob_at_true_tips else None,
        "prob_at_true_tip_median": float(np.median(prob_at_true_tips)) if prob_at_true_tips else None,
    }


def get_fresh_run_dir(base_name: str) -> Path:
    """Find a fresh directory that does NOT contain existing resolved_config.yaml."""
    runs_dir = PROJECT_ROOT / "runs"
    candidate = runs_dir / base_name
    if not (candidate / "control_uniform" / "resolved_config.yaml").exists() and \
       not (candidate / "experiment_controlled" / "resolved_config.yaml").exists():
        return candidate

    idx = 2
    while True:
        candidate = runs_dir / f"{base_name}_v{idx}"
        if not (candidate / "control_uniform" / "resolved_config.yaml").exists() and \
           not (candidate / "experiment_controlled" / "resolved_config.yaml").exists():
            return candidate
        idx += 1


def assert_comparison_integrity(results: dict):
    print("\n" + "=" * 80)
    print("ENFORCING COMPARISON INTEGRITY ASSERTIONS")
    print("=" * 80)
    ctrl = results["control_uniform"]
    exp = results["experiment_controlled"]

    # 1. Budget: Exactly 1,000 successful updates on both arms
    ctrl_steps = ctrl["final_validation"]["step"]
    exp_steps = exp["final_validation"]["step"]
    assert ctrl_steps == 1000, f"Control step mismatch: {ctrl_steps} != 1000"
    assert exp_steps == 1000, f"Experiment step mismatch: {exp_steps} != 1000"

    ctrl_updates = ctrl["telemetry"]["successful_updates"]
    exp_updates = exp["telemetry"]["successful_updates"]
    assert ctrl_updates == 1000, f"Control successful updates mismatch: {ctrl_updates} != 1000"
    assert exp_updates == 1000, f"Experiment successful updates mismatch: {exp_updates} != 1000"
    print("[PASS] Assertion passed: Both arms completed exactly 1,000 successful optimizer updates.")

    # 2. Initialization: Identical backbone and init checkpoint hashes
    ctrl_bb = ctrl["initialization"]["backbone_hash"]
    exp_bb = exp["initialization"]["backbone_hash"]
    assert ctrl_bb == exp_bb, f"Backbone hash mismatch: {ctrl_bb} vs {exp_bb}"

    ctrl_init_ckpt = ctrl["initialization"]["init_ckpt_hash"]
    exp_init_ckpt = exp["initialization"]["init_ckpt_hash"]
    assert ctrl_init_ckpt == exp_init_ckpt, f"Init checkpoint hash mismatch: {ctrl_init_ckpt} vs {exp_init_ckpt}"
    print(f"[PASS] Assertion passed: Identical initialization (backbone hash: {ctrl_bb}).")

    # 3. Supervision & Evaluation Cohort Parity
    ctrl_val_n = ctrl["final_validation"]["reviewed_targets_total"]
    exp_val_n = exp["final_validation"]["reviewed_targets_total"]
    assert ctrl_val_n == 25 and exp_val_n == 25, f"Validation targets mismatch: {ctrl_val_n} vs {exp_val_n}"

    ctrl_train_n = ctrl["supervised_training_eval"]["active_training_targets_total"]
    exp_train_n = exp["supervised_training_eval"]["active_training_targets_total"]
    assert ctrl_train_n == 39 and exp_train_n == 39, f"Active training targets mismatch: {ctrl_train_n} vs {exp_train_n}"

    ctrl_hashes = ctrl["resolved_config"]["input_hashes"]
    exp_hashes = exp["resolved_config"]["input_hashes"]
    for k in ["reviewed_tips_hash", "splits_hash", "inventory_hash", "study_inventory_hash", "config_hash"]:
        assert ctrl_hashes[k] == exp_hashes[k], f"Input hash {k} mismatch: {ctrl_hashes[k]} vs {exp_hashes[k]}"
    print("[PASS] Assertion passed: Identical label tables, splits, and validation targets (N=25 val, N=39 train).")

    # 4. Settings & Hyperparameters Parity (except sampling)
    ctrl_cfg = ctrl["resolved_config"]
    exp_cfg = exp["resolved_config"]
    assert ctrl_cfg["batch_size"] == exp_cfg["batch_size"] == 4, "Batch size must be 4"
    assert ctrl_cfg["losses"] == exp_cfg["losses"], f"Loss config mismatch: {ctrl_cfg['losses']} vs {exp_cfg['losses']}"
    assert ctrl_cfg["optimizer"] == exp_cfg["optimizer"], f"Optimizer mismatch: {ctrl_cfg['optimizer']} vs {exp_cfg['optimizer']}"
    assert ctrl_cfg["geometry_and_supervision"] == exp_cfg["geometry_and_supervision"], "Geometry mismatch"
    print("[PASS] Assertion passed: Identical loss weights, optimizer, and geometry settings.")

    # 5. Distinct Sampling Policies as Specified
    assert ctrl["sampling"] == "uniform", f"Expected uniform sampling for control, got {ctrl['sampling']}"
    assert exp["sampling"] == "tip_balanced", f"Expected tip_balanced sampling for experiment, got {exp['sampling']}"
    print("[PASS] Assertion passed: Sampling policy strictly isolated (control=uniform, experiment=tip_balanced).")
    print("=" * 80 + "\n")


def format_decision_summary(results: dict) -> str:
    ctrl_val = results["control_uniform"]["final_validation"]["tip_heatmap"]
    exp_val = results["experiment_controlled"]["final_validation"]["tip_heatmap"]
    ctrl_trn = results["control_uniform"]["supervised_training_eval"]
    exp_trn = results["experiment_controlled"]["supervised_training_eval"]

    ctrl_val_s15 = ctrl_val["success_at_15"]
    exp_val_s15 = exp_val["success_at_15"]
    ctrl_trn_s15 = ctrl_trn["success_at_15"]
    exp_trn_s15 = exp_trn["success_at_15"]

    lines = []
    lines.append("=" * 80)
    lines.append("SAMPLING COMPARISON PILOT RESULTS & DECISION ANALYSIS")
    lines.append("=" * 80)
    lines.append("Control (Dual-head + Uniform Sampling):")
    lines.append(f"  Validation Success@15 (N=25): {ctrl_val['matched_15_count']}/25 ({ctrl_val_s15:.2%})")
    lines.append(f"  Missing Candidates Count:     {ctrl_val.get('targets_missing_candidates_count', 'N/A')}")
    lines.append(f"  Supervised Training (N=39):   {ctrl_trn['matched_15_count']}/39 ({ctrl_trn_s15:.2%})")
    lines.append(f"  Negative False Peaks (N=8):   {ctrl_val['negative_false_peaks_total']}")
    lines.append("")
    lines.append("Experiment (Dual-head + Controlled / Tip-Balanced Sampling):")
    lines.append(f"  Validation Success@15 (N=25): {exp_val['matched_15_count']}/25 ({exp_val_s15:.2%})")
    lines.append(f"  Missing Candidates Count:     {exp_val.get('targets_missing_candidates_count', 'N/A')}")
    lines.append(f"  Supervised Training (N=39):   {exp_trn['matched_15_count']}/39 ({exp_trn_s15:.2%})")
    lines.append(f"  Negative False Peaks (N=8):   {exp_val['negative_false_peaks_total']}")
    lines.append("-" * 80)
    lines.append("Decision Analysis Framework & Qualification:")
    lines.append(
        "Qualification: Controlled sampling changes the whole sampling policy (study repeat frequency,\n"
        "negative mix, and segmentation balance), not positive-tip exposure alone."
    )
    if exp_trn_s15 > ctrl_trn_s15 and exp_val_s15 <= ctrl_val_s15:
        lines.append(
            "\nObservation: Controlled sampling improved supervised training accuracy without improving validation accuracy.\n"
            "Interpretation: This suggests a generalization problem across studies rather than proving label diversity\n"
            "is the exclusive bottleneck. Annotation quality/noise, overfitting to the small positive training cohort (23 studies),\n"
            "peak decoding parameters/spatial suppression, and the overall sampling distribution could all contribute."
        )
    elif exp_val_s15 > ctrl_val_s15:
        lines.append(
            "\nObservation: Controlled sampling improved validation Success@15.\n"
            "Interpretation: Whole sampling policy change provided measurable generalization benefit across studies."
        )
    else:
        lines.append(
            "\nObservation: Validation performance remained comparable across both arms.\n"
            "Interpretation: Direct tip localization remains constrained; investigate annotation consistency and decoder thresholds."
        )
    lines.append("=" * 80)
    return "\n".join(lines)


def run_sampling_comparison(execute: bool = False, run_dir_name: str = None):
    # Fix 1: Preserve dry-run outputs and use fresh execution directories
    if not execute:
        base_name = f"{run_dir_name or 'sampling_comparison_20261014'}_dryrun"
        run_dir = get_fresh_run_dir(base_name)
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Dry Run] Using fresh dry-run verification directory: {run_dir}")
    else:
        if run_dir_name:
            run_dir = PROJECT_ROOT / "runs" / run_dir_name
        else:
            base_name = "sampling_pilot_20261014"
            run_dir = get_fresh_run_dir(base_name)
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Execution] Using execution directory: {run_dir}")

    v3_path = PROJECT_ROOT / "data" / "reviewed_distal_tips_v3.parquet"
    if not v3_path.exists():
        raise FileNotFoundError(f"V3 label table not found at {v3_path}")
    v3_hash = compute_file_hash(v3_path)

    init_ckpt = PROJECT_ROOT / "runs/pilot_coarse_repaired/checkpoints/best_coarse.pt"
    if not init_ckpt.exists():
        raise FileNotFoundError(f"Initial checkpoint not found at {init_ckpt}")
    init_hash = compute_file_hash(init_ckpt)

    status = {
        "status": "running" if execute else "dry_run_configured",
        "mode": "execution" if execute else "dry_run",
        "run_dir": str(run_dir),
        "description": "Controlled sampling pilot: Uniform (Control) vs Tip-Balanced (Experiment)",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "v3_label_hash": v3_hash,
        "init_ckpt_hash": init_hash,
        "arms": {},
    }
    write_json(run_dir, "status.json", status)

    arm_configs = [
        {
            "arm_name": "control_uniform",
            "sampling": "uniform",
            "output_dir": run_dir / "control_uniform",
        },
        {
            "arm_name": "experiment_controlled",
            "sampling": "tip_balanced",
            "output_dir": run_dir / "experiment_controlled",
        },
    ]

    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        for arm in arm_configs:
            arm_name = arm["arm_name"]
            sampling_mode = arm["sampling"]
            output_path = arm["output_dir"]

            cmd = [
                sys.executable, "-u", str(PROJECT_ROOT / "scripts/05_train_coarse_dual.py"),
                "--mode", "dual_head",
                "--sampling", sampling_mode,
                "--output_dir", str(output_path),
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
                "--init_ckpt", str(init_ckpt),
            ]
            if execute:
                cmd.append("--execute_training")

            print(f"\n{'='*80}")
            print(f"Launching Arm: {arm_name} (sampling={sampling_mode})")
            print(f"Output: {output_path}")
            print(f"{'='*80}")

            status["arms"][arm_name] = {
                "status": "running" if execute else "configured",
                "sampling": sampling_mode,
                "command": cmd,
                "started_utc": datetime.now(timezone.utc).isoformat(),
            }
            status["current_arm"] = arm_name
            write_json(run_dir, "status.json", status)

            log_path = run_dir / f"{arm_name}.log"
            final_ckpt = output_path / "checkpoints" / "final_coarse_dual_head.pt"
            if execute and final_ckpt.exists() and (output_path / "history.json").exists():
                print(f"Checkpoint already exists and verified for {arm_name}. Skipping training and proceeding directly to evaluation.")
            else:
                with open(log_path, "w", encoding="utf-8") as log_file:
                    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)

                if proc.returncode != 0:
                    status["arms"][arm_name]["status"] = "failed"
                    status["arms"][arm_name]["returncode"] = proc.returncode
                    status["arms"][arm_name]["error"] = f"Returncode {proc.returncode}. See {log_path.name}"
                    crash_diag_path = output_path / "crash_diagnostics.json"
                    if crash_diag_path.exists():
                        try:
                            status["arms"][arm_name]["crash_diagnostics"] = json.loads(crash_diag_path.read_text(encoding="utf-8"))
                        except Exception:
                            pass
                    status["status"] = "failed"
                    status["error"] = f"Arm {arm_name} failed with returncode {proc.returncode}. See {log_path.name}"
                    write_json(run_dir, "status.json", status)
                    raise RuntimeError(f"Arm {arm_name} failed with returncode {proc.returncode}. See {log_path}")

            if not execute:
                status["arms"][arm_name]["status"] = "dry_run_verified"
                status["arms"][arm_name]["completed_utc"] = datetime.now(timezone.utc).isoformat()
                write_json(run_dir, "status.json", status)
                print(f"Configuration verified successfully for {arm_name}.")
                continue

            history_path = output_path / "history.json"
            if not history_path.exists():
                raise FileNotFoundError(f"history.json missing at {history_path}")
            history = json.loads(history_path.read_text(encoding="utf-8"))

            final_ckpt = output_path / "checkpoints" / "final_coarse_dual_head.pt"
            print(f"Evaluating supervised training targets for {arm_name}...")
            train_eval = evaluate_supervised_training_targets(final_ckpt, v3_path, device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            results[arm_name] = {
                "sampling": sampling_mode,
                "final_validation": history["val_evaluations"][-1],
                "supervised_training_eval": train_eval,
                "telemetry": history.get("training_telemetry", {}),
                "best_step": history["best_step"],
                "best_metric": history["best_metric"],
                "seconds": history["total_time_seconds"],
                "initialization": history["resolved_config"]["initialization"],
                "resolved_config": history["resolved_config"],
            }
            status["arms"][arm_name]["status"] = "complete"
            status["arms"][arm_name]["completed_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(run_dir, "comparison.json", results)
            write_json(run_dir, "status.json", status)

        # Fix 3: Enforce strict comparison integrity assertions before accepting results
        if execute:
            assert_comparison_integrity(results)

            summary_text = format_decision_summary(results)
            print(summary_text)

            status["status"] = "complete"
            status["completed_utc"] = datetime.now(timezone.utc).isoformat()
            status["current_arm"] = None
            write_json(run_dir, "comparison.json", results)
            write_json(run_dir, "status.json", status)
            (run_dir / "summary.txt").write_text(summary_text, encoding="utf-8")
            print(f"\nBoth arms completed successfully and passed all integrity assertions.")
            print(f"Results saved to {run_dir / 'comparison.json'}")
        else:
            status["status"] = "dry_run_verified"
            status["current_arm"] = None
            status["completed_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(run_dir, "status.json", status)
            print("\nDry-run configuration verification completed for both arms. Ready for execution.")

    except Exception as exc:
        status["status"] = "failed"
        status["error"] = str(exc)
        if "current_arm" in status and status["current_arm"] in status.get("arms", {}):
            if status["arms"][status["current_arm"]].get("status") == "running":
                status["arms"][status["current_arm"]]["status"] = "failed"
                status["arms"][status["current_arm"]]["error"] = str(exc)
        print(f"\nExecution failed: {exc}", file=sys.stderr)
        raise
    finally:
        if status.get("status") == "running":
            status["status"] = "failed"
        for arm_k, arm_v in status.get("arms", {}).items():
            if arm_v.get("status") == "running":
                arm_v["status"] = "failed"
        status["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(run_dir, "status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Launch training execution")
    parser.add_argument("--run_dir", type=str, default=None, help="Custom run directory name")
    args = parser.parse_args()
    run_sampling_comparison(execute=args.execute, run_dir_name=args.run_dir)
