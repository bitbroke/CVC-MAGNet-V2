"""
Generate baseline_manifest.json recording paths, sizes, and SHA-256 hashes of:
- T0 skeleton predictions
- T1 seed-42 and seed-43 checkpoints
- Saved heatmaps and decoded predictions
- Original 41-target benchmark manifest and registry
- Training configuration and preprocessing code
"""
import hashlib
import json
from pathlib import Path

EXP_V1 = Path("C:/Ankshit/CVC-MAGNet-V2/experiments/hrnet_tip_pilot_v1")
EXP_V2 = Path("C:/Ankshit/CVC-MAGNet-V2/experiments/hrnet_tip_improvement_v2")

def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def make_manifest():
    files_to_record = {
        "t0_skeleton_predictions": EXP_V1 / "predictions" / "t0_baseline_predictions.json",
        "t1_seed42_checkpoint": EXP_V1 / "runs" / "seed_42" / "t1_joint_hrnet" / "checkpoints" / "final_model.pt",
        "t1_seed43_checkpoint": EXP_V1 / "runs" / "seed_43" / "t1_joint_hrnet" / "checkpoints" / "final_model.pt",
        "t1_seed42_decoded_candidates": EXP_V1 / "runs" / "seed_42" / "t1_joint_hrnet" / "decoded_candidates.csv",
        "t1_seed43_decoded_candidates": EXP_V1 / "runs" / "seed_43" / "t1_joint_hrnet" / "decoded_candidates.csv",
        "original_41_target_benchmark_manifest": EXP_V1 / "manifests" / "development_benchmark_manifest.json",
        "original_instance_target_registry_csv": EXP_V1 / "manifests" / "instance_target_registry.csv",
        "original_instance_target_registry_parquet": EXP_V1 / "manifests" / "instance_target_registry.parquet",
        "training_config_seed42": EXP_V1 / "runs" / "seed_42" / "t1_joint_hrnet" / "resolved_config.json",
        "training_config_seed43": EXP_V1 / "runs" / "seed_43" / "t1_joint_hrnet" / "resolved_config.json",
        "model_code_hrnet_joint": EXP_V1 / "src" / "models" / "hrnet_joint_tip.py",
        "geometry_code": EXP_V1 / "src" / "geometry.py",
        "decoder_code": EXP_V1 / "src" / "decoder.py",
        "evaluation_code": EXP_V1 / "src" / "evaluation.py",
        "train_runner_code": EXP_V1 / "src" / "train_runner_tip.py",
    }

    manifest_entries = {}
    for key, p in files_to_record.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")
        manifest_entries[key] = {
            "path": str(p),
            "size_bytes": p.stat().st_size,
            "sha256": compute_sha256(p),
        }

    out_doc = {
        "experiment_name": "hrnet_tip_improvement_v2",
        "created_at": "2026-10-18T07:28:00Z",
        "baseline_source": "hrnet_tip_pilot_v1",
        "summary": "Preserved baseline references for HRNet distal-tip localization improvement sprint.",
        "artifacts": manifest_entries,
    }

    out_path = EXP_V2 / "baseline_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, indent=2)

    print(f"Recorded {len(manifest_entries)} baseline artifacts in {out_path}")
    for k, v in manifest_entries.items():
        print(f"  {k}: {v['sha256'][:16]}... ({v['size_bytes']} bytes)")

if __name__ == "__main__":
    make_manifest()
