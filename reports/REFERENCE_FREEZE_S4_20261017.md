# Reference Baseline Freeze: S3 and S4 Checkpoints

**Date**: 2026-10-17  
**Scope**: Locked reference baselines prior to launching Seed-43 confirmation and controlled cohort expansion.

---

## 1. Preserved Checkpoints & Artifact Provenance

| Model Identifier | Run Directory | Checkpoint Path | SHA256 Hash |
| :--- | :--- | :--- | :--- |
| **S3 Reference Baseline** | `runs/s3_experiment_hard_neg` | `checkpoints/final_segmentation_v3.pt` | `6520e32629e5ea0cf710682790495241f7fc79266b3ee500a324f1323338a53a` |
| **S4 Arm A Control (Seed 42)** | `runs/s4_control_uniform_neg` | `checkpoints/final_segmentation_v3.pt` | `41af15c52d5cb80b33b00fdfc9ce519e917d5c7ee077227d8dbccf7f41f0d367` |
| **S4 Arm B Experiment (Seed 42)** | `runs/s4_experiment_selective_hard_neg` | `checkpoints/final_segmentation_v3.pt` | `89531f91740ffe91da0ec00730b91d44fe115e523f6eeeb79be810b42ddcf5ad` |
| **S4 Arm A Control (Seed 43)** | `runs/s4_control_seed43` | `checkpoints/final_segmentation_v3.pt` | `2d2bb6ffbfe8902008f107c8a6e87f87fdfebcf78696ecdd2c0c7b2e8a1562b7` |
| **S4 Arm B Experiment (Seed 43)** | `runs/s4_experiment_seed43` | `checkpoints/final_segmentation_v3.pt` | `f01089be1b7507bfb1d1a2762911f6ee1f0d425ce49628c6d0083daecf5e94e2` |

All runs maintained fixed architecture (`CVCSegmentationV3Net`), 768px native letterbox geometry, BCE + Soft Dice loss weights, and $\tau=0.50$ evaluation grid.

---

## 2. Locked Reference Evaluation Metrics ($\tau=0.50$, Native Grid, 193 Studies)

Evaluated on the frozen `dev_pilot_val` partition (185 positive studies, 8 negative studies, 132 unique patients):

| Model Arm | Seed | Sampling Policy | Binary Dice | Precision | Recall | Cov @ 10px | Missed @ 10px | Neg False Px | Neg False % |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S3 Reference Baseline** | - | Frozen Reference (Pre-S4) | 0.4000 | 27.91% | 73.59% | 79.6% | 4.8% | 10,307 px | **0.162%** |
| **S4 Arm A: Control** | **42** | 100% Uniform Negatives | 0.4330 | 31.02% | 74.64% | 80.6% | 3.6% | 8,538 px | **0.125%** |
| **S4 Arm B: Experiment** | **42** | 50% Uniform + 50% Top-99 Hard | **0.4363** | **31.35%** | **74.24%** | **80.2%** | **3.6%** | **6,755 px** | **0.102%** |
| **S4 Arm A: Control** | **43** | 100% Uniform Negatives | 0.4394 | 31.98% | 74.24% | 80.3% | 5.6% | 6,932 px | **0.096%** |
| **S4 Arm B: Experiment** | **43** | 50% Uniform + 50% Top-99 Hard | **0.4419** | **32.10%** | **74.47%** | **80.5%** | **4.4%** | **5,714 px** | **0.078%** |

---

## 3. Preserved Artifacts & Ledgers

- **Manifest**: `data/selective_hard_negatives_manifest.json` (99 audited training negatives).
- **Exposure Ledgers**:
  - `runs/s4_control_uniform_neg/exposure_ledger.jsonl` (1,000 updates, 4,000 exposures).
  - `runs/s4_experiment_selective_hard_neg/exposure_ledger.jsonl` (1,000 updates, 4,000 exposures).
- **Resolved Configurations**:
  - `runs/s4_control_uniform_neg/resolved_config.json`
  - `runs/s4_experiment_selective_hard_neg/resolved_config.json`
- **Benchmark Report**: `reports/s4_selective_hard_negative_benchmark_report.md`
- **Bootstrap JSONs**: `reports/bootstrap_ci_s4_exp_vs_ctrl.json`, `reports/bootstrap_ci_s4_exp_vs_ref.json`.
