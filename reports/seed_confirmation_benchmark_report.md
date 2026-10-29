# Seed Confirmation Benchmark Report: Seed 42 vs Seed 43

**Protocol & Parity Guarantees**:
- **Initialization**: Both arms and both seeds initialized from identical frozen S3 checkpoint (`runs/s3_experiment_hard_neg`).
- **Seed 42 & Seed 43**: Matched pilots executed sequentially under strict single-process GPU discipline with same-batch AMP retries.
- **Predefined Tolerance**: Maximum 1.0 percentage point recall reduction allowed (Minimum acceptable recall $\ge 72.59\%$).

## 1. Dual-Seed Primary Decision Gate Comparison at $\tau=0.50$ (Native Grid)

| Model Arm | Seed | Sampling Policy | Binary Dice | Precision | Recall | Cov @ 10px | Missed @ 10px | Neg False Px | Neg False % |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S3 Reference Baseline** | - | Frozen Baseline | 0.4000 | 27.91% | 73.59% | 79.6% | 4.8% | 10,307 px | **0.162%** |
| **Arm A: Control** | **42** | 100% Uniform Neg | 0.4330 | 31.02% | 74.64% | 80.6% | 3.6% | 8,538 px | **0.125%** |
| **Arm B: Experiment** | **42** | 50% Uniform + 50% Top-99 Hard | 0.4363 | 31.35% | 74.24% | 80.2% | 3.6% | 6,755 px | **0.102%** |
| **Arm A: Control** | **43** | 100% Uniform Neg | 0.4394 | 31.98% | 74.24% | 80.3% | 5.6% | 6,932 px | **0.096%** |
| **Arm B: Experiment** | **43** | 50% Uniform + 50% Top-99 Hard | 0.4419 | 32.10% | 74.47% | 80.5% | 4.4% | 5,714 px | **0.078%** |

## 2. Seed 43 Bootstrap Confidence Intervals (95% CI, $\tau=0.50$)

- **Arm B (Selective Hard Neg) vs Arm A (Uniform Control) [Seed 43]** (N=132 patients, 185 positive studies):
  - $\Delta$ Macro Binary Dice: **[-0.0013, +0.0059]**
  - $\Delta$ Pooled Precision: **[-0.18%, +0.41%]**
  - $\Delta$ Pooled Recall: **[-0.29%, +0.64%]**
- **Arm B [Seed 43] Absolute Recall Bootstrap CI**: **[71.48%, 77.62%]** (Point estimate: 74.47%)
- **Arm B [Seed 43] Paired Recall Change vs S3 Reference Baseline (73.59%)**: **+0.88%** (Point estimate positive)

## 3. Evaluation Against Predefined Tolerance & Decision Rules
- **Point Estimate Tolerance ($\ge 72.59\%$)**:
  - Seed 42 Recall: 74.24% vs Baseline 73.59% -> **Preserved**
  - Seed 43 Recall: 74.47% vs Baseline 73.59% -> **Preserved** (+0.88% paired gain)
- **Statistical Nuance on Absolute CI Lower Bound**:
  - The absolute recall 95% bootstrap CI is [71.48%, 77.62%]. Because the lower bound (71.48%) is below 72.59%, declaring unqualified "all criteria passed" under that literal wording is inaccurate. The paired delta is positive and supports expansion, but the absolute interval width reflects sample size ($N=185$).
- **Seed 42 Parity Caveat**:
  - Seed 42 was trained prior to the in-place AMP retry fix; only Seed 43 used the repaired retry mechanism with guaranteed 3,000/3,000 positive exposure parity.
- **Decision Outcome**:
  - Confirmed repeatability of false-mask suppression (-18% to -21%) and precision improvement (32.10% vs 27.91% baseline) with preserved catheter recall justifies proceeding directly to the controlled 2,500-update expansion comparison.
