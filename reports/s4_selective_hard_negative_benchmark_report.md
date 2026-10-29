# S4 Selective Hard-Negative Experiment Benchmark Report

**Experimental Parity & Protocol**:
- **Reference Freeze**: Checkpoint `runs/s3_experiment_hard_neg` (Dice 0.4000, Prec 27.91%, Recall 73.59%, False Area 0.162%).
- **Arm A (Control)**: 100% Uniform negative sampling from 396 training negatives (1,000 updates, 768px).
- **Arm B (Experiment)**: 50% Uniform + 50% Selective Top-99 Hard Negatives (top 25% by false foreground area, 1,000 updates, 768px).
- **Strict Equivalence**: Same initialization checkpoint, decoupled positive RNG guaranteeing 100% identical positive study sequences and augmentations, effective batch size 4 (micro-batch 1, accum 4), cosine decay, fixed losses.

## 1. Primary Decision Gate Comparison at Threshold 0.50 (Native Grid)

| Model Arm | Sampling Policy | Binary Dice | Macro Soft Dice | Precision | Recall | Cov @ 10px | Missed @ 10px | Neg False Px | Neg False % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S3 Reference Baseline** | Frozen Reference (Pre-S4) | 0.4000 | 0.0501 | 27.91% | 73.59% | 79.6% | 4.8% | 10,307 px | **0.162%** |
| **S4 Arm A: Control (Final)** | 100% Uniform Negatives | 0.4330 | 0.0843 | 31.02% | 74.64% | 80.6% | 3.6% | 8,538 px | **0.125%** |
| **S4 Arm B: Experiment (Final)** | 50% Uniform + 50% Top-99 Hard | 0.4363 | 0.0838 | 31.35% | 74.24% | 80.2% | 3.6% | 6,755 px | **0.102%** |

## 2. Multi-Threshold Performance Profile (Arm B Experiment vs Arm A Control)

| Model | $\tau$ | Binary Dice | Precision | Recall | Cov @ 10px | Missed @ 10px | Neg False Px | Neg False % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S4 Arm A Control (Final)** | **0.30** | 0.3822 | 25.06% | 78.17% | 82.4% | 3.2% | 12,245 px | 0.179% |
| **S4 Arm A Control (Final)** | **0.40** | 0.4073 | 27.77% | 76.40% | 81.5% | 3.6% | 10,047 px | 0.147% |
| **S4 Arm A Control (Final)** | **0.50** | 0.4330 | 31.02% | 74.64% | 80.6% | 3.6% | 8,538 px | 0.125% |
| **S4 Arm A Control (Final)** | **0.60** | 0.4545 | 33.94% | 72.67% | 79.7% | 3.6% | 7,223 px | 0.105% |
| **S4 Arm A Control (Final)** | **0.70** | 0.4741 | 36.94% | 70.19% | 78.7% | 4.0% | 5,958 px | 0.087% |
| **S4 Arm B Experiment (Final)** | **0.30** | 0.3850 | 25.31% | 77.86% | 82.1% | 3.6% | 9,682 px | 0.148% |
| **S4 Arm B Experiment (Final)** | **0.40** | 0.4106 | 28.09% | 76.03% | 81.1% | 3.6% | 7,965 px | 0.121% |
| **S4 Arm B Experiment (Final)** | **0.50** | 0.4363 | 31.35% | 74.24% | 80.2% | 3.6% | 6,755 px | 0.102% |
| **S4 Arm B Experiment (Final)** | **0.60** | 0.4577 | 34.32% | 72.26% | 79.4% | 3.6% | 5,681 px | 0.085% |
| **S4 Arm B Experiment (Final)** | **0.70** | 0.4778 | 37.42% | 69.76% | 78.4% | 4.0% | 4,698 px | 0.070% |

## 3. Paired Patient-Cluster Bootstrap CIs (95% CI, $\tau=0.50$)

- **Arm B (Selective Hard Neg) vs Arm A (Uniform Control)** (N=132 patients, 185 positive studies):
  - $\Delta$ Macro Binary Dice: **[+0.0006, +0.0057]**
  - $\Delta$ Pooled Precision: **[+0.06%, +0.63%]**
  - $\Delta$ Pooled Recall: **[-0.90%, +0.09%]**

- **Arm B (Selective Hard Neg) vs S3 Frozen Reference Baseline** (N=132 patients, 185 positive studies):
  - $\Delta$ Macro Binary Dice: **[+0.0290, +0.0441]**
  - $\Delta$ Pooled Precision: **[+2.71%, +4.17%]**
  - $\Delta$ Pooled Recall: **[-0.09%, +1.41%]**

## 4. Assessment Against User Decision Criteria
- **False Track Reduction**: Evaluated on confirmed negative radiographs.
- **Precision & Coverage**: Evaluated across native resolution grid.
- **Mask Size Verification**: Confirmed model is actively segmenting CVC paths rather than collapsing foreground to empty.
- **Caveat**: Validation set contains 8 negative studies (66 non-CVC devices across cohort); specificity findings are directional.
