# Controlled Hard-Negative False-Track Suppression Benchmark Report

**Experimental Parity & Protocol**:
- **Reference Freeze**: Architecture (`CVCSegmentationV3Net`, 768px), starting checkpoint (`s2_repaired_768`), threshold 0.50, and 193-study validation split frozen.
- **Arm A (Control)**: Uniform negative sampling from 396 training confirmed negatives (1,000 updates).
- **Arm B (Experiment)**: 50% uniform, 50% audited hard negatives mined from training confirmed negatives (1,000 updates).
- **Guaranteed Constants**: Same initialization, 3:1 positive/negative ratio, effective batch size 4, cosine LR decay, and synchronized augmentation.

## 1. Primary Decision Gate Comparison at Threshold 0.50

| Model Arm | Sampling Policy | Binary Dice | Macro Soft Dice | Precision | Recall | Cov @ 10px | Missed @ 10px | Neg False Px | Neg False % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Frozen Reference Baseline** | None (Pre-experiment) | 0.3185 | 0.0299 | 19.74% | 69.52% | 75.7% | 5.6% | 22,920 px | **0.357%** |
| **Arm A: Control (Final)** | 100% Uniform Negatives | 0.3984 | 0.0501 | 27.78% | 73.54% | 79.4% | 5.2% | 11,127 px | **0.174%** |
| **Arm B: Experiment (Final)** | 50% Uniform, 50% Hard Neg | 0.4000 | 0.0501 | 27.91% | 73.59% | 79.6% | 4.8% | 10,307 px | **0.162%** |

## 2. Multi-Threshold Performance Profile (Arm B Experiment vs Arm A Control)

| Model | $\tau$ | Binary Dice | Precision | Recall | Cov @ 10px | Missed @ 10px | Neg False Px | Neg False % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm A Control (Final)** | **0.30** | 0.3232 | 19.62% | 78.47% | 82.3% | 3.6% | 19,119 px | 0.303% |
| **Arm A Control (Final)** | **0.40** | 0.3644 | 23.61% | 75.93% | 80.6% | 4.4% | 13,939 px | 0.220% |
| **Arm A Control (Final)** | **0.50** | 0.3984 | 27.78% | 73.54% | 79.4% | 5.2% | 11,127 px | 0.174% |
| **Arm A Control (Final)** | **0.60** | 0.4248 | 31.32% | 70.93% | 78.0% | 5.2% | 8,953 px | 0.139% |
| **Arm A Control (Final)** | **0.70** | 0.4505 | 35.32% | 67.45% | 76.4% | 5.6% | 7,028 px | 0.108% |
| **Arm B Experiment (Final)** | **0.30** | 0.3235 | 19.61% | 78.56% | 82.3% | 4.0% | 18,112 px | 0.289% |
| **Arm B Experiment (Final)** | **0.40** | 0.3653 | 23.64% | 75.96% | 80.8% | 4.8% | 12,838 px | 0.204% |
| **Arm B Experiment (Final)** | **0.50** | 0.4000 | 27.91% | 73.59% | 79.6% | 4.8% | 10,307 px | 0.162% |
| **Arm B Experiment (Final)** | **0.60** | 0.4266 | 31.53% | 70.90% | 78.2% | 5.2% | 8,323 px | 0.130% |
| **Arm B Experiment (Final)** | **0.70** | 0.4524 | 35.57% | 67.37% | 76.2% | 5.6% | 6,519 px | 0.101% |

## 3. Paired Patient-Cluster Bootstrap CIs (95% CI, $\tau=0.50$)

- **Arm B (Hard Neg Experiment) vs Arm A (Uniform Control)** (N=132 patients, 185 positive studies):
  - $\Delta$ Macro Binary Dice: **[+0.0003, +0.0029]**
  - $\Delta$ Pooled Precision: **[+0.01%, +0.25%]**
  - $\Delta$ Pooled Recall: **[-0.16%, +0.24%]**

- **Arm B (Hard Neg Experiment) vs Frozen Reference Baseline** (N=132 patients, 185 positive studies):
  - $\Delta$ Macro Binary Dice: **[+0.0741, +0.0900]**
  - $\Delta$ Pooled Precision: **[+7.12%, +9.18%]**
  - $\Delta$ Pooled Recall: **[+2.91%, +5.17%]**

## 4. Assessment Against User Decision Criteria
- **False Track Reduction**: Evaluated on confirmed negative radiographs.
- **Precision & Coverage**: Evaluated across native resolution grid.
- **Mask Size Verification**: Confirmed model is actively segmenting CVC paths rather than collapsing foreground to empty.
