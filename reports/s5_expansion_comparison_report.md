# S5 Cohort Expansion Benchmark Report: Duration Control vs Expanded Cohort

**Date**: 2026-10-17  
**Evaluation Protocol**: Native radiograph grid on frozen `dev_pilot_val` (193 studies: 185 positive, 8 negative, 132 unique patients).  
**Experimental Design**:
- **Initialization**: Identical frozen S3 baseline (`runs/s3_experiment_hard_neg`).
- **Update Budget**: Exactly 2,500 successful optimizer updates for both arms.
- **Negative Sampling**: Exactly identical 50% Top-99 hard negative + 50% uniform negative policy (396 confirmed negatives).
- **Arm A (Duration Control)**: Current cohort (400 positive studies + 396 negatives = 796 studies).
- **Arm B (Expanded Cohort)**: Audited 2x cohort (800 positive studies + 396 negatives = 1,196 studies).
- **Geometry & Distal Tip Audit**: Repaired native-dimension checks; 35 boundary-reaching instances safely clipped for segmentation while preserving raw annotations; clipped endpoints explicitly non-equivalent to distal tips.

---

## 1. Primary Benchmark Comparison at $\tau=0.50$ (Native Grid)

| Arm / Model                                                          | Seed   |   Binary Dice | Precision   | Recall   | Cov @ 10px   | Missed @ 10px   | Neg False Px   | Neg False %   |
|:---------------------------------------------------------------------|:-------|--------------:|:------------|:---------|:-------------|:----------------|:---------------|:--------------|
| S3 Frozen Baseline (1,000 updates, 400 pos)                          | -      |        0.4    | 27.91%      | 73.59%   | 79.6%        | 4.8%            | 10307 px       | 0.162%        |
| S4 Seed-43 Confirmed (1,000 updates, 400 pos, 50% Top-99 Hard)       | 43     |        0.4419 | 32.10%      | 74.47%   | 80.5%        | 4.4%            | 5714 px        | 0.078%        |
| S5 Arm A: Duration Control (2,500 updates, 400 pos, 50% Top-99 Hard) | 43     |        0.0053 | 12.31%      | 0.37%    | 0.9%         | 99.2%           | 0 px           | 0.000%        |
| S5 Arm B: Expanded Cohort (2,500 updates, 800 pos, 50% Top-99 Hard)  | 43     |        0      | 0.00%       | 0.00%    | 0.0%         | 100.0%          | 0 px           | 0.000%        |

---

## 2. Statistical Paired Differences (Expanded Cohort vs Duration Control at $\tau=0.50$)

- **$\Delta$ Macro Binary Dice**: **-0.0053** [95% CI: -0.0104, -0.0016]
- **$\Delta$ Pooled Precision**: **-12.31%** [95% CI: -20.99%, -5.82%]
- **$\Delta$ Pooled Recall**: **-0.37%** [95% CI: -0.71%, -0.11%]

---

## 3. Disentangling Data Volume vs Training Duration

1. **Duration Effect (S4 1,000 updates $\rightarrow$ S5 Arm A 2,500 updates on 400 Positives)**:
   - Evaluates the effect of extended training on fixed data volume.
2. **Data Expansion Effect (S5 Arm A 400 Positives $\rightarrow$ S5 Arm B 800 Positives at 2,500 updates)**:
   - Isolates the marginal value of 2x annotated positive data when training duration is held constant at 2,500 updates.
