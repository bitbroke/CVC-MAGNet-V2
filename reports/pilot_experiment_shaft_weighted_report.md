# Matched Pilot Report: Controlled Shaft-Weighting Experiment (Arm C)

**Date**: 2026-10-16  
**Artifact**: [runs/pilot_experiment_shaft_weighted/checkpoints/final_coarse_dual_head.pt](file:///C:/Ankshit/CVC-MAGNet-V2/runs/pilot_experiment_shaft_weighted/checkpoints/final_coarse_dual_head.pt)  
**SHA-256 Digest**: `3cbc44789f04c282fba1086d3b4aacff594ef4d2c00f68eb75613ca3aeee9eda`  
**Raw Evaluation Data**: [reports/pilot_three_arms_comparison.json](file:///C:/Ankshit/CVC-MAGNet-V2/reports/pilot_three_arms_comparison.json)

---

## 1. Mathematical Formulation & Correction

### The Bug in Naive Shaft Scaling
In standard weighted binary cross-entropy, logit derivatives are given by:
$$\frac{\partial L}{\partial z} = (1 - y) \sigma(z) - w_{\text{pos}} \cdot y (1 - \sigma(z))$$
Multiplying the entire per-pixel loss by $w_{\text{shaft}} = 2.0$ on pixels where $y \le 0.01$ inadvertently doubled the negative component $-w_{\text{pos}} \cdot y (1 - \sigma(z))$ whenever $y > 0$. At $\sigma(z) = 0.3865$ and $y = 0.01$, the unweighted derivative was $-0.2309$ (pulling logits UP); multiplying by 2 deepened this upward pull to $-0.4617$.

### The Corrected Negative-Only Shaft Penalty
We implemented the corrected formulation:
$$\text{extra} = (w_{\text{shaft}} - 1.0) \cdot M_{\text{shaft}} \cdot (1 - y) \cdot \text{softplus}(z)$$
where $M_{\text{shaft}} = (M_{\text{seg}} > 0.5) \land (Y_{\text{tip}} \le \text{thresh})$.

Its derivative with respect to logit $z$ is:
$$\frac{\partial \text{extra}}{\partial z} = (w_{\text{shaft}} - 1.0) \cdot M_{\text{shaft}} \cdot (1 - y) \cdot \sigma(z) \ge 0$$
This is strictly non-negative everywhere, ensuring that:
1. True zeros on the shaft ($y=0$) receive an exact doubled downward gradient: $\sigma(z) + (2 - 1)\sigma(z) = 2\sigma(z)$.
2. Fractional Gaussian tails ($0 < y \le 0.01$) receive only an additive downward push; the positive-target term $-w_{\text{pos}} \cdot y (1 - \sigma(z))$ is untouched.
3. Protected tip neighborhoods ($Y_{\text{tip}} > 0.01$) have $M_{\text{shaft}} = 0$ and receive zero extra penalty.
4. Masked studies (`tip_mask = 0`) and spatial ignore masks (`tip_spatial_mask = 0`) contribute zero penalty.
5. The loss denominator remains strictly `valid_pixels`.

Verified by unit tests in [tests/test_coarse_dual.py](file:///C:/Ankshit/CVC-MAGNet-V2/tests/test_coarse_dual.py) (11/11 tests pass).

---

## 2. Experimental Controls

| Dimension | Arm A (Control) | Arm B (Expanded Baseline) | Arm C (Shaft-Weighted Pilot) | Match Status |
| :--- | :--- | :--- | :--- | :--- |
| **Backbone Init** | `best_coarse.pt` (`13aa7eb75e869c94`) | `best_coarse.pt` (`13aa7eb75e869c94`) | `best_coarse.pt` (`13aa7eb75e869c94`) | Bitwise Identical |
| **Training Manifest** | `reviewed_distal_tips_v2.parquet` | `training_expansion_manifest.parquet` | `training_expansion_manifest.parquet` | Identical to Arm B |
| **Active Positive Studies** | 10 studies (10 tips) | 290 studies (339 tips) | 290 studies (339 tips) | Identical to Arm B |
| **Random Seed / Sampler** | Seed 42, Uniform | Seed 42, Uniform | Seed 42, Uniform | Identical |
| **Successful Updates** | 1,000 updates | 1,000 updates | 1,000 updates | Identical |
| **Actual Image Exposures** | Pos: 50, Neg: 1988, Unk: 1966 | Pos: 1461, Neg: 1988, Unk: 555 | Pos: 1461, Neg: 1988, Unk: 555 | **Bitwise Identical to Arm B** |
| **Experimental Variable** | None | Supervision Expansion | **`tip_shaft_weight = 2.0`** | **Strictly Isolated** |

---

## 3. Standardized Comparative Evaluation (Step 1000 Checkpoints)

All evaluations use the unified reference decoder (`min_distance=10`, Hungarian radius $15.0$ px) evaluated across all 193 validation images:

| Evaluation Metric | Arm A (Reviewed Control) | Arm B (Expanded Baseline, 1x) | Arm C (Shaft-Weighted, 2x) | Arm C vs Arm B Delta |
| :--- | :--- | :--- | :--- | :--- |
| **Tip-Head Success@15 (248 Registry)** | 1 / 248 (0.40%) | 33 / 248 (13.31%) | **40 / 248 (16.13%)** | **+7 targets (+2.82%)** |
| **Tip-Head Success@15 (25 Audited Subset)** | 0 / 25 (0.00%) | 1 / 25 (4.00%) | **3 / 25 (12.00%)** | **+2 targets (+8.00%)** |
| **Segmentation Dice** | 0.1285 | 0.1387 | **0.1433** | **+0.0046** |
| **False Peaks / Neg Study (Capped 10, $\tau=0.30$)** | 10.00 (80 total) | 10.00 (80 total) | 10.00 (80 total) | Capped at limit |
| **False Peaks / Neg Study (Uncapped, $\tau=0.50$)** | 0.62 (5 total) | 3.62 (29 total) | **1.62 (13 total)** | **-55% False Peaks** |
| **False Peaks / Neg Study (Uncapped, $\tau=0.30$)** | 331.25 (2650 total) | 233.25 (1866 total) | **225.38 (1803 total)** | -3.4% False Peaks |
| **Uncapped Success@15 ($\tau=0.30$, `max_peaks=None`)** | 54 / 248 (21.77%) | 58 / 248 (23.39%) | **63 / 248 (25.40%)** | **+5 targets** |
| **Uncapped Success@15 on 25 ($\tau=0.30$)** | 7 / 25 (28.00%) | 4 / 25 (16.00%) | **8 / 25 (32.00%)** | **+4 targets** |
| **Seg Skeleton Endpoints Success@15 (248)** | **35 / 248 (14.11%)** | 24 / 248 (9.68%) | 23 / 248 (9.27%) | Stable |

---

## 4. Key Diagnostic Observations

1. **Surpasses Control Skeleton Endpoints for the First Time**:
   - In Arm B, the tip head trailed the control's segmentation skeleton (33 vs 35 targets).
   - In Arm C, the tip head achieves **40 / 248 (16.13%)**, surpassing the control's skeleton endpoints (35/248, 14.11%) under the exact reference decoder contract.
2. **Tripling on the Audited 25-Case Cohort**:
   - Success@15 rose from **1 / 25 (4%) to 3 / 25 (12%)**.
   - Suppressing the shaft allowed true tip peaks that were previously ranked below shaft peaks to enter the top-10 list.
3. **Halving High-Confidence False Positives on Negatives**:
   - At $\tau=0.50$, uncapped false peaks dropped from 3.62 to **1.62 per negative study** (13 total across all 8 negative images vs 29 in Arm B).
4. **Segmentation Synergy**:
   - Segmentation Dice improved to **0.1433**, demonstrating that penalizing the tip head on catheter shafts does not hurt segmentation feature learning.
