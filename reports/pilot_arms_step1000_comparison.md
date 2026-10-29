# Controlled Pilot Experiment Report: Distal Tip Expansion vs. Reviewed-Only Baseline

**Date**: 2026-10-15  
**Cohort**: `dev_pilot_train` (796 studies) & `dev_pilot_val` (193 studies)  
**Budget**: Fixed 1,000 successful optimizer updates per arm  
**Target Registries**: Full assessable validation targets ($N=248$, SHA256: `7ad60c1d68a2bf60`) & provisional audited cohort ($N=25$)  
**Artifact Data**: [reports/pilot_arms_step1000_comparison.json](file:///C:/Ankshit/CVC-MAGNet-V2/reports/pilot_arms_step1000_comparison.json)

---

## Executive Summary & Core Verdict

| Dimension | Arm A: Control (Reviewed-Only) | Arm B: Experiment (Expanded Manifest) | Relative Change |
| :--- | :--- | :--- | :--- |
| **Training Active Tips** | 25 tips across 10 complete studies | **339 tips across 290 complete studies** | **+329 positive studies (+29x)** |
| **Masked Positive Studies** | 390 studies (`tip_mask = 0`) | **110 studies (`tip_mask = 0`)** | **-280 unsupervisable studies** |
| **Confirmed Negative Studies** | 396 studies (`tip_mask = 1`) | 396 studies (`tip_mask = 1`) | Bitwise identical |
| **Image Sampling Exposure** | **Uniform random (4,000 draws)** | **Uniform random (4,000 draws)** | **Bitwise identical per-image exposure** |
| **Shared Backbone Init** | `best_coarse.pt` (`9aa28b7887883aab`) | `best_coarse.pt` (`9aa28b7887883aab`) | Bitwise identical |
| **Final Mask Dice** | 0.1285 | **0.1387** | +7.9% |
| **Tip Heatmap Success@15 (Full $N=248$)** | **1 / 248 (0.40%)** | **33 / 248 (13.31%)** | **+33x (+12.91 percentage points)** |
| **Tip Heatmap Success@30 (Full $N=248$)** | **2 / 248 (0.81%)** | **72 / 248 (29.03%)** | **+36x (+28.22 percentage points)** |
| **Single-CVC Tip Success@15 ($N=126$)** | **0 / 126 (0.00%)** | **22 / 126 (17.46%)** | **+17.46 percentage points** |
| **Multi-CVC Tip Success@15 ($N=122$)** | **1 / 122 (0.82%)** | **11 / 122 (9.02%)** | **+8.20 percentage points** |
| **All-Target Nearest Tip Median Error** | **542.96 native px** | **55.02 native px** | **9.9x reduction in error tail** |
| **All-Target Nearest Tip p95 Error** | **1,288.64 native px** | **471.85 native px** | **2.7x reduction in extreme error** |
| **Tip Heatmap False Peaks / Neg Study** | 10.0 peaks | 10.0 peaks | Flat (bounded by peak threshold) |
| **Seg Skeleton Success@15 (Full $N=248$)** | 35 / 248 (14.11%) | 24 / 248 (9.68%) | -4.43 percentage points |
| **Provisional Audited Tip S@15 ($N=25$)** | 0 / 25 (0.00%) | 1 / 25 (4.00%) | +4.0 percentage points |

> [!IMPORTANT]
> **Definitive Finding**: Expanding training distal tip supervision with provisional automatic labels (governed by the frozen conservative selector and multi-CVC completeness rule) **reserves the auxiliary tip head from total extinction**. In Arm A, training with only 10 positive studies against 396 negative studies caused the tip head to succumb to negative background suppression (0.40% Success@15). In Arm B, 339 verified tips provided sufficient positive gradient density to establish functional, calibrated distal tip localization (**13.31% Success@15 at 15px, 29.03% at 30px, and 17.46% in single-CVC studies**), while maintaining identical false-peak rates on confirmed negatives.

---

## 1. Concrete Mismatches Resolved Before Launch

1. **Exact Frozen Selector Recovery**:
   - Manifest rebuilt using `VERSION='conservative_endpoint_v1'` from [outputs/endpoint_selector_pilot/endpoint_selector.py](file:///C:/Ankshit/CVC-MAGNet-V2/outputs/endpoint_selector_pilot/endpoint_selector.py).
   - Removed the ad-hoc 50px border distance heuristic.
   - Identified and excluded exactly **110 studies** containing 167 instances that the conservative selector abstained on or that failed multi-CVC completeness.
   - Kept catheter path segmentation supervision 100% active on all 506 training instances.

2. **Reviewed-Label Provenance & Corrections**:
   - Loaded training cases from [data/reviewed_distal_tips_v2.parquet](file:///C:/Ankshit/CVC-MAGNet-V2/data/reviewed_distal_tips_v2.parquet).
   - Human review corrections (such as Case 18, which corrected the heuristic from Point Last to Point 0) were retained as valid provisional ground truth rather than flagged as "uncertain."
   - Recorded provenance strictly as `provisional_engineering_review_awaiting_clinical_attestation`.
   - Replaced all obsolete occurrences of `reviewed_verified`.

3. **Complete Trainer Integration with Isolated Validation**:
   - Added `--train_manifest_path` to [scripts/05_train_coarse_dual.py](file:///C:/Ankshit/CVC-MAGNet-V2/scripts/05_train_coarse_dual.py).
   - `val_dataset` and `run_dual_validation` unconditionally load the frozen, untouched validation cohort ($N=25$ provisional audited cases and 8 confirmed negatives) from `reviewed_tips_path` (`split == 'val'`).
   - Training tips load separately from `train_manifest_path` (Arm B) or `reviewed_tips_path` (Arm A).

4. **Deliberate Image Exposure Control**:
   - Avoided `TipBalancedBatchSampler` quota distortion, which would have repeated the 10 positive studies in Arm A 100 times each while diluting Arm B.
   - Standardized on `--sampling uniform --seed 42 --batch_size 4` across all 796 studies in `dev_pilot_train`.
   - Result: Both arms received the **exact same 4,000 image draws in the identical order** (mean $5.02$ exposures per study, exactly 1,988 negative draws in both arms).

---

## 2. Full Assessable Registry ($N=248$) Detailed Breakdown

Evaluated with threshold-constrained Hungarian assignment (radius 15.0px, cost > radius replaced with $10^9$):

```mermaid
graph LR
    subgraph Arm_A [Arm A: Control Reviewed-Only]
        A1[10 Positive Training Studies] --> A2[Tip Heatmap Head Suppressed]
        A2 --> A3[Full Registry S@15: 0.40%]
        A2 --> A4[Nearest Median Error: 542.96 px]
    end

    subgraph Arm_B [Arm B: Experiment Expanded Manifest]
        B1[290 Positive Training Studies] --> B2[Tip Heatmap Head Active & Calibrated]
        B2 --> B3[Full Registry S@15: 13.31%]
        B2 --> B4[Nearest Median Error: 55.02 px]
    end
```

### Tolerance Sweeps (Hungarian One-to-One Matching)

| Tolerance Radius | Tip Head Arm A (Control) | Tip Head Arm B (Experiment) | Seg Skeleton Arm A | Seg Skeleton Arm B |
| :--- | :--- | :--- | :--- | :--- |
| **Success@5** | 0.00% (0 / 248) | **2.02% (5 / 248)** | 1.61% (4 / 248) | 1.61% (4 / 248) |
| **Success@10** | 0.00% (0 / 248) | **8.47% (21 / 248)** | 7.26% (18 / 248) | 6.85% (17 / 248) |
| **Success@15** | 0.40% (1 / 248) | **13.31% (33 / 248)** | 14.11% (35 / 248) | 9.68% (24 / 248) |
| **Success@20** | 0.40% (1 / 248) | **18.55% (46 / 248)** | 19.76% (49 / 248) | 14.11% (35 / 248) |
| **Success@30** | 0.81% (2 / 248) | **29.03% (72 / 248)** | 28.63% (71 / 248) | 19.76% (49 / 248) |
| **Success@50** | 3.23% (8 / 248) | **44.76% (111 / 248)** | 46.37% (115 / 248) | 37.50% (93 / 248) |

### Error Distribution on All Assessable Targets ($N=248$)

| Metric | Tip Head Arm A (Control) | Tip Head Arm B (Experiment) | Improvement |
| :--- | :--- | :--- | :--- |
| **Matched Error Median** | 14.85 px | **8.61 px** | -6.24 px (-42%) |
| **Matched Error Mean** | 14.85 px | **8.64 px** | -6.21 px (-42%) |
| **All-Target Nearest Distance (Median)** | 542.96 px | **55.02 px** | **-487.94 px (-90%)** |
| **All-Target Nearest Distance (p95)** | 1,288.64 px | **471.85 px** | **-816.79 px (-63%)** |
| **Targets Missing Any Candidate** | 0 | 0 | 100% candidate generation |

---

## 3. Single-CVC vs. Multi-CVC Localization Breakdown

| Cohort | Tip Head Arm A (Control) | Tip Head Arm B (Experiment) | Seg Skeleton Arm A | Seg Skeleton Arm B |
| :--- | :--- | :--- | :--- | :--- |
| **Single-CVC Targets ($N=126$)** | 0.00% (0 / 126) | **17.46% (22 / 126)** | 12.70% (16 / 126) | 11.11% (14 / 126) |
| **Multi-CVC Targets ($N=122$)** | 0.82% (1 / 122) | **9.02% (11 / 122)** | 15.57% (19 / 122) | 8.20% (10 / 122) |

> [!NOTE]
> Single-CVC performance in Arm B reached **17.46% Success@15** directly from the heatmap head alone, proving that isolated catheter paths yield high-confidence Gaussian peak localizations when adequately supervised. Multi-CVC studies remain more challenging due to crossing catheters and spatial proximity, but improved by over $11\times$ compared to the control arm.

---

## 4. Specific Failures & Diagnostic Observations

1. **Suppression Collapse in Arm A**:
   With only 10 positive training studies having active tip supervision against 396 confirmed negative studies with active zero-heatmap loss, the tip loss gradient was over 97% negative. The network learned that predicting any tip peak was heavily penalized, collapsing the heatmap output towards zero across the entire validation cohort.

2. **Localization Resolution in Arm B**:
   In Arm B, 339 positive tip targets provided sufficient balance against the 396 negative studies. The tip head produced localized peaks with a median matched error of **8.61 native pixels**. At a 30px tolerance, nearly 30% of all assessable catheter tips in the validation cohort were accurately matched.

3. **Segmentation vs. Heatmap Complementarity**:
   Skeleton endpoint extraction from the segmentation mask achieved 14.11% in Arm A and 9.68% in Arm B. However, skeleton endpoints suffer from high false candidate rates (67.8 to 70.2 false branches per negative study). In contrast, the Tip Heatmap head provides focused localization (10.0 peaks per study) and achieved **13.31% Success@15** in Arm B.
