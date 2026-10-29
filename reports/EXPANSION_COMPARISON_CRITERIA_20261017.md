# Predefined Comparison Criteria: S5 Cohort Expansion Benchmark

**Date**: 2026-10-17  
**Scope**: Locked comparison criteria prior to evaluating the 2,500-update comparison (Arm A Duration Control vs Arm B Expanded Cohort).

---

## 1. Experimental Invariants Held Constant

Both arms share:
- **Initialization**: Frozen S3 reference baseline (`runs/s3_experiment_hard_neg/checkpoints/final_segmentation_v3.pt`, `SHA256: 6520e326...`).
- **Update Budget**: Exactly 2,500 successful optimizer updates.
- **Optimization**: AdamW, initial LR 1e-4 with cosine decay to 1e-6, effective batch size 4 (micro-batch 2, accumulation 2).
- **Mixed Precision**: Automatic Mixed Precision (AMP) with in-place same-batch retry upon non-finite gradients.
- **Negative Sampling**: Exactly identical 50% Top-99 hard negative + 50% uniform negative policy drawn from the 396 confirmed negatives.
- **Native Geometry**: 768px letterbox input, stroke thickness 10px, boundary-clipped coordinates, evaluation threshold $\tau=0.50$.
- **Validation Isolation**: Exact zero study and zero patient overlap with `dev_pilot_val` (193 studies, 185 positive, 8 negative, 132 patients).

---

## 2. Experimental Variable

- **Arm A (Duration Control Baseline)**: 400 annotated positive studies + 396 confirmed negatives (796 studies).
- **Arm B (Expanded Cohort)**: 800 annotated positive studies + 396 confirmed negatives (1,196 studies).

---

## 3. Predefined Decision Rules & Acceptance Criteria

For cohort expansion to be judged successful and justify scaling further:

1. **Catheter Recall & Centerline Coverage Preservation**:
   - Point estimate recall in Arm B must not drop more than **1.0 percentage points** compared to Arm A (and must satisfy pooled recall $\ge 72.59\%$).
   - Centerline coverage @ 10px must remain **$\ge 79.0\%$**.
   - Missed catheter rate @ 10px must remain **$\le 5.0\%$**.
2. **False Foreground Suppression**:
   - Negative false foreground area must remain **$\le 0.100\%$** of valid lung field area (no regression in specificity).
3. **Precision & Segmentation Dice**:
   - Precision in Arm B must match or exceed Arm A (Duration Control).
   - Binary Dice in Arm B must match or exceed Arm A.
4. **Statistical Reporting Nuance**:
   - Report paired patient-cluster bootstrap confidence intervals (N=132 patients) for $\Delta$ Dice, $\Delta$ Precision, and $\Delta$ Recall.
   - Note: Absolute recall confidence interval width reflects finite validation cohort size ($N=185$ positives); paired differences against matched control isolate the data expansion effect.

---

## 4. Decision Outcomes

- **Pass (Data Expansion Validated)**: Arm B satisfies all coverage criteria while improving or matching Dice/precision and false-mask suppression over Arm A. Justifies scaling to full-dataset training.
- **Equivocal (Duration Dominant)**: Arm B matches Arm A within measurement noise without meaningful gain. Indicates current data volume is not yet the primary bottleneck at 2,500 updates.
- **Fail (Coverage Loss or False Foreground Regression)**: Arm B shows $> 1.0\%$ recall drop or regression in false mask area. Halts further expansion to re-examine annotation stratification.
