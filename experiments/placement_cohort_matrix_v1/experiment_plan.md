# Controlled placement dataset and architecture experiments

## Objective

Compare data selection and a small set of practical image-classification backbones for RANZCR CVC placement labels. This is a research development series; no result can be called perfect or clinical placement accuracy because labels are radiograph-level Normal/Borderline/Abnormal indicators, not catheter-specific distances to a carina.

## Frozen references and protections

- Keep all existing segmentation, tip-localization, and placement checkpoints untouched.
- Keep the locked test partition (2,284 studies) closed. It is not read by training, checkpoint selection, threshold selection, or this pilot evaluator.
- Inherit the existing patient and duplicate-group split. Validation and test are unexposed; training can contain studies used in earlier development, which is recorded in the manifests.
- Run GPU jobs sequentially on the 4 GB RTX 3050.
- Use AUROC/AUPRC and Brier score as primary screening metrics; do not select a model based on training performance. Report per-class and trace-count strata.

## Queued experiment sequence

1. Finish the full-cohort EfficientNet-B0 multi-label reference (currently active, 2,500 updates).
2. Train the strict single-annotated-trace, exclusive-label three-class EfficientNet-B0 pilot (already queued; 1,000 updates). This is paper-inspired, but an annotation count of one does not prove the image contains only one catheter.
3. Train EfficientNet-B0 multi-label on the subset with at least one recorded CVC trace (1,000 updates). The three labels remain multi-label. Compare this checkpoint with the full-cohort baseline on the *same complete locked validation rows* and on trace-count strata.
4. Train ResNet-18 multi-label on the full locked training cohort (1,000 updates). Compare on the same validation rows to screen a different backbone under the same task and budget.

## Interpretation gates

- Dataset filtering is useful only if traced-only training improves validation discrimination/calibration on comparable validation subsets without harming the relevant unseen strata. A score computed on its smaller in-cohort validation set is not directly comparable to a score from all validation images.
- An architecture advances only if its same-task, same-validation comparison improves class-wise and macro metrics consistently; a 1,000-update screen is not final evidence.
- Confirm a promising choice with matched longer training and a second seed. Calibrate thresholds on validation, lock them, then evaluate the test partition once.
- The binary single-CVC cohort is available for a later “Normal vs Borderline/Abnormal” task. That is a different target from the multi-label classifier and should be reported separately.

## Known limits

- RANZCR image-level labels can co-occur and disagree across individual catheters in multi-CVC studies; they cannot support catheter-specific placement decisions by themselves.
- “One annotated trace” can still omit a second visible catheter.
- The full cohort includes many CVC-positive studies without CVC polylines. They remain valid for image-level classification but cannot supervise CVC segmentation.
- No millimetre distance or carina location is inferred from these labels.
