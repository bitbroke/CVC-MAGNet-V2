# Constrained recall pilot — 2026-10-18

Both arms completed 250 successful updates on an identical image/augmentation ledger, initialized from the frozen HRNet T1 baseline. Only segmentation-head parameters changed. Shared encoder, fusion and tip-head tensors remained bitwise unchanged. Both arms used BCE + Dice + 0.25 instance recall loss. The constrained arm additionally used 0.25 hard-background penalty and 0.25 reference-foreground retention penalty. No labels were promoted or thresholds tuned.

| Model | Missed paths /248 | Coverage @10px | Dice | Mean-study precision | Negative false area | Tips @15px /41 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 8 | 87.93% | 0.4625 | 34.03% | 0.06105% | 14 |
| control | 7 | 89.96% | 0.3902 | 25.31% | 0.12770% | 14 |
| constrained | 10 | 86.32% | 0.4777 | 35.63% | 0.05366% | 14 |

**Locked all-criteria decision: FAIL. No checkpoint promoted.**

- fewer_misses_than_baseline: FAIL
- fewer_misses_than_control: FAIL
- dice_preserved: PASS
- precision_preserved: PASS
- coverage_preserved: FAIL
- negative_false_area_preserved: PASS
- tip_metrics_identical: PASS

Paired study-mean coverage difference versus control: -3.44 percentage points; patient-cluster bootstrap 95% interval [-4.19, -2.76].

Dice is positive-study macro Dice (185 studies). Coverage is catheter-weighted (248 paths). Missed means under 30% path coverage at 10 native pixels, not a tip-localization failure. Precision uses the inherited mean-study metric including negative-study conventions. Negative false area is the mean across only eight negatives.

The 41 tip targets remain provisional, including 16 unverified endpoints. Exact preservation verifies no tip regression on this model path; it does not establish clinical correctness. Freezing inherited weights does not remove their historical label or development-selection exposure. No claim of absent overfitting, clinical safety, millimeter accuracy or perfect-model performance is supported.

Next: retain the baseline if any locked criterion fails. If all pass, repeat the fixed experiment on a second seed and a larger independently reviewed evaluation cohort before promotion. Do not extend this configuration unchanged merely to seek a better development score.

## Interpretation and disposition
The constrained objective improved mask Dice and mean-study precision relative to the frozen baseline while preserving tip predictions exactly, but increased missed paths from 8 to 10 and reduced mean path coverage by 1.60 percentage points. This is not an overall upgrade. The control recovered one additional path but degraded masks. These results support a precision/coverage trade-off in this restricted segmentation-head experiment; they do not prove a universal architecture limitation or establish the cause of each individual failure. Keep all checkpoints as experimental evidence; do not replace the baseline or extend this configuration unchanged.

For the next investigation, identify which individual paths were recovered or lost using paired native predictions, then test a targeted feature or resolution change on training-only difficult cases while retaining the exact tip-preservation constraint. Avoid another broad scalar-weight sweep on this same development set. No additional training was launched.


## Visual artifacts
Four comparison panels and twelve binary masks are in overlays/. The multi-CVC panel repeats low_coverage_2 because it also meets the baseline-selected multi-CVC criterion (three distinct studies, not four). Inspection of that panel shows the unsegmented companion path remains absent in all three models; narrowing the visible predicted path does not recover it. This is an illustrative case, not a cohort-wide causal diagnosis. training_components.png contains all four recorded objective components.
