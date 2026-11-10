# Lost and recovered CVC paths: frozen-model audit

Date: 2026-10-18. Inference and inspection only; no training, checkpoint changes, annotation edits, or new operating-threshold selection.

## What was checked

The study-mean coverage and number of CVC instances give a conservative bound: a study with n paths cannot contain a path below 30% coverage if its mean is at least (n-1+0.3)/n. Applying this to all three arms selected 11 studies containing 19 CVC instances. Every selected study's recomputed mean exactly agrees with the earlier native evaluator (tolerance 1e-7). The recovered per-instance totals reproduce all 248-path benchmark misses: baseline 8, control 7, constrained 10. No potentially missed paths were discarded by looking only at low study means.

All 19 audited polylines exactly match a unique CVC-labeled polyline in the original train_annotations.csv. This rules out a local inventory join/device-index mismatch in this audited subset. It does NOT certify the clinical correctness, completeness or fine pixel alignment of those source annotations.

All ten distinct paths missed by at least one model were visually inspected in full-image and cropped comparison panels. The interpretation is an engineering review, not radiologist attestation. The three checkpoint hashes were unchanged after inference.

## Findings

1. **The constrained model recovered none of the eight original misses and lost two previously counted paths.** All eight original misses remain missed; the two added misses explain the total of ten.
2. **The control's one counted recovery is not a complete-path recovery.** Case 03 moves from 29.03% to 30.81% coverage, just crossing the arbitrary 30% missed-path cutoff. The overlay remains a short proximal fragment; the long continuation is still absent. Report the continuous coverage, not only the binary recovered count.
3. **One constrained loss is a boundary flip; the other is substantial suppression.** Case 05 falls from 30.79% to 29.38%; a small change flips its status. Case 09 falls from 48.45% to 24.05%, with visibly shorter, separated mask fragments. At threshold 0.30 these become 30.51% and 43.99%, respectively. Lowering a threshold partly restores their former coverage, but does not restore complete paths and has not been validated for false positives across the cohort.
4. **Persistent failures concentrate in companion CVCs.** Six of eight persistent misses occur in two-CVC images (cases 01, 02, 04, 06, 07, 08). The model generally retains a different catheter/fragment while much of the selected path is absent. This is a repeated observable output pattern; it does not prove an internal attention mechanism or that the encoder contains no usable information.
5. **A global threshold is not the main solution.** All eight persistent misses remain below 30% coverage even at threshold 0.20. Their constrained-model median probabilities sampled on the source polyline lie between 0.0747 and 0.0871. Cases 02, 07, 08 and 10 have zero coverage at threshold 0.50. Case 08 remains zero at 0.20. These are extended low-confidence/absent paths, not solely thin masks sitting just below 0.50.
6. **The constraints suppress weak fragments as well as false foreground.** None of the ten audited target paths improves over baseline at the locked threshold: eight decrease and two remain at zero. The constrained penalties are a package, so the audit cannot attribute this separately to the background penalty versus retention penalty. Visually, masks tend to narrow or lose fragments rather than acquire a missing trajectory.
7. **Resolution and feature adaptation remain plausible bottlenecks, not proven diagnoses.** These images are reduced to approximately 25–31% of native dimensions. The 10-native-pixel reference ribbon is therefore only about 2.5–3.1 model pixels wide. Several missed trajectories are faint or lie among overlapping structures in the panels. The latest experiment only trained 23,153 segmentation-head parameters; its encoder and fusion were frozen. It could reshape existing output confidence but did not test learning improved shared features for difficult companion lines.
8. **Coverage does not establish catheter identity.** The evaluator measures distance to the union foreground. Near crossings or closely parallel lines, foreground from a companion catheter can contribute to the selected path's coverage. This is a useful geometric metric, but not an instance-association guarantee. Likewise a path above 30% coverage may remain unusable for tracing or tip localization.
9. **Embedded black-frame false positives are visible.** Cases 02 and 06 show some predictions on the radiograph's native black border. Existing valid-area handling excludes added letterbox padding, not all embedded black image regions. This is a separate artifact failure; removing it alone would not recover the missing catheter. Do not quietly exclude these pixels from the benchmark to inflate scores.

## Case-by-case inspection

Coverage is the percentage of sampled source-polyline points within 10 native pixels of foreground at probability threshold 0.50. All statuses below use the existing 30% cutoff.

| Panel | CVCs in study | Baseline | Control | Constrained | Observed pattern |
|---|---:|---:|---:|---:|---|
| 01 | 2 | 9.51% | 10.90% | 8.82% | Mostly absent selected neck-to-central path while companion is segmented; crossing may contribute coverage. |
| 02 | 2 | 0.00% | 0.00% | 0.00% | Selected upper-neck path absent; another path segmented; multiple devices and native black-border artifacts present. |
| 03 | 1 | 29.03% | 30.81% | 22.26% | Short proximal fragment retained; long continuation absent. Control's recovery is only a cutoff crossing. |
| 04 | 2 | 17.31% | 19.26% | 13.96% | Companion path dominates; selected long lateral trajectory is largely absent. |
| 05 | 2 | 30.79% | 33.05% | 29.38% | Partially aligned, incomplete neck segment; small narrowing crosses the missed cutoff. Pixel alignment merits targeted review before assigning a cause. |
| 06 | 2 | 17.65% | 20.72% | 13.24% | One lateral path is segmented; much of the other remains absent; native border false foreground also visible. |
| 07 | 2 | 2.09% | 12.98% | 0.00% | Very limited fragments along the selected long arch; constrained model suppresses them. |
| 08 | 2 | 0.00% | 0.00% | 0.00% | Complete omission of selected companion path despite segmentation of the other. |
| 09 | 1 | 48.45% | 64.60% | 24.05% | Neck-region line becomes short separated fragments under constraints; clear coverage regression. |
| 10 | 1 | 1.95% | 18.60% | 0.00% | Baseline nearly silent; control adds a short arch fragment; constrained model removes it. |

## What to fix next, in order

1. **Fix the interpretation and retained evaluation artifacts first.** Keep continuous per-instance coverage, longest uncovered segment, fragmentation and full-path completeness beside the old missed count. Keep the old metric unchanged for comparability. A 29% to 31% change must not be described as recovering a catheter. Preserve all target identities and one-to-one tip matching. For crossings, flag ambiguous union-mask coverage rather than silently treating it as correct association.
2. **Build a training-only difficult-positive inventory.** Use the frozen model to score each annotated TRAINING catheter, then identify absent companion lines, long unsupported segments, faint lines, and fragmented positives. Include every annotated CVC in multi-line crops and retain representative negatives. Do not put these eleven validation studies or their patients into training/mining. Check exact original polyline correspondence and native-image alignment for selected training examples. No need to relabel the entire dataset or declare the whole dataset defective.
3. **Stop trying to solve absent paths with broader masks alone.** Do not extend this constrained run or run another broad positive-weight sweep. The current modifications predominantly exchange precision for coverage. Prioritize per-path evidence and continuity rather than increasing union-foreground area.
4. **Test a representation/resolution change next, with tip protection maintained.** First inspect training-only failed lines at native resolution versus the 768 input to assess how much visible detail is lost. The next controlled training candidate should permit a SEGMENTATION copy's encoder/fusion to adapt at a low learning rate, using difficult-positive crops or higher-resolution overlapping tiles alongside full-image context. Compare against the current frozen-feature control with the same data exposure and budget; do not change both resolution and trainable layers in one purported single-factor test. Start with one factor and profile on the 4GB GPU.
5. **Keep the original full tip predictor frozen as a separate branch if segmentation features are unfrozen.** Freezing only the tip head does not preserve its output when its input features change. A separate frozen tip model preserves the current reference behavior, while segmentation is improved independently. This increases inference cost and must be profiled. Tip improvement remains a later task; the provisional benchmark is not clinical truth.
6. **Avoid candidate-dependent crop coverage failures.** Ground-truth paths may select training crops, but validation/inference crops must use image-only fixed overlapping tiling or a separately evaluated proposal mechanism. Do not assume a missed line will generate its own crop. Report all-target coverage and inference cost.
7. **Treat the two threshold-sensitive losses separately from the eight persistent omissions.** A later operating-point study needs false-positive and completeness metrics on the entire fixed cohort. The diagnostic 0.20/0.30 values here are not a recommendation to lower the deployed threshold.

No new training was launched. Baseline, control and constrained models remain frozen. This audit identifies observable failures and prioritizes the next controlled test; it does not establish a unique causal explanation for every radiograph.

## Saved evidence

- `instance_audit.csv`: all 19 instances in the conservative audit subset.
- `missed_union.csv`: all ten paths missed by at least one arm, including threshold-sensitivity diagnostics.
- `original_annotation_checks.csv`: exact matches to original CVC source polylines.
- `panels/`: ten full-image plus target-crop comparisons, all visually inspected.
- `point_traces/`: sampled coordinates, probabilities and foreground distances within the evaluation ROI. Infinite distance means no foreground within that ROI, not necessarily none elsewhere in the image.
- `probabilities/`: frozen FP32 768-grid prediction maps for independent re-analysis.
- `summary.json` and `provenance.json`: reconciled counts, source hashes and selection rule.
