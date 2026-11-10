# Frozen-reference segmentation feature experiment

Authorized on 2026-10-18. This is a bounded development experiment, not final model training.

All existing experiment checkpoints are hashed before and after execution. All new outputs live here. The production segmentation and tip models are not replaced.

## Fixed comparison

- Source: frozen HRNet T1 seed-42 checkpoint.
- Input: 768 pixels, unchanged preprocessing and geometry.
- Two sequential arms: segmentation-head adaptation versus segmentation encoder/projections/fusion/head adaptation.
- 250 successful optimizer updates per arm, seed 42, microbatch 1, effective batch 4.
- Head LR 5e-5 in both arms; newly trainable features LR 1e-5. BatchNorm statistics remain frozen.
- Same BCE (positive weight 10), Dice and 0.25 equal-instance recall loss in both arms. The prior additional background/retention penalties are absent from both arms.
- Training-only hard-positive mining: lowest coverage quartile or longest gap >=100 native pixels. Replace 50% of positive slots from this pool in a single frozen ledger shared by both arms. Negative slots stay unchanged.
- The effect isolated by this comparison is feature adaptation on that shared sampling policy, not the separate benefit of hard-positive sampling versus uniform sampling.

## Protection of tips

The adapted model is a segmentation-only candidate. Its internal tip predictions are invalid for deployment even though its tip-head parameters are frozen, because shared features can change. Tip predictions must continue to come from the complete original frozen reference model through the separate inference adapter.

## Evaluation

193 development studies, 248 annotated CVC paths, native-grid threshold 0.50. Report canonical mask metrics plus continuous individual path coverage, longest missing visible segment, gap runs, >=90% coverage and complete coverage. Union-mask proximity is not proof of correct catheter-instance association.

Progression requires fewer misses than baseline and control while preserving baseline Dice, mean-study precision, coverage, negative false area and mean longest gap. No checkpoint is automatically promoted. This repeatedly used development cohort and its eight negative studies do not support independent clinical accuracy claims. Tip annotations remain provisional.

## Files

- `smoke/`: completed two-update GPU checks for both arms; originals unchanged.
- `comparison.log`: persistent main log.
- `comparison/status.json`: live lifecycle and update state.
- `comparison/training_path_inventory.csv`, `training_hard_paths.csv`: training-only mining evidence.
- `comparison/matched_batch_ledger.json`: identical exposures and augmentation parameters.
- `comparison/*_paths.csv`, `path_transitions.csv`: individual validation paths.
- `comparison/*_probabilities/`: saved probability maps for independent re-analysis.
- `comparison/RESULTS.md`, `decision.json`: locked decision, with no model promotion.
- `comparison/overlays/`: all paths missed by baseline or feature-adapted model.

Run implementation: `../hrnet_recovery_v3/run_feature_pilot.py`; contract tests: `../hrnet_recovery_v3/test_feature_pilot.py` (3 passed).
