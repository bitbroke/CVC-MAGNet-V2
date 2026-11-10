# Boundary precision experiment — 2026-10-22

The previous segmentation and tip models remain frozen. All new outputs belong in this directory. No candidate is automatically promoted.

## Registered comparison

Both arms start from `../hrnet_segmentation_features_v1/comparison/feature_adaptation/last.pt`. Both use the previously frozen hard-positive batch ledger, seed 42, 768-pixel input, effective batch 4 with microbatch 1, 250 successful optimizer updates, head LR 1e-5 and feature LR 2e-6. BatchNorm buffers and tip-head parameters stay frozen. Actual tip predictions use the entire separate original tip model.

Shared losses: BCE positive weight 10, Dice, 0.25 instance recall, and 0.5 foreground retention against the frozen starting model. Retention acts only on annotated foreground with teacher probability >=0.5 and allows a 0.02 probability margin.

The experimental arm alone adds 0.25 mean softplus(logit) on a band at distance >1 and <=8 model pixels outside annotated masks. This penalizes widening near annotations while leaving a one-pixel tolerance band. Negative studies retain their standard background supervision; no artificial boundary ring is constructed for an empty mask. The training-only diagnostic measures near-boundary versus distant false pixels before training.

## Fixed evaluation

193 studies /248 paths, threshold 0.50, native-grid evaluator. Report Dice, mean-study precision, continuous path coverage, longest gaps, missed paths, mean negative false area and total negative false pixels separately.

Every one of the four previously recovered paths must retain coverage within 1 percentage point and longest gap within 10 native pixels of the parent. There must be no new missed path relative to the parent. Overall coverage may fall at most 0.5 percentage points, mean longest gap may increase at most 10 pixels, and neither negative false-area metric may worsen. Dice and precision must improve over parent and matched control. Restoring original-baseline Dice and precision is an additional requirement before considering promotion.

These are development checks against rasterized polylines, not clinically attested catheter boundaries or an independent external test. Eight negative studies do not establish clinical specificity.

## Storage prerequisite

At implementation time C: had only about 0.92 GB free. Full training must not launch until at least 3 GB is available (5 GB preferred) or a separate output drive is provided. Existing data and checkpoints must not be deleted to make room without explicit user direction. Lightweight contract tests and the two-update smoke test are separate from the full comparison.

Implementation: `../hrnet_recovery_v3/run_boundary_pilot.py`.
Tests: `../hrnet_recovery_v3/test_boundary_pilot.py`.
