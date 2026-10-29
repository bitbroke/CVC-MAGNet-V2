# Carina and CVC Placement Annotation Pilot

## Purpose

The local annotation workstation now supports a separate carina landmark and a reviewer-entered placement category for each inventory-listed CVC. The 50-study calibration queue is intended to establish annotation feasibility and reviewer agreement before those annotations are used as training targets. No model training was performed and no checkpoint was modified.

## Annotation fields

- **Carina state:** `visible`, `uncertain`, or `not_visible`. A native-image `(x, y)` coordinate is required only for `visible`. The reviewer must select **Mark Carina** before clicking; the application does not snap to anatomy or show model predictions.
- **CVC tip state and coordinate:** the existing assessable/uncertain/outside-image/annotation-incorrect workflow remains in place.
- **Per-CVC placement:** `normal`, `borderline`, `malpositioned`, or `uncertain`, attached to the stable catheter instance ID. Source dataset placement labels are not prefilled into the review control.
- **Reviewer and provenance:** each save is marked `provisional_engineering_review_awaiting_clinical_attestation`; separate reviewer IDs are stored independently for the dual-review queue.

The placement categories are raw reviewer judgments for calibration. Before using them to train a clinical classifier, a qualified clinical panel must agree on operational definitions (including what anatomical zone counts as normal/borderline/malpositioned) and adjudicate disagreements. These labels do not establish clinical truth by themselves.

## Queue and separation

The fixed queue is `carina_placement_pilot_50`, generated with seed 20261025 from the training partition (`dev_full_train`). It contains 50 studies, each from a distinct patient and duplicate group. The queue has zero study, patient, or duplicate-group overlap with frozen `dev_pilot_val`.

Strata are sampling descriptors: 10 multi-CVC, 10 single-CVC with other annotated lines, 10 single-CVC with low clutter, 10 border-contact or short-path difficulty proxies, and 10 additional placement-mix cases. The source manifest labels every selected image `readable`; it does not provide graded image-quality labels. Border contact and path length are only difficulty proxies, not image-quality annotations. The cohort includes 64 inventory-listed CVC instances (37 one-CVC studies, 12 two-CVC, 1 three-CVC). All 50 source image files were found.

Artifacts:

- `data/curated_cvc_v1/carina_placement_pilot_50.csv`
- `data/curated_cvc_v1/carina_placement_pilot_50.json`
- `scripts/create_carina_placement_pilot_queue.py`

The script refuses to overwrite an existing queue. Queue SHA-256 and exact study list are stored in the JSON manifest.

## Running the dual review

1. Use a distinct reviewer ID for each reviewer; do not share IDs.
2. Select **Carina + Placement Calibration (50 Training Studies)** in the queue selector.
3. Review CVC polylines while blinded to model predictions. For the carina, select **Mark Carina**, click the landmark, then set the state to `visible`; use `uncertain` or `not_visible` when appropriate.
4. Select each CVC card, assign a placement label, and complete the existing tip review state. Save drafts freely; attestation requires all CVC states, placement labels, and a carina state, plus a point if visible.
5. Each reviewer repeats all 50 studies under their own reviewer ID. Their records are stored separately and do not overwrite one another.

The updated app is running at `http://127.0.0.1:8001`. The previous server at port 8000 was left untouched so any unsaved review there is preserved. The new port runs the updated API and static interface; refresh or open port 8001 to use this pilot.

## Exports

The queue writes per-CVC rows to `data/reviews_carina_placement_pilot_50.csv` and `.parquet`, including `placement_label`, and one per-study carina record to `data/carina_reviews_carina_placement_pilot_50.csv` and `.parquet`. Exports preserve reviewer IDs and provisional status. Multiple reviewer rows remain separate for agreement analysis.

The coordinate system is native image pixels. This dataset contains no calibrated pixel spacing, so these coordinates cannot be interpreted as millimetres.
