# CVC-Curated-v1 implementation audit

**Scope:** Read-only review of the generated curation code and artifacts on 25 October 2026. No training, label changes, source-data changes, or model changes were made.

## Finding

The saved manifest is structurally usable as a **provisional source-supported segmentation cohort**. It is not evidence for a clinically reviewed or trustworthy-core dataset.

The 30-study review script, `scripts/review_30_study_queue.py`, does not collect or read reviewer decisions. It deterministically assigns:

- `incomplete_annotation` if minimum polyline length is below 450 pixels;
- `border_truncated_valid_path` if an endpoint is close to an image border;
- `multi_cvc_verified_complete`, `cluttered_positive_verified_complete`, or `single_cvc_verified_complete` otherwise.

Thus, the reported 29/30 eligibility is an output of a rule, not a manual/clinical audit. The word **verified** is unsupported in these fields.

## Specific evidence

1. The audited queue has no reviewer ID, timestamp per decision, explicit manual state, coordinates added by a reviewer, or attestation field. It contains geometric metrics and deterministic classifications only.
2. The assertion that all multi-CVC companion paths were complete is not established by the script. It checks that source polylines are retained, which is necessary but cannot establish all visible catheters were annotated.
3. The 450-pixel rule was inferred from one short example among 30. Applying it to all 4,352 annotated positive studies excludes 32 studies automatically. This may be a sensible **review priority** but has not been validated as a reliable completeness classifier.
4. The statement that an average adult CVC is about 1,800 pixels is not a safe basis for filtering: pixel length changes with image size, projection, crop, patient size and catheter route. There is no DICOM spacing here.
5. The manifest claims a five-pixel boundary halo and uncertainty handling, and `scripts/validate_curated_tensors.py` can construct a halo around CVC source polylines. However, its target-building function does not construct the documented `companion_uncertainty_mask` for ETT/NGT/Swan annotations. Therefore the claim that non-CVC companion lines are masked as uncertain background is not implemented by that tensor engine.
6. The tensor-validation script demonstrates internal consistency for selected examples. It does not prove that an existing production trainer consumes these masks. No training integration was verified in this audit.
7. Twelve Group-A studies exist in the base manifest, but their 14 instances use `manual_audit_v3` with review status `engineering_audit_endpoint_review_v3`. Earlier project provenance classifies this review as engineering/provisional rather than clinically attested. It must not be described as clinical verification.

## What is valid to retain

- All 4,352 annotated training positives are source-supported candidates.
- The 396 label-supported negatives may be retained, with a separate review of high-risk false tracks before claiming strong negative supervision.
- The 32 short-path studies should be placed in a **review queue**, not permanently called defective based on length alone.
- The 20,480 CVC-positive/no-polyline studies must remain excluded from dense background segmentation supervision.
- All images, manifests, overlays, models, and original labels remain preserved.

## Required corrections before a matched training experiment

1. Rename the current artifact from `trustworthy_core` to `provisional_length_filtered_cohort_v1`, or add an explicit status field that marks it provisional.
2. Do not use length alone as an exclusion rule. Make the 32 studies a blinded review queue with outcomes: complete source path, incomplete source path, border-truncated, uncertain, or unusable.
3. Create a review record that stores reviewer identity/type, timestamp, study ID, per-instance outcome, and a free-text reason. Only a qualified human review can upgrade a case to reviewed-complete.
4. Implement and test companion-device uncertainty masks from source ETT/NGT/Swan polylines, or remove that claim from the protocol. Do not silently treat those regions as confirmed background.
5. Wire the selected cohort and loss masks into the actual trainer, then perform an end-to-end tensor/gradient test using that trainer. The standalone validation engine is insufficient.
6. Freeze the revised manifest and predeclare one controlled comparison. Retain full validation and report Dice, positive-only precision, negative false area, per-instance coverage, and longest uncovered gaps together.

## Current decision

**Do not launch the matched training experiment using this artifact as a trustworthy-core claim.** It may be used only as a provisional, length-filtered source-supported cohort after the naming and mask-contract corrections above are completed.
