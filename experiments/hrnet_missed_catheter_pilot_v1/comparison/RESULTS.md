# Missed-catheter recall pilot — completed

All checkpoints were preserved. Both arms started from the same frozen T1 weights and received the identical 250-update image/augmentation ledger. The only arm difference was a 0.25 equal-instance soft recall loss. BatchNorm running statistics were frozen in both arms. Unverified tips were excluded from tip supervision; segmentation used existing CVC polylines.

| Arm | Dice | Coverage @10 | Missed /248 | Precision | Negative false area | Tips @15 /41 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.4625 | 87.93% | 8 | 34.03% | 0.0610% | 14 |
| control | 0.4812 | 84.30% | 16 | 38.31% | 0.0121% | 10 |
| instance_recall | 0.4142 | 88.36% | 8 | 30.68% | 0.0002% | 12 |

Metric definitions: Dice is the mean over 185 positive studies. Centerline coverage is the mean over 248 catheter paths; a path with less than 30% of sampled points within 10 native pixels of foreground is counted as missed. Precision is the legacy evaluator mean across all 193 studies (including negative-study conventions), not pooled pixel precision. Negative false area is the mean across eight negative studies.
Predeclared progression rule: **PASS**. Requires fewer missed paths than control and baseline coverage retained within one percentage point.
Paired study-mean coverage difference (experiment minus control): 3.48 percentage points; patient-cluster 95% bootstrap interval [2.16, 4.99].

A progression pass alone does not justify promotion: inspect precision, negative false area, tip performance and paired overlays. The tip benchmark remains mixed provisional (including 16 unverified endpoints); it is a development diagnostic, not clinical accuracy. Complete-provisional-subset results are retained in each metrics JSON. No external generalization, overfitting exclusion or millimeter accuracy claim is justified. Masking weak labels during continuation does not undo their historical influence in the inherited checkpoint. This is a controlled development continuation, not a clean-label retraining.

Tip-supervision exposure: the common ledger contains 17 provisional-positive exposures, 250 confirmed-negative exposures and 733 masked exposures. This imbalance is a plausible contributor to tip suppression, not an isolated causal finding. Do not treat this experiment as a sufficient test of the achievable tip-localization performance. A subsequent segmentation-focused experiment should explicitly protect the existing tip behavior rather than assume sparse joint supervision preserves it.

Next: retain the baseline unless the complete trade-off is acceptable. Replicate a promising improvement on another seed before scaling. If the primary rule fails, do not extend this loss configuration or overwrite the baseline.

## Final disposition: do not promote
The experiment passes the narrow coverage progression rule but has lower Dice and fewer correctly localized provisional tips than the frozen baseline. Its missed-path count equals, rather than improves upon, the baseline. Keep the frozen baseline as the main reference; retain this checkpoint only as evidence of a recall-versus-mask-quality trade-off. The reduced false area on eight negatives does not compensate for poorer positive-study masks or establish reliable specificity.
A next experiment should constrain positive-study false foreground and protect tip behavior while retaining the instance-recall signal. Do not expand training duration for this configuration unchanged. Do not launch a second-seed confirmation merely because the narrow coverage rule passed.

## Saved visual inspection
Four panels and twelve binary mask files are saved in overlays/. The multi-CVC panel repeats low_coverage_2 because that study also has the lowest baseline multi-CVC coverage (three distinct studies total). In this multi-line example, the recall arm recovers more of one previously missed path but produces broader masks and still lacks complete distal segments. The first low-coverage example remains incomplete in every arm. The selected negative shows substantially less false foreground in the recall arm. These examples illustrate the measured trade-off and are not a representative accuracy estimate.
