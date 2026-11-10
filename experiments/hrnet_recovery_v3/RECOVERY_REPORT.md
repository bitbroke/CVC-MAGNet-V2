# HRNet recovery audit — 2026-10-18

## Completed work

All three saved checkpoints were independently re-evaluated on the same native-resolution segmentation benchmark (193 studies, 248 catheter paths). Tip evaluations use the same decoder and coordinate convention. Source checkpoint SHA-256 hashes remain unchanged. No training was launched.

| Frozen checkpoint / decoder | Native macro Dice | Centerline coverage at 10 px | Missed paths | Tip Success@15, mixed provisional targets |
|---|---:|---:|---:|---:|
| T1 baseline / quadratic | 0.4625 | 87.93% | 8/248 | 14/41 |
| Continuation control / quadratic | 0.4857 | 84.83% | 15/248 | 14/41 |
| Offset continuation / quadratic, learned offsets disabled | 0.4930 | 85.12% | 13/248 | 15/41 |
| Same offset checkpoint / learned offsets enabled | 0.4930 | 85.12% | 13/248 | 12/41 |

Disabling learned offsets recovers three matched targets on the SAME checkpoint. This identifies an operating decoder problem, not proof of destructive backbone gradient interference. The gain over the baseline is only one target and is not evidence of statistical superiority. At 25 pixels the learned-offset decoder finds 19 targets versus 18 for quadratic decoding, so the choice depends on the predeclared primary tolerance.

Continuation improves Dice and reduces false-mask area but loses centerline coverage and misses more catheter paths. Neither continuation is an unconditional replacement for the baseline. Preserve the baseline as the recall reference.

## Label and evaluation limitations

The user confirmed there are no actual reviewer records for the 16 automatically labeled Blinded_Audit_Panel_V2 endpoints. They remain unverified. Automatic review attestation in batch_repair_benchmark.py is now disabled, with its source preserved before editing. The restored provenance CSV records original coordinates and proposed changes separately; original experimental artifacts remain untouched.

The 41-target scores above deliberately reproduce the existing mixed provisional V2 development benchmark. They are NOT independently verified accuracy. The complete-study provisional subset has only 11 targets across 11 positive studies plus 8 negatives. Quadratic decoding matches 6/11 for all three checkpoints; learned offsets match 5/11. This small subset contains no complete multi-CVC evaluation and cannot validate the every-catheter goal.

The former failure categories are preserved alongside explicitly unverified causes. Distance alone does not establish quantization, anatomical ambiguity, or a missed catheter. Geometry consistency is not clinical verification.

## Code repairs and verification

- Added explicit instance-completeness and supervision contracts. Unknown or incomplete positive studies cannot become empty tip negatives. Weak supervision requires explicit opt-in. These contracts are staged for the next trainer; legacy trainers are not silently rewritten.
- Added exact continuous-coordinate offset encoding and masking of same-cell collisions; no Gaussian-fit approximation is required. These contracts are not retroactively applied to trained weights.
- Three contract tests passed, including negative-label conflicts, unknown masking, exact residuals and collisions.
- Decoder heatmap caches now use checkpoint, image, manifest, preprocessing, geometry and model source hashes. Legacy caches are retained; future runs cannot silently reuse them under a seed-only name. New cache writes are atomic and cached heatmaps are checked for finite values and exact study inventory.
- Canonical segmentation measurements replace the unsupported 0.7816 retention claim.

## Next implementation sequence

1. Use the frozen T1 checkpoint as the segmentation-recall reference. Keep the offset checkpoint with quadratic decoding as a development challenger, not a validated winner. No additional decoder sweep on this repeatedly used benchmark.
2. Incorporate the new supervision contracts into a separate versioned dataset/trainer. Keep provisional engineering labels, heuristic labels and truly attested labels distinct. Mask unknown tip loss without removing valid segmentation supervision. Require complete per-study instance accounting for full-image tip supervision.
3. Review the 16 unresolved targets and complete all coexisting CVCs before making verified multi-CVC accuracy claims. Automated tools can prepare overlays and flag disagreement but cannot manufacture reviewer attestation. The user explicitly confirmed these records must stay provisional.
4. For a future offset experiment, derive residual targets directly from transformed continuous coordinates, test real augmentation geometry, and compare offsets off/on on identical weights. Require a real forward/backward/decode test, exact optimizer exposure records, and baseline coverage retention before extending training.
5. Prioritize missed-path and faint/multiple-catheter cases in a training-only expansion. Preserve negatives and keep label-policy changes separate from architecture changes. Run a single matched control/challenger pilot with fixed initialization, image exposure, update budget and primary metric. Do not claim a model upgrade from lower false area alone.
6. Report patient-cluster uncertainty, all-target one-to-one localization, false tips per study, segmentation coverage and missed paths together. Use an independently reserved evaluation cohort for final claims; current development results cannot certify absence of overfitting or project-level selection leakage.
7. Anatomy landmarks, millimeter distances and placement classification remain separate tasks. Millimeters require calibration metadata; no such metadata has been established here. Do not infer correct placement merely from localization confidence.

The project has a reproducible development result, not a perfect or clinically validated model. The audit and repairs are complete; model improvement and independent validation remain unfinished.
