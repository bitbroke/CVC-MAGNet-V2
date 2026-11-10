# Completed feature-adaptation experiment: interpretation

Both arms completed 250 successful updates without AMP retries. The original checkpoints and source inputs are unchanged. Post-run tensor comparison confirms that only the segmentation head changed in the control; encoder, projection, fusion and segmentation-head tensors changed in the feature arm. BatchNorm buffers and tip-head weights stayed unchanged. The candidate's internal tip outputs must never be used: the separate complete frozen tip model supplies inference tips, with bitwise output equality verified on the deployment adapter's test input.

## Outcome

Feature adaptation reduced missed annotated paths from 8/248 to 5/248, compared with 7/248 in the matched head-only control. Four old misses crossed above 30% coverage and one previously covered path became a new miss. The four gains are substantial (9.5% to 81.9%, 2.1% to 53.9%, 0% to 67.1%, and 2.0% to 85.5%), but none implies complete recovery of a catheter or its distal tip.

Overall coverage increased from 87.93% to 92.50%; paths with at least 90% coverage increased from 173 to 197. Mean longest uncovered visible segment declined from 156.0 to 102.2 native pixels. Dice fell from 0.4625 to 0.3643, and inherited mean-study precision fell from 34.03% to 24.87%. Therefore the locked preservation criteria FAIL and no model is promoted.

Negative-image metrics require careful interpretation: mean per-study false-area percentage decreased from 0.06105% to 0.05059%, but total false pixels increased from 25,692 to 27,460. Unequal native image dimensions make these different estimands. Only eight negatives were evaluated; this is not evidence of general clinical specificity.

## Visual inspection

All nine panels in `overlays/` were inspected. This is engineering inspection of source polylines and predictions, not clinical adjudication.

| Panel | Observation |
|---|---|
| 01 | A previously absent companion trajectory appears over much of its length. The new corridor is broad and merges at the crossing. |
| 02 | Selected neck-region source path remains absent in all arms; embedded black-border artifacts persist. No automatic label correction is justified. |
| 03 | Long descending continuation remains missing; the feature arm also suppresses the proximal fragment. |
| 04 | New miss: coverage falls from 51.3% to 25.1%, with substantial lateral path loss. |
| 05 | Companion trajectory remains absent; proximity to the other catheter supplies part of the apparent coverage. |
| 06 | Some coverage gain, but the long arch is still missing and the companion mask broadens. |
| 07 | A longer arch is recovered, with broad/spurred foreground and a remaining 618-pixel gap. |
| 08 | A previously absent companion arch becomes substantially visible but remains fragmented; longest gap 320 pixels. |
| 09 | A long previously absent trajectory is recovered to 85.5% coverage, still with residual gaps and mask-width error. |

## What this does and does not establish

This matched experiment supports further investigation of feature adaptation: it recovers trajectories that head-only optimization does not. It does not isolate the benefit of hard-positive mining, because both arms used the same mining policy. It does not show that every recovered mask corresponds exclusively to the assigned catheter; evaluation remains proximity to a union mask. It does not establish clinically correct tips, independent generalization or absence of overfitting.

## Next focused work

Keep the production baseline and separate tip model frozen. Before another training run, quantify mask widening versus distant false foreground on positive training studies, using the saved maps and source paths, and inspect the new lost path. A subsequent controlled candidate should target boundary/false-foreground precision on positive images while retaining feature adaptation and the same training ledger. Do not simply continue this run, increase recall weight, lower the threshold, or promote a model on coverage alone. No further training has been launched.

## Saved verification

- Three contract tests passed; both real-batch two-update GPU smoke arms passed.
- Control training: 137.5 seconds; feature adaptation: 332.3 seconds.
- Feature training peak: 1294.6 MiB allocated / 1372.0 MiB reserved (not a claim of free GPU memory).
- `postrun_verification.json`, `frozen_hashes.json`, `source_hashes.json`: preservation evidence.
- `path_transitions.csv`: all 248 paths with continuous coverage and gap metrics.
- `training_path_inventory.csv`: 506 training paths; 129 hard paths across 110 studies.
- `decision.json`: explicit failed criteria and no promotion.
