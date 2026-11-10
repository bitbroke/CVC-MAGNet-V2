# Low-clutter-first versus mixed-data continuation

Authorized local controlled comparison, 24 October 2026. Existing datasets and models remain frozen. This is an adaptation of the autoresearch workflow for one predeclared two-arm pilot, not an open-ended optimization loop.

The sample review found that a single annotated CVC and no labeled ETT/NGT/Swan is not equivalent to a clean radiograph. Loops, faint tracks, external wires and short peripheral annotations remain. See `comparison/inspection/visual_review.json` and the three raw/annotation contact sheets. All findings are provisional engineering observations, not clinical attestation. No labels were changed or discarded.

## Matched experiment

- Original frozen HRNet joint checkpoint copied separately into each training arm. Tip deployment remains the separate complete original model.
- Seed 42, 250 successful updates each, 768 input, effective batch four, microbatch one, frozen BatchNorm buffers.
- Exactly 500 positive exposures (375 low-clutter, 125 other annotated positives) and 500 negative exposures per arm. 896 unique training images; 396 unique confirmed negatives.
- The same image/augmentation multiset in both arms. The same negative examples at the same update/slot. Positive order and co-batching differ.
- `simple_first`: first 125 updates have two low-clutter positives; last 125 updates mix the remaining positives.
- `mixed_data`: the identical positive exposure multiset shuffled throughout.
- Identical optimizer, schedule and losses from the existing feature-adaptation runner. No new boundary penalty.
- Full 193-study / 248-path native validation, threshold 0.50. All targets retained; eight negative validation studies.
- Primary fixed-step comparison: macro Dice. Safeguards cover precision, coverage, missed paths and negative false area/pixels. No automatic promotion.

This isolates presentation order within a fixed low-clutter-enriched sample. It does not compare against natural-prevalence full-dataset training. A single seed cannot establish a general curriculum benefit.

## Artifacts

`comparison/protocol.json`, exposure ledgers, source snapshots/hashes, existing checkpoint hashes, image hashes, parity checks, visual inspection record, bounded on-disk image cache, per-arm checkpoints and histories, probability maps, full and subgroup metrics, patient-cluster bootstrap interval, path transitions and comparison overlays.

Preparation and training source revisions are recorded separately because the training loader was changed to bound RAM use after preparation began. Preparation provenance is preserved rather than overwritten.

No result represents independent clinical accuracy; labels remain polyline-derived and the development cohort has been used repeatedly. No physical millimetres are inferred.
