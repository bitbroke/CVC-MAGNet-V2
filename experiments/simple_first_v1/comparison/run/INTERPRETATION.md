# Interpretation — 25 October 2026

Completed two 250-successful-update arms, one seed, with zero AMP retries in either full arm. All 219 pre-existing `.pt` checkpoint files and tracked source inputs passed post-run SHA-256 verification. No model promoted and no source labels changed. The complete original tip model remains the separate inference branch, verified against the composed inference output.

## Outcome

Simple-first failed the predeclared quality safeguards. Relative to mixed-data training, mean study Dice fell from 0.3818 to 0.3603, precision from 26.17% to 23.60%, while path coverage rose from 91.53% to 92.43% and misses decreased from 7 to 6 of 248. Negative false pixels rose from 18,616 to 46,819 (2.51x). Both trained candidates have worse mask Dice and precision than the original frozen reference (0.4625 and 34.03%).

The paired Dice difference is -0.0215, with a patient-cluster bootstrap interval of [-0.0267, -0.0161]. This describes this repeatedly used development cohort, not independent generalization. There are only eight negative validation images; 42,469 of the simple-first arm's 46,819 false pixels occur in one study. Do not generalize the total false-area ratio to a population estimate.

Even on the 66 simple validation studies, simple-first Dice is 0.3789 versus 0.3965 for mixed and 0.4932 for the original model. Easier-image selection did not solve mask precision in this pilot.

## Visual evidence personally inspected

Twelve raw/annotation training examples were reviewed before training. The metadata rule (one inventory-listed CVC, no ETT/NGT/Swan positive label) still includes low contrast, loops, external wires and short edge-adjacent traces. None is declared clinically verified; no bad-label percentage is inferred from this deliberately varied small sample.

Six positive comparison panels were then inspected, containing the largest paired Dice changes and all four previously recovered-path studies. In panels 3, 4 and 6, simple-first restores more of faint upper-chest/arching annotated paths. It also produces wider orange regions and spur branches beyond the cyan reference, especially panel 6. In panel 2, the simple-first mask retains a long extra lateral branch that the mixed-data model mostly suppresses. In panel 5, simple-first loses part of the neck companion path: its coverage is 56.1% versus 79.4% for mixed. This demonstrates why a total missed-path count alone is insufficient.

All four previously recovered paths are above the 30% coverage detection threshold in simple-first, but this does NOT mean their full paths are preserved. Against the earlier recovery parent, the first of those paths drops from 81.9% to 56.1% coverage. The current experiment does not replace that parent either.

The two highest-false-area negative panels were inspected. In negative panel 1, simple-first extends a long upper/lateral chest arc absent from the reference annotations and counted as false foreground under the CVC-absent study label. The second negative has much smaller scattered predictions. These observations support increased false tracking, but do not independently validate clinical absence or establish which loss component caused it.

## Decision and next step

Keep all frozen references; do not replace the dataset with only easy cases and do not launch full-scale simple-first training based on this result. This experiment tests ordering within the SAME low-clutter-enriched image/augmentation multiset; it does not prove every possible curriculum or selection strategy will fail.

Before another training run, inspect training-side negative false tracks and sampled positive annotation completeness. A next controlled experiment should address precision loss during continuation (for example conservative learning rates or a separately controlled precision constraint), while retaining mixed cases and negative supervision. This is a hypothesis for a future experiment, not an authorized automatic follow-on or a confirmed causal diagnosis.

## Artifacts

- `REPORT.md`: full and subgroup metrics.
- `subgroup_metrics.csv`, `paired_paths.csv`, `previously_recovered_paths.csv`: auditable denominators and path transitions.
- `paired_dice_bootstrap.json`: patient-level paired interval.
- `overlays/`: six positive panels, columns annotation / original / mixed / simple-first.
- `negative_overlays/`: two negative panels with the same column order.
- `../inspection/`: 12-example pre-training visual inspection, raw/annotation panels and notes.
- `status.json`, `decision.json`, per-arm histories/checkpoints and `deployment.json`: completion and freeze/inference records.

Timing caveat: the control run contains a long wall-clock interruption; its elapsed seconds do not measure uninterrupted GPU training time.
