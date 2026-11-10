# Boundary precision experiment — 2026-10-22

| Arm | Misses /248 | Coverage10 | Dice | Mean-study precision | Mean negative area % | Total negative FP pixels | Mean longest gap px |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 8 | 87.93% | 0.4625 | 34.03% | 0.06105 | 25692 | 156.0 |
| parent | 5 | 92.50% | 0.3643 | 24.87% | 0.05059 | 27460 | 102.2 |
| control | 5 | 92.68% | 0.3574 | 23.27% | 0.07727 | 44025 | 100.4 |
| boundary | 6 | 90.80% | 0.4739 | 33.88% | 0.05375 | 31037 | 117.4 |

Controlled experiment decision: FAIL. No model promoted.

Both arms start from the frozen feature-adaptation checkpoint and use identical batches, augmentations, learning rates, successful-update budgets and foreground-retention loss. Only the 0.25 boundary term differs. The boundary target is distance 1 < d <= 8 model pixels outside the annotated mask; one pixel is ignored for alignment uncertainty. This is supervision against rasterized polylines, not independently traced anatomical boundaries.

Recovered-path preservation requires each of the four paths to retain coverage within 1 percentage point and longest gap within 10 native pixels of the parent. Overall coverage may fall at most 0.5 percentage points. These tolerances are predeclared, not exact non-regression guarantees. Tips use the original separate frozen model.