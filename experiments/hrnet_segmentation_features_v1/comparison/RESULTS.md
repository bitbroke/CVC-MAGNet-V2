# Segmentation feature adaptation pilot

| Arm | Misses /248 | Coverage10 | Dice | Precision | Negative false area % | Mean longest gap px | Paths >=90% |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 8 | 87.93% | 0.4625 | 34.03% | 0.06105 | 156.0 | 173 |
| head_control | 7 | 91.14% | 0.3471 | 21.21% | 0.18722 | 120.5 | 191 |
| feature_adaptation | 5 | 92.50% | 0.3643 | 24.87% | 0.05059 | 102.2 | 197 |

Locked progression decision: FAIL. No model promoted.

Only trainable parameter scope differs between arms. Same training-mined hard-positive ledger, losses, resolution, seed, successful updates and segmentation-head learning rate. Newly trainable features use LR 1e-5. BN buffers stay fixed. Tip outputs must use the complete original frozen model, never the adapted copy.

Coverage is proximity to a union mask, not instance association. Gap metrics integrate consecutive uncovered samples along the visible annotated path, in native pixels. Source annotations may be incomplete. Eight negatives cannot establish reliable clinical specificity. No millimeter calibration or independent test claim.