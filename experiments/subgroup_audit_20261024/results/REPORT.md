# Frozen-model dataset subgroup audit

Native-resolution segmentation, threshold 0.50; unchanged 193-study development cohort.
Simple = one annotated CVC and no labeled ETT/NGT/Swan. Counts are inventory-based, not clinical confirmation.

| Model | Subset | Studies / patients / paths | Dice | Path coverage @10px | Missed paths |
|---|---|---:|---:|---:|---:|
| original_hrnet | all_cvc_positive | 185 / 132 / 248 | 0.463 | 87.9% | 8 |
| original_hrnet | single_cvc | 126 / 99 / 126 | 0.463 | 92.0% | 2 |
| original_hrnet | two_cvcs | 55 / 43 / 110 | 0.464 | 83.3% | 6 |
| original_hrnet | three_or_more_cvcs | 4 / 3 / 12 | 0.435 | 87.5% | 0 |
| original_hrnet | simple_single_no_other_device | 66 / 63 / 66 | 0.493 | 95.1% | 1 |
| original_hrnet | single_with_other_devices | 60 / 43 / 60 | 0.429 | 88.6% | 1 |
| original_hrnet | multiple_no_other_device | 25 / 21 / 51 | 0.491 | 89.0% | 2 |
| original_hrnet | multiple_with_other_devices | 34 / 25 / 71 | 0.441 | 79.9% | 4 |
| recovery_parent | all_cvc_positive | 185 / 132 / 248 | 0.364 | 92.5% | 5 |
| recovery_parent | single_cvc | 126 / 99 / 126 | 0.360 | 95.2% | 2 |
| recovery_parent | two_cvcs | 55 / 43 / 110 | 0.375 | 89.6% | 3 |
| recovery_parent | three_or_more_cvcs | 4 / 3 / 12 | 0.364 | 91.0% | 0 |
| recovery_parent | simple_single_no_other_device | 66 / 63 / 66 | 0.384 | 97.8% | 0 |
| recovery_parent | single_with_other_devices | 60 / 43 / 60 | 0.333 | 92.3% | 2 |
| recovery_parent | multiple_no_other_device | 25 / 21 / 51 | 0.386 | 92.6% | 1 |
| recovery_parent | multiple_with_other_devices | 34 / 25 / 71 | 0.365 | 87.7% | 2 |
| boundary_candidate | all_cvc_positive | 185 / 132 / 248 | 0.474 | 90.8% | 6 |
| boundary_candidate | single_cvc | 126 / 99 / 126 | 0.478 | 94.1% | 2 |
| boundary_candidate | two_cvcs | 55 / 43 / 110 | 0.469 | 87.2% | 4 |
| boundary_candidate | three_or_more_cvcs | 4 / 3 / 12 | 0.418 | 88.8% | 0 |
| boundary_candidate | simple_single_no_other_device | 66 / 63 / 66 | 0.511 | 96.9% | 0 |
| boundary_candidate | single_with_other_devices | 60 / 43 / 60 | 0.441 | 91.0% | 2 |
| boundary_candidate | multiple_no_other_device | 25 / 21 / 51 | 0.497 | 91.8% | 1 |
| boundary_candidate | multiple_with_other_devices | 34 / 25 / 71 | 0.442 | 84.2% | 3 |

Patient-bootstrap Dice intervals are in subgroup_metrics.csv. Groups overlap. Do not interpret small-subgroup rankings as proven superiority.
Training-only simple and negative manifests are candidates for a controlled curriculum, not a replacement dataset. No records removed; no training launched.
Keep complex cases for later training and all cohorts for evaluation. Tip labels and full visible catheter counts still require verification.