# Simple-first vs mixed-data pilot

Same positive/negative image and augmentation multisets; only presentation order differs. Both start from original frozen HRNet. No checkpoints promoted.

| Model | Dice | Precision | Coverage10 | Misses | Negative FP pixels |
|---|---:|---:|---:|---:|---:|
| original | 0.4625 | 34.03% | 87.93% | 8/248 | 25692 |
| mixed_data | 0.3818 | 26.17% | 91.53% | 7/248 | 18616 |
| simple_first | 0.3603 | 23.60% | 92.43% | 6/248 | 46819 |

Paired mean study Dice difference (simple-first minus mixed): -0.0215; patient-bootstrap 95% interval [-0.0267, -0.0161].
Predeclared safeguards: FAIL.

| Model | Group | Studies | Dice | Coverage10 | Misses |
|---|---|---:|---:|---:|---:|
| original | all_positive | 185 | 0.4625 | 87.93% | 8/248 |
| original | simple | 66 | 0.4932 | 95.10% | 1/66 |
| original | single_with_other | 60 | 0.4287 | 88.61% | 1/60 |
| original | two_cvcs | 55 | 0.4645 | 83.30% | 6/110 |
| original | three_plus | 4 | 0.4349 | 87.51% | 0/12 |
| original | multiple_with_other | 34 | 0.4412 | 79.93% | 4/71 |
| mixed_data | all_positive | 185 | 0.3818 | 91.53% | 7/248 |
| mixed_data | simple | 66 | 0.3965 | 96.94% | 1/66 |
| mixed_data | single_with_other | 60 | 0.3535 | 91.60% | 1/60 |
| mixed_data | two_cvcs | 55 | 0.3957 | 88.82% | 4/110 |
| mixed_data | three_plus | 4 | 0.3728 | 86.28% | 1/12 |
| mixed_data | multiple_with_other | 34 | 0.3843 | 85.03% | 4/71 |
| simple_first | all_positive | 185 | 0.3603 | 92.43% | 6/248 |
| simple_first | simple | 66 | 0.3789 | 97.92% | 0/66 |
| simple_first | single_with_other | 60 | 0.3237 | 91.81% | 2/60 |
| simple_first | two_cvcs | 55 | 0.3792 | 89.60% | 4/110 |
| simple_first | three_plus | 4 | 0.3423 | 91.26% | 0/12 |
| simple_first | multiple_with_other | 34 | 0.3692 | 87.41% | 3/71 |

One seed, 250-update continuation pilot, repeatedly used development cohort; eight negatives. Simple means metadata-based low clutter, not clinical confirmation. A nonzero subgroup difference is not proof of generalization. No claim of improved tip accuracy: complete original tip branch remains separate and frozen.

Labels remain rasterized polyline annotations. Visual sample review is provisional engineering inspection. Full dataset and all existing models are preserved.

The visual review of 12 training images found loops, faint lines, external wires, poor contrast and short peripheral traces despite passing the low-clutter metadata rule. None was relabeled or called expert-verified.

Timing caveat: the control history contains a long wall-clock gap. Its reported elapsed seconds must not be interpreted as uninterrupted GPU compute time.