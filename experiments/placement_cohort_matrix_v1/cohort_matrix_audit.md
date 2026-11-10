# Placement Cohort Matrix v1

Derived without changing split assignments or checkpoints.

## all_cvc_traced_multilabel

At least one annotated CVC trace; original three image-level labels retained.

| Split | Studies | Patients | Normal | Borderline | Abnormal | Multi-label | Prior exposed |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 6600 | 2538 | 4708 | 1970 | 874 | 935 | 4352 |
| val | 778 | 256 | 588 | 261 | 87 | 154 | 0 |
| test | 630 | 256 | 450 | 201 | 85 | 105 | 0 |

## single_cvc_binary_normal_vs_non_normal

Exactly one CVC trace, exactly one concordant image-level label; Normal=0, Borderline or Abnormal=1.

| Split | Studies | Patients | Normal | Borderline | Abnormal | Multi-label | Prior exposed |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 4788 | 2264 | 3159 | 1071 | 558 | 0 | 3226 |
| val | 537 | 232 | 392 | 102 | 43 | 0 | 0 |
| test | 451 | 233 | 304 | 101 | 46 | 0 | 0 |

## Interpretation safeguards

- An annotation trace count is not proof that the image contains exactly that many visible catheters.
- Image-level placement labels are not catheter-specific outcomes in multi-CVC studies.
- The locked test subset must remain unused until final model selection is complete.
- The binary target means not-normal by dataset label; it is not a millimetre-based placement judgment.
