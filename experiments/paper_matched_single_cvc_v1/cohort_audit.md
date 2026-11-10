# Paper-Matched Single-Trace CVC Cohort v1

Exactly one annotated CVC trace per study and exactly one image-level CVC placement label, matching the trace label. This is an annotation-defined single-trace cohort; it is not radiologist-confirmed single-catheter truth.

| Split | Studies | Patients | Normal | Borderline | Abnormal | Prior exposure |
|---|---:|---:|---:|---:|---:|---:|
| train | 4788 | 2264 | 3159 | 1071 | 558 | 3226 |
| val | 537 | 232 | 392 | 102 | 43 | 0 |
| test | 451 | 233 | 304 | 101 | 46 | 0 |

Candidates with one CVC trace in locked train/val/test: 5786.
Removed for trace/image-label disagreement or multiple active labels: 10.
All reported patient and duplicate-group overlaps are zero; validation and test are unexposed.
The test partition is reserved for one final evaluation after model selection.

This is a focused research subset, not the final deployment cohort. The annotation count does not prove that no unannotated CVC is present in an image.
