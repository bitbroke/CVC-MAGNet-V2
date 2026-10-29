# RANZCR CLiP Locked Patient-Separated CVC Placement Split Manifest

**Date**: 2026-10-25  
**Status**: LOCKED & VERIFIED (Zero Patient / Zero Duplicate Leakage)  
**Source Studies**: 30,083 total radiographs across 3,253 patient-duplicate clusters.  

---

## 1. Split Allocation Summary

| Split Name | Role & Exposure Status | Disjoint Clusters | Total Studies | % of Dataset |
|---|---|---|---|---|
| **`train`** | Model fitting (Includes 7,105 unexposed + 17,998 prior dev studies) | 2,597 | **22,444** | 74.61% |
| **`val`** | **Model & threshold tuning (100% UNEXPOSED CLEAN)** | 260 | **2,558** | 8.50% |
| **`test`** | **Untouched evaluation benchmark (100% UNEXPOSED CLEAN)** | 260 | **2,284** | 7.59% |
| **`dev_val_historical`** | Historical development comparator (Prior 193-study val set) | 136 | **2,797** | 9.30% |

---

## 2. Label Distribution Across Splits

| Split | CVC Normal | CVC Borderline | CVC Abnormal | Multi-Label Studies | CVC-Absent (Neg) | Annotated Line Traces |
|---|---|---|---|---|---|---|
| **`train`** | 15,908 | 6,146 | 2,364 | 2,520 | 589 | 8,599 |
| **`val`** | 1,848 | 729 | 266 | 332 | 53 | 1,060 |
| **`test`** | 1,575 | 724 | 245 | 299 | 40 | 818 |
| **`dev_val_historical`** | 1,993 | 861 | 320 | 424 | 68 | 1,152 |

---

## 3. Strict Clinical & Methodological Integrity Rules

1. **Never Touch `test` During Development**: Checkpoints, hyperparameter choices, and decision thresholds MUST be selected strictly on `val`. The `test` partition ({len(df_train[df_train['split'] == 'test']):,} studies) must remain untouched until the final evaluation report.
2. **Multi-Label Formulation**: Placement labels are evaluated as multi-label binary cross-entropy (BCE) with sigmoid outputs. Each class has an independent threshold selected on `val`.
3. **Zero Trace Masking Policy**: In multi-task models, segmentation loss is applied ONLY to the {len(df_train[df_train['annotated_cvc_count'] > 0]):,} studies that have verified line annotations. For studies without line annotations, segmentation loss is masked out (zero gradient) while placement classification loss remains active.

