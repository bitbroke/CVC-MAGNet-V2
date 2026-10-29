# RANZCR CLiP CVC Label, Multi-Device, and Exposure Audit Report

**Date**: 2026-10-25  
**Source Data**: `train.csv` (30,083 studies) and `train_annotations.csv` (17,999 total device traces).  

---

## 1. Executive Summary & Core Label Finding

- **Total Radiographs**: 30,083 across 3,255 unique patients.
- **CVC Placement Labels Are NOT Mutually Exclusive**: Exactly **3,575 studies** (12.19%) have **multiple positive CVC placement labels** simultaneously (e.g., Normal + Borderline: 2,536; Normal + Abnormal: 658; Borderline + Abnormal: 310; All three: 71).
- **Clinical Implication**: The CVC placement task **must be formulated as multi-label binary cross-entropy (BCE)**, NOT single-label multi-class Softmax.
- **Annotated Catheter Traces**: Exactly **8,853 studies** have line annotations (11,629 total CVC polylines). The remaining **21,230 studies have NO polyline traces**.
- **Crucial Polyline Caveat**: 20,480 studies have positive CVC placement labels but 0 polyline traces. Missing polyline traces **do not imply CVC absence**; image-level labels confirm CVC presence even when unannotated.
- **Multi-Catheter Discordance**: Among the 2,489 studies with multiple annotated CVC lines, **1,338 (53.76%)** have catheters with **different individual placement labels** (e.g., one Normal catheter and one Abnormal catheter in the same image). This mathematically explains why image-level labels are multi-label!
- **Prior Exposure Audit**: 15,643 studies were previously used across earlier development, pilot, or gold segmentation experiments. Exactly **14,440 studies (1,301 patients)** remain **completely unexposed**.

---

## 2. CVC Placement Label Counts & Combinations

| Label | Positive Count | % of All Studies | % of CVC-Positive Studies |
|---|---|---|---|
| **CVC - Normal** | 21,324 | 70.88% | 72.70% |
| **CVC - Borderline** | 8,460 | 28.12% | 28.84% |
| **CVC - Abnormal** | 3,195 | 10.62% | 10.89% |
| **Any CVC Present (Union)** | 29,333 | 97.51% | 100.00% |
| **CVC-Absent Studies** | 750 | 2.49% | - |

### Exact Placement Label Combination Breakdown

| Placement Combination | Study Count | Share of CVC-Positive | Label Category |
|---|---|---|---|
| **Normal** | 18,059 | 61.57% | Single Label (Pure) |
| **Borderline** | 5,543 | 18.90% | Single Label (Pure) |
| **Normal+Borderline** | 2,536 | 8.65% | **MULTI-LABEL (Co-occurring)** |
| **Abnormal** | 2,156 | 7.35% | Single Label (Pure) |
| **Normal+Abnormal** | 658 | 2.24% | **MULTI-LABEL (Co-occurring)** |
| **Borderline+Abnormal** | 310 | 1.06% | **MULTI-LABEL (Co-occurring)** |
| **Normal+Borderline+Abnormal** | 71 | 0.24% | **MULTI-LABEL (Co-occurring)** |

> [!IMPORTANT]
> **Multi-Label Finding**: In clinical practice, multi-CVC patients frequently have one properly placed line and one misplaced line. The **3,575 multi-label studies** prove that RANZCR annotators assigned labels independently. Forcing mutually exclusive classes (e.g., using Softmax or priority hierarchies) would discard critical clinical signal.

---

## 3. Catheter Polyline Annotations vs Image-Level Labels

- Total CVC polylines recorded in `train_annotations.csv`: **11,629 lines**.
  - `CVC - Normal`: **7,437**
  - `CVC - Borderline`: **2,986**
  - `CVC - Abnormal`: **1,206**
- Radiographs with **0 annotated CVC traces**: 21,230 (70.6%).
- Radiographs with **exactly 1 annotated CVC trace**: 6,364 (21.2%).
- Radiographs with **2 or more annotated CVC traces**: 2,489 (8.3%).

### Multi-Catheter Discordance Analysis
- Studies with multiple annotated CVCs: **2,489**
- Studies with **discordant placement** across catheters: **1,338 (53.76%)**
- **Proof of Image-Level Ambiguity**: When an image contains multiple CVCs with different placements, an image-level classifier cannot indicate which specific catheter is malpositioned. This formalizes why per-catheter assessment requires catheter-instance grounding.

### Cross-Tabulation: Image-Level Placement vs Annotated Polyline Traces

| Radiograph Status | 0 Polyline Traces | Exactly 1 Polyline Trace | 2+ Polyline Traces | Total Studies |
|---|---|---|---|---|
| **CVC Label Positive** | 20,480 | 6,364 | 2,489 | 29,333 |
| **CVC Label Negative** | 750 | 0 | 0 | 750 |
| **Total** | 21,230 | 6,364 | 2,489 | 30,083 |

> [!WARNING]
> **Supervision Contract Mandate**: Exactly **20,480 studies** have positive CVC placement labels, but **0 polyline annotations**. Treating these 20,480 studies as empty segmentation masks would penalize the model for detecting genuine catheters. When training segmentation-assisted placement or multi-task heads, segmentation loss MUST be zeroed out for unannotated CVC-positive studies while keeping placement classification loss active.

---

## 4. Cohort Exposure & Split Integrity Audit

| Cohort / Partition | Study Count | Exposure Status in Project |
|---|---|---|
| `dev_pilot_val` | 193 | Previously Exposed in Segmentation Dev / Pilot / Gold |
| `dev_pilot_train` | 796 | Previously Exposed in Segmentation Dev / Pilot / Gold |
| `dev_full_train` | 15,450 | Previously Exposed in Segmentation Dev / Pilot / Gold |
| `previously_exposed_held_out` | 0 | Previously Exposed in Segmentation Dev / Pilot / Gold |
| `gold_clean_manifest_2974` | 2,974 | Previously Exposed in Segmentation Dev / Pilot / Gold |
| **Total Exposed Studies** | **15,643** | **Exposed (Known to pipeline development)** |
| **Total Unexposed Studies** | **14,440** | **CLEAN / UNTOUCHED (Zero prior exposure)** |

### Unexposed Patient Cohort
- Unique Unexposed Patients: **1,301**
- Unexposed CVC-Positive Studies: **14,106**
- Unexposed CVC-Negative Studies: **334**

---

## 5. Locked Evaluation and Split Protocol

To guarantee clinical validity and eliminate optimistic data leakage:
1. **Group by PatientID and Duplicate Group**: No patient or near-duplicate radiograph may span train and validation/test splits.
2. **Stratified Partitioning**: Stratify across CVC placement classes (Normal, Borderline, Abnormal, Multi-label, Negative) and companion clutter.
3. **Exposed Cohort Classification**: Reserve all previously exposed studies strictly for development and training, or evaluate them with an explicit 'development-set' caveat.
4. **Locked Untouched Test Partition**: Carve an untouched test partition strictly from the unexposed patients.

