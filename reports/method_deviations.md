# Method Deviations: Adapted MAG-Net vs. Published Paper

**Document Version**: 2.0  
**Date**: October 11, 2026  
**Reference Paper**: Wang et al., *"A Multi-stage Attention-Guided Network for PICC Line Segmentation in Chest X-rays"*, Neurocomputing, 2022.  
**Pinned Vendor Repository**: `Bubble-bubble77/muti-stage-PICC-segmentation` (Commit `a5597438589382eebe5867b7173105128988e650`).

---

## 1. Licensing & Source Isolation
- **Paper / Vendor State**: The vendor repository on GitHub has no root `LICENSE` file.
- **V2 Policy**: The vendor code in `vendor/MAG-Net/` is kept strictly read-only and frozen. All adapted modules are implemented cleanly under `src/models/`, `src/data/`, and `src/infer/` to avoid copyright entanglement and enable independent verification.

---

## 2. Supervised Target & Catheter Scope
- **Published Paper**: Single PICC (Peripherally Inserted Central Catheter) line per patient. Private dataset from a single clinical institution.
- **V2 Adaptation**: Central Venous Catheter (CVC) scope on the public, multi-institutional RANZCR CLiP dataset. Radiographs frequently contain co-occurring lines (ETT, NGT, Swan-Ganz, and multi-CVC crossings).
- **Negative Supervision**: The published paper trained solely on catheter-positive images. Our adaptation includes verified study-level negative radiographs (no CVCs present in radiology reports) to ensure the network suppresses false positives on other catheters and anatomy.

---

## 3. Polyline Rasterization vs. Dense Manual Masks
- **Published Paper**: Supervised with dense manual segmentation masks of PICC lines.
- **V2 Adaptation**: RANZCR annotations are discrete, unoriented polyline coordinates. We adapt dense mask supervision by rasterizing each polyline with an empirical stroke width hyperparameter of **10 native pixels** (uncalibrated to physical millimeters).
- **Fine-Stage Mask Attention**: In line with the published paper, an 11×11 Gaussian blur kernel ($\sigma = 2.0$) is applied to the rasterized mask to provide continuous target supervision for the learned spatial attention module ($L_1$ loss weight = 0.5).

---

## 4. Geometric Transforms & Coordinate Preservation
- **Published Paper**: Anisotropic scaling to 512×512 using `skimage.transform.resize(mode='reflect')`. This alters anatomical aspect ratios and prevents exact projection back to native pixel space.
- **V2 Adaptation**: Strict **isotropic letterbox resizing** with symmetric padding. Every transform step records scale and padding metadata (`TransformMetadata`). Roundtrip coordinate reconstruction achieves subpixel alignment, enabling exact evaluation in native pixels (Success@15).

---

## 5. Numerical Stability & Modernized PyTorch Operations
- **Published Paper**:
  - Used deprecated `F.upsample(..., mode='bilinear')`.
  - Applied `nn.Sigmoid()` inside model definitions and trained with `nn.BCELoss()`, which is prone to numerical instability (NaN/Inf) under extreme saturating logits.
  - Squeezed tensors via `torch.squeeze()`, which collapses batch dimensions when `batch_size = 1`.
- **V2 Adaptation**:
  - Replaced with standard `F.interpolate(..., align_corners=False)`.
  - Models output raw unconstrained logits; optimization uses numerically stable `torch.nn.BCEWithLogitsLoss`.
  - Explicit shape indexing prevents accidental batch dimension collapse.

---

## 6. Safe Empty-Mask Handling (Bug Fix)
- **Published Paper**: Crop extraction in `Train_2_return.py` used `choice = np.where(seg_pre > 0.1); xmin = choice[0].min()`. If the coarse model predicted zero foreground pixels, this crashed with `ValueError: zero-size array to reduction operation minimum`.
- **V2 Adaptation**: If coarse prediction contains no foreground pixels above threshold, `has_roi` is set to `False`. The pipeline falls back to full-field center crop or outputs an explicit failure state (`no_catheter_detected`), ensuring robust training and zero crashes on negative cases.

---

## 7. Endpoint Extraction Heuristic Audit & Adaptation
- **Published Paper**: `findpoint()` heuristic in `config/test_function.py` searched for the bottom-most pixel to the right of the centroid (`maxx >= cx`), tailored specifically for PICC lines entering via the patient's right arm.
- **V2 Adaptation**: CVCs enter from either internal jugular or subclavian veins (both left and right sides) or femoral veins, and terminate centrally at the cavoatrial junction. The vendor's right-sided assumption fails on left-sided CVCs. We isolate segmentation quality from distal endpoint selection, and evaluate candidates using graph skeletonization and border distance rules.

---

## 8. Split Exposure Status & Generalization Scope
- **V1 Fitting Exposure**: Folds 1–4 were previously exposed to model fitting during V1 development.
- **V2 Holdout Policy**: Fold 4 is held out from V2 model fitting. However, because Fold 4 was previously exposed in V1, it **cannot support a genuinely untouched project-level test claim**. Final project-level generalization remains unmeasured.

---

## 9. Annotation Provenance & Clinical Verification Status
- **Audit Findings**: The 74 diagnostic packet cases in `reports/expert_review_manifest.csv` record `provenance_type = automated_pipeline_derived (awaiting_clinical_expert_audit)`.
- **Strict Provenance Rule**: Zero labels are assigned "expert verified" status. Documented human clinical expert sign-off is required before any label can be treated as expert-verified. All distal tips currently retain automated/heuristic provenance.
