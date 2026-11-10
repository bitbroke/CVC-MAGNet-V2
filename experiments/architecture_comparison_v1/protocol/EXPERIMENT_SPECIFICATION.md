# Architecture Comparison v1: Experimental Protocol & Specification

**Objective**: Improve CVC segmentation precision and continuity without losing catheters, then determine whether the improvement helps distal-tip localization.

---

## 1. Experimental Framing: Two Separate Tracks

To avoid conflating architecture differences with initialization advantages or training duration:

### Track A: Architecture Comparison (Screening)
- **Question**: Which architecture configuration learns best under comparable starting conditions?
- **Models**:
  1. `EfficientNet-B0 Baseline`: EfficientNet-B0 encoder + multiscale spatial decoder (reproducible control).
  2. `SegFormer-B0`: Hierarchical Transformer encoder + lightweight MLP/all-MLP segmentation decoder.
  3. `Small HRNet`: HRNet-W18-Small-v2 maintaining parallel high-resolution representations throughout.
  4. `EfficientNet-B0–Mamba Hybrid`: EfficientNet-B0 encoder with a residual Mamba state-space block at stride 32.
- **Initialization**: Documented ImageNet-pretrained encoders where available; newly initialized segmentation heads. Hybrid shares the exact baseline encoder initialization; its Mamba module starts fresh.
- **Budget**: Matched 1,000 successful updates from seed 42.

### Track B: Practical Upgrade (Extension of Established Pipeline)
- **Question**: Does adding a Mamba state-space block improve the high-performing model we already have?
- **Models**:
  1. `Frozen Current Model`: Preserved S4 seed-43 checkpoint (Reference baseline).
  2. `Matched Continuation`: Continued training of S4 seed-43 without structural modifications.
  3. `Current Model + Added Mamba Block`: S4 seed-43 checkpoint with an added fresh residual Mamba block at stride 32.

---

## 2. Common Implementation Constraints
- Single-channel output for CVC union segmentation.
- No anatomy landmarks, placement classifier, or tip head in this comparison.
- Input resolution: 768 $\times$ 768 (subject to sequential GPU feasibility check).
- Micro-batch: 1, Effective batch: 4 (gradient accumulation).
- Mixed precision (AMP FP16) with FP32 loss reductions.
- In-place same-batch retry on gradient overflow.
- Single GPU execution (RTX 3050 Laptop GPU, 4,096 MB VRAM).

---

## 3. Data & Supervision Contracts
- Frozen 400-positive / 396-negative training cohort (matching S4/S5 manifests).
- Top-99 selective hard-negative sampling (50% uniform, 50% selective hard negatives).
- Identical effective-batch ledger saved before training: `update_id, study_ids, sampling_category, augmentation_parameters, source_image_hashes`.
- Development validation set (193 studies, 293 assessable tips) used strictly for evaluation.

---

## 4. Evaluation Contract
- Evaluated at native resolution using inverse affine coordinate projections.
- Fixed primary threshold $\tau = 0.50$:
  - Macro Binary Dice (positive studies)
  - Pooled Precision & Recall
  - Centerline coverage within 5px and 10px
  - Missed-catheter rate (< 50% coverage within 10px)
  - Negative false-mask area and negative studies with false foreground
- Development threshold sweep $\tau \in [0.10, 0.90]$ reported separately.
