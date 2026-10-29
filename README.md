# CVC-MAGNet V2

Fresh research workspace: published MAG-Net adaptation for CVC segmentation and tip localization.

Previous project: CVC-Refiner V1, paused at C:/Ankshit/Catheter/archives/CVC-Refiner-V1_20261011.

## Sources and provenance
- Paper: https://www.sciencedirect.com/science/article/pii/S0925231222000583
- Author-linked code: https://github.com/Bubble-bubble77/muti-stage-PICC-segmentation
- Pinned commit: a5597438589382eebe5867b7173105128988e650
- Original reference code: vendor/MAG-Net (keep unchanged).

## Scope
Reproduce the method's essential coarse-to-fine segmentation components, then evaluate an explicitly documented CVC adaptation. This is not yet a successful replication. Original private PICC data and our CVC cohort differ.

## First implementation milestones
1. Audit source architecture, target generation, preprocessing, losses and inference. Check licensing before redistribution and determine availability of pretrained weights.
2. Establish target quality: polylines are not dense catheter masks; endpoint heuristics are not expert distal-tip labels. Document any rasterization adaptation and obtain expert endpoint review.
3. Build patient/duplicate-separated development splits and a truly untouched test source. Do not reuse upstream-exposed evaluation as an independent test.
4. Port minimal modules into src under a separate local virtual environment. Do not modify the existing Anaconda environment or execute vendor trainers unchanged.
5. Verify source/code parity, geometry, tiny-batch learning and one forward/backward memory measurement on RTX3050 4GB before choosing resolution or launching training.
6. Pilot full-image/coarse-to-fine segmentation and tip localization against the frozen old pipeline using identical target definitions and one-to-one all-target evaluation.
7. Assess instance association separately on reviewed multi-CVC images. Semantic segmentation alone does not assign catheter identities.

## Initial audit caveats
The reference training scripts appear to instantiate models before cross-validation fold loops. Our harness must initialize a fresh model and optimizer per fold and use registry-based splits, never file-order slicing. Verify all deviations in reports before training.

## Evaluation
Primary: all-target CVC tip Success@15 native pixels; report additional pixel tolerances, proposal coverage, false outputs, median/p95 localization, runtime and peak GPU memory. No millimeters without valid calibration. Report segmentation metrics separately from tip metrics. Preserve abstentions and missed targets in denominators.

## Current state
Source cloned and pinned; no packages installed, no new training launched. First next action: source audit and isolated environment/memory preflight.
