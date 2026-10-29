# MAG-Net source audit — 2026-10-11

Pinned source commit is recorded in README. Static inspection only; no model executed.

- Coarse CS-Net: base-32 U-Net with pyramid pooling, grayscale 512x512 input, binary 1024x1024 output.
- Fine UNet_SW1_PAM_D: 512x512 image plus coarse prediction, learned mask attention, stride-8 position attention, pixel-shuffle decoding. A 4096x4096 FP32 attention matrix alone occupies 64 MiB; complete 4GB GPU fit remains unmeasured.
- Fine supervision: binary catheter mask and 11x11 Gaussian-blurred mask. BCE segmentation plus 0.5 L1 attention loss. These model modules have no explicit distal-tip coordinate head; audit endpoint extraction separately.
- Original data are private; no confirmed downloadable weights or root license file located. Resolve reuse/redistribution terms before distributing vendor-derived code.
- Trainers instantiate the model before fold loops; use fresh initialization per fold in our harness. File-order splits must be replaced by patient/duplicate-group manifests.
- Coarse train/test preprocessing differs (test CLAHE). Establish one explicit contract and document replication differences.
- Fine crop extraction min/max on mask >0.1 needs an empty-mask failure state.

Next test, after isolated local environment setup: model definitions only, batch-one synthetic forward/backward coarse and fine separately; finite outputs/gradients, peak CUDA memory and timing. Then real-image/mask geometry overlays. No heavy training until these checks and annotation contracts pass.

Sources:
https://github.com/Bubble-bubble77/muti-stage-PICC-segmentation
https://www.sciencedirect.com/science/article/pii/S0925231222000583
