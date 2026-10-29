# Reference Baseline Freeze: S3-Experiment Checkpoint

**Date**: 2026-10-17  
**Scope**: Locked reference baseline prior to launching the selective hard-negative experiment (S4).

---

## 1. Preserved Checkpoint & Artifact Provenance

| Property | Value |
| :--- | :--- |
| **Model Architecture** | `CVCSegmentationV3Net` (EfficientNet-B0 + GroupNorm/U-Net Decoder, 768×768) |
| **Source Checkpoint** | `runs/s3_experiment_hard_neg/checkpoints/final_segmentation_v3.pt` |
| **Checkpoint SHA256** | `6520e32629e5ea0cb45d4c979a4dbba91d577ca226d97e7f6e0b7a8fa6a84ef3` |
| **Best Val Checkpoint** | `runs/s3_experiment_hard_neg/checkpoints/best_segmentation_v3.pt` |
| **Best Val Checkpoint SHA256** | `13d3957eb6fb91bbcfc9fe8a32a6858e72750e32f05cf0019e078ba151978249` |
| **Exposure Ledger** | `runs/s3_experiment_hard_neg/exposure_ledger.jsonl` (1,000 updates, 4,000 exposures) |
| **Resolved Config** | `runs/s3_experiment_hard_neg/resolved_config.json` |
| **Training Summary** | `runs/s3_experiment_hard_neg/train_summary.json` |

---

## 2. Locked Reference Evaluation Metrics ($\tau=0.50$, Native Grid, 193 Studies)

Evaluated via `scripts/evaluate_hard_negative_experiment.py` on the frozen `dev_pilot_val` partition (185 positive studies, 8 negative studies, 132 unique patients):

| Metric | Reference Value |
| :--- | :---: |
| **Macro Binary Dice** | **0.4000** |
| **Macro Soft Dice** | 0.0501 |
| **Pooled Pixel Precision** | **27.91%** |
| **Pooled Pixel Recall** | **73.59%** |
| **Centerline Coverage @ 10px** | **79.6%** |
| **Missed Catheter Rate @ 10px** | **4.8%** |
| **Negative False Area** | **10,307 px (0.162%)** |

---

## 3. Reference Decision Gate Standards for S4 Selective Hard-Negative Experiment

Any successor model must satisfy:
1. **False Track Reduction**: Lower negative false-mask area ($< 0.162\%$).
2. **Precision Improvement**: Pooled precision $> 27.91\%$.
3. **Catheter Coverage Preservation**: Recall $\ge 73.0\%$, Centerline coverage @ 10px $\ge 79.0\%$.
4. **Safety Margin**: Missed catheter rate $< 5.5\%$.
5. **Mask Collapse Check**: Non-trivial foreground mask generation; no empty/collapsed masks on positive studies.
