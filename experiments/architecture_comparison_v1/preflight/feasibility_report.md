# Architecture Candidates Feasibility & Profiling Report (768px)

**Date**: 2026-10-17  
**Hardware**: NVIDIA GeForce RTX 3050 Laptop GPU (4096 MB VRAM)  
**Configuration**: Micro-batch 1, Effective batch 4, Mixed Precision (AMP), Resolution 768px  

---

## Candidate Profiling Summary

| Candidate Architecture    | Params Total   | 512px Step   | 768px Step   | Save/Reload   | Sec / Update (768px)   | Peak VRAM (Alloc / Res)   | 100-Upd Stability   | Verdict           |
|:--------------------------|:---------------|:-------------|:-------------|:--------------|:-----------------------|:--------------------------|:--------------------|:------------------|
| EfficientNet_B0_Baseline  | 4,176,941      | PASSED       | PASSED       | PASSED        | 0.33s                  | 933 / 1078 MB             | PASSED (0 ovf)      | FEASIBLE_AT_768PX |
| SegFormer_B0              | 3,714,401      | PASSED       | PASSED       | PASSED        | 0.46s                  | 1262 / 1374 MB            | PASSED (0 ovf)      | FEASIBLE_AT_768PX |
| Small_HRNet_W18           | 5,479,329      | PASSED       | PASSED       | PASSED        | 0.96s                  | 1317 / 1620 MB            | PASSED (0 ovf)      | FEASIBLE_AT_768PX |
| EfficientNet_Mamba_Hybrid | 4,297,517      | PASSED       | PASSED       | PASSED        | 2.75s                  | 985 / 1066 MB             | PASSED (0 ovf)      | FEASIBLE_AT_768PX |

---

## Feasibility Conclusions
1. All four candidate architectures successfully passed instantiation, 512px and 768px forward/loss/backward passes, checkpoint resumability, timed throughput profiling, and 100-update stability without gradient overflow retries.
2. VRAM headroom is abundant across all candidates (peak allocated VRAM < 1,500 MB against 4,096 MB device capacity).
3. The four candidates are 100% FEASIBLE at the primary 768px resolution for full pilot training.
