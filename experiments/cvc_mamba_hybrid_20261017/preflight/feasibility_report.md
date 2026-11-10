# CVC-Mamba-Hybrid-01: Three-Arm Feasibility & Profiling Report (768px)

**Date**: 2026-10-17  
**Hardware**: NVIDIA GeForce RTX 3050 Laptop GPU (4096 MB VRAM)  
**Configuration**: Micro-batch 1, Accumulation 4, Mixed Precision (AMP), Resolution 768px  

---

## Three-Arm Profiling Matrix

| Arm                | Params Total   | 512px Step   | 768px Step   | Save/Reload   | Sec / Update (768px)   | Peak VRAM (Alloc / Res)   | 100-Upd Stability   | Verdict           |
|:-------------------|:---------------|:-------------|:-------------|:--------------|:-----------------------|:--------------------------|:--------------------|:------------------|
| Arm_A_Continuation | 4,176,941      | PASSED       | PASSED       | PASSED        | 0.33s                  | 931 / 1090 MB             | PASSED (0 ovf)      | FEASIBLE_AT_768PX |
| Arm_B_Mamba_Hybrid | 4,219,053      | PASSED       | PASSED       | PASSED        | 7.09s                  | 1007 / 1154 MB            | PASSED (0 ovf)      | FEASIBLE_AT_768PX |
| Arm_C_Conv_Control | 4,220,749      | PASSED       | PASSED       | PASSED        | 0.36s                  | 952 / 1094 MB             | PASSED (0 ovf)      | FEASIBLE_AT_768PX |

---

## Conclusions
1. All three arms (Arm A Continuation, Arm B Mamba Hybrid, Arm C Conv Control) share the exact frozen S4 weights and pass 100% of functional and stability requirements at 768×768 resolution.
2. Capacity matching: Arm B adds 42,112 parameters; Arm C adds 43,808 parameters (matched within 4.0%).
3. Memory headroom is abundant (>2,700 MB available on 4 GB GPU).
4. All three arms are 100% FEASIBLE at 768px for full 1,000-update pilot training.
