# Predefined Decision Rule: Seed-43 Confirmation Benchmark

**Date**: 2026-10-17  
**Scope**: Explicit decision criteria locked prior to launching Seed-43 confirmation runs.

---

## 1. Primary Success Criteria (Arm B Selective Hard Neg vs Arm A Uniform Control at $\tau=0.50$)

For the seed-43 confirmation to pass:
1. **False-Mask Area Reduction**:
   - Negative false-mask area must be lower in Arm B than in Arm A (Seed 42 achieved 0.102% vs 0.125%).
   - Target: $\le 0.110\%$ on negative studies (and lower than Control).
2. **Precision Improvement**:
   - Pooled precision in Arm B must exceed Arm A (Seed 42 achieved 31.35% vs 31.02%).
   - Target: $> 31.0\%$.
3. **Catheter Coverage Preservation (Predefined Tolerance)**:
   - Engineering pilot criterion: **Maximum 1.0 percentage point recall reduction** compared to the baseline reference (73.59%).
   - Minimum acceptable pooled recall: **$\ge 72.59\%$**.
   - Centerline coverage @ 10px: **$\ge 79.0\%$**.
   - Missed catheter rate: **$\le 5.0\%$**.
4. **Non-Triviality / No Mask Collapse**:
   - Predicted mask pixels on positives must remain non-empty and consistent with anatomical catheter tracks across the 50-study audit panel.
5. **Separate Seed Reporting**:
   - Both seeds (Seed 42 and Seed 43) will be reported side-by-side in distinct columns.
   - Statistical Note: A paired confidence interval crossing zero does not prove equivalence. Point estimate preservation ($\ge 72.59\%$) and paired change versus baseline are the primary engineering decision criteria. Under strict interpretation requiring the *absolute* recall 95% CI lower bound to exceed 72.59%, the finite validation cohort (185 positives) yields [71.48%, 77.62%], whose lower bound falls slightly below 72.59% despite positive paired change (+0.88%).
   - Exposure Caveat: Only Seed 43 used the repaired in-place AMP retry mechanism guaranteeing identical positive batch exposure parity; Seed 42 retains the legacy skipped-update exposure caveat.

---

## 2. Decision Branches

- **Pass (Confirmation Succeeded / Proceed to Expansion Comparison)**:
  - Both Seed 42 and Seed 43 demonstrate lower negative false area, higher precision, and recall point estimates within the $\le 1.0\%$ drop tolerance, with positive paired recall change.
  - Action: Proceed directly to Item 6 (Controlled 2,500-update expansion experiment comparing current cohort vs expanded cohort).
- **Fail (Inconsistent or Coverage Loss)**:
  - Seed 43 shows $> 1.0\%$ point-estimate recall drop or fails to suppress false area compared to uniform control.
  - Action: Re-tune hard negative blending ratio (e.g., 25% hard / 75% uniform) or investigate thin-line supervision in an isolated ablation before expanding.
