# Numerical validation report — 16 September 2026

## Repeated cross-validation results

The manuscript-associated reproducibility script was executed against the processed 142-molecule C6–C10 boiling-point analysis table. The recomputed values agree with the reference results to the displayed precision:

| Model | Q²_CV | RMSE_CV |
|---|---:|---:|
| N | 0.939375 | 6.717811 |
| N + ABC | 0.967117 | 4.947505 |
| N + GA | 0.960005 | 5.456355 |
| N + DRSO | 0.959827 | 5.468479 |
| N + Randić | 0.959815 | 5.469306 |
| N + SDD | 0.959727 | 5.475295 |
| N + mDRSO | 0.959508 | 5.490159 |
| N + SO | 0.952255 | 5.961650 |
| N + M1 | 0.949606 | 6.124748 |
| N + M2 | 0.939286 | 6.722720 |

The categorical carbon-number robustness analysis is also reproduced:

- C(N): Q² = 0.938575
- C(N) + DRSO: Q² = 0.960659

## Permutation-analysis verification

The 10,000-permutation boiling-point analysis was recomputed using the documented randomization protocol. The complete 45-test exceedance matrix agrees with the reference analysis.

For DRSO/mDRSO, the exceedance counts are:

- C6: 79 / 298
- C7: 9 / 5
- C8: 0 / 2
- C9: 0 / 0
- C10: 0 / 0

With the plus-one correction, these counts reproduce the reported raw permutation p-values and the within-size Benjamini–Hochberg conclusions.

## Exact enumeration verification

A new enumeration through order 18 reproduced the reference equality-class counts, including:

- the first DRSO–SDD class-count separation at `n = 9`;
- the first SO–DRSO class-count separation at `n = 16`;
- at `n = 18`: 60,523 chemical trees, 1,725 SO classes, 1,708 DRSO/mDRSO classes, and 249 SDD classes.

The reference table additionally contains the independently checked order-19 and order-20 counts used in the manuscript. A complete order-20 rerun is computationally more demanding and was not repeated in the constrained validation environment used for this report.
