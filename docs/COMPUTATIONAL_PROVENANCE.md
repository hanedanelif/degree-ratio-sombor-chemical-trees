# Computational provenance and protocol selection

## Purpose

This document records the computational provenance of the manuscript-associated analyses and identifies the procedures used to generate the reported numerical results.

## Exact chemical-tree enumeration

The exact enumeration implementation uses `networkx.generators.nonisomorphic_trees` and retains trees with maximum degree at most four. Equality classes are represented without floating-point comparisons:

- DRSO/mDRSO equality is represented by the exact six-component degree-ratio profile;
- SDD is evaluated using exact rational arithmetic;
- SO equality is represented by exact radical-coefficient signatures.

The reference enumeration table reports the first DRSO–SDD class-count separation at `n = 9`, the first SO–DRSO class-count separation at `n = 16`, and 366,319 chemical trees at `n = 20`.

## QDB.122 data engineering

The data-engineering workflow is implemented in `src/qsardb_pipeline.py`. It reconstructs the common 134-compound panel from QsarDB QDB.122, converts alkane names to hydrogen-suppressed carbon trees, verifies graph identities, and computes the pre-specified descriptor panel.

## Repeated cross-validation protocol

The repeated cross-validation analysis reported in the manuscript uses:

```python
RepeatedKFold(n_splits=10, n_repeats=10, random_state=20260812)
```

All held-out predictions from the 100 validation folds are pooled into one out-of-fold prediction vector, from which Q², RMSE, and MAE are calculated. The same fold assignments are used for all model comparisons.

Earlier exploratory calculations used a different aggregation of repeated out-of-fold predictions. Those exploratory results are not used in the manuscript.

## Boiling-point response-permutation protocol

For the complete C6–C10 boiling-point analysis, the manuscript-associated randomization protocol uses:

- 10,000 response permutations per test;
- the in-sample simple-regression R² statistic;
- `numpy.random.default_rng(20260812)`;
- one continuous pseudo-random number stream;
- carbon-number order C6, C7, C8, C9, C10;
- descriptor order `DRSO, mDRSO, SO, SDD, M1, M2, Randic, ABC, GA`;
- plus-one p-value correction;
- Benjamini–Hochberg adjustment within each nine-descriptor size family.

The automated test suite verifies the corresponding permutation exceedance counts used in the reference analysis.

## Reference outputs

The directory `results/reference/` contains the numerical reference outputs corresponding to the tables and computational claims reported in the manuscript. Recomputed outputs are written separately to `results/recomputed/`.

The distinction between reference and recomputed outputs is organizational only; the reference tables are included to permit direct numerical comparison during reproducibility checks.
