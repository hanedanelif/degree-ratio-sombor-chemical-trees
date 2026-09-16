#!/usr/bin/env python3
"""Reproducibility code for the DRSO chemical-tree manuscript.

This script implements the computational protocol used for the manuscript-associated
analyses and provides automated checks against the reference numerical results included
in this repository.

Computational protocol
----------------------
1. Exact chemical-tree equality classes:
   * DRSO/mDRSO equality is represented by the exact 6-component ratio profile.
   * SDD is evaluated with Fraction arithmetic.
   * SO equality is represented by an exact radical-coefficient signature.
2. Pooled QSPR repeated CV:
   RepeatedKFold(n_splits=10, n_repeats=10, random_state=20260812).
   All held-out predictions from all 100 folds are stacked, then Q2/RMSE/MAE
   are calculated on that pooled out-of-fold prediction vector.
3. Fixed-size QSPR:
   LeaveOneOut cross-validation.
4. Response-permutation association checks:
   10,000 permutations, in-sample simple-regression R2 test statistic,
   plus-one p-value correction. For the complete C6-C10 boiling-point analysis,
   one numpy.default_rng(20260812) stream is consumed in N=6,...,10 order and
   descriptor order DRSO,mDRSO,SO,SDD,M1,M2,Randic,ABC,GA. This reproduces
   the reference 10,000-permutation p-values.
5. Benjamini-Hochberg FDR is applied separately within each 9-descriptor
   fixed-size family.

The bundled 142-molecule boiling-point master table is sufficient to reproduce
all BP results in Sections 7.5 and the structural-resolution comparisons.
The five-endpoint C7/C8 analysis requires the 134-compound common QsarDB panel,
which can be rebuilt with src/qsardb_pipeline.py when network access is available.
"""

from __future__ import annotations

import argparse
import ast
import math
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, RepeatedKFold

SEED = 20260812
N_PERM = 10_000
DESCRIPTORS = ["DRSO", "mDRSO", "SO", "SDD", "M1", "M2", "Randic", "ABC", "GA"]

# ---------------------------------------------------------------------------
# Graph descriptors and exact equality signatures
# ---------------------------------------------------------------------------

def edge_type_counts(G: nx.Graph) -> Dict[Tuple[int, int], int]:
    d = dict(G.degree())
    out = {(i, j): 0 for i in range(1, 5) for j in range(i, 5)}
    for u, v in G.edges():
        i, j = sorted((d[u], d[v]))
        out[(i, j)] += 1
    return out


def ratio_profile(G: nx.Graph) -> Tuple[int, int, int, int, int, int]:
    m = edge_type_counts(G)
    return (
        m[(1, 1)] + m[(2, 2)] + m[(3, 3)] + m[(4, 4)],
        m[(1, 2)] + m[(2, 4)],
        m[(1, 3)],
        m[(1, 4)],
        m[(2, 3)],
        m[(3, 4)],
    )


def drso_numeric(G: nx.Graph) -> float:
    d = dict(G.degree())
    return sum(math.sqrt((d[u] / d[v]) ** 2 + (d[v] / d[u]) ** 2) for u, v in G.edges())


def mdrso_numeric(G: nx.Graph) -> float:
    d = dict(G.degree())
    return sum(1.0 / math.sqrt((d[u] / d[v]) ** 2 + (d[v] / d[u]) ** 2) for u, v in G.edges())


def sdd_signature(G: nx.Graph) -> Fraction:
    d = dict(G.degree())
    total = Fraction(0, 1)
    for u, v in G.edges():
        i, j = d[u], d[v]
        total += Fraction(i, j) + Fraction(j, i)
    return total


def _squarefree_decomposition(n: int) -> Tuple[int, int]:
    """Return a,s with sqrt(n)=a*sqrt(s), s square-free."""
    a = 1
    s = n
    p = 2
    while p * p <= s:
        sq = p * p
        while s % sq == 0:
            s //= sq
            a *= p
        p += 1
    return a, s


def so_signature(G: nx.Graph) -> Tuple[int, int, int, int, int, int]:
    """Exact coefficient signature in basis (1,sqrt2,sqrt5,sqrt10,sqrt13,sqrt17)."""
    basis = [1, 2, 5, 10, 13, 17]
    coeff = {q: 0 for q in basis}
    d = dict(G.degree())
    for u, v in G.edges():
        a, sf = _squarefree_decomposition(d[u] ** 2 + d[v] ** 2)
        if sf not in coeff:
            raise AssertionError(f"Unexpected SO radical sqrt({sf})")
        coeff[sf] += a
    return tuple(coeff[q] for q in basis)


def basic_descriptors(G: nx.Graph) -> Dict[str, float]:
    d = dict(G.degree())
    vals = {
        "DRSO": drso_numeric(G),
        "mDRSO": mdrso_numeric(G),
        "SO": sum(math.sqrt(d[u] ** 2 + d[v] ** 2) for u, v in G.edges()),
        "SDD": float(sdd_signature(G)),
        "M1": float(sum(x * x for x in d.values())),
        "M2": float(sum(d[u] * d[v] for u, v in G.edges())),
        "Randic": float(sum(1.0 / math.sqrt(d[u] * d[v]) for u, v in G.edges())),
        "ABC": float(sum(math.sqrt((d[u] + d[v] - 2) / (d[u] * d[v])) for u, v in G.edges())),
        "GA": float(sum(2.0 * math.sqrt(d[u] * d[v]) / (d[u] + d[v]) for u, v in G.edges())),
    }
    return vals

# ---------------------------------------------------------------------------
# Exact enumeration
# ---------------------------------------------------------------------------

def enumerate_chemical_trees(n_min: int = 4, n_max: int = 20) -> pd.DataFrame:
    rows = []
    for n in range(n_min, n_max + 1):
        graphs = [T for T in nx.generators.nonisomorphic_trees(n) if max(dict(T.degree()).values()) <= 4]
        sig_so = [so_signature(T) for T in graphs]
        sig_drso = [ratio_profile(T) for T in graphs]
        sig_sdd = [sdd_signature(T) for T in graphs]

        def class_sizes(xs: Sequence[object]) -> List[int]:
            counts: Dict[object, int] = {}
            for x in xs:
                counts[x] = counts.get(x, 0) + 1
            return list(counts.values())

        so_sizes = class_sizes(sig_so)
        drso_sizes = class_sizes(sig_drso)
        sdd_sizes = class_sizes(sig_sdd)

        rows.append({
            "n": n,
            "chemical_trees": len(graphs),
            "SO_JDM_distinct": len(set(sig_so)),
            "DRSO_mDRSO_distinct": len(set(sig_drso)),
            "SDD_distinct": len(set(sig_sdd)),
            "DRSO_graph_collisions": len(graphs) - len(set(sig_drso)),
            "SDD_graph_collisions": len(graphs) - len(set(sig_sdd)),
            "SO_graph_collisions": len(graphs) - len(set(sig_so)),
            "max_DRSO_class": max(drso_sizes),
            "max_SDD_class": max(sdd_sizes),
            "max_SO_class": max(so_sizes),
            # Because SO equality is equivalent to equality of the chemical-tree edge-type vector
            # on the fixed-order classes used in the manuscript, differences in SO-vs-DRSO
            # distinct counts measure edge-type/JDM classes lost by the ratio projection.
            "ratio_profiles_merging_JDM": len(set(sig_so)) - len(set(sig_drso)),
            "JDM_classes_lost_by_ratio_projection": len(set(sig_so)) - len(set(sig_drso)),
            "SDD_values_merging_ratio_profiles": len(set(sig_drso)) - len(set(sig_sdd)),
        })
        print(f"n={n:2d}: CT={len(graphs):6d}, SO={len(set(sig_so)):4d}, DRSO={len(set(sig_drso)):4d}, SDD={len(set(sig_sdd)):4d}")
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------

def fit_r2(X: np.ndarray, y: np.ndarray) -> float:
    model = LinearRegression().fit(X, y)
    return float(r2_score(y, model.predict(X)))


def pooled_repeated_cv(X: np.ndarray, y: np.ndarray, splits=None) -> Dict[str, float]:
    if splits is None:
        splits = list(RepeatedKFold(n_splits=10, n_repeats=10, random_state=SEED).split(X))
    y_true, y_pred = [], []
    for train, test in splits:
        model = LinearRegression().fit(X[train], y[train])
        pred = model.predict(X[test])
        y_true.extend(y[test].tolist())
        y_pred.extend(pred.tolist())
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "Q2_CV": float(r2_score(y_true, y_pred)),
        "RMSE_CV": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE_CV": float(mean_absolute_error(y_true, y_pred)),
    }


def loocv_metrics(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    X = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    loo = LeaveOneOut()
    preds = np.empty(len(y), dtype=float)
    for train, test in loo.split(X):
        preds[test] = LinearRegression().fit(X[train], y[train]).predict(X[test])
    return {
        "R2": fit_r2(X, y),
        "Q2_LOO": float(r2_score(y, preds)),
        "RMSE_LOO": float(math.sqrt(mean_squared_error(y, preds))),
        "MAE_LOO": float(mean_absolute_error(y, preds)),
    }


def bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, monotone and capped at 1."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj_ranked = ranked * m / np.arange(1, m + 1)
    adj_ranked = np.minimum.accumulate(adj_ranked[::-1])[::-1]
    adj_ranked = np.minimum(adj_ranked, 1.0)
    out = np.empty(m)
    out[order] = adj_ranked
    return out


def simple_r2_fast(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    r = np.corrcoef(x, y)[0, 1]
    return float(r * r)

# ---------------------------------------------------------------------------
# Complete C6-C10 boiling-point analysis
# ---------------------------------------------------------------------------

def reproduce_bp(master_csv: Path, outdir: Path) -> Dict[str, pd.DataFrame]:
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(master_csv)
    required = {"BP", "N", *DESCRIPTORS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Master CSV missing required columns: {sorted(missing)}")

    y = df["BP"].to_numpy(float)
    # One split list reused for every comparator, ensuring identical fold assignments.
    common_splits = list(RepeatedKFold(n_splits=10, n_repeats=10, random_state=SEED).split(np.arange(len(df))))

    # Primary pooled models used in the manuscript.
    primary_rows = []
    for label, X in [
        ("N", df[["N"]].to_numpy(float)),
        ("DRSO", df[["DRSO"]].to_numpy(float)),
        ("N+DRSO", df[["N", "DRSO"]].to_numpy(float)),
    ]:
        ins = fit_r2(X, y)
        cv = pooled_repeated_cv(X, y, common_splits)
        primary_rows.append({"model": label, "R2": ins, **cv})

    # Categorical carbon-number robustness analysis.
    cat = pd.get_dummies(df["N"].astype(str), prefix="N", dtype=float).to_numpy()
    for label, X in [("C(N)", cat), ("C(N)+DRSO", np.column_stack([cat, df["DRSO"].to_numpy(float)]))]:
        ins = fit_r2(X, y)
        cv = pooled_repeated_cv(X, y, common_splits)
        primary_rows.append({"model": label, "R2": ins, **cv})
    primary = pd.DataFrame(primary_rows)
    primary.to_csv(outdir / "pooled_bp_results.csv", index=False)

    # Table 6: N + each fixed descriptor.
    baseline = pooled_repeated_cv(df[["N"]].to_numpy(float), y, common_splits)
    table6_rows = [{"model": "N (size only)", "Q2_CV": baseline["Q2_CV"], "delta_Q2_vs_N": np.nan,
                    "RMSE_CV": baseline["RMSE_CV"], "MAE_CV": baseline["MAE_CV"]}]
    for d in ["ABC", "GA", "DRSO", "Randic", "SDD", "mDRSO", "SO", "M1", "M2"]:
        X = df[["N", d]].to_numpy(float)
        cv = pooled_repeated_cv(X, y, common_splits)
        table6_rows.append({"model": f"N + {d}", "Q2_CV": cv["Q2_CV"],
                            "delta_Q2_vs_N": cv["Q2_CV"] - baseline["Q2_CV"],
                            "RMSE_CV": cv["RMSE_CV"], "MAE_CV": cv["MAE_CV"]})
    table6 = pd.DataFrame(table6_rows)
    table6.to_csv(outdir / "table6_size_adjusted_bp.csv", index=False)

    # Complete fixed-size LOOCV for all 9 descriptors.
    fixed_rows = []
    for N in range(6, 11):
        sub = df[df["N"] == N]
        for d in DESCRIPTORS:
            met = loocv_metrics(sub[d].to_numpy(float), sub["BP"].to_numpy(float))
            fixed_rows.append({"N": N, "n": len(sub), "descriptor": d, **met})
    fixed = pd.DataFrame(fixed_rows)
    fixed.to_csv(outdir / "bp_c6_c10_loocv.csv", index=False)

    # Exact historical 10,000-permutation stream. DO NOT reset RNG within cells.
    rng = np.random.default_rng(SEED)
    perm_rows = []
    for N in range(6, 11):
        sub = df[df["N"] == N]
        yy = sub["BP"].to_numpy(float)
        p_block = []
        block_rows = []
        for d in DESCRIPTORS:
            xx = sub[d].to_numpy(float)
            observed = simple_r2_fast(xx, yy)
            exceed = 0
            for _ in range(N_PERM):
                if simple_r2_fast(xx, rng.permutation(yy)) >= observed - 1e-15:
                    exceed += 1
            p = (exceed + 1) / (N_PERM + 1)
            p_block.append(p)
            block_rows.append({"N": N, "descriptor": d, "R2_observed": observed,
                               "p_permutation": p, "exceedances": exceed})
        q = bh_adjust(p_block)
        for row, qq in zip(block_rows, q):
            row["p_BH_FDR_within_N"] = float(qq)
            perm_rows.append(row)
    perm = pd.DataFrame(perm_rows)
    perm.to_csv(outdir / "bp_permutation_10000_bh.csv", index=False)

    # Structural-resolution Table 5.
    struct_rows = []
    for N in range(6, 11):
        sub = df[df["N"] == N]
        if "sig_DRSO" in sub.columns:
            nclasses = sub["sig_DRSO"].nunique()
        else:
            nclasses = sub[["R_equal", "R_2", "R_3", "R_4", "R_3_2", "R_4_3"]].drop_duplicates().shape[0]
        dr = fixed[(fixed.N == N) & (fixed.descriptor == "DRSO")].iloc[0]
        md = fixed[(fixed.N == N) & (fixed.descriptor == "mDRSO")].iloc[0]
        struct_rows.append({"N": N, "isomers": len(sub), "DRSO_equality_classes": nclasses,
                            "collision_deficit": len(sub) - nclasses,
                            "equality_class_isomer_ratio": nclasses / len(sub),
                            "DRSO_Q2_LOO": dr.Q2_LOO, "mDRSO_Q2_LOO": md.Q2_LOO})
    structural = pd.DataFrame(struct_rows)
    structural.to_csv(outdir / "table5_structural_resolution_bp.csv", index=False)

    # Representation-loss / collision-oracle floors.
    oracle_rows = []
    for N in range(6, 11):
        sub = df[df["N"] == N].copy()
        yy = sub["BP"].to_numpy(float)
        tss = float(np.sum((yy - yy.mean()) ** 2))
        for index_name, sig_col in [("SO", "sig_SO"), ("DRSO/mDRSO", "sig_DRSO"), ("SDD", "sig_SDD")]:
            means = sub.groupby(sig_col)["BP"].transform("mean").to_numpy(float)
            sse = float(np.sum((yy - means) ** 2))
            oracle_rows.append({"N": N, "n": len(sub), "index": index_name,
                                "n_classes": sub[sig_col].nunique(),
                                "collision_excess": len(sub) - sub[sig_col].nunique(),
                                "L_I_BP": sse,
                                "empirical_oracle_R2_ceiling": 1.0 - sse / tss if tss > 0 else np.nan,
                                "empirical_oracle_RMSE_floor": math.sqrt(sse / len(sub)),
                                "empirical_oracle_MAE_floor": float(np.mean(np.abs(yy - means)))})
    oracle = pd.DataFrame(oracle_rows)
    oracle.to_csv(outdir / "bp_partition_representation_limits.csv", index=False)

    return {"primary": primary, "table6": table6, "fixed": fixed, "permutation": perm,
            "structural": structural, "oracle": oracle}

# ---------------------------------------------------------------------------
# Manuscript reference-result checks
# ---------------------------------------------------------------------------
EXPECTED = {
    "N_Q2": 0.939375,
    "N_RMSE": 6.717811,
    "N_DRSO_Q2": 0.959827,
    "N_DRSO_RMSE": 5.468479,
    "CAT_N_Q2": 0.938575,
    "CAT_N_DRSO_Q2": 0.960659,
    "C6_DRSO_LOO": 0.935136,
    "C7_DRSO_LOO": 0.709319,
    "C8_DRSO_LOO": 0.503153,
    "C9_DRSO_LOO": 0.377680,
    "C10_DRSO_LOO": 0.180778,
}


def manuscript_checks(outputs: Dict[str, pd.DataFrame], tol=5e-7) -> None:
    primary = outputs["primary"].set_index("model")
    fixed = outputs["fixed"]
    checks = {
        "N_Q2": primary.loc["N", "Q2_CV"],
        "N_RMSE": primary.loc["N", "RMSE_CV"],
        "N_DRSO_Q2": primary.loc["N+DRSO", "Q2_CV"],
        "N_DRSO_RMSE": primary.loc["N+DRSO", "RMSE_CV"],
        "CAT_N_Q2": primary.loc["C(N)", "Q2_CV"],
        "CAT_N_DRSO_Q2": primary.loc["C(N)+DRSO", "Q2_CV"],
    }
    for N in range(6, 11):
        checks[f"C{N}_DRSO_LOO"] = fixed[(fixed.N == N) & (fixed.descriptor == "DRSO")].iloc[0].Q2_LOO
    bad = []
    for key, expected in EXPECTED.items():
        got = float(checks[key])
        if abs(got - expected) > tol:
            bad.append((key, got, expected))
    if bad:
        raise AssertionError("Manuscript reference-result mismatch: " + repr(bad))

    # Reference permutation exceedance counts provide an exact consistency check for the RNG protocol.
    perm = outputs["permutation"].set_index(["N", "descriptor"])
    expected_exceed = {
        (6, "DRSO"): 79, (6, "mDRSO"): 298,
        (7, "DRSO"): 9, (7, "mDRSO"): 5,
        (8, "DRSO"): 0, (8, "mDRSO"): 2,
        (9, "DRSO"): 0, (9, "mDRSO"): 0,
        (10, "DRSO"): 0, (10, "mDRSO"): 0,
    }
    for key, expected in expected_exceed.items():
        got = int(perm.loc[key, "exceedances"])
        if got != expected:
            raise AssertionError(f"Permutation stream mismatch {key}: got {got}, expected {expected}")
    print("PASS: manuscript-associated BP/CV/permutation reference checks.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bp", "enumeration", "all"], default="bp")
    ap.add_argument("--bp-master", type=Path,
                    default=Path(__file__).resolve().parent / "data" / "processed" / "alkane_bp_c6_c10_142_master.csv")
    ap.add_argument("--outdir", type=Path,
                    default=Path(__file__).resolve().parent / "results" / "recomputed")
    ap.add_argument("--n-max", type=int, default=20)
    args = ap.parse_args()

    if args.mode in {"bp", "all"}:
        outputs = reproduce_bp(args.bp_master, args.outdir)
        manuscript_checks(outputs)
        print(outputs["table6"].to_string(index=False))

    if args.mode in {"enumeration", "all"}:
        enum = enumerate_chemical_trees(4, args.n_max)
        args.outdir.mkdir(parents=True, exist_ok=True)
        path = args.outdir / f"exact_enumeration_n4_n{args.n_max}.csv"
        enum.to_csv(path, index=False)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
