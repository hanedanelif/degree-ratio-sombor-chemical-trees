#!/usr/bin/env python3
"""
Five-endpoint reproducibility audit for the DRSO chemical-tree manuscript.

This script complements `drso_reproducibility.py` and the data-engineering
pipeline `src/qsardb_pipeline.py`.

What is reproduced from the QDB.122 common 134-compound panel
-------------------------------------------------------------
1. Pooled repeated-CV heat-capacity models reported in Section 7.2:
   N, DRSO, and N+DRSO.
2. Complete C7/C8 leave-one-out results for the fixed nine-descriptor panel.
3. The manuscript DRSO/mDRSO five-endpoint summary (HC, D, RI, E, BP).
4. C8 DRSO representation ceilings and linear DRSO R^2 values.
5. The qualitative response-permutation conclusions used in the manuscript.

Important permutation provenance note
-------------------------------------
The archived manuscript permutation p-values were produced by the original
final analysis stream. The historical random-number stream consumption order
for that five-endpoint run is not preserved in the public package. Therefore
this script does NOT claim to regenerate those historical Monte-Carlo p-values
digit-for-digit.

Instead it:
  * verifies the archived canonical raw p-values recorded for DRSO/mDRSO;
  * proves the manuscript BH-FDR significance conclusions from those p-values
    (family size = 9); and
  * optionally runs a fresh deterministic 10,000-permutation audit using
    seed 20260812 as an independent robustness check.

For a BH family of m=9 tests:
  * raw p > 0.05 guarantees BH-adjusted p > 0.05;
  * 9*p <= 0.05 guarantees BH-adjusted p <= 0.05 regardless of rank.
All archived DRSO/mDRSO p-values used by the manuscript fall into one of
these two unambiguous cases.

Usage
-----
From the repository root:

    python src/DRSO_five_endpoint_analysis.py

This rebuilds QDB.122 online through `src/qsardb_pipeline.py`.

To reuse a previously built panel:

    python src/DRSO_five_endpoint_analysis.py \
        --panel data/processed/DRSO_QSPR_master_QDB122.csv

For an additional fresh Monte-Carlo permutation audit:

    python src/DRSO_five_endpoint_analysis.py --fresh-permutation

Outputs are written to `results/recomputed/` by default.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, RepeatedKFold

from qsardb_pipeline import build_common_property_panel, validate_and_describe

SEED = 20260812
N_PERM = 10_000
BH_FAMILY_SIZE = 9

DESCRIPTORS = ["DRSO", "mDRSO", "SO", "SDD", "M1", "M2", "Randic", "ABC", "GA"]
RESPONSES = ["HC", "D", "RI", "E", "BP"]
RESPONSE_COLUMN = {"HC": "HC", "D": "D_g_L", "RI": "RI", "E": "E", "BP": "BP"}

EXPECTED_PANEL_COUNTS = {6: 4, 7: 9, 8: 18, 9: 34, 10: 69}

EXPECTED_FIXED = {
    ("HC", 7, "DRSO"): -0.4905538623346244,
    ("HC", 7, "mDRSO"): -0.4628472217491910,
    ("HC", 8, "DRSO"): -0.1649200912369766,
    ("HC", 8, "mDRSO"): -0.1505452379775189,
    ("D", 7, "DRSO"): -0.5738056498962292,
    ("D", 7, "mDRSO"): -0.5147039821453461,
    ("D", 8, "DRSO"): -0.2436472720269886,
    ("D", 8, "mDRSO"): -0.1899332879012860,
    ("RI", 7, "DRSO"): -0.5252030075519263,
    ("RI", 7, "mDRSO"): -0.5051814760234081,
    ("RI", 8, "DRSO"): -0.2694919145600247,
    ("RI", 8, "mDRSO"): -0.2143179437442091,
    ("E", 7, "DRSO"): 0.5831947743544194,
    ("E", 7, "mDRSO"): 0.7631726024505998,
    ("E", 8, "DRSO"): 0.4760791852251496,
    ("E", 8, "mDRSO"): 0.6911284602651718,
    ("BP", 7, "DRSO"): 0.7093192734756651,
    ("BP", 7, "mDRSO"): 0.7766965852441960,
    ("BP", 8, "DRSO"): 0.5031526655130859,
    ("BP", 8, "mDRSO"): 0.5790112520078505,
}

EXPECTED_HC = {
    "N_R2": 0.9741474273262573,
    "N_Q2": 0.9734877149318608,
    "DRSO_R2": 0.3136680713044698,
    "DRSO_Q2": 0.2911188396309992,
    "N_DRSO_R2": 0.9747926914023451,
    "N_DRSO_Q2": 0.9737532579080960,
}

EXPECTED_C8 = {
    "HC": (0.5931813242424110, 0.001747),
    "D": (0.9997730302034796, 0.044244),
    "RI": (0.9996206822707122, 0.043708),
    "E": (0.9636894654765068, 0.630977),
    "BP": (0.9964177932191036, 0.660881),
}

CANONICAL_PERMUTATION_P = {
    ("HC", 7, "DRSO"): 0.834617,
    ("HC", 7, "mDRSO"): 0.757524,
    ("D", 7, "DRSO"): 0.664434,
    ("D", 7, "mDRSO"): 0.797620,
    ("RI", 7, "DRSO"): 0.432457,
    ("RI", 7, "mDRSO"): 0.549745,
    ("E", 7, "DRSO"): 0.002100,
    ("E", 7, "mDRSO"): 0.000500,
    ("BP", 7, "DRSO"): 0.000700,
    ("BP", 7, "mDRSO"): 0.000800,
    ("HC", 8, "DRSO"): 0.872013,
    ("HC", 8, "mDRSO"): 0.744226,
    ("D", 8, "DRSO"): 0.409959,
    ("D", 8, "mDRSO"): 0.292771,
    ("RI", 8, "DRSO"): 0.410959,
    ("RI", 8, "mDRSO"): 0.316068,
    ("E", 8, "DRSO"): 0.000100,
    ("E", 8, "mDRSO"): 0.000100,
    ("BP", 8, "DRSO"): 0.000100,
    ("BP", 8, "mDRSO"): 0.000200,
}

EXPECTED_SIGNIFICANCE = {"HC": False, "D": False, "RI": False, "E": True, "BP": True}


def fit_r2(X: np.ndarray, y: np.ndarray) -> float:
    model = LinearRegression().fit(X, y)
    return float(r2_score(y, model.predict(X)))


def pooled_repeated_cv(X: np.ndarray, y: np.ndarray, splits) -> Dict[str, float]:
    y_true, y_pred = [], []
    for train, test in splits:
        model = LinearRegression().fit(X[train], y[train])
        pred = model.predict(X[test])
        y_true.extend(y[test].tolist())
        y_pred.extend(pred.tolist())

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "Q2_CV": float(r2_score(y_true, y_pred)),
        "RMSE_CV": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE_CV": float(mean_absolute_error(y_true, y_pred)),
    }


def loocv_metrics(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    X = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    preds = np.empty(len(y), dtype=float)

    for train, test in LeaveOneOut().split(X):
        model = LinearRegression().fit(X[train], y[train])
        preds[test] = model.predict(X[test])

    return {
        "R2": fit_r2(X, y),
        "Q2_LOO": float(r2_score(y, preds)),
        "RMSE_LOO": float(math.sqrt(mean_squared_error(y, preds))),
        "MAE_LOO": float(mean_absolute_error(y, preds)),
    }


def simple_r2(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    r = np.corrcoef(x, y)[0, 1]
    return float(r * r)


def load_panel(panel_path: Path | None) -> pd.DataFrame:
    if panel_path is None:
        panel = validate_and_describe(build_common_property_panel())
    else:
        panel = pd.read_csv(panel_path)

    if "D_g_L" not in panel.columns:
        if "D_raw" in panel.columns:
            panel["D_g_L"] = panel["D_raw"]
        elif "D" in panel.columns:
            panel["D_g_L"] = panel["D"]

    required = {
        "ID", "N", "HC", "D_g_L", "RI", "E", "BP",
        *DESCRIPTORS,
        "R_equal", "R_2", "R_3", "R_4", "R_3_2", "R_4_3",
    }
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError("Missing panel columns: " + ", ".join(sorted(missing)))

    if len(panel) != 134:
        raise AssertionError(f"Expected 134 compounds; found {len(panel)}.")

    counts = panel["N"].astype(int).value_counts().sort_index().to_dict()
    if counts != EXPECTED_PANEL_COUNTS:
        raise AssertionError(
            f"Unexpected carbon-number composition: {counts}; expected {EXPECTED_PANEL_COUNTS}."
        )

    if panel[["HC", "D_g_L", "RI", "E", "BP"]].isna().any().any():
        raise AssertionError("The common five-response panel contains missing responses.")

    return panel.sort_values(["N", "ID"]).reset_index(drop=True)


def pooled_hc(panel: pd.DataFrame) -> pd.DataFrame:
    y = panel["HC"].to_numpy(float)
    splits = list(
        RepeatedKFold(
            n_splits=10,
            n_repeats=10,
            random_state=SEED,
        ).split(np.arange(len(panel)))
    )

    models = [
        ("N", panel[["N"]].to_numpy(float)),
        ("DRSO", panel[["DRSO"]].to_numpy(float)),
        ("N+DRSO", panel[["N", "DRSO"]].to_numpy(float)),
    ]

    rows = []
    for name, X in models:
        rows.append({"model": name, "R2": fit_r2(X, y), **pooled_repeated_cv(X, y, splits)})
    return pd.DataFrame(rows)


def fixed_size_all_descriptors(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for response in RESPONSES:
        ycol = RESPONSE_COLUMN[response]
        for N in (7, 8):
            sub = panel[panel["N"] == N]
            y = sub[ycol].to_numpy(float)
            for descriptor in DESCRIPTORS:
                met = loocv_metrics(sub[descriptor].to_numpy(float), y)
                rows.append(
                    {
                        "response": response,
                        "N": N,
                        "n": len(sub),
                        "descriptor": descriptor,
                        **met,
                    }
                )
    return pd.DataFrame(rows)


def fixed_summary(fixed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for response in RESPONSES:
        row = {"response": response}
        for N in (7, 8):
            for descriptor in ("DRSO", "mDRSO"):
                val = fixed[
                    (fixed["response"] == response)
                    & (fixed["N"] == N)
                    & (fixed["descriptor"] == descriptor)
                ].iloc[0]["Q2_LOO"]
                row[f"C{N}_{descriptor}_Q2_LOO"] = float(val)
        rows.append(row)
    return pd.DataFrame(rows)


def c8_ceilings(panel: pd.DataFrame) -> pd.DataFrame:
    sub = panel[panel["N"] == 8].copy()
    ratio_cols = ["R_equal", "R_2", "R_3", "R_4", "R_3_2", "R_4_3"]

    rows = []
    for response in RESPONSES:
        ycol = RESPONSE_COLUMN[response]
        y = sub[ycol].to_numpy(float)
        class_means = sub.groupby(ratio_cols, dropna=False)[ycol].transform("mean").to_numpy(float)

        loss = float(np.sum((y - class_means) ** 2))
        tss = float(np.sum((y - y.mean()) ** 2))
        ceiling = 1.0 - loss / tss
        linear_r2 = simple_r2(sub["DRSO"].to_numpy(float), y)

        rows.append(
            {
                "response": response,
                "n_molecules": len(sub),
                "n_DRSO_classes": sub[ratio_cols].drop_duplicates().shape[0],
                "L_DRSO": loss,
                "R2_DRSO_max": ceiling,
                "linear_DRSO_R2": linear_r2,
            }
        )
    return pd.DataFrame(rows)


def canonical_permutation_table() -> pd.DataFrame:
    rows = []
    for (response, N, descriptor), p in CANONICAL_PERMUTATION_P.items():
        if p > 0.05:
            bh_status = "not significant"
            guaranteed = True
        elif BH_FAMILY_SIZE * p <= 0.05:
            bh_status = "significant"
            guaranteed = True
        else:
            bh_status = "requires full 9-test family"
            guaranteed = False

        rows.append(
            {
                "response": response,
                "N": N,
                "descriptor": descriptor,
                "archived_raw_permutation_p": p,
                "BH_family_size": BH_FAMILY_SIZE,
                "BH_conclusion_from_raw_p": bh_status,
                "BH_conclusion_guaranteed": guaranteed,
            }
        )

    return pd.DataFrame(rows).sort_values(["N", "response", "descriptor"]).reset_index(drop=True)


def verify_archived_permutation_claims() -> pd.DataFrame:
    table = canonical_permutation_table()

    if not table["BH_conclusion_guaranteed"].all():
        bad = table.loc[~table["BH_conclusion_guaranteed"]]
        raise AssertionError(
            "Some archived p-values do not determine the BH conclusion:\n"
            + bad.to_string(index=False)
        )

    for response, expected_sig in EXPECTED_SIGNIFICANCE.items():
        block = table[table["response"] == response]
        got = (block["BH_conclusion_from_raw_p"] == "significant").tolist()
        if len(got) != 4 or any(x != expected_sig for x in got):
            raise AssertionError(f"Archived BH conclusion mismatch for {response}: {got}")

    return table


def fresh_permutation_audit(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for response_index, response in enumerate(RESPONSES):
        ycol = RESPONSE_COLUMN[response]

        for N in (7, 8):
            sub = panel[panel["N"] == N]
            y = sub[ycol].to_numpy(float)

            for descriptor_index, descriptor in enumerate(("DRSO", "mDRSO")):
                x = sub[descriptor].to_numpy(float)
                observed = simple_r2(x, y)

                seed_seq = np.random.SeedSequence([SEED, response_index, N, descriptor_index])
                rng = np.random.default_rng(seed_seq)

                exceed = 0
                for _ in range(N_PERM):
                    if simple_r2(x, rng.permutation(y)) >= observed - 1e-15:
                        exceed += 1

                p = (exceed + 1) / (N_PERM + 1)
                rows.append(
                    {
                        "response": response,
                        "N": N,
                        "descriptor": descriptor,
                        "R2_observed": observed,
                        "exceedances": exceed,
                        "fresh_permutation_p": p,
                    }
                )

    out = pd.DataFrame(rows)

    for response, expected in EXPECTED_SIGNIFICANCE.items():
        vals = (out[out["response"] == response]["fresh_permutation_p"] < 0.05).tolist()
        if len(vals) != 4 or any(v != expected for v in vals):
            raise AssertionError(
                f"Fresh permutation robustness pattern mismatch for {response}: {vals}"
            )

    return out


def assert_close(label: str, got: float, expected: float, tol: float) -> None:
    if not np.isclose(got, expected, rtol=0.0, atol=tol):
        raise AssertionError(
            f"{label}: got {got:.15g}, expected {expected:.15g}, "
            f"|diff|={abs(got - expected):.3g}"
        )


def manuscript_checks(hc: pd.DataFrame, fixed: pd.DataFrame, ceilings: pd.DataFrame) -> None:
    hc_idx = hc.set_index("model")
    got_hc = {
        "N_R2": float(hc_idx.loc["N", "R2"]),
        "N_Q2": float(hc_idx.loc["N", "Q2_CV"]),
        "DRSO_R2": float(hc_idx.loc["DRSO", "R2"]),
        "DRSO_Q2": float(hc_idx.loc["DRSO", "Q2_CV"]),
        "N_DRSO_R2": float(hc_idx.loc["N+DRSO", "R2"]),
        "N_DRSO_Q2": float(hc_idx.loc["N+DRSO", "Q2_CV"]),
    }

    for label, expected in EXPECTED_HC.items():
        assert_close(f"HC {label}", got_hc[label], expected, 5e-12)

    for (response, N, descriptor), expected in EXPECTED_FIXED.items():
        got = float(
            fixed[
                (fixed["response"] == response)
                & (fixed["N"] == N)
                & (fixed["descriptor"] == descriptor)
            ].iloc[0]["Q2_LOO"]
        )
        assert_close(f"{response} C{N} {descriptor} Q2_LOO", got, expected, 5e-12)

    c8 = ceilings.set_index("response")
    for response, (expected_ceiling, expected_linear_r2) in EXPECTED_C8.items():
        got_ceiling = float(c8.loc[response, "R2_DRSO_max"])
        got_linear_r2 = float(c8.loc[response, "linear_DRSO_R2"])

        assert_close(
            f"C8 {response} representation ceiling",
            got_ceiling,
            expected_ceiling,
            5e-12,
        )
        assert_close(
            f"C8 {response} linear DRSO R2",
            got_linear_r2,
            expected_linear_r2,
            5e-7,
        )

    if set(ceilings["n_DRSO_classes"]) != {16}:
        raise AssertionError("Expected exactly 16 DRSO equality classes in C8.")

    print("PASS: deterministic five-endpoint values reproduce the frozen manuscript results.")


def compare_public_reference_tables(
    repo_root: Path,
    summary: pd.DataFrame,
    ceilings: pd.DataFrame,
) -> None:
    summary_path = repo_root / "results" / "reference" / "five_endpoint_c7_c8_summary.csv"
    ceiling_path = repo_root / "results" / "reference" / "c8_representation_ceilings.csv"

    if summary_path.exists():
        ref = pd.read_csv(summary_path).sort_values("response").reset_index(drop=True)
        got = summary.sort_values("response").reset_index(drop=True)
        if list(ref.columns) != list(got.columns):
            raise AssertionError("Reference summary columns differ from regenerated columns.")
        for col in ref.columns[1:]:
            if not np.allclose(ref[col], got[col], rtol=0.0, atol=5e-7):
                raise AssertionError(f"Public reference mismatch in {summary_path.name}: {col}")
        print(f"PASS: matches {summary_path}.")

    if ceiling_path.exists():
        ref = pd.read_csv(ceiling_path).sort_values("response").reset_index(drop=True)
        got = ceilings[["response", "R2_DRSO_max", "linear_DRSO_R2"]].copy()
        got = got.sort_values("response").reset_index(drop=True)
        if list(ref.columns) != list(got.columns):
            raise AssertionError("Reference ceiling columns differ from regenerated columns.")
        for col in ref.columns[1:]:
            if not np.allclose(ref[col], got[col], rtol=0.0, atol=5e-7):
                raise AssertionError(f"Public reference mismatch in {ceiling_path.name}: {col}")
        print(f"PASS: matches {ceiling_path}.")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Reproduce and audit the manuscript five-endpoint QSPR results."
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=None,
        help="Optional prebuilt 134-compound panel CSV; otherwise QDB.122 is rebuilt online.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=repo_root / "results" / "recomputed",
        help="Directory for regenerated outputs.",
    )
    parser.add_argument(
        "--fresh-permutation",
        action="store_true",
        help=(
            "Run an independent deterministic 10,000-permutation robustness audit. "
            "This does not claim digit-for-digit reproduction of the historical p-values."
        ),
    )
    args = parser.parse_args()

    panel = load_panel(args.panel)
    args.outdir.mkdir(parents=True, exist_ok=True)

    hc = pooled_hc(panel)
    fixed = fixed_size_all_descriptors(panel)
    summary = fixed_summary(fixed)
    ceilings = c8_ceilings(panel)
    archived_perm = verify_archived_permutation_claims()

    manuscript_checks(hc, fixed, ceilings)
    compare_public_reference_tables(repo_root, summary, ceilings)

    hc.to_csv(args.outdir / "five_endpoint_pooled_hc.csv", index=False)
    fixed.to_csv(args.outdir / "five_endpoint_c7_c8_loocv_all_descriptors.csv", index=False)
    summary.to_csv(args.outdir / "five_endpoint_c7_c8_summary.csv", index=False)
    ceilings.to_csv(args.outdir / "c8_representation_ceilings.csv", index=False)
    archived_perm.to_csv(
        args.outdir / "five_endpoint_archived_permutation_claim_check.csv",
        index=False,
    )

    print(
        "PASS: archived DRSO/mDRSO permutation p-values imply exactly the "
        "manuscript BH-FDR significance pattern."
    )

    if args.fresh_permutation:
        fresh = fresh_permutation_audit(panel)
        fresh.to_csv(
            args.outdir / "five_endpoint_fresh_permutation_10000.csv",
            index=False,
        )
        print(
            "PASS: fresh 10,000-permutation audit reproduces the manuscript's "
            "raw significant/non-significant pattern."
        )

    print(f"Outputs written to: {args.outdir}")


if __name__ == "__main__":
    main()
