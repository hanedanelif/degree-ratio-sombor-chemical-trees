from pathlib import Path
import importlib.util
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("drso_reproducibility", ROOT / "drso_reproducibility.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


@pytest.fixture(scope="session")
def outputs(tmp_path_factory):
    out = tmp_path_factory.mktemp("drso")
    result = mod.reproduce_bp(ROOT / "data/processed/alkane_bp_c6_c10_142_master.csv", out)
    return result


def test_bp_reference_results(outputs):
    mod.manuscript_checks(outputs)
    tab = outputs["table6"].set_index("model")
    assert round(tab.loc["N + ABC", "Q2_CV"], 6) == 0.967117
    assert round(tab.loc["N + DRSO", "Q2_CV"], 6) == 0.959827
    assert round(tab.loc["N + M2", "Q2_CV"], 6) == 0.939286


def test_full_permutation_reference_counts(outputs):
    got = outputs["permutation"].set_index(["N", "descriptor"])["exceedances"].astype(int)
    expected = {
        6:[79,298,83,80,174,356,174,184,80],
        7:[9,5,36,9,47,391,9,5,11],
        8:[0,2,0,2,8,350,0,0,0],
        9:[0,0,3,0,12,1852,0,0,0],
        10:[0,0,20,0,123,7385,0,0,0],
    }
    for N, counts in expected.items():
        for d, c in zip(mod.DESCRIPTORS, counts):
            assert got.loc[(N, d)] == c


def test_enumeration_small_orders():
    enum = mod.enumerate_chemical_trees(4, 10).set_index("n")
    expected = {
        4:(2,2,2,2), 5:(3,3,3,3), 6:(5,5,5,5), 7:(9,9,9,9),
        8:(18,16,16,16), 9:(35,28,28,24), 10:(75,49,49,40),
    }
    for n, (ct, so, drso, sdd) in expected.items():
        row = enum.loc[n]
        assert int(row.chemical_trees) == ct
        assert int(row.SO_JDM_distinct) == so
        assert int(row.DRSO_mDRSO_distinct) == drso
        assert int(row.SDD_distinct) == sdd
