#!/usr/bin/env python3
"""Build the common 134-compound QDB.122 panel and DRSO descriptor table.

This is a cleaned manuscript-aligned descendant of the archived
`qsardb_drso_master_pipeline.py` (12 Aug 2026). It performs data engineering
only; statistical modelling is handled by drso_reproducibility.py.

Network access is required to download QsarDB tables.
"""
from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import requests

QDB_BASE = "https://qsardb.org/repository/compounds/10967/122"
PRIMARY_ENDPOINTS = ("BP", "D", "RI", "HC", "E")
ANCHOR_ENDPOINTS_134 = ("D", "RI", "HC", "E")
ROOTS = {"methane":1,"ethane":2,"propane":3,"butane":4,"pentane":5,
         "hexane":6,"heptane":7,"octane":8,"nonane":9,"decane":10}
MULTIPLICITY_PREFIX = {"":1,"di":2,"tri":3,"tetra":4,"penta":5,"hexa":6}
EDGE_TYPES = ((1,2),(1,3),(1,4),(2,2),(2,3),(2,4),(3,3),(3,4),(4,4))


def fetch_property_table(property_id: str, timeout: int = 60) -> pd.DataFrame:
    url = f"{QDB_BASE}?property={property_id}"
    response = requests.get(url, headers={"User-Agent":"academic-QSPR-reproducibility/1.0"}, timeout=timeout)
    response.raise_for_status()
    candidate = None
    for table in pd.read_html(response.text):
        cols = [str(c).strip() for c in table.columns]
        if "ID" in cols and "Name" in cols and property_id in cols:
            candidate = table.copy(); break
    if candidate is None:
        raise RuntimeError(f"Could not identify QsarDB table for property {property_id}")
    candidate = candidate[["ID","Name",property_id]].copy()
    candidate["ID"] = pd.to_numeric(candidate["ID"], errors="raise").astype(int)
    candidate[property_id] = pd.to_numeric(candidate[property_id], errors="raise")
    candidate["Name"] = candidate["Name"].astype(str).str.strip()
    if candidate["ID"].duplicated().any():
        raise AssertionError(f"Duplicate QsarDB IDs in {property_id}")
    return candidate.sort_values("ID").reset_index(drop=True)


def build_common_property_panel() -> pd.DataFrame:
    tables = {p: fetch_property_table(p) for p in PRIMARY_ENDPOINTS}
    anchor = {p:set(tables[p]["ID"]) for p in ANCHOR_ENDPOINTS_134}
    for p, ids in anchor.items():
        if len(ids) != 134:
            raise AssertionError(f"{p}: expected 134 compounds; found {len(ids)}")
    common = set.intersection(*anchor.values())
    if len(common) != 134:
        raise AssertionError(f"Common D/RI/HC/E panel should contain 134 IDs; found {len(common)}")
    master = tables["D"].loc[tables["D"].ID.isin(common), ["ID","Name","D"]].copy()
    for p in ("RI","HC","E","BP"):
        right = tables[p].loc[tables[p].ID.isin(common), ["ID","Name",p]].copy().rename(columns={"Name":f"Name_{p}"})
        master = master.merge(right, on="ID", how="left", validate="one_to_one")
        bad = master[master[f"Name_{p}"].notna() & (master[f"Name_{p}"].str.lower()!=master["Name"].str.lower())]
        if not bad.empty:
            raise AssertionError(f"Name mismatch D vs {p}:\n{bad[['ID','Name',f'Name_{p}']].to_string(index=False)}")
        master = master.drop(columns=f"Name_{p}")
    if len(master) != 134 or master[list(PRIMARY_ENDPOINTS)].isna().any().any():
        raise AssertionError("The pre-specified five-response common panel is not complete.")
    master = master.rename(columns={"D":"D_raw"})
    # QsarDB metadata says g cm^-3 but the archived numerical scale is several hundred.
    # The manuscript reports those numbers on the g L^-1 scale, numerically equal to kg m^-3.
    master["D_g_L"] = master["D_raw"]
    master["D_kg_m3"] = master["D_raw"]
    master["D_g_cm3"] = master["D_raw"] / 1000.0
    return master.sort_values("ID").reset_index(drop=True)


def _find_parent_alkane(name: str) -> Tuple[str,int,str]:
    lname = name.lower().strip()
    # tolerate the occasional n- prefix on normal alkanes
    if lname.startswith("n-"):
        lname = lname[2:]
    for root, n in sorted(ROOTS.items(), key=lambda kv:-len(kv[0])):
        if lname.endswith(root):
            return root, n, lname[:-len(root)].rstrip("-")
    raise ValueError(f"No supported parent alkane suffix in: {name}")


def _split_substituent_chunks(prefix: str) -> List[str]:
    if not prefix: return []
    pattern = r"[0-9,]+-(?:(?:di|tri|tetra|penta|hexa)?(?:methyl|ethyl|isopropyl|propyl))"
    chunks = re.findall(pattern, prefix)
    if "-".join(chunks) != prefix:
        raise ValueError(f"Unsupported substituent expression: {prefix!r}; parsed={chunks}")
    return chunks


def _parse_chunk(chunk: str) -> Tuple[List[int],str]:
    m = re.fullmatch(r"([0-9,]+)-((?:di|tri|tetra|penta|hexa)?)(methyl|ethyl|isopropyl|propyl)", chunk)
    if not m: raise ValueError(f"Cannot parse substituent chunk: {chunk}")
    locants = [int(x) for x in m.group(1).split(",")]
    mult = MULTIPLICITY_PREFIX[m.group(2)]
    if len(locants) != mult:
        raise ValueError(f"{chunk}: {len(locants)} locants but prefix implies {mult}")
    return locants, m.group(3)


def _add_substituent(G: nx.Graph, parent: int, sub: str) -> None:
    nxt = max(G.nodes, default=-1)+1
    if sub == "methyl": G.add_edge(parent, nxt)
    elif sub == "ethyl": G.add_edges_from([(parent,nxt),(nxt,nxt+1)])
    elif sub == "propyl": G.add_edges_from([(parent,nxt),(nxt,nxt+1),(nxt+1,nxt+2)])
    elif sub == "isopropyl": G.add_edges_from([(parent,nxt),(nxt,nxt+1),(nxt,nxt+2)])
    else: raise ValueError(sub)


def alkane_name_to_carbon_tree(name: str) -> nx.Graph:
    _, parent_n, prefix = _find_parent_alkane(name)
    G = nx.path_graph(parent_n)
    for chunk in _split_substituent_chunks(prefix):
        locants, sub = _parse_chunk(chunk)
        for locant in locants:
            if not 1 <= locant <= parent_n: raise ValueError(f"{name}: bad locant {locant}")
            _add_substituent(G, locant-1, sub)
    if not nx.is_tree(G) or max(dict(G.degree()).values()) > 4 or not (6 <= G.number_of_nodes() <= 10):
        raise AssertionError(f"Invalid chemical tree generated for {name}")
    return G


def edge_type_counts(G: nx.Graph) -> Counter:
    d = dict(G.degree()); c=Counter()
    for u,v in G.edges(): c[tuple(sorted((d[u],d[v])))] += 1
    return c


def degree_ratio_profile(c: Counter):
    return (c[(2,2)]+c[(3,3)]+c[(4,4)], c[(1,2)]+c[(2,4)], c[(1,3)], c[(1,4)], c[(2,3)], c[(3,4)])


def calculate_descriptors(G: nx.Graph) -> Dict[str,float]:
    d=dict(G.degree()); c=edge_type_counts(G); R=degree_ratio_profile(c)
    out={
        "N":G.number_of_nodes(), "m_edges":G.number_of_edges(),
        "n1":sum(x==1 for x in d.values()), "n2":sum(x==2 for x in d.values()),
        "n3":sum(x==3 for x in d.values()), "n4":sum(x==4 for x in d.values()),
        "DRSO":sum(math.sqrt((d[u]/d[v])**2+(d[v]/d[u])**2) for u,v in G.edges()),
        "mDRSO":sum(1/math.sqrt((d[u]/d[v])**2+(d[v]/d[u])**2) for u,v in G.edges()),
        "SO":sum(math.sqrt(d[u]**2+d[v]**2) for u,v in G.edges()),
        "SDD":sum(d[u]/d[v]+d[v]/d[u] for u,v in G.edges()),
        "M1":sum(d[u]+d[v] for u,v in G.edges()),
        "M2":sum(d[u]*d[v] for u,v in G.edges()),
        "Randic":sum(1/math.sqrt(d[u]*d[v]) for u,v in G.edges()),
        "ABC":sum(math.sqrt((d[u]+d[v]-2)/(d[u]*d[v])) for u,v in G.edges()),
        "GA":sum(2*math.sqrt(d[u]*d[v])/(d[u]+d[v]) for u,v in G.edges()),
        "R_equal":R[0], "R_2":R[1], "R_3":R[2], "R_4":R[3], "R_3_2":R[4], "R_4_3":R[5],
    }
    for i,j in EDGE_TYPES: out[f"m{i}{j}"] = c[(i,j)]
    out["WL_hash"] = nx.weisfeiler_lehman_graph_hash(G)
    return out


def validate_and_describe(master: pd.DataFrame) -> pd.DataFrame:
    rows=[]; graphs={}
    for row in master.itertuples(index=False):
        G=alkane_name_to_carbon_tree(row.Name); graphs[row.ID]=G
        rows.append({"ID":row.ID, **calculate_descriptors(G)})
    out=master.merge(pd.DataFrame(rows),on="ID",validate="one_to_one")
    if not np.all(out.m_edges.to_numpy()==out.N.to_numpy()-1): raise AssertionError("Tree identity failed")
    if not np.array_equal(out.n1.to_numpy(), (out.n3+2*out.n4+2).to_numpy()): raise AssertionError("n1 identity failed")
    if not np.array_equal(sum(out[f"m{i}{j}"] for i,j in EDGE_TYPES).to_numpy(), out.m_edges.to_numpy()): raise AssertionError("edge incidence failed")
    for N, block in out.groupby("N"):
        ids=block.ID.tolist()
        for ai in range(len(ids)):
            for bi in range(ai+1,len(ids)):
                if nx.is_isomorphic(graphs[ids[ai]], graphs[ids[bi]]):
                    raise AssertionError(f"Duplicate constitutional tree C{N}: {ids[ai]}, {ids[bi]}")
    return out.sort_values(["N","ID"]).reset_index(drop=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",default="DRSO_QSPR_master_QDB122.csv")
    args=ap.parse_args()
    master=validate_and_describe(build_common_property_panel())
    master.to_csv(args.output,index=False)
    print(f"Wrote {len(master)} compounds to {args.output}")
    print(master.N.value_counts().sort_index().to_string())

if __name__ == "__main__": main()
