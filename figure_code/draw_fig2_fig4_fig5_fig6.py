#!/usr/bin/env python3
"""Reproducible sources for manuscript Figures 2, 4, 5 and 6.

Fig2  Equality-partition hierarchy diagram (Theorems on SO / DRSO / SDD).
Fig4  The 56-vertex witness trees A and B from the proof of Theorem (sharp
      SO--DRSO comparison).  The trees are built exactly as described in the
      proof, then checked by code: order 56, edge-type vectors m(A), m(B),
      SO(A) = SO(B) and DRSO(A) - DRSO(B) = 2*sqrt(2) - sqrt(17) (exact, SymPy).
Fig5  Equality-class gaps for orders 4..20, read from
      data/exact_enumeration_counts_n4_n20.csv (exact enumeration output).
Fig6  Leave-one-out Q^2 of DRSO and mDRSO for C6..C10, read from
      data/bp_c6_c10_loocv.csv (QSPR output).

Both CSV files are the reference outputs of the GitHub repository
https://github.com/hanedanelif/degree-ratio-sombor-chemical-trees
(results/reference/).

Run:  python draw_fig2_fig4_fig5_fig6.py --output output --dpi 600
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import networkx as nx
import sympy as sp

HERE = Path(__file__).resolve().parent
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'savefig.facecolor': 'white'})
SERIF = {'font.family': 'DejaVu Serif', 'mathtext.fontset': 'dejavuserif'}


def save(fig, out: Path, name: str, dpi: int):
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight', pad_inches=0.05)
    fig.savefig(out / f'{name}_600dpi.png', dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


# --------------------------------------------------------------------- Fig2
def draw_fig2(out: Path, dpi: int):
    with plt.rc_context(SERIF):
        fig, ax = plt.subplots(figsize=(11.0, 3.7))
        ax.set(xlim=(0, 110), ylim=(0, 37)); ax.axis('off')
        boxes = {'SO': (1, 17.5, '#dbe8f7', '#2f6db5', r'$\Pi_{\mathrm{SO}}$'),
                 'DR': (39, 30, '#f8d9d9', '#b44a4a',
                        r'$\Pi_{\mathrm{DRSO}}=\Pi_{m\mathrm{DRSO}}$'),
                 'SDD': (90, 19, '#dcefd8', '#4a8a45', r'$\Pi_{\mathrm{SDD}}$')}
        h = 6.2
        rows = [(26.5, r'(a) $3\leq n\leq 55$', r'$3\leq n\leq 55$',
                 'strict:\n' + r'$16\leq n\leq 55$', 'strict:\n' + r'$9\leq n\leq 55$', False),
                (8.0, r'(b) $n\geq 56$', r'$n\geq 56$',
                 'incomparable', 'strict', True)]
        for y, sub, rng, lab1, lab2, incomparable in rows:
            for key, (x, w, fc, ec, txt) in boxes.items():
                ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.0',
                                            fc=fc, ec=ec, lw=1.6))
                ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=17)
            ax.text(1.5, y + h + 3.0, rng, ha='left', va='center', fontsize=15)
            ax.text(55, y - 3.0, sub, ha='center', va='center', fontsize=14)
            cy = y + h / 2
            so_r, dr_l, dr_r, sdd_l = 18.5, 39, 69, 90
            if incomparable:
                ax.add_patch(FancyArrowPatch((so_r + 0.5, cy), (dr_l - 0.5, cy), arrowstyle='<|-|>',
                                             mutation_scale=16, lw=1.6, ls=(0, (5, 4)), color='black'))
            else:
                ax.add_patch(FancyArrowPatch((so_r + 0.5, cy), (dr_l - 0.5, cy), arrowstyle='-|>',
                                             mutation_scale=16, lw=1.6, color='black'))
            ax.add_patch(FancyArrowPatch((dr_r + 0.5, cy), (sdd_l - 0.5, cy), arrowstyle='-|>',
                                         mutation_scale=16, lw=1.6, color='black'))
            ax.text((so_r + dr_l) / 2, cy + 2.2, lab1, ha='center', va='bottom', fontsize=13,
                    linespacing=1.3)
            ax.text((dr_r + sdd_l) / 2, cy + 2.2, lab2, ha='center', va='bottom', fontsize=13,
                    linespacing=1.3)
        save(fig, out, 'Fig2', dpi)


# --------------------------------------------------------------------- Fig4
def tree_A():
    """Witness A of the proof, built literally from its description."""
    G = nx.Graph()
    q = [f'q{i}' for i in range(8)]
    a = [f'a{i}' for i in range(6)]
    b = [f'b{i}' for i in range(5)]
    c = [f'c{i}' for i in range(5)]
    r = [f'r{i}' for i in range(7)]
    for path in (q, a, b, c):
        nx.add_path(G, path)
    G.add_nodes_from(r)
    G.add_edges_from([('q0', 'a0'), ('q1', 'b0')])
    G.add_edges_from((f'q{i}', f'r{i - 2}') for i in (2, 3, 4, 5))
    G.add_edges_from([('r0', 'c0'), ('r4', 'a5'), ('r5', 'b4'), ('r6', 'c4')])
    target = {**{v: 4 for v in q}, **{v: 3 for v in r}, **{v: 2 for v in a + b + c}}
    leaves = attach_leaves(G, target)

    pos = {f'q{i}': (float(i), 0.0) for i in range(8)}
    pos.update({f'a{k}': (-0.6 - 0.9 * k, 1.6) for k in range(6)}); pos['r4'] = (-6.0, 1.6)
    pos.update({f'b{k}': (0.2 - 0.9 * k, -1.6) for k in range(5)}); pos['r5'] = (-4.3, -1.6)
    pos['r0'] = (2.0, -1.3)
    pos.update({f'c{k}': (1.0 + 0.9 * k, -3.0) for k in range(5)}); pos['r6'] = (5.5, -3.0)
    for i, rr in zip((3, 4, 5), ('r1', 'r2', 'r3')):
        pos[rr] = (float(i), 1.3)
    offsets = {'q0': [(-0.9, 0.0), (-0.55, -0.75)], 'q1': [(0.0, 0.9)], 'q2': [(0.0, 0.9)],
               'q3': [(0.0, -0.9)], 'q4': [(0.0, -0.9)], 'q5': [(0.0, -0.9)],
               'q6': [(0.0, 0.9), (0.0, -0.9)], 'q7': [(0.9, 0.0), (0.6, 0.75), (0.6, -0.75)],
               'r0': [(0.8, -0.45)], 'r1': [(-0.28, 0.75), (0.28, 0.75)],
               'r2': [(-0.28, 0.75), (0.28, 0.75)], 'r3': [(-0.28, 0.75), (0.28, 0.75)],
               'r4': [(-0.6, 0.5), (-0.6, -0.5)], 'r5': [(-0.6, 0.5), (-0.6, -0.5)],
               'r6': [(0.6, 0.5), (0.6, -0.5)]}
    place_leaves(pos, leaves, offsets)
    return G, pos


def tree_B():
    G = nx.Graph()
    s = [f's{j}' for j in range(19)]
    nx.add_path(G, s)
    for i in range(4):
        G.add_edge(f's{2 + 2 * i}', f'u{i}')
        G.add_edge(f's{10 + 2 * i}', f'v{i}')
    target = {**{v: 3 for v in s}, **{f'u{i}': 4 for i in range(4)}, **{f'v{i}': 2 for i in range(4)}}
    leaves = attach_leaves(G, target)
    pos = {f's{j}': (float(j), 0.0) for j in range(19)}
    offsets = {}
    for i in range(4):
        pos[f'u{i}'] = (2.0 + 2 * i, 1.0)
        offsets[f'u{i}'] = [(-0.55, 0.55), (0.0, 0.8), (0.55, 0.55)]
        pos[f'v{i}'] = (10.0 + 2 * i, -1.0)
        offsets[f'v{i}'] = [(0.0, -0.9)]
    offsets['s0'] = [(-0.6, 0.55), (-0.6, -0.55)]
    offsets['s18'] = [(0.6, 0.55), (0.6, -0.55)]
    for j in range(1, 18):
        if j in (2, 4, 6, 8, 10, 12, 14, 16):
            continue
        offsets[f's{j}'] = [(0.0, -0.9)] if j < 9 else [(0.0, 0.9)]
    place_leaves(pos, leaves, offsets)
    return G, pos


def attach_leaves(G, target):
    leaves = {}
    for v, d in target.items():
        need = d - G.degree(v)
        assert need >= 0, (v, 'degree already too large')
        leaves[v] = []
        for t in range(need):
            leaf = f'{v}_L{t}'
            G.add_edge(v, leaf)
            leaves[v].append(leaf)
    return leaves


def place_leaves(pos, leaves, offsets):
    for v, ls in leaves.items():
        assert len(ls) == len(offsets.get(v, [])), (v, len(ls), offsets.get(v))
        for leaf, (dx, dy) in zip(ls, offsets.get(v, [])):
            pos[leaf] = (pos[v][0] + dx, pos[v][1] + dy)


ORDER = ['12', '13', '14', '22', '23', '24', '33', '34', '44']   # eq. (mvec)


def edge_type_vector(G):
    cnt = Counter()
    for u, v in G.edges():
        i, j = sorted((G.degree(u), G.degree(v)))
        cnt[f'{i}{j}'] += 1
    return [cnt[k] for k in ORDER]


def exact_index(G, f):
    return sp.nsimplify(sum(f(G.degree(u), G.degree(v)) for u, v in G.edges()))


def verify_fig4(A, B):
    for G in (A, B):
        assert nx.is_tree(G) and G.number_of_nodes() == 56
        assert max(d for _, d in G.degree()) <= 4
    mA, mB = edge_type_vector(A), edge_type_vector(B)
    assert mA == [0, 13, 12, 13, 4, 2, 0, 4, 7], mA
    assert mB == [4, 13, 12, 0, 4, 0, 18, 4, 0], mB
    so = lambda i, j: sp.sqrt(i ** 2 + j ** 2)
    drso = lambda i, j: sp.sqrt(sp.Rational(i, j) ** 2 + sp.Rational(j, i) ** 2)
    assert sp.simplify(exact_index(A, so) - exact_index(B, so)) == 0
    diff = sp.simplify(exact_index(A, drso) - exact_index(B, drso))
    assert sp.simplify(diff - (2 * sp.sqrt(2) - sp.sqrt(17))) == 0, diff
    return {'m(A)': mA, 'm(B)': mB, 'SO(A)-SO(B)': 0, 'DRSO(A)-DRSO(B)': str(diff)}


def draw_fig4(out: Path, dpi: int):
    A, posA = tree_A()
    B, posB = tree_B()
    report = verify_fig4(A, B)
    fills = {1: '#ffffff', 2: '#cfd8e3', 3: '#f4a96a', 4: '#3a7bb0'}
    fig = plt.figure(figsize=(8.6, 7.0))
    axA = fig.add_axes([0.02, 0.36, 0.96, 0.54])
    axB = fig.add_axes([0.02, 0.02, 0.96, 0.30])
    for ax, G, pos, title in ((axA, A, posA, r'$A$  (56 vertices)'),
                              (axB, B, posB, r'$B$  (56 vertices)')):
        ax.axis('off'); ax.set_aspect('equal')
        nx.draw_networkx_edges(G, pos, ax=ax, width=1.3, edge_color='#333a44')
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=120, linewidths=1.0, edgecolors='#222222',
                               node_color=[fills[G.degree(v)] for v in G.nodes()])
        ax.set_title(title, loc='left', fontsize=12)
    handles = [plt.Line2D([], [], ls='', marker='o', ms=9, mfc=fills[d], mec='#222222',
                          label=f'degree $d={d}$') for d in (1, 2, 3, 4)]
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 0.995))
    save(fig, out, 'Fig4', dpi)
    return report


# --------------------------------------------------------------------- Fig5
def read_csv(path):
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def draw_fig5(out: Path, dpi: int):
    rows = read_csv(HERE / 'data' / 'exact_enumeration_counts_n4_n20.csv')
    n = [int(r['n']) for r in rows]
    so = [int(r['SO_JDM_distinct']) for r in rows]
    dr = [int(r['DRSO_mDRSO_distinct']) for r in rows]
    sdd = [int(r['SDD_distinct']) for r in rows]
    g1 = [d - s for d, s in zip(dr, sdd)]
    g2 = [a - d for a, d in zip(so, dr)]
    assert next(k for k, g in zip(n, g1) if g > 0) == 9
    assert next(k for k, g in zip(n, g2) if g > 0) == 16
    with plt.rc_context(SERIF):
        fig, ax = plt.subplots(figsize=(8.0, 4.4))
        ax.plot(n, g1, '-o', color='black', ms=5, lw=1.6,
                label=r'$D_{\mathrm{DRSO}}(n)-D_{\mathrm{SDD}}(n)$')
        ax.plot(n, g2, '--s', color='#555555', ms=5, lw=1.6,
                label=r'$D_{\mathrm{SO}}(n)-D_{\mathrm{DRSO}}(n)$')
        ax.set_yscale('symlog', linthresh=1)
        ax.set_ylim(-0.3, 6000)
        for x, txt in ((9, r'first DRSO–SDD gap: $n=9$'), (16, r'first SO–DRSO gap: $n=16$')):
            ax.axvline(x, color='#666666', lw=0.9, ls=':')
            # horizontal label right of the guide line, in the empty band between 0 and 1
            ax.text(x + 0.15, 0.42, txt, ha='left', va='center', fontsize=9.5)
        ax.set_xticks(range(4, 21, 2)); ax.set_xlim(3.4, 21.9)
        ax.set_xlabel(r'Order $n$')
        ax.set_ylabel('Difference in equality-class counts')
        ax.grid(axis='y', color='#e3e3e3', lw=0.7)
        ax.legend(loc='upper left', frameon=False)
        save(fig, out, 'Fig5', dpi)


# --------------------------------------------------------------------- Fig6
def draw_fig6(out: Path, dpi: int):
    rows = read_csv(HERE / 'data' / 'bp_c6_c10_loocv.csv')
    data = {d: sorted((int(r['N']), float(r['Q2_LOO'])) for r in rows if r['descriptor'] == d)
            for d in ('DRSO', 'mDRSO')}
    with plt.rc_context(SERIF):
        fig, ax = plt.subplots(figsize=(8.0, 4.4))
        for d, style, col in (('DRSO', '-o', 'black'), ('mDRSO', '--s', '#555555')):
            xs, ys = zip(*data[d])
            ax.plot(xs, ys, style, color=col, ms=5, lw=1.6, label=d)
        ax.axhline(0, color='#777777', lw=0.9)
        ax.set_ylim(-0.03, 1.0); ax.set_xticks(range(6, 11))
        ax.set_xlabel('Carbon number'); ax.set_ylabel(r'LOOCV $Q^{2}$')
        ax.grid(axis='y', color='#e3e3e3', lw=0.7)
        ax.legend(frameon=False)
        save(fig, out, 'Fig6', dpi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=HERE / 'output')
    p.add_argument('--dpi', type=int, default=600)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    draw_fig2(a.output, a.dpi)
    print('Fig4 verification:', draw_fig4(a.output, a.dpi))
    draw_fig5(a.output, a.dpi)
    draw_fig6(a.output, a.dpi)
    print('Figures 2, 4, 5, 6 written to', a.output)


if __name__ == '__main__':
    main()
