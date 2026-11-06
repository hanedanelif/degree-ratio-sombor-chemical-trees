# Figure sources for "Degree-Ratio Sombor Indices of Chemical Trees"

This directory contains reproducible Python plotting implementations for Figures 1–6 of the manuscript. The scripts create vector-PDF and 600-dpi PNG outputs from the specified mathematical constructions and reference data. Use the outputs of these scripts when a direct match to the supplied source code is required.

## Run

```bash
python -m pip install -r requirements-figures.txt
python make_all_figures.py
```

Python 3.10 or newer is required. Each figure is written to `output/` as a vector PDF and a 600-dpi PNG.

## What each figure is built from

| Figure | Script | Source of the content | Checks performed by the code |
|---|---|---|---|
| Fig1 | `draw_figures_1_3.py` | Ten chemical edge-degree types $(i,j)$, $1\le i\le j\le 4$ | All 10 types occur exactly once, and each reduced ratio equals $\max/\min$. |
| Fig2 | `draw_fig2_fig4_fig5_fig6.py` | Partition-hierarchy theorem statements | Diagram of the theorem statements; no independent computational theorem check. |
| Fig3 | `draw_figures_1_3.py` | Witness trees $T_1,T_2$ ($n=9$) and $H_1,H_2$ ($n=16$), built as NetworkX graphs | Tree property, correct order, degree labels and edge-degree counts $m_{ij}$. |
| Fig4 | `draw_fig2_fig4_fig5_fig6.py` | 56-vertex trees $A,B$ constructed in the proof | Both orders, edge-degree vectors, $\mathrm{SO}(A)=\mathrm{SO}(B)$ and $\mathrm{DRSO}(A)-\mathrm{DRSO}(B)=2\sqrt2-\sqrt{17}$ checked exactly with SymPy. |
| Fig5 | `draw_fig2_fig4_fig5_fig6.py` | `data/exact_enumeration_counts_n4_n20.csv` | First DRSO–SDD separation at $n=9$ and first SO–DRSO separation at $n=16$. |
| Fig6 | `draw_fig2_fig4_fig5_fig6.py` | `data/bp_c6_c10_loocv.csv` | Input CSV is read; see analysis scripts for numerical provenance. |

The two CSV files are copies of reference outputs in the repository's `results/reference/` directory.

## Output-to-manuscript file mapping

| Manuscript file | Generated file |
|---|---|
| `Fig1.pdf` | `output/Fig1_reproducible_vector.pdf` |
| `Fig2.pdf` | `output/Fig2.pdf` |
| `Fig3.pdf` | `output/Fig3_reproducible_vector.pdf` |
| `Fig4.pdf` | `output/Fig4.pdf` |
| `Fig5.pdf` | `output/Fig5.pdf` |
| `Fig6.pdf` | `output/Fig6.pdf` |

`output/graph_verification.json` records the computed edge-degree counts for Figure 3.
