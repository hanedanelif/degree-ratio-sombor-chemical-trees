#!/usr/bin/env python3
"""Regenerate all six manuscript figures (vector PDF + 600-dpi PNG) into ./output."""
import subprocess, sys
from pathlib import Path
here = Path(__file__).resolve().parent
out = here / 'output'
for script in ('draw_figures_1_3.py', 'draw_fig2_fig4_fig5_fig6.py'):
    subprocess.run([sys.executable, str(here / script), '--output', str(out), '--dpi', '600'], check=True)
mapping = {'Fig1': 'Fig1_reproducible_vector.pdf', 'Fig2': 'Fig2.pdf', 'Fig3': 'Fig3_reproducible_vector.pdf',
           'Fig4': 'Fig4.pdf', 'Fig5': 'Fig5.pdf', 'Fig6': 'Fig6.pdf'}
for fig, f in mapping.items():
    assert (out / f).exists(), f
    print(f'{fig}: output/{f}')
