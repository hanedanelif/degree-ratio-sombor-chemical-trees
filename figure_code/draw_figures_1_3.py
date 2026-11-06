#!/usr/bin/env python3
"""Reproducible and mathematically checked drawings for DRSO manuscript Figures 1 and 3.

Generates both vector PDFs and 600-dpi PNGs from the underlying mathematical
specifications. The drawings produced by this file are NEW CODE-BASED RECREATIONS,
not a claim that earlier AI-rendered illustration pixels were made using this script.

Install: python -m pip install 'matplotlib>=3.8' 'networkx>=3.0'
Run:     python draw_figures_1_3.py --output . --dpi 600

Mathematical construction source: manuscript, Theorem 7 and Theorem 8
(n=9 witnesses T1/T2, n=16 witnesses H1/H2), and Definition 2 (ratio classes).
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import networkx as nx

# No network calls or random elements are used.
plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'mathtext.fontset': 'dejavuserif',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.facecolor': 'white',
})

COLORS = {
    'blue':   ('#0b55ba', '#deebfc'),
    'teal':   ('#007985', '#e0f5f5'),
    'green':  ('#087a16', '#e7f5df'),
    'orange': ('#ff5d06', '#fff0df'),
}

# All 10 unordered chemical edge-degree types classified by reduced ratio.
RATIO_CLASSES = (
    ('1', ( (1,1), (2,2), (3,3), (4,4) )),
    ('2', ( (1,2), (2,4) )),
    ('3', ( (1,3), )),
    ('4', ( (1,4), )),
    ('3/2', ( (2,3), )),
    ('4/3', ( (3,4), )),
)


def check_ratio_classes():
    expected={(i,j) for i in range(1,5) for j in range(i,5)}
    found=[pair for _, group in RATIO_CLASSES for pair in group]
    assert len(found)==10 and set(found)==expected, 'Ratio classification incomplete'
    from fractions import Fraction
    for rho, group in RATIO_CLASSES:
        for i, j in group:
            assert Fraction(j,i)==Fraction(rho), (rho,i,j)


def add_box(ax, x, y, w, h, text, fc, *, fontsize=15.0, bold=False,
            edgecolor='#4c6077', inset_px=8, shrink_to_fit=True):
    box=FancyBboxPatch((x,y),w,h,
        boxstyle='round,pad=0.003,rounding_size=0.012',
        linewidth=1.2,facecolor=fc,edgecolor=edgecolor)
    ax.add_patch(box)
    artist=ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fontsize,
                   fontweight='bold' if bold else 'normal')
    return (box,artist,inset_px,shrink_to_fit)


def auto_fit_box_labels(fig,ax,labels):
    """Guarantee each header/row label is INSIDE its corresponding box.
    Fit uses actual renderer measurements, not visually guessed fontsize.
    """
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for box,artist,inset,shrink in labels:
        x,y=box.get_x(),box.get_y(); w,h=box.get_width(),box.get_height()
        xy1=ax.transData.transform((x,y))
        xy2=ax.transData.transform((x+w,y+h))
        max_w=xy2[0]-xy1[0]-2*inset
        max_h=xy2[1]-xy1[1]-2*inset
        bbox=artist.get_window_extent(renderer)
        if shrink:
            while (bbox.width>max_w or bbox.height>max_h) and artist.get_fontsize()>10.5:
                artist.set_fontsize(artist.get_fontsize()-0.4)
                bbox=artist.get_window_extent(renderer)
        assert bbox.width<=max_w+1 and bbox.height<=max_h+1, (
            'Box overflow for',artist.get_text(),bbox.width,max_w,bbox.height,max_h)
        # spatial center check
        center=ax.transData.transform((x+w/2,y+h/2))
        assert abs((bbox.x0+bbox.x1)/2-center[0])<3,artist.get_text()


def draw_figure_1(output:Path, dpi:int):
    check_ratio_classes()
    fig,ax=plt.subplots(figsize=(12.8,8.15))
    fig.subplots_adjust(left=.015,right=.985,top=.98,bottom=.02)
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    fig.patch.set_facecolor('white')
    labels=[]
    ax.text(.50,.937,r'$\rho(i,j)=\dfrac{\max\{i,j\}}{\min\{i,j\}}$',
            ha='center',va='center',fontsize=25)
    # Leave explicit internal padding; mid header was the historic overflow.
    left=(.013,.332); middle=(.405,.190); right=(.656,.332)
    head_y=.805; head_h=.075
    headers=(('Chemical edge-degree types',left,15.3),
             (r'Reduced ratio $\rho(i,j)$',middle,14.5),
             ('Degree-ratio profile class',right,15.2))
    for label,(x,w),size in headers:
        labels.append(add_box(ax,x,head_y,w,head_h,label,'#e6edf4',fontsize=size,bold=True,inset_px=12))

    left_texts=(
        r'$(1,1),\ (2,2),\ (3,3),\ (4,4)$',
        r'$(1,2),\ (2,4)$',r'$(1,3)$',r'$(1,4)$',r'$(2,3)$',r'$(3,4)$')
    middle_texts=('1','2','3','4',r'$3/2$',r'$4/3$')
    right_texts=tuple(f'ratio {x} component' for x,_ in RATIO_CLASSES)
    for k,(lt,mt,rt) in enumerate(zip(left_texts,middle_texts,right_texts)):
        y=.695-k*.103; h=.070
        labels.append(add_box(ax,left[0],y,left[1],h,lt,'#dceffd',fontsize=17.7,inset_px=10))
        labels.append(add_box(ax,middle[0]+.03,y,middle[1]-.06,h,mt,'#d2e6f6',fontsize=19.8,bold=True,inset_px=8))
        labels.append(add_box(ax,right[0],y,right[1],h,rt,'#ecedfb',fontsize=16.8,inset_px=10))
        cy=y+h/2
        ax.add_patch(FancyArrowPatch((left[0]+left[1]+.002,cy),
                                    (middle[0]+.03-.006,cy),
                                    arrowstyle='-|>', mutation_scale=12,lw=1.2,color='#252525'))
        ax.add_patch(FancyArrowPatch((middle[0]+middle[1]-.03+.002,cy),
                                    (right[0]-.005,cy),
                                    arrowstyle='-|>',mutation_scale=12,lw=1.2,color='#252525'))
    ax.text(.5,.069,r'10 unordered edge-degree types $\longrightarrow$ 6 reduced ratio classes',
            ha='center',va='center',fontsize=17)
    auto_fit_box_labels(fig,ax,labels)
    fig.savefig(output/'Fig1_reproducible_vector.pdf',bbox_inches='tight',pad_inches=.05)
    fig.savefig(output/'Fig1_reproducible_600dpi.png',dpi=dpi,bbox_inches='tight',pad_inches=.05)
    plt.close(fig)


def add_edge(G,p,u,v):
    G.add_edge(u,v)


def graph_T1():
    G=nx.Graph(); pos={}
    for i in range(1,8):
        pos[f'v{i}']=(i-1,0)
    nx.add_path(G,[f'v{i}' for i in range(1,8)])
    for leaf,y in [('a',1.10),('b',-1.10)]:
        G.add_edge('v7',leaf);pos[leaf]=(6,y)
    return G,pos


def graph_T2():
    G=nx.Graph()
    pos={'c1':(0,0),'c2':(2.0,0),
         'a2':(-1.0,1.0),'a1':(-2.0,2.0),
         'b2':(-1.0,-1.0),'b1':(-2.0,-2.0),
         'd2':(3.0,1.0),'d1':(4.0,2.0),'e1':(3.0,-1.0)}
    G.add_edges_from([
        ('c1','c2'),
        ('c1','a2'),('a2','a1'),
        ('c1','b2'),('b2','b1'),
        ('c2','d2'),('d2','d1'),('c2','e1'),
    ])
    return G,pos


def graph_H1():
    G=nx.Graph();pos={}
    spine=[f'q{i}' for i in range(8)]
    nx.add_path(G,spine)
    for i,v in enumerate(spine):pos[v]=(float(i),0.0)
    # q0 and q7: each 3 leaves; q3: 2 leaves
    leaves={
        'q0':[(-.80,.88),(-1.05,0),(-.80,-.88)],
        'q3':[(3,1.15),(3,-1.15)],
        'q7':[(7.80,.88),(8.05,0),(7.80,-.88)]}
    for u,coords in leaves.items():
        for i,xy in enumerate(coords):
            v=f'{u}_leaf{i}';G.add_edge(u,v);pos[v]=xy
    return G,pos


def graph_H2():
    G=nx.Graph(); pos={}
    core=[f'c{i}' for i in range(4)]
    nx.add_path(G,core)
    for i,u in enumerate(core):pos[u]=(2.0*i,0)
    terminals={'c0':[(-.75,.86),(-1.05,0),(-.75,-.86)],
               'c3':[(6.75,.86),(7.05,0),(6.75,-.86)]}
    for u,coords in terminals.items():
        for i,xy in enumerate(coords):
            v=f'{u}_leaf{i}';G.add_edge(u,v);pos[v]=xy
    for i in [1,2]:
        u=f'c{i}';x=2.0*i
        leaf=f'{u}_leaf';d=f'{u}_deg2';upper=f'{u}_top_leaf'
        G.add_edges_from([(u,leaf),(u,d),(d,upper)])
        pos[leaf]=(x,-1.10);pos[d]=(x,1.00);pos[upper]=(x,2.00)
    return G,pos


EXPECTED={
'T1':{'order':9, 'counts':{'12':1,'22':4,'23':1,'13':2}},
'T2':{'order':9, 'counts':{'33':1,'23':3,'12':3,'13':1}},
'H1':{'order':16,'counts':{'22':3,'24':4,'14':8}},
'H2':{'order':16,'counts':{'44':3,'24':2,'12':2,'14':8}},
}

def validate_tree(G, expected, key):
    assert nx.is_tree(G), f'{key} is not a tree'
    assert G.number_of_nodes()==expected['order'],(key,'wrong node count')
    assert G.number_of_edges()==expected['order']-1,(key,'wrong edge count')
    assert max(dict(G.degree()).values())<=4,(key,'non-chemical graph')
    counts=Counter()
    for u,v in G.edges():
        a,b=sorted((G.degree(u),G.degree(v)))
        counts[f'{a}{b}']+=1
    expected_counts=expected['counts']
    assert dict(counts)==expected_counts, (key,dict(counts),expected_counts)
    # Validate every node label from its actual degree; do not trust manual labels.
    return {'n':G.number_of_nodes(), 'm':G.number_of_edges(),
            'max_degree':max(dict(G.degree()).values()),
            'edge_degree_counts':dict(sorted(counts.items()))}


def draw_tree_panel(ax, key, G, pos, tone, expected):
    border,fill=COLORS[tone]
    ax.axis('off')
    # Provide consistent geometric margins with equal circular nodes.
    xs=[p[0] for p in pos.values()]
    ys=[p[1] for p in pos.values()]
    xr=max(xs)-min(xs);yr=max(ys)-min(ys)
    ax.set_xlim(min(xs)-max(.25,xr*.06),max(xs)+max(.25,xr*.06))
    ax.set_ylim(min(ys)-max(.25,yr*.15),max(ys)+max(.25,yr*.15))
    ax.set_aspect('equal', adjustable='box')
    # Draw behind opaque node disks so edges do not cross vertex-degree labels.
    nx.draw_networkx_edges(G,pos,ax=ax,edge_color='#222222',width=1.6)
    nx.draw_networkx_nodes(G,pos,ax=ax,node_color=fill,
        edgecolors=border,linewidths=1.6,node_size=400)
    nx.draw_networkx_labels(G,pos,{v:str(G.degree(v)) for v in G.nodes()},
        ax=ax,font_size=11.1,font_family='DejaVu Serif',font_color='#101010')
    return validate_tree(G,expected,key)


def draw_figure_3(output:Path,dpi:int):
    specs=[
       ('T1',graph_T1(),'blue',r'$T_1\ \ (n=9)$',
        r'$m_{12}=1,\ m_{22}=4,\ m_{23}=1,\ m_{13}=2$'),
       ('T2',graph_T2(),'teal',r'$T_2\ \ (n=9)$',
        r'$m_{33}=1,\ m_{23}=3,\ m_{12}=3,\ m_{13}=1$'),
       ('H1',graph_H1(),'green',r'$H_1\ \ (n=16)$',
        r'$m_{22}=3,\ m_{24}=4,\ m_{14}=8$'),
       ('H2',graph_H2(),'orange',r'$H_2\ \ (n=16)$',
        r'$m_{44}=3,\ m_{24}=2,\ m_{12}=2,\ m_{14}=8$'),
    ]
    fig=plt.figure(figsize=(12.3,9.5),facecolor='white')
    # Four independent panels preserve geometric proportions.
    regions=[(.035,.55,.44,.29),(.52,.55,.44,.29),
             (.035,.10,.44,.28),(.52,.10,.44,.28)]
    title_positions=[(.255,.91),(.745,.91),(.255,.43),(.745,.43)]
    label_positions=[(.255,.52),(.745,.52),(.255,.06),(.745,.06)]
    report={}
    for (key,(G,pos),tone,title,annotation),reg,tpos,lpos in zip(specs,regions,title_positions,label_positions):
        ax=fig.add_axes(reg)
        report[key]=draw_tree_panel(ax,key,G,pos,tone,EXPECTED[key])
        fc,_=COLORS[tone]
        fig.text(tpos[0],tpos[1],title,color=fc,fontsize=20.0,
                 ha='center',va='center')
        fig.text(lpos[0],lpos[1],annotation,color=fc,fontsize=14.4,
                 ha='center',va='center')
    fig.savefig(output/'Fig3_reproducible_vector.pdf',bbox_inches='tight',pad_inches=.08)
    fig.savefig(output/'Fig3_reproducible_600dpi.png',dpi=dpi,bbox_inches='tight',pad_inches=.08)
    plt.close(fig)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--dpi',type=int,default=600)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    draw_figure_1(args.output,args.dpi)
    report=draw_figure_3(args.output,args.dpi)
    report['ratio_classes']={rho:[list(pair) for pair in groups] for rho,groups in RATIO_CLASSES}
    (args.output/'graph_verification.json').write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    print(f'Figures 1 and 3 saved as {args.dpi}-dpi PNGs and vector PDFs: {args.output}')

if __name__=='__main__':
    main()
