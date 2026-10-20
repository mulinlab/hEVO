"""Fig.4g world map plus adaptive-allele benchmark results.

The map is a geographic index for the frozen 30-locus benchmark. The two
quantitative columns retain the S38 directional-preference estimates and the
S39 raw-score Calinski-Harabasz calculation used by the former Fig.4h.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from matplotlib.ticker import NullLocator

HERE = Path(__file__).resolve()
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from common.matplotlib_common import configure_matplotlib
from common.public_paths import publicize_paths
from Fig4 import h_adaptive_allele_preference as metrics


BLACK = "#000000"
LAND = "#E5EBEF"
COAST = "#8B99A4"
MARKER = "#3F5668"
GRID = "#E1E6EA"
HEVO_WASH = "#FBF2EA"
NEUTRAL = "#69727D"
NEUTRAL_LIGHT = "#C7CED6"

ASSET_PATH = HERE.parents[2] / "assets" / "natural_earth" / "ne_110m_land.geojson"
LOCI_RELATIVE = Path(
    "hEvo_benchmark/datasets/Adaptation_Variants/Adaptive_Allele_Preference_Benchmark/benchmark_loci_expanded.tsv"
)

REGIONS = {
    "1": {
        "title": "Europe",
        "lon": 13,
        "lat": 50,
        "variants": ["rs4988235", "rs16891982", "rs12913832", "rs1042602", "rs12821256", "rs12203592"],
        "genes": "MCM6/LCT, SLC45A2, HERC2/OCA2, TYR, KITLG, IRF4",
        "trait": "lactase persistence; pigmentation",
    },
    "2": {
        "title": "Africa / Arabia",
        "lon": 38,
        "lat": 19,
        "variants": ["rs41380347", "rs145946881"],
        "genes": "MCM6/LCT",
        "trait": "lactase persistence",
    },
    "3": {
        "title": "Sub-Saharan Africa",
        "lon": 19,
        "lat": 1,
        "variants": ["rs2814778", "rs334", "rs33930165", "rs1871534"],
        "genes": "ACKR1, HBB, SLC39A4",
        "trait": "malaria resistance; zinc transport",
    },
    "4": {
        "title": "Tibetan Plateau",
        "lon": 89,
        "lat": 31,
        "variants": ["rs186996510", "rs12097901", "rs149594770"],
        "genes": "EGLN1, EPAS1",
        "trait": "high-altitude adaptation",
    },
    "5": {
        "title": "East Asia",
        "lon": 116,
        "lat": 37,
        "variants": ["rs1800414", "rs74653330", "rs3827760", "rs885479", "rs17822931", "rs671"],
        "genes": "OCA2, EDAR, MC1R, ABCC11, ALDH2",
        "trait": "pigmentation; ectodermal traits; metabolism",
    },
    "6": {
        "title": "Andean highlands",
        "lon": -70,
        "lat": -16,
        "variants": ["rs10035235"],
        "genes": "PRKAA1",
        "trait": "high-altitude adaptation",
    },
    "7": {
        "title": "Northern Eurasia",
        "lon": 62,
        "lat": 61,
        "variants": ["rs10166942"],
        "genes": "TRPM8",
        "trait": "cold-climate adaptation",
    },
    "8": {
        "title": "Western Eurasia",
        "lon": 37,
        "lat": 45,
        "variants": ["rs174546", "rs5743618"],
        "genes": "FADS1/FADS2, TLR1",
        "trait": "diet; innate immunity",
    },
    "B": {
        "title": "Broad distributions",
        "lon": -17,
        "lat": 30,
        "variants": ["rs1426654", "rs642742", "rs10774671", "rs1047626", "rs497116"],
        "genes": "SLC24A5, KITLG, OAS1, SLC30A9, CASP12",
        "trait": "pigmentation; immunity; metabolism",
    },
}


def _repo_root() -> Path:
    return HERE.parents[4]


def prepare_map_data(repo_root: Path | None = None) -> pd.DataFrame:
    root = Path(repo_root) if repo_root is not None else _repo_root()
    loci = pd.read_csv(root / LOCI_RELATIVE, sep="\t")
    if len(loci) != metrics.EXPECTED_LOCI or loci["variant_id"].nunique() != metrics.EXPECTED_LOCI:
        raise ValueError("Fig.4g requires exactly 30 unique frozen benchmark loci")
    rows = []
    assigned = []
    by_variant = loci.set_index("variant_id")
    for region_id, region in REGIONS.items():
        for variant_id in region["variants"]:
            if variant_id not in by_variant.index:
                raise ValueError(f"Unknown benchmark locus in Fig.4g map: {variant_id}")
            row = by_variant.loc[variant_id]
            rows.append(
                {
                    "region_id": region_id,
                    "region": region["title"],
                    "map_lon": region["lon"],
                    "map_lat": region["lat"],
                    "variant_id": variant_id,
                    "gene": row["gene"],
                    "target_population": row["target_population"],
                    "benchmark_tier": row["benchmark_tier"],
                }
            )
            assigned.append(variant_id)
    if len(assigned) != metrics.EXPECTED_LOCI or set(assigned) != set(loci["variant_id"]):
        raise ValueError("Fig.4g regional groups must assign every benchmark locus exactly once")
    if len(assigned) != len(set(assigned)):
        raise ValueError("Fig.4g regional groups contain duplicate loci")
    return pd.DataFrame(rows)


def _iter_polygon_rings(geometry: dict):
    if geometry["type"] == "Polygon":
        yield geometry["coordinates"][0]
    elif geometry["type"] == "MultiPolygon":
        for polygon in geometry["coordinates"]:
            yield polygon[0]


def _draw_world(ax: mpl.axes.Axes, asset_path: Path) -> None:
    payload = json.loads(asset_path.read_text(encoding="utf-8"))
    for feature in payload["features"]:
        for ring in _iter_polygon_rings(feature["geometry"]):
            points = np.asarray(ring, dtype=float)
            if len(points) < 3:
                continue
            ax.add_patch(
                Polygon(points, closed=True, facecolor=LAND, edgecolor=COAST, linewidth=0.28, zorder=1)
            )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-58, 85)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_map_notes(ax, map_data, *, vertical_shift_pts: float = 0.0):
    ax.set_axis_off()
    notes = [
        ("1  Europe (6)", "LCT/MCM6 · SLC45A2 · TYR\nHERC2/OCA2 · IRF4 · KITLG", "Milk digestion · pigmentation"),
        ("2  Africa / Arabia (2)", "LCT/MCM6", "Lactase persistence"),
        ("3  Sub-Saharan Africa (4)", "HBB · ACKR1 · SLC39A4", "Malaria protection · zinc"),
        ("4  Tibetan Plateau (3)", "EGLN1 · EPAS1", "High-altitude hypoxia"),
        ("5  East Asia (6)", "OCA2 · MC1R · EDAR\nABCC11 · ALDH2", "Pigmentation · hair / glands\nEarwax · alcohol metabolism"),
        ("6  Andean highlands (1)", "PRKAA1", "Hypoxic ventilation"),
        ("7  N. Eurasia (1)", "TRPM8", "Cold perception"),
        ("8  W. Eurasia (2)", "FADS1/FADS2 · TLR1", "Lipids · innate immunity"),
        ("B  Broad distributions (5)", "SLC24A5 · KITLG · OAS1\nSLC30A9 · CASP12", "Pigmentation · zinc · immunity"),
    ]
    fig=ax.figure
    W,H=595.2755905512,841.8897637795
    for i,(title,genes,trait) in enumerate(notes):
        x=[19.8,105.2,190.5][i%3]
        y=[719.6,751.87,790.06][i//3] + vertical_shift_pts
        fig.text(x/W,1-(y+4.7)/H,title,fontsize=6,fontweight="bold",va="baseline")
        lines=genes.split("\n")
        for j,line in enumerate(lines):
            fig.text(x/W,1-(y+11.0+6.1*j)/H,line,fontsize=6,fontstyle="italic",va="baseline")
        for j,line in enumerate(trait.split("\n")):
            fig.text(x/W,1-(y+11.2+6.1*len(lines)+6.1*j)/H,line,fontsize=6,va="baseline")

REFERENCE_MODELS = [
    "hEvo-base","hEvo-large","Mistral-DNA-17M","GENA-LM-base",
    "NT_500m_MS","Caduceus-PS","HyenaDNA_tiny_1k","AIDO.DNA-300M",
    "Caduceus-PH","GROVER","NT_2.5b_MS","NTv3-100M-pre",
    "SpeciesLM","DNABERT2","evo2_1b_base","GENERator-1.2B",
    "GPN-promoter","Genos-1.2B","GPN-Star-P","GPN-MSA",
]

def build_panel(tables_dir, *, figure, slot=None, letter="g", repo_root=None, vertical_shift_pts: float = 0.0):
    configure_matplotlib()
    ordered,ch=metrics.prepare_data(tables_dir)
    map_data=prepare_map_data(repo_root)
    indexed=ordered.set_index("model_id",drop=False)
    if not set(REFERENCE_MODELS).issubset(indexed.index):
        raise ValueError("A reference-display model is missing from the current source")
    plot_data=indexed.loc[REFERENCE_MODELS].copy().reset_index(drop=True)
    if not plot_data["status"].eq("PASS").all():
        raise ValueError("A reference-display model has no PASS preference result")
    plot_data["y"]=np.arange(len(plot_data))
    W,H=595.2755905512,841.8897637795
    def axes(x,top,width,height):
        top += vertical_shift_pts
        return figure.add_axes([x/W,1-(top+height)/H,width/W,height/H])
    map_ax=axes(19.796,650.0,252.3,60.0)
    _draw_world(map_ax,ASSET_PATH)
    map_ax.set_aspect("auto")
    marker_positions={"1":(10,52),"2":(39,10),"3":(10,0),"4":(88,30),
                      "5":(117,34),"6":(-69,-20),"7":(62,65),"8":(35,40)}
    for region_id,(lon,lat) in marker_positions.items():
        map_ax.scatter(lon,lat,s=29,facecolor=MARKER,edgecolor="white",linewidth=.45,zorder=3)
        map_ax.text(lon,lat,region_id,ha="center",va="center",fontsize=6,color="white",zorder=4)
    figure.text(81.88/W,1-(640.5+vertical_shift_pts)/H,"Adaptive-allele benchmark (30 variants)",
                fontsize=6.8,fontweight="bold",ha="left",va="baseline")
    notes_ax=axes(19.796,719.6,252.3,94)
    _draw_map_notes(notes_ax,map_data,vertical_shift_pts=vertical_shift_pts)
    names_ax=axes(297.0,651.0,91.0,138.9)
    pref=axes(399.7,651.0,76.5,138.9)
    ch_ax=axes(496.1,651.0,62.4,138.9)
    names_ax.set_xlim(0,1)
    names_ax.set_axis_off()
    for ax in [names_ax,pref,ch_ax]:
        ax.set_ylim(19.5,-.5)
    plot_data.loc[plot_data["model_id"].eq("NT_2.5b_MS"), "plot_label"] = "NTv1-2.5B-MS"
    for row in plot_data.itertuples(index=False):
        style=metrics._family_style(row.model_id)
        highlight=row.model_id in {"hEvo-base","hEvo-large"}
        if highlight:
            for ax in [pref,ch_ax]:
                ax.axhspan(row.y-.46,row.y+.46,color=HEVO_WASH,lw=0,zorder=0)
        names_ax.scatter(.03,row.y,s=6,marker=style["marker"],facecolor=style["color"],
                         edgecolor="black",linewidth=.3)
        names_ax.text((305.296-297.0)/91,row.y,row.plot_label,fontsize=6,
                      fontweight="bold" if highlight else "normal",va="center")
        color=style["color"] if metrics.BACKEND.model_family(row.model_id) in {"hEVO","Evo 2","GPN","NT"} else NEUTRAL_LIGHT
        pref.errorbar(row.accuracy*100,row.y,
            xerr=[[100*(row.accuracy-row.accuracy_ci_low)],[100*(row.accuracy_ci_high-row.accuracy)]],
            fmt="none",ecolor=NEUTRAL,elinewidth=.55,capsize=1.2,zorder=2)
        pref.scatter(row.accuracy*100,row.y,marker=style["marker"],s=15,
                     facecolor=color,edgecolor=BLACK,linewidth=.4,zorder=3)
    for ax in [pref,ch_ax]:
        ax.set_yticks([])
        for edge in ["left","top","right"]:
            ax.spines[edge].set_visible(False)
        ax.spines["bottom"].set_linewidth(1)
        ax.tick_params(axis="x",length=2,width=.55,pad=1.5,labelsize=6)
    pref.set_xlim(0,100)
    pref.set_xticks([0,50,100])
    pref.set_xlabel("Loci (%)",fontsize=6,labelpad=4)
    pref.set_title("Selected-allele\npreference",fontsize=6.4,fontweight="bold",pad=5,linespacing=1)
    lookup=ch.set_index("model_id")
    for row in plot_data.itertuples(index=False):
        value=lookup.loc[row.model_id,"ch_index"]
        if lookup.loc[row.model_id,"ch_status"]=="PASS":
            style=metrics._family_style(row.model_id)
            ch_ax.scatter(value,row.y,marker=style["marker"],s=15,facecolor=style["color"],
                          edgecolor=BLACK,linewidth=.4,zorder=3)
        else:
            ch_ax.text(.5,row.y,"NA",transform=ch_ax.get_yaxis_transform(),ha="center",va="center",fontsize=6)
    ch_ax.set_xscale("log")
    ch_ax.set_xlim(1e-6,200)
    ch_ax.set_xticks([1e-4,1e-2,1],["$10^{-4}$","$10^{-2}$","1"])
    ch_ax.xaxis.set_minor_locator(NullLocator())
    ch_ax.set_xlabel("Raw scores, log scale",fontsize=6,labelpad=4)
    ch_ax.set_title("Calinski-Harabasz\nindex",fontsize=6.4,fontweight="bold",pad=5,linespacing=1)
    figure._hevo_fig4_g_map_source_data=map_data
    figure._hevo_fig4_g_source_data=ch
    figure._hevo_fig4_g_ordered_data=ordered
    return map_ax,pref,ch_ax

def write_map_source_data(data: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
