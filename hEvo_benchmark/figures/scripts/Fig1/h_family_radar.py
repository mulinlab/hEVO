"""Fig.1h: render the current canonical family-rank radar from S10."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from common.data_io import require_columns  # noqa: E402
from common.legacy_loader import load_module  # noqa: E402
from common.panel_runtime import input_files, panel_output_dir, panel_tables_dir  # noqa: E402
from common.paths import figure_dir  # noqa: E402
from common.reportlab_common import export_drawing, register_arial_aliases  # noqa: E402

BACKEND = load_module("Fig1/plot_fig1h_family_rank_radars_compact.py")
WIDTH_MM, HEIGHT_MM = 64.0, 128.0
SOURCE_NAME = "S10_best_model_family_ranks_across_functional_variant_and_evolutionary_tasks.csv"
FAMILIES = {"hEVO family", "Evo 2 family", "NT family", "GPN family"}


def _load(tables_dir: Path | None = None):
    root = panel_tables_dir("Fig1", tables_dir)
    source = root / SOURCE_NAME
    table = pd.read_csv(source)
    require_columns(
        table,
        [
            "category", "axis", "axis_label", "family", "best_model",
            "rank", "n_eligible", "canonical_roster_size", "display_radius",
            "source_tables", "status",
        ],
        "Fig1h",
    )
    if set(table["family"]) != FAMILIES:
        raise ValueError("Fig1h: family roster mismatch")
    if set(table["canonical_roster_size"].astype(int)) != {29}:
        raise ValueError("Fig1h: S10 must use the canonical 29-model roster")
    if (pd.to_numeric(table["n_eligible"], errors="raise") > 29).any():
        raise ValueError("Fig1h: eligible denominator exceeds the canonical roster")
    if table.duplicated(["category", "axis", "family"]).any():
        raise ValueError("Fig1h: duplicate category-axis-family rows")
    passed = table["status"].eq("PASS")
    finite = pd.to_numeric(table.loc[passed, "display_radius"], errors="raise")
    if not finite.between(0.0, 1.0).all():
        raise ValueError("Fig1h: normalized rank radius must be within [0, 1]")
    unavailable = table["status"].eq("NOT_AVAILABLE")
    if table.loc[unavailable, "display_radius"].notna().any():
        raise ValueError("Fig1h: unavailable family rows must not carry a radius")
    return table, source



# Reference geometry is in mm, with native Arial text and vector strokes.
# The source-table radius is used directly on the shared 0--1 scale.
def render_vertical(table):
    from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
    from reportlab.lib.colors import HexColor
    from reportlab.lib.units import mm

    drawing = Drawing(WIDTH_MM * mm, HEIGHT_MM * mm)
    drawing.add(Rect(0, 0, drawing.width, drawing.height,
                     fillColor=HexColor("#FFFFFF"), strokeColor=None))

    def text(x, y, label, size=6.0, anchor="start", bold=False):
        drawing.add(String(x * mm, y * mm, label, fontSize=size,
                           fontName="Arial-Bold" if bold else "Arial",
                           fillColor=HexColor("#000000"), textAnchor=anchor))

    def line(x1, y1, x2, y2, color, width):
        drawing.add(Line(x1 * mm, y1 * mm, x2 * mm, y2 * mm,
                         strokeColor=HexColor(color), strokeWidth=width))

    styles = {
        "hEVO family": dict(color="#B35C1E", marker="circle", width=1.1),
        "Evo 2 family": dict(color="#0E7C86", marker="triangle", width=0.85),
        "NT family": dict(color="#8A95A5", marker="square", width=0.85),
        "GPN family": dict(color="#0B2E83", marker="diamond", width=0.85),
    }
    text(0.3, 123, "h", 14, bold=True)
    for family, x, y in (("hEVO family",14,125.1), ("Evo 2 family",44,125.1),
                         ("NT family",14,121.1), ("GPN family",44,121.1)):
        style = styles[family]
        line(x,y,x+4,y,style["color"],0.9)
        BACKEND.marker(drawing,x+2,y,style,radius=0.45)
        text(x+5,y-0.7,family)
    text(36.2,116.5,"Family-best normalized rank",anchor="middle")

    for (title, axes), cy in zip(BACKEND.REVISED_CATEGORY_SPECS,(96.0,58.0,20.0)):
        cx, radius = 36.2, 12.7
        subset = table.loc[table["category"].eq(title)]
        if set(subset["axis"]) != set(axes):
            raise ValueError(f"Fig1h: axis roster mismatch for {title}")
        if len(subset) != len(axes) * len(styles):
            raise ValueError(f"Fig1h: incomplete source rows for {title}")
        labels = subset.drop_duplicates("axis").set_index("axis")["axis_label"]
        text(cx,cy+16.1,title,6.6,anchor="middle",bold=True)
        outer=[]
        for i,axis in enumerate(axes):
            angle=math.pi/2-2*math.pi*i/len(axes)
            dx,dy=math.cos(angle),math.sin(angle)
            vx,vy=cx+radius*dx,cy+radius*dy
            outer.extend([vx*mm,vy*mm])
            line(cx,cy,vx,vy,"#BBC3CB",0.3)
            anchor="middle" if abs(dx)<0.25 else "start" if dx>0 else "end"
            lx,ly=cx+(radius+1.8)*dx,cy+(radius+1.8)*dy-0.7
            if abs(dx)<0.25 and dy>0:ly=cy+13.5
            text(lx,ly,str(labels[axis]),anchor=anchor)
        drawing.add(Polygon(outer,fillColor=None,strokeColor=HexColor("#A5AFB8"),strokeWidth=0.4))
        for family in reversed(tuple(styles)):
            style=styles[family]
            rows=subset.loc[subset["family"].eq(family)].set_index("axis")
            coords=[]
            for i,axis in enumerate(axes):
                row=rows.loc[axis]
                if row["status"] == "NOT_AVAILABLE":
                    coords.append(None)
                    continue
                if row["status"] != "PASS":
                    raise ValueError(f"Fig1h: unexpected status for {family}/{axis}")
                value=float(row["display_radius"])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Fig1h: invalid radius")
                angle=math.pi/2-2*math.pi*i/len(axes)
                coords.append((cx+radius*value*math.cos(angle),cy+radius*value*math.sin(angle)))
            for i,first in enumerate(coords):
                second=coords[(i+1)%len(coords)]
                if first is not None and second is not None:
                    line(*first,*second,style["color"],style["width"])
            for point in coords:
                if point is not None:
                    BACKEND.marker(drawing,*point,style,radius=0.48 if family=="hEVO family" else 0.43)
    if table["status"].eq("NOT_AVAILABLE").any():
        text(36.2,77.8,"Open edges indicate unavailable family-task results",4.7,anchor="middle")
    text(36.2,1,"0 = centre; 1 = outer",anchor="middle")
    return drawing


def build_panel(tables_dir: Path | None = None):
    table, _ = _load(tables_dir)
    register_arial_aliases()
    return render_vertical(table)


def export_panel(output_dir: Path | None = None, tables_dir: Path | None = None):
    _, source = _load(tables_dir)
    return export_drawing(
        build_panel(tables_dir),
        panel_output_dir("Fig1", output_dir),
        "h_family_radar",
        input_files(source),
        width_mm=WIDTH_MM,
        height_mm=HEIGHT_MM,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", "--data-dir", dest="tables_dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=figure_dir("Fig1"))
    args = parser.parse_args()
    export_panel(args.output_dir, args.tables_dir)


if __name__ == "__main__":
    main()
