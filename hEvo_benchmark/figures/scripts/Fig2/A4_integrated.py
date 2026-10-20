"""Fig.2 A4 整合入口。

每个 panel 脚本独立构建 Drawing；本文件只负责按当前六-panel 版式的 mm
坐标组合图形，因而不会再次实现任何统计或数据筛选逻辑。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pypdf import PdfReader
from reportlab.graphics.shapes import Group, Rect, String
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm

HERE = Path(__file__).resolve()
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from common.panel_runtime import input_files, panel_tables_dir  # noqa: E402
from common.paths import tables_dir, figure_dir  # noqa: E402
from common.public_paths import publicize_paths  # noqa: E402
from common.reportlab_common import export_drawing, mm_drawing, register_arial_aliases  # noqa: E402
from Fig2 import a_model_ch, c_genomic_benchmark, d_nt_benchmark, e_ure_binary, f_ure_multiclass, g_deepsea  # noqa: E402

PAGE_W_MM, PAGE_H_MM = 210.0, 297.0
GPN_PROMOTER = "GPN-Promoter"
GPN_MSA = "GPN-MSA"
FIG2C_EXPECTED_MODEL_COUNT = 19
FIG2C_HEVO_MODELS = ("hEvo-large", "hEvo-base")
PANEL_LAYOUT_MM = {
    "a": {"x": 10.0, "y": 220.1, "width": 190.0, "height": 69.0, "source": "Genomic Benchmark"},
    "b": {"x": 10.0, "y": 145.0, "width": 190.0, "height": 73.0, "source": "NT benchmark"},
    "c": {"x": 10.0, "y": 89.5, "width": 190.0, "height": 53.0, "source": "Model-level CH"},
    "d": {"x": 10.0, "y": 23.0, "width": 64.0, "height": 63.0, "source": "URE multiclass F1"},
    "e": {"x": 77.0, "y": 23.0, "width": 61.0, "height": 63.0, "source": "URE binary MCC"},
    "f": {"x": 141.0, "y": 23.0, "width": 58.0, "height": 63.0, "source": "DeepSEA AUROC"},
}


def translated(panel, x: float, y: float) -> Group:
    """把 panel 的本地原点平移到 A4 的既定左下坐标。"""

    group = Group(*panel.contents)
    group.translate(x * mm, y * mm)
    return group


def draw_family_legend(drawing) -> None:
    """Draw the shared model-family key centered under panels d–f."""

    y = 19.2
    drawing.add(String(39.0 * mm, y * mm, "Model family", fontName="Arial-Bold",
                       fontSize=5.2, fillColor=HexColor("#000000")))
    entries = (
        ("hEVO", "#B35C1E", 59.0, 3.2, 21.0),
        ("Evo 2", "#0E7C86", 82.0, 3.2, 22.0),
        ("NT", "#65717F", 106.0, 3.2, 18.0),
        ("GPN", "#0B2E83", 126.0, 3.2, 20.0),
        ("Other models", "#9AA3AF", 148.0, 3.2, 34.0),
    )
    for label, color, x, width, _step in entries:
        drawing.add(Rect(x * mm, (y - 0.05) * mm, width * mm, 1.45 * mm,
                         fillColor=HexColor(color), strokeColor=None))
        drawing.add(String((x + width + 1.3) * mm, y * mm, label,
                           fontName="Arial", fontSize=5.2,
                           fillColor=HexColor("#000000")))


def _visible_text(node) -> list[str]:
    """Collect text from a Drawing/Group tree for display-contract checks."""

    result: list[str] = []
    value = getattr(node, "text", None)
    if value is not None:
        result.append(str(value))
    for child in getattr(node, "contents", []):
        result.extend(_visible_text(child))
    return result


def _normalized_model_text(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _validate_panel_model_contract(panels: dict[str, object]) -> dict[str, object]:
    """Validate the focused CH panel and existing sequence-model contracts."""

    panel_text = {label: "\n".join(_visible_text(panel)) for label, panel in panels.items()}
    expected_genomic_models = tuple(d_nt_benchmark.FIG2A_REFERENCE_MODELS)
    expected_nt_models = tuple(c_genomic_benchmark.FIG2B_DISPLAY_MODELS)
    if tuple(getattr(panels["a"], "_fig2a_display_models", ())) != expected_genomic_models:
        raise ValueError("Fig.2a model order differs from the supplied PDF reference")
    if tuple(getattr(panels["b"], "_fig2b_display_models", ())) != expected_nt_models:
        raise ValueError("Fig.2b model order differs from the current S17 data")
    if set(getattr(panels["d"], "_fig2d_display_models", ())) != set(f_ure_multiclass.REFERENCE_MODEL_ORDER):
        raise ValueError("Fig.2d model roster differs from the supplied PDF reference")
    if set(getattr(panels["e"], "_fig2e_display_models", ())) != set(e_ure_binary.REFERENCE_MODELS):
        raise ValueError("Fig.2e model roster differs from the supplied PDF reference")
    if _normalized_model_text(GPN_PROMOTER) not in _normalized_model_text(panel_text["a"]):
        raise ValueError("Fig.2a must retain the complete GPN-Promoter row")
    if _normalized_model_text(GPN_PROMOTER) not in _normalized_model_text(panel_text["b"]):
        raise ValueError("Fig.2b must retain the GPN-Promoter row")
    if _normalized_model_text(GPN_MSA) not in _normalized_model_text(panel_text["b"]):
        raise ValueError("Fig.2b must retain the complete GPN-MSA row")
    if "gpnstar" in _normalized_model_text(panel_text["a"]):
        raise ValueError("Fig.2a must not display a GPN-Star series")
    if "na" in {token.strip(" ,;:") for token in panel_text["a"].split()}:
        raise ValueError("Fig.2a must not display an NA cell or row")
    if "Silencer" not in panel_text["d"]:
        raise ValueError("Fig.2d must display the Silencer class from the five-class URE task")
    gpn_star_selection = getattr(panels["d"], "_gpn_star_ure_selection", None)
    if (
        not isinstance(gpn_star_selection, dict)
        or set(gpn_star_selection.get("candidates", []))
        != {"GPN-Star-M", "GPN-Star-P", "GPN-Star-V"}
        or len(gpn_star_selection.get("selected", [])) != 1
        or "gpn-star" not in panel_text["d"].lower()
    ):
        raise ValueError("Fig.2d must select and display one registered GPN-Star M/P/V model")
    displayed_ch_models = tuple(getattr(panels["c"], "_fig2c_display_models", ()))
    displayed_hevo_models = tuple(getattr(panels["c"], "_fig2c_hevo_models", ()))
    expected_representatives = set(a_model_ch.FIG2C_REFERENCE_MODEL_ORDER)
    if len(displayed_ch_models) != FIG2C_EXPECTED_MODEL_COUNT:
        raise ValueError(
            f"Fig.2c must display the {FIG2C_EXPECTED_MODEL_COUNT} models in the supplied reference, got {len(displayed_ch_models)}"
        )
    if set(displayed_ch_models) != expected_representatives:
        raise ValueError("Fig.2c representative selection differs from the prior figure")
    if set(displayed_hevo_models) != set(FIG2C_HEVO_MODELS):
        raise ValueError(
            f"Fig.2c must include only hEvo-large/base from the hEvo family, got {displayed_hevo_models}"
        )
    return {
        "sequence_classification_panels": ["a", "b"],
        "genomic_benchmark_model_order": list(expected_genomic_models),
        "nt_benchmark_model_order": list(expected_nt_models),
        "gpn_sequence_models": [GPN_PROMOTER, GPN_MSA],
        "gpn_promoter_present_in_genomic_and_nt_panels": True,
        "gpn_msa_present_in_nt_panel_only": True,
        "gpn_star_excluded_from_genomic_panel": True,
        "na_cells_absent_from_genomic_panel": True,
        "ure_five_class_panel": "d",
        "silencer_present_in_ure_multiclass_panel": True,
        "gpn_star_ure_multiclass_selection": gpn_star_selection,
        "genomic_element_ch_panel": {
            "representative_model_count": len(displayed_ch_models),
            "representative_models": list(displayed_ch_models),
            "hEvo_family_models": list(displayed_hevo_models),
            "selection_source": "supplied Figure 2 PDF; 19 visible model labels",
            "other_family_selection": "exact reference roster",
            "exact_model_scope_validated": True,
        },
        "benchmark_source_summary": getattr(panels["b"], "_benchmark_source_summary", None),
    }


def _validate_exported_pdf(pdf_path: Path) -> dict[str, object]:
    """Reopen the final artifact so stale vector text cannot bypass the gate."""

    reader = PdfReader(str(pdf_path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    normalized = _normalized_model_text(text)
    gpn_occurrences = {
        name: normalized.count(_normalized_model_text(name))
        for name in (GPN_PROMOTER, GPN_MSA)
    }
    if gpn_occurrences[GPN_PROMOTER] < 2 or gpn_occurrences[GPN_MSA] < 1:
        raise ValueError(
            "Exported Fig.2 PDF must show GPN-Promoter in panels a/b and "
            f"GPN-MSA in panel b: {gpn_occurrences}"
        )
    if "Silencer" not in text:
        raise ValueError("Exported Fig.2 PDF must include Silencer in panel d")
    if "gpnstar" not in normalized:
        raise ValueError("Exported Fig.2 PDF must include the selected GPN-Star model in panel d")
    return {
        "pdf_reopened": True,
        "gpn_model_text_occurrences": gpn_occurrences,
        "silencer_text_present": True,
        "gpn_star_text_present": True,
    }


def build_panel(tables_dir: Path | None = None):
    """按参考图组合 a–f；函数名保持统一接口以便上层自动化调用。"""

    root = panel_tables_dir("Fig2", tables_dir)
    register_arial_aliases()
    page = mm_drawing(PAGE_W_MM, PAGE_H_MM)
    page.add(Rect(0, 0, page.width, page.height, fillColor=HexColor("#FFFFFF"), strokeColor=None))
    # Reference order: genomic benchmarks, NT, model-level CH, URE F1,
    # URE binary MCC, and DeepSEA AUROC.
    panels = {
        "a": d_nt_benchmark.build_panel(root, panel_letter="a"),
        "b": c_genomic_benchmark.build_panel(root, panel_letter="b"),
        "c": a_model_ch.build_panel(
            root,
            panel_letter="c",
            standalone_legend=True,
            height_mm=PANEL_LAYOUT_MM["c"]["height"],
        ),
        "d": f_ure_multiclass.build_panel(root, panel_letter="d"),
        "e": e_ure_binary.build_panel(root, panel_letter="e"),
        "f": g_deepsea.build_panel(root, panel_letter="f"),
    }
    model_contract = _validate_panel_model_contract(panels)
    for label, panel in panels.items():
        x, y = PANEL_LAYOUT_MM[label]["x"], PANEL_LAYOUT_MM[label]["y"]
        page.add(translated(panel, x, y))
    draw_family_legend(page)
    page._fig2_model_contract = model_contract  # type: ignore[attr-defined]
    return page


def export_panel(
    output_dir: Path | None = None,
    tables_dir: Path | None = None,
):
    root = panel_tables_dir("Fig2", tables_dir)
    files = [path for path in root.iterdir() if path.is_file()]
    out = output_dir or figure_dir("Fig2")
    page = build_panel(root)
    result = export_drawing(
        page,
        out,
        "A4_integrated",
        input_files(*files),
        width_mm=PAGE_W_MM,
        height_mm=PAGE_H_MM,
    )
    model_contract = dict(page._fig2_model_contract)  # type: ignore[attr-defined]
    model_contract.update(_validate_exported_pdf(Path(result["pdf"])))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", "--data-dir", dest="tables_dir", type=Path, default=tables_dir())
    parser.add_argument("--output-dir", type=Path, default=figure_dir("Fig2"))
    args = parser.parse_args()
    export_panel(args.output_dir, args.tables_dir)


if __name__ == "__main__":
    main()
