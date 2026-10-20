"""Fig.3c–h 的小型 panel 构建辅助。

主脚本仍是 c/d/e/f/g/h 各自的独立入口；本文件只抽出完全相同的
ReportLab 坐标裁剪和 `draw_numbered_bar_panel()` 调用，避免六份代码产生
细小漂移。传入的 `source_name` 决定该 panel 使用的独立数据副本。
"""

from __future__ import annotations

from pathlib import Path

from common.panel_runtime import panel_tables_dir
from common.reportlab_common import crop_drawing, mm_drawing, register_arial_aliases
from Fig3._backend import BACKEND, load_sources, static_checks


# The source panel header extends 3–4 mm above its plotting box. Include that
# header in the standalone export canvas instead of cropping through capitals.
WIDTH_MM, HEIGHT_MM = 64.6666666667, 60.0
CONTENT_HEIGHT_MM = 55.5
X_BY_PANEL = {"c": 5.0, "d": 72.6666666667, "e": 140.3333333334, "f": 5.0, "g": 72.6666666667, "h": 140.3333333334}
Y_BY_PANEL = {"c": 130.0, "d": 130.0, "e": 130.0, "f": 66.5, "g": 66.5, "h": 66.5}
HEADER_BY_TASK = {
    "causal": "d",
    "rare_causal": "e",
    "rare_ukb": "f",
    "ddd_clinvar": "g",
    "mpra": "h",
}


def build_static(tables_dir: Path | None, source_name: str, task: str, letter: str):
    """从指定静态快照抽取一个 task，绘制并裁出主图的 64.67×55.5 mm panel。"""

    root = panel_tables_dir("Fig3", tables_dir)
    _, static, _ = load_sources(root, static_name=source_name)
    rows = static[static["task"].eq(task)]
    checks = static_checks(static)[task]
    meta = BACKEND.core.STATIC_META[task]
    x, y = X_BY_PANEL[letter], Y_BY_PANEL[letter]
    register_arial_aliases()
    full = mm_drawing(210.0, 297.0)
    BACKEND.draw_numbered_bar_panel(full, rows, x=x, bottom=y, width=WIDTH_MM, top=y + CONTENT_HEIGHT_MM, letter=letter, header_key=HEADER_BY_TASK[task], n_samples=int(checks["num_samples"]), baseline=float(checks["baseline"]), limits=meta["scale"], ticks=meta["ticks"], score_col="score", low_col="ci_lower", high_col="ci_upper")
    return crop_drawing(full, x, y, WIDTH_MM, HEIGHT_MM)
