"""Fig.3i–k temporal ClinVar panel 的共享适配。"""

from __future__ import annotations

from pathlib import Path

from common.panel_runtime import panel_tables_dir
from common.reportlab_common import crop_drawing, mm_drawing, register_arial_aliases
from Fig3._backend import BACKEND, load_sources

WIDTH_MM, HEIGHT_MM = 64.6666666667, 55.0
LETTERS = {"blb_stable_vs_upgraded": "i", "plp_stable_vs_downgraded": "j", "vus_resolved_path_vs_benign": "k"}
TASK_BY_LETTER = {value: key for key, value in LETTERS.items()}
X_BY_LETTER = {"i": 5.0, "j": 72.6666666667, "k": 140.3333333334}


def build_temporal(
    tables_dir: Path | None,
    source_name: str,
    task_letter: str,
    *,
    panel_letter: str | None = None,
):
    """读取指定 temporal 快照，补 prevalence 后绘制一个时间序列 panel。"""

    root = panel_tables_dir("Fig3", tables_dir)
    _, _, temporal = load_sources(root, temporal_name=source_name)
    task = TASK_BY_LETTER[task_letter]
    display_letter = panel_letter or task_letter
    # Display letter is independent from task semantics so the reference i/j
    # order can be reproduced without exchanging the underlying task data.
    titles = {"i": "LB/B → VUS or P/LP", "j": "P/LP → VUS or LB/B", "k": "VUS → P/LP"}
    BACKEND.core.TEMPORAL_META[task]["letter"] = display_letter
    BACKEND.core.TEMPORAL_META[task]["title"] = titles[task_letter]
    x, y = X_BY_LETTER[display_letter], 4.5
    register_arial_aliases()
    full = mm_drawing(210.0, 297.0)
    BACKEND.core.draw_temporal_panel(full, temporal, task=task, x=x, width=WIDTH_MM, top=59.5, bottom=y, inner_left=4.25, inner_right=4.25)
    return crop_drawing(full, x, y, WIDTH_MM, HEIGHT_MM)
