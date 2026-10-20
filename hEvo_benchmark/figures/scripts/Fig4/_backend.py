"""Fig.4 重构版的冻结 Matplotlib backend 和本地数据适配。"""

from __future__ import annotations

from pathlib import Path

from common.legacy_loader import load_module
from common.panel_runtime import panel_tables_dir


BACKEND = load_module("Fig4/build_fig4abc_compact_main_and_supp.py")


def load_data(data_dir: Path | None = None):
    """把当前 S33/S34/S35 的独立副本注入兼容绘图逻辑。"""

    root = panel_tables_dir("Fig4", data_dir)
    BACKEND.S26 = root / "S33_unsupervised_separation_of_1000_genomes_ancestry_groups.csv"
    BACKEND.S27 = root / "S34_supervised_classification_of_1000_genomes_ancestry_groups.csv"
    BACKEND.S28 = root / "S35_few_shot_regression_of_ancestry_proportions.csv"
    return BACKEND.load_data()


def selected(data_dir: Path | None = None):
    """返回主图 a、b、c 所需的代表模型表。"""

    unsup, cls, few = load_data(data_dir)
    unsup_main = BACKEND.representative_rows(unsup, "rank_score")
    cls_main = BACKEND.representative_rows(cls, "classification_mcc")
    endpoint = few.loc[few["shot"].eq(60)].sort_values("pearson_r_mean", ascending=False).reset_index(drop=True)
    end_main = BACKEND.representative_rows(endpoint, "pearson_r_mean")
    return unsup, cls, few, unsup_main, cls_main, end_main
