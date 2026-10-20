#!/usr/bin/env python3
"""Collect review copies, source tables, change receipts and remaining gaps."""

from pathlib import Path
import json
import shutil

import pandas as pd


def main():
    root = Path(__file__).resolve().parents[3]
    out = root / "article/revision_20261010"
    tables = root / "hEvo_benchmark/tables"
    companion = tables / "revision_consistency"
    sensitivity = out / "cllr_sensitivity"
    for name in [
        "cllr_sensitivity_summary.csv",
        "retrospective_cllr_summary.csv",
        "retrospective_cllr_by_seed.csv",
    ]:
        shutil.copy2(sensitivity / name, companion / name)
    for name in [
        "hEvo_Methods_中文论文版.md",
        "hEvo_附加材料Methods_中文论文版.md",
        "hEvo_附表说明_中文论文版.md",
        "hEvo_Supplementary_Table_Titles_EN.md",
    ]:
        shutil.copy2(root / "article" / name, out / name)
    delivery_tables = out / "tables"
    delivery_tables.mkdir(exist_ok=True)
    for p in tables.glob("*.csv"):
        # Do not issue deferred ablations as newly revised evidence.
        if not p.name.startswith(("S41_", "S42_")):
            shutil.copy2(p, delivery_tables / p.name)
    shutil.copytree(
        companion, delivery_tables / "revision_consistency", dirs_exist_ok=True
    )
    for figure in [
        "Fig1",
        "Fig2",
        "Fig3",
        "Fig4",
        "Fig5",
        "FigS6",
        "FigS8",
        "FigS9",
        "FigS10",
    ]:
        folder = out / "figures" / figure
        folder.mkdir(parents=True, exist_ok=True)
        for ext in ["pdf", "svg", "png"]:
            p = root / "hEvo_benchmark/figures" / figure / ("A4_integrated." + ext)
            if p.exists():
                shutil.copy2(p, folder / p.name)
    receipts = json.loads((out / "document_revision_receipts.json").read_text())
    rank = pd.read_csv(companion / "common_task_rank_sensitivity.csv")
    _s = pd.read_csv(sensitivity / "cllr_sensitivity_summary.csv")
    (out / "README.md").write_text(f"""# 论文与结果一致性修订交付（2026-10-10）

本轮以实施开始时的工作区为基准，保留原有作者修改。**按你的最新要求，消融部分暂停：S41/S42 已恢复原始字节，正文消融段落、中文方法消融章节和 Fig.5 a–c 均保持原样。消融未纳入本轮结论验收。**

## 先看这些文件

- [英文主稿（带修订）](Manuscript_to_XQY_revised_with_changes.docx)
- [英文补充方法，Science/Arial 版（带修订）](hEvo_Supplementary_Methods_English_Science_Arial11_revised_with_changes.docx)
- [英文补充方法，可编辑公式版（带修订）](hEvo_Supplementary_Methods_English_Editable_Equations_revised_with_changes.docx)
- [逐项修改清单](修改清单.md)
- [校准敏感性图](cllr_sensitivity/cllr_sensitivity.pdf)及[完整汇总](cllr_sensitivity/cllr_sensitivity_summary.csv)

原 Word 未覆盖。三份副本合计 {sum(len(r["changes"]) for r in receipts)} 项带修订文本修改；主稿保留 {receipts[0]["preserved_field_instructions"]} 个引用字段、{receipts[0]["preserved_equations"]} 个公式和 {receipts[0]["preserved_prior_revisions"]} 个已有修订。新修订可以独立拒绝，拒绝后可恢复原稿可读文本。

## 实际变更

- S49 按当前 S33–S35 等来源重建，共 {len(rank)} 个模型×任务记录、34 个模型、45 个排名单位；S43 同步排名并按 S44 复算，资源模型为 32 个。
- 全体模型的共同任务集合为 38 项。hEvo-large、Evo2-7B、hEvo-base 的平均排名分别为 {rank.set_index("model").loc["hEvo-large", "common_task_mean_rank"]:.3f}、{rank.set_index("model").loc["evo2_7b_base", "common_task_mean_rank"]:.3f}、{rank.set_index("model").loc["hEvo-base", "common_task_mean_rank"]:.3f}；共同任务比较改变了后两者的顺序。
- 建立 48 endpoint / 15 group 清单，保持与 45 个排名单位的区别；删除无法复现规则的 14/15、34/48 精确胜出声明。
- S48 和其生成器统一为直接等位基因配对评分；修正 S39 相关协议描述。
- DeepSEA 的派生汇总使用已有 run24 选择规则，保留 run22 历史记录，不将不同行为的运行混平均。
- GPN-Star raw 主表保持不变。完成 24 项匹配点估计对照（3 模型×8 分析），以及 81 个年份×事件的回顾性对照，每项保留 5 个种子。raw 分数可复现主表；cLLR 为独立、描述性敏感性结果。
- 图和正文采用当前 Fig.4 a–f 布局：祖源分类 a、少样本回归 b、HAR c、HAQER d、区域轨道 e、等位基因偏好 f。Fig. 4e 的 hEvo-large 与 GPN-Star-M 轨道合并保存在 `Fig4e_track_source_data.csv`。
- Fig.3 按实际后端与成图采用 i=downgrade、j=upgrade、k=VUS→P/LP；a/b 示意图的编号为 ClinVar 1、DDD 2、GWAS 3、rare-causal 4、UKB rare 5、MPRA 6。
- HAR/HAQER、MPRA、URE、ClinVar 和资源数字及比较范围同步修订；删除未经证实的模型公开可获取性和硬件匹配表述。
- rare-causal 当前最强外部比较为 GPN-MSA，raw AUPRC 约 0.448，hEvo-large 相对提升约 25.2%。CH 仅衡量 raw-score 分布分离，不能判断是否偏好文献指定等位基因；已删除错误的 hEvo CH 全面优势描述。

## 核验范围

数值与来源哈希、原始 raw 表、消融恢复、Word 字段/公式/已有修订以及 Fig.5 消融区域像素核对均通过。Fig.5 a–c 与基准图对应区域的不同像素数为 0。已重建并查看受影响图件；图源和完整 PDF/SVG/PNG 在 `figures/`。

硬件、dtype 和实际 batch 以资源表中记录的配置为准，空值表示来源未提供该字段。本目录的文件对应仓库内保存的版本。

当前环境没有 Word/LibreOffice 排版引擎，因此 Word 完成了包结构、最终文本及字段/公式保护检查，未宣称完成逐页渲染检查。`*.final.txt` 仅供文字核对，公式对象保留在 DOCX 中。

`article/` 在当前仓库中被忽略；本目录是可直接打开的交付文件，不依赖 Git 记录。代码、来源表及仓库图的修改仍在当前工作区。本轮未自动提交或发布。

## 保存的证据

修订核验脚本位于 `hEvo_benchmark/scripts/revision_consistency/`。结果表、endpoint 清单和敏感性分析表保留在 `tables/revision_consistency/`；未保留实施前快照、状态转储或差异审计包。
""")
    lines = [
        "# 逐项修改清单",
        "",
        "消融部分按用户要求暂停并恢复；以下修改清单不包含消融结论调整。",
        "",
    ]
    for receipt in receipts:
        lines.extend(["## " + Path(receipt["output"]).name, ""])
        for i, change in enumerate(receipt["changes"], 1):
            lines.extend(
                [
                    f"### {i}. {change['reason']}",
                    "",
                    f"原稿段落索引：{change['paragraph_index']}（从 0 起；用于定位 XML，不是 Word 页码）。",
                    "",
                    "**原文**",
                    "",
                    change["old"],
                    "",
                    "**修订后**",
                    "",
                    change["new"],
                    "",
                ]
            )
    (out / "修改清单.md").write_text("\n".join(lines) + "\n")
    print("Review package:", out)


if __name__ == "__main__":
    main()
