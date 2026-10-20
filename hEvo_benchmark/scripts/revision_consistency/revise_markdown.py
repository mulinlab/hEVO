#!/usr/bin/env python3
"""Synchronize method prose while leaving deferred ablation sections intact."""

from pathlib import Path
import json
import re
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[3]
    tables = root / "hEvo_benchmark/tables"
    s43 = pd.read_csv(next(tables.glob("S43_*.csv")))
    inventory = json.loads(
        (tables / "revision_consistency/rank_inventory.json").read_text()
    )
    common = inventory["common_task_units"]
    methods = root / "article/hEvo_Methods_中文论文版.md"
    original = methods.read_text()
    text = original
    text = text.replace(
        "## Public resource model training and release",
        "## Models trained on public genomic resources",
    )
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "[GPN-Star]" in line and "原生 VEP" in line and "cLLR" not in line:
            lines[i] += (
                " GPN-Star 主表使用未经突变率校准的 raw LLR，并沿用预先确定的任务方向；另行计算 cLLR 敏感性结果。cLLR 从 raw LLR 中减去与模型固定 revision 对应、按正向参考链中心五核苷酸及 ALT 查询的官方中性均值，之后应用相同的任务方向。raw/cLLR 比较使用相同的有效变异集合，校准覆盖率、模型版本和输入哈希随结果保存；主表不被敏感性结果覆盖。"
            )
        if line.startswith("[DeepSEA]"):
            lines[i] = line.replace(
                "以逐标签二元交叉熵优化 3 个 epoch",
                "参考配置以逐标签二元交叉熵优化 3 个 epoch；当前选定的 hEvo-open-large S21_run_24 为一轮 FP16，实际输出为 pooler，声明配置却写 mean-pooling；历史 S21_run_22 为三轮 BF16、mean-pooling，两者分别保留，不作为匹配训练比较",
            )
        if line.startswith("无监督元件分析分别使用") and "主图 Fig.2c" not in line:
            lines[i] += (
                " 主图 Fig.2c 只展示 19 个代表模型在三套数据上的二维 PCA CH 指标；PCA/t-SNE 的分别比较保留在 Fig.S4 和 S11–S16。"
            )
        if "该指标量化文献支持" in line and "区间均跨越" not in line:
            lines[i] += (
                " 30 位点的 hEvo-base/large 精确区间均跨越 0.5，因此按描述性的等位基因方向一致性解释，不据此声称可靠检出正选择。"
            )
        if line.startswith("每个模型的任务表现以可用任务 MCC 的均值汇总"):
            lines[i] = (
                f"每个模型在 18 项 NT 任务的 MCC 及 S43 的各资源指标按算术平均汇总；资源比较要求对应指标及性能具有 18/18 有效记录。S43/S44 当前包含 {len(s43)} 个模型、{len(s43) * 18} 个模型×任务记录。Fig.5 e–h 纵轴另用 S49 的 45 个排名单位中可用任务的平均 rank，越低越好，缺失任务不插补；S49 共 {inventory['models']} 个模型，其中 18/45（40%）单位来自 NT。这与 48 个主文 endpoint、15 个任务组是不同口径。共同任务敏感性使用 {common} 项任务。输入长度及可恢复的运行设置由 companion manifest 记录，缺失 GPU 型号、dtype、实际 batch 等信息明确记为 unknown；资源差异只按记录配置解释，不宣称统一硬件下的速度比较。"
            )
    text = "\n".join(lines) + "\n"
    # Release provenance is separate from the intentionally preserved ablation prose.
    release_note = "模型的公开训练数据来源不等于权重已公开发布。S47 中 hEvo 的 release license 尚待确认；最终权重、许可、代码仓库及 annotation portal 地址须在投稿前补齐。"
    if release_note not in text:
        pos = text.index("## Comparator models and evaluation interfaces")
        text = text[:pos] + release_note + "\n\n" + text[pos:]
    if "## Regional HAQER score tracks" not in text:
        pos = text.index("## Literature supported adaptive allele preference")
        note = (
            "## Regional HAQER score tracks\n\n"
            "当前 Fig.4f 为 hEvo-large 与 GPN-Star-M 的监督窗口评分比较，主展示 seed 为 20260515。两模型分别在验证阴性上校准；共同阈值 0.59 在验证集上以两模型 summit-hit F1 的宏平均选定。共享阈值不保证概率校准一致。12 个示例包含 6 个验证及 6 个测试位点，按 hEvo summit 落入金标准、显示区间内无 GPN 调用、局部分数差至少 0.15 和 hEvo 峰宽限制选择；缺少 GPN 调用不等于没有模型分数。214 个共享测试 HAQER 和 1,216 个标注窗口的整体指标另行报告。窗口分类与峰调用是不同评估单位；邻近 COL4A2 只提供位置背景。完整参数见补充方法 S11.3。\n\n"
        )
        text = text[:pos] + note + text[pos:]
    misplaced = "## Literature supported adaptive allele preference\n\n## Regional HAQER score tracks"
    if misplaced in text:
        start = text.index("## Literature supported adaptive allele preference")
        region = text.index("## Regional HAQER score tracks", start)
        end = text.index("适应性等位基因偏好在", region)
        block = text[region:end]
        text = (
            text[:start]
            + block
            + "## Literature supported adaptive allele preference\n\n"
            + text[end:]
        )

    # The complete ablation block must remain exactly as in the original.
    def ablation_block(value):
        return value.split("## Ablation and public corpus evaluations", 1)[1].split(
            "## Computational performance and deployment scope", 1
        )[0]

    assert ablation_block(text) == ablation_block(original)
    text = text.replace("Fig.4f", "Fig.4e")
    text = text.replace(
        "以包含三个 ALT 的位置组为单位",
        "以预先指定的阳性变异及其两个其他 ALT 构成的候选组为单位（不同组可共享坐标）",
    )
    methods.write_text(text)
    supp = root / "article/hEvo_附加材料Methods_中文论文版.md"
    original_supp = supp.read_text()
    text = original_supp
    text = text.replace(
        "延迟、吞吐量和显存对有效测量取中位数",
        "延迟、吞吐量和显存分别对 18 项 NT 任务的有效测量取算术平均",
    )
    text = text.replace(
        "当前记录范围为 29 个模型×18 项任务，共 522 个组合",
        f"当前记录范围为 {len(s43)} 个模型×18 项任务，共 {len(s43) * 18} 个组合",
    )
    if "GPN-Star 主表使用 raw" not in text:
        needle = "适应性位点任务采用总对数似然差，见 S12。"
        text = text.replace(
            needle,
            needle
            + " GPN-Star 主表使用 raw、未校准的原生 VEP LLR。独立 cLLR 敏感性使用各 checkpoint 固定 revision 的官方中心五核苷酸/ALT 中性均值，保持正向参考链查表及原有任务方向，并报告匹配变异集合、覆盖率及来源哈希。",
        )
    if "### S11.3 当前 HAQER" not in text:
        marker = "## S12."
        pos = text.index(marker)
        note = (
            "### S11.3 当前 HAQER 窗口评分与示例轨道\n\n"
            "Fig.4e 的现行来源为 `tables/fig4e_haqer_tracks/Fig4e_track_source_data.csv`，其中合并保存 hEvo-large 与 GPN-Star-M 的轨道数据。两者使用冻结表示及单独训练的监督窗口评分头，GPN-Star 对 1,024-bp 原生 MSA 输入的中心 512 bp 作官方正反向均值池化。窗口扫描步长为 64 bp；chr1–12 用于训练、chr13–17 验证、chr18–22 测试，标准化统计只由训练数据拟合，head 与 hard-negative 方案依据验证窗口 AUPRC 选择。显示 seed 为 20260515，整体测试指标汇总 20260515–20260517 三个种子。\n\n"
            "两模型分别在验证阴性上校准。共享阈值在 0.30–0.90、步长 0.01 的网格上最大化两模型验证 summit-hit F1 的宏平均；并列时依次取宏平均 recall 更高、总峰数更少者，选定阈值为 0.59。hEvo/GPN 的 Gaussian sigma 分别为 128/256 bp，truncate 都为 8，prominence 为 0.10/0.15，最小峰距离均为 1,024 bp，最小宽度为 64/256 bp。峰命中以 summit 落入原始半开 HAQER 区间定义。独立校准意味着同一阈值不保证概率校准一致。\n\n"
            "12 个示例为 6 个验证及 6 个测试位点；选择条件为 hEvo summit 位于金标准内、显示的 ±2 kb 内无 GPN 调用、局部平滑分数差至少 0.15、hEvo 峰宽至多 1.5 kb。HAQER0218 的 COL4A2 邻近背景不证明其调控该基因。整体指标另用完整共享测试集合（214 个 HAQER、1,216 个标注窗口）。完整轨道源数据与注释随图表脚本保存在仓库中。\n\n"
        )
        text = text[:pos] + note + text[pos:]
    resource_note = f"Fig.5 的 Pareto 纵轴使用 S49 的 45 个任务单位的平均 rank（越低越好），而不是 18 项 NT 的平均 MCC；缺失分数不插补。S49 有 {inventory['models']} 个模型，共同任务敏感性使用 {common} 个单位，NT 占总排名单位的 40%。硬件、精度和实际 batch 的逐次运行缺项在资源 companion manifest 标为 unknown，因此按现存配置解释测量差异。"
    if resource_note not in text:
        marker = "## S14."
        pos = text.index(marker)
        text = text[:pos] + resource_note + "\n\n" + text[pos:]

    def supp_ablation(value):
        return value.split("### S13.1 训练条件消融", 1)[1].split("### S13.2", 1)[0]

    assert supp_ablation(text) == supp_ablation(original_supp)
    text = text.replace("Fig.4f", "Fig.4e")
    text = text.replace(
        "| DDD 同位点 | 三 ALT 位置组 |",
        "| DDD 同位点 | 预先指定阳性变异的三候选组；不同组可共享坐标 |",
    )
    supp.write_text(text)
    readme = tables / "README.md"
    value = readme.read_text()
    # Publication table guide is maintained separately from local revision evidence.
    value = readme.read_text().replace(
        "Fig. 4f HAQER tracks:", "Fig. 4e HAQER tracks:"
    )
    readme.write_text(value)
    crosswalk = root / "hEvo_benchmark/BENCHMARK_METHODS_CROSSWALK.zh.md"
    value = crosswalk.read_text()
    old = "| Fig. 4e genomic tracks |"
    lines = value.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(old):
            lines[i] = (
                "| Fig. 4e genomic tracks | hEvo-large 与 GPN-Star-M 的 12 个验证/测试示例；完整共享测试指标单独记录，不将选择性轨道作为整体优势证据 | 主文区域评分方法；补充方法 S11.3 | `tables/fig4e_haqer_tracks/Fig4e_track_source_data.csv`；HAR/HAQER 分类仍来自 S36–S37 | `figures/scripts/Fig4/e_haqer_gpn_comparison.py`；仓库内源表和注释；不增加 endpoint |"
            )
    value = "\n".join(lines) + "\n"
    value = value.replace("现行 Fig4f", "现行 Fig4e").replace("Fig.4f", "Fig.4e")
    value = value.replace("Fig.4d–e", "Fig.4c–d").replace("Fig.4g", "Fig.4f")
    value = value.replace("Fig.4a–c", "Fig.4a–b（无监督分离见 Fig.S8a）")
    crosswalk.write_text(value)
    tracks_readme = tables / "fig4e_haqer_tracks/README.md"
    value = tracks_readme.read_text()
    if "both hEvo-large and GPN-Star-M tracks" not in value:
        value = (
            "# Current Fig.4e: hEvo-large vs GPN-Star-M source data\n\n"
            "This source bundle contains both hEvo-large and GPN-Star-M tracks used in the current Fig. 4e panel. The combined source table is `Fig4e_track_source_data.csv`.\n\n"
            + "\n".join(value.splitlines()[1:])
            + "\n"
        )
        tracks_readme.write_text(value)
    legends = root / "article/hEvo_附表说明_中文论文版.md"
    value = legends.read_text()
    value = value.replace("S1–S48", "S1–S49")
    value = value.replace(
        "按 29 个可用模型汇总 NT 任务平均 MCC，以及完整预测器参数量、吞吐量、延迟和峰值显存的中位实测值及覆盖数",
        f"按 {len(s43)} 个可用模型汇总 18 项 NT 任务平均 MCC，以及完整预测器参数量、吞吐量、延迟和增量峰值显存的算术平均实测值及覆盖数；Fig.5 e–h 纵轴另用 S49 的 45 单位平均 rank，越低越好",
    )
    value = value.replace(
        "保存 29 个模型在十八项 NT 任务", f"保存 {len(s43)} 个模型在十八项 NT 任务"
    )
    value = value.replace(
        "522 个任务–模型组合", f"{len(s43) * 18} 个任务–模型组合"
    ).replace("522 行；核心字段", f"{len(s43) * 18} 行；核心字段")
    value = value.replace(
        "GPN-Star 的原生 MSA/VEP 评分",
        "GPN-Star 的 raw、未校准原生 MSA/VEP LLR（cLLR 敏感性另报）",
    )
    if "S38/S39 使用直接 selected−other" not in value:
        value += (
            "\nS38/S39 使用直接 selected−other 等位基因评分，无 embedding 分类器、群体 centroid 或交叉验证；汇总方向偏好比例与精确二项区间。\n\n"
            "#### Table S49: 跨任务分数、排名与覆盖率\n\n"
            f"45 个排名单位、{inventory['models']} 个模型；每任务按源表指标排名，并列使用 minimum rank，缺失分数不插补。共同任务敏感性使用 {common} 个单位。S49 不是 48 个主文 endpoint 的清单，不能用于直接验证 14/15 组或 34/48 endpoint 的胜出计数。\n"
        )
    panels = {
        "S33": "Supplementary Fig. S8a",
        "S34": "Fig. 4a；Supplementary Fig. S8b",
        "S35": "Fig. 4b；Supplementary Fig. S8c",
        "S36": "Fig. 4d；Supplementary Fig. S9b",
        "S37": "Fig. 4c；Supplementary Fig. S9a",
        "S38": "Fig. 4f；Supplementary Fig. S9c",
        "S39": "Fig. 4f；Supplementary Fig. S9d、Fig. S10",
    }
    for tid, figure in panels.items():
        data = pd.read_csv(next(tables.glob(tid + "_*.csv")))
        start = value.index("#### Table " + tid + ":")
        next_pos = value.find("#### Table ", start + 1)
        block = value[start : next_pos if next_pos >= 0 else len(value)]
        block = re.sub(
            r"- \*\*对应图件：\*\*[^\n]*", "- **对应图件：** " + figure + "。", block
        )
        block = re.sub(
            r"(\*\*规模和统计单位：\*\* )\d+( 行；)",
            lambda m: m[1] + str(len(data)) + m[2],
            block,
        )
        if tid == "S39":
            block = block.replace(
                "记录 30 个模型 ×30 个候选位点",
                f"记录 {data.model_id.nunique()} 个模型 ×30 个候选位点",
            )
            block = block.replace(
                "该逐位点表包含 hEvo-open-large；hEvo-open-base 未有逐位点行，不能依据 S38 的模型级汇总补造。",
                "该逐位点表包含两个 hEvo-open 模型；DNABERT 不在该逐位点表中。",
            )
        value = (
            value[:start] + block + value[next_pos if next_pos >= 0 else len(value) :]
        )
    legends.write_text(value)
    titles = root / "article/hEvo_Supplementary_Table_Titles_EN.md"
    value = (
        titles.read_text()
        .replace("S1–S48", "S1–S49")
        .replace("through “Supplementary Table 48”", "through “Supplementary Table 49”")
    )
    row = "| **Supplementary Table 49** | S49 | Model–task scores, ranks and coverage for the cross-task comparison |"
    value = value.replace(row + "\n", "")
    value = value.replace(
        "## Supplementary Tables 45–48", "## Supplementary Tables 45–49"
    )
    lines = value.splitlines()
    i = next(i for i, line in enumerate(lines) if "| S48 |" in line)
    lines.insert(i + 1, row)
    value = "\n".join(lines) + "\n"
    titles.write_text(value)
    print("Methods and source documentation synchronized; ablation blocks preserved.")


if __name__ == "__main__":
    main()
