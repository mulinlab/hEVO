"""Fig5 with the supplied PDF's exact vector layout and local S-table data.

The reference supplies only geometry, typography, marker styles and static labels.
Numerical marks and annotations are updated from S41, S42, S43 and S49, including
both marker layers in each Pareto panel and the complete-coverage frontiers.
"""
from __future__ import annotations

from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd
from pypdf import PdfWriter
from pypdf.generic import FloatObject

from common.reference_pdf import ReferencePage, load_table, marker_center

HERE = Path(__file__).resolve()
REFERENCE = HERE.parent / "assets" / "reference.pdf"
CORPORA = ["PopReduce_60K", "1000G_3K", "UKB_500K"]
WINDOWS = ["NoSW", "SW"]
STAGES = ["Stage I", "Stage II"]

# These IDs identify the reference's existing marks, rather than determine the
# updated ranking. Current overall numbers are recalculated from S49 below.
REFERENCE_ORDER = [
    "hEvo-large", "hEvo-base", "evo2_7b_base", "GPN-Star-M", "GPN-Star-P",
    "GPN-Star-V", "GENA-LM-large", "NT_2.5b_1kG", "SpeciesLM", "NT_2.5b_MS",
    "Genos-1.2B", "DNABERT2", "AIDO.DNA-300M", "DNABERT", "GPN-MSA",
    "NT_500m_MS", "GROVER", "Mistral-DNA-422M", "NTv3-650M-pre",
    "NTv3-100M-pre", "GENA-LM-base", "GENERator-3B", "GENERator-1.2B",
    "GPN-promoter", "evo2_1b_base", "Mistral-DNA-17M", "NT_500m_1kG",
    "NTv3-8M-pre", "Caduceus-PS", "Caduceus-PH", "HyenaDNA_tiny_1k",
    "Mistral-DNA-138M",
]
FRONTIER_COLOR = (.560784, .603922, .682353)
LEADER_COLOR = (.615686, .654902, .705882)


# Page points measured directly from the reference's labeled logarithmic ticks.
PANELS = [
    dict(letter="d", metric="params_m", left=48.18898, bottom=439.3701,
         tick_x=78.40743, tick_value=1., log_step=50.02914, higher=False),
    dict(letter="e", metric="latency_ms", left=340.1575, bottom=439.3701,
         tick_x=359.5370, tick_value=1., log_step=70.45460, higher=False),
    dict(letter="f", metric="throughput_bps", left=48.18898, bottom=204.0945,
         tick_x=115.7321, tick_value=10000., log_step=72.18410, higher=True),
    dict(letter="g", metric="vram_mb", left=340.1575, bottom=204.0945,
         tick_x=442.7487, tick_value=1000., log_step=104.3476, higher=False),
]


def center(path):
    """Account for this reference's triangle geometry when placing markers."""
    if path["vertices"] == 3:
        x0, y0, x1, _ = path["bbox"]
        return (x0 + x1) / 2, y0 + .375 * (x1 - x0)
    return marker_center(path)


def translate_text(page, record, dx=0., dy=0.):
    delta = np.linalg.inv(record["ctm"][:2, :2]) @ np.array([dx, dy])
    operands, operator = page.ops[record["tm_op"]]
    operands = list(operands)
    operands[4] = FloatObject(float(operands[4]) + delta[0])
    operands[5] = FloatObject(float(operands[5]) + delta[1])
    page.ops[record["tm_op"]] = (operands, operator)


def set_line(page, path, start, end):
    inverse = np.linalg.inv(path["ctm"])
    points = iter([start, end])
    for index in path["indices"]:
        _, operator = page.ops[index]
        if operator in (b"m", b"l"):
            point = inverse @ np.array([*next(points), 1.])
            page.ops[index] = ([FloatObject(value) for value in point[:2]], operator)


def nondominated(data, metric, higher):
    ordered = data.sort_values([metric, "mean_rank", "model"],
                               ascending=[not higher, True, True])
    selected = []
    best_rank = np.inf
    for row in ordered.itertuples():
        if row.mean_rank < best_rank:
            selected.append(row.model)
            best_rank = row.mean_rank
    return data.set_index("model").loc[selected].sort_values(metric)


def in_panel(path, panel):
    return (panel["left"] < path["x"] < panel["left"] + 233
            and panel["bottom"] < path["y"] < panel["bottom"] + 176)


def build_reference(tables_dir):
    root = Path(tables_dir)
    page = ReferencePage(REFERENCE)
    for label in page.texts:
        if label["text"] == "hEVO corpus (~64K)":
            page.set_text(label, "hEVO corpus (63,809)", align="left")
        elif label["text"] == "1000G (~3K)":
            page.set_text(label, "1000G (3,202)", align="left")
    s41, s42, s43, s49 = [load_table(root, n) for n in (41, 42, 43, 49)]

    # b: mean MCC across the 18 local NT tasks, paired by windowing scheme.
    if s41.duplicated(["training_dataset", "windowing", "dataset"]).any():
        raise ValueError("S41 has duplicate corpus/window/task records")
    nt = s41.groupby(["training_dataset", "windowing"]).metric_mean.agg(
        mean_mcc="mean", n_tasks="count"
    )
    if not nt.n_tasks.eq(18).all():
        raise ValueError("Fig5b requires 18 NT tasks per corpus/window scheme")
    row_y = [771.0236, 742.6772, 714.3307]
    nt_x = lambda value: 232.4409 + (value - .52) / .09 * 130.3937
    for row, corpus in enumerate(CORPORA):
        values = [float(nt.loc[(corpus, window), "mean_mcc"]) for window in WINDOWS]
        marks = [path for path in page.paths if path["paint"] == b"B*"
                 and 232 < path["x"] < 364 and abs(path["y"] - row_y[row]) < .1]
        if len(marks) != 2:
            raise ValueError("Unexpected Fig5b reference marker layout")
        for mark, value in zip(marks, values):
            dx = nt_x(value) - mark["x"]
            page.move(mark, x=nt_x(value))
            annotation = next(t for t in page.texts if t["text"].startswith("0.")
                              and abs(t["y"] - (row_y[row] - 9.9213)) < .1
                              and abs(t["x"] + 7.007 - mark["x"]) < .1)
            page.set_text(annotation, f"{value:.3f}")
            translate_text(page, annotation, dx=dx)
        line = next(path for path in page.paths if path["paint"] == b"S"
                    and abs(path["y"] - row_y[row]) < .1 and 232 < path["x"] < 364)
        page.line(line, nt_x(values[0]), nt_x(values[1]))

    # c: Stage I vs Stage II AUPRC, with the two window schemes shown separately.
    clinvar = s42.loc[s42.dataset.eq("all") & s42.metric.eq("AUPRC")
                     & s42.evidence_status.eq("PASS")].copy()
    if clinvar.duplicated(["training_dataset", "windowing", "stage"]).any():
        raise ValueError("S42 has duplicate ClinVar ablation conditions")
    indexed = clinvar.set_index(["training_dataset", "windowing", "stage"])
    clinvar_x = lambda value: 464.8819 + (value - .3) / .65 * 107.7165
    clinvar_records = []
    for row, corpus in enumerate(CORPORA):
        for window_index, window in enumerate(WINDOWS):
            y = 773.8583 - row * 28.34646 - window_index * 11.33858
            values = [float(indexed.loc[(corpus, window, stage), "metric_mean"])
                      for stage in STAGES]
            marks = [path for path in page.paths if path["paint"] == b"B*"
                     and 464 < path["x"] < 573 and abs(path["y"] - y) < .1]
            if len(marks) != 2:
                raise ValueError("Unexpected Fig5c reference marker layout")
            for mark, value in zip(marks, values):
                page.move(mark, x=clinvar_x(value))
            line = next(path for path in page.paths if path["paint"] == b"S"
                        and abs(path["y"] - y) < .1 and 464 < path["x"] < 573)
            page.line(line, clinvar_x(values[0]), clinvar_x(values[1]))
            if row == 0 and window_index == 1:
                annotation = next(t for t in page.texts if t["text"] == "0.904"
                                  and t["y"] > 700)
                page.set_text(annotation, f"{values[1]:.3f}", align="left")
                translate_text(page, annotation, dx=clinvar_x(values[1]) - marks[1]["x"])
            for stage, value in zip(STAGES, values):
                clinvar_records.append(dict(corpus=corpus, windowing=window,
                                            stage=stage, auprc=value))

    # d-g: resource means from S43; available-task mean ranks from S49.
    if s49.duplicated(["model", "task_unit"]).any():
        raise ValueError("S49 has duplicate model/task rank records")
    summary = s43.set_index("model").reindex(REFERENCE_ORDER).copy()
    if summary.params_m.isna().any():
        raise ValueError("S43 is missing a reference model")
    ranks = s49.groupby("model").task_rank.agg(mean_rank="mean", n_rank_tasks="count")
    total_tasks = int(s49.task_unit.nunique())
    summary["mean_rank"] = ranks.mean_rank.reindex(REFERENCE_ORDER)
    summary["n_rank_tasks"] = ranks.n_rank_tasks.reindex(REFERENCE_ORDER)
    if not np.isfinite(summary[["mean_rank", "params_m", "latency_ms",
                                "throughput_bps", "vram_mb"]].to_numpy(float)).all():
        raise ValueError("A displayed model has a nonfinite rank/resource value")
    overall = summary.sort_values(["mean_rank"], kind="stable").index.tolist()
    ids = {model: index + 1 for index, model in enumerate(overall)}
    legend_texts = [t for t in page.texts if t["y"] < 170
                    and re.fullmatch(r"\s*\d+\s+\S.*", t["text"])]
    legend_texts.sort(key=lambda t: int(t["text"].split()[0]))
    if len(legend_texts) != 32:
        raise ValueError("Unexpected Fig5 ranking-key layout")
    aliases = {model: re.sub(r"^\s*\d+\s+", "", text["text"])
               for model, text in zip(REFERENCE_ORDER, legend_texts)}
    direct_models = {"hEvo-large", "hEvo-base", "evo2_7b_base", "GPN-Star-M",
                     "GENA-LM-large", "NT_2.5b_1kG", "SpeciesLM", "Genos-1.2B", "GPN-MSA"}
    alias_to_model = {alias: model for model, alias in aliases.items()}
    frontier_records = []
    insertions = []

    for panel in PANELS:
        metric = panel["metric"]
        tx = lambda value: panel["tick_x"] + np.log10(value / panel["tick_value"]) * panel["log_step"]
        ty = lambda rank: panel["bottom"] + (35 - rank) * (175.7480 / 34.5)
        marks = [path for path in page.paths if path["paint"] == b"B*"
                 and in_panel(path, panel)]
        # The PDF paints each point twice, with labels between the two layers.
        if len(marks) != 64:
            raise ValueError(f"Unexpected Fig5{panel['letter']} marker layout")
        original = {model: center(path) for model, path in zip(REFERENCE_ORDER, marks[:32])}
        shifts = {}
        for index, mark in enumerate(marks):
            model = REFERENCE_ORDER[index % 32]
            row = summary.loc[model]
            x, y = tx(float(row[metric])), ty(float(row.mean_rank))
            old_x, old_y = center(mark)
            dx, dy = x - old_x, y - old_y
            page.transform(mark, lambda xx, yy, dx=dx, dy=dy: (xx + dx, yy + dy))
            complete = (int(row["n_" + metric]) == int(row.n_tasks)
                        and int(row.n_perf) == int(row.n_tasks)
                        and int(row.n_rank_tasks) == total_tasks)
            page.color(mark, mark["stroke"] if complete else (1., 1., 1.))
            shifts[model] = (dx, dy)

        annotations = [t for t in page.texts if panel["left"] < t["x"] < panel["left"] + 233
                       and panel["bottom"] < t["y"] < panel["bottom"] + 176]
        if len(annotations) != 32:
            raise ValueError("Unexpected Pareto annotation layout")
        for annotation in annotations:
            label = annotation["text"].strip()
            model = REFERENCE_ORDER[int(label) - 1] if label.isdigit() else alias_to_model[label]
            if model not in direct_models:
                page.set_text(annotation, ids[model], align="left")
            translate_text(page, annotation, *shifts[model])
        leaders = [path for path in page.paths if path["paint"] == b"S"
                   and in_panel(path, panel) and path["width"] == .3]
        for leader in leaders:
            a, _ = page.ops[leader["indices"][0]]
            first = (leader["ctm"] @ np.array([float(a[0]), float(a[1]), 1.]))[:2]
            model = min(REFERENCE_ORDER, key=lambda m: np.linalg.norm(first - original[m]))
            dx, dy = shifts[model]
            page.transform(leader, lambda x, y, dx=dx, dy=dy: (x + dx, y + dy))

        complete = summary.loc[(summary["n_" + metric] == summary.n_tasks)
                               & summary.n_perf.eq(summary.n_tasks)
                               & summary.n_rank_tasks.eq(total_tasks)].reset_index()
        frontier = nondominated(complete, metric, panel["higher"])
        vertices = [(tx(float(row[metric])), ty(float(row.mean_rank)))
                    for _, row in frontier.iterrows()]
        segments = []
        for start, end in zip(vertices[:-1], vertices[1:]):
            corner = (start[0], end[1]) if panel["higher"] else (end[0], start[1])
            segments.extend([(start, corner), (corner, end)])
        old_frontier = [path for path in page.paths if path["paint"] == b"S"
                        and in_panel(path, panel) and path["stroke"] == FRONTIER_COLOR]
        if not old_frontier or not segments:
            raise ValueError("Pareto frontier needs at least two complete models")
        for path in old_frontier[1:]:
            page.remove(path)
        first = old_frontier[0]
        set_line(page, first, *segments[0])
        # Add the remaining segments inside the first path's original graphics
        # state, preserving its exact stroke, dash pattern and behind-point order.
        extra = []
        inverse = np.linalg.inv(first["ctm"])
        for start, end in segments[1:]:
            a = (inverse @ np.array([*start, 1.]))[:2]
            b = (inverse @ np.array([*end, 1.]))[:2]
            extra.extend([([], b"n"), ([FloatObject(v) for v in a], b"m"),
                          ([FloatObject(v) for v in b], b"l"), ([], b"S")])
        insertions.append((first["index"] + 1, extra))
        for model, row in frontier.iterrows():
            frontier_records.append(dict(panel=panel["letter"], metric=metric, model=model,
                                         resource_mean=row[metric], mean_task_rank=row.mean_rank))

    # Keep the measured key slots, replacing row contents with the current order.
    legend_marks = [path for path in page.paths if path["paint"] == b"B*"
                    and path["y"] < 170]
    if len(legend_marks) != 32:
        raise ValueError("Unexpected Fig5 ranking-key marker layout")
    slots = [center(mark) for mark in legend_marks]
    for model, mark in zip(REFERENCE_ORDER, legend_marks):
        old_x, old_y = center(mark)
        x, y = slots[ids[model] - 1]
        page.transform(mark, lambda xx, yy, dx=x-old_x, dy=y-old_y: (xx + dx, yy + dy))
    for slot, annotation in enumerate(legend_texts):
        model = overall[slot]
        page.set_text(annotation, f"{ids[model]:2d}  {aliases[model]}", align="left",
                      bold=model in {"hEvo-large", "hEvo-base"})

    for index, operations in sorted(insertions, reverse=True):
        page.ops[index:index] = operations
    page.source_data = {
        "Fig5b_corpus_window_mean_MCC": nt.reset_index(),
        "Fig5c_ClinVar_stage_AUPRC": pd.DataFrame(clinvar_records),
        "Fig5d_g_resources_and_ranks": summary.assign(
            overall_rank=[ids[model] for model in summary.index],
            display_model=[aliases[model] for model in summary.index]
        ).reset_index(),
        "Fig5d_g_Pareto_frontiers": pd.DataFrame(frontier_records),
        "Fig5_task_ranks": s49.loc[s49.model.isin(REFERENCE_ORDER)].copy(),
    }
    return page


def export(tables_dir, output_dir, export_panels=False):
    out = Path(output_dir)
    page = build_reference(tables_dir)
    src = out / "source_data"
    src.mkdir(parents=True, exist_ok=True)
    for name, frame in page.source_data.items():
        frame.to_csv(src / f"{name}.csv", index=False)
    pdf = page.finish(out)
    if export_panels:
        export_panel_crops(pdf, out)
    return {kind: str(out / f"A4_integrated.{kind}") for kind in ("pdf", "png", "svg")}


def export_panel_crops(pdf, out):
    """Export the same generated vectors as individual panels, without an audit."""
    boxes = {
        "a_ablation_design": (15, 680, 170, 825),
        "b_sequence_representation": (188, 665, 368, 825),
        "c_allelic_refinement": (380, 665, 580, 825),
        "d_parameters": (12, 413, 288, 647),
        "e_latency": (304, 413, 583, 647),
        "f_throughput": (12, 178, 288, 411),
        "g_vram": (304, 178, 583, 411),
    }
    # Historical names used by the existing Fig5 panel export interface.
    aliases = {
        "b_nt_ablation_matrix": "b_sequence_representation",
        "c_nt_factor_delta": "c_allelic_refinement",
        "d_clinvar_auprc_stage_effect": "c_allelic_refinement",
        "f_nt_pareto_params": "d_parameters",
        "g_nt_pareto_latency": "e_latency",
        "e_nt_pareto": "f_throughput",
        "h_nt_pareto_vram": "g_vram",
    }
    all_boxes = {**boxes, **{alias: boxes[name] for alias, name in aliases.items()}}
    for name, box in all_boxes.items():
        cropped = ReferencePage(pdf)
        # The shared family key belongs to the whole page. Keep each panel's
        # letter while removing the partial key from individual top-row crops.
        if box[1] == 413:
            for text in cropped.texts:
                if 620 < text['y'] < 625:
                    cropped.set_text(text, '', align='left')
            for path in cropped.paths:
                if path['paint'] == b'B*' and 619 < path['y'] < 626:
                    cropped.remove(path)
        cropped.stream.operations = cropped.ops
        cropped.page.replace_contents(cropped.stream)
        page = cropped.page
        page.cropbox.lower_left = box[:2]
        page.cropbox.upper_right = box[2:]
        page.mediabox.lower_left = box[:2]
        page.mediabox.upper_right = box[2:]
        writer = PdfWriter()
        writer.add_page(page)
        path = Path(out) / f"{name}.pdf"
        with path.open("wb") as handle:
            writer.write(handle)
        subprocess.run(["pdftoppm", "-f", "1", "-singlefile", "-cropbox", "-png", "-r", "600",
                        str(path), str(Path(out) / name)], check=True, capture_output=True)
        subprocess.run(["pdftocairo", "-f", "1", "-l", "1", "-svg", str(path),
                        str(Path(out) / f"{name}.svg")], check=True, capture_output=True)
