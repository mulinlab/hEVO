#!/usr/bin/env python3
"""Reorient two retained allele pairs and rebuild dependent public tables.
No inference is run; immutable source snapshots make rebuilding idempotent.
"""

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from xml.dom import minidom
import argparse
import hashlib
import json
import shutil
import sys
import io
import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest, wilcoxon
from sklearn.metrics import calinski_harabasz_score
from allele_direction import orient_alleles, DIRECTION_CORRECTIONS

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "hEvo_benchmark"
TABLES = BENCH / "tables"
OUT = BENCH / "reports/adaptive_allele_direction_20261014"
BACKUP = OUT / "before_update"
sys.path.insert(0, str(BENCH / "figures/scripts"))
RELEASE = Path("/mnt/hpc/home/xuqinyi/adaptive_allele_preference_release_20260928")
WORKBOOK = TABLES / "Supplementary Tables.xlsx"
S38 = (
    TABLES / "S38_model_level_preferences_for_literature_supported_adaptive_alleles.csv"
)
S39 = TABLES / "S39_locus_level_scores_for_literature_supported_adaptive_alleles.csv"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def backup(p):
    destination = BACKUP / p.relative_to(ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(p, destination)
    return destination


def ci(k, n):
    return (
        0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)),
        1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k)),
    )


def wtest(v):
    if np.allclose(v, 0):
        return np.nan, 1.0
    z = wilcoxon(v, alternative="greater", zero_method="wilcox", method="auto")
    return float(z.statistic), float(z.pvalue)


def metric(v):
    n = len(v)
    k = int((v > 0).sum())
    lo, hi = ci(k, n)
    ws, wp = wtest(v)
    q1, q3 = np.quantile(v, [0.25, 0.75])
    return dict(
        n_loci=n,
        positive_loci=k,
        accuracy=k / n,
        accuracy_ci_low=lo,
        accuracy_ci_high=hi,
        one_sided_exact_binom_p=float(
            binomtest(k, n, 0.5, alternative="greater").pvalue
        ),
        median_score=float(np.median(v)),
        q1_score=float(q1),
        q3_score=float(q3),
        iqr_score=float(q3 - q1),
        wilcoxon_statistic=ws,
        wilcoxon_one_sided_p=wp,
    )


def reader(path):
    text = path.read_text(encoding="utf-8")
    header = (
        "\n".join(locus for locus in text.splitlines() if locus.startswith("##")) + "\n"
    )
    return header, pd.read_csv(
        path, sep="\t", comment="#", dtype=str, keep_default_na=False
    )


def reader_write(path, header, frame):
    path.write_text(header + frame.to_csv(index=False, sep="\t"), encoding="utf-8")


def elements(doc, name):
    return [x for x in doc.getElementsByTagName("*") if x.localName == name]


def main():
    global RELEASE, OUT, BACKUP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=RELEASE,
        help="Retained 2026-09-28 release snapshot",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT,
        help="Repair evidence and before-update backups; this command also writes public tables",
    )
    args = parser.parse_args()
    RELEASE = args.release_dir.resolve()
    OUT = args.output_dir.resolve()
    BACKUP = OUT / "before_update"
    OUT.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(backup(S39))
    oldsummary = pd.read_csv(backup(S38))
    scores = orient_alleles(original)
    family_by_model = oldsummary.set_index("model_id")["family"]
    scores["family"] = scores["family"].fillna(scores["model_id"].map(family_by_model))
    r14 = TABLES / "repository_tables/R14_adaptive_allele_scores_at_individual_loci.tsv"
    _, oldreader = reader(backup(r14))
    reader_names = {}
    for model, group in original.groupby("model_id", sort=False):
        candidates = []
        for label, reader_group in oldreader.groupby("Model", sort=False):
            aligned = reader_group.set_index("Variant").reindex(group.variant_id)
            if aligned["Selected-minus-alternative score"].isna().any():
                continue
            if np.allclose(
                pd.to_numeric(aligned["Selected-minus-alternative score"]).to_numpy(),
                group.locus_score.to_numpy(),
                atol=1e-9,
                rtol=0,
            ):
                candidates.append(label)
        assert len(candidates) == 1, (model, candidates)
        reader_names[model] = candidates[0]
    reader_names["DNABERT"] = "DNABERT"

    def display(model):
        return reader_names[model]

    assert (
        len(scores) == 990 and scores.model_id.nunique() == 33 and len(oldsummary) == 34
    )
    assert not scores.duplicated(["model_id", "variant_id"]).any()
    assert scores.groupby("model_id").size().eq(30).all()
    assert np.isfinite(
        scores[["selected_logprob", "other_logprob", "locus_score"]].to_numpy()
    ).all()
    target = scores.variant_id.isin(DIRECTION_CORRECTIONS)
    assert target.sum() == 66
    assert scores.loc[target, "selected_allele"].eq("C").all()
    assert original.loc[~target].equals(scores.loc[~target])
    assert np.array_equal(
        scores.loc[target, "locus_score"].to_numpy(),
        -original.loc[target, "locus_score"].to_numpy(),
    )
    assert np.array_equal(
        scores.loc[target, "selected_logprob"].to_numpy(),
        original.loc[target, "other_logprob"].to_numpy(),
    )
    assert orient_alleles(scores).equals(scores)
    release = pd.read_csv(RELEASE / "locus_scores.tsv", sep="\t")
    aliases = {"Generator-1.2B": "GENERator-1.2B", "Generator-3B": "GENERator-3B"}
    release.model_id = release.model_id.replace(aliases)
    joined = original.merge(
        release[["model_id", "variant_id", "locus_score"]],
        on=["model_id", "variant_id"],
        suffixes=("_current", "_release"),
        validate="one_to_one",
    )
    assert len(joined) == 990 and np.allclose(
        joined.locus_score_current, joined.locus_score_release, atol=1e-9, rtol=0
    )
    order = (
        pd.read_csv(RELEASE / "model_summary.tsv", sep="\t")
        .model_id.replace(aliases)
        .tolist()
    )
    assert len(set(order)) == 33 and set(order) == set(scores.model_id)
    summary = oldsummary.copy().set_index("model_id")
    rng = np.random.default_rng(42)
    boots = []
    changes = []
    ch = []
    baseline = []
    for model in order:
        old = original.loc[original.model_id.eq(model)].sort_values("variant_id")
        new = scores.loc[scores.model_id.eq(model)].sort_values("variant_id")
        v = new.locus_score.to_numpy()
        oldv = old.locus_score.to_numpy()
        m = metric(v)
        prior = metric(oldv)
        assert int(summary.loc[model, "positive_loci"]) == prior["positive_loci"]
        indices = rng.integers(0, 30, size=(10000, 30))
        if model == "GPN-MSA":
            indices = np.random.default_rng(42).integers(0, 30, size=(10000, 30))
        bm = np.median(v[indices], axis=1)
        oldbm = np.median(oldv[indices], axis=1)
        ba = np.mean(v[indices] > 0, axis=1)
        m["median_score_bootstrap_ci_low"], m["median_score_bootstrap_ci_high"] = (
            np.quantile(bm, [0.025, 0.975])
        )
        for key, value in m.items():
            summary.loc[model, key] = value
        baseline.append(
            {
                "model_id": model,
                "old_median_bootstrap_ci_low_reproduced": bool(
                    np.isclose(
                        np.quantile(oldbm, 0.025),
                        oldsummary.set_index("model_id").loc[
                            model, "median_score_bootstrap_ci_low"
                        ],
                        atol=1e-7,
                        rtol=0,
                    )
                ),
                "old_median_bootstrap_ci_high_reproduced": bool(
                    np.isclose(
                        np.quantile(oldbm, 0.975),
                        oldsummary.set_index("model_id").loc[
                            model, "median_score_bootstrap_ci_high"
                        ],
                        atol=1e-7,
                        rtol=0,
                    )
                ),
            }
        )
        for name, values, observed in [
            ("median_score", bm, m["median_score"]),
            ("accuracy", ba, m["accuracy"]),
        ]:
            low, high = np.quantile(values, [0.025, 0.975])
            boots.append(
                dict(
                    model_id=model,
                    metric=name,
                    observed=observed,
                    ci_low=low,
                    ci_high=high,
                    reps=10000,
                    seed=42,
                    unit="locus",
                )
            )
        changes.append(
            {
                "model_id": model,
                "display_model": display(model),
                "before_positive_loci": prior["positive_loci"],
                "after_positive_loci": m["positive_loci"],
                "before_accuracy": prior["accuracy"],
                "after_accuracy": m["accuracy"],
                "delta_accuracy": m["accuracy"] - prior["accuracy"],
                "after_ci_low": m["accuracy_ci_low"],
                "after_ci_high": m["accuracy_ci_high"],
                "after_binomial_p": m["one_sided_exact_binom_p"],
                "before_median_score": prior["median_score"],
                "after_median_score": m["median_score"],
            }
        )
        x = new.selected_logprob.to_numpy()
        y = new.other_logprob.to_numpy()
        ch_value = float(
            calinski_harabasz_score(
                np.r_[x, y].reshape(-1, 1), np.r_[np.ones(30), np.zeros(30)]
            )
        )
        grand = np.r_[x, y].mean()
        between = 30 * ((x.mean() - grand) ** 2 + (y.mean() - grand) ** 2)
        within = ((x - x.mean()) ** 2).sum() + ((y - y.mean()) ** 2).sum()
        manual = between / (within / 58)
        assert np.isclose(manual, ch_value, rtol=1e-9, atol=1e-12)
        ch.append(
            {
                "model_id": model,
                "Model": display(model),
                "Locus pairs": 30,
                "Calinski–Harabasz index": ch_value,
            }
        )
    passmask = summary.status.eq("PASS")
    summary.loc[passmask, "rank_by_accuracy"] = summary.loc[passmask, "accuracy"].rank(
        method="min", ascending=False
    )
    summary = summary.reset_index()
    scores.to_csv(S39, index=False, float_format="%.10g")
    summary.to_csv(S38, index=False, float_format="%.10g")
    synchronize_adaptation_radar()
    pd.DataFrame(changes).to_csv(OUT / "model_changes.csv", index=False)
    pd.DataFrame(boots).to_csv(
        OUT / "bootstrap_summary.tsv", sep="\t", index=False, float_format="%.10g"
    )
    (OUT / "bootstrap_baseline_reproduction.json").write_text(
        json.dumps(baseline, indent=2) + "\n"
    )
    scores.to_csv(OUT / "locus_scores.tsv", sep="\t", index=False, float_format="%.10g")
    # Complete leave-one-locus-out and paired signed-rank summaries.
    loo = []
    paired = []
    by = {
        model: scores.loc[scores.model_id.eq(model)].sort_values("variant_id")
        for model in order
    }
    for model in order:
        frame = by[model]
        v = frame.locus_score.to_numpy()
        for i, variant in enumerate(frame.variant_id):
            keep = np.delete(v, i)
            loo.append(
                dict(
                    model_id=model,
                    omitted_locus=variant,
                    n_loci=29,
                    positive_loci=int((keep > 0).sum()),
                    accuracy=float((keep > 0).mean()),
                    median_score=float(np.median(keep)),
                )
            )
    import itertools

    for a, b in itertools.combinations(order, 2):
        delta = by[a].locus_score.to_numpy() - by[b].locus_score.to_numpy()
        stat, p = wtest(delta)
        paired.append(
            dict(
                model_a=a,
                model_b=b,
                n_loci=30,
                median_delta_a_minus_b=float(np.median(delta)),
                wilcoxon_statistic=stat,
                wilcoxon_one_sided_p_a_gt_b=p,
            )
        )
    pd.DataFrame(loo).to_csv(
        OUT / "leave_one_locus_out.tsv", sep="\t", index=False, float_format="%.10g"
    )
    pd.DataFrame(paired).to_csv(
        OUT / "paired_model_tests.tsv", sep="\t", index=False, float_format="%.10g"
    )
    # Mirror the reader-facing locus and raw-score separation tables.
    r14 = TABLES / "repository_tables/R14_adaptive_allele_scores_at_individual_loci.tsv"
    header, frame = reader(backup(r14))
    lookup = {
        (display(row.model_id), row.variant_id): row for row in scores.itertuples()
    }
    for i, row in frame.iterrows():
        if row["Variant"] not in DIRECTION_CORRECTIONS:
            continue
        source = lookup[(row["Model"], row["Variant"])]
        for col, value in [
            ("Selected allele", source.selected_allele),
            ("Alternative allele", source.other_allele),
            ("Selected allele log probability", source.selected_logprob),
            ("Alternative allele log probability", source.other_logprob),
            ("Selected-minus-alternative score", source.locus_score),
            ("Selected allele preferred", int(source.locus_score > 0)),
        ]:
            frame.at[i, col] = str(value) if isinstance(value, str) else f"{value:.10g}"
    reader_write(
        r14,
        header.replace(
            "No ancestral allele assignment is inferred.",
            "Two corrected allele directions are supported by literature and explicit ancestral/strand audit.",
        ),
        frame,
    )
    r15 = (
        TABLES
        / "repository_tables/R15_direction_independent_adaptive_allele_score_separation.tsv"
    )
    header, frame = reader(backup(r15))
    lookupch = {x["Model"]: x["Calinski–Harabasz index"] for x in ch}
    for i, row in frame.iterrows():
        frame.at[i, "Calinski–Harabasz index"] = str(lookupch[row.Model])
    reader_write(r15, header, frame)
    # Reorient calibrated selected-minus-other values without changing ALT-minus-REF cLLR.
    calibration = []
    sensitivity = TABLES / "revision_consistency/cllr_sensitivity_summary.csv"
    manifestpath = TABLES / "revision_consistency/sensitivity_manifest.json"
    csum = pd.read_csv(backup(sensitivity))
    records = json.loads(backup(manifestpath).read_text())
    r27 = (
        TABLES
        / "repository_tables/R27_gpn_star_calibration_across_variant_and_adaptive_allele_endpoints.tsv"
    )
    rheader, rframe = reader(backup(r27))
    for model in ["GPN-Star-V", "GPN-Star-M", "GPN-Star-P"]:
        source = ROOT / (
            "article/revision_20261010/cllr_sensitivity/scores/"
            + model
            + "/S38.cllr.tsv.gz"
        )
        original_cal = pd.read_csv(source, sep="\t")
        corrected = orient_alleles(original_cal)
        eligible = corrected.calibration_status.eq("PASS")
        n = int(eligible.sum())
        assert n == 30
        original_primary = (
            original.loc[original.model_id.eq(model)]
            .set_index("variant_id")
            .locus_score
        )
        assert np.allclose(
            original_cal.set_index("variant_id").locus_score.reindex(
                original_primary.index
            ),
            original_primary,
            atol=1e-6,
            rtol=0,
        )
        raw_k = int(corrected.loc[eligible, "raw_benchmark_score"].gt(0).sum())
        cal_k = int(corrected.loc[eligible, "calibrated_benchmark_score"].gt(0).sum())
        lo, hi = ci(cal_k, n)
        assert raw_k == int(summary.set_index("model_id").loc[model, "positive_loci"])
        assert np.array_equal(
            corrected.raw_llr, original_cal.raw_llr
        ) and np.array_equal(corrected.calibrated_llr, original_cal.calibrated_llr)
        path = OUT / "calibrated_scores" / (model + ".tsv.gz")
        path.parent.mkdir(exist_ok=True)
        corrected.to_csv(path, sep="\t", index=False)
        values = dict(
            raw_positive_loci=raw_k,
            cllr_positive_loci=cal_k,
            raw_preference_fraction=raw_k / n,
            cllr_preference_fraction=cal_k / n,
            cllr_preference_ci_low=lo,
            cllr_preference_ci_high=hi,
        )
        mask = csum.model.eq(model) & csum.table_id.eq("S38/S39")
        assert mask.sum() == 1
        for col, value in values.items():
            csum.loc[mask, col] = value
        csum.loc[mask, "scores_sha256"] = sha(path)
        record = next(
            x
            for x in records
            if x.get("model") == model and x.get("table_id") == "S38/S39"
        )
        record.update(values)
        record.update(
            scores_sha256=sha(path),
            corrected_scores=str(path),
            orientation_corrections=DIRECTION_CORRECTIONS,
            original_calibrated_scores=str(source),
            original_calibrated_scores_sha256=sha(source),
        )
        rm = rframe.Model.eq(display(model)) & rframe[
            "Original endpoint identifier"
        ].eq("S38/S39")
        assert rm.sum() == 1
        for col, value in [
            ("Raw selected-allele preference", raw_k / n),
            ("Calibrated selected-allele preference", cal_k / n),
            ("Calibrated preference lower bound", lo),
            ("Calibrated preference upper bound", hi),
        ]:
            rframe.loc[rm, col] = f"{value:.10g}"
        calibration.append(
            dict(
                model_id=model,
                raw_positive_loci=raw_k,
                calibrated_positive_loci=cal_k,
                n_loci=n,
            )
        )
    csum.to_csv(sensitivity, index=False)
    manifestpath.write_text(json.dumps(records, indent=2) + "\n")
    reader_write(r27, rheader, rframe)
    # Correct canonical locus annotations used by future scoring and maps.
    dataset = (
        BENCH / "datasets/Adaptation_Variants/Adaptive_Allele_Preference_Benchmark"
    )
    for name, sep in [
        ("human_selection_snv_master_curated_20260905.csv", ","),
        ("benchmark_loci_expanded.tsv", "\t"),
    ]:
        p = dataset / name
        frame = pd.read_csv(
            backup(p), sep=sep, dtype=str, keep_default_na=False, encoding="utf-8-sig"
        )
        for variant, (selected, other) in DIRECTION_CORRECTIONS.items():
            mask = frame.variant_id.eq(variant)
            assert mask.sum() == 1
            frame.loc[mask, "selected_allele"] = selected
            if "other_allele" in frame:
                frame.loc[mask, "other_allele"] = other
            pmids = (
                "29961894;18083106" if variant == "rs642742" else "24586184;25927708"
            )
            for col in ["source_pmid", "selection_pmid"]:
                if col in frame:
                    frame.loc[mask, col] = pmids
            if "primary_source_url" in frame:
                frame.loc[mask, "primary_source_url"] = (
                    "https://pubmed.ncbi.nlm.nih.gov/" + pmids.split(";")[0] + "/"
                )
            note = (
                "GRCh38 plus: ancestral "
                + other
                + "; selected C. Literature/ancestral-sequence/strand audit corrected the former ancestral focal label on 2026-10-14."
            )
            for col in ["source_note", "curation_note"]:
                if col in frame:
                    frame.loc[mask, col] = note
        frame.to_csv(
            p, sep=sep, index=False, encoding="utf-8-sig" if sep == "," else "utf-8"
        )
    # Patch only S23 and S22 in the custom 29-sheet workbook.
    _original_workbook = backup(WORKBOOK)
    workbook_input = WORKBOOK.read_bytes()
    with ZipFile(io.BytesIO(workbook_input)) as archive:
        wb = minidom.parseString(archive.read("xl/workbook.xml"))
        rels = minidom.parseString(archive.read("xl/_rels/workbook.xml.rels"))
        parts = {}
        for name in ["S22", "S23"]:
            rid = next(
                s.getAttributeNS(
                    "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
                    "id",
                )
                for s in elements(wb, "sheet")
                if s.getAttribute("name") == name
            )
            target = next(
                x.getAttribute("Target")
                for x in elements(rels, "Relationship")
                if x.getAttribute("Id") == rid
            )
            parts[name] = (
                target.lstrip("/") if target.startswith("/") else "xl/" + target
            )
        updated = {}
        model_lookup = {display(row.model_id): row for row in summary.itertuples()}
        import openpyxl

        xlsx = openpyxl.load_workbook(io.BytesIO(workbook_input))
        for name, part in parts.items():
            doc = minidom.parseString(archive.read(part))
            cells = {x.getAttribute("r"): x for x in elements(doc, "c")}

            def set_cell(address, value):
                cell = cells[address]
                for node in list(cell.childNodes):
                    cell.removeChild(node)
                prefix = (cell.prefix + ":") if cell.prefix else ""
                if isinstance(value, (int, float)) and np.isfinite(value):
                    if cell.hasAttribute("t"):
                        cell.removeAttribute("t")
                    node = doc.createElementNS(cell.namespaceURI, prefix + "v")
                    node.appendChild(doc.createTextNode(f"{value:.10g}"))
                    cell.appendChild(node)
                else:
                    cell.setAttribute("t", "inlineStr")
                    node = doc.createElementNS(cell.namespaceURI, prefix + "is")
                    t = doc.createElementNS(cell.namespaceURI, prefix + "t")
                    t.appendChild(doc.createTextNode(str(value)))
                    node.appendChild(t)
                    cell.appendChild(node)

            if name == "S23":
                for index, row in enumerate(list(xlsx[name].values)[2:], 3):
                    model = model_lookup[row[0]]
                    if model.status != "PASS":
                        continue
                    for col, value in zip(
                        "BCDEFG",
                        [
                            int(model.n_loci),
                            int(model.positive_loci),
                            model.accuracy,
                            model.accuracy_ci_low,
                            model.accuracy_ci_high,
                            model.one_sided_exact_binom_p,
                        ],
                    ):
                        set_cell(col + str(index), value)
            else:
                for index, row in enumerate(list(xlsx[name].values)[2:], 3):
                    if row[0] in DIRECTION_CORRECTIONS:
                        set_cell(
                            "L" + str(index),
                            str(row[11])
                            .replace(
                                "historical direction summaries require reorientation.",
                                "all 33 scored-model directional summaries were reoriented on 2026-10-14.",
                            )
                            .replace(
                                "historical summaries require reorientation.",
                                "all 33 scored-model summaries were reoriented on 2026-10-14.",
                            ),
                        )
            updated[part] = doc.toxml(encoding="UTF-8")
        temp = WORKBOOK.with_suffix(".updated.xlsx")
        with ZipFile(temp, "w", compression=ZIP_DEFLATED) as result:
            for info in archive.infolist():
                result.writestr(
                    info, updated.get(info.filename, archive.read(info.filename))
                )
    temp.replace(WORKBOOK)
    # Receipt and checks rely on independent combinatorial binomial probabilities.
    from math import comb

    final = pd.read_csv(S38).set_index("model_id")
    finalscores = pd.read_csv(S39)
    for model in order:
        k = int(
            finalscores.loc[finalscores.model_id.eq(model), "locus_score"].gt(0).sum()
        )
        p = sum(comb(30, j) for j in range(k, 31)) / 2**30
        assert k == int(final.loc[model, "positive_loci"]) and np.isclose(
            p, final.loc[model, "one_sided_exact_binom_p"], atol=1e-9, rtol=0
        )
    with ZipFile(io.BytesIO(workbook_input)) as old, ZipFile(WORKBOOK) as new:
        assert new.testzip() is None
        for name in old.namelist():
            if name not in updated:
                assert old.read(name) == new.read(name), name
    current = openpyxl.load_workbook(WORKBOOK)
    for row in list(current["S23"].values)[2:]:
        model = model_lookup[row[0]]
        if model.status == "PASS":
            assert row[2] == int(model.positive_loci) and np.isclose(
                row[3], model.accuracy, atol=1e-9, rtol=0
            )
    receipt = dict(
        date="2026-10-14",
        corrections=DIRECTION_CORRECTIONS,
        scored_models=33,
        blocked_models=["DNABERT"],
        locus_rows=990,
        corrected_locus_rows=66,
        model_inference_rerun=False,
        bootstrap_replicates=10000,
        bootstrap_seed=42,
        bootstrap_unit="locus",
        bootstrap_order="preserved 20260928 release model order; GPN-MSA retains its documented per-model seed-42 override",
        accuracy_ci="two-sided 95% Clopper-Pearson",
        binomial_test="one-sided greater, null p=0.5",
        ties="score=0 is not directional success",
        baseline_bootstrap_bounds_reproduced=sum(
            x["old_median_bootstrap_ci_low_reproduced"]
            and x["old_median_bootstrap_ci_high_reproduced"]
            for x in baseline
        ),
        calibration=calibration,
        source_release={
            "path": str(RELEASE / "locus_scores.tsv"),
            "sha256": sha(RELEASE / "locus_scores.tsv"),
            "all_990_retained_scores_matched": True,
        },
        outputs=[
            {"path": str(p.relative_to(ROOT)), "sha256": sha(p)}
            for p in [S38, S39, r14, r15, r27, sensitivity, manifestpath, WORKBOOK]
        ],
    )
    (OUT / "recompute_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))
    print(
        pd.DataFrame(changes)
        .loc[lambda x: x.model_id.str.startswith("hEvo")]
        .to_string(index=False)
    )


def synchronize_adaptation_radar():
    """Propagate corrected S38 preferences to the radar and public mirror."""
    import csv
    sys.path.insert(0, str(TABLES / "scripts"))
    from build_fig1_family_radar import build

    source = TABLES / "S10_best_model_family_ranks_across_functional_variant_and_evolutionary_tasks.csv"
    backup(source)
    updated = build(TABLES)
    updated = updated.loc[updated.axis.eq("Adaptation")].set_index("family", drop=False)
    lines = source.read_text(encoding="utf-8").splitlines()
    records = list(csv.DictReader(lines))
    output = [lines[0]]
    for line, row in zip(lines[1:], records):
        if row["axis"] == "Adaptation":
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=list(row), lineterminator="")
            writer.writerow(updated.loc[row["family"]].to_dict())
            line = buffer.getvalue()
        output.append(line)
    source.write_text("\n".join(output) + "\n", encoding="utf-8")
    public = TABLES / "repository_tables/R08_family_level_benchmark_summaries.tsv"
    backup(public)
    lines = public.read_text(encoding="utf-8").splitlines()
    header_index = next(i for i, line in enumerate(lines) if not line.startswith("##"))
    fields = lines[header_index].split("\t")
    for index in range(header_index + 1, len(lines)):
        row = dict(zip(fields, lines[index].split("\t")))
        if row["Radar axis"] != "Adaptation (30)":
            continue
        values = updated.loc[row["Family"]]
        row.update({"Primary summary": str(values.metric),
                    "Within-axis rank": str(int(values["rank"])),
                    "Eligible models": str(values.n_eligible),
                    "Radar radius": str(values.display_radius)})
        lines[index] = "\t".join(row[field] for field in fields)
    public.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
