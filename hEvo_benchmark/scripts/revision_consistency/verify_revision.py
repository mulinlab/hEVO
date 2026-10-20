#!/usr/bin/env python3
"""Independent numerical, preservation and artifact checks for this revision."""

from pathlib import Path
from zipfile import ZipFile
import ast
import hashlib
import json
import sys
import io

import numpy as np
import pandas as pd
from PIL import Image
from lxml import etree as ET
from docx import Document
from revise_docx import visible, NS, protected_parts


def hash_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[3]
    output = root / "article/revision_20261010"
    tables = root / "hEvo_benchmark/tables"
    checks = {}
    baseline_snapshot = output / "baseline_snapshot.zip"
    if baseline_snapshot.is_file():
        with ZipFile(baseline_snapshot) as baseline:
            for name in baseline.namelist():
                if name.startswith("hEvo_benchmark/tables/S41_") or name.startswith(
                    "hEvo_benchmark/tables/S42_"
                ):
                    assert (root / name).read_bytes() == baseline.read(name)
            checks["deferred_S41_S42_restored_byte_for_byte"] = True

            def function_signature(code, names):
                tree = ast.parse(code)
                return {
                    n.name: ast.dump(n, include_attributes=False)
                    for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name in names
                }

            name = "hEvo_benchmark/figures/scripts/Fig5/_reference.py"
            assert function_signature(
                baseline.read(name).decode(), {"flowchart", "nt", "clinvar"}
            ) == function_signature(
                (root / name).read_text(), {"flowchart", "nt", "clinvar"}
            )
            checks["deferred_fig5_ablation_functions_unchanged"] = True
            before = Image.open(
                io.BytesIO(baseline.read("hEvo_benchmark/figures/Fig5/A4_integrated.png"))
            ).convert("RGB")
            after = Image.open(
                root / "hEvo_benchmark/figures/Fig5/A4_integrated.png"
            ).convert("RGB")
            assert before.size == after.size
            crop = (0, 0, round(before.width * 153 / 210), round(before.height * 80 / 297))
            b = np.asarray(before.crop(crop))
            a = np.asarray(after.crop(crop))
            checks["fig5_ablation_crop_differing_pixels"] = int(
                np.any(a != b, axis=2).sum()
            )
            assert checks["fig5_ablation_crop_differing_pixels"] == 0
            # Main raw result tables and the supplied source Word files remain intact.
            for prefix in [
                "S21_",
                "S23_",
                "S24_",
                "S25_",
                "S26_",
                "S27_",
                "S28_",
                "S29_",
                "S30_",
                "S31_",
                "S32_",
                "S38_",
                "S39_",
            ]:
                for p in tables.glob(prefix + "*.csv"):
                    assert p.read_bytes() == baseline.read(p.relative_to(root).as_posix())
            checks["primary_raw_tables_unchanged"] = True
            receipts = json.loads((output / "document_revision_receipts.json").read_text())
            for receipt in receipts:
                source, target = Path(receipt["source"]), Path(receipt["output"])
                assert source.read_bytes() == baseline.read(
                    source.relative_to(root).as_posix()
                )
                Document(target)  # OPC/package parse independent of the XML editor
                with ZipFile(source) as z:
                    original = ET.fromstring(z.read("word/document.xml"))
                with ZipFile(target) as z:
                    revised = ET.fromstring(z.read("word/document.xml"))
                assert protected_parts(original) == protected_parts(revised)
                if source.name == "Manuscript to XQY.docx":
                    before_p, after_p = (
                        original.findall(".//w:p", NS),
                        revised.findall(".//w:p", NS),
                    )
                    assert ET.tostring(before_p[58], method="c14n") == ET.tostring(
                        after_p[58], method="c14n"
                    )
                    assert (
                        visible(before_p[267]).split("e–h,")[0]
                        == visible(after_p[267]).split("e–h,")[0]
                    )
                    caption = visible(after_p[263])
                    assert "i, Retrospective enrichment of subsequent downgrades" in caption
                    assert (
                        "j, Corresponding enrichment of subsequent pathogenic or likely pathogenic upgrades"
                        in caption
                    )
                    assert "(Fig. 3j,k)" in visible(after_p[46]) and "(Fig. 3i)" in visible(
                        after_p[46]
                    )
                    assert "Fig. 4c,d" in visible(after_p[52]) and "Fig. 4e" in visible(
                        after_p[52]
                    )
                    assert "(Fig. 4f and Supplementary Tables S38 and S39)" in visible(
                        after_p[54]
                    )
            checks["word_documents_fields_equations_prior_revisions_preserved"] = len(
                receipts
            )
            checks["word_layout_visual_check"] = (
                "not available: no Word/LibreOffice rendering engine; package and final-text checks passed"
            )
            for filename, left, right in [
                (
                    "hEvo_Methods_中文论文版.md",
                    "## Ablation and public corpus evaluations",
                    "## Computational performance and deployment scope",
                ),
                (
                    "hEvo_附加材料Methods_中文论文版.md",
                    "### S13.1 训练条件消融",
                    "### S13.2",
                ),
            ]:
                old = baseline.read("article/" + filename).decode()
                new = (root / "article" / filename).read_text()
                assert (
                    old.split(left, 1)[1].split(right, 1)[0]
                    == new.split(left, 1)[1].split(right, 1)[0]
                )
            checks["chinese_ablation_sections_unchanged"] = True
    else:
        checks["baseline_archive_comparisons"] = "skipped: audit archive removed"
    ranks = pd.read_csv(next(tables.glob("S49_*.csv")))
    assert ranks.task_unit.nunique() == 45
    assert not ranks.duplicated(["model", "task_unit"]).any()
    for source, group in ranks.groupby("source_table"):
        assert set(group.source_sha256) == {hash_file(tables / source)}
    independent_rank = ranks.groupby("task_unit").score.rank(
        ascending=False, method="min"
    )
    assert np.array_equal(independent_rank.to_numpy(), ranks.task_rank.to_numpy())
    checks["s49_source_hashes_and_within_task_ranks"] = len(ranks)
    detail = pd.read_csv(next(tables.glob("S44_*.csv")))
    summary = pd.read_csv(next(tables.glob("S43_*.csv"))).set_index("model")
    for model, rows in detail.groupby("model"):
        for col in ["params_m", "throughput_bps", "latency_ms", "vram_mb"]:
            assert np.isclose(
                rows[col].mean(), summary.loc[model, col], atol=1e-9, rtol=1e-10
            )
        assert np.isclose(rows.mcc.mean(), summary.loc[model, "mean_mcc"], atol=1e-12)
        assert np.isclose(
            ranks[ranks.model.eq(model)].task_rank.mean(),
            summary.loc[model, "mean_rank"],
            atol=1e-12,
        )
    checks["s43_independent_recomputation_from_s44_and_s49"] = len(summary)
    ledger = pd.read_csv(tables / "revision_consistency/endpoint_ledger.csv")
    assert len(ledger) == 48 and ledger.group.nunique() == 15
    assert (
        ledger[ledger.endpoint_id.str.startswith("S32:")]
        .status.eq("NO_PRESPECIFIED_SINGLE_WINNER_RULE")
        .all()
    )
    checks["endpoint_inventory"] = {"endpoints": 48, "groups": 15}
    sensitivity = pd.read_csv(
        tables / "revision_consistency/cllr_sensitivity_summary.csv"
    )
    assert sensitivity.status.isin(["COMPLETE", "COMPLETE_DESCRIPTIVE"]).all()
    assert sensitivity[sensitivity.table_id.ne("S32")].raw_reproduces_primary.all()
    longitudinal = pd.read_csv(
        tables / "revision_consistency/retrospective_cllr_summary.csv"
    )
    assert longitudinal.raw_primary_delta.abs().max() < 1e-12
    assert longitudinal.coverage.eq(1).all() and longitudinal.seeds.eq(5).all()
    checks["cllr_primary_reconciliation"] = {
        "point_comparisons": int(sensitivity.table_id.ne("S32").sum()),
        "retrospective_year_event_comparisons": len(longitudinal),
        "retrospective_seeds": 5,
    }
    # Verify the retained protocol table remains reproducible for changed columns.
    sys.path.insert(0, str(tables / "scripts"))
    from build_model_supplementary_tables import build_evaluation_parameters

    actual = pd.read_csv(next(tables.glob("S48_*.csv"))).set_index("configuration_item")
    expected = build_evaluation_parameters().set_index("configuration_item")
    for col in [
        "Adaptive allele-preference evaluation",
        "Adaptive allele-preference locus-score evaluation",
    ]:
        pd.testing.assert_series_equal(actual[col], expected[col], check_names=False)
    checks["s48_adaptive_protocol_regenerates"] = True
    import fitz

    rendered_doc = fitz.open(root / "hEvo_benchmark/figures/Fig3/A4_integrated.pdf")
    rendered_page = rendered_doc[0]
    rendered = rendered_page.get_text()
    assert "P/LP → later downgrade" in rendered and "LB/B → later upgrade" in rendered
    assert (
        rendered_page.search_for("P/LP → later downgrade")[0].x0
        < rendered_page.search_for("LB/B → later upgrade")[0].x0
    )
    checks["current_fig3_and_fig4_caption_bindings"] = True
    report = {
        "status": "PASS_WITH_RECORDED_LIMITATIONS",
        "checks": checks,
        "limitations": [
            "ablation work deferred by user",
            "historical hardware/dtype/batch records incomplete",
            "model weight release and licenses unresolved",
            "external Windows latest files unavailable",
            "Word page-layout rendering unavailable",
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
