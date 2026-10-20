import tempfile
import unittest
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from lxml import etree as ET

from revise_docx import revise, visible, NS, W
from run_cllr_sensitivity import paired_top1

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tables" / "scripts"))
from deepsea_run_selection import active_deepsea_rows


class RevisionTests(unittest.TestCase):
    def test_main_deepsea_preserves_both_primary_scales(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "figures/scripts"))
        from Fig2.g_deepsea import select_family_representatives

        models = [
            "hEvo-base",
            "hEvo-large",
            "hEvo-open-large",
            "NT_a",
            "NT_b",
            "NT_c",
            "evo2_a",
            "GPN_a",
            "GPN_b",
            "other_a",
            "other_b",
            "other_c",
            "other_d",
        ]
        data = pd.DataFrame(
            {
                "model": models,
                "display_model": models,
                "recomputed_mean": [0.7, 0.8, 0.99] + [0.6] * 10,
            }
        )
        selected, _ = select_family_representatives(data)
        self.assertEqual(
            set(selected.loc[selected.model.str.startswith("hEvo"), "model"]),
            {"hEvo-base", "hEvo-large"},
        )

    def test_explicit_deepsea_run_not_best_test_score(self):
        rows = pd.DataFrame(
            {
                "model": ["hEvo-open-large"] * 6,
                "task": ["DHS", "HM", "TF"] * 2,
                "run_id": ["S21_run_22"] * 3 + ["S21_run_24"] * 3,
                "auroc": [0.99] * 3 + [0.8] * 3,
            }
        )
        selected = active_deepsea_rows(rows)
        self.assertEqual(set(selected.run_id), {"S21_run_24"})
        self.assertTrue(selected.auroc.eq(0.8).all())
        self.assertEqual(len(rows), 6)
        with self.assertRaises(ValueError):
            active_deepsea_rows(rows.iloc[:3])

    def test_distinct_positive_trios_at_same_coordinates(self):
        rows = []
        for i, positive in enumerate([3.0, 0.0]):
            for suffix, label, score in [
                ("", 1, positive),
                ("__samepos_other_A>C", 0, 1.0),
                ("__samepos_other_A>T", 0, 2.0),
            ]:
                rows.append(
                    dict(
                        variant_id=f"positive_{i}" + suffix,
                        chrom="chr1",
                        pos=5,
                        label=label,
                        score=score,
                    )
                )
        self.assertEqual(paired_top1(pd.DataFrame(rows), "score"), (0.5, 2))

    def test_tracked_copy_preserves_old_revision_equation_and_field(self):
        root = ET.fromstring(
            f'''<w:document xmlns:w="{W}" xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"><w:body><w:p><w:ins w:id="1" w:author="Author"><w:r><w:t>old text</w:t><w:br/></w:r></w:ins><w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText> ADDIN EN.CITE retained </w:instrText></w:r><w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>(1)</w:t></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r><m:oMath><m:r><m:t>x</m:t></m:r></m:oMath></w:p></w:body></w:document>'''
        )
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "before.docx", Path(folder) / "after.docx"
            with ZipFile(source, "w") as z:
                z.writestr("word/document.xml", ET.tostring(root))
                z.writestr("retained.bin", b"unchanged")
            receipt = revise(
                source,
                output,
                [(0, "old text", "new text", "test nested historical insertion")],
            )
            self.assertTrue(receipt["reject_new_restores_original"])
            self.assertEqual(receipt["preserved_prior_revisions"], 1)
            with ZipFile(output) as z:
                revised = ET.fromstring(z.read("word/document.xml"))
                self.assertEqual(visible(revised.find(".//w:p", NS)), "new text(1)")
                self.assertEqual(z.read("retained.bin"), b"unchanged")
            with self.assertRaises(ValueError):
                revise(
                    source,
                    output,
                    [(0, "(1)", "(2)", "must reject a field-result edit")],
                )


if __name__ == "__main__":
    unittest.main()
