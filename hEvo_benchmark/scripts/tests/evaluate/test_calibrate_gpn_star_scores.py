"""Protocol checks for independent GPN-Star calibration post-processing."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from revision_consistency.calibrate_gpn_star_scores import (
    Reference,
    calibrate,
    evaluate,
)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        p = Path(self.temp.name) / "reference.fa"
        p.write_text(">chr1\nAACGT\nAACGT\n")
        Path(str(p) + ".fai").write_text("chr1\t10\t6\t5\t6\n")
        self.reference = Reference(p)
        self.calibration = pd.DataFrame(
            {"pentanuc_mut": ["ACGTA_A"], "llr_neutral_mean": [-1.0]}
        )
        self.rows = pd.DataFrame(
            [
                dict(
                    model_id="model",
                    model_revision="revision",
                    variant_id=str(i),
                    chrom="chr1",
                    pos=4,
                    ref="G",
                    alt="A",
                    raw_llr=2.0,
                    score_mode=mode,
                    label=i % 2,
                    selected_allele="G",
                    other_allele="A",
                    status="PASS",
                )
                for i, mode in enumerate(
                    ["raw_llr", "neg_llr", "abs_llr", "selected_other"]
                )
            ]
        )

    def tearDown(self):
        self.reference.close()
        self.temp.cleanup()

    def test_context_crosses_fasta_line_and_contig_alias(self):
        self.assertEqual(self.reference.context("1", 4), "ACGTA")

    def test_calibration_then_task_orientation_and_input_preservation(self):
        before = self.rows.copy(deep=True)
        out = calibrate(
            self.rows, self.calibration, self.reference, "model", "revision"
        )
        self.assertEqual(
            out.calibrated_benchmark_score.tolist(), [3.0, -3.0, 3.0, -3.0]
        )
        pd.testing.assert_frame_equal(self.rows, before)
        self.assertEqual(evaluate(out)["calibrated_rows"], 4)

    def test_reference_mismatch_and_boundary_excluded_from_both_metrics(self):
        self.rows.loc[0, "ref"] = "C"
        self.rows.loc[1, "pos"] = 1
        out = calibrate(
            self.rows, self.calibration, self.reference, "model", "revision"
        )
        self.assertEqual(out.loc[0, "calibration_status"], "reference_mismatch")
        self.assertEqual(
            out.loc[1, "calibration_status"], "pentanucleotide_crosses_contig_boundary"
        )
        self.assertTrue(np.isnan(out.loc[0, "calibrated_llr"]))
        self.assertEqual(evaluate(out)["num_samples"], 2)

    def test_duplicate_variants_and_wrong_checkpoint_rejected(self):
        with self.assertRaises(ValueError):
            calibrate(
                self.rows, self.calibration, self.reference, "wrong_model", "revision"
            )
        self.rows.loc[1, "variant_id"] = "0"
        with self.assertRaises(ValueError):
            calibrate(self.rows, self.calibration, self.reference, "model", "revision")

    def test_missing_and_duplicate_lookup_keys(self):
        out = calibrate(
            self.rows, self.calibration.iloc[:0], self.reference, "model", "revision"
        )
        self.assertTrue(
            out.calibration_status.eq("calibration_context_unavailable").all()
        )
        with self.assertRaises(ValueError):
            calibrate(
                self.rows,
                pd.concat([self.calibration] * 2),
                self.reference,
                "model",
                "revision",
            )


if __name__ == "__main__":
    unittest.main()
