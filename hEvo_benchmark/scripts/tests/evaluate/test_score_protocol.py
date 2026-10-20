import unittest

from core.score_protocol import metric_mode_for_model


class TaskOrientationTest(unittest.TestCase):
    @staticmethod
    def transform(value, mode):
        if mode == "neg_llr":
            return -value
        if mode == "abs_llr":
            return abs(value)
        return value

    def test_generator_converts_canonical_modes_to_cached_ref_minus_alt(self):
        for mode in ("neg_llr", "raw_llr", "abs_llr"):
            for llr in (-2.0, 0.0, 3.0):
                cached = -llr
                effective = metric_mode_for_model("GENERator-1.2B", mode)
                self.assertEqual(
                    self.transform(cached, effective), self.transform(llr, mode)
                )

    def test_effect_distance_cannot_be_treated_as_llr(self):
        self.assertEqual(metric_mode_for_model("Mistral-DNA-17M", "raw_llr"), "raw_llr")
        for mode in ("neg_llr", "abs_llr"):
            with self.assertRaises(ValueError):
                metric_mode_for_model("Mistral-DNA-17M", mode)

    def test_mlm_orientation_is_unchanged(self):
        for mode in ("neg_llr", "raw_llr", "abs_llr"):
            self.assertEqual(metric_mode_for_model("hEvo-large", mode), mode)


if __name__ == "__main__":
    unittest.main()
