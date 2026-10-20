"""Routing checks for the single-sequence job generator."""

import unittest
from argparse import Namespace
from pathlib import Path

from create_probe_single_seq import BENCHMARK_ROOT, resolve_evaluator


class ProbeGeneratorRoutingTests(unittest.TestCase):
    def args(self, evaluate_script=None, protocol=None):
        return Namespace(evaluate_script=evaluate_script, protocol=protocol)

    def test_nt_pareto_defaults_to_pareto_protocol(self):
        script, protocol, protocol_arg = resolve_evaluator(
            self.args(), {"protocol": "auto"}, "NT_Pareto"
        )
        self.assertEqual(
            script,
            BENCHMARK_ROOT / "scripts" / "evaluate" / "probe_single_seq_input.py",
        )
        self.assertEqual(protocol, "pareto")
        self.assertTrue(protocol_arg)

    def test_other_task_family_defaults_to_standard_protocol(self):
        script, protocol, protocol_arg = resolve_evaluator(
            self.args(), {"protocol": "auto"}, "GenomicBenchmark"
        )
        self.assertEqual(
            script,
            BENCHMARK_ROOT / "scripts" / "evaluate" / "probe_single_seq_input.py",
        )
        self.assertEqual(protocol, "standard")
        self.assertTrue(protocol_arg)

    def test_explicit_evaluator_path_wins_and_receives_no_protocol_flag(self):
        custom_script = Path("custom_evaluator.py")
        script, protocol, protocol_arg = resolve_evaluator(
            self.args(evaluate_script=custom_script, protocol="pareto"),
            {"protocol": "standard"},
            "GenomicBenchmark",
        )
        self.assertEqual(script, custom_script)
        self.assertEqual(protocol, "pareto")
        self.assertFalse(protocol_arg)


if __name__ == "__main__":
    unittest.main()
