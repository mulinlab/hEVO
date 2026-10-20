"""Parser and dispatch regressions for the unified probe entrypoint."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import random
import sys
import types
import unittest

import numpy as np


EVALUATOR = (
    Path(__file__).resolve().parents[2] / "evaluate" / "probe_single_seq_input.py"
)


def load_cli_functions():
    tree = ast.parse(EVALUATOR.read_text(encoding="utf-8"), filename=str(EVALUATOR))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"build_parser", "main"}
    ]
    namespace = {
        "argparse": argparse,
        "os": __import__("os"),
        "sys": sys,
        "random": random,
        "np": np,
    }

    class CudaStub:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def manual_seed_all(seed):
            return None

    class TorchStub:
        cuda = CudaStub()

        @staticmethod
        def manual_seed(seed):
            return None

    namespace["torch"] = TorchStub()
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(EVALUATOR), "exec"),
        namespace,
    )
    return namespace


def base_args(*extra):
    return [
        "--model_paths",
        "/model",
        "--model_name",
        "DNABERT2",
        "--dataset_files",
        "train.txt,test.txt",
        "--dataset_name",
        "task",
        *extra,
    ]


class UnifiedProbeCliTests(unittest.TestCase):
    def setUp(self):
        self.cli = load_cli_functions()

    def test_protocol_stage_defaults_and_shared_column_options(self):
        parser = self.cli["build_parser"]()
        args = parser.parse_args(base_args())
        self.assertEqual(args.protocol, "standard")
        self.assertEqual(args.stage, "full")

        args = parser.parse_args(
            base_args(
                "--protocol",
                "standard",
                "--stage",
                "precompute",
                "--shard_dir",
                "shards",
                "--seq_col",
                "4",
                "--label_col",
                "5",
                "--shard_index",
                "1",
                "--shard_count",
                "2",
            )
        )
        self.assertEqual((args.seq_col, args.label_col), (4, 5))
        self.assertFalse(
            any(
                action.dest in {"shard_seq_col", "shard_label_col"}
                for action in parser._actions
            )
        )

    def test_sharded_stage_dispatch_forwards_columns_and_stage_args(self):
        parser = self.cli["build_parser"]()
        args = parser.parse_args(
            base_args(
                "--stage",
                "precompute",
                "--shard_dir",
                "shards",
                "--seq_col",
                "4",
                "--label_col",
                "5",
                "--shard_index",
                "1",
                "--shard_count",
                "2",
            )
        )
        calls = []
        fake_engine = types.SimpleNamespace(
            precompute=lambda value: calls.append(("precompute", value)) or "ok",
            merge_probe=lambda value: calls.append(("merge_probe", value)) or "ok",
        )
        old_core = sys.modules.get("core")
        core_stub = types.ModuleType("core")
        core_stub.probe_shard_engine = fake_engine
        sys.modules["core"] = core_stub
        try:
            self.cli["build_parser"] = lambda: argparse.Namespace(
                parse_args=lambda ignored: args
            )
            self.assertEqual(self.cli["main"]([]), "ok")
        finally:
            if old_core is None:
                sys.modules.pop("core", None)
            else:
                sys.modules["core"] = old_core
        mode, forwarded = calls[0]
        self.assertEqual(mode, "precompute")
        self.assertEqual((forwarded.seq_col, forwarded.label_col), (4, 5))
        self.assertEqual((forwarded.shard_index, forwarded.shard_count), (1, 2))

    def test_pareto_protocol_routes_to_legacy_engine_with_protocol_removed(self):
        args = self.cli["build_parser"]().parse_args(
            base_args("--protocol", "pareto", "--stage", "full")
        )
        calls = []
        fake_engine = types.SimpleNamespace(
            main=lambda values: calls.append(values) or "pareto"
        )
        old_core = sys.modules.get("core")
        core_stub = types.ModuleType("core")
        core_stub.probe_pareto_engine = fake_engine
        sys.modules["core"] = core_stub
        try:
            self.cli["build_parser"] = lambda: argparse.Namespace(
                parse_args=lambda ignored: args
            )
            self.assertEqual(
                self.cli["main"](base_args("--protocol", "pareto", "--stage", "full")),
                "pareto",
            )
        finally:
            if old_core is None:
                sys.modules.pop("core", None)
            else:
                sys.modules["core"] = old_core
        self.assertNotIn("--protocol", calls[0])
        self.assertNotIn("--stage", calls[0])
        self.assertIn("--model_paths", calls[0])

    def test_coordinate_msa_models_are_rejected_for_pareto_and_sharded_stages(self):
        route_cases = (
            (
                ("--protocol", "pareto", "--stage", "full"),
                "Pareto protocol does not support coordinate-aware",
            ),
            (
                (
                    "--protocol",
                    "standard",
                    "--stage",
                    "precompute",
                    "--shard_dir",
                    "shards",
                ),
                "Sharded probe stages do not support coordinate-aware",
            ),
        )
        for model_name in ("GPN-MSA", "GPN-Star-V"):
            for extra, expected in route_cases:
                with self.subTest(model_name=model_name, extra=extra):
                    values = base_args("--model_name", model_name, *extra)
                    with self.assertRaisesRegex(ValueError, expected):
                        self.cli["main"](values)


if __name__ == "__main__":
    unittest.main()
