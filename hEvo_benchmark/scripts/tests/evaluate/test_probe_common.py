"""Regression checks for shared probe metrics and dataset plumbing."""

from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader

from core import probe_common


class _LogitProbe(nn.Module):
    def forward(self, values):
        return values


class ProbeCommonRegressionTests(unittest.TestCase):
    def test_protocol_evaluator_preserves_metric_difference(self):
        logits = torch.tensor(
            [
                [4.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 4.0],
                [3.0, 1.0, 0.0],
                [0.0, 3.0, 1.0],
                [1.0, 0.0, 3.0],
            ]
        )
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        loader = DataLoader(
            probe_common.EmbeddingDataset(logits, labels),
            batch_size=2,
            shuffle=False,
            collate_fn=probe_common.collate_emb_fn,
        )

        standard = probe_common.eval_probe_on_embeddings(
            _LogitProbe(),
            loader,
            device=torch.device("cpu"),
            include_per_class_f1=True,
        )
        pareto = probe_common.eval_probe_on_embeddings(
            _LogitProbe(),
            loader,
            device=torch.device("cpu"),
            include_per_class_f1=False,
        )

        self.assertEqual(
            {key for key in standard if key != "per_class_f1"}, set(pareto)
        )
        self.assertEqual(len(standard["per_class_f1"]), 3)
        self.assertEqual(standard["per_class_f1"], [1.0, 1.0, 1.0])
        for metric in pareto:
            if isinstance(pareto[metric], list):
                self.assertEqual(standard[metric], pareto[metric])
            else:
                self.assertAlmostEqual(standard[metric], pareto[metric])

    def test_shared_linear_probe_keeps_historical_shapes(self):
        probe = probe_common.FlexibleProbe(input_dim=5, num_classes=3)
        self.assertIsInstance(probe.probe, nn.Linear)
        self.assertEqual(tuple(probe(torch.zeros(2, 5)).shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
