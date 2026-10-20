#!/usr/bin/env python3
"""Replace GPN-Star blank raw-score columns in the Fig. 4f S39 table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from allele_direction import orient_alleles


MODEL_IDS = ["GPN-Star-V", "GPN-Star-M", "GPN-Star-P"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s39", type=Path, required=True)
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    table = orient_alleles(pd.read_csv(args.s39, dtype=str, keep_default_na=False))
    for model_id in MODEL_IDS:
        score_path = args.scores_dir / f"{model_id}.tsv"
        scores = pd.read_csv(score_path, sep="\t", dtype=str)
        if len(scores) != 30 or scores["variant_id"].nunique() != 30:
            raise ValueError(f"{model_id}: expected 30 unique score rows")
        mask = table["model_id"].eq(model_id)
        if int(mask.sum()) != 30:
            raise ValueError(f"{model_id}: S39 has {int(mask.sum())} rows")
        scores = orient_alleles(scores).set_index("variant_id")
        target = table.loc[mask].copy()
        target = target.set_index("variant_id")
        if set(target.index) != set(scores.index):
            raise ValueError(f"{model_id}: S39 and raw scores have different loci")
        for column in ["selected_logprob", "other_logprob", "locus_score"]:
            target[column] = (
                scores[column]
                .reindex(target.index)
                .astype(float)
                .map(lambda x: f"{x:.10g}")
            )
        target["directional_correct"] = (
            scores["directional_correct"].reindex(target.index).astype(int).astype(str)
        )
        target["score_definition"] = (
            "GPN-Star full-vocabulary log-softmax selected_logprob - other_logprob"
        )
        target["benchmark_status"] = "PASS"
        update_columns = [column for column in target.columns if column != "variant_id"]
        ordered_target = target.reindex(table.loc[mask, "variant_id"])
        table.loc[mask, update_columns] = ordered_target[update_columns].to_numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"wrote {args.output} rows={len(table)}")


if __name__ == "__main__":
    main()
