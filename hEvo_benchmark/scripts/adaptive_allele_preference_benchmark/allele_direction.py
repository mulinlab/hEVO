"""Apply the two literature/ancestry-verified GRCh38-plus allele directions.

REF/ALT stays unchanged. Existing inference values are exchanged, not inferred.
This operation is idempotent and applies before aggregation/calibration.
"""

from __future__ import annotations
import pandas as pd

DIRECTION_CORRECTIONS = {"rs642742": ("C", "T"), "rs1871534": ("C", "G")}


def orient_alleles(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for variant, (selected, other) in DIRECTION_CORRECTIONS.items():
        mask = result["variant_id"].eq(variant)
        if not mask.any():
            continue
        valid = result.loc[mask, "selected_allele"].eq(selected) & result.loc[
            mask, "other_allele"
        ].eq(other)
        reverse = result.loc[mask, "selected_allele"].eq(other) & result.loc[
            mask, "other_allele"
        ].eq(selected)
        if not (valid | reverse).all():
            raise ValueError(f"{variant}: unexpected allele pair")
        indices = reverse.index[reverse]
        if not len(indices):
            continue
        result.loc[indices, "selected_allele"] = selected
        result.loc[indices, "other_allele"] = other
        if "selected_logprob" in result and "other_logprob" in result:
            values = result.loc[
                indices, ["other_logprob", "selected_logprob"]
            ].to_numpy()
            result.loc[indices, ["selected_logprob", "other_logprob"]] = values
        for column in (
            "locus_score",
            "raw_benchmark_score",
            "calibrated_benchmark_score",
        ):
            if column in result:
                numeric = pd.to_numeric(result.loc[indices, column], errors="raise")
                result.loc[indices, column] = -numeric
        if "directional_correct" in result and "locus_score" in result:
            result.loc[indices, "directional_correct"] = (
                pd.to_numeric(result.loc[indices, "locus_score"]) > 0
            ).astype(int)
        # raw_llr and calibrated_llr are ALT-minus-REF and retain their signs.
    return result
