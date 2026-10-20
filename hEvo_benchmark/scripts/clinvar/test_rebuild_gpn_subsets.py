"""Regression checks for score direction, ties, and ID-based reconstruction."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from rebuild_gpn_subsets import bootstrap, identity, load_scores, sha256
import json


def test_bootstrap_matches_sklearn_with_ties():
    y = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    s = np.array([0.0, 0.0, 2.0, 1.0, 2.0, 3.0, 1.0, 4.0])
    rng = np.random.RandomState(42)
    values = []
    for _ in range(100):
        idx = rng.randint(0, len(y), size=len(y))
        if len(np.unique(y[idx])) == 2:
            values.append(
                [average_precision_score(y[idx], s[idx]), roc_auc_score(y[idx], s[idx])]
            )
    a = np.array(values)
    for actual, expected in zip(
        bootstrap(y, s, 100, 42),
        (a.mean(0), np.percentile(a, 2.5, axis=0), np.percentile(a, 97.5, axis=0)),
    ):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)


def test_identity_rejects_multiple_changes():
    with pytest.raises(ValueError):
        identity(dict(refSeq="AAA", altSeq="TAT", chrom="1", pos=2))


def fixture(tmp_path, rows):
    p = tmp_path / "scores.tsv"
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({"output_sha256": sha256(p)}))
    spec = dict(model="GPN-Star-M", format="tsv", path=str(p), manifest=str(m))
    source = {
        "all_1": ("coding", 1, ("1", 2, "A", "T")),
        "all_2": ("noncoding", 0, ("2", 3, "C", "G")),
    }
    return spec, source


def row(sid, label, chrom, pos, ref, alt, llr):
    return dict(
        model_id="GPN-Star-M",
        task_name="ClinVar260615_all",
        variant_id=sid,
        label=label,
        chrom=chrom,
        pos=pos,
        ref=ref,
        alt=alt,
        raw_llr=llr,
        benchmark_score=-llr,
        score_mode="neg_llr",
        status="PASS",
    )


def test_reordering_preserves_mapping(tmp_path):
    rows = [
        row("all_1", 1, "chr1", 2, "A", "T", -3),
        row("all_2", 0, "2", 3, "C", "G", 1),
    ]
    spec, source = fixture(tmp_path, rows)
    a, _, _ = load_scores(spec, source)
    spec, source = fixture(tmp_path, rows[::-1])
    b, _, _ = load_scores(spec, source)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize(
    "fault", ["duplicate", "label", "unknown", "nonfinite", "direction", "identity"]
)
def test_bad_records_fail(tmp_path, fault):
    rows = [row("all_1", 1, "1", 2, "A", "T", -3), row("all_2", 0, "2", 3, "C", "G", 1)]
    if fault == "duplicate":
        rows[1] = rows[0].copy()
    if fault == "label":
        rows[0]["label"] = 0
    if fault == "unknown":
        rows[0]["variant_id"] = "all_9"
    if fault == "nonfinite":
        rows[0]["raw_llr"] = float("inf")
        rows[0]["benchmark_score"] = -float("inf")
    if fault == "direction":
        rows[0]["benchmark_score"] = -3
    if fault == "identity":
        rows[0]["alt"] = "G"
    spec, source = fixture(tmp_path, rows)
    with pytest.raises((ValueError, KeyError)):
        load_scores(spec, source)


def test_new_masked_hdf5_sequence_validation(tmp_path):
    import h5py
    import hashlib

    p = tmp_path / "masked.hdf5"
    source = {
        "all_1": (
            "coding",
            1,
            ("1", 2, "A", "T"),
            hashlib.sha256(b"AAA|ATA").hexdigest(),
        )
    }
    with h5py.File(p, "w") as f:
        f.attrs.update(
            variant_scoring_method="masked_marginal_changed_tokens_v1",
            score_mode="neg_llr",
            model_id="GPN-MSA",
            dataset_name="ClinVar260615_all",
        )
        f.create_dataset(
            "details",
            data=[
                json.dumps(
                    dict(idx="all_1", ref_char="AAA", alt_char="ATA", label=1)
                ).encode()
            ],
        )
        f.create_dataset("labels", data=[1])
        f.create_dataset("llrs", data=[-2.0])
    spec = dict(
        model="GPN-MSA",
        format="hdf5",
        path=str(p),
        scoring_method="masked_marginal_changed_tokens_v1",
    )
    frame, skipped, _ = load_scores(spec, source)
    assert frame.score.iloc[0] == 2 and skipped.empty
    source["all_1"] = source["all_1"][:3] + ("incorrect_digest",)
    with pytest.raises(ValueError, match="Sequence mismatch"):
        load_scores(spec, source)
