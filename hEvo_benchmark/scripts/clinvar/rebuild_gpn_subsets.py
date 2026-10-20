#!/usr/bin/env python3
"""Validate and reconstruct ClinVar subsets from explicit per-model all score files."""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from export_clinvar_all_subsets import atomic_write_json, atomic_write_text, sha256

BENCHMARK = Path(__file__).resolve().parents[2]
TABLE_FILES = {
    s: f"S{n}_zero_shot_classification_of_{s}_clinvar_variants.csv"
    for s, n in [("coding", 24), ("noncoding", 25)]
}


def identity(row):
    ref, alt = row["refSeq"], row["altSeq"]
    center = len(ref) // 2
    if (
        len(ref) == len(alt)
        and ref[:center] == alt[:center]
        and ref[center + 1 :] == alt[center + 1 :]
        and ref[center] != alt[center]
    ):
        differences = [center]
    else:
        differences = [i for i, (r, a) in enumerate(zip(ref, alt)) if r != a]
    if len(ref) != len(alt) or len(differences) != 1:
        raise ValueError("Expected one SNV per paired sequence")
    i = differences[0]
    return (str(row["chrom"]).removeprefix("chr"), int(row["pos"]), ref[i], alt[i])


def mapping(data):
    members = {}
    counts = {}
    sources = {}
    for subset in TABLE_FILES:
        path = data / f"clinvar_{subset}_1023bp.csv"
        sources[subset] = sha256(path)
        frame = pd.read_csv(path)
        counts[subset] = len(frame)
        for row in frame.to_dict("records"):
            key = identity(row)
            if key in members:
                raise ValueError(f"Duplicate or overlapping variant: {key}")
            members[key] = (subset, int(row["label"]))
    if counts != {"coding": 144285, "noncoding": 119907}:
        raise ValueError(f"Unexpected input counts: {counts}")
    path = data / "clinvar_all_1023bp.csv"
    sources["all"] = sha256(path)
    result = {}
    seen = set()
    for row in pd.read_csv(path).to_dict("records"):
        key = identity(row)
        subset, label = members[key]
        sid = row["SeqID"]
        if sid in result or key in seen:
            raise ValueError("Duplicate all ID/identity")
        if label != int(row["label"]):
            raise ValueError("Source label mismatch")
        # Verify the historical numbering convention against variant identity.
        if (int(sid.split("_")[1]) <= counts["coding"]) != (subset == "coding"):
            raise ValueError("all_N boundary disagrees with source subset identity")
        result[sid] = (
            subset,
            label,
            key,
            hashlib.sha256((row["refSeq"] + "|" + row["altSeq"]).encode()).hexdigest(),
        )
        seen.add(key)
    if seen != set(members):
        raise ValueError("Source subsets do not partition all")
    return result, counts, sources


def decode(value):
    return json.loads(value.decode() if isinstance(value, bytes) else str(value))


def load_scores(spec, source):
    path = Path(spec["path"])
    provenance = {"path": str(path), "sha256": sha256(path), "format": spec["format"]}
    rows = []
    skipped = []
    if spec["format"] == "hdf5":
        with h5py.File(path, "r") as f:
            provenance["attributes"] = {
                k: v.item() if isinstance(v, np.generic) else v
                for k, v in f.attrs.items()
            }
            if spec.get("scoring_method") == "masked_marginal_changed_tokens_v1":
                if (
                    f.attrs.get("variant_scoring_method") != spec["scoring_method"]
                    or f.attrs.get("score_mode") != "neg_llr"
                    or f.attrs.get("model_id") != spec["model"]
                    or f.attrs.get("dataset_name") != "ClinVar260615_all"
                ):
                    raise ValueError("Unexpected masked scoring protocol/model")
            elif f.attrs.get("scoring_method") != "gpn_msa_official_bidirectional_llr":
                raise ValueError("Unexpected MSA scoring protocol")
            if not len(f["details"]) == len(f["llrs"]) == len(f["labels"]):
                raise ValueError("HDF5 alignment mismatch")
            for value, llr, label in zip(f["details"][:], f["llrs"][:], f["labels"][:]):
                d = decode(value)
                rows.append((d["idx"], int(label), float(llr), d))
            skipped = (
                [decode(v) for v in f["skipped_details"][:]]
                if "skipped_details" in f
                else []
            )
    else:
        manifest = Path(spec["manifest"])
        meta = json.loads(manifest.read_text())
        if meta["output_sha256"] != provenance["sha256"]:
            raise ValueError("Manifest checksum mismatch")
        provenance.update(
            manifest=str(manifest),
            manifest_sha256=sha256(manifest),
            manifest_contents=meta,
        )
        with path.open() as handle:
            for d in csv.DictReader(handle, delimiter="\t"):
                if (
                    d["model_id"] != spec["model"]
                    or d["task_name"] != "ClinVar260615_all"
                ):
                    raise ValueError("Wrong model/task")
                d["idx"] = d["variant_id"]
                if d["status"] != "PASS":
                    skipped.append(d)
                    continue
                llr = float(d["raw_llr"])
                if d["score_mode"] != "neg_llr" or not np.isclose(
                    float(d["benchmark_score"]), -llr, rtol=0, atol=1e-12
                ):
                    raise ValueError("Score direction mismatch")
                rows.append((d["idx"], int(d["label"]), llr, d))
    ids = set()
    output = []
    skipout = []
    for sid, label, llr, d in rows:
        if sid in ids:
            raise ValueError(f"Duplicate scored ID {sid}")
        ids.add(sid)
        subset, expected, key = source[sid][:3]
        if label != expected or int(d["label"]) != expected:
            raise ValueError(f"Label mismatch {sid}")
        if not np.isfinite(llr):
            raise ValueError(f"Nonfinite score {sid}")
        if "chrom" not in d:
            digest = hashlib.sha256(
                (d["ref_char"] + "|" + d["alt_char"]).encode()
            ).hexdigest()
            if digest != source[sid][3]:
                raise ValueError(f"Sequence mismatch {sid}")
            actual = key
        else:
            actual = (
                str(d["chrom"]).removeprefix("chr"),
                int(d["pos"]),
                d.get("ref_char", d.get("ref")),
                d.get("alt_char", d.get("alt")),
            )
        if actual != key:
            raise ValueError(f"Variant identity mismatch {sid}: {actual} != {key}")
        output.append(
            dict(
                SeqID=sid,
                subset=subset,
                label=label,
                raw_llr=llr,
                score=-llr,
                chrom=key[0],
                pos=key[1],
                ref=key[2],
                alt=key[3],
            )
        )
    skipids = set()
    for d in skipped:
        sid = d["idx"]
        if sid in ids or sid in skipids:
            raise ValueError(f"Duplicate/overlapping skipped ID {sid}")
        subset, label, key = source[sid][:3]
        skipids.add(sid)
        if int(d["label"]) != label:
            raise ValueError("Skipped label mismatch")
        skipout.append(
            dict(
                SeqID=sid,
                subset=subset,
                label=label,
                reason=d.get("reason", d.get("failure_reason", d.get("status"))),
            )
        )
    if ids | skipids != set(source):
        raise ValueError("Scored + skipped IDs do not exhaust all source records")
    frame = (
        pd.DataFrame(output)
        .sort_values("SeqID", key=lambda x: x.str.split("_").str[1].astype(int))
        .reset_index(drop=True)
    )
    return (
        frame,
        pd.DataFrame(skipout, columns=["SeqID", "subset", "label", "reason"]),
        provenance,
    )


def bootstrap(y, s, repeats=1000, seed=42):
    """Exact row bootstrap with fixed score tie groups; avoids repeated sorting."""
    levels, group = np.unique(-s, return_inverse=True)
    rng = np.random.RandomState(seed)
    values = []
    for _ in range(repeats):
        indices = rng.randint(0, len(y), size=len(y))
        yi = y[indices]
        p = np.bincount(group[indices], weights=yi, minlength=len(levels))
        n = np.bincount(group[indices], weights=1 - yi, minlength=len(levels))
        if not p.sum() or not n.sum():
            continue
        cp = np.cumsum(p)
        cn = np.cumsum(n)
        ap = (
            np.sum(
                np.divide(cp, cp + cn, out=np.zeros_like(cp), where=(cp + cn) > 0) * p
            )
            / p.sum()
        )
        auc = np.sum(p * (n.sum() - cn + 0.5 * n)) / (p.sum() * n.sum())
        values.append((ap, auc))
    if not values:
        raise ValueError("No valid bootstrap resamples")
    a = np.array(values)
    return a.mean(axis=0), np.percentile(a, 2.5, axis=0), np.percentile(a, 97.5, axis=0)


def metrics(frame, model, subset, repeats, seed):
    y = frame.label.to_numpy()
    s = frame.score.to_numpy()
    mean, lo, hi = bootstrap(y, s, repeats, seed)
    result = dict(
        model=model,
        dataset=subset,
        num_samples=len(y),
        num_positive=int(y.sum()),
        num_negative=int(len(y) - y.sum()),
        baseline=float(y.mean()),
        bootstrap_n=repeats,
        bootstrap_seed=seed,
    )
    for i, (name, fn) in enumerate(
        [("auprc", average_precision_score), ("auroc", roc_auc_score)]
    ):
        result.update(
            {
                name + "_mean": float(fn(y, s)),
                name + "_bootstrap_mean": float(mean[i]),
                name + "_ci_lower": float(lo[i]),
                name + "_ci_upper": float(hi[i]),
            }
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--data-dir", type=Path, default=BENCHMARK / "datasets/newClinvar_260615"
    )
    p.add_argument("--tables-dir", type=Path, default=BENCHMARK / "tables")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    if args.bootstrap <= 0 or (args.apply and args.validate_only):
        p.error("positive bootstrap required; validate-only cannot apply")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source, counts, sourcehash = mapping(args.data_dir)
    s23 = pd.read_csv(
        args.tables_dir
        / "S23_zero_shot_classification_of_pathogenic_and_benign_clinvar_variants.csv"
    ).set_index("model")
    audit = {
        "source_counts": counts,
        "source_sha256": sourcehash,
        "models": {},
        "bootstrap_method": "row_iid_with_replacement_percentile",
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_n": args.bootstrap,
    }
    records = []
    for spec in json.loads(args.input_manifest.read_text())["models"]:
        model = spec["model"]
        frame, skipped, provenance = load_scores(spec, source)
        y = frame.label.to_numpy()
        s = frame.score.to_numpy()
        actual = dict(
            num_samples=len(y),
            num_positive=int(y.sum()),
            num_negative=int(len(y) - y.sum()),
            auprc_mean=float(average_precision_score(y, s)),
            auroc_mean=float(roc_auc_score(y, s)),
        )
        old = {k: float(s23.loc[model, k]) for k in actual}
        delta = {k: actual[k] - old[k] for k in actual}
        gate = all(abs(delta[k]) < 1e-10 for k in actual)
        reconciliation = None
        if spec.get("all_summary"):
            summary_path = Path(spec["all_summary"])
            summary = json.loads(summary_path.read_text())
            if summary.get("source_sha256") != sourcehash["all"]:
                raise ValueError("New summary source checksum differs from all CSV")
            summary_values = {
                k: summary[k.replace("_mean", "")]
                if k.endswith("_mean")
                else summary[k]
                for k in actual
            }
            if not all(abs(actual[k] - summary_values[k]) < 1e-10 for k in actual):
                raise ValueError("New all summary does not reproduce raw scores")
            if spec.get("scoring_method") == "masked_marginal_changed_tokens_v1":
                gate = True
                reconciliation = {
                    "explanation": "User selected the 2026-09-28 masked-marginal rerun. S23 retains the older official-MSA result; scoring protocol and coverage differ. New raw all scores reproduce the new run summary. S23 is outside this subset update.",
                    "summary_path": str(summary_path),
                    "summary_sha256": sha256(summary_path),
                    "summary_values": summary_values,
                }
        audit["models"][model] = dict(
            source=provenance,
            all_recomputed=actual,
            s23=old,
            difference=delta,
            formal_table_gate="PASS" if gate else "BLOCKED_HISTORICAL_ALL_MISMATCH",
            scored_by_subset=frame.groupby("subset").size().to_dict(),
            skipped_by_subset=skipped.groupby("subset").size().to_dict(),
            skip_reasons=skipped.groupby(["subset", "reason"])
            .size()
            .reset_index(name="count")
            .to_dict("records"),
        )
        if reconciliation:
            audit["models"][model]["reconciliation"] = reconciliation
        atomic_write_json(args.output_dir / "audit.json", audit)
        print(model, actual, "gate", gate, flush=True)
        if args.validate_only:
            continue
        frame.to_csv(
            args.output_dir / f"{model}_subset_scores.tsv.gz",
            sep="\t",
            index=False,
            compression="gzip",
        )
        skipped.to_csv(args.output_dir / f"{model}_skipped.tsv", sep="\t", index=False)
        for subset in TABLE_FILES:
            sub = frame.loc[frame.subset == subset]
            record = metrics(sub, model, subset, args.bootstrap, args.bootstrap_seed)
            record.update(
                input_samples=counts[subset],
                coverage=len(sub) / counts[subset],
                formal_table_gate=audit["models"][model]["formal_table_gate"],
            )
            records.append(record)
        pd.DataFrame(records).to_csv(args.output_dir / "metrics.csv", index=False)
    atomic_write_json(args.output_dir / "audit.json", audit)
    if args.apply:
        changes = []
        for subset, filename in TABLE_FILES.items():
            path = args.tables_dir / filename
            table = pd.read_csv(path)
            backup = args.output_dir / (filename + ".before")
            history = pd.read_csv(backup) if backup.exists() else table.copy()
            for record in records:
                if record["dataset"] != subset or record["formal_table_gate"] != "PASS":
                    continue
                oldrows = history.loc[history.model == record["model"]].to_dict(
                    "records"
                )
                row = {
                    k: record.get(
                        k, f"ClinVar ({subset})" if k == "plot_title" else None
                    )
                    for k in table.columns
                }
                table = table.loc[table.model != record["model"]]
                table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
                changes.append(
                    dict(
                        table=filename, model=record["model"], before=oldrows, after=row
                    )
                )
            # Retain exact original text for unmodified historical records.
            original_text = path.read_text()
            if not backup.exists():
                atomic_write_text(backup, original_text)
            changed = {
                r["model"]
                for r in records
                if r["dataset"] == subset and r["formal_table_gate"] == "PASS"
            }
            lines = original_text.splitlines(keepends=True)
            kept = [lines[0]] + [
                line for line in lines[1:] if next(csv.reader([line]))[2] not in changed
            ]
            additions = table.loc[table.model.isin(changed)].to_csv(
                index=False, header=False
            )
            atomic_write_text(path, "".join(kept) + additions)
        atomic_write_json(
            args.output_dir / "table_changes.json",
            {
                "changes": changes,
                "blocked_models": [
                    m
                    for m, v in audit["models"].items()
                    if v["formal_table_gate"] != "PASS"
                ],
            },
        )


if __name__ == "__main__":
    main()
