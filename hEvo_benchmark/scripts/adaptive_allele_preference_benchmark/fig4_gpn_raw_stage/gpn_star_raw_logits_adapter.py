#!/usr/bin/env python3
"""Score Fig. 4f loci with GPN-Star's full raw masked-position logits.

The ordinary GPN-Star VEP adapter returns only alt-ref LLR.  This adapter
keeps the full vocabulary logits, maps reverse-complement logits back to the
forward orientation, averages forward/reverse logits, and then computes full
vocabulary log-probabilities.  It therefore provides the two raw values
required by Fig. 4f: selected_logprob and other_logprob.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_io import sha256


REQUIRED = {
    "variant_id",
    "chrom",
    "pos",
    "ref",
    "alt",
    "selected_allele",
    "other_allele",
    "input_status",
}


def resolve_store(path: str) -> Path:
    root = Path(path)
    if root.is_dir() and (root / "all.zarr").is_dir():
        return root / "all.zarr"
    if root.is_dir() and root.name.endswith(".zarr"):
        return root
    raise ValueError(f"MSA store is not an extracted Zarr directory: {root}")


def load_msa(stores: list[str], species: list[int], chroms: list[str]):
    from gpn.star.data import GenomeMSA

    if len(stores) != len(species):
        raise ValueError("MSA store/species counts differ")
    result = []
    requested = set(chroms)
    requested.update(chrom[3:] for chrom in chroms if chrom.startswith("chr"))
    for store, n_species in zip(stores, species):
        msa = GenomeMSA(
            str(resolve_store(store)),
            n_species=n_species,
            subset_chroms=requested,
            in_memory=False,
        )
        if not msa.data:
            raise ValueError(f"MSA has no requested chromosomes: {store}")
        result.append(msa)
    return result


def shared_chrom(msa_list, chrom: str) -> str | None:
    candidates = [chrom]
    if chrom.startswith("chr"):
        candidates.append(chrom[3:])
    else:
        candidates.append("chr" + chrom)
    return next((x for x in candidates if all(x in msa.data for msa in msa_list)), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--msa-store", action="append", required=True)
    parser.add_argument("--msa-species", nargs="+", type=int, required=True)
    parser.add_argument("--msa-dataset", default="")
    parser.add_argument("--msa-revision", default="")
    parser.add_argument("--window-bp", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    if args.window_bp <= 0 or args.window_bp % 2:
        raise ValueError("window must be a positive even number")
    if args.model_id == "GPN-Star-V" and sum(args.msa_species) != 100:
        raise ValueError("GPN-Star-V requires 100 MSA species")
    if args.model_id == "GPN-Star-M" and sum(args.msa_species) != 447:
        raise ValueError("GPN-Star-M requires 447 MSA species")
    if args.model_id == "GPN-Star-P" and sum(args.msa_species) != 243:
        raise ValueError("GPN-Star-P requires 243 MSA species")
    if len(args.msa_store) != len(args.msa_species):
        raise ValueError("MSA store/species counts differ")

    # The first store must be the target-bearing alignment for multi-store M.
    pairs = sorted(zip(args.msa_species, args.msa_store), reverse=True)
    args.msa_species = [x[0] for x in pairs]
    args.msa_store = [x[1] for x in pairs]

    frame = pd.read_csv(args.input, sep="\t", dtype=str, keep_default_na=False)
    missing = REQUIRED.difference(frame.columns)
    if missing:
        raise ValueError(f"input missing columns: {sorted(missing)}")
    if len(frame) != 30:
        raise ValueError(f"expected 30 input loci, found {len(frame)}")

    import torch
    from gpn.data import Tokenizer
    from gpn.star.logits import LogitsInference

    msa_list = load_msa(args.msa_store, args.msa_species, sorted(frame.chrom.unique()))
    frame["_msa_chrom"] = frame["chrom"].map(
        lambda value: shared_chrom(msa_list, value)
    )
    if frame["_msa_chrom"].isna().any():
        bad = frame.loc[frame["_msa_chrom"].isna(), "chrom"].unique().tolist()
        raise ValueError(f"MSA chromosome keys not found for: {bad}")
    inference = LogitsInference(str(args.model_dir), msa_list, args.window_bp)
    inference.model.to(args.device).eval()
    actual_model = inference.model.model
    tokenizer = Tokenizer()
    token_ids = {base: tokenizer.vocab.index(base) for base in "ACGT"}
    # vocab order is - A C G T ?, so this maps reverse-complement outputs back
    # to the forward orientation before averaging.
    reverse_map = torch.tensor([0, 4, 3, 2, 1, 5], device=args.device)

    output_rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for start in range(0, len(frame), args.batch_size):
            batch = frame.iloc[start : start + args.batch_size]
            encoded = inference.tokenize_function(
                {
                    "chrom": batch["_msa_chrom"].tolist(),
                    "pos": [int(x) for x in batch.pos.tolist()],
                    "ref": batch.ref.tolist(),
                    "alt": batch.alt.tolist(),
                }
            )
            tensors = {
                key: torch.as_tensor(value, device=args.device)
                for key, value in encoded.items()
            }
            fwd = actual_model(
                input_ids=tensors["input_ids_fwd"],
                source_ids=tensors["source_ids_fwd"],
                target_species=tensors["target_species"],
            ).logits
            rev = actual_model(
                input_ids=tensors["input_ids_rev"],
                source_ids=tensors["source_ids_rev"],
                target_species=tensors["target_species"],
            ).logits
            fwd = fwd[
                torch.arange(len(batch), device=args.device), tensors["pos_fwd"]
            ].squeeze(-2)
            rev = rev[
                torch.arange(len(batch), device=args.device), tensors["pos_rev"]
            ].squeeze(-2)
            rev = rev.index_select(-1, reverse_map)
            # Use full-vocabulary log-softmax, not a reconstructed LLR.
            log_probs = (
                torch.log_softmax((fwd + rev) / 2.0, dim=-1).float().cpu().numpy()
            )
            for row, values in zip(batch.itertuples(index=False), log_probs):
                selected = str(row.selected_allele).upper()
                other = str(row.other_allele).upper()
                selected_score = float(values[token_ids[selected]])
                other_score = float(values[token_ids[other]])
                output_rows.append(
                    {
                        "model_id": args.model_id,
                        "model_revision": args.model_revision,
                        "variant_id": row.variant_id,
                        "chrom": row.chrom,
                        "pos": row.pos,
                        "ref": row.ref,
                        "alt": row.alt,
                        "selected_allele": selected,
                        "other_allele": other,
                        "selected_logprob": selected_score,
                        "other_logprob": other_score,
                        "locus_score": selected_score - other_score,
                        "directional_correct": int(selected_score > other_score),
                        "benchmark_status": "PASS",
                        "score_definition": "GPN-Star full-vocabulary log-softmax selected_logprob - other_logprob",
                    }
                )

    result = pd.DataFrame(output_rows)
    if (
        len(result) != 30
        or not np.isfinite(
            result[["selected_logprob", "other_logprob"]].to_numpy()
        ).all()
    ):
        raise RuntimeError("raw-logit output is incomplete or non-finite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, sep="\t", index=False)
    manifest = {
        "schema": "fig4f_gpn_star_raw_logits.v1",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "output": str(args.output.resolve()),
        "rows": len(result),
        "score_definition": "full vocabulary log-softmax after forward/reverse-complement logit averaging",
        "vocab": tokenizer.vocab,
        "msa_species": args.msa_species,
        "msa_stores": [str(Path(x).resolve()) for x in args.msa_store],
        "batch_size": args.batch_size,
        "device": args.device,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} rows={len(result)}")


if __name__ == "__main__":
    main()
