#!/usr/bin/env python3
"""Extract native GPN-MSA selected/other raw log-probabilities for Fig. 4f."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_io import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--msa-path", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    import zarr

    if int(zarr.__version__.split(".", 1)[0]) >= 3:
        raise RuntimeError(f"GPN-MSA requires zarr<3; found {zarr.__version__}")

    import gpn.model  # noqa: F401
    from gpn.data import GenomeMSA
    from gpn.msa.vep import VEPInference

    data = pd.read_csv(args.input, sep="\t", dtype=str)
    required = {
        "variant_id",
        "chrom",
        "pos",
        "ref",
        "alt",
        "selected_allele",
        "other_allele",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"input is missing columns: {sorted(missing)}")
    if len(data) != 30 or data["variant_id"].nunique() != 30:
        raise ValueError("expected exactly 30 unique loci")

    inference = VEPInference(
        str(args.model_dir), GenomeMSA(args.msa_path, in_memory=False), args.window_size
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = inference.model.model.to(device).eval()
    rows: list[dict[str, object]] = []

    for start in range(0, len(data), args.batch_size):
        batch = data.iloc[start : start + args.batch_size].copy()
        # The published multiz100way store uses bare chromosome keys ("1"),
        # while the benchmark table uses UCSC-style names ("chr1").
        msa_chroms = batch["chrom"].str.replace(r"^chr", "", regex=True).tolist()
        inputs = inference.tokenize_function(
            {
                "chrom": msa_chroms,
                "pos": batch["pos"].astype(int).tolist(),
                "ref": batch["ref"].str.upper().tolist(),
                "alt": batch["alt"].str.upper().tolist(),
            }
        )
        tensors = {
            key: torch.as_tensor(value, device=device) for key, value in inputs.items()
        }
        logps: dict[str, torch.Tensor] = {}
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ),
        ):
            for strand in ("fwd", "rev"):
                logits = model(
                    input_ids=tensors[f"input_ids_{strand}"],
                    aux_features=tensors[f"aux_features_{strand}"],
                ).logits
                index = torch.arange(len(batch), device=device)
                site_logits = logits[index, tensors[f"pos_{strand}"]]
                logps[strand] = torch.log_softmax(site_logits.float(), dim=-1)

        for local_index, record in enumerate(batch.itertuples(index=False)):
            selected_is_ref = record.selected_allele.upper() == record.ref.upper()
            other_is_alt = record.other_allele.upper() == record.alt.upper()
            if selected_is_ref != other_is_alt:
                raise ValueError(
                    f"{record.variant_id}: selected/other do not partition REF/ALT"
                )
            selected_key = "ref" if selected_is_ref else "alt"
            other_key = "alt" if selected_is_ref else "ref"
            selected = np.mean(
                [
                    float(
                        logps[strand][
                            local_index,
                            tensors[f"{selected_key}_{strand}"][local_index],
                        ].cpu()
                    )
                    for strand in ("fwd", "rev")
                ]
            )
            other = np.mean(
                [
                    float(
                        logps[strand][
                            local_index, tensors[f"{other_key}_{strand}"][local_index]
                        ].cpu()
                    )
                    for strand in ("fwd", "rev")
                ]
            )
            rows.append(
                {
                    "variant_id": record.variant_id,
                    "selected_logprob": selected,
                    "other_logprob": other,
                    "locus_score": selected - other,
                    "directional_correct": int(selected > other),
                }
            )

    result = pd.DataFrame(rows)
    values = result[["selected_logprob", "other_logprob", "locus_score"]].to_numpy(
        float
    )
    if len(result) != 30 or not np.isfinite(values).all():
        raise RuntimeError("raw-score output failed the 30-row finite-value contract")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, sep="\t", index=False, float_format="%.10g")
    payload = {
        "schema": "fig4f_gpn_msa_raw_logits.v1",
        "model_id": "GPN-MSA",
        "model_dir": str(args.model_dir),
        "msa_path": args.msa_path,
        "window_size": args.window_size,
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "rows": len(result),
        "zarr_version": zarr.__version__,
        "torch_version": torch.__version__,
        "device": str(device),
        "chromosome_mapping": "strip leading chr from benchmark chromosome IDs",
        "score_definition": "mean forward/reverse full-vocabulary log-softmax at masked site",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} rows={len(result)}")


if __name__ == "__main__":
    main()
