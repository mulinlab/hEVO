import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluate"))

try:
    from adapters.extract_embeddings_classes import (
        ExtractEmbeddingForAIDO,
        ExtractEmbeddingForCaduceus,
        ExtractEmbeddingForDNABert,
        ExtractEmbeddingForEnformer,
        ExtractEmbeddingForEvo2_1B,
        ExtractEmbeddingForEvo2_7B,
        ExtractEmbeddingForGENALM,
        ExtractEmbeddingForGENERator,
        ExtractEmbeddingForGenos_1B,
        ExtractEmbeddingForGPN,
        ExtractEmbeddingForGROVER,
        ExtractEmbeddingForHyena,
        ExtractEmbeddingForMistralDNA,
        ExtractEmbeddingForNT,
        ExtractEmbeddingForNTv3,
        ExtractEmbeddingForSpeciesLM,
        ExtractEmbeddingForhEvo,
        ExtractEmbeddingForhEvo_rand,
        ExtractEmbeddingForhEvo_rare,
    )
except ImportError:
    print("Error: cannot import extract_embeddings_classes.py")
    sys.exit(1)


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


MODEL_MAP = {
    "DNABERT2": ExtractEmbeddingForDNABert,
    "DNABERT": ExtractEmbeddingForDNABert,
    "hEvo-1kG-86M": ExtractEmbeddingForhEvo,
    "hEvo-1kG-sliding128-86M": ExtractEmbeddingForhEvo_rare,
    "hEvo-open-base": ExtractEmbeddingForhEvo_rare,
    "hEvo-open-large": ExtractEmbeddingForhEvo_rare,
    "hEvo-base": ExtractEmbeddingForhEvo_rare,
    "hEvo-large": ExtractEmbeddingForhEvo_rare,
    "hEvo-fullUKB-86M": ExtractEmbeddingForhEvo_rand,
    "hEvo-60k-86M": ExtractEmbeddingForhEvo_rare,
    "evo2_7b_base": ExtractEmbeddingForEvo2_7B,
    "evo2_1b_base": ExtractEmbeddingForEvo2_1B,
    "NT_2.5b_1kG": ExtractEmbeddingForNT,
    "NT_500m_1kG": ExtractEmbeddingForNT,
    "NT_500m_MS": ExtractEmbeddingForNT,
    "NT_2.5b_MS": ExtractEmbeddingForNT,
    "HyenaDNA_tiny_1k": ExtractEmbeddingForHyena,
    "GPN-MSA": ExtractEmbeddingForGPN,
    "GPN-promoter": ExtractEmbeddingForGPN,
    "SpeciesLM": ExtractEmbeddingForSpeciesLM,
    "Genos-1.2B": ExtractEmbeddingForGenos_1B,
    "AIDO.DNA-300M": ExtractEmbeddingForAIDO,
    "GENA-LM-base": ExtractEmbeddingForGENALM,
    "GENA-LM-large": ExtractEmbeddingForGENALM,
    "GROVER": ExtractEmbeddingForGROVER,
    "Mistral-DNA-17M": ExtractEmbeddingForMistralDNA,
    "Mistral-DNA-138M": ExtractEmbeddingForMistralDNA,
    "Mistral-DNA-422M": ExtractEmbeddingForMistralDNA,
    "NTv3-8M-pre": ExtractEmbeddingForNTv3,
    "NTv3-100M-pre": ExtractEmbeddingForNTv3,
    "NTv3-650M-pre": ExtractEmbeddingForNTv3,
    "GENERator-1.2B": ExtractEmbeddingForGENERator,
    "GENERator-3B": ExtractEmbeddingForGENERator,
    "Enformer": ExtractEmbeddingForEnformer,
    "Caduceus-PH": ExtractEmbeddingForCaduceus,
    "Caduceus-PS": ExtractEmbeddingForCaduceus,
}


def aggregate_feature_matrix(final_matrix, aggregation, topk):
    norms = np.linalg.norm(final_matrix, axis=1)
    valid_indices = np.where(np.isfinite(norms) & (norms > 0))[0]

    if len(valid_indices) == 0:
        if aggregation == "multi_stats":
            return np.zeros(final_matrix.shape[1] * 3 + 4, dtype=np.float32)
        return np.zeros(final_matrix.shape[1], dtype=np.float32)

    actual_k = min(topk, len(valid_indices))
    top_k_idx = valid_indices[np.argsort(norms[valid_indices])[-actual_k:]]
    topk_mean = np.mean(final_matrix[top_k_idx], axis=0).astype(np.float32)

    if aggregation == "topk_mean":
        return topk_mean

    valid_matrix = final_matrix[valid_indices]
    valid_norms = norms[valid_indices]
    global_mean = np.mean(valid_matrix, axis=0).astype(np.float32)
    if float(np.sum(valid_norms)) > 0:
        weighted_mean = np.average(valid_matrix, axis=0, weights=valid_norms).astype(
            np.float32
        )
    else:
        weighted_mean = global_mean

    scalar_stats = np.array(
        [
            len(valid_indices) / max(len(final_matrix), 1),
            float(np.mean(valid_norms)),
            float(np.std(valid_norms)),
            float(np.max(valid_norms)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([topk_mean, global_mean, weighted_mean, scalar_stats], axis=0)


def get_window_mean(emb_seq, center_idx, radius, max_len):
    start = max(0, center_idx - radius)
    end = min(max_len, center_idx + radius + 1)
    window = emb_seq[start:end, :]
    return np.mean(window, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-sample variant delta embeddings."
    )
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=255)
    parser.add_argument(
        "--aggregation",
        type=str,
        default="topk_mean",
        choices=["topk_mean", "multi_stats"],
    )
    parser.add_argument("--topk", type=int, default=500)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    sample_id = os.path.basename(args.input_file).replace(".jsonl", "")
    output_path = os.path.join(args.output_dir, f"{sample_id}.npy")

    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.exists(output_path) and not args.recompute:
        return

    extractor_class = MODEL_MAP[args.model_name]
    extractor = extractor_class(model_path=args.model_path, model_name=args.model_name)

    alt_texts = []
    ref_texts = []
    with open(args.input_file, "r") as handle:
        for line in handle:
            data = json.loads(line)
            seq_alt = data["seq"]
            ref_base = data["ref_base"]
            mid_idx = len(seq_alt) // 2
            seq_ref = seq_alt[:mid_idx] + ref_base + seq_alt[mid_idx + 1 :]
            alt_texts.append(seq_alt)
            ref_texts.append(seq_ref)

    if not alt_texts:
        return

    all_deltas = []
    window_radius = 2

    for i in tqdm(
        range(0, len(alt_texts), args.batch_size), desc=f"processing {sample_id}"
    ):
        b_alt = alt_texts[i : i + args.batch_size]
        b_ref = ref_texts[i : i + args.batch_size]

        with torch.no_grad():
            enc_alt = extractor.tokenize(
                b_alt, max_length=args.max_length, padding=True, truncation=True
            )
            enc_ref = extractor.tokenize(
                b_ref, max_length=args.max_length, padding=True, truncation=True
            )

            ids_alt = enc_alt["input_ids"]
            ids_ref = enc_ref["input_ids"]
            emb_alt = extractor.get_embedding(
                ids_alt, pooling="none", return_numpy=True
            ).astype(np.float32)
            emb_ref = extractor.get_embedding(
                ids_ref, pooling="none", return_numpy=True
            ).astype(np.float32)

            emb_alt = np.nan_to_num(emb_alt, nan=0.0, posinf=1e4, neginf=-1e4)
            emb_ref = np.nan_to_num(emb_ref, nan=0.0, posinf=1e4, neginf=-1e4)

            batch_res = []
            for b in range(len(b_alt)):
                curr_alt_ids = ids_alt[b]
                curr_ref_ids = ids_ref[b]
                min_len = min(len(curr_alt_ids), len(curr_ref_ids))
                diff_idx = None
                for j in range(min_len):
                    if curr_alt_ids[j] != curr_ref_ids[j]:
                        diff_idx = j
                        break
                if diff_idx is None:
                    diff_idx = min_len // 2

                vec_alt = get_window_mean(
                    emb_alt[b], diff_idx, window_radius, ids_alt.shape[1]
                )
                vec_ref = get_window_mean(
                    emb_ref[b], diff_idx, window_radius, ids_ref.shape[1]
                )
                batch_res.append(vec_alt - vec_ref)

            all_deltas.append(np.stack(batch_res))

    final_matrix = np.concatenate(all_deltas, axis=0)
    final_vec = aggregate_feature_matrix(
        final_matrix=final_matrix,
        aggregation=args.aggregation,
        topk=args.topk,
    )
    np.save(output_path, final_vec.astype(np.float32))
    print(f"Done: {sample_id}")


if __name__ == "__main__":
    main()
