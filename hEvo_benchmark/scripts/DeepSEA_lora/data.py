from __future__ import annotations

import ast
from functools import lru_cache
import json
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


NUM_LABELS = 919
GPN_STAR_CHROMS = [*(f"chr{i}" for i in range(1, 23)), "chrX", "chrY", "chrM"]


def parse_deepsea_line(line: str) -> tuple[str, np.ndarray | None]:
    parts = line.rstrip("\n").split(maxsplit=3)
    if not parts:
        raise ValueError("Empty DeepSEA row")
    inline = None
    if len(parts) >= 4:
        blob = parts[3].strip()
        if blob.startswith('"') and blob.endswith('"'):
            blob = blob[1:-1]
        if blob:
            inline = np.asarray(ast.literal_eval(blob), dtype=np.float32)
    return parts[0], inline


def count_csv_rows(path: str | Path) -> int:
    with Path(path).open("rb") as handle:
        return sum(1 for _ in handle)


class DeepSEAIterableDataset(IterableDataset):
    """Streaming DeepSEA reader with deterministic rank/worker sharding."""

    def __init__(
        self,
        csv_path: str | Path,
        labels_path: str | Path | None,
        split: str,
        max_samples: int | None = None,
        num_labels: int = NUM_LABELS,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = False,
        shuffle_buffer_size: int = 10_000,
        seed: int = 42,
        offsets_path: str | Path | None = None,
        row_stride: int = 1,
    ) -> None:
        super().__init__()
        self.csv_path = Path(csv_path)
        self.labels_path = Path(labels_path) if labels_path else None
        self.split = split
        self.max_samples = max_samples
        self.num_labels = num_labels
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.shuffle_buffer_size = max(1, shuffle_buffer_size)
        self.seed = seed
        self.offsets_path = Path(offsets_path) if offsets_path else None
        self.row_stride = max(1, int(row_stride))
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _label_array(self) -> np.ndarray | None:
        return np.load(self.labels_path, mmap_mode="r") if self.labels_path else None

    def _local_shard(self) -> tuple[int, int]:
        worker = get_worker_info()
        if worker is None:
            return self.rank, self.world_size
        return (
            self.rank * worker.num_workers + worker.id,
            self.world_size * worker.num_workers,
        )

    def _iter_records(self) -> Iterator[tuple[int, str, np.ndarray]]:
        shard_id, shard_count = self._local_shard()
        labels = self._label_array()
        buffer: list[tuple[int, str, np.ndarray]] = []
        rng = random.Random(self.seed + 1009 * self.epoch + shard_id)

        def emit_buffer(
            force_all: bool = False,
        ) -> Iterator[tuple[int, str, np.ndarray]]:
            while buffer and (force_all or len(buffer) >= self.shuffle_buffer_size):
                yield buffer.pop(rng.randrange(len(buffer)))

        def records_from_line(
            row_index: int, line: str
        ) -> Iterator[tuple[int, str, np.ndarray]]:
            sequence, inline_labels = parse_deepsea_line(line)
            raw = inline_labels
            if raw is None and labels is not None:
                raw = np.asarray(labels[row_index], dtype=np.float32)
            if raw is None:
                raise ValueError(f"No labels for {self.csv_path}:{row_index}")
            if raw.shape != (self.num_labels,):
                raise ValueError(
                    f"Expected {self.num_labels} labels at {self.csv_path}:{row_index}, got {raw.shape}"
                )
            record = (row_index, sequence, raw)
            if self.shuffle:
                buffer.append(record)
                yield from emit_buffer()
            else:
                yield record

        with self.csv_path.open("r", encoding="utf-8") as handle:
            if self.offsets_path is not None:
                checkpoints = json.loads(self.offsets_path.read_text(encoding="utf-8"))
                row_count = int(checkpoints["row_count"])
                row_limit = (
                    row_count
                    if self.row_stride > 1
                    else min(row_count, self.max_samples)
                    if self.max_samples is not None
                    else row_count
                )
                first = row_limit * shard_id // shard_count
                last = row_limit * (shard_id + 1) // shard_count
                if first < last:
                    stride = int(checkpoints["stride"])
                    checkpoint_row = (first // stride) * stride
                    handle.seek(int(checkpoints["offsets"][checkpoint_row // stride]))
                    row_index = checkpoint_row
                    while row_index < last:
                        line = handle.readline()
                        if not line:
                            break
                        if row_index >= first and row_index % self.row_stride == 0:
                            yield from records_from_line(row_index, line)
                        row_index += 1
            else:
                if self.row_stride > 1:
                    raise ValueError("row_stride > 1 requires sparse CSV offsets")
                for row_index, line in enumerate(handle):
                    if self.max_samples is not None and row_index >= self.max_samples:
                        break
                    if row_index % shard_count != shard_id:
                        continue
                    yield from records_from_line(row_index, line)
        if self.shuffle:
            yield from emit_buffer(force_all=True)

    def __iter__(self) -> Iterator[dict[str, object]]:
        for row_index, sequence, labels in self._iter_records():
            yield {
                "row_index": row_index,
                "sequence": sequence,
                "labels": torch.from_numpy(np.asarray(labels, dtype=np.float32)),
            }


class DeepSEACollator:
    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        sequences = [str(feature["sequence"]) for feature in features]
        is_rnabert = self.tokenizer.__class__.__name__ == "RNABertTokenizer"
        if is_rnabert:
            # RNABert's tokenizer is explicitly whitespace-tokenized.  Passing
            # a continuous DNA string makes every example [UNK].  Its custom
            # attention also cannot safely softmax completely padded query
            # rows, so use dynamic padding; DeepSEA sequences are fixed-length
            # and therefore receive no padding in normal batches.
            sequences = [" ".join(sequence) for sequence in sequences]
        encoded = self.tokenizer(
            sequences,
            padding=True if is_rnabert else "max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if "attention_mask" not in encoded:
            pad_id = getattr(self.tokenizer, "pad_token_id", 0)
            encoded["attention_mask"] = (encoded["input_ids"] != pad_id).long()
        encoded["labels"] = torch.stack([feature["labels"] for feature in features])
        return dict(encoded)


@lru_cache(maxsize=2)
def _gpn_star_msa_store(path: str, n_species: int):
    from gpn.star.data import GenomeMSA

    return GenomeMSA(path, n_species=n_species, in_memory=False)


class GPNStarMSACollator:
    """Fetch native 243+204-way GPN-Star MSA rows for mapped DeepSEA windows."""

    def __init__(
        self,
        split: str,
        coordinate_root: str | Path,
        msa_243: str | Path,
        msa_204: str | Path,
    ):
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Unsupported DeepSEA split: {split}")
        self.split = split
        root = Path(coordinate_root)
        self.coordinates = np.load(root / f"{split}.coordinates.npy", mmap_mode="r")
        self.msa_243_path = str(msa_243)
        self.msa_204_path = str(msa_204)
        self._msas = None
        self._available_chroms: set[str] | None = None

    def _open_msa(self):
        if self._msas is None:
            self._msas = (
                _gpn_star_msa_store(self.msa_243_path, 243),
                _gpn_star_msa_store(self.msa_204_path, 204),
            )
            self._available_chroms = set(self._msas[0].data) & set(self._msas[1].data)
            missing_autosomes = [
                chrom
                for chrom in GPN_STAR_CHROMS[:22]
                if not any(
                    candidate in self._available_chroms
                    for candidate in (chrom, chrom.removeprefix("chr"))
                )
            ]
            if missing_autosomes:
                raise ValueError(
                    f"GPN-Star MSA is missing autosomes: {missing_autosomes}"
                )
        return self._msas

    def _resolve_chrom(self, chrom: str) -> str:
        if self._available_chroms is None:
            raise RuntimeError("GPN-Star MSA stores have not been opened")
        without = chrom.removeprefix("chr")
        for candidate in dict.fromkeys((chrom, without, "chr" + without)):
            if candidate in self._available_chroms:
                return candidate
        raise ValueError(f"Chromosome {chrom} is absent from both GPN-Star MSA stores")

    def __call__(self, features: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        msas = self._open_msa()
        tokens: list[np.ndarray] = []
        validity: list[bool] = []
        for feature in features:
            row = int(feature["row_index"])
            if row < 0 or row >= len(self.coordinates):
                raise IndexError(
                    f"{self.split} row index outside coordinate map: {row}"
                )
            coordinate = self.coordinates[row]
            chrom_id = int(coordinate["chrom_id"])
            valid = bool(coordinate["valid"]) and 1 <= chrom_id <= len(GPN_STAR_CHROMS)
            chrom = GPN_STAR_CHROMS[chrom_id - 1] if valid else None
            start = int(coordinate["start"]) if valid else -1
            strand = "+" if int(coordinate["strand"]) >= 0 else "-"
            if valid:
                try:
                    chrom = self._resolve_chrom(chrom)
                except ValueError:
                    valid = False
            if valid:
                end = start + 1000
                if (
                    end > msas[0].data[chrom].shape[0]
                    or end > msas[1].data[chrom].shape[0]
                ):
                    valid = False
            if valid:
                pieces = [
                    msa.get_msa(chrom, start, end, strand=strand, tokenize=True)
                    for msa in msas
                ]
                source = np.concatenate(pieces, axis=1)
                if source.shape != (1000, 447):
                    raise ValueError(
                        f"Unexpected GPN-Star MSA shape {source.shape} at {chrom}:{start}-{end}"
                    )
            else:
                # A zero-loss dummy keeps every DDP rank on the same step count.
                source = np.zeros((1000, 447), dtype=np.uint8)
            tokens.append(source.astype(np.int64, copy=False))
            validity.append(valid)
        source_ids = torch.as_tensor(np.stack(tokens), dtype=torch.long)
        batch = {
            "input_ids": source_ids[:, :, :1].contiguous(),
            "source_ids": source_ids,
            "target_species": torch.zeros((len(features), 1), dtype=torch.long),
            "sample_valid": torch.as_tensor(validity, dtype=torch.bool),
            "labels": torch.stack([feature["labels"] for feature in features]),
        }
        return batch


def make_loader(
    dataset: DeepSEAIterableDataset,
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    collator=None,
) -> DataLoader:
    worker_count = max(0, int(num_workers))
    loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": collator
        if collator is not None
        else DeepSEACollator(tokenizer, max_length),
        "num_workers": worker_count,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": worker_count > 0,
    }
    if worker_count > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    return DataLoader(
        dataset,
        **loader_kwargs,
    )


def validate_dataset(root: str | Path) -> dict[str, object]:
    root = Path(root)
    expected = {
        "train_csv": root / "train_out.csv",
        "train_labels": root / "train_labels.npy",
        "valid_csv": root / "valid_out.csv",
        "test_csv": root / "test_out.csv",
        "test_labels": root / "test_labels.npy",
    }
    missing = [str(path) for path in expected.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing DeepSEA files: " + ", ".join(missing))
    train_labels = np.load(expected["train_labels"], mmap_mode="r")
    test_labels = np.load(expected["test_labels"], mmap_mode="r")
    counts = {
        key: count_csv_rows(path)
        for key, path in expected.items()
        if key.endswith("_csv")
    }
    if train_labels.ndim != 2 or train_labels.shape[1] != NUM_LABELS:
        raise ValueError(f"Unexpected train label shape: {train_labels.shape}")
    if test_labels.ndim != 2 or test_labels.shape[1] != NUM_LABELS:
        raise ValueError(f"Unexpected test label shape: {test_labels.shape}")
    if counts["train_csv"] != train_labels.shape[0]:
        raise ValueError("train_out.csv and train_labels.npy row counts differ")
    if counts["test_csv"] != test_labels.shape[0]:
        raise ValueError("test_out.csv and test_labels.npy row counts differ")
    return {
        "root": str(root),
        "files": {key: str(path) for key, path in expected.items()},
        "csv_rows": counts,
        "train_labels_shape": list(train_labels.shape),
        "test_labels_shape": list(test_labels.shape),
        "num_labels": NUM_LABELS,
    }
