"""Reproducible, local-only SNLI preparation for the frozen NLI comparison.

Run from the project directory with ``python -m experiments.neurosymbolic_nli.data``.
Only the sampled official training partition supplies the vocabulary. The official
development and test partitions are encoded after that vocabulary is frozen.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any

import torch


LABELS = {"entailment": 0, "contradiction": 1, "neutral": 2}
DEFAULT_RUN = Path("runs/neurosymbolic_nli/20260915_order_v1")
CACHE_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenize_parse(text: str) -> list[str]:
    """Match the notebook read_snli: remove parentheses, preserve token case."""
    return re.sub(r"[()]", "", text).split()


def _records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        header = next(handle).rstrip("\r\n").split("\t")
        if header[:3] != ["gold_label", "sentence1_binary_parse", "sentence2_binary_parse"]:
            raise ValueError(f"Unexpected SNLI header in {path}")
        for line_number, line in enumerate(handle, start=2):
            fields = line.rstrip("\r\n").split("\t")
            if fields[0] not in LABELS:
                continue
            if len(fields) < 9:
                raise ValueError(f"Malformed SNLI row at {path}:{line_number}")
            yield {
                "row_id": line_number,
                "pair_id": fields[8],
                "premise_tokens": tokenize_parse(fields[1]),
                "hypothesis_tokens": tokenize_parse(fields[2]),
                "premise_text": fields[5],
                "hypothesis_text": fields[6],
                "label": LABELS[fields[0]],
            }


def _sample_train(path: Path, total: int, seed: int):
    if total % len(LABELS):
        raise ValueError("Stratified sample size must be divisible by three")
    per_label = total // len(LABELS)
    reservoirs: dict[int, list[dict]] = {label: [] for label in LABELS.values()}
    counts = Counter()
    rngs = {label: random.Random(seed + label) for label in LABELS.values()}
    for record in _records(path):
        label = record["label"]
        counts[label] += 1
        pool = reservoirs[label]
        if len(pool) < per_label:
            pool.append(record)
        else:
            replacement = rngs[label].randrange(counts[label])
            if replacement < per_label:
                pool[replacement] = record
    if any(len(pool) != per_label for pool in reservoirs.values()):
        raise ValueError("Not enough official training examples for stratification")
    sampled = [record for label in sorted(reservoirs) for record in reservoirs[label]]
    random.Random(seed).shuffle(sampled)
    return sampled, dict(sorted(counts.items()))


def _encode(records: list[dict], token_to_id: dict[str, int], max_length: int):
    premises = torch.zeros((len(records), max_length), dtype=torch.long)
    hypotheses = torch.zeros_like(premises)
    premise_lengths = torch.empty(len(records), dtype=torch.long)
    hypothesis_lengths = torch.empty_like(premise_lengths)
    unknown_counts = Counter()
    token_counts = Counter()
    truncated_counts = Counter()
    empty_counts = Counter()
    for index, record in enumerate(records):
        for name, output, lengths in (
            ("premise", premises, premise_lengths),
            ("hypothesis", hypotheses, hypothesis_lengths),
        ):
            source_tokens = record[f"{name}_tokens"]
            truncated_counts[name] += int(len(source_tokens) > max_length)
            empty_counts[name] += int(not source_tokens)
            tokens = source_tokens[:max_length]
            ids = [token_to_id.get(token, 1) for token in tokens] or [1]
            output[index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            lengths[index] = len(ids)
            unknown_counts[name] += sum(token_id == 1 for token_id in ids)
            token_counts[name] += len(ids)
    encoded = {
        "premises": premises,
        "hypotheses": hypotheses,
        "labels": torch.tensor([record["label"] for record in records], dtype=torch.long),
        "premise_lengths": premise_lengths,
        "hypothesis_lengths": hypothesis_lengths,
        "row_ids": [record["row_id"] for record in records],
        "pair_ids": [record["pair_id"] for record in records],
        "premise_texts": [record["premise_text"] for record in records],
        "hypothesis_texts": [record["hypothesis_text"] for record in records],
    }
    stats = {
        "count": len(records),
        "label_counts": dict(sorted(Counter(record["label"] for record in records).items())),
        "unknown_tokens": dict(unknown_counts),
        "encoded_tokens": dict(token_counts),
        "truncated_sentences": dict(truncated_counts),
        "empty_sentences_replaced_by_unk": dict(empty_counts),
    }
    return encoded, stats


def _embeddings(path: Path, vocab: list[str], dimension: int, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    embeddings = torch.randn((len(vocab), dimension), generator=generator) * 0.05
    embeddings[0].zero_()
    relevant = {token: index for index, token in enumerate(vocab) if index > 1}
    found: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            token, separator, values = line.rstrip("\r\n").partition(" ")
            if not separator or token not in relevant:
                continue
            vector = values.split()
            if len(vector) != dimension:
                raise ValueError(f"Invalid GloVe dimension at {path}:{line_number}")
            index = relevant[token]
            embeddings[index] = torch.tensor([float(value) for value in vector])
            found.add(index)
    if not torch.isfinite(embeddings).all():
        raise ValueError("Non-finite values in embeddings")
    return embeddings, {
        "pretrained_tokens": len(found),
        "eligible_vocab_tokens": len(relevant),
        "coverage_fraction": len(found) / max(len(relevant), 1),
        "unknown_initialization": "CPU torch.randn * 0.05",
        "unknown_seed": seed,
        "pad_row_zero": bool(torch.count_nonzero(embeddings[0]) == 0),
    }


def _pair_keys(records: list[dict]):
    return {
        (tuple(record["premise_tokens"]), tuple(record["hypothesis_tokens"]))
        for record in records
    }


def prepare_data(
    project_root: str | Path | None = None,
    run_dir: str | Path | None = None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Build and persist the fixed experiment inputs; reuse a verified cache."""
    project_root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    run_dir = Path(run_dir) if run_dir else project_root / DEFAULT_RUN
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = run_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    if (run_dir / "data.pt").exists() and not force:
        return load_data(run_dir)
    print("Preparing fixed SNLI data from local source files", flush=True)
    sources = {
        name: project_root / "data/snli_1.0" / f"snli_1.0_{name}.txt"
        for name in ("train", "dev", "test")
    }
    glove_path = project_root / "data/glove.6B.100d/vec.txt"
    for path in (*sources.values(), glove_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes = {name: sha256_file(path) for name, path in sources.items()}
    train, original_train_counts = _sample_train(
        sources["train"], protocol["train_examples"], protocol["train_sampling_seed"]
    )
    frequencies = Counter(
        token for record in train
        for field in ("premise_tokens", "hypothesis_tokens") for token in record[field]
    )
    vocab = ["<pad>", "<unk>"] + [
        token for token, frequency in sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
        if frequency >= 2 and token not in ("<pad>", "<unk>")
    ]
    token_to_id = {token: index for index, token in enumerate(vocab)}
    print(f"Sampled {len(train):,} training pairs; frozen vocabulary has {len(vocab):,} tokens", flush=True)
    records = {"train": train}
    # Evaluation partitions cannot affect the already frozen vocabulary or sample.
    records.update({name: list(_records(sources[name])) for name in ("dev", "test")})
    splits, split_stats = {}, {}
    for name, split_records in records.items():
        splits[name], split_stats[name] = _encode(split_records, token_to_id, protocol["max_length"])
        print(f"Encoded {name}: {len(split_records):,} examples", flush=True)
    embeddings, embedding_stats = _embeddings(
        glove_path, vocab, protocol["embedding_dim"], protocol["train_sampling_seed"]
    )
    pairs = {name: _pair_keys(split_records) for name, split_records in records.items()}
    pair_ids = {name: {record["pair_id"] for record in split_records} for name, split_records in records.items()}
    overlaps = {
        f"{left}_{right}": {
            "normalized_sentence_pairs": len(pairs[left] & pairs[right]),
            "pair_ids": len(pair_ids[left] & pair_ids[right]),
        }
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    }
    vocab_path = run_dir / "vocab.json"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    payload = {
        "cache_version": CACHE_VERSION,
        "splits": splits,
        "embeddings": embeddings,
        "vocab": vocab,
        "pad_id": 0,
        "unk_id": 1,
        "label_to_id": LABELS,
        "protocol_sha256": sha256_file(protocol_path),
        "source_sha256": hashes,
    }
    cache_path = run_dir / "data.pt"
    torch.save(payload, cache_path)
    manifest = {
        "cache_version": CACHE_VERSION,
        "protocol_sha256": payload["protocol_sha256"],
        "source_files": {
            name: {"path": str(path.relative_to(project_root)), "sha256": hashes[name], "size_bytes": path.stat().st_size}
            for name, path in sources.items()
        },
        "embedding_source": {"path": str(glove_path.relative_to(project_root)), "sha256": sha256_file(glove_path)},
        "sampling": {
            "method": "Independent class-wise reservoir sampling, followed by fixed shuffle",
            "seed": protocol["train_sampling_seed"],
            "class_rng_seeds": {str(label): protocol["train_sampling_seed"] + label for label in LABELS.values()},
            "original_train_valid_label_counts": original_train_counts,
            "row_id_definition": "One-based physical source-file line number; header is line 1",
        },
        "tokenization": "Remove all parentheses from binary parse columns 1 and 2; split whitespace; preserve case",
        "vocabulary": {"source": "sampled official train only", "min_frequency": 2, "size": len(vocab), "pad_id": 0, "unk_id": 1, "sha256": sha256_file(vocab_path)},
        "max_length": protocol["max_length"],
        "splits": {
            name: {
                **split_stats[name],
                "row_ids": splits[name]["row_ids"],
                "pair_ids": splits[name]["pair_ids"],
            }
            for name in records
        },
        "cross_split_overlap": overlaps,
        "within_split_duplicate_sentence_pairs": {name: len(records[name]) - len(pairs[name]) for name in records},
        "embedding": embedding_stats,
        "cache": {"file": "data.pt", "sha256": sha256_file(cache_path), "size_bytes": cache_path.stat().st_size},
        "torch_version": str(torch.__version__),
    }
    (run_dir / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"splits": split_stats, "embedding": embedding_stats, "overlap": overlaps}, indent=2), flush=True)
    return payload


def load_data(run_dir: str | Path, *, verify_cache: bool = True) -> dict[str, Any]:
    """Load CPU tensors and validate provenance before experiment training."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8"))
    if sha256_file(run_dir / "protocol.json") != manifest["protocol_sha256"]:
        raise ValueError("Protocol changed after data preparation")
    if verify_cache and sha256_file(run_dir / "data.pt") != manifest["cache"]["sha256"]:
        raise ValueError("Data cache checksum mismatch")
    data = torch.load(run_dir / "data.pt", map_location="cpu", weights_only=True)
    if data["cache_version"] != CACHE_VERSION:
        raise ValueError("Unsupported cache version")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare_data(args.project_root, args.run_dir, force=args.force)


if __name__ == "__main__":
    main()
