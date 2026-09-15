"""Freeze an unused-SNLI-train audit set without changing any v1 inputs.

The set is an in-domain sanity check, not an official test or a causal benchmark.
Neither model predictions nor linguistic difficulty enter its sampling rule.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import unicodedata

import torch

from .data import LABELS, _encode, load_data, sha256_file, tokenize_parse


SEED = 20260916
PER_LABEL = 2000


def normalized(tokens):
    return tuple(unicodedata.normalize("NFKC", token).casefold() for token in tokens)


def source_records(path):
    with path.open(encoding="utf-8") as handle:
        header = next(handle).rstrip("\r\n").split("\t")
        assert header[:3] == ["gold_label", "sentence1_binary_parse", "sentence2_binary_parse"]
        assert header[7:9] == ["captionID", "pairID"]
        for row_id, line in enumerate(handle, 2):
            fields = line.rstrip("\r\n").split("\t")
            if fields[0] not in LABELS:
                continue
            if len(fields) < 9 or not fields[7] or not fields[8]:
                raise ValueError(f"Malformed source record {path}:{row_id}")
            yield {
                "row_id": row_id, "pair_id": fields[8],
                "caption_id": fields[7], "image_id": fields[7].split("#", 1)[0],
                "premise_tokens": tokenize_parse(fields[1]),
                "hypothesis_tokens": tokenize_parse(fields[2]),
                "premise_text": fields[5], "hypothesis_text": fields[6],
                "label": LABELS[fields[0]],
            }


def pair_key(record):
    return normalized(record["premise_tokens"]), normalized(record["hypothesis_tokens"])


class Reservoir:
    def __init__(self):
        self.counts = Counter()
        self.pools = {label: [] for label in LABELS.values()}
        self.rngs = {label: random.Random(SEED + label) for label in LABELS.values()}

    def add(self, record):
        label = record["label"]
        self.counts[label] += 1
        pool = self.pools[label]
        if len(pool) < PER_LABEL:
            pool.append(record)
        else:
            replacement = self.rngs[label].randrange(self.counts[label])
            if replacement < PER_LABEL:
                pool[replacement] = record

    def feasible(self):
        return all(len(pool) == PER_LABEL for pool in self.pools.values())

    def selected(self):
        result = [record for label in sorted(self.pools) for record in self.pools[label]]
        random.Random(SEED).shuffle(result)
        return result


def exclusive_atomic_write(path, writer):
    """Publish a complete file via atomic hard link; never replace an existing file."""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        writer(temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)  # Atomic, and raises FileExistsError on collision.
    finally:
        temporary.unlink(missing_ok=True)


def prepare(project_root=None, output_dir=None):
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    previous = root / "runs/neurosymbolic_nli/20260915_order_v1"
    output = Path(output_dir or root / "runs/neurosymbolic_nli/20260915_review_v3").resolve()
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / "holdout.pt", output / "holdout_manifest.json"]
    if any(path.exists() for path in targets):
        raise FileExistsError("Frozen holdout already exists; refusing to replace any output")
    cache = load_data(previous, verify_cache=True)
    protocol = json.loads((previous / "protocol.json").read_text(encoding="utf-8-sig"))
    sources = {split: root / "data/snli_1.0" / f"snli_1.0_{split}.txt" for split in ("train", "dev", "test")}
    hashes = {name: sha256_file(path) for name, path in sources.items()}
    if hashes != cache["source_sha256"]:
        raise ValueError("SNLI source hashes differ from the frozen v1 cache")
    forbidden = {name: {"pairs": set(), "pair_ids": set(), "premises": set(), "images": set(), "captions": set()} for name in sources}
    referenced = {}
    for name, source in sources.items():
        split = cache["splits"][name]
        row_to_index = {row: index for index, row in enumerate(split["row_ids"])}
        if len(row_to_index) != len(split["row_ids"]):
            raise ValueError(f"Duplicate cached row IDs in {name}")
        matched = 0
        for record in source_records(source):
            index = row_to_index.get(record["row_id"])
            if index is None:
                continue
            assert record["pair_id"] == split["pair_ids"][index]
            assert record["label"] == int(split["labels"][index])
            assert record["premise_text"] == split["premise_texts"][index]
            assert record["hypothesis_text"] == split["hypothesis_texts"][index]
            keys = forbidden[name]
            keys["pairs"].add(pair_key(record))
            keys["pair_ids"].add(record["pair_id"])
            keys["premises"].add(normalized(record["premise_tokens"]))
            keys["images"].add(record["image_id"])
            keys["captions"].add(record["caption_id"])
            matched += 1
        assert matched == len(row_to_index)
        referenced[name] = {"verified_source_rows": matched, **{key: len(value) for key, value in forbidden[name].items()}}
        print(f"Verified v1 {name}: {matched:,} source rows, {len(forbidden[name]['images']):,} image groups", flush=True)
    unions = {key: set().union(*(part[key] for part in forbidden.values())) for key in next(iter(forbidden.values()))}
    reservoirs = {name: Reservoir() for name in ("image_disjoint", "premise_disjoint")}
    exclusions = Counter()
    unique_pairs, unique_ids = set(), set()
    eligible_images, eligible_premises = set(), set()
    for record in source_records(sources["train"]):
        exclusions["source_valid_rows"] += 1
        key = pair_key(record)
        if key in unions["pairs"]:
            exclusions["v1_normalized_pair_overlap"] += 1
            continue
        if record["pair_id"] in unions["pair_ids"]:
            exclusions["v1_pair_id_overlap_after_pair_filter"] += 1
            continue
        if key in unique_pairs or record["pair_id"] in unique_ids:
            exclusions["within_candidate_duplicate_pair_or_id_first_source_row_retained"] += 1
            continue
        unique_pairs.add(key)
        unique_ids.add(record["pair_id"])
        if key[0] in unions["premises"]:
            exclusions["v1_premise_overlap_after_pair_deduplication"] += 1
            continue
        reservoirs["premise_disjoint"].add(record)
        eligible_premises.add(key[0])
        if record["image_id"] in unions["images"]:
            exclusions["v1_image_overlap_after_premise_filter"] += 1
            continue
        eligible_images.add(record["image_id"])
        reservoirs["image_disjoint"].add(record)
    feasibility = {
        name: {"eligible_label_counts": dict(sorted(reservoir.counts.items())), "feasible": reservoir.feasible()}
        for name, reservoir in reservoirs.items()
    }
    print(json.dumps({"feasibility": feasibility, "exclusions": exclusions}, indent=2), flush=True)
    mode = next((name for name in ("image_disjoint", "premise_disjoint") if reservoirs[name].feasible()), None)
    if mode is None:
        raise ValueError(f"Neither group rule supplies 2,000 examples per class: {feasibility}")
    selected = reservoirs[mode].selected()
    split, encoded_stats = _encode(selected, {token: index for index, token in enumerate(cache["vocab"])}, protocol["max_length"])
    split["image_ids"] = [record["image_id"] for record in selected]
    split["caption_ids"] = [record["caption_id"] for record in selected]
    selected_keys = {
        "pairs": {pair_key(record) for record in selected},
        "pair_ids": set(split["pair_ids"]),
        "premises": {normalized(record["premise_tokens"]) for record in selected},
        "images": set(split["image_ids"]), "captions": set(split["caption_ids"]),
    }
    overlap_checks = {name: {key: len(selected_keys[key] & part[key]) for key in selected_keys} for name, part in forbidden.items()}
    for checks in overlap_checks.values():
        assert all(checks[key] == 0 for key in ("pairs", "pair_ids", "premises", "captions"))
        if mode == "image_disjoint":
            assert checks["images"] == 0
    assert len(selected_keys["pairs"]) == len(selected) == 6000
    assert len(selected_keys["pair_ids"]) == len(selected)
    assert torch.equal(torch.bincount(split["labels"], minlength=3), torch.tensor([2000, 2000, 2000]))
    assert set(split["row_ids"]).isdisjoint(cache["splits"]["train"]["row_ids"])
    for name in ("premise", "hypothesis"):
        tensor = split["premises" if name == "premise" else "hypotheses"]
        lengths = split[f"{name}_lengths"]
        assert tensor.dtype == torch.long and tensor.shape == (6000, protocol["max_length"])
        assert (lengths >= 1).all() and (lengths <= protocol["max_length"]).all()
        assert tensor.min() >= 0 and tensor.max() < len(cache["vocab"])
        assert (tensor[torch.arange(protocol["max_length"])[None, :] >= lengths[:, None]] == 0).all()
    canonical_rows = json.dumps([{key: record[key] for key in ("row_id", "pair_id", "caption_id", "image_id", "label")} for record in selected], sort_keys=True, separators=(",", ":")).encode()
    content_hash = hashlib.sha256(canonical_rows).hexdigest()
    provenance = {
        "source_sha256": hashes, "v1_cache_sha256": sha256_file(previous / "data.pt"),
        "v1_protocol_sha256": sha256_file(previous / "protocol.json"),
        "v1_vocab_sha256": sha256_file(previous / "vocab.json"),
        "preparer_sha256": sha256_file(Path(__file__).resolve()),
        "selected_rows_sha256": content_hash,
    }
    payload = {"cache_version": 1, "split": split, "label_to_id": LABELS,
               "pad_id": 0, "unk_id": 1, "max_length": protocol["max_length"],
               "sampling_seed": SEED, "group_rule": mode, **provenance}
    manifest = {
        "purpose": "Frozen unused-official-train SNLI in-domain audit; not official test, independent domain, or causal inference.",
        "selection_used_model_predictions": False, "selection_used_linguistic_difficulty": False,
        "sampling_seed": SEED, "target_per_label": PER_LABEL, "label_to_id": LABELS,
        "source": "data/snli_1.0/snli_1.0_train.txt", "group_rule": mode,
        "normalization": "Parse tokens from sentence{1,2}_binary_parse, parentheses removed and whitespace split; each token NFKC normalized and casefolded. Ordered full token tuples before truncation.",
        "group_definition": "SNLI captionID before first # is image ID; exact full normalized premise is premise group.",
        "selection_rule": "Exclude normalized pair or pairID seen in any v1 train/dev/test; keep first source occurrence of candidate duplicate pair or ID; exclude normalized premise seen in any v1 split; prefer exclusion of all seen image IDs when >=2000 eligible rows in each label, otherwise explicitly use premise-only exclusion. Class-wise reservoir sampling with Random(seed+label), then shuffle with Random(seed).",
        "encoding": "Reuse v1 vocabulary and data._encode unchanged: case-preserving parse tokens, OOV=1, PAD=0, max_length=50, empty -> UNK. No vocabulary fitting on holdout.",
        "reference_split_stats": referenced, "filter_counts_in_order": dict(exclusions),
        "candidate_feasibility": feasibility, "premise_disjoint_candidate_unique_premises": len(eligible_premises),
        "image_disjoint_candidate_unique_images": len(eligible_images),
        "selected_unique_groups": {key: len(value) for key, value in selected_keys.items()},
        "overlap_with_v1_by_split": overlap_checks,
        "encoded_stats": encoded_stats, "source_row_numbering": "1-based physical line; header is line 1",
        "dependence_note": "Several selected examples may share an image/premise; use image-group resampling for confidence intervals. Image-disjointness controls direct group reuse but does not remove SNLI annotation artifacts or domain dependence.",
        "all_checks_passed": True, **provenance,
    }
    exclusive_atomic_write(targets[0], lambda temporary: torch.save(payload, temporary))
    manifest["holdout_sha256"] = sha256_file(targets[0])
    exclusive_atomic_write(targets[1], lambda temporary: temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"))
    print(json.dumps({"group_rule": mode, "count": len(selected), "holdout_sha256": manifest["holdout_sha256"], "selected_rows_sha256": content_hash, "output": str(output)}, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    prepare(args.project_root, args.output_dir)
