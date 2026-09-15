"""Independently reconstruct holdout tensors and audit group exclusions from TSV."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unicodedata

import torch


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify(root=None):
    root = Path(root or Path(__file__).resolve().parents[2])
    run = root / "runs/neurosymbolic_nli/20260915_review_v3"
    previous = root / "runs/neurosymbolic_nli/20260915_order_v1"
    report_path = run / "holdout_verification.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    manifest = json.loads((run / "holdout_manifest.json").read_text())
    payload = torch.load(run / "holdout.pt", map_location="cpu", weights_only=True)
    reference = torch.load(previous / "data.pt", map_location="cpu", weights_only=True)
    split = payload["split"]
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise AssertionError(name)

    check("holdout_file_hash", digest(run / "holdout.pt") == manifest["holdout_sha256"])
    check("v1_cache_hash", digest(previous / "data.pt") == manifest["v1_cache_sha256"])
    check("v1_protocol_hash", digest(previous / "protocol.json") == manifest["v1_protocol_sha256"])
    check("v1_vocab_hash", digest(previous / "vocab.json") == manifest["v1_vocab_sha256"])
    check("preparer_hash", digest(root / "experiments/neurosymbolic_nli/prepare_holdout_v3.py") == manifest["preparer_sha256"])
    check("count_and_balance", len(split["row_ids"]) == 6000 and split["labels"].bincount(minlength=3).tolist() == [2000] * 3)
    check("image_rule_not_relaxed", payload["group_rule"] == "image_disjoint")
    row_positions = {value: index for index, value in enumerate(split["row_ids"])}
    check("unique_source_rows", len(row_positions) == 6000)
    vocab = {token: index for index, token in enumerate(reference["vocab"])}
    label_map = {"entailment": 0, "contradiction": 1, "neutral": 2}

    def norm(tokens):
        return tuple(unicodedata.normalize("NFKC", token).casefold() for token in tokens)

    def keys(row):
        premise = row["sentence1_binary_parse"].replace("(", "").replace(")", "").split()
        hypothesis = row["sentence2_binary_parse"].replace("(", "").replace(")", "").split()
        return {"pairs": (norm(premise), norm(hypothesis)), "pair_ids": row["pairID"],
                "premises": norm(premise), "images": row["captionID"].partition("#")[0],
                "captions": row["captionID"]}, premise, hypothesis

    selected_sets = {key: set() for key in ("pairs", "pair_ids", "premises", "images", "captions")}
    reference_sets = {name: {key: set() for key in selected_sets} for name in ("train", "dev", "test")}
    matched = 0
    reconstructed = {name: torch.zeros_like(split[name]) for name in ("premises", "hypotheses")}
    reconstruction_lengths = {name: torch.zeros_like(split[name]) for name in ("premise_lengths", "hypothesis_lengths")}
    for name in reference_sets:
        source = root / "data/snli_1.0" / f"snli_1.0_{name}.txt"
        check(f"{name}_source_hash", digest(source) == manifest["source_sha256"][name])
        reference_rows = set(reference["splits"][name]["row_ids"])
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row_id, row in enumerate(reader, 2):
                selected_index = row_positions.get(row_id) if name == "train" else None
                if row_id not in reference_rows and selected_index is None:
                    continue
                row_keys, premise, hypothesis = keys(row)
                if row_id in reference_rows:
                    for key, value in row_keys.items():
                        reference_sets[name][key].add(value)
                if selected_index is None:
                    continue
                matched += 1
                check(f"row_{row_id}_metadata", row["pairID"] == split["pair_ids"][selected_index]
                      and row["captionID"] == split["caption_ids"][selected_index]
                      and row_keys["images"] == split["image_ids"][selected_index]
                      and row["sentence1"] == split["premise_texts"][selected_index]
                      and row["sentence2"] == split["hypothesis_texts"][selected_index]
                      and label_map[row["gold_label"]] == int(split["labels"][selected_index]))
                for key, value in row_keys.items():
                    selected_sets[key].add(value)
                for tokens, tensor_name, length_name in (
                    (premise, "premises", "premise_lengths"),
                    (hypothesis, "hypotheses", "hypothesis_lengths"),
                ):
                    ids = [vocab.get(token, 1) for token in tokens[:50]] or [1]
                    reconstructed[tensor_name][selected_index, :len(ids)] = torch.tensor(ids)
                    reconstruction_lengths[length_name][selected_index] = len(ids)
    check("all_selected_source_rows_found", matched == 6000)
    for key, value in {**reconstructed, **reconstruction_lengths}.items():
        check(f"independent_source_reconstruction_{key}", torch.equal(value, split[key]))
    overlaps = {name: {key: len(values & selected_sets[key]) for key, values in groups.items()} for name, groups in reference_sets.items()}
    check("all_reference_pair_premise_caption_image_overlaps_zero", all(count == 0 for group in overlaps.values() for count in group.values()))
    check("recorded_overlap_checks_match", overlaps == manifest["overlap_with_v1_by_split"])
    check("unique_pairs_and_ids", len(selected_sets["pairs"]) == len(selected_sets["pair_ids"]) == 6000)
    check("recorded_group_counts_match", {key: len(values) for key, values in selected_sets.items()} == manifest["selected_unique_groups"])
    # Keep successful per-row metadata checks compact in the durable report.
    metadata_checks = sum(key.startswith("row_") for key in checks)
    report = {"all_passed": all(checks.values()), "checked_source_records": metadata_checks,
              "checks": {key: value for key, value in checks.items() if not key.startswith("row_")},
              "overlaps": overlaps, "selected_groups": {key: len(values) for key, values in selected_sets.items()},
              "holdout_sha256": manifest["holdout_sha256"], "verifier_sha256": digest(Path(__file__).resolve()),
              "verification": "Separate csv.DictReader parser and independent tensor reconstruction; preparer functions not imported."}
    with tempfile.NamedTemporaryFile(dir=run, prefix=".holdout_verification.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.link(temporary, report_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    verify()
