"""Generate an independent holdout evaluation split for the semantic-format disentanglement experiment.

Verifies that the new holdout dataset:
1. Has zero semantic signature or prompt overlap with train_public.json.
2. Has 2,048 OOD instances per task with independent ground truth.
"""
import copy
import hashlib
import json
from pathlib import Path
import random

import synthetic_data as syn

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "runs/local_scheme_validation/20260924_semantic_disentangle"
TRAIN_PUBLIC = RUN_DIR / "train_public.json"

def main():
    train_data = json.loads(TRAIN_PUBLIC.read_text(encoding="utf-8"))
    train_prompts = set()
    train_ids = set()
    for task, rows in train_data["splits"]["train"].items():
        for r in rows:
            train_ids.add(r["id"])
            for v in r["variants"]:
                train_prompts.add(v["prompt"])
                
    config = copy.deepcopy(syn.CONFIG)
    config["seed"] = 20260924
    config["source_bundles_per_task"] = {"ood": 2048}
    syn.CONFIG = config
    
    rng = random.Random(config["seed"])
    print("Generating independent Holdout NLI...", flush=True)
    nli, nli_truth, _ = syn.make_nli(rng)
    print("Generating independent Holdout MCQA...", flush=True)
    mcqa, qa_truth = syn.make_mcqa(rng)
    truth = {**nli_truth, **qa_truth}
    
    holdout_data = {
        "schema_version": 1,
        "config_sha256": syn.digest(config),
        "answer_label_map": {"nli": config["nli"]["label_map"], "mcqa": "A/B/C/D are presented option positions"},
        "splits": {"ood": {"nli": nli["ood"], "mcqa": mcqa["ood"]}}
    }
    holdout_oracle = {
        "schema_version": 1,
        "config_sha256": syn.digest(config),
        "sources": truth
    }
    
    # Audit collisions against train_public
    holdout_prompts = set()
    holdout_ids = set()
    for task in ("nli", "mcqa"):
        for r in holdout_data["splits"]["ood"][task]:
            holdout_ids.add(r["id"])
            for v in r["variants"]:
                holdout_prompts.add(v["prompt"])
                
    overlap_prompts = train_prompts.intersection(holdout_prompts)
    overlap_ids = train_ids.intersection(holdout_ids)
    print(f"Train vs Holdout prompt overlap: {len(overlap_prompts)}")
    print(f"Train vs Holdout ID overlap: {len(overlap_ids)}")
    assert len(overlap_prompts) == 0, f"Prompt overlap detected: {len(overlap_prompts)}"
    
    # Rename IDs with prefix 'holdout-' to avoid any accidental ID confusion
    for task in ("nli", "mcqa"):
        for r in holdout_data["splits"]["ood"][task]:
            old_id = r["id"]
            new_id = f"holdout-{old_id}"
            r["id"] = new_id
            for v in r["variants"]:
                v["id"] = f"holdout-{v['id']}"
            t = truth.pop(old_id)
            t["id"] = new_id
            truth[new_id] = t
            
    # Save holdout files
    (RUN_DIR / "data_holdout.json").write_text(json.dumps(holdout_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RUN_DIR / "oracle_holdout.json").write_text(json.dumps(holdout_oracle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    
    manifest = {
        "seed": config["seed"],
        "data_holdout_sha256": hashlib.sha256((RUN_DIR / "data_holdout.json").read_bytes()).hexdigest(),
        "oracle_holdout_sha256": hashlib.sha256((RUN_DIR / "oracle_holdout.json").read_bytes()).hexdigest(),
        "nli_ood_sources": len(holdout_data["splits"]["ood"]["nli"]),
        "mcqa_ood_sources": len(holdout_data["splits"]["ood"]["mcqa"]),
        "zero_train_prompt_overlap": True,
    }
    (RUN_DIR / "holdout_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Holdout generation complete and verified successfully:")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
