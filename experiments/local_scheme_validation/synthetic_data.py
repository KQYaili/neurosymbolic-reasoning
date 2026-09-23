"""Frozen controlled-language pilot data; standard library only, no model/GPU use.

Train inputs and evaluator-only truth are separate artifacts.  All transformations
of a source belong to one split.  This is propositional NLI and arithmetic MCQA,
not a benchmark of open-world language or causal intervention identification.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "runs/local_scheme_validation/20260923_relational_v1"
LABELS = ("A", "B", "C")  # entailment, contradiction, neutral
MC_LABELS = ("A", "B", "C", "D")
CONFIG = {
    "version": "controlled_relational_v1",
    "seed": 20260923,
    "source_bundles_per_task": {"train": 1024, "dev": 128, "test": 256, "ood": 256},
    "supervised_sources_per_task": 102,
    "supervision_fraction_actual": 102 / 1024,
    "supervision_sampling": "fixed-seed stratified by base NLI label or MCQA correct position",
    "nli": {
        "id_atoms": 3, "id_max_literal_leaves": 3,
        "ood_atoms": 4, "ood_literal_leaves": 4,
        "ood_each_formula_depends_on_all_four_atoms": True,
        "satisfiable_premise_and_hypothesis": True,
        "variants": ["base", "swap"],
        "label_map": {"A": "entailment", "B": "contradiction", "C": "neutral"},
        "swap_allowed": {"A": ["A", "C"], "B": ["B"], "C": ["A", "C"]},
        "cross_split_exclusion": "unordered pair of complete truth masks plus atom count",
        "supervised_swap_target": "allowed set derived only from supervised base label; never reverse oracle label",
    },
    "mcqa": {
        "id_operators": ["add", "subtract", "multiply"], "id_operand_range": [2, 40],
        "ood_operators": ["add_then_multiply", "multiply_then_add", "subtract_then_multiply"],
        "ood_main_operand_range": [41, 99], "ood_third_operand_range": [2, 7],
        "variants": 4, "permutation_family": "four cyclic shifts of randomized base order",
        "correct_position_and_numeric_rank": "balanced jointly; canonical option ID also balanced",
        "distinct_options": True, "distractor_offsets_id": [1, 12],
        "distractor_offsets_ood": [1, 25],
        "cross_split_exclusion": "expression signature; commutative operands canonicalized",
    },
    "test_policy": "Frozen before training; do not alter after viewing test/OOD outcomes.",
    "truth_policy": "All hidden labels and structured oracles only in oracle.json; trainer uses data.json and training_view().",
}


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(obj):
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atom_mask(index, atoms):
    return sum(1 << world for world in range(1 << atoms) if (world >> index) & 1)


def evaluate_ast(ast, world):
    """Independent Boolean evaluator, intentionally not the bit-mask builder."""
    if ast[0] == "atom":
        return bool((world >> ast[1]) & 1)
    if ast[0] == "not":
        return not evaluate_ast(ast[1], world)
    if ast[0] == "and":
        return evaluate_ast(ast[1], world) and evaluate_ast(ast[2], world)
    if ast[0] == "or":
        return evaluate_ast(ast[1], world) or evaluate_ast(ast[2], world)
    raise ValueError(ast)


def truth_mask(ast, atoms):
    return sum(1 << world for world in range(1 << atoms) if evaluate_ast(ast, world))


def nli_label(premise_mask, hypothesis_mask):
    if not premise_mask or not hypothesis_mask:
        raise ValueError("This NLI protocol requires two satisfiable propositions")
    overlap = premise_mask & hypothesis_mask
    return "A" if overlap == premise_mask else "B" if overlap == 0 else "C"


def formula_pool(atoms, max_leaves):
    """One shortest deterministic NNF representative for each reachable truth set."""
    full = (1 << (1 << atoms)) - 1
    best, by_cost = {}, defaultdict(list)
    for index in range(atoms):
        atom = ("atom", index)
        for mask, ast in ((atom_mask(index, atoms), atom), (full ^ atom_mask(index, atoms), ("not", atom))):
            best[mask] = (1, ast)
            by_cost[1].append(mask)
    for cost in range(2, max_leaves + 1):
        candidates = {}
        for left_cost in range(1, cost):
            for left in by_cost[left_cost]:
                for right in by_cost[cost - left_cost]:
                    children = sorted((best[left][1], best[right][1]), key=repr)
                    for operation, mask in (("and", left & right), ("or", left | right)):
                        if mask in best:
                            continue
                        ast = (operation, *children)
                        if mask not in candidates or repr(ast) < repr(candidates[mask]):
                            candidates[mask] = ast
        for mask, ast in sorted(candidates.items()):
            best[mask] = (cost, ast)
            by_cost[cost].append(mask)
    return {mask: (cost, ast) for mask, (cost, ast) in best.items() if mask}


def depends_on_all_atoms(mask, atoms):
    return all(any(bool(mask & (1 << w)) != bool(mask & (1 << (w ^ (1 << atom))))
                   for w in range(1 << atoms)) for atom in range(atoms))


def render_formula(ast, ood=False):
    if ast[0] == "atom":
        return ("red", "blue", "green", "white")[ast[1]] + (" lamp is lit" if ood else " is on")
    if ast[0] == "not":
        child = render_formula(ast[1], ood)
        return f"not ({child})" if ood else f"(not {child})"
    left, right = (render_formula(child, ood) for child in ast[1:])
    if ood:
        connective = "both" if ast[0] == "and" else "either"
        return f"({connective} {left} {ast[0]} {right})"
    return f"({left} {ast[0]} {right})"


def nli_prompt(premise, hypothesis, style, ood):
    rule = "A: necessarily true. B: necessarily false. C: possibly true and possibly false."
    if ood:
        stems = [
            f"Lamps vary independently; 'or' includes both. Assume {premise}. Judge: {hypothesis}.",
            f"Each lamp is a Boolean fact; 'or' is inclusive. In all settings where {premise}, classify {hypothesis}.",
        ]
    else:
        stems = [
            f"Switches are independent; 'or' is inclusive. Premise: {premise}. Hypothesis: {hypothesis}.",
            f"Treat switches as independent Boolean facts; 'or' is inclusive. Given {premise}, judge {hypothesis}.",
        ]
    return stems[style % 2] + "\n" + rule + "\nAnswer with one letter:"


def quotas(total, labels):
    return {label: total // len(labels) + (i < total % len(labels)) for i, label in enumerate(labels)}


def make_nli(rng):
    data, truth, used = {}, {}, set()
    id_pool = formula_pool(3, 3)
    ood_pool = {mask: value for mask, value in formula_pool(4, 4).items()
                if value[0] == 4 and depends_on_all_atoms(mask, 4)}
    buckets = {}
    for name, pool in (("id", id_pool), ("ood", ood_pool)):
        by_label = defaultdict(list)
        for p in sorted(pool):
            for h in sorted(pool):
                by_label[nli_label(p, h)].append((p, h))
        for rows in by_label.values():
            rng.shuffle(rows)
        buckets[name] = by_label
    for split, count in CONFIG["source_bundles_per_task"].items():
        ood, atoms = split == "ood", 4 if split == "ood" else 3
        pool, source_buckets = (ood_pool, buckets["ood"]) if ood else (id_pool, buckets["id"])
        rows = []
        for label, size in quotas(count, LABELS).items():
            taken = 0
            while taken < size:
                p, h = source_buckets[label].pop()
                signature = f"nli:{atoms}:{min(p, h)}:{max(p, h)}"
                if signature in used:
                    continue
                used.add(signature)
                idx = len(rows)
                source_id = f"nli-{split}-{idx:04d}"
                p_ast, h_ast = pool[p][1], pool[h][1]
                p_text, h_text = render_formula(p_ast, ood), render_formula(h_ast, ood)
                style = rng.randrange(2)
                variants = []
                for name, x, y in (("base", p_text, h_text), ("swap", h_text, p_text)):
                    variants.append({"id": f"{source_id}:{name}", "kind": name,
                                     "prompt": nli_prompt(x, y, style, ood),
                                     "answer_labels": list(LABELS),
                                     "supervised_label": None, "allowed_labels": None})
                row = {"id": source_id, "task": "nli", "split": split,
                       "source_signature": digest(signature), "supervised": False,
                       "gold_label": None, "variants": variants}
                rows.append(row)
                truth[source_id] = {"label": label, "variant_labels": [label, nli_label(h, p)],
                                    "atoms": atoms, "premise_mask": p, "hypothesis_mask": h,
                                    "premise_ast": p_ast, "hypothesis_ast": h_ast,
                                    "literal_leaves": [pool[p][0], pool[h][0]],
                                    "source_signature_raw": signature}
                taken += 1
        rng.shuffle(rows)
        data[split] = rows
    return data, truth, {"id_semantic_formula_count": len(id_pool), "ood_semantic_formula_count": len(ood_pool)}


def expression_signature(op, a, b, c=None):
    if op in ("add", "multiply", "add_then_multiply", "multiply_then_add"):
        a, b = sorted((a, b))
    return (op, a, b) if c is None else (op, a, b, c)


def arithmetic_answer(signature):
    op, a, b, *rest = signature
    if op == "add": return a + b
    if op == "subtract": return a - b
    if op == "multiply": return a * b
    c = rest[0]
    if op == "add_then_multiply": return (a + b) * c
    if op == "multiply_then_add": return a * b + c
    if op == "subtract_then_multiply": return (a - b) * c
    raise ValueError(signature)


def arithmetic_text(signature, style):
    op, a, b, *rest = signature
    if op == "add":
        return [f"What is {a} plus {b}?", f"Find the sum of {a} and {b}."][style]
    if op == "subtract":
        return [f"What is {a} minus {b}?", f"Subtract {b} from {a}."][style]
    if op == "multiply":
        return [f"What is {a} times {b}?", f"Find the product of {a} and {b}."][style]
    c = rest[0]
    if op == "add_then_multiply":
        return [f"Add {a} to {b}, then multiply the result by {c}.", f"Compute ({a} + {b}) * {c}."][style]
    if op == "multiply_then_add":
        return [f"Multiply {a} by {b}, then add {c}.", f"Compute ({a} * {b}) + {c}."][style]
    if op == "subtract_then_multiply":
        return [f"Take {b} away from {a}, then multiply by {c}.", f"Compute ({a} - {b}) * {c}."][style]
    raise ValueError(signature)


def make_mcqa(rng):
    data, truth, used = {}, {}, set()
    for split, count in CONFIG["source_bundles_per_task"].items():
        ood = split == "ood"
        operators = CONFIG["mcqa"]["ood_operators" if ood else "id_operators"]
        low, high = CONFIG["mcqa"]["ood_main_operand_range" if ood else "id_operand_range"]
        rows = []
        for idx in range(count):
            op = operators[idx % len(operators)]
            while True:
                a, b = rng.randint(low, high), rng.randint(low, high)
                c = rng.randint(2, 7) if ood else None
                signature = expression_signature(op, a, b, c)
                if signature not in used:
                    used.add(signature)
                    break
            answer = arithmetic_answer(signature)
            position, numeric_rank, semantic_id = idx % 4, (idx // 4) % 4, (idx // 16) % 4
            offsets = rng.sample(range(1, (25 if ood else 12) + 1), 3)
            incorrect = [answer - offset if j < numeric_rank else answer + offset
                         for j, offset in enumerate(offsets)]
            rng.shuffle(incorrect)
            canonical_values = incorrect[:]
            canonical_values.insert(semantic_id, answer)
            other_ids = [i for i in range(4) if i != semantic_id]
            rng.shuffle(other_ids)
            base_ids = other_ids[:]
            base_ids.insert(position, semantic_id)
            source_id = f"mcqa-{split}-{idx:04d}"
            question = arithmetic_text(signature, rng.randrange(2))
            variants, targets = [], []
            for shift in range(4):
                option_ids = base_ids[shift:] + base_ids[:shift]
                choices = [canonical_values[i] for i in option_ids]
                target = MC_LABELS[option_ids.index(semantic_id)]
                targets.append(target)
                prompt = question + "\n" + " ".join(f"{label}: {value}." for label, value in zip(MC_LABELS, choices))
                prompt += "\nAnswer with one letter:"
                variants.append({"id": f"{source_id}:perm{shift}", "kind": "base" if shift == 0 else "permutation",
                                 "prompt": prompt, "answer_labels": list(MC_LABELS),
                                 "option_ids": option_ids, "option_values": choices,
                                 "base_to_variant": [option_ids.index(i) for i in base_ids],
                                 "supervised_label": None, "allowed_labels": None})
            rows.append({"id": source_id, "task": "mcqa", "split": split,
                         "source_signature": digest(signature), "supervised": False,
                         "gold_label": None, "variants": variants})
            truth[source_id] = {"label": targets[0], "variant_labels": targets,
                                "answer": answer, "correct_option_id": semantic_id,
                                "numeric_rank": numeric_rank, "expression": signature}
        rng.shuffle(rows)
        data[split] = rows
    return data, truth


def apply_supervision(rows, truth, task, seed):
    rng = random.Random(seed)
    labels = LABELS if task == "nli" else MC_LABELS
    by_label = {label: [row for row in rows if truth[row["id"]]["label"] == label] for label in labels}
    for label, count in quotas(CONFIG["supervised_sources_per_task"], labels).items():
        selected = rng.sample(by_label[label], count)
        for row in selected:
            row["supervised"] = True
            row["gold_label"] = label
            row["variants"][0]["supervised_label"] = label
            if task == "nli":
                row["variants"][1]["allowed_labels"] = CONFIG["nli"]["swap_allowed"][label][:]
            else:
                correct_id = row["variants"][0]["option_ids"][MC_LABELS.index(label)]
                for variant in row["variants"][1:]:
                    variant["supervised_label"] = MC_LABELS[variant["option_ids"].index(correct_id)]


def training_view(data):
    """Strict allow-list: no evaluator oracle, expressions, truth masks or hidden labels.

    NLI unlabeled swaps have allowed_labels=None: their allowed set C vs non-C
    cannot be inferred without the base label.  MCQA mappings need no gold.
    """
    allowed_row_keys = ("id", "task", "supervised", "gold_label")
    allowed_variant_keys = ("id", "kind", "prompt", "answer_labels", "supervised_label",
                            "allowed_labels", "option_ids", "base_to_variant")
    result = []
    for task in ("nli", "mcqa"):
        for row in data["splits"]["train"][task]:
            cleaned = {key: row[key] for key in allowed_row_keys}
            cleaned["variants"] = [{key: variant[key] for key in allowed_variant_keys if key in variant}
                                   for variant in row["variants"]]
            result.append(cleaned)
    return result


def audit(data, oracle):
    checks, stats = {}, {}
    all_truth = oracle["sources"]
    exhaustive = Counter()
    failure = []
    for p in range(1, 256):
        for h in range(1, 256):
            forward, reverse = nli_label(p, h), nli_label(h, p)
            exhaustive[f"{forward}->{reverse}"] += 1
            if reverse not in CONFIG["nli"]["swap_allowed"][forward]:
                failure.append([p, h])
    checks["swap_rule_all_65025_nonempty_three_atom_truth_set_pairs"] = not failure
    checks["entailment_reverse_is_not_uniquely_determined"] = exhaustive["A->A"] > 0 and exhaustive["A->C"] > 0
    checks["neutral_reverse_is_not_uniquely_determined"] = exhaustive["C->A"] > 0 and exhaustive["C->C"] > 0
    stats["exhaustive_swap_counts"] = dict(sorted(exhaustive.items()))
    stats["exhaustive_domain"] = "All 255 nonempty subsets of 8 worlds (3 Boolean atoms), ordered pairs; not all 4-atom truth sets."
    split_signatures = {split: set() for split in data["splits"]}
    prompts_by_split = {split: set() for split in data["splits"]}
    source_ids = []
    label_counts, supervised_counts, rank_counts, option_counts = {}, {}, {}, {}
    samples_ok, mappings_ok, supervision_ok, independent_truth_ok, ood_ok = True, True, True, True, True
    ambiguous_generated = Counter()
    for split, tasks in data["splits"].items():
        for task, rows in tasks.items():
            label_counts[f"{split}/{task}"] = dict(Counter(all_truth[row["id"]]["label"] for row in rows))
            supervised_counts[f"{split}/{task}"] = dict(Counter(row["gold_label"] for row in rows if row["supervised"]))
            samples_ok &= len(rows) == CONFIG["source_bundles_per_task"][split]
            counts = list(label_counts[f"{split}/{task}"].values())
            samples_ok &= max(counts) - min(counts) <= 1
            for row in rows:
                source_ids.append(row["id"])
                split_signatures[split].add(row["source_signature"])
                prompts_by_split[split].update(digest(v["prompt"]) for v in row["variants"])
                truth = all_truth[row["id"]]
                if not row["supervised"]:
                    supervision_ok &= row["gold_label"] is None
                    supervision_ok &= all(v["supervised_label"] is None and v["allowed_labels"] is None for v in row["variants"])
                else:
                    supervision_ok &= split == "train" and row["gold_label"] == truth["label"]
                if task == "nli":
                    p = truth_mask(truth["premise_ast"], truth["atoms"])
                    h = truth_mask(truth["hypothesis_ast"], truth["atoms"])
                    independent_truth_ok &= p == truth["premise_mask"] and h == truth["hypothesis_mask"] and p > 0 and h > 0
                    independent_truth_ok &= [nli_label(p, h), nli_label(h, p)] == truth["variant_labels"]
                    if split == "ood":
                        ood_ok &= truth["atoms"] == 4 and truth["literal_leaves"] == [4, 4]
                        ood_ok &= depends_on_all_atoms(p, 4) and depends_on_all_atoms(h, 4)
                    if row["supervised"]:
                        allowed = row["variants"][1]["allowed_labels"]
                        supervision_ok &= allowed == CONFIG["nli"]["swap_allowed"][row["gold_label"]]
                        supervision_ok &= truth["variant_labels"][1] in allowed
                        supervision_ok &= row["variants"][1]["supervised_label"] is None
                        ambiguous_generated["non_unique_allowed" if len(allowed) == 2 else "unique_allowed"] += 1
                else:
                    independent_truth_ok &= arithmetic_answer(truth["expression"]) == truth["answer"]
                    variants = row["variants"]
                    mappings_ok &= len({tuple(v["option_ids"]) for v in variants}) == 4
                    mappings_ok &= len(set(truth["variant_labels"])) == 4
                    for j, variant in enumerate(variants):
                        mappings_ok &= len(set(variant["option_values"])) == 4
                        label_position = MC_LABELS.index(truth["variant_labels"][j])
                        mappings_ok &= variant["option_values"][label_position] == truth["answer"]
                        mappings_ok &= variant["option_ids"][label_position] == truth["correct_option_id"]
                        mappings_ok &= variant["base_to_variant"] == [variant["option_ids"].index(i) for i in variants[0]["option_ids"]]
                        if row["supervised"]:
                            supervision_ok &= variant["supervised_label"] == truth["variant_labels"][j]
            if task == "mcqa":
                rank_counts[split] = dict(Counter(str(all_truth[r["id"]]["numeric_rank"]) for r in rows))
                option_counts[split] = dict(Counter(str(all_truth[r["id"]]["correct_option_id"]) for r in rows))
    intersections = {}
    prompt_intersections = {}
    for x, y in itertools.combinations(data["splits"], 2):
        intersections[f"{x}/{y}"] = len(split_signatures[x] & split_signatures[y])
        prompt_intersections[f"{x}/{y}"] = len(prompts_by_split[x] & prompts_by_split[y])
    checks.update({
        "all_source_sizes_and_base_label_balances": bool(samples_ok),
        "unique_source_ids": len(source_ids) == len(set(source_ids)),
        "no_cross_split_source_signature_overlap": not any(intersections.values()),
        "no_cross_split_prompt_overlap_including_variants": not any(prompt_intersections.values()),
        "independent_truth_table_and_arithmetic_recomputation": bool(independent_truth_ok),
        "ood_all_four_atoms_and_more_complex_than_id": bool(ood_ok),
        "all_mcqa_permutations_and_gold_transport": bool(mappings_ok),
        "no_hidden_gold_or_unjustified_nli_allowed_masks_in_data": bool(supervision_ok),
        "exact_102_supervised_sources_per_task": all(sum(supervised_counts[f"train/{task}"].values()) == 102 for task in ("nli", "mcqa")),
        "mcqa_numeric_rank_and_option_identity_balanced": all(max(c.values()) - min(c.values()) <= 1 for c in list(rank_counts.values()) + list(option_counts.values())),
        "all_gold_entries_accounted_for": set(source_ids) == set(all_truth),
    })
    stats.update({"label_counts": label_counts, "supervised_label_counts": supervised_counts,
                  "mcqa_correct_numeric_rank_counts": rank_counts, "mcqa_correct_semantic_id_counts": option_counts,
                  "split_signature_intersections": intersections, "split_prompt_intersections": prompt_intersections,
                  "supervised_nli_swap_allowed_cardinality_counts": dict(ambiguous_generated),
                  "source_bundles": len(source_ids), "variants": sum(len(r["variants"]) for tasks in data["splits"].values() for rows in tasks.values() for r in rows)})
    dummy = {}
    for split, tasks in data["splits"].items():
        nli_rows, qa_rows = tasks["nli"], tasks["mcqa"]
        dummy[split] = {
            "nli_constant_label": {label: {"swap_consistency": 1.0,
                "base_accuracy": sum(all_truth[r["id"]]["label"] == label for r in nli_rows) / len(nli_rows)} for label in LABELS},
            "mcqa_uniform_conditional_distribution": {"mapped_js_divergence": 0.0, "expected_accuracy": 0.25},
            "mcqa_constant_semantic_option_0": {"equivariance_consistency": 1.0,
                "accuracy": sum(all_truth[r["id"]]["correct_option_id"] == 0 for r in qa_rows) / len(qa_rows)},
            "mcqa_always_position_A": {"base_accuracy": sum(all_truth[r["id"]]["label"] == "A" for r in qa_rows) / len(qa_rows),
                "base_to_nonidentity_variant_consistency": 0.0},
        }
    return {"all_passed": all(checks.values()), "checks": checks, "statistics": stats,
            "consistency_gaming_diagnostics": dummy,
            "diagnostic_caveat": "Consistency alone permits vacuous/constant solutions; evaluate accuracy, invalid full-vocabulary outputs, and teacher confidence separately."}


def generate(output_dir=DEFAULT_OUT):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / name for name in ("data.json", "oracle.json", "data_manifest.json")}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Frozen artifacts already exist; inspect them instead of regenerating or overwriting")
    rng = random.Random(CONFIG["seed"])
    nli, nli_truth, pool_stats = make_nli(rng)
    mcqa, qa_truth = make_mcqa(rng)
    truth = {**nli_truth, **qa_truth}
    apply_supervision(nli["train"], truth, "nli", CONFIG["seed"] + 101)
    apply_supervision(mcqa["train"], truth, "mcqa", CONFIG["seed"] + 202)
    data = {"schema_version": 1, "config_sha256": digest(CONFIG),
            "answer_label_map": {"nli": CONFIG["nli"]["label_map"], "mcqa": "A/B/C/D are presented option positions"},
            "splits": {split: {"nli": nli[split], "mcqa": mcqa[split]} for split in CONFIG["source_bundles_per_task"]}}
    oracle = {"schema_version": 1, "usage": "Evaluator-only. Do not load in trainer or use hidden labels to construct training constraints.",
              "config_sha256": digest(CONFIG), "sources": truth}
    report = audit(data, oracle)
    if not report["all_passed"]:
        raise AssertionError(report["checks"])
    for filename, obj in (("data.json", data), ("oracle.json", oracle)):
        paths[filename].write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "frozen": True,
                "generation_config": CONFIG, "config_sha256": digest(CONFIG),
                "generator_sha256": file_hash(__file__),
                "artifacts_sha256": {name: file_hash(paths[name]) for name in ("data.json", "oracle.json")},
                "formula_pool_statistics": pool_stats, "audit": report}
    paths["data_manifest.json"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        data = json.loads((args.output_dir / "data.json").read_text(encoding="utf-8"))
        oracle = json.loads((args.output_dir / "oracle.json").read_text(encoding="utf-8"))
        result = audit(data, oracle)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result["all_passed"] else 1)
    result = generate(args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "all_passed": result["audit"]["all_passed"],
                      "config_sha256": result["config_sha256"], "statistics": result["audit"]["statistics"]}, indent=2))
