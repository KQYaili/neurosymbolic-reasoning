"""Audit order geometry and graph inference separately from NLI accuracy.

No model is trained and no natural-language labels are read.  The exact
mathematical relation is coordinate dominance, not a thresholded score.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable

import torch


def order_energy(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Return E(u,v) = sum_k max(0, v_k-u_k)**2 along the last axis.

    This operation supports broadcasting and autograd.  E(u,v)=0 means
    u >= v coordinatewise.  Positive thresholds do not preserve transitivity.
    """
    return torch.relu(v - u).square().sum(dim=-1)


def transitive_closure_with_witnesses(
    edges: Iterable[tuple[str, str]], nodes: Iterable[str] = ()
) -> dict:
    """Compute reflexive reachability, shortest path witnesses, and SCCs.

    Nodes are stable string identifiers.  Mutual reachability defines an
    equivalence relation on this graph; its SCC quotient is a DAG.  Calling
    these components *semantic* equivalences additionally requires sound
    input edges.  Reachability cannot establish that a learned edge is true.
    """
    observed = set()
    node_set = set(nodes)
    for edge in edges:
        if not isinstance(edge, (tuple, list)) or len(edge) != 2:
            raise ValueError("Each directed edge must contain two node IDs")
        source, target = edge
        observed.add((source, target))
        node_set.update((source, target))
    if any(not isinstance(node, str) for node in node_set):
        raise TypeError("Graph node IDs must be strings")
    ordered_nodes = sorted(node_set)
    adjacency = {node: [] for node in ordered_nodes}
    for source, target in sorted(observed):
        adjacency[source].append(target)

    paths = {}
    reachable = {}
    for source in ordered_nodes:
        parents = {source: None}
        queue = deque([source])
        while queue:
            current = queue.popleft()
            for target in adjacency[current]:
                if target not in parents:
                    parents[target] = current
                    queue.append(target)
        reachable[source] = set(parents)
        for target in sorted(parents):
            path, current = [], target
            while current is not None:
                path.append(current)
                current = parents[current]
            paths[source, target] = list(reversed(path))

    # Once reachability is available, SCCs are exactly its mutually reachable
    # equivalence classes.  This also handles isolated nodes and self-loops.
    remaining = set(ordered_nodes)
    components = []
    while remaining:
        first = min(remaining)
        members = sorted(
            node for node in remaining
            if node in reachable[first] and first in reachable[node]
        )
        components.append(members)
        remaining.difference_update(members)
    component_of = {
        node: component
        for component, members in enumerate(components)
        for node in members
    }
    quotient_edges = sorted({
        (component_of[source], component_of[target])
        for source, target in observed
        if component_of[source] != component_of[target]
    })
    quotient_adjacency = {i: [] for i in range(len(components))}
    indegree = {i: 0 for i in range(len(components))}
    for source, target in quotient_edges:
        quotient_adjacency[source].append(target)
        indegree[target] += 1
    ready = deque(i for i in range(len(components)) if indegree[i] == 0)
    topological_order = []
    while ready:
        source = ready.popleft()
        topological_order.append(source)
        for target in quotient_adjacency[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    return {
        "nodes": ordered_nodes,
        "observed_edges": [list(edge) for edge in sorted(observed)],
        "reflexive_closure": [
            {"source": source, "target": target, "path": path,
             "observed_edge": (source, target) in observed}
            for (source, target), path in sorted(paths.items())
        ],
        "strongly_connected_components": components,
        "component_of": component_of,
        "quotient_edges": [list(edge) for edge in quotient_edges],
        "quotient_topological_order": topological_order,
        "quotient_is_dag": len(topological_order) == len(components),
        "semantics": "Reflexive graph reachability; input edge truth is assumed.",
    }


def _closure_reference(nodes: list[str], edges: list[list[str]]) -> set:
    """Independent Floyd-Warshall reference for the small audit graph."""
    relation = {(node, node) for node in nodes} | {tuple(edge) for edge in edges}
    for middle in nodes:
        for source in nodes:
            for target in nodes:
                if (source, middle) in relation and (middle, target) in relation:
                    relation.add((source, target))
    return relation


def run_logic_audit(output_dir: str | Path) -> dict:
    """Write logic_audit.json and return its report; raise if a check fails.

    Exact checks use integer coordinates. Randomized numerical checks use a
    private CPU generator, leaving training RNG state and devices untouched.
    Proof explanations are separate from finite implementation checks.
    """
    seed = 20260915
    training_rng_before = torch.random.get_rng_state().clone()
    checks = {}
    grid = torch.tensor(list(itertools.product(range(4), repeat=3)),
                        dtype=torch.int64)
    energies = order_energy(grid[:, None, :], grid[None, :, :])
    relation = energies == 0
    identity = torch.eye(len(grid), dtype=torch.bool)
    equality = (grid[:, None, :] == grid[None, :, :]).all(dim=-1)
    transitive_violations = 0
    for middle in range(len(grid)):
        antecedent = relation[:, middle, None] & relation[middle, None, :]
        transitive_violations += int((antecedent & ~relation).sum())
    checks["exact_energy_characterizes_coordinate_order"] = bool(torch.equal(
        relation, (grid[:, None, :] >= grid[None, :, :]).all(dim=-1)))
    checks["exact_reflexivity"] = bool(torch.all(relation[identity]))
    checks["exact_coordinate_antisymmetry"] = bool(torch.equal(
        relation & relation.T, equality))
    checks["exact_transitivity"] = transitive_violations == 0

    a = torch.tensor([0.0], dtype=torch.float64)
    b = torch.tensor([0.2], dtype=torch.float64)
    c = torch.tensor([0.4], dtype=torch.float64)
    e_ab, e_bc, e_ac = [float(order_energy(x, y))
                       for x, y in ((a, b), (b, c), (a, c))]
    threshold = 0.05
    checks["positive_threshold_transitivity_counterexample"] = (
        e_ab < threshold and e_bc < threshold and e_ac >= threshold)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    sample_count, dimension = 4096, 17
    triples = torch.rand((3, sample_count, dimension), generator=generator,
                         dtype=torch.float64) * 5
    u, v, w = triples.unbind(0)
    left = order_energy(u, w).sqrt()
    right = order_energy(u, v).sqrt() + order_energy(v, w).sqrt()
    tolerance = 1e-12
    excess = left - right
    checks["randomized_directed_triangle_bound"] = bool(torch.all(
        excess <= tolerance))
    checks["tight_directed_triangle_bound"] = (
        e_ac ** 0.5 <= e_ab ** 0.5 + e_bc ** 0.5 + tolerance)

    grad_u = torch.tensor([[2., 0., 4.], [1., 3., 2.]],
                          dtype=torch.float64, requires_grad=True)
    grad_v = torch.tensor([1., 2., 4.], dtype=torch.float64, requires_grad=True)
    du, dv = torch.autograd.grad(order_energy(grad_u, grad_v).sum(),
                                 (grad_u, grad_v))
    expected_du = torch.tensor([[0., -4., 0.], [0., 0., -4.]],
                               dtype=torch.float64)
    expected_dv = torch.tensor([0., 4., 4.], dtype=torch.float64)
    checks["broadcast_autograd_matches_analytic_derivatives"] = (
        bool(torch.equal(du, expected_du)) and bool(torch.equal(dv, expected_dv)))

    graph = transitive_closure_with_witnesses(
        [("A", "B"), ("B", "C"), ("A", "A_equiv"),
         ("A_equiv", "A")],
        nodes=["A", "A_equiv", "B", "C", "D"],
    )
    closure = {(edge["source"], edge["target"]): edge
               for edge in graph["reflexive_closure"]}
    observed = {tuple(edge) for edge in graph["observed_edges"]}
    held_out = ("A", "C")
    checks["held_out_edge_is_not_observed"] = held_out not in observed
    checks["held_out_edge_has_observed_path_witness"] = (
        closure[held_out]["path"] == ["A", "B", "C"])
    checks["all_path_witnesses_use_only_observed_edges"] = all(
        path[0] == source and path[-1] == target and
        all(edge in observed for edge in zip(path, path[1:]))
        for (source, target), item in closure.items()
        for path in [item["path"]]
    )
    checks["graph_closure_matches_independent_reference"] = (
        set(closure) == _closure_reference(graph["nodes"], graph["observed_edges"]))
    checks["graph_does_not_invent_reverse_or_unrelated_edges"] = (
        ("B", "A") not in closure and ("A", "D") not in closure)
    checks["scc_handles_mutual_reachability"] = (
        ["A", "A_equiv"] in graph["strongly_connected_components"])
    checks["scc_quotient_is_dag"] = graph["quotient_is_dag"]
    checks["empty_graph_supported"] = (
        transitive_closure_with_witnesses([])["reflexive_closure"] == [])
    isolated = transitive_closure_with_witnesses([("X", "X")], ["Y"])
    checks["self_loop_and_isolated_node_supported"] = (
        isolated["strongly_connected_components"] == [["X"], ["Y"]]
        and isolated["quotient_edges"] == [])
    checks["training_cpu_rng_state_unchanged"] = bool(torch.equal(
        training_rng_before, torch.random.get_rng_state()))

    report = {
        "schema_version": 1,
        "scope": "Mathematical derivations and deterministic implementation checks",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "seed": seed,
        "all_passed": all(checks.values()),
        "checks": checks,
        "proof_notes": {
            "energy": "A sum of nonnegative squares is zero iff each v_k-u_k <= 0.",
            "reflexivity": "Every coordinate satisfies u_k >= u_k, so E(u,u)=0.",
            "antisymmetry": "u>=v and v>=u imply equality in every coordinate.",
            "transitivity": "u>=v and v>=w imply u>=w coordinatewise.",
            "triangle_bound": (
                "relu(w-u) <= relu(v-u)+relu(w-v) coordinatewise; monotonicity "
                "and the triangle inequality of the Euclidean norm give "
                "sqrt(E(u,w)) <= sqrt(E(u,v))+sqrt(E(v,w))."),
            "scc_quotient": (
                "Mutual reachability is an equivalence relation. A cycle between "
                "distinct quotient components would make those components mutually "
                "reachable, contradicting their separation."),
        },
        "exact_finite_grid": {
            "coordinates": "{0,1,2,3}^3", "vectors": len(grid),
            "ordered_pairs_checked": len(grid) ** 2,
            "triples_checked": len(grid) ** 3,
            "transitivity_violations": transitive_violations,
            "numeric_type": "int64; no floating threshold",
        },
        "threshold_counterexample": {
            "A": a.tolist(), "B": b.tolist(), "C": c.tolist(),
            "E_AB": e_ab, "E_BC": e_bc, "E_AC": e_ac,
            "threshold": threshold, "decision_rule": "E(x,y) < threshold",
            "conclusion": "The positive-threshold relation is not transitive.",
        },
        "randomized_triangle_check": {
            "trials": sample_count, "dimension": dimension,
            "dtype": "float64", "absolute_tolerance": tolerance,
            "maximum_lhs_minus_rhs": float(excess.max()),
            "note": "Random checks test implementation; the proof is above.",
        },
        "discrete_graph": graph,
        "held_out_edge_demo": {
            "edge": list(held_out), "path_witness": closure[held_out]["path"],
            "evaluation_type": "Logical closure of supplied edges over seen node IDs",
            "training_performed": False,
        },
        "limitations": [
            "No empirical SNLI or other natural-language accuracy is measured here.",
            "Theorems concern the exact vector order and graph relation, not a learned encoder's semantic correctness.",
            "Graph closure can propagate incorrect learned edges; a path witness proves only derivability from supplied edges.",
            "An unseen edge among known nodes is not zero-shot generalization to unseen sentences.",
            "Sentence entailment is a preorder; quotienting by semantic equivalence yields a partial order.",
            "Graph SCC equivalence need not be real semantic equivalence when edges are noisy.",
            "Order energy alone distinguishes entailment from non-entailment, not contradiction from neutral.",
            "Natural-language inference does not by itself identify intervention effects.",
        ],
        "primary_sources": [
            {"title": "Order-Embeddings of Images and Language (ICLR 2016)",
             "url": "https://arxiv.org/pdf/1511.06361",
             "relevance": "Reversed product order, order-violation energy, and binary SNLI adaptation."},
            {"title": "Stanford Natural Language Inference Corpus",
             "url": "https://nlp.stanford.edu/projects/snli/",
             "relevance": "SNLI's entailment, contradiction, and neutral labels."},
        ],
    }
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / "logic_audit.json"
    report["output_path"] = str(output_path)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination,
        prefix="logic_audit_", suffix=".tmp", delete=False,
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary_path = handle.name
    os.replace(temporary_path, output_path)
    if not report["all_passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Logic audit failed: {failed}; report: {output_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run_logic_audit(args.output_dir)
    print(json.dumps({"all_passed": result["all_passed"],
                      "checks_passed": sum(result["checks"].values()),
                      "output_path": result["output_path"]}, indent=2))
