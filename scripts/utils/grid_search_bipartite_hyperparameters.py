"""
Grid search utilities for bipartite matching hyperparameter optimization.
"""

import networkx as nx
import numpy as np
import pandas as pd
from bipartite_matching import (
    EPS, REFERENCE_PARTITION, HYPOTHESIS_PARTITION,
    build_full_bipartite_graph, solve_matching, build_reduced_graph_by_matching,
    get_node_name, extract_index_from_node_name, is_eps_node,
    get_bipartite_edges,
)
from word_similarity_calculator import WordSimilarityCalculator


def gt_alignment_to_matching(
        alignment: dict[int, int | None],
) -> dict[str, str]:
    """Convert a JSON-style ground-truth alignment dict to a matching dict
    (node names follow the convention in bipartite_matching.py).

    Output is in the same format as solve_matching. None entries route through a
    hyp epsilon node (deletion). Insertions (ref_ε → hyp word) are omitted because
    evaluation only scores ref-side edges to avoid double-counting wrong edges.
    """
    matching = {}
    for ref_idx, hyp_idx in alignment.items():
        ref_node = get_node_name(REFERENCE_PARTITION, ref_idx)
        if hyp_idx is None:
            hyp_node = get_node_name(HYPOTHESIS_PARTITION, ref_idx, eps=True)
        else:
            hyp_node = get_node_name(HYPOTHESIS_PARTITION, hyp_idx)
        matching[ref_node] = hyp_node
    return matching


def evaluate_alignment(M_solver: nx.DiGraph, M_gt: nx.DiGraph) -> float:
    """Compute alignment accuracy: fraction of reference words where the solver's pick
    matches the ground truth (real hyp word or epsilon).

    Only ref-side edges are scored. Hyp-side insertions are fully determined by ref-side
    correctness in a 1-to-1 matching, so they carry no independent signal.
    """
    solver_edges = _ref_side_edges(M_solver)
    gt_edges = _ref_side_edges(M_gt)
    if not gt_edges:
        return 1.0
    return len(solver_edges & gt_edges) / len(gt_edges)


def _ref_side_edges(M: nx.DiGraph) -> set[tuple]:
    """Bipartite edges from real ref words (no insertions); for deletions, the specific hyp epsilon index doesn't matter."""
    edges = set()
    for u, v in get_bipartite_edges(M):
        if is_eps_node(M.nodes[u]):
            continue
        u_key = extract_index_from_node_name(u)
        v_key = EPS if is_eps_node(M.nodes[v]) else extract_index_from_node_name(v)
        edges.add((u_key, v_key))
    return edges


def grid_search(entries, alphas, lambdas, lexical_normalization_modes):
    """Run grid search over all (alpha, lambda, lexical_normalization_modes) combinations. Returns a DataFrame."""
    if not entries:
        raise ValueError("grid_search() requires a non-empty 'entries' list")
    global_max_word_len = max(len(w) for entry in entries for w in entry["reference"] + entry["hypothesis"])

    results = []
    for alpha in alphas:
        for lambda_ in lambdas:
            for use_global_lexical_normalization in lexical_normalization_modes:
                accuracies = []
                for entry in entries:
                    ref, hyp = entry["reference"], entry["hypothesis"]

                    similarity_calculator = WordSimilarityCalculator(
                        sent_len=max(len(ref), len(hyp)),
                        alpha=alpha, lambda_=lambda_,
                        use_global_lexical_normalization=use_global_lexical_normalization,
                        max_word_len=global_max_word_len if use_global_lexical_normalization else None,
                    )
                    G = build_full_bipartite_graph(ref, hyp, similarity_calculator)
                    solver_matching = solve_matching(G)
                    gt_matching = gt_alignment_to_matching(entry["alignment"])

                    M_solver = build_reduced_graph_by_matching(G, solver_matching)
                    M_gt = build_reduced_graph_by_matching(G, gt_matching)

                    accuracies.append(evaluate_alignment(M_solver, M_gt))

                results.append({
                    "alpha": alpha,
                    "lambda": lambda_,
                    "use_global_lexical_normalization": use_global_lexical_normalization,
                    "accuracy": np.mean(accuracies),
                })
    return pd.DataFrame(results)


def pivot_accuracy_grids(df, alphas=None, lambdas=None):
    """Pivot a grid search DataFrame into {use_global_lexical_normalization: 2D array} grids.
    """
    alpha_order = list(alphas) if alphas is not None else sorted(df["alpha"].unique())
    lambda_order = list(lambdas) if lambdas is not None else sorted(df["lambda"].unique())

    grids = {}
    for mode in sorted(df["use_global_lexical_normalization"].unique()):
        subset = df[df["use_global_lexical_normalization"] == mode]
        pivoted = subset.pivot(index="alpha", columns="lambda", values="accuracy")
        pivoted = pivoted.reindex(index=alpha_order, columns=lambda_order)
        grids[mode] = pivoted.to_numpy()
    return grids


def print_best_accuracy_summary(df, label=""):
    """Print the best accuracy and the tied alpha/lambda values for a grid search DataFrame."""
    best_accuracy = df["accuracy"].max()
    tied = df[df["accuracy"] == best_accuracy]
    tied_alphas = sorted(tied["alpha"].unique())
    tied_lambdas = sorted(tied["lambda"].unique())
    print(f"{label}Best accuracy = {best_accuracy:.3f} ({len(tied)} tied)")
    print(f"  α tied: {[f'{a:.2f}' for a in tied_alphas]}")
    print(f"  λ tied: {[f'{l:.2f}' for l in tied_lambdas]}")
