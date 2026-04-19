"""
Grid search utilities for bipartite matching hyperparameter optimization.
"""

import numpy as np
import pandas as pd
from bipartite_matching import (
    build_full_bipartite_graph, solve_matching, build_reduced_graph_by_matching,
    extract_index_from_node_name, is_eps_node,
    get_bipartite_edges, get_word_edges, get_epsilon_edges,
)
from word_similarity_calculator import WordSimilarityCalculator


def map_solver_matching_to_idx_based_alignment(solver_matching: dict, G) -> dict:
    """Convert graph bipartite alignment solver output to ground-truth index-based format (ref_idx -> hyp_idx | None)."""
    M = build_reduced_graph_by_matching(G, solver_matching)
    bipartite_edges = get_bipartite_edges(M)
    word_edges = get_word_edges(M, bipartite_edges)
    eps_edges = get_epsilon_edges(M, bipartite_edges)

    idx_based_solver_alignment = {}
    for ref_node, hyp_node in word_edges:
        idx_based_solver_alignment[extract_index_from_node_name(ref_node)] = extract_index_from_node_name(hyp_node)
    for ref_node, hyp_node in eps_edges:
        if not is_eps_node(M.nodes[ref_node]):
            idx_based_solver_alignment[extract_index_from_node_name(ref_node)] = None
    return idx_based_solver_alignment


def evaluate_alignment(idx_based_solver_alignment: dict, ground_truth_alignment: dict) -> dict:
    """Compute precision, recall, F1 on word-to-word matches (epsilon = no match)."""
    gt_edges = {(r, h) for r, h in ground_truth_alignment.items() if h is not None}
    solver_edges = {(r, h) for r, h in idx_based_solver_alignment.items() if h is not None}

    tp = len(gt_edges & solver_edges)  # true positives, correct matchings (intersection of both sets)
    fp = len(solver_edges - gt_edges)  # false positives, wrong matchings
    fn = len(gt_edges - solver_edges)  # false negatives, missed matchings

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if not gt_edges else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def grid_search(entries, alphas, lambdas, lexical_normalization_modes):
    """Run grid search over all (alpha, lambda, lexical_normalization_modes) combinations. Returns a DataFrame."""
    results = []
    for alpha in alphas:
        for lambda_ in lambdas:
            for use_global_lexical_normalization in lexical_normalization_modes:
                precisions, recalls, f1s = [], [], []
                for entry in entries:
                    ref, hyp = entry["ref"], entry["hyp"]
                    max_word_len = max(len(w) for w in ref + hyp) if use_global_lexical_normalization else None

                    similarity_calculator = WordSimilarityCalculator(
                        sent_len=max(len(ref), len(hyp)),
                        alpha=alpha, lambda_=lambda_,
                        use_global_lexical_normalization=use_global_lexical_normalization, max_word_len=max_word_len,
                    )
                    G = build_full_bipartite_graph(ref, hyp, similarity_calculator)
                    solver_matching = solve_matching(G)
                    idx_based_solver_alignment = map_solver_matching_to_idx_based_alignment(solver_matching, G)

                    scores = evaluate_alignment(idx_based_solver_alignment, entry["alignment"])
                    precisions.append(scores["precision"])
                    recalls.append(scores["recall"])
                    f1s.append(scores["f1"])

                results.append({
                    "alpha": alpha,
                    "lambda": lambda_,
                    "use_global_lexical_normalization": use_global_lexical_normalization,
                    "precision": np.mean(precisions),
                    "recall": np.mean(recalls),
                    "f1": np.mean(f1s),
                })
    return pd.DataFrame(results)


def pivot_f1_grids(df, alphas=None, lambdas=None):
    """Pivot a grid search DataFrame into {use_global_lexical_normalization: 2D array} grids.
    """
    alpha_order = list(alphas) if alphas is not None else sorted(df["alpha"].unique())
    lambda_order = list(lambdas) if lambdas is not None else sorted(df["lambda"].unique())

    grids = {}
    for mode in sorted(df["use_global_lexical_normalization"].unique()):
        subset = df[df["use_global_lexical_normalization"] == mode]
        pivoted = subset.pivot(index="alpha", columns="lambda", values="f1")
        pivoted = pivoted.reindex(index=alpha_order, columns=lambda_order)
        grids[mode] = pivoted.to_numpy()
    return grids


def print_best_f1_summary(df, label=""):
    """Print the best F1 and the tied alpha/lambda values for a grid search DataFrame."""
    best_f1 = df["f1"].max()
    tied = df[df["f1"] == best_f1]
    tied_alphas = sorted(tied["alpha"].unique())
    tied_lambdas = sorted(tied["lambda"].unique())
    print(f"{label}Best F1 = {best_f1:.3f} ({len(tied)} tied)")
    print(f"  α tied: {[f'{a:.2f}' for a in tied_alphas]}")
    print(f"  λ tied: {[f'{l:.2f}' for l in tied_lambdas]}")
