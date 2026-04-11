import networkx as nx
from calculations import (
    clean,
    calculate_word_similarity_local,
    calculate_word_similarity_global,
    calculate_position_score,
    calculate_score_weighted,
)

COST_SCALE = 1000  # scale float costs to int for network_simplex


def build_bipartite_graph(
    ref_words: list[str],
    hyp_words: list[str],
    max_word_len: int,
    max_sent_len: int,
    is_levenshtein_normalization_global: bool,
    alpha: float = 0.7,
    lambda_: float = 0.3,
) -> nx.DiGraph:
    """Build a weighted bipartite flow network between reference and hypothesis words.

    The alignment is modelled as finding the minimum-cost flow in a bipartite
    network G = (R' ∪ H', E) with source s and sink t.

    Each partition is padded with ε-nodes so that both sides have N = n_r + n_h nodes,
    enabling a perfect matching where unmatched words flow through ε at a fixed penalty.

    Edge weights represent cost = 1 - score (lower = better match).
    All edge capacities are 1 (unit flow).
    """
    # --- Graph ---
    G = nx.DiGraph()

    n_r = len(ref_words)  # |R|, reference word count / amount of ε-nodes in H'
    n_h = len(hyp_words)  # |H|, hypothesis word count / amount of ε-nodes in R'
    N = n_r + n_h  # padded partition size

    # --- Nodes ---
    # source
    G.add_node("s", demand=-N, side="source")

    # word nodes
    for i, word in enumerate(ref_words):
        G.add_node(f"ref_{i}", word=word, side="ref")
    for j, word in enumerate(hyp_words):
        G.add_node(f"hyp_{j}", word=word, side="hyp")

    # ε-nodes
    for j in range(n_h):
        G.add_node(f"ref_ε_{j}", word="ε", side="ref")
    for i in range(n_r):
        G.add_node(f"hyp_ε_{i}", word="ε", side="hyp")

    # sink
    G.add_node("t", demand=N, side="sink")

    # --- Edges ---
    # edges from s to R'
    for i in range(n_r):
        G.add_edge("s", f"ref_{i}", capacity=1, weight=0)
    for j in range(n_h):
        G.add_edge("s", f"ref_ε_{j}", capacity=1, weight=0)

    # edges from ref word nodes to hyp word nodes
    for i in range(n_r):
        for j in range(n_h):
            ref_word = clean(ref_words[i])
            hyp_word = clean(hyp_words[j])

            cost = calculate_cost(
                src_word=ref_word,
                src_position=i,
                target_word=hyp_word,
                target_position=j,
                global_max_word_length=max_word_len,
                global_max_sentence_length=max_sent_len,
                use_gloabal_levenshtein_normalization=is_levenshtein_normalization_global, # TODO rename
                alpha=alpha,
            )
            G.add_edge(
                f"ref_{i}",
                f"hyp_{j}",
                capacity=1,
                weight=int(cost * COST_SCALE), # TODO belongs to calculations (maybe use ints from lev distance & pos gap)
                score=1 - cost,
            )
    # edges from ref word nodes to hyp ε-nodes
    for i in range(n_r):
        for k in range(n_r):
            G.add_edge(
                f"ref_{i}",
                f"hyp_ε_{k}",
                capacity=1,
                weight=int(lambda_ * COST_SCALE),
                score=1 - lambda_,
            )

    # edges from ref ε-nodes to hyp word nodes
    for j in range(n_h):
        for k in range(n_h):
            G.add_edge(
                f"ref_ε_{j}",
                f"hyp_{k}",
                capacity=1,
                weight=int(lambda_ * COST_SCALE),
                score=1 - lambda_,
            )
    # edges from ref ε-nodes to hyp ε-nodes
    for j in range(n_h):
        for i in range(n_r):
            G.add_edge(f"ref_ε_{j}", f"hyp_ε_{i}", capacity=1, weight=0, score=1)

    # edges from H' to t
    for j in range(n_h):
        G.add_edge(f"hyp_{j}", "t", capacity=1, weight=0)
    for i in range(n_r):
        G.add_edge(f"hyp_ε_{i}", "t", capacity=1, weight=0)

    return G


def calculate_cost( # TODO move to calculations, costs over the whole sentence also possible, also rename
    src_word: str,
    src_position: int,
    target_word: str,
    target_position: int,
    global_max_word_length: int,
    global_max_sentence_length: int,
    use_gloabal_levenshtein_normalization: bool, # logic shouldnt be here, maybe object for params
    alpha: float = 0.5,
) -> float:
    # Calculate word similarity (lexical similarity) using Levenshtein distance, normalized either globally or locally. TODO also levels?
    if use_gloabal_levenshtein_normalization:
        word_similarity = calculate_word_similarity_global(
            src_word=src_word,
            target_word=target_word,
            global_max_word_length=global_max_word_length,
        )
    else:
        word_similarity = calculate_word_similarity_local(
            src_word=src_word, target_word=target_word
        )

    # Calculate position similarity (positional similarity) based on distance in sentence, normalized by global max sentence length.
    position_score = calculate_position_score( # TODO maybe in levels (based on neighbourhood)
        src_index=src_position,
        target_index=target_position,
        global_max_sentence_length=global_max_sentence_length,
    )

    # Combine lexical and positional similarity into a single score using a weighted average.
    score = calculate_score_weighted(word_similarity, position_score, alpha=alpha)

    return 1 - score  # Convert similarity to cost


def solve_matching(G: nx.DiGraph) -> dict[str, str]:
    """Find optimal minimum-cost flow on the bipartite flow network.
    Uses Network Simplex algorithm from NetworkX."
    Returns a dict mapping each src node to its matched tgt node."""
    flow_dict = nx.min_cost_flow(G, weight="weight") # TODO hungarian as alternative

    # Extract matching: for each node w in W', find the matched node v in V' with flow=1
    matching = {}
    for w, neighbors in flow_dict.items():
        for v, flow in neighbors.items():
            if flow == 1 and G.nodes[v].get("side") == "hyp":
                matching[w] = v
    return matching
