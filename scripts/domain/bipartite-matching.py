import networkx as nx
import numpy as np
from calculations import (
    clean,
    calculate_word_similarity_global,
    calculate_word_similarity_local,
    calculate_position_score,
    calculate_score_weighted,
)

EPSILON_PENALTY = 0.6
COST_SCALE = 1000  # scale float costs to int for network_simplex


def build_bipartite_graph(
    src_words: list[str],
    target_words: list[str],
    max_word_len: int,
    max_sent_len: int,
    alpha: float = 0.5,
) -> nx.DiGraph:
    """Build a weighted bipartite flow network between source (DIT) and target (DAT) words.

    The alignment is modelled as finding the minimum-cost flow in a bipartite
    network G = (W' ∪ V', E) with source s and sink t.

    Each partition is padded with ε-nodes so that both sides have N = m + n nodes,
    enabling a perfect matching where unmatched words flow through ε at a fixed penalty.

    Edge weights represent cost = 1 - similarity (lower = better match).
    All edge capacities are 1 (unit flow).
    """
    # --- Graph ---
    G = nx.DiGraph()

    m = len(src_words)  # |W|, source word count / amount of ε-nodes in V'
    n = len(target_words)  # |V|, target word count / amount of ε-nodes in W'
    N = m + n  # padded partition size

    # --- Nodes ---
    # source
    G.add_node("s", demand=-N)

    # word nodes
    for i, word in enumerate(src_words):
        G.add_node(f"src_{i}", word=word)
    for j, word in enumerate(target_words):
        G.add_node(f"tgt_{j}", word=word)

    # ε-nodes
    for j in range(n):
        G.add_node(f"src_ε_{j}", word="ε")
    for i in range(m):
        G.add_node(f"tgt_ε_{i}", word="ε")

    # sink
    G.add_node("t", demand=N)

    # --- Edges ---
    # edges from s to W'
    for i in range(m):
        G.add_edge("s", f"src_{i}", capacity=1, weight=0)
    for j in range(n):
        G.add_edge("s", f"src_ε_{j}", capacity=1, weight=0)

    # edges from source word nodes to target word nodes
    for i in range(m):
        # edges to target word nodes
        for j in range(n):
            src_word = clean(src_words[i])
            target_word = clean(target_words[j])
            
            cost = calculate_cost(
                src_word=src_word,
                src_position=i,
                target_word=target_word,
                target_position=j,
                global_max_word_length=max_word_len,
                global_max_sentence_length=max_sent_len,
                is_levenshtein_normalization_global=True,
            )
            G.add_edge(f"src_{i}", f"tgt_{j}", capacity=1, weight=int(cost * COST_SCALE), score=1 - cost)
    # edges from source word nodes to target ε-nodes
    for i in range(m):
        for k in range(m):
            G.add_edge(f"src_{i}", f"tgt_ε_{k}", capacity=1, weight=int(EPSILON_PENALTY * COST_SCALE), score=1 - EPSILON_PENALTY)
            
    # edges from source ε-nodes to target word nodes
    for j in range(n):
        for k in range(n):
            G.add_edge(f"src_ε_{j}", f"tgt_{k}", capacity=1, weight=int(EPSILON_PENALTY * COST_SCALE), score=1 - EPSILON_PENALTY)
    # edges from source ε-nodes to target ε-nodes
    for j in range(n):
        for i in range(m):
            G.add_edge(f"src_ε_{j}", f"tgt_ε_{i}", capacity=1, weight=0, score=1)

    # edges from V' to t
    for j in range(n):
        G.add_edge(f"tgt_{j}", "t", capacity=1, weight=0)
    for i in range(m):
        G.add_edge(f"tgt_ε_{i}", "t", capacity=1, weight=0)

    return G


def calculate_cost(
    src_word: str,
    src_position: int,
    target_word: str,
    target_position: int,
    global_max_word_length: int,
    global_max_sentence_length: int,
    is_levenshtein_normalization_global: bool,
) -> float:
    # Calculate word similarity (lexical similarity) using Levenshtein distance, normalized either globally or locally.
    if is_levenshtein_normalization_global:
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
    position_score = calculate_position_score(
        src_index=src_position,
        target_index=target_position,
        global_max_sentence_length=global_max_sentence_length,
    )

    # Combine lexical and positional similarity into a single score using a weighted average.
    score = calculate_score_weighted(word_similarity, position_score, alpha=1)

    return 1 - score  # Convert similarity to cost


def solve_matching(G: nx.DiGraph) -> dict[str, str]:
    """Find optimal minimum-cost flow on the bipartite flow network.
    Returns a dict mapping each src node to its matched tgt node."""
    flow_dict = nx.min_cost_flow(G, weight="weight")

    # Extract matching: for each node w in W', find the matched node v in V' with flow=1
    matching = {}
    for w, neighbors in flow_dict.items():
        for v, flow in neighbors.items():
            if flow == 1 and v.startswith("tgt"):
                matching[w] = v
    return matching


# ── quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dit = clean("Ein Missgeschick nach dem andern traf sie: die Geschirre zerrissen, die Wagen brachen, Pferde und Ochsen fielen oder weigerten den Gehorsam.").split()
    dat = clean("Ein Missgeschick nach dem andern hat sie troffen: die Geschirre sind zerrissen, die Wagen sind brochen, Rose und Ochse sind Gehege oder Hände sich ähm Gehorsam geweigert").split()


    G = build_bipartite_graph(dit, dat, max_word_len=10, max_sent_len=5)
    matching = solve_matching(G)

    print("Matching:")
    for src, tgt in matching.items():
        if src.startswith("src"):
            src_word = G.nodes[src]["word"]
            tgt_word = G.nodes[tgt]["word"]
            edge_data = G.edges[src, tgt]
            print(f"  {src_word:>12} → {tgt_word:<12}  (score: {edge_data['score']:.3f})")
