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
EPSILON_SYMBOL = "ε"


def build_bipartite_graph(
    src_words: list[str],
    target_words: list[str],
    max_word_len: int,
    max_sent_len: int,
    alpha: float = 0.5,
) -> nx.Graph:
    """Build a weighted bipartite graph between source (DIT) and target (DAT) words.
    TODO: add correct mathematical definition here.

    Each partition is padded with ε-nodes (one per word in the opposite partition)
    so that both sides have (m + n) nodes, enabling a full matching that can
    leave words unmatched (mapped to ε) at a fixed penalty cost.

    Edge weights represent cost = 1 - similarity (lower = better match).
    Edges to ε-nodes carry a fixed EPSILON_PENALTY cost.
    """
    # Initialize an empty bipartite graph
    G = nx.Graph()

    # ------
    
    # Node Creation: Add nodes for each source and target word, plus ε-nodes
    
    # Add real word nodes
    for i, word in enumerate(src_words):
        G.add_node(f"src_{i}", bipartite=0, word=word)
    for j, word in enumerate(target_words):
        G.add_node(f"tgt_{j}", bipartite=1, word=word)

    # Fill both partitions respective to amount of words in other 
    # partition to achieve same amount of nodes on both sides.
    for j in range(len(target_words)):
        G.add_node(f"src_ε_{j}", bipartite=0, word=EPSILON_SYMBOL)
    for i in range(len(src_words)):
        G.add_node(f"tgt_ε_{i}", bipartite=1, word=EPSILON_SYMBOL)
        
    m = len(src_words)
    n = len(target_words)
    N = m + n   # N = total nodes per partition after adding ε-nodes

    # ------
    
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
    score = calculate_score_weighted(word_similarity, position_score, alpha=0.5)

    return 1 - score  # Convert similarity to cost


def solve_matching(G: nx.Graph) -> dict:
    """Find optimal minimum-weight matching on the bipartite graph."""
    src_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
    return nx.bipartite.minimum_weight_full_matching(
        G, top_nodes=src_nodes, weight="weight"
    )


# ── quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dit = clean("ich habe das gemacht").split()
    dat = clean("i ha das gmacht").split()

    G = build_bipartite_graph(dit, dat, max_word_len=10, max_sent_len=5)
    matching = solve_matching(G)

    print("Matching:")
    for node, match in matching.items():
        if node.startswith("src"):
            src_word = G.nodes[node]["word"]
            tgt_word = G.nodes[match]["word"]
            score = G.edges[node, match]["score"]
            print(f"  {src_word:>12} → {tgt_word:<12}  (score: {score:.3f})")
