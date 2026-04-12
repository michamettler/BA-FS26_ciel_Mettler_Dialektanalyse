import networkx as nx
from calculations import (
    clean,
    calculate_word_similarity_local,
    calculate_word_similarity_global,
    calculate_position_score,
    calculate_score_weighted,
)

COST_SCALE = 1000  # scale float costs to int for network_simplex TODO remove

EPS = "ε"  # epsilon symbol, used for unmatched padding nodes
SOURCE_NODE = "s"
SINK_NODE = "t"
REFERENCE_PARTITION = "ref"
HYPOTHESIS_PARTITION = "hyp"

# Node/edge attribute keys
ATTR_WORD = "word"
ATTR_PARTITION = "partition"
ATTR_LABEL = "label"
ATTR_SCORE = "score"


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

    The alignment is modeled as finding the min-cost max-flow in a bipartite network
    G = (R' ∪ H', E) with source s and sink t.
    Each partition is padded with ε-nodes so that both sides have N = n_r + n_h nodes,
    enabling a perfect matching where unmatched words flow through ε at a fixed penalty.
    Edge weights represent cost = 1 - score (lower cost = better match).
    All edge capacities are 1 (unit flow).

    Args:
        ref_words: List of words in the reference sentence.
        hyp_words: List of words in the hypothesis sentence.
        max_word_len: Global max word length for normalizing Levenshtein distance (if global normalization is used).
        max_sent_len: Global max sentence length for normalizing position score.
        is_levenshtein_normalization_global: Whether to use global or local normalization for Levenshtein distance when calculating word similarity.
        alpha: Weight for combining lexical and positional similarity into a single score (default 0.7 means 70% lexical, 30% positional).
        lambda_: Penalty cost for unmatched words (flow through ε-nodes), in [0, 1] (default 0.3 means 30% penalty).

    Returns:
        A NetworkX directed graph representing the bipartite flow network with nodes and edges as described above.
    """
    # --- Graph ---
    G = nx.DiGraph()

    n_r = len(ref_words)  # |R|, reference word count / number of ε-nodes in H'
    n_h = len(hyp_words)  # |H|, hypothesis word count / number of ε-nodes in R'
    N = n_r + n_h  # padded partition size

    # --- Nodes ---
    # source
    G.add_node(SOURCE_NODE, demand=-N)

    # word nodes
    for i, word in enumerate(ref_words):
        G.add_node(_get_node_name(REFERENCE_PARTITION, i), word=word, partition=REFERENCE_PARTITION)
    for j, word in enumerate(hyp_words):
        G.add_node(_get_node_name(HYPOTHESIS_PARTITION, j), word=word, partition=HYPOTHESIS_PARTITION)

    # epsilon-nodes
    for i in range(n_r):
        G.add_node(_get_node_name(HYPOTHESIS_PARTITION, i, eps=True), word=EPS, partition=HYPOTHESIS_PARTITION)
    for j in range(n_h):
        G.add_node(_get_node_name(REFERENCE_PARTITION, j, eps=True), word=EPS, partition=REFERENCE_PARTITION)

    # sink
    G.add_node(SINK_NODE, demand=N)

    # --- Edges ---
    # edges from s to R'
    for i in range(n_r):
        G.add_edge(SOURCE_NODE, _get_node_name(REFERENCE_PARTITION, i), capacity=1, weight=0)
    for j in range(n_h):
        G.add_edge(SOURCE_NODE, _get_node_name(REFERENCE_PARTITION, j, eps=True), capacity=1, weight=0)

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
                use_gloabal_levenshtein_normalization=is_levenshtein_normalization_global,  # TODO rename
                alpha=alpha,
            )
            G.add_edge(
                _get_node_name(REFERENCE_PARTITION, i),
                _get_node_name(HYPOTHESIS_PARTITION, j),
                capacity=1,
                weight=int(cost * COST_SCALE),
                # TODO belongs to calculations (maybe use ints from lev distance & pos gap)
                score=1 - cost,
            )
    # edges from ref word nodes to hyp ε-nodes
    for i in range(n_r):
        for k in range(n_r):
            G.add_edge(
                _get_node_name(REFERENCE_PARTITION, i),
                _get_node_name(HYPOTHESIS_PARTITION, k, eps=True),
                capacity=1,
                weight=int(lambda_ * COST_SCALE),
                score=1 - lambda_,
            )

    # edges from ref ε-nodes to hyp word nodes
    for j in range(n_h):
        for k in range(n_h):
            G.add_edge(
                _get_node_name(REFERENCE_PARTITION, j, eps=True),
                _get_node_name(HYPOTHESIS_PARTITION, k),
                capacity=1,
                weight=int(lambda_ * COST_SCALE),
                score=1 - lambda_,
            )
    # edges from ref ε-nodes to hyp ε-nodes
    for j in range(n_h):
        for i in range(n_r):
            G.add_edge(_get_node_name(REFERENCE_PARTITION, j, eps=True),
                       _get_node_name(HYPOTHESIS_PARTITION, i, eps=True),
                       capacity=1, weight=0, score=1)

    # edges from H' to t
    for i in range(n_r):
        G.add_edge(_get_node_name(HYPOTHESIS_PARTITION, i, eps=True), SINK_NODE, capacity=1, weight=0)
    for j in range(n_h):
        G.add_edge(_get_node_name(HYPOTHESIS_PARTITION, j), SINK_NODE, capacity=1, weight=0)

    return G


def calculate_cost(  # TODO move to calculations, costs over the whole sentence also possible, also rename
        src_word: str,
        src_position: int,
        target_word: str,
        target_position: int,
        global_max_word_length: int,
        global_max_sentence_length: int,
        use_gloabal_levenshtein_normalization: bool,  # TODO logic shouldnt be here, maybe object for params
        alpha: float = 0.5,
) -> float:
    """Calculate the cost between two words based on their lexical and positional similarity.

    This function determines the similarity between two words using the Levenshtein distance, which
    can be normalized either globally or locally. It also considers the positional difference
    of the words in their respective sentences, normalized by the global maximum sentence length.
    The final cost is computed as one minus the weighted combination of lexical and positional
    similarity scores.

    Args:
        src_word: The source word for comparison.
        src_position: The position of the source word in the sentence.
        target_word: The target word for comparison.
        target_position: The position of the target word in the sentence.
        global_max_word_length: The globally known maximum length of any word in the dataset.
        global_max_sentence_length: The globally known maximum length of any sentence in the dataset.
        use_gloabal_levenshtein_normalization: A flag indicating whether to normalize Levenshtein
            distance globally or locally.
        alpha: A weighting factor for combining lexical and positional similarity. Defaults to 0.5.

    Returns:
        A float representing the cost calculated as one minus the combined similarity score.
    """
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
    position_score = calculate_position_score(  # TODO maybe in levels (based on neighbourhood)
        src_index=src_position,
        target_index=target_position,
        global_max_sentence_length=global_max_sentence_length,
    )

    # Combine lexical and positional similarity into a single score using a weighted average.
    score = calculate_score_weighted(word_similarity, position_score, alpha=alpha)

    return 1 - score  # Convert similarity to cost


def solve_matching(G: nx.DiGraph) -> dict[str, str]:
    """Find the optimal minimum-cost flow on the bipartite flow network.
    Uses Network Simplex algorithm from NetworkX."
    Returns a dict mapping each src node to its matched tgt node."""
    flow_dict = nx.min_cost_flow(G, weight="weight")  # TODO hungarian as alternative

    # Extract matching: for each node w in W', find the matched node v in V' with flow=1
    matching = {}
    for w, neighbors in flow_dict.items():
        for v, flow in neighbors.items():
            if flow == 1 and G.nodes[v].get(ATTR_PARTITION) == HYPOTHESIS_PARTITION:
                matching[w] = v
    return matching


def _get_node_name(partition: str, index: int, eps: bool = False) -> str:
    """Build a node name from partition, index, and optional epsilon flag.

    Args:
        partition: The partition identifier (REFERENCE_PARTITION or HYPOTHESIS_PARTITION).
        index: The numerical index of the node.
        eps: Whether the node is an epsilon (padding) node.

    Returns:
        A node name string, e.g. "ref_0" or "hyp_ε_2".
    """
    if eps:
        return f"{partition}_{EPS}_{index}"
    return f"{partition}_{index}"
