import networkx as nx

from calculations import (
    calculate_similarity_for_word_pair_by_weighted_lexical_and_positional_similarities,
    calculate_cost_for_word_pair_by_similarity,
    calculate_cost_for_epsilon_by_penalty,
)
from models import CalculationParameters
from preprocessing import clean_word

EPS = "ε"  # epsilon symbol, used for unmatched padding nodes
SOURCE_NODE = "s"
SINK_NODE = "t"
REFERENCE_PARTITION = "ref"
HYPOTHESIS_PARTITION = "hyp"

# Node/edge attribute keys
ATTR_WORD = "word"
ATTR_PARTITION = "partition"
ATTR_LABEL = "label"
ATTR_SCORE = "score"  # float for word-word edges, None for epsilon routing edges


def build_full_bipartite_graph(
        ref_words: list[str],
        hyp_words: list[str],
        calculation_parameters: CalculationParameters,
) -> nx.DiGraph:
    """Build a weighted bipartite flow network between reference and hypothesis words.

    The alignment is modeled as finding the min-cost max-flow in a bipartite network
    G = (R' ∪ H', E) with source s and sink t.
    Each partition is padded with epsilon-nodes so that both sides have N = n_r + n_h nodes,
    enabling a perfect matching where unmatched words flow through epsilon at a fixed penalty.
    Edge weights represent cost = 1 - score (lower cost = better match).
    All edge capacities are 1 (unit flow).

    Args:
        ref_words: List of words in the reference sentence.
        hyp_words: List of words in the hypothesis sentence.
        calculation_parameters: Matching configuration (alpha, lambda_, normalization mode, max lengths).

    Returns:
        A NetworkX directed graph representing the bipartite flow network with nodes and edges as described above.
    """

    # --- Graph ---
    G = nx.DiGraph()

    n_r = len(ref_words)  # |R|, reference word count / number of epsilon-nodes in H'
    n_h = len(hyp_words)  # |H|, hypothesis word count / number of epsilon-nodes in R'
    N = n_r + n_h  # padded partition size

    # --- Nodes ---
    # source
    G.add_node(SOURCE_NODE, demand=-N)

    # word nodes
    for i, word in enumerate(ref_words):
        G.add_node(get_node_name(REFERENCE_PARTITION, i), word=word, partition=REFERENCE_PARTITION)
    for j, word in enumerate(hyp_words):
        G.add_node(get_node_name(HYPOTHESIS_PARTITION, j), word=word, partition=HYPOTHESIS_PARTITION)

    # epsilon-nodes
    for i in range(n_r):
        G.add_node(get_node_name(HYPOTHESIS_PARTITION, i, eps=True), word=EPS, partition=HYPOTHESIS_PARTITION)
    for j in range(n_h):
        G.add_node(get_node_name(REFERENCE_PARTITION, j, eps=True), word=EPS, partition=REFERENCE_PARTITION)

    # sink
    G.add_node(SINK_NODE, demand=N)

    # --- Edges ---
    # edges from s to R'
    for i in range(n_r):
        G.add_edge(SOURCE_NODE, get_node_name(REFERENCE_PARTITION, i), capacity=1, weight=0)
    for j in range(n_h):
        G.add_edge(SOURCE_NODE, get_node_name(REFERENCE_PARTITION, j, eps=True), capacity=1, weight=0)

    # edges from ref word nodes to hyp word nodes
    for i in range(n_r):
        for j in range(n_h):
            ref_word = clean_word(ref_words[i])
            hyp_word = clean_word(hyp_words[j])

            similarity = calculate_similarity_for_word_pair_by_weighted_lexical_and_positional_similarities(
                ref_word=ref_word,
                ref_position=i,
                hyp_word=hyp_word,
                hyp_position=j,
                calculation_parameters=calculation_parameters,
            )
            cost = calculate_cost_for_word_pair_by_similarity(similarity)

            G.add_edge(
                get_node_name(REFERENCE_PARTITION, i),
                get_node_name(HYPOTHESIS_PARTITION, j),
                capacity=1,
                weight=cost,
                score=similarity,
            )
    # edges from ref word nodes to hyp epsilon-nodes
    epsilon_cost = calculate_cost_for_epsilon_by_penalty(calculation_parameters.lambda_)
    for i in range(n_r):
        for k in range(n_r):
            G.add_edge(
                get_node_name(REFERENCE_PARTITION, i),
                get_node_name(HYPOTHESIS_PARTITION, k, eps=True),
                capacity=1,
                weight=epsilon_cost,
                score=None,
            )

    # edges from ref epsilon-nodes to hyp word nodes
    for j in range(n_h):
        for k in range(n_h):
            G.add_edge(
                get_node_name(REFERENCE_PARTITION, j, eps=True),
                get_node_name(HYPOTHESIS_PARTITION, k),
                capacity=1,
                weight=epsilon_cost,
                score=None,
            )
    # edges from ref epsilon-nodes to hyp epsilon-nodes
    for j in range(n_h):
        for i in range(n_r):
            G.add_edge(get_node_name(REFERENCE_PARTITION, j, eps=True),
                       get_node_name(HYPOTHESIS_PARTITION, i, eps=True),
                       capacity=1, weight=0, score=None)

    # edges from H' to t
    for i in range(n_r):
        G.add_edge(get_node_name(HYPOTHESIS_PARTITION, i, eps=True), SINK_NODE, capacity=1, weight=0)
    for j in range(n_h):
        G.add_edge(get_node_name(HYPOTHESIS_PARTITION, j), SINK_NODE, capacity=1, weight=0)

    return G


def solve_matching(G: nx.DiGraph) -> dict[str, str]:
    """Solve the min-cost max-flow problem on the given bipartite graph to find the optimal matching.

    Args:
        G: A NetworkX directed graph representing the bipartite flow network.

    Returns:
        Mapping of reference word nodes to hypothesis word nodes, representing the optimal matching.
    """
    flow_dict = nx.min_cost_flow(G, weight="weight")  # TODO hungarian as alternative

    # Extract matching: for each node w in W', find the matched node v in V' with flow=1
    matching = {}
    for w, neighbors in flow_dict.items():
        for v, flow in neighbors.items():
            if flow == 1 and G.nodes[v].get(ATTR_PARTITION) == HYPOTHESIS_PARTITION:
                matching[w] = v
    return matching


def build_reduced_graph_by_matching(G: nx.DiGraph, matching: dict[str, str]) -> nx.DiGraph:
    """Build a reduced bipartite graph from the full network and its matching.

    Keeps only matched ref/hyp pairs (excluding epsilon-to-epsilon matches), with source and sink.
    Only includes edges where both endpoints are in the matching.

    Args:
        G: Full flow network.
        matching: Dict mapping each reference node to its matched hypothesis node.

    Returns:
        A reduced NetworkX directed graph with s -> ref -> hyp -> t edges for non-trivial matches.
    """
    M = nx.DiGraph()

    M.add_node(SOURCE_NODE, label=SOURCE_NODE)
    M.add_node(SINK_NODE, label=SINK_NODE)

    for ref_node, hyp_node in matching.items():
        is_ref_eps = is_eps_node(G.nodes[ref_node])
        is_hyp_eps = is_eps_node(G.nodes[hyp_node])
        if is_ref_eps and is_hyp_eps:
            continue

        M.add_node(ref_node, word=G.nodes[ref_node][ATTR_WORD], label=G.nodes[ref_node][ATTR_WORD],
                   partition=REFERENCE_PARTITION)
        M.add_edge(SOURCE_NODE, ref_node, score=None)
        M.add_node(hyp_node, word=G.nodes[hyp_node][ATTR_WORD], label=G.nodes[hyp_node][ATTR_WORD],
                   partition=HYPOTHESIS_PARTITION)
        M.add_edge(hyp_node, SINK_NODE, score=None)

        score = G.edges[ref_node, hyp_node].get(ATTR_SCORE)
        M.add_edge(ref_node, hyp_node, score=score)

    return M


def get_node_name(partition: str, index: int, eps: bool = False) -> str:
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


def extract_index_from_node_name(node_name: str) -> int:
    """Extract the numerical index from a node name like "ref_3" or "hyp_ε_2".

    Splits on underscores and returns the last segment as an integer.

    Args:
        node_name: Node name string where the last underscore-separated segment is a numeric index.

    Returns:
        The extracted integer index.
    """
    return int(node_name.split("_")[-1])


def is_eps_node(attrs) -> bool:
    return attrs.get(ATTR_WORD) == EPS


# --- Graph Query Helpers ---

def get_nodes_by_partition(graph: nx.DiGraph, partition: str) -> list[str]:
    return [str(node) for node, attrs in graph.nodes(data=True)
            if attrs.get(ATTR_PARTITION) == partition]

def get_source_edges(graph: nx.DiGraph) -> list[tuple[str, str]]:
    return [(u, v) for u, v in graph.edges() if u == SOURCE_NODE]


def get_sink_edges(graph: nx.DiGraph) -> list[tuple[str, str]]:
    return [(u, v) for u, v in graph.edges() if v == SINK_NODE]


def get_bipartite_edges(graph: nx.DiGraph) -> list[tuple[str, str]]:
    """Get all edges between the reference and hypothesis partitions (ref -> hyp).

    Args:
        graph: A NetworkX directed graph with partition attributes on nodes.

    Returns:
        A list of (u, v) tuples for edges from reference to hypothesis nodes.
    """
    return [(u, v) for u, v in graph.edges()
            if graph.nodes[u].get(ATTR_PARTITION) == REFERENCE_PARTITION
            and graph.nodes[v].get(ATTR_PARTITION) == HYPOTHESIS_PARTITION]


def get_word_edges(graph: nx.DiGraph, matching_edges: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Extract word-level edges from a list of matching edges.

    Filters out edges where either endpoint is an epsilon node.

    Args:
        graph: A NetworkX directed graph with "word" attributes on nodes.
        matching_edges: A list of tuples, where each tuple represents an edge between two nodes.

    Returns:
        A list of tuples representing edges where neither node is an epsilon node.
    """
    return [(u, v) for u, v in matching_edges
            if not is_eps_node(graph.nodes[u]) and not is_eps_node(graph.nodes[v])]


def get_epsilon_edges(graph: nx.DiGraph, matching_edges: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Extract epsilon edges from a list of matching edges.

    Returns only edges where at least one endpoint is an epsilon node.

    Args:
        graph: A NetworkX directed graph with "word" attributes on nodes.
        matching_edges: A list of edges represented as tuples of strings.

    Returns:
        A list of edges where at least one node is an epsilon node.
    """
    return [(u, v) for u, v in matching_edges
            if is_eps_node(graph.nodes[u]) or is_eps_node(graph.nodes[v])]
