import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import networkx as nx
from domain.bipartite_matching import (
    EPS,
    SOURCE_NODE,
    SINK_NODE,
    REFERENCE_PARTITION,
    HYPOTHESIS_PARTITION,
    ATTR_WORD,
    ATTR_PARTITION,
    ATTR_LABEL,
    ATTR_SCORE,
)

# --- Colors ---

_NODE_COLOR_MAP = {
    SOURCE_NODE: "#c8e6c9",
    REFERENCE_PARTITION: "#d5e8f0",
    HYPOTHESIS_PARTITION: "#f0e0d5",
    SINK_NODE: "#ffcdd2",
}
_COLOR_NODE_BORDER = "#aaa"
_COLOR_NODE_EPS = "#e0e0e0"

_COLOR_SCORE_LOW = "#e74c3c"
_COLOR_SCORE_MID = "#f0c929"
_COLOR_SCORE_HIGH = "#2ecc71"
_COLOR_SCORE_LABEL = "#555"

_COLOR_EDGE_GRAY = "#ccc"
_COLOR_EDGE_BIPARTITE = "#b0c4de"

_COLOR_HIST_BAR = "#4C72B0"
_COLOR_HIST_MEAN = "#DD5544"


# --- Plotting Functions ---

def plot_bipartite_graph_full(
        G: nx.DiGraph,
):
    """Plot the full bipartite flow network (all edges, before matching) using NetworkX.

    Shows the complete graph structure: s → R' (all edges) → H' (all edges) → t,
    including all possible connections.

    Args:
        G: Full flow network.
    """
    # --- collect nodes by partition & build layout ---
    ref_nodes = _collect_nodes_by_partition(G, REFERENCE_PARTITION)
    hyp_nodes = _collect_nodes_by_partition(G, HYPOTHESIS_PARTITION)

    pos, N = _build_layout(ref_nodes, hyp_nodes)

    # --- node styling ---
    labels, node_colors, node_sizes = _get_style_for_nodes(G)

    # --- classify edges ---
    source_edges = _get_source_edges(G)
    sink_edges = _get_sink_edges(G)
    bipartite_edges = _get_bipartite_edges(G)

    # --- draw ---
    fig, ax = plt.subplots(figsize=(12, N * 0.5 + 1.5))

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_size=node_sizes, node_color=node_colors,
                           edgecolors=_COLOR_NODE_BORDER, linewidths=1.0)
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=9, font_weight="bold")

    # s → R' edges (gray)
    nx.draw_networkx_edges(G, pos, edgelist=source_edges, ax=ax,
                           edge_color=_COLOR_EDGE_GRAY, width=0.8, alpha=0.5,
                           connectionstyle="arc3,rad=0.0")

    # H' → t edges (gray)
    nx.draw_networkx_edges(G, pos, edgelist=sink_edges, ax=ax,
                           edge_color=_COLOR_EDGE_GRAY, width=0.8, alpha=0.5,
                           connectionstyle="arc3,rad=0.0")

    # R' → H' bipartite edges (light blue, thin)
    nx.draw_networkx_edges(G, pos, edgelist=bipartite_edges, ax=ax,
                           edge_color=_COLOR_EDGE_BIPARTITE, width=0.5, alpha=0.3,
                           connectionstyle="arc3,rad=0.0")

    ax.set_title("Full Bipartite Flow Network", fontsize=14, pad=15)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_bipartite_graph(
        G: nx.DiGraph,
        matching: dict[str, str]
):
    """Plot the reduced bipartite flow network using NetworkX.

    The reduced graph doesn't include epsilon-to-epsilon matchings to reduce complexity.

    Args:
        G: Full flow network.
        matching: Dict mapping of reference and hypothesis partitions.
    """
    # color map to visualize scores
    cmap = mcolors.LinearSegmentedColormap.from_list("rg", [_COLOR_SCORE_LOW, _COLOR_SCORE_MID, _COLOR_SCORE_HIGH])

    # build matching graph (doesn't include epsilon-to-epsilon matchings): s -> R' -> (matching) -> H' -> t
    M = _build_reduced_graph(G, matching)

    # collect nodes by partition & build layout
    ref_nodes = _collect_nodes_by_partition(M, REFERENCE_PARTITION)
    hyp_nodes = _collect_nodes_by_partition(M, HYPOTHESIS_PARTITION)

    pos, N = _build_layout(ref_nodes, hyp_nodes)

    # node styling
    labels, node_colors, node_sizes = _get_style_for_nodes(M)

    # classify edges
    matching_edges = _get_bipartite_edges(M)
    source_sink_edges = _get_source_edges(M) + _get_sink_edges(M)

    # --- draw ---
    fig, ax = plt.subplots(figsize=(12, N * 0.5 + 1.5))

    # source/sink edges: s -> ref, hyp -> t (thin, gray, straight)
    nx.draw_networkx_edges(M, pos, edgelist=source_sink_edges, ax=ax,
                           edge_color=_COLOR_EDGE_GRAY, width=1.0, style="-", alpha=0.6)

    # matching edges: routed as horizontal–diagonal–horizontal
    eps_edges = [(u, v) for u, v in matching_edges if EPS in str(u) or EPS in str(v)]
    word_edges = [(u, v) for u, v in matching_edges if EPS not in str(u) and EPS not in str(v)]

    stub_len = 0.15  # length of horizontal stub from each node

    def draw_hdh_edge(start_pos, end_pos, color, width, style, alpha):
        """Draw horizontal-diagonal-horizontal edge."""
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        path_x = [start_x, start_x + stub_len, end_x - stub_len, end_x]
        path_y = [start_y, start_y, end_y, end_y]
        ax.plot(path_x, path_y, color=color, linewidth=width, linestyle=style, alpha=alpha, zorder=1)

    for u, v in eps_edges:
        draw_hdh_edge(pos[u], pos[v], _COLOR_EDGE_GRAY, 1.2, "--", 0.6)

    for u, v in word_edges:
        score = M.edges[u, v][ATTR_SCORE]
        draw_hdh_edge(pos[u], pos[v], cmap(score), 2.5, "-", 1.0)

    # draw nodes ON TOP of edges so start/end points are clearly visible
    nx.draw_networkx_nodes(M, pos, ax=ax, node_size=node_sizes, node_color=node_colors,
                           edgecolors=_COLOR_NODE_BORDER, linewidths=1.0)
    nx.draw_networkx_labels(M, pos, labels=labels, ax=ax, font_size=9, font_weight="bold")

    # score labels on word -> word edges - on the horizontal stub near the ref node
    scored_edges = [(u, v, attrs[ATTR_SCORE]) for u, v, attrs in M.edges(data=True)
                    if attrs[ATTR_SCORE] is not None and EPS not in u and EPS not in v]
    for u, v, score in scored_edges:
        label_x = pos[u][0] + stub_len * 0.7  # towards the end of the horizontal stub
        label_y = pos[u][1]
        ax.text(label_x, label_y, f"{score:.2f}", fontsize=7, ha="center", va="center",
                color=_COLOR_SCORE_LABEL, zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))

    ax.set_title("Reduced Bipartite Flow Network (s → R' → H' → t)", fontsize=14, pad=15)
    ax.axis("off")

    # score colorbar (red=0 -> yellow=0.5 -> green=1)
    score_colorbar = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    score_colorbar.set_array([])
    fig.colorbar(score_colorbar, ax=ax, fraction=0.03, pad=0.04).set_label("Score", fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_score_distribution(data, title):
    """Plot a histogram of scores with a mean value indicated by a dashed line.

    Args:
        data: Series or array of score values.
        title: Title for the plot.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data.dropna(), bins=50, color=_COLOR_HIST_BAR, edgecolor="white", linewidth=0.4)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Score (0 – 1)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.set_xlim(0, 1)
    mean_val = data.dropna().mean()
    ax.axvline(
        mean_val,
        color=_COLOR_HIST_MEAN,
        linewidth=1.2,
        linestyle="--",
        label=f"Mean: {mean_val:.3f}",
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# --- Helper Functions ---

def _collect_nodes_by_partition(graph: nx.DiGraph, partition: str) -> list[str]:
    """Collect and sort nodes belonging to a given partition.

    Args:
        graph: A NetworkX directed graph with "partition" attributes on nodes.
        partition: The partition value to filter by (e.g. REFERENCE_PARTITION or HYPOTHESIS_PARTITION).

    Returns:
        A sorted list of node names belonging to the given partition.
    """
    nodes = [str(node) for node, attrs in graph.nodes(data=True)
             if attrs.get(ATTR_PARTITION) == partition]
    return sorted(nodes, key=_get_sort_key_for_node_order)


def _build_reduced_graph(G: nx.DiGraph, matching: dict[str, str]) -> nx.DiGraph:
    """Build a reduced bipartite graph from the full network and its matching.

    Keeps only matched ref/hyp pairs (excluding epsilon-to-epsilon matches), with source and sink.

    Args:
        G: Full flow network.
        matching: Dict mapping each reference node to its matched hypothesis node.

    Returns:
        A reduced NetworkX directed graph with s → ref → hyp → t edges for non-trivial matches.
    """
    M = nx.DiGraph()

    M.add_node(SOURCE_NODE, label=SOURCE_NODE)
    M.add_node(SINK_NODE, label=SINK_NODE)

    for ref_node, hyp_node in matching.items():
        is_ref_eps = G.nodes[ref_node][ATTR_WORD] == EPS
        is_hyp_eps = G.nodes[hyp_node][ATTR_WORD] == EPS
        if is_ref_eps and is_hyp_eps:
            continue

        M.add_node(ref_node, label=G.nodes[ref_node][ATTR_WORD], partition=REFERENCE_PARTITION)
        M.add_edge(SOURCE_NODE, ref_node, score=None)
        M.add_node(hyp_node, label=G.nodes[hyp_node][ATTR_WORD], partition=HYPOTHESIS_PARTITION)
        M.add_edge(hyp_node, SINK_NODE, score=None)

        score = G.edges[ref_node, hyp_node].get(ATTR_SCORE, 0)
        M.add_edge(ref_node, hyp_node, score=score)

    return M


def _get_source_edges(graph: nx.DiGraph) -> list[tuple[str, str]]:
    """Get all edges originating from the source node.

    Args:
        graph: A NetworkX directed graph with a source node.

    Returns:
        A list of (u, v) tuples for edges where u is the source node.
    """
    return [(u, v) for u, v in graph.edges() if u == SOURCE_NODE]


def _get_sink_edges(graph: nx.DiGraph) -> list[tuple[str, str]]:
    """Get all edges leading into the sink node.

    Args:
        graph: A NetworkX directed graph with a sink node.

    Returns:
        A list of (u, v) tuples for edges where v is the sink node.
    """
    return [(u, v) for u, v in graph.edges() if v == SINK_NODE]


def _get_bipartite_edges(graph: nx.DiGraph) -> list[tuple[str, str]]:
    """Get all edges between the reference and hypothesis partitions (ref → hyp).

    Args:
        graph: A NetworkX directed graph with partition attributes on nodes.

    Returns:
        A list of (u, v) tuples for edges from reference to hypothesis nodes.
    """
    return [(u, v) for u, v in graph.edges()
            if graph.nodes[u].get(ATTR_PARTITION) == REFERENCE_PARTITION
            and graph.nodes[v].get(ATTR_PARTITION) == HYPOTHESIS_PARTITION]


def _build_layout(ref_nodes: list[str], hyp_nodes: list[str]) -> tuple[dict, int]:
    """Compute 4-column layout: source (s), reference nodes (ref), hypothesis nodes (hyp) and sink (t).

    Args:
        ref_nodes: List of reference node names.
        hyp_nodes: List of hypothesis node names.

    Returns:
        A tuple (pos, N), where pos is a dict mapping node name to (column, row) position,
        and N is the row count (max of ref and hyp side).
    """
    N = max(len(ref_nodes), len(hyp_nodes))
    y_center = (N - 1) / 2

    pos: dict = {SOURCE_NODE: (-0.5, y_center), SINK_NODE: (1.5, y_center)}
    for i, ref_node in enumerate(ref_nodes):
        pos[ref_node] = (0, N - 1 - i)
    for j, hyp_node in enumerate(hyp_nodes):
        pos[hyp_node] = (1, N - 1 - j)
    return pos, N


def _get_style_for_nodes(graph: nx.DiGraph) -> tuple[dict, list[str], list[int]]:
    """Compute labels, colors, and sizes for all nodes in a bipartite flow graph.

    Node colors: gray for ε-nodes, partition color for ref/hyp word nodes,
    and dedicated colors for source/sink.

    Args:
        graph: A NetworkX directed graph with bipartite flow network structure.

    Returns:
        A tuple (labels, node_colors, node_sizes) aligned with graph.nodes() iteration order.
    """
    labels = {}
    node_colors = []
    node_sizes = []

    for node, attrs in graph.nodes(data=True):
        node_str = str(node)
        labels[node] = attrs.get(ATTR_LABEL, attrs.get(ATTR_WORD, node_str))

        if node_str in (SOURCE_NODE, SINK_NODE):
            node_colors.append(_NODE_COLOR_MAP[node_str])
            node_sizes.append(1600)
        else:
            if EPS in node_str:
                node_colors.append(_COLOR_NODE_EPS)
            else:
                node_colors.append(_NODE_COLOR_MAP[attrs[ATTR_PARTITION]])
            node_sizes.append(1200)

    return labels, node_colors, node_sizes


def _get_sort_key_for_node_order(node_name: str) -> tuple[int, int]:
    """Sort key for bipartite graph nodes: real words first, then epsilon; each group sorted by index.

    Args:
        node_name: Node name like "ref_3" or "hyp_ε_2".

    Returns:
        A tuple (is_eps, idx), where is_eps is 0 for real words and 1 for epsilon (so real words
        come first), and idx is the extracted index for sorting inside words/epsilons.
    """
    is_eps = 1 if EPS in node_name else 0
    idx = _extract_index_from_node_name(node_name)
    return is_eps, idx


def _extract_index_from_node_name(node_name: str) -> int:
    """Extract the numerical index from a node name like "ref_3" or "hyp_ε_2".

    Splits on underscores and returns the last segment as an integer.

    Args:
        node_name: Node name string where the last underscore-separated segment is a numeric index.

    Returns:
        The extracted integer index.
    """
    return int(node_name.split("_")[-1])
