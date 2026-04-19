import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import seaborn as sns
import networkx as nx
from bipartite_matching import (
    EPS,
    SOURCE_NODE,
    SINK_NODE,
    REFERENCE_PARTITION,
    HYPOTHESIS_PARTITION,
    ATTR_WORD,
    ATTR_PARTITION,
    ATTR_LABEL,
    ATTR_SIMILARITY,
    build_reduced_graph_by_matching,
    extract_index_from_node_name,
    is_eps_node,
    get_source_edges,
    get_sink_edges,
    get_bipartite_edges,
    get_word_edges,
    get_epsilon_edges,
    get_nodes_by_partition,
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

    Shows the complete graph structure: s -> R' (all edges) -> H' (all edges) -> t,
    including all possible connections.

    Args:
        G: Full flow network.
    """
    # collect nodes by partition & build layout
    ref_nodes = _sort_nodes_for_display(
        get_nodes_by_partition(G, REFERENCE_PARTITION)
    )
    hyp_nodes = _sort_nodes_for_display(
        get_nodes_by_partition(G, HYPOTHESIS_PARTITION)
    )

    pos, N = _build_layout(ref_nodes, hyp_nodes)

    # node styling
    labels, node_colors, node_sizes = _get_style_for_nodes(G)

    # classify edges
    source_edges = get_source_edges(G)
    sink_edges = get_sink_edges(G)
    bipartite_edges = get_bipartite_edges(G)

    # --- draw ---
    fig, ax = plt.subplots(figsize=(12, N * 0.5 + 1.5))

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_size=node_sizes, node_color=node_colors,
                           edgecolors=_COLOR_NODE_BORDER, linewidths=1.0)
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=9, font_weight="bold")

    # s -> R' edges (gray)
    nx.draw_networkx_edges(G, pos, edgelist=source_edges, ax=ax,
                           edge_color=_COLOR_EDGE_GRAY, width=0.8, alpha=0.5,
                           connectionstyle="arc3,rad=0.0")

    # H' -> t edges (gray)
    nx.draw_networkx_edges(G, pos, edgelist=sink_edges, ax=ax,
                           edge_color=_COLOR_EDGE_GRAY, width=0.8, alpha=0.5,
                           connectionstyle="arc3,rad=0.0")

    # R' -> H' bipartite edges (light blue, thin)
    nx.draw_networkx_edges(G, pos, edgelist=bipartite_edges, ax=ax,
                           edge_color=_COLOR_EDGE_BIPARTITE, width=0.5, alpha=0.3,
                           connectionstyle="arc3,rad=0.0")

    ax.set_title("Full Bipartite Flow Network", fontsize=14, pad=15)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_reduced_bipartite_graph_with_matching(
        G: nx.DiGraph,
        matching: dict[str, str]
):
    """Plot the reduced bipartite flow network using NetworkX.

    The reduced graph only shows the matching edges (ref -> hyp), and the source/sink edges.
    Epsilon-to-epsilon matches are excluded for clarity.
    """
    # color map to visualize scores
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rg", [_COLOR_SCORE_LOW, _COLOR_SCORE_MID, _COLOR_SCORE_HIGH]
    )

    # build matching graph (doesn't include epsilon-to-epsilon matchings): s -> R' -> (matching) -> H' -> t
    M = build_reduced_graph_by_matching(G, matching)

    # collect nodes by partition & build layout
    ref_nodes = _sort_nodes_for_display(
        get_nodes_by_partition(M, REFERENCE_PARTITION)
    )
    hyp_nodes = _sort_nodes_for_display(
        get_nodes_by_partition(M, HYPOTHESIS_PARTITION)
    )

    pos, N = _build_layout(ref_nodes, hyp_nodes)

    # node styling
    labels, node_colors, node_sizes = _get_style_for_nodes(M)

    # classify edges
    matching_edges = get_bipartite_edges(M)
    source_sink_edges = get_source_edges(M) + get_sink_edges(M)
    eps_edges = get_epsilon_edges(M, matching_edges)
    word_edges = get_word_edges(M, matching_edges)

    # --- draw ---
    fig, ax = plt.subplots(figsize=(12, N * 0.5 + 1.5))

    # source/sink edges: s -> ref, hyp -> t (thin, gray, straight)
    nx.draw_networkx_edges(M, pos, edgelist=source_sink_edges, ax=ax,
                           edge_color=_COLOR_EDGE_GRAY, width=1.0, style="-", alpha=0.6)

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
        similarity = M.edges[u, v][ATTR_SIMILARITY]
        draw_hdh_edge(pos[u], pos[v], cmap(similarity), 2.5, "-", 1.0)

    # draw nodes ON TOP of edges so start/end points are clearly visible
    nx.draw_networkx_nodes(M, pos, ax=ax, node_size=node_sizes, node_color=node_colors,
                           edgecolors=_COLOR_NODE_BORDER, linewidths=1.0)
    nx.draw_networkx_labels(M, pos, labels=labels, ax=ax, font_size=9, font_weight="bold")

    # similarity labels on word -> word edges - on the horizontal stub near the ref node
    for u, v in word_edges:
        similarity = M.edges[u, v][ATTR_SIMILARITY]
        if similarity is None:
            continue
        label_x = pos[u][0] + stub_len * 0.7  # towards the end of the horizontal stub
        label_y = pos[u][1]
        ax.text(label_x, label_y, f"{similarity:.3f}", fontsize=7, ha="center", va="center",
                color=_COLOR_SCORE_LABEL, zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))

    ax.set_title("Reduced Bipartite Flow Network (s -> R' -> H' -> t)", fontsize=14, pad=15)
    ax.axis("off")

    # similarity colorbar (red=0 -> yellow=0.5 -> green=1)
    score_colorbar = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    score_colorbar.set_array([])
    fig.colorbar(score_colorbar, ax=ax, fraction=0.03, pad=0.04).set_label("Similarity", fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_similarity_distribution(data, title):
    """Plot a histogram of similarities with a mean value indicated by a dashed line.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data.dropna(), bins=50, color=_COLOR_HIST_BAR, edgecolor="white", linewidth=0.4)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Similarity (0 – 1)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.set_xlim(0, 1)
    mean_val = data.dropna().mean()
    ax.axvline(
        mean_val,
        color=_COLOR_HIST_MEAN,
        linewidth=1.2,
        linestyle="--",
        label=f"Mean: {mean_val:.5f}",
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_grid_search_heatmaps(
        panels: list[tuple[np.ndarray, str]],
        alphas: np.ndarray,
        lambdas: np.ndarray,
        suptitle: str,
        cmap: str = "YlGn",
        highlight_best: bool = True,
        tick_step: int = 5,
):
    """Plot one or more grid-search heatmaps side by side with a shared color scale.

    Args:
        panels: List of (2D array, subtitle) tuples. Each array has shape (len(alphas), len(lambdas)).
        alphas: Alpha values used in the grid search (y-axis).
        lambdas: Lambda values used in the grid search (x-axis).
        suptitle: Overall figure title.
        cmap: Colormap name for seaborn heatmap.
        highlight_best: If True, mark all cells tied for the best value with a red rectangle.
        tick_step: Show every N-th tick label on both axes.
    """
    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    all_values = np.concatenate([data.ravel() for data, _ in panels])
    vmin, vmax = all_values.min(), all_values.max()

    tick_positions = list(range(0, len(alphas), tick_step))
    alpha_tick_labels = [f"{alphas[k]:.2f}" for k in tick_positions]
    lambda_tick_labels = [f"{lambdas[k]:.2f}" for k in tick_positions]

    for ax, (data, title) in zip(axes, panels):
        sns.heatmap(data, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                    xticklabels=False, yticklabels=False, cbar_kws={"label": "Mean F1"})
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(lambda_tick_labels)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(alpha_tick_labels, rotation=0)
        ax.set_xlabel("λ (epsilon penalty)")
        ax.set_ylabel("α (lexical weight)")
        ax.set_title(title)

        if highlight_best:
            best_val = data.max()
            for i, j in np.argwhere(data == best_val):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="red", lw=1.5))

    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


# --- Helper Functions ---

def _sort_nodes_for_display(nodes: list[str]) -> list[str]:
    """Sort nodes for display: real words first, then epsilon; each group sorted by index."""
    return sorted(nodes, key=_get_sort_key_for_node_order)


def _get_sort_key_for_node_order(node_name: str) -> tuple[int, int]:
    """Sort key for bipartite graph nodes: real words first, then epsilon; each group sorted by index.

    Args:
        node_name: Node name like "ref_3" or "hyp_ε_2".

    Returns:
        A tuple (is_eps, idx), where is_eps is 0 for real words and 1 for epsilon (so real words
        come first), and idx is the extracted index for sorting inside words/epsilons.
    """
    is_eps = 1 if EPS in node_name else 0
    idx = extract_index_from_node_name(node_name)
    return is_eps, idx


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
            if is_eps_node(attrs):
                node_colors.append(_COLOR_NODE_EPS)
            else:
                node_colors.append(_NODE_COLOR_MAP[attrs[ATTR_PARTITION]])
            node_sizes.append(1200)

    return labels, node_colors, node_sizes
