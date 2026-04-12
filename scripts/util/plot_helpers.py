import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import networkx as nx


# --- Helper Functions ---

def _sort_key(node_name: str) -> tuple[int, int]:
    """Sort key for bipartite graph nodes: real words first, then epsilon; each group sorted by index.

    Args:
        node_name: Node name like "ref_3" or "hyp_ε_2".

    Returns:
        A tuple (is_eps, idx), where is_eps is 0 for real words and 1 for epsilon (so real words
        come first), and idx is the extracted index for sorting inside words/epsilons.
    """
    is_eps = 1 if "ε" in node_name else 0
    idx = int(node_name.split("_")[-1])  # extract index from node name (last segment)
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

    pos: dict = {"s": (-0.5, y_center), "t": (1.5, y_center)}
    for i, ref_node in enumerate(ref_nodes):
        pos[ref_node] = (0, N - 1 - i)
    for j, hyp_node in enumerate(hyp_nodes):
        pos[hyp_node] = (1, N - 1 - j)
    return pos, N


# --- Plotting Functions ---

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
    cmap = mcolors.LinearSegmentedColormap.from_list("rg", ["#e74c3c", "#f0c929", "#2ecc71"])

    # --- build reduced graph: s -> R' -> (matching) -> H' -> t ---
    R = nx.DiGraph()

    # add source and sink
    R.add_node("s", label="s", side="source")
    R.add_node("t", label="t", side="sink")

    for ref_node, hyp_node in matching.items():
        is_ref_eps = G.nodes[ref_node]["word"] == "ε"
        is_hyp_eps = G.nodes[hyp_node]["word"] == "ε"
        if is_ref_eps and is_hyp_eps:  # epsilon to epsilon matches - not part of reduced graph
            continue

        # add reference nodes & connect with source
        R.add_node(ref_node, label=G.nodes[ref_node]["word"], side="ref")
        R.add_edge("s", ref_node, score=None)  # s -> ref node
        # add hypothesis nodes & connect with sink
        R.add_node(hyp_node, label=G.nodes[hyp_node]["word"], side="hyp")
        R.add_edge(hyp_node, "t", score=None)  # hyp node -> t

        # add matching edge with score
        score = G.edges[ref_node, hyp_node].get("score", 0)
        R.add_edge(ref_node, hyp_node, score=score)

    # --- layout: s (left), ref nodes (center-left), hyp nodes (center-right), t (right) ---
    ref_nodes: list[str] = [str(node) for node, attrs in R.nodes(data=True) if attrs["side"] == "ref"]
    hyp_nodes: list[str] = [str(node) for node, attrs in R.nodes(data=True) if attrs["side"] == "hyp"]

    # sort reference & hypothesis nodes
    ref_nodes = sorted(ref_nodes, key=_sort_key)
    hyp_nodes = sorted(hyp_nodes, key=_sort_key)

    pos, N = _build_layout(ref_nodes, hyp_nodes)

    # --- node styling ---
    labels = {node: attrs["label"] for node, attrs in R.nodes(data=True)}
    color_map = {"source": "#c8e6c9", "ref": "#d5e8f0", "hyp": "#f0e0d5", "sink": "#ffcdd2"}
    node_colors = []
    for node, attrs in R.nodes(data=True):
        if "ε" in node:
            node_colors.append("#e0e0e0")  # grey for epsilon-nodes
        else:
            node_colors.append(color_map[attrs["side"]])
    node_sizes = [1600 if attrs["side"] in ("source", "sink") else 1200 for _, attrs in R.nodes(data=True)]

    # --- classify edges ---
    matching_edges = [(u, v) for u, v in R.edges() if u != "s" and v != "t"]
    source_target_edges = [(u, v) for u, v in R.edges() if u == "s" or v == "t"]

    # --- draw ---
    fig, ax = plt.subplots(figsize=(12, N * 0.5 + 1.5))

    # source/sink edges: s -> ref, hyp -> t (thin, gray, straight)
    nx.draw_networkx_edges(R, pos, edgelist=source_target_edges, ax=ax,
                           edge_color="#ccc", width=1.0, style="-", alpha=0.6)

    # matching edges: routed as horizontal–diagonal–horizontal
    # short horizontal stub from each node, diagonal in the middle
    eps_edges = [(u, v) for u, v in matching_edges if "ε" in str(u) or "ε" in str(v)]
    word_edges = [(u, v) for u, v in matching_edges if "ε" not in str(u) and "ε" not in str(v)]

    stub_len = 0.15  # length of horizontal stub from each node

    def draw_hdh_edge(start_pos, end_pos, color, width, style, alpha):
        """Draw horizontal-diagonal-horizontal edge."""
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        path_x = [start_x, start_x + stub_len, end_x - stub_len, end_x]
        path_y = [start_y, start_y, end_y, end_y]
        ax.plot(path_x, path_y, color=color, linewidth=width, linestyle=style, alpha=alpha, zorder=1)

    for u, v in eps_edges:
        draw_hdh_edge(pos[u], pos[v], "#ccc", 1.2, "--", 0.6)

    for u, v in word_edges:
        score = R.edges[u, v]["score"]
        draw_hdh_edge(pos[u], pos[v], cmap(score), 2.5, "-", 1.0)

    # draw nodes ON TOP of edges so start/end points are clearly visible
    nx.draw_networkx_nodes(R, pos, ax=ax, node_size=node_sizes, node_color=node_colors,
                           edgecolors="#aaa", linewidths=1.0)
    nx.draw_networkx_labels(R, pos, labels=labels, ax=ax, font_size=9, font_weight="bold")

    # score labels on word -> word edges - on the horizontal stub near the ref node
    scored_edges = [(u, v, attrs["score"]) for u, v, attrs in R.edges(data=True)
                    if attrs["score"] is not None and "ε" not in u and "ε" not in v]
    for u, v, score in scored_edges:
        label_x = pos[u][0] + stub_len * 0.7  # towards the end of the horizontal stub
        label_y = pos[u][1]
        ax.text(label_x, label_y, f"{score:.2f}", fontsize=7, ha="center", va="center",
                color="#555", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))

    ax.set_title("Reduced Bipartite Flow Network (s → R' → H' → t)", fontsize=14, pad=15)
    ax.axis("off")

    # score colorbar (red=0 -> yellow=0.5 -> green=1)
    score_colorbar = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    score_colorbar.set_array([])
    fig.colorbar(score_colorbar, ax=ax, fraction=0.03, pad=0.04).set_label("Score", fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_bipartite_graph_full(
        G: nx.DiGraph,
):
    """Plot the full bipartite flow network (all edges, before matching) using NetworkX.

    Shows the complete graph structure: s → R' (all edges) → H' (all edges) → t,
    including all possible connections.

    Args:
        G: Full flow network.
    """
    # --- collect nodes by side/partition ---
    ref_nodes: list[str] = [str(node) for node, attrs in G.nodes(data=True) if attrs.get("side") == "ref"]
    hyp_nodes: list[str] = [str(node) for node, attrs in G.nodes(data=True) if attrs.get("side") == "hyp"]

    ref_nodes = sorted(ref_nodes, key=_sort_key)
    hyp_nodes = sorted(hyp_nodes, key=_sort_key)

    pos, n_max = _build_layout(ref_nodes, hyp_nodes)

    # --- node styling ---
    all_nodes = ["s"] + ref_nodes + hyp_nodes + ["t"]
    labels = {}
    node_colors = []
    node_sizes = []
    color_map = {"source": "#c8e6c9", "ref": "#d5e8f0", "hyp": "#f0e0d5", "sink": "#ffcdd2"}

    for node in all_nodes:
        side = G.nodes[node]["side"]
        if side in ("source", "sink"):
            labels[node] = node
            node_colors.append(color_map[side])
            node_sizes.append(1600)
        else:
            labels[node] = G.nodes[node]["word"]
            node_colors.append(color_map[side])
            node_sizes.append(1200)

    # --- classify edges ---
    s_edges = [(u, v) for u, v in G.edges() if u == "s"]
    t_edges = [(u, v) for u, v in G.edges() if v == "t"]
    bipartite_edges = [(u, v) for u, v in G.edges()
                       if G.nodes[u].get("side") == "ref" and G.nodes[v].get("side") == "hyp"]

    # --- draw ---
    fig, ax = plt.subplots(figsize=(12, n_max * 0.5 + 1.5))

    nx.draw_networkx_nodes(G, pos, nodelist=all_nodes, ax=ax,
                           node_size=node_sizes, node_color=node_colors,
                           edgecolors="#aaa", linewidths=1.0)
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=9, font_weight="bold")

    # s → R' edges (grey)
    nx.draw_networkx_edges(G, pos, edgelist=s_edges, ax=ax,
                           edge_color="#ccc", width=0.8, alpha=0.5,
                           connectionstyle="arc3,rad=0.0")

    # H' → t edges (grey)
    nx.draw_networkx_edges(G, pos, edgelist=t_edges, ax=ax,
                           edge_color="#ccc", width=0.8, alpha=0.5,
                           connectionstyle="arc3,rad=0.0")

    # R' → H' bipartite edges (light blue, thin)
    nx.draw_networkx_edges(G, pos, edgelist=bipartite_edges, ax=ax,
                           edge_color="#b0c4de", width=0.5, alpha=0.3,
                           connectionstyle="arc3,rad=0.0")

    ax.set_title("Full Bipartite Flow Network", fontsize=14, pad=15)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_score_distribution(data, title):
    """Plot a histogram of scores with a mean value indicated by a dashed line.

    Args:
        data: Series or array of score values.
        title: Title for the plot.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data.dropna(), bins=50, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Score (0 – 1)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.set_xlim(0, 1)
    mean_val = data.dropna().mean()
    ax.axvline(
        mean_val,
        color="#DD5544",
        linewidth=1.2,
        linestyle="--",
        label=f"Mean: {mean_val:.3f}",
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
