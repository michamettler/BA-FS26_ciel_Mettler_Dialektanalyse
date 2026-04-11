import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np


def plot_score_distribution(data, title):
    """_summary_
    Helper function to plot a histogram of scores with mean value indicated by a dashed line.

    Args:
        data (_type_): _description_ 
        title (_type_): _description_
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


def plot_bipartite_matching(
    G: nx.DiGraph,
    matching: dict[str, str],
    reference: list[str],
    hypothesis: list[str],
):
    """_summary_
    Helper function to plot the bipartite matching between reference and hypothesis words.
    Only show matched edges (word-word, word-ε) and colour word-word edges by their score.

    Args:
        G (nx.DiGraph): _description_
        matching (dict[str, str]): _description_
        reference (list[str]): _description_
        hypothesis (list[str]): _description_
    """
    n_r, n_h = len(reference), len(hypothesis)
    # -- horizontal (x) positions --
    x_ref, x_hyp = 0.0, 1.0
    # colour map for word-word edge scores (red=0 → yellow=0.5 → green=1)
    cmap = mcolors.LinearSegmentedColormap.from_list("rg", ["#e74c3c", "#f0c929", "#2ecc71"])
    eps_box = dict(boxstyle="round,pad=0.2", fc="#f5f5f5", ec="#ddd", lw=0.8)

    # --- classify matching edges ---
    matches = [] # (ref_idx, hyp_idx, score)
    deletions_idx = [] # reference indices -> ε
    insertions_idx = [] # ε -> hypothesis indices

    for ref_node, hyp_node in matching.items():
        ref_is_eps = ref_node.startswith("ref_ε")
        hyp_is_eps = hyp_node.startswith("hyp_ε")
        if ref_is_eps and hyp_is_eps: # epsilon to epsilon matching - not part of reduced graph
            continue
        if ref_is_eps: # reference epsilon to hypothesis word matching
            idx = int(hyp_node.split("_")[1]) # epsilon in reference (insertion), extracts index from hypothesis word node
            insertions_idx.append(idx)
        elif hyp_is_eps: # reference word to hypothesis epsilon matching
            idx = int(ref_node.split("_")[1]) # epsilon in hypothesis (deletion), extracts index from reference word node
            deletions_idx.append(idx)
        else: # word to word matching
            ref_idx = int(ref_node.split("_")[1]) # extracts index from reference word node
            hyp_idx = int(hyp_node.split("_")[1]) # extracts index from hypothesis word node
            matches.append((ref_idx, hyp_idx, G.edges[ref_node, hyp_node]["score"]))

    # sort deletions & insertions based on index
    deletions_idx.sort()
    insertions_idx.sort()

    # --- vertical (y) positions ---
    # Layout: real words at the top, ε-nodes padded at the bottom.
    # The y-axis is inverted: row 0 (first word) is at the top, last row at the bottom.
    n_eps_ref = len(insertions_idx)   # ε-nodes on the reference (left) side
    n_eps_hyp = len(deletions_idx)    # ε-nodes on the hypothesis (right) side
    n_rows = max(n_r + n_eps_ref, n_h + n_eps_hyp)
    row_to_y = lambda row: n_rows - 1 - row  # row 0 -> top, row n_rows-1 -> bottom

    # word nodes: rows 0..n_r-1 (left) and 0..n_h-1 (right)
    ref_y = {i: row_to_y(i) for i in range(n_r)}
    hyp_y = {j: row_to_y(j) for j in range(n_h)}

    # ε-nodes: placed below word nodes on each side
    eps_ref_y = [row_to_y(n_r + k) for k in range(n_eps_ref)]
    eps_hyp_y = [row_to_y(n_h + k) for k in range(n_eps_hyp)]

    # --- drawing helpers ---
    fig, ax = plt.subplots(figsize=(12, n_rows * 0.55 + 1))
    edge_offset = 0.10  # horizontal inset so edges don't overlap labels

    def draw_edge(y_from, y_to, color="#ddd", dashed=False):
        """Draw a connecting line between a reference (left) and hypothesis (right) node."""
        ax.plot(
            [x_ref + edge_offset, x_hyp - edge_offset],
            [y_from, y_to],
            color=color,
            linewidth=1.2 if dashed else 2,
            linestyle="--" if dashed else "-",
            alpha=0.85,
            zorder=1 if dashed else 2,
        )

    def draw_label(x, y_pos, text, color="#222", bold=True):
        """Draw a word label on the left (reference) or right (hypothesis) side."""
        is_left = x < 0.5
        ha = "right" if is_left else "left"
        label_offset = -0.02 if is_left else 0.02
        ax.text(
            x + label_offset, y_pos, text,
            fontsize=11, ha=ha, va="center",
            fontweight="bold" if bold else "normal", color=color,
        )

    def draw_eps_label(x, y_pos):
        """Draw an ε (epsilon) label with a rounded box."""
        ax.text(
            x, y_pos, "ε",
            fontsize=11, ha="center", va="center",
            color="#aaa", fontweight="bold", bbox=eps_box,
        )

    # --- draw matching edges ---
    # word -> word matches (coloured by score)
    for ref_idx, hyp_idx, score in matches:
        draw_edge(ref_y[ref_idx], hyp_y[hyp_idx], color=cmap(score))
        label_x = 0.5
        label_y = (ref_y[ref_idx] + hyp_y[hyp_idx]) / 2
        ax.text(
            label_x, label_y, f"{score:.2f}",
            fontsize=7, ha="center", va="bottom", color="#555", zorder=3,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
        )

    # deletions: ref word -> ε on hypothesis side (dashed)
    for k, ref_idx in enumerate(deletions_idx):
        draw_edge(ref_y[ref_idx], eps_hyp_y[k], dashed=True)

    # insertions: ε on reference side -> hyp word (dashed)
    for k, hyp_idx in enumerate(insertions_idx):
        draw_edge(eps_ref_y[k], hyp_y[hyp_idx], dashed=True)

    # --- draw word labels ---
    # greyed-out words are matched to ε (deleted / inserted)
    del_set = set(deletions_idx)
    ins_set = set(insertions_idx)
    for i, word in enumerate(reference):
        color = "#bbb" if i in del_set else "#222"
        draw_label(x_ref, ref_y[i], word, color=color)
    for j, word in enumerate(hypothesis):
        color = "#bbb" if j in ins_set else "#222"
        draw_label(x_hyp, hyp_y[j], word, color=color)

    # --- draw ε labels ---
    for y_pos in eps_ref_y:
        draw_eps_label(x_ref + 0.06, y_pos)
    for y_pos in eps_hyp_y:
        draw_eps_label(x_hyp - 0.06, y_pos)

    # --- headers & colorbar ---
    header_y = n_rows - 1 + 0.8
    ax.text(x_ref - 0.02, header_y, "Reference",  fontsize=12, ha="right", va="center", fontstyle="italic", color="#888")
    ax.text(x_hyp + 0.02, header_y, "Hypothesis", fontsize=12, ha="left",  va="center", fontstyle="italic", color="#888")

    # score colorbar (red=0 → yellow=0.5 → green=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04).set_label("Score", fontsize=10)

    # --- axis limits & cleanup ---
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-0.8, n_rows + 0.5)
    ax.axis("off")
    ax.set_title("Bipartite Matching: Reference → Hypothesis", fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()