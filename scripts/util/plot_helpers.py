import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np


def plot_score_distribution(data, title):
    """
    Helper function to create create a plot.
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
    src: list[str],
    tgt: list[str],
):
    m, n = len(src), len(tgt)
    x_left, x_right = 0.0, 1.0
    cmap = mcolors.LinearSegmentedColormap.from_list("rg", ["#e74c3c", "#f0c929", "#2ecc71"])
    eps_box = dict(boxstyle="round,pad=0.2", fc="#f5f5f5", ec="#ddd", lw=0.8)

    # --- classify matching edges ---
    matches = []       # (src_idx, tgt_idx, score)
    deletions = []     # src indices → ε
    insertions = []    # tgt indices ← ε

    for s_node, t_node in matching.items():
        s_is_eps = s_node.startswith("ref_ε")
        t_is_eps = t_node.startswith("transc_ε")
        if s_is_eps and t_is_eps:
            continue
        if s_is_eps:
            insertions.append(int(t_node.split("_")[1]))
        elif t_is_eps:
            deletions.append(int(s_node.split("_")[1]))
        else:
            i = int(s_node.split("_")[1])
            j = int(t_node.split("_")[1])
            matches.append((i, j, G.edges[s_node, t_node]["score"]))

    deletions.sort()
    insertions.sort()

    # --- y positions: words top, ε-nodes padded at bottom ---
    total_rows = m + len(insertions)
    y = lambda idx: total_rows - 1 - idx

    src_y = {i: y(i) for i in range(m)}
    tgt_y = {j: y(j) for j in range(n)}
    eps_left_y = [y(m + k) for k in range(len(insertions))]
    eps_right_y = [y(n + k) for k in range(len(deletions))]

    # --- draw ---
    fig, ax = plt.subplots(figsize=(12, total_rows * 0.55 + 1))

    def draw_edge(y1, y2, color="#ddd", dashed=False):
        ax.plot([x_left + 0.10, x_right - 0.10], [y1, y2],
                color=color, linewidth=2 if not dashed else 1.2,
                linestyle="--" if dashed else "-",
                alpha=0.85, zorder=1 if dashed else 2)

    def draw_label(x, y_pos, text, color="#222", bold=True):
        ha = "right" if x < 0.5 else "left"
        offset = -0.02 if x < 0.5 else 0.02
        ax.text(x + offset, y_pos, text, fontsize=11, ha=ha, va="center",
                fontweight="bold" if bold else "normal", color=color)

    def draw_eps(x, y_pos):
        ax.text(x, y_pos, "ε", fontsize=11, ha="center", va="center",
                color="#aaa", fontweight="bold", bbox=eps_box)

    # real matches
    for i, j, score in matches:
        draw_edge(src_y[i], tgt_y[j], color=cmap(score))
        mx, my = 0.5, (src_y[i] + tgt_y[j]) / 2
        ax.text(mx, my, f"{score:.2f}", fontsize=7, ha="center", va="bottom",
                color="#555", zorder=3,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    # deletions: src word → ε (bottom right)
    for k, i in enumerate(deletions):
        draw_edge(src_y[i], eps_right_y[k], dashed=True)

    # insertions: ε (bottom left) → tgt word
    for k, j in enumerate(insertions):
        draw_edge(eps_left_y[k], tgt_y[j], dashed=True)

    # word labels
    del_set, ins_set = set(deletions), set(insertions)
    for i, word in enumerate(src):
        draw_label(x_left, src_y[i], word, color="#bbb" if i in del_set else "#222")
    for j, word in enumerate(tgt):
        draw_label(x_right, tgt_y[j], word, color="#bbb" if j in ins_set else "#222")

    # ε labels
    for ey in eps_left_y:
        draw_eps(x_left + 0.06, ey)
    for ey in eps_right_y:
        draw_eps(x_right - 0.06, ey)

    # headers & colorbar
    header_y = total_rows - 1 + 0.8
    ax.text(x_left - 0.02, header_y, "Source", fontsize=12, ha="right", va="center", fontstyle="italic", color="#888")
    ax.text(x_right + 0.02, header_y, "DIT", fontsize=12, ha="left", va="center", fontstyle="italic", color="#888")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04).set_label("Score", fontsize=10)

    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-0.8, total_rows + 0.5)
    ax.axis("off")
    ax.set_title("Bipartite Matching: Source → DIT", fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()