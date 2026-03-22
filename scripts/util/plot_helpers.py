import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


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