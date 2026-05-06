"""
Regional Distance from Hochdeutsch — Page 2.

Aggregate dialect distance per region, computed as mean per-sentence alignment
cost (substitution rows: 1 − sim; epsilon rows: λ = 0.45) divided by reference
word count. Restricted to STT4SG-350 train_balanced (~25k sentences per region;
sample-size-balanced subset of train_all) so per-region samples are comparable.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _data import (  # noqa: E402
    DAT_COLOR, DIT_COLOR, LAMBDA, REGIONS,
    load_alignments, load_balanced_paths,
)


@st.cache_data
def per_sentence_cost() -> pd.DataFrame:
    """Per-sentence alignment cost for both models, restricted to train_balanced paths."""
    balanced = load_balanced_paths()
    balanced_paths = set(balanced["path"])

    align = load_alignments()
    align = align[align["path"].isin(balanced_paths)].copy()
    align["cost"] = np.where(align["similarity"].notna(), 1 - align["similarity"], LAMBDA)
    align["is_ref"] = align["reference_word"].notna()

    grouped = (
        align.groupby(["path", "model"], observed=True)
        .agg(total_cost=("cost", "sum"), n_ref_words=("is_ref", "sum"))
        .reset_index()
    )
    grouped = grouped[grouped["n_ref_words"] > 0]
    grouped = grouped.merge(balanced, on="path", how="inner")
    grouped["cost_per_ref_word"] = grouped["total_cost"] / grouped["n_ref_words"]
    return grouped


@st.cache_data
def regional_summary() -> pd.DataFrame:
    """Per-region mean cost (DAT, DIT), DIT−DAT delta, and sample size."""
    df = per_sentence_cost()
    summary = (
        df.groupby(["dialect_region", "model"], observed=True)
        .agg(mean_cost=("cost_per_ref_word", "mean"),
             n_sentences=("path", "size"))
        .reset_index()
    )
    pivoted = summary.pivot(
        index="dialect_region", columns="model",
        values=["mean_cost", "n_sentences"],
    )
    out = pd.DataFrame({
        "DAT cost": pivoted[("mean_cost", "dialect-aware")],
        "DIT cost": pivoted[("mean_cost", "dialect-ignorant")],
        "delta (DIT − DAT)": pivoted[("mean_cost", "dialect-ignorant")] - pivoted[("mean_cost", "dialect-aware")],
        "n sentences": pivoted[("n_sentences", "dialect-aware")].astype(int),
    })
    return out.reindex([r for r in REGIONS if r in out.index]).reset_index()


# ── Page ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Regional Distance", layout="wide")
st.title("Regional Distance from Hochdeutsch")

st.markdown(
    "Aggregate dialect distance per region, measured as mean per-sentence alignment cost "
    f"(substitution: 1 − sim; ε rows: λ = {LAMBDA}) divided by reference word count. "
    "Computed on the **train_balanced** subset (~25k sentences per region; subset of train_all). "
    "Lower cost = closer to Hochdeutsch; higher DIT − DAT delta = stronger dialect signal."
)

with st.spinner("Computing per-sentence alignment costs…"):
    summary = regional_summary()
    per_sentence = per_sentence_cost()

# ── Headline plots ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), gridspec_kw={"height_ratios": [3, 2]})

regions = summary["dialect_region"].tolist()
x = np.arange(len(regions))
w = 0.4

axes[0].bar(x - w / 2, summary["DAT cost"], width=w, label="DAT", color=DAT_COLOR, alpha=0.9)
axes[0].bar(x + w / 2, summary["DIT cost"], width=w, label="DIT", color=DIT_COLOR, alpha=0.9)
axes[0].set_xticks(x)
axes[0].set_xticklabels(regions)
axes[0].set_ylabel("Mean cost per ref word")
axes[0].set_title("Mean alignment cost per region")
axes[0].legend(loc="upper right")
axes[0].grid(axis="y", alpha=0.3)

axes[1].bar(x, summary["delta (DIT − DAT)"], color=DIT_COLOR, alpha=0.9)
axes[1].set_xticks(x)
axes[1].set_xticklabels(regions)
axes[1].axhline(0, color="black", linewidth=0.6)
axes[1].set_ylabel("DIT − DAT")
axes[1].set_title("Dialect-specific distance per region (positive = DIT pays more cost than DAT)")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ── Table ────────────────────────────────────────────────────────────────────
st.markdown("### Per-region summary")
display_table = summary.copy()
for col in ("DAT cost", "DIT cost", "delta (DIT − DAT)"):
    display_table[col] = display_table[col].round(4)
st.dataframe(display_table, hide_index=True, use_container_width=True)

# ── Distribution ─────────────────────────────────────────────────────────────
st.markdown("### Per-sentence cost distribution")
st.caption(
    "Each box shows the within-region distribution of per-sentence alignment cost. "
    "Wide IQR = mixed dialect compliance across speakers; tight = consistent."
)
fig, ax = plt.subplots(figsize=(12, 4))
sns.boxplot(
    data=per_sentence,
    x="dialect_region", y="cost_per_ref_word",
    hue="model", order=regions,
    hue_order=["dialect-aware", "dialect-ignorant"],
    palette={"dialect-aware": DAT_COLOR, "dialect-ignorant": DIT_COLOR},
    ax=ax,
    fliersize=2,
)
ax.set_xlabel("")
ax.set_ylabel("Cost per ref word")
ax.legend(title="", loc="upper right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)