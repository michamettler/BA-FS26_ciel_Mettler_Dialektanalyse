"""
Dialect Word Lexicon — Page 1.

Search a Hochdeutsch reference word, see how DIT (dialect-ignorant Whisper) and
DAT (dialect-aware FHNW STT4SG) transcribed it across regions. Click-through
from a word cloud or autocomplete search.

Run: streamlit run visualization/Dialect_Word_Lexicon.py
"""
import html
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data import DAT_COLOR, DIT_COLOR, REGIONS, joined_view  # noqa: E402

# ── Page ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Dialect Word Lexicon", layout="wide")
st.title("Dialect Word Lexicon")

# Sidebar filters
selected_regions = st.sidebar.multiselect("Regions", REGIONS, default=REGIONS)
min_count = st.sidebar.slider("Minimum occurrences per reference word", 1, 50, 5)
max_examples = st.sidebar.slider("Max examples in detail view (per-region cap applied)", 50, 1000, 200, step=50)

if not selected_regions:
    st.info("Select at least one region in the sidebar to begin.")
    st.stop()

with st.spinner("Loading alignment data…"):
    df = joined_view(tuple(selected_regions))

st.sidebar.metric("Rows in view", f"{len(df):,}")
st.sidebar.metric("Unique sentences", f"{df['path'].nunique():,}")

# Reference-word frequencies (substitution + deletion edges only — drop insertions where ref_word is NA)
ref_counts = (
    df[df["reference_word"].notna()]
    .groupby("reference_word")
    .size()
    .sort_values(ascending=False)
)
eligible_words = ref_counts[ref_counts >= min_count]

# Search box (autocomplete via st.selectbox)
search_options = [""] + eligible_words.index.tolist()
selected_word = st.selectbox(
    f"Search a reference word ({len(eligible_words):,} eligible above the frequency threshold)",
    options=search_options,
    index=0,
    placeholder="Start typing a Hochdeutsch word…",
)


def _render_overview(eligible: pd.Series) -> None:
    st.markdown("### Reference vocabulary overview")
    st.caption(
        "Word cloud of the most frequent Hochdeutsch reference words across the selected regions. "
        "Click an entry in the autocomplete above to drill into a word."
    )
    if eligible.empty:
        st.warning("No reference words meet the frequency threshold. Lower it in the sidebar.")
        return

    top = eligible.head(200)  # limit for readability
    wc = WordCloud(width=1400, height=600, background_color="white", colormap="viridis")
    wc.generate_from_frequencies(top.to_dict())
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)

    with st.expander("Top reference words by frequency"):
        st.dataframe(
            top.reset_index().rename(columns={"reference_word": "word", 0: "count"}),
            use_container_width=True,
            hide_index=True,
        )


def _hypothesis_table(slice_df: pd.DataFrame) -> pd.DataFrame:
    """Group by hypothesis, return count + mean similarity, sorted by count desc."""
    return (
        slice_df.groupby("hypothesis_word", dropna=False)
        .agg(count=("path", "size"), mean_similarity=("similarity", "mean"))
        .sort_values("count", ascending=False)
        .reset_index()
    )


def _render_similarity_plots(word_rows: pd.DataFrame) -> None:
    """Side-by-side: per-alignment similarity strip plot + per-region delta bar."""
    real = word_rows[word_rows["hypothesis_word"].notna()]
    if real.empty:
        return

    region_order = [r for r in REGIONS if r in real["dialect_region"].unique()]
    if not region_order:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))

    sns.stripplot(
        data=real, x="dialect_region", y="similarity",
        hue="model", order=region_order,
        hue_order=["dialect-aware", "dialect-ignorant"],
        palette={"dialect-aware": DAT_COLOR, "dialect-ignorant": DIT_COLOR},
        dodge=True, alpha=0.65, size=5, jitter=0.25, ax=axes[0],
    )
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Similarity")
    axes[0].set_title("Per-alignment similarity by region")
    axes[0].set_ylim(-0.02, 1.05)
    axes[0].legend(title="", loc="lower right", fontsize=9)
    axes[0].tick_params(axis="x", rotation=20)

    delta = (
        real.pivot_table(index="dialect_region", columns="model", values="similarity", aggfunc="mean")
        .reindex(region_order)
    )
    delta["delta"] = delta.get("dialect-aware") - delta.get("dialect-ignorant")
    bar_colors = [DIT_COLOR if v >= 0 else DAT_COLOR for v in delta["delta"].fillna(0)]
    axes[1].bar(range(len(delta)), delta["delta"], color=bar_colors, alpha=0.85)
    axes[1].set_xticks(range(len(delta)))
    axes[1].set_xticklabels(region_order, rotation=20)
    axes[1].set_ylabel("Mean DAT − Mean DIT")
    axes[1].set_title("Dialect signal per region (positive = DIT diverges more)")
    axes[1].axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── Alignment visualization helpers ──────────────────────────────────────────
def _alignment_columns(rows: pd.DataFrame) -> list[dict]:
    """Convert per-(path, model) alignment rows into ordered display columns.

    Substitutions and deletions are placed in reference-index order; insertions
    are interleaved by their hypothesis index relative to surrounding substitutions.
    """
    records = rows.to_dict("records")
    subs_dels = sorted(
        [r for r in records if pd.notna(r["reference_index"])],
        key=lambda r: r["reference_index"],
    )
    inss = sorted(
        [r for r in records if pd.isna(r["reference_index"])],
        key=lambda r: r["hypothesis_index"],
    )

    columns: list[dict] = []
    ins_iter = iter(inss)
    next_ins = next(ins_iter, None)

    for r in subs_dels:
        if pd.isna(r["hypothesis_index"]):
            columns.append({"ref": r["reference_word"], "hyp": "ε", "kind": "deletion"})
            continue
        while next_ins is not None and next_ins["hypothesis_index"] < r["hypothesis_index"]:
            columns.append({"ref": "ε", "hyp": next_ins["hypothesis_word"], "kind": "insertion"})
            next_ins = next(ins_iter, None)
        kind = "match" if r["reference_word"] == r["hypothesis_word"] else "substitution"
        columns.append({"ref": r["reference_word"], "hyp": r["hypothesis_word"], "kind": kind})

    while next_ins is not None:
        columns.append({"ref": "ε", "hyp": next_ins["hypothesis_word"], "kind": "insertion"})
        next_ins = next(ins_iter, None)

    return columns


_LABEL_STYLE = (
    "padding: 4px 12px 4px 0; font-weight: 600; text-align: right; color: #555; "
    "white-space: nowrap; border-bottom: 1px solid #eee;"
)


def _render_alignment_html(columns: list[dict], hyp_label: str, searched_word: str | None = None) -> str:
    """Build a 2-row HTML alignment table; highlights the column whose ref word == searched_word."""
    ref_cells = [f'<td style="{_LABEL_STYLE}">Reference</td>']
    hyp_cells = [f'<td style="{_LABEL_STYLE}">{html.escape(hyp_label)}</td>']
    for col in columns:
        is_searched = searched_word is not None and col["ref"] == searched_word
        bg = " background-color: #fff3cd;" if is_searched else ""
        base = f"padding: 4px 10px;{bg} border-bottom: 1px solid #eee;"
        ref_style = base + (" color: #999; font-style: italic;" if col["kind"] == "insertion" else "")
        hyp_style = base + (" color: #999; font-style: italic;" if col["kind"] == "deletion" else "")
        ref_cells.append(f'<td style="{ref_style}">{html.escape(str(col["ref"]))}</td>')
        hyp_cells.append(f'<td style="{hyp_style}">{html.escape(str(col["hyp"]))}</td>')

    table_style = "border-collapse: collapse; font-family: ui-monospace, monospace; font-size: 0.95em;"
    return (
        f'<div style="overflow-x: auto; margin-bottom: 6px;">'
        f'<table style="{table_style}">'
        f'<tr>{"".join(ref_cells)}</tr>'
        f'<tr>{"".join(hyp_cells)}</tr>'
        f'</table></div>'
    )


def _render_detail(df_view: pd.DataFrame, word: str) -> None:
    st.markdown(f"### `{word}`")

    word_rows = df_view[df_view["reference_word"] == word]
    n_total = len(word_rows)
    n_sentences = word_rows["path"].nunique()
    st.caption(f"{n_total:,} alignment rows across {n_sentences:,} sentences "
               f"(in {len(selected_regions)} selected region(s))")

    col_dat, col_dit = st.columns(2)
    with col_dat:
        st.markdown("**Dialect-aware (DAT, FHNW STT4SG)**")
        dat_table = _hypothesis_table(word_rows[word_rows["model"] == "dialect-aware"])
        st.dataframe(dat_table, use_container_width=True, hide_index=True)
    with col_dit:
        st.markdown("**Dialect-ignorant (DIT, Whisper-large-v2)**")
        dit_table = _hypothesis_table(word_rows[word_rows["model"] == "dialect-ignorant"])
        st.dataframe(dit_table, use_container_width=True, hide_index=True)

    real = word_rows[word_rows["hypothesis_word"].notna()]
    if not real.empty:
        delta = (
            real.pivot_table(
                index="dialect_region", columns="model",
                values="similarity", aggfunc="mean",
            )
            .reindex(columns=["dialect-aware", "dialect-ignorant"])
        )
        delta["delta (DAT − DIT)"] = delta["dialect-aware"] - delta["dialect-ignorant"]
        st.markdown("**Mean similarity per region**")
        st.dataframe(
            delta.round(3).reset_index(),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Higher delta = DIT struggles more than DAT on this word in that region "
            "→ stronger candidate for a region-specific dialect transformation."
        )

    _render_similarity_plots(word_rows)

    # Example sentences with word-level alignment, grouped by region, collapsed by default.
    unique_paths = (
        word_rows.drop_duplicates("path")[
            ["path", "dialect_region", "gender", "age",
             "reference", "dat_hypothesis", "dit_hypothesis"]
        ]
        .assign(_region_idx=lambda d: d["dialect_region"].map(
            lambda r: REGIONS.index(r) if r in REGIONS else len(REGIONS)
        ))
        .sort_values(["_region_idx", "path"])
        .reset_index(drop=True)
    )

    n_regions_present = unique_paths["dialect_region"].nunique()
    per_region_cap = max(1, max_examples // max(1, n_regions_present))
    shown = unique_paths.groupby("dialect_region", sort=False).head(per_region_cap).reset_index(drop=True)

    n_total, n_shown = len(unique_paths), len(shown)
    header = f"**Example sentences with word-level alignment** — {n_shown:,} of {n_total:,} sentences"
    if n_shown < n_total:
        header += f" (capped at {per_region_cap} per region; raise the limit in the sidebar to see more)"
    st.markdown(header)

    rows_by_path = dict(tuple(
        df_view[df_view["path"].isin(set(shown["path"]))]
        .groupby("path", sort=False)
    ))

    prev_region = None
    for _, row in shown.iterrows():
        region = row["dialect_region"]
        if region != prev_region:
            n_region_shown = (shown["dialect_region"] == region).sum()
            n_region_total = (unique_paths["dialect_region"] == region).sum()
            label = f"{region} ({n_region_shown}"
            if n_region_shown < n_region_total:
                label += f" of {n_region_total}"
            label += ")"
            st.markdown(f"#### {label}")
            prev_region = region

        path = row["path"]
        sentence_rows = rows_by_path[path]
        with st.expander(f"{row['gender']} · {row['age']} · …{path[-12:]}"):
            st.markdown(f"**Clip ID:** `{path}`")
            st.markdown(
                f"**Reference:** {row['reference']}  \n"
                f"**Hypothesis DAT:** {row['dat_hypothesis']}  \n"
                f"**Hypothesis DIT:** {row['dit_hypothesis']}"
            )
            dat_cols = _alignment_columns(sentence_rows[sentence_rows["model"] == "dialect-aware"])
            dit_cols = _alignment_columns(sentence_rows[sentence_rows["model"] == "dialect-ignorant"])
            st.markdown(_render_alignment_html(dat_cols, "Hypothesis DAT", searched_word=word), unsafe_allow_html=True)
            st.markdown(_render_alignment_html(dit_cols, "Hypothesis DIT", searched_word=word), unsafe_allow_html=True)


if selected_word:
    _render_detail(df, selected_word)
else:
    _render_overview(eligible_words)
