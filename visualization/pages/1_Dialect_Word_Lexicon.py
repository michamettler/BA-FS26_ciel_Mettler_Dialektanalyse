"""
Dialect Word Lexicon: Page 1.

Search a Standard German reference word, see how DIT (dialect-ignorant Whisper) and
DAT (dialect-aware FHNW STT4SG) transcribed it across regions. Click-through
from a word cloud or autocomplete search.

Run via the entry script: streamlit run visualization/Home.py
"""
import html
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "domain"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "utils"))
from _data import DAT_COLOR, DIT_COLOR, REGIONS, joined_view  # noqa: E402
from bipartite_matching import build_full_bipartite_graph, solve_matching  # noqa: E402
from plot_helpers import plot_reduced_bipartite_graph_with_matching  # noqa: E402
from preprocessing import clean_word  # noqa: E402
from word_similarity_calculator import WordSimilarityCalculator  # noqa: E402

# Calibrated hyperparameters: must match build_alignment_table.py.
_ALPHA = 0.85
_LAMBDA = 0.45
_USE_GLOBAL_LEX_NORM = False
_USE_SQUARED_POSITIONAL = True


@st.cache_resource(max_entries=64, show_spinner=False)
def _reduced_graph_figure(reference: str, hypothesis: str):
    """Run the calibrated solver on one (ref, hyp) pair and return a matplotlib Figure
    of the reduced bipartite matching. None when both sides are empty after cleaning."""
    ref = clean_word(reference).split()
    hyp = clean_word(hypothesis).split()
    if not ref and not hyp:
        return None
    calc = WordSimilarityCalculator(
        sent_len=max(len(ref), len(hyp)),
        alpha=_ALPHA, lambda_=_LAMBDA,
        use_global_lexical_normalization=_USE_GLOBAL_LEX_NORM,
        use_squared_positional=_USE_SQUARED_POSITIONAL,
    )
    G = build_full_bipartite_graph(ref, hyp, calc)
    matching = solve_matching(G)
    return plot_reduced_bipartite_graph_with_matching(G, matching)

# --- Page ---
st.set_page_config(page_title="Dialect Word Lexicon", layout="wide")
st.title("Dialect Word Lexicon")

# Sidebar filters
selected_regions = st.sidebar.multiselect("Regions", REGIONS, default=REGIONS)
min_count = st.sidebar.slider("Word cloud: minimum occurrences", 1, 50, 5,
                              help="Sample-size guard for the aggregate delta ranking. "
                                   "Does not filter the search box or detail views.")
include_praet = st.sidebar.toggle("Include Preterite sentences", value=True,
                                  help="Preterite forms are themselves dialect-distinctive "
                                       "(Swiss German rarely uses them): keep them on by default.")

if not selected_regions:
    st.info("Select at least one region in the sidebar to begin.")
    st.stop()

with st.spinner("Loading alignment data…"):
    df = joined_view(tuple(selected_regions))

if not include_praet:
    df = df[~df["is_praeteritum"].fillna(False).astype(bool)]

st.sidebar.metric("Rows in view", f"{len(df):,}")
st.sidebar.metric("Unique sentences", f"{df['path'].nunique():,}")

# Reference-word frequencies (substitution + deletion edges only: drop insertions where ref_word is NA)
ref_counts = (
    df[df["reference_word"].notna()]
    .groupby("reference_word")
    .size()
    .sort_values(ascending=False)
)

# Search box: every ref word is searchable; the alignment is local, no frequency floor needed.
search_options = [""] + ref_counts.index.tolist()
selected_word = st.selectbox(
    f"Search a reference word ({len(ref_counts):,} unique)",
    options=search_options,
    key="selected_word",
    placeholder="Start typing a Standard German word…",
)


def _back_to_cloud():
    """Clear the search selection so the next render returns to the word-cloud overview."""
    st.session_state["selected_word"] = ""


def _render_overview(df: pd.DataFrame, min_count_threshold: int) -> None:
    """Word cloud of best dialect-candidate ref words: highest mean DAT − DIT similarity delta."""
    st.markdown("### Dialect-distinctive vocabulary")
    st.caption(
        "Reference words ranked by **mean DAT sim − mean DIT sim** (similarity delta) "
        "across the selected regions, restricted to words above the minimum-occurrences "
        "threshold (sidebar). Larger = DIT struggles more than DAT on this word: "
        "the strongest dialect-candidate signal. "
        "Click an entry in the autocomplete above to drill into a word."
    )

    real = df[df["hypothesis_word"].notna() & df["reference_word"].notna()]
    if real.empty:
        st.warning("No alignment data available for the selected filters.")
        return

    sim = real.pivot_table(
        index="reference_word", columns="model", values="similarity", aggfunc="mean"
    )
    counts = real[real["model"] == "dialect-aware"].groupby("reference_word").size()
    delta = (sim.get("dialect-aware") - sim.get("dialect-ignorant")).dropna()
    delta = delta[counts >= min_count_threshold]
    delta = delta[delta > 0].sort_values(ascending=False)

    if delta.empty:
        st.warning(
            "No dialect-candidate words above the frequency threshold with positive delta. "
            "Lower the minimum-occurrences slider in the sidebar."
        )
        return

    top = delta.head(200)
    cloud_data = [
        {"name": str(word), "value": round(float(value), 4)}
        for word, value in top.items()
    ]
    option = {
        "tooltip": {"show": True},
        "series": [{
            "type": "wordCloud",
            "shape": "circle",
            "left": "center",
            "top": "center",
            "width": "100%",
            "height": "100%",
            "sizeRange": [14, 80],
            "rotationRange": [-30, 30],
            "rotationStep": 15,
            "gridSize": 8,
            "drawOutOfBound": False,
            "emphasis": {
                "focus": "self",
                "textStyle": {"shadowBlur": 8, "shadowColor": "#999"},
            },
            "data": cloud_data,
        }],
    }
    st_echarts(options=option, height="520px")

    with st.expander("Top dialect-candidate words"):
        display = pd.DataFrame({
            "word": top.index,
            "delta": top.round(3).values,
            "count": [int(counts[w]) for w in top.index],
            "DAT sim": [round(sim.loc[w, "dialect-aware"], 3) for w in top.index],
            "DIT sim": [round(sim.loc[w, "dialect-ignorant"], 3) for w in top.index],
        })
        st.dataframe(display, use_container_width=True, hide_index=True)


def _hypothesis_table(slice_df: pd.DataFrame) -> pd.DataFrame:
    """Hypothesis variants with count, mean similarity, and weighted divergence.

    `divergence = count * (1 - mean_similarity)`: surfaces variants that are both
    frequent AND linguistically distant from the reference.
    """
    out = (
        slice_df.groupby("hypothesis_word", dropna=False)
        .agg(count=("path", "size"), mean_similarity=("similarity", "mean"))
        .reset_index()
    )
    out["count * (1 - mean_similarity)"] = (out["count"] * (1 - out["mean_similarity"])).round(2)
    out["mean_similarity"] = out["mean_similarity"].round(3)
    return out.sort_values("count", ascending=False)


def _render_word_charts(word_rows: pd.DataFrame) -> None:
    """Side-by-side: regional DIT-variant breakdown + per-region similarity delta bar."""
    region_order = [r for r in REGIONS if r in word_rows["dialect_region"].unique()]
    if not region_order:
        return

    # regional DIT-variant word breakdown (stacked bar)
    dit = word_rows[word_rows["model"] == "dialect-ignorant"].copy()
    dit["variant"] = dit["hypothesis_word"].fillna("(deletion)")

    top_n = 6
    top_variants = dit["variant"].value_counts().head(top_n).index.tolist()
    dit["variant_grouped"] = dit["variant"].where(dit["variant"].isin(top_variants), "(other)")

    counts = (
        dit.groupby(["dialect_region", "variant_grouped"], observed=True)
        .size()
        .reset_index(name="count")
    )

    variant_chart = (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("dialect_region:N", sort=region_order, title=None,
                    axis=alt.Axis(labelAngle=0, labelOverlap=False)),
            y=alt.Y("count:Q", title="DIT alignments"),
            color=alt.Color("variant_grouped:N",
                            scale=alt.Scale(scheme="category10"),
                            legend=alt.Legend(title="DIT variant")),
            tooltip=[
                alt.Tooltip("dialect_region:N", title="Region"),
                alt.Tooltip("variant_grouped:N", title="Variant"),
                alt.Tooltip("count:Q", title="Count"),
            ],
        )
        .properties(height=340, title="DIT variants per region")
    )

    # per-region similarity delta bar
    real = word_rows[word_rows["hypothesis_word"].notna()]
    delta_data = (
        real.pivot_table(index="dialect_region", columns="model", values="similarity", aggfunc="mean")
        .reindex(region_order)
    )
    delta_data["delta"] = delta_data.get("dialect-aware") - delta_data.get("dialect-ignorant")
    delta_df = delta_data.reset_index()[["dialect_region", "delta"]].dropna()

    delta_bar = (
        alt.Chart(delta_df)
        .mark_bar(opacity=0.9)
        .encode(
            x=alt.X("dialect_region:N", sort=region_order, title=None,
                    axis=alt.Axis(labelAngle=0, labelOverlap=False)),
            y=alt.Y("delta:Q", title="Mean DAT − Mean DIT similarity"),
            color=alt.condition(alt.datum.delta >= 0, alt.value(DIT_COLOR), alt.value(DAT_COLOR)),
            tooltip=[
                alt.Tooltip("dialect_region:N", title="Region"),
                alt.Tooltip("delta:Q", format=".3f", title="Delta"),
            ],
        )
        .properties(height=280, title="Similarity delta per region (positive = DIT diverges more)")
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeWidth=0.6, color="black").encode(y="y:Q")

    st.altair_chart(variant_chart, use_container_width=True)
    st.altair_chart(delta_bar + zero, use_container_width=True)


# --- Alignment visualization helpers ---
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
    "padding: 4px 12px; font-weight: 600; text-align: center; color: #888; "
    "white-space: nowrap; border-bottom: 1px solid #eee;"
)


def _render_alignment_html(columns: list[dict], hyp_label: str, searched_word: str | None = None) -> str:
    """Build a 2-row HTML alignment table; highlights the column whose ref word == searched_word."""
    ref_cells = [f'<td style="{_LABEL_STYLE}">Reference</td>']
    hyp_cells = [f'<td style="{_LABEL_STYLE}">{html.escape(hyp_label)}</td>']
    for col in columns:
        is_searched = searched_word is not None and col["ref"] == searched_word
        bg = " background-color: rgba(247, 181, 0, 0.4);" if is_searched else ""
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
    st.button("← Back to word cloud", on_click=_back_to_cloud, key="back_btn")
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

    _render_word_charts(word_rows)

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

    # sentences with word-level alignment, grouped by region, collapsed by default.
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

    st.markdown(f"**Example sentences with word-level alignment**: {len(unique_paths):,} sentences")

    rows_by_path = dict(tuple(
        df_view[df_view["path"].isin(set(unique_paths["path"]))]
        .groupby("path", sort=False)
    ))

    page_size = 15
    for region in unique_paths["dialect_region"].unique():
        region_paths = unique_paths[unique_paths["dialect_region"] == region].reset_index(drop=True)
        n_region = len(region_paths)
        st.markdown(f"#### {region} ({n_region})")

        n_pages = max(1, (n_region + page_size - 1) // page_size)
        if n_pages > 1:
            page = st.selectbox(
                "Page",
                options=list(range(1, n_pages + 1)),
                format_func=lambda p, n=n_pages, r=n_region: (
                    f"Page {p} of {n} (sentences {(p - 1) * page_size + 1}–{min(p * page_size, r)})"
                ),
                key=f"page_{word}_{region}",
                label_visibility="collapsed",
            )
        else:
            page = 1
        start = (page - 1) * page_size
        page_paths = region_paths.iloc[start:start + page_size]

        for _, row in page_paths.iterrows():
            path = row["path"]
            sentence_rows = rows_by_path[path]
            with st.expander(f"{row['gender']} · {row['age']} · …{path[-12:]}"):
                st.markdown(f"**Clip ID:** `{path}`")
                st.markdown(
                    f"**Reference:** {row['reference']}  \n"
                    f"**Hypothesis DIT:** {row['dit_hypothesis']}  \n"
                    f"**Hypothesis DAT:** {row['dat_hypothesis']}"
                )
                dat_cols = _alignment_columns(sentence_rows[sentence_rows["model"] == "dialect-aware"])
                dit_cols = _alignment_columns(sentence_rows[sentence_rows["model"] == "dialect-ignorant"])
                st.markdown(_render_alignment_html(dit_cols, "Hypothesis DIT", searched_word=word), unsafe_allow_html=True)
                st.markdown(_render_alignment_html(dat_cols, "Hypothesis DAT", searched_word=word), unsafe_allow_html=True)

                if st.checkbox("Show technical alignment", key=f"graph_{path}"):
                    fig_dit = _reduced_graph_figure(row["reference"], row["dit_hypothesis"])
                    if fig_dit is not None:
                        st.markdown("**DIT: reduced bipartite matching**")
                        st.pyplot(fig_dit)
                    fig_dat = _reduced_graph_figure(row["reference"], row["dat_hypothesis"])
                    if fig_dat is not None:
                        st.markdown("**DAT: reduced bipartite matching**")
                        st.pyplot(fig_dat)


if selected_word:
    _render_detail(df, selected_word)
else:
    _render_overview(df, min_count)
