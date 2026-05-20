"""
Dialect Word Lexicon: Page 1.

Search a Standard German reference word, compare DIT and DAT across regions.

Run via the entry script: streamlit run visualization/Home.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _lexicon_detail as detail  # noqa: E402
import _lexicon_overview as overview  # noqa: E402
from _data import REGION_COLORS, REGIONS, CloudMode, joined_view  # noqa: E402


# --- Page 1 ---
st.set_page_config(page_title="Dialect Word Lexicon", layout="wide")
st.title("Dialect Word Lexicon")

# Sidebar filters
region_filter = st.sidebar.multiselect(
    "Filter regions",
    REGIONS,
    default=[],
    help="Leave empty to include all 7 regions. Pick specific regions to narrow the view.",
    placeholder="All regions",
)
selected_regions = region_filter if region_filter else REGIONS

_chip_css = "\n".join(
    f'.stMultiSelect [data-baseweb="tag"][aria-label*="{region}"],'
    f'.stMultiSelect [data-baseweb="tag"]:has([aria-label*="{region}"])'
    f' {{ background-color: {color} !important; color: white !important; }}'
    for region, color in REGION_COLORS.items()
)
st.markdown(f"<style>{_chip_css}</style>", unsafe_allow_html=True)

include_preterite = st.sidebar.toggle(
    "Include Preterite sentences",
    value=True,
    help="As Swiss German does not have the Preterite tense and replaces it with the Perfect, including such cases "
         "can introduce noise in the transcripts. Exclude them to focus on dialect specifics; include for more data.")

with st.spinner("Loading alignment data..."):
    df = joined_view(tuple(selected_regions))

if not include_preterite:
    df = df[~df["is_praeteritum"].fillna(False).astype(bool)]

st.sidebar.metric("Alignments", f"{len(df):,}")
st.sidebar.metric("Unique sentences", f"{df['path'].nunique():,}")

# Reference-word frequencies (substitution + deletion edges only: drop insertions where ref_word is NA (epsilon))
ref_counts = (
    df[df["reference_word"].notna()]
    .groupby("reference_word")
    .size()
    .sort_values(ascending=False)
)

# Search box
search_options = [""] + ref_counts.index.tolist()
selected_word = st.selectbox(
    f"Search a reference word ({len(ref_counts):,} unique) to see its dialect specific variants.",
    options=search_options,
    key="selected_word",
    placeholder="Start typing a Standard German word...",
)


def render_detail(df_view: pd.DataFrame, word: str, regions: list[str],
                  include_preterite: bool) -> None:
    """Per-reference-word detail view: header, hypothesis tables, charts, regional table, examples."""
    word_rows = df_view[(df_view["reference_word"] == word) & (df_view["model"] != "dat-dit")]
    detail.render_header(word, word_rows, regions)
    detail.render_word_chart(word_rows)
    detail.render_hypothesis_tables(word, word_rows, regions, include_preterite)
    detail.render_example_sentences(df_view, word_rows, word, regions, include_preterite)


def render_overview(df_view: pd.DataFrame, regions: list[str], include_preterite: bool) -> None:
    """Word-cloud overview of regionally distinctive substitution pairs."""
    mode_label = st.radio(
        "Compare",
        options=["Reference → DIT (Whisper)", "DAT (FHNW) → DIT (Whisper)"],
        horizontal=True,
        key="cloud_mode",
    )
    mode: CloudMode = "ref_dit" if mode_label.startswith("Reference") else "dat_dit"
    overview.render_caption(mode)
    table = overview.compute_top_table(df_view, regions, include_preterite, mode)
    if table.empty:
        st.warning("No regionally distinctive pairs in the selected regions.")
        return
    overview.render_word_cloud(table)
    overview.render_top_candidates_expander(table, mode)


if selected_word:
    render_detail(df, selected_word, selected_regions, include_preterite)
else:
    render_overview(df, selected_regions, include_preterite)