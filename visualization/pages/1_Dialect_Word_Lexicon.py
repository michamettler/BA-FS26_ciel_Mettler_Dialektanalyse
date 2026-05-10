"""
Dialect Word Lexicon: Page 1.

Search a Standard German reference word, compare DIT and DAT across regions.

Run via the entry script: streamlit run visualization/Home.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _lexicon_detail as detail  # noqa: E402
import _lexicon_overview as overview  # noqa: E402
from _data import REGIONS, joined_view  # noqa: E402


# --- Page 1 ---
st.set_page_config(page_title="Dialect Word Lexicon", layout="wide")
st.title("Dialect Word Lexicon")

# Sidebar filters
selected_regions = st.sidebar.multiselect("Regions", REGIONS, default=REGIONS)
include_preterite = st.sidebar.toggle(
    "Include Preterite sentences",
    value=True,
    help="As Swiss German does not have the Preterite tense and replaces it with the Perfect, including such cases "
         "can introduce noise in the transcripts. Exclude them to focus on dialect specifics; include for more data.")

if not selected_regions:
    st.info("Select at least one region in the sidebar to begin.")
    st.stop()

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

# Word-cloud-only slider
min_count_for_word_cloud = None
if not selected_word:
    min_count_for_word_cloud = st.sidebar.slider(
        "Word cloud: minimum word occurrences",
        1, 50, 5,
        help="Minimum occurrences of a reference word to be considered for the word cloud.")


def render_detail(df_view: pd.DataFrame, word: str, regions: list[str]) -> None:
    """Per-reference-word detail view: header, hypothesis tables, charts, regional table, examples."""
    word_rows = df_view[df_view["reference_word"] == word]
    detail.render_header(word, word_rows, regions)
    detail.render_hypothesis_tables(word_rows)
    detail.render_word_chart(word_rows)
    detail.render_regional_similarity_table(word_rows)
    detail.render_regional_similarity_chart(word_rows)
    detail.render_example_sentences(df_view, word_rows, word)


def render_overview(df_view: pd.DataFrame, min_count_threshold: int) -> None:
    """Word-cloud overview of dialect-candidate ref words: caption, cloud, and top-table expander."""
    overview.render_caption()
    table = overview.compute_top_table(df_view, min_count_threshold)
    if table.empty:
        st.warning(
            "No dialect-candidate words above the frequency threshold with positive delta. "
            "Lower the minimum-occurrences slider in the sidebar."
        )
        return
    overview.render_word_cloud(table)
    overview.render_top_candidates_expander(table)


if selected_word:
    render_detail(df, selected_word, selected_regions)
else:
    render_overview(df, min_count_for_word_cloud)