"""
Regional Distance from Standard German: Page 2.

Dialect distance per region, computed as the mean of (per-sentence total alignment cost / sentence's ref-word count)
across all sentences in the region.
Restricted to STT4SG-350 train_balanced (~25k sentences per region; sample-size-balanced
subset of train_all) so per-region samples are comparable.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _regional_distance as regional  # noqa: E402
from _data import load_alignments  # noqa: E402

# --- Page 2 ---
st.set_page_config(page_title="Regional Distance", layout="wide")
st.title("Regional Distance from Standard German")
regional.render_intro()

include_preterite = st.sidebar.toggle(
    "Include Preterite sentences",
    value=False,
    help="As Swiss German does not have the Preterite tense and replaces it with the Perfect, including such cases "
         "can introduce noise in the transcripts. Exclude them to focus on dialect specifics; include for more data.")

with st.spinner("Computing per-sentence alignment costs..."):
    summary = regional.regional_summary(include_preterite)
    per_sentence = regional.per_sentence_cost()
    if not include_preterite:
        per_sentence = per_sentence[~per_sentence["is_praeteritum"].fillna(False).astype(bool)]

align_in_view = load_alignments()
align_in_view = align_in_view[align_in_view["path"].isin(set(per_sentence["path"]))]
st.sidebar.metric("Alignments", f"{len(align_in_view):,}")
st.sidebar.metric("Unique sentences", f"{per_sentence['path'].nunique():,}")

regions_sorted = summary["dialect_region"].tolist()
regional.render_headline_plots(summary, regions_sorted)
regional.render_summary_table(summary)
regional.render_cost_distribution(per_sentence, regions_sorted)
