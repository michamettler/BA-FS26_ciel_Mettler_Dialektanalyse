"""
Regional Distance from Standard German: Page 2.

Dialect distance per region, computed as the mean of (per-sentence total alignment cost / sentence's ref-word count)
across all sentences in the region. STT4SG-350 restricts to the train_balanced subset
(~25k sentences per region) for comparable samples; SDS-200 has no balanced subset and runs on the
full filtered set (caveat banner shown).
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _regional_distance as regional  # noqa: E402
from _auth import require_password  # noqa: E402
from _data import (  # noqa: E402
    DATASET_CHOICES, DEFAULT_DATASET,
    load_alignments, load_balanced_paths,
)

# --- Page 2 ---
st.set_page_config(page_title="Regional Distance", layout="wide")
require_password()
st.title("Regional Distance from Standard German")

dataset = st.sidebar.selectbox(
    "Dataset",
    DATASET_CHOICES,
    index=DATASET_CHOICES.index(DEFAULT_DATASET),
    key="selected_dataset",
    help="Switch between STT4SG-350 and SDS-200.",
)
uses_balanced = load_balanced_paths(dataset) is not None
regional.render_intro(uses_balanced)
if not uses_balanced:
    st.warning(
        f"**{dataset}** has no curated balanced subset. Per-region sentence counts vary "
        "considerably, so read per-region means as descriptive, not statistically balanced."
    )

include_preterite = st.sidebar.toggle(
    "Include Preterite sentences",
    value=False,
    help="As Swiss German does not have the Preterite tense and replaces it with the Perfect, including such cases "
         "can introduce noise in the transcripts. Exclude them to focus on dialect specifics; include for more data.")

with st.spinner("Computing per-sentence alignment costs..."):
    summary = regional.regional_summary(include_preterite, dataset)
    per_sentence = regional.per_sentence_cost(dataset)
    if not include_preterite:
        per_sentence = per_sentence[~per_sentence["is_praeteritum"].fillna(False).astype(bool)]

align_in_view = load_alignments(dataset)
align_in_view = align_in_view[align_in_view["path"].isin(set(per_sentence["path"]))]
st.sidebar.metric("Alignments", f"{len(align_in_view):,}")
st.sidebar.metric("Unique sentences", f"{per_sentence['path'].nunique():,}")

regions_sorted = summary["dialect_region"].tolist()
regional.render_headline_plots(summary, regions_sorted)
regional.render_summary_table(summary, uses_balanced)
regional.render_cost_distribution(per_sentence, regions_sorted)
