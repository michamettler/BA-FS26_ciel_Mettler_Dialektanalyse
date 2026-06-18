"""
Swiss German Dialect Analysis: landing page.

Run: streamlit run visualization/Home.py
"""
import streamlit as st

st.set_page_config(page_title="Swiss German Dialect Analysis", layout="wide")
st.title("Swiss German Dialect Analysis")

st.markdown(
    """
This tool explores **dialect-specific words in Swiss German speech** by
comparing two automatic speech recognition (ASR) systems against the Standard German reference:

- **DAT: Dialect-Aware-Transcript** (FHNW STT4SG): trained on Swiss German; produces clean Standard German.
- **DIT: Dialect-Ignorant-Transcript** (Whisper-large-v2): a general-purpose ASR; produces phonetic
  Standard German approximations of dialect speech.

The gap between the two — *what DIT got wrong but DAT got right* — indicates whether a variant is
dialect-specific.

### Datasets

Two Swiss German speech corpora feed the analysis. Each clip has been transcribed by both
ASR systems, and word-level alignments between the Standard German reference and each
transcript are precomputed by the bipartite matching solver (see
`experiments/analysis/build_alignment_table.py`).

- **STT4SG-350** (FHNW): ~198k usable sentences across the 7 dialect regions. Ships with a
  curated `train_balanced` subset (~25k sentences per region) used by Regional Distance for a
  statistically balanced per-region comparison.
- **SDS-200**: ~95k usable sentences across the same 7 regions, after the same usability filter.
  No curated balanced subset, so Regional Distance falls back to the full filtered set and shows
  a caveat banner.

Use the **Dataset** selector in each page's sidebar to switch between them. A third
**Combined** option concatenates the two datasets and treats them as a single sample for
downstream aggregations (per-region counts, TF-IDF, mean costs). Recording conditions differ
between corpora, so patterns confirmed in both are more robust than patterns from a single
dataset.

### Pages

- **Dialect Word Lexicon**: search any Standard German reference word and see its dialect variants from both
  DIT and DAT. Inspect each clip's hypotheses with word-level alignments.
- **Regional Distance**: which Swiss German dialect is most distant from Standard German?

Use the sidebar to navigate.
"""
)
