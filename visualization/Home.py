"""
Swiss German Dialect Analysis — landing page.

Run: streamlit run visualization/Home.py
"""
import streamlit as st

st.set_page_config(page_title="Swiss German Dialect Analysis", layout="wide")
st.title("Swiss German Dialect Analysis")

st.markdown(
    """
This tool explores **dialect-distinctive transformations in Swiss German speech** by
comparing two automatic speech recognition (ASR) systems against the canonical
Hochdeutsch reference:

- **DAT — Dialect-Aware** (FHNW STT4SG): trained on Swiss German; produces clean Hochdeutsch.
- **DIT — Dialect-Ignorant** (Whisper-large-v2): a general-purpose ASR; produces phonetic
  Hochdeutsch attempts when it hears dialect.

The gap between the two — *what DIT got wrong but DAT got right* — surfaces
dialect-specific transformations.

### Pages

- **Dialect Word Lexicon** — search any Hochdeutsch reference word and see how each region's
  speakers transformed it. Inspect example sentences with word-level alignments.
- **Regional Distance** — aggregate dialect distance per region. Which dialect is most
  distant from Hochdeutsch?

Use the sidebar to navigate.
"""
)