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

### Pages

- **Dialect Word Lexicon**: search any Standard German reference word and see its dialect variants from both
  DIT and DAT. Inspect each clip's hypotheses with word-level alignments.
- **Regional Distance**: which Swiss German dialect is most distant from Standard German?

Use the sidebar to navigate.
"""
)
