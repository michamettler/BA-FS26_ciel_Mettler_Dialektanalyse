# BA-FS26_ciel_Mettler_Dialektanalyse

Source code, transcripts, and analysis artifacts of the bachelor's thesis
*Word Alignment for Swiss German ASR-Based Dialect Analysis: A Weighted Bipartite Matching Approach* (ZHAW, FS26).

The pipeline compares two automatic speech recognition systems against the Standard German reference:

- **DAT** (Dialect-Aware Transcript): FHNW STT4SG, trained on Swiss German;
  produces clean Standard German.
- **DIT** (Dialect-Ignorant Transcript): Whisper large-v2, general-purpose ASR;
  produces phonetic Standard German approximations of dialect speech.

The gap between the two — what DIT got wrong but DAT got right — surfaces dialectal variation.

---

## Setup

Python 3.12+ is required.

### macOS

```bash
brew install ffmpeg

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Windows

```bash
winget install ffmpeg

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

### Using uv

[uv](https://github.com/astral-sh/uv) is an alternative to the venv + pip flow above
and handles Python version selection in one step. After installing ffmpeg:

```bash
uv venv --python 3.12
source .venv/bin/activate      # use .venv\Scripts\activate on Windows

uv pip install -r requirements.txt
```

---

## Key Artifacts

All of the pipeline's main outputs are committed to the repository, so no
re-transcription or re-alignment is required to use them:

- **Transcripts** (DAT and DIT): [`transcripts/`](transcripts). One TSV per split per model.
- **Word-level alignment tables**: [`experiments/analysis/`](experiments/analysis). Three
  Parquets covering REF ↔ DAT, REF ↔ DIT, and DAT ↔ DIT for `train_all`.
- **Dialect-word lexicon**: [`experiments/lexicon/`](experiments/lexicon). Top regionally distinctive
  substitution pairs for both alignment directions, ranked by TF–IDF.

File naming and column structure are documented at each artifact's location.
The full reproduction procedure (calibration => alignment => visualization) is
described in *Appendix B* of the thesis.

---

## Interactive Web Tool

The Streamlit application provides interactive access to the analysis results:
a word-cloud overview of dialect-distinctive substitution pairs, a per-word
detail view with side-by-side DIT and DAT hypothesis tables, audio playback
per clip, and the bipartite-matching plots for individual sentences.

> **Not yet publicly deployed.** The hosted URL and deployment instructions
> will be added to this README before the thesis defense.

To run the tool locally after completing setup:

```bash
streamlit run visualization/Home.py
```
