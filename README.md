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

### Datasets

The two speech corpora are **not** in the repository (the audio is large and licensed). Obtain them
from their sources and place them under `datasets/` in the layout below. The metadata `.tsv` files
are committed; the **audio directories** are what you add.

```
datasets/
├── STT4SG-350 v2.1/
│   ├── clips__train_valid-001/      # train + valid audio
│   ├── clips__test/                 # test audio
│   ├── train_all.tsv                # full train split
│   ├── train_balanced.tsv           # region-balanced subset (used by Regional Distance)
│   ├── valid.tsv
│   └── test.tsv
└── SDS-200 Corpus/
    ├── export_20211220_clips-001/   # all audio (.mp3)
    ├── export_20211220.tsv          # full metadata export, one row per clip
    ├── splits/
    │   ├── train.tsv, train_{clean,other,raw,removed,unvalidated}.tsv
    │   ├── valid.tsv
    │   └── test.tsv
    ├── README_columns.txt
    └── ATTRIBUTION_DATA.txt
```

Naming conventions:
- `clips__train_valid-001` / `clips__test` (STT4SG-350) and `export_20211220_clips-001` (SDS-200) are
  the audio directories. Each holds the `<uuid>/<hash>.{flac,mp3}` files referenced by the metadata
  (`path` in STT4SG-350, `clip_path` in SDS-200).
- SDS-200 is a dated export: `export_20211220.tsv` is the metadata and `export_20211220_clips-001`
  the matching audio. Column meanings are documented in the corpus's own `README_columns.txt`.
- SDS-200 split definitions (see `splits/README_splits.txt`): `train = train_clean + train_other +
  train_unvalidated`, and `train_removed = train_raw - train` (so `train_raw` is the full set and
  `train_removed` the clips left out of `train`).

The visualization only needs the audio directories (for playback); the `.tsv` files and `splits/`
support the transcription and alignment pipeline.

---

## Key Artifacts

All of the pipeline's main outputs are committed to the repository, so no
re-transcription or re-alignment is required to use them:

- **Transcripts** (DAT and DIT): [`transcripts/`](transcripts). One TSV per split per model.
- **Word-level alignment tables**: [`experiments/analysis/`](experiments/analysis). Three
  Parquets covering REF ↔ DAT, REF ↔ DIT, and DAT ↔ DIT for `train_all`.
- **Dialect-word lexicon**: [`experiments/lexicon/`](experiments/lexicon). Top regionally distinctive
  substitution pairs for both alignment directions, ranked by TF–IDF.

File naming and column structure are defined by the generating scripts (e.g., `experiments/analysis/build_alignment_table.py`) and by the header rows of the TSV artifacts.
The full reproduction procedure (calibration => alignment => visualization) is described in *Appendix B* of the thesis.

---

## Interactive Web Tool

**Access (ZHAW-internal):** http://dialectanalysis-bipartitematching.engineering.zhaw.ch
Audio playback is password-gated; the rest of the tool is open within the ZHAW network.

The Streamlit application provides interactive access to the analysis results:
a word-cloud overview of dialect-distinctive substitution pairs, a per-word
detail view with side-by-side DIT and DAT hypothesis tables, audio playback
per clip, and the bipartite-matching plots for individual sentences.

To run the tool locally after completing setup:

```bash
streamlit run visualization/Home.py
```

For the ZHAW-internal server deployment (HTTP), see
[`deploy/DEPLOY.md`](deploy/DEPLOY.md).
