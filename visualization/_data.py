"""Shared data layer for the Streamlit visualization pages."""
import sys
from pathlib import Path
from typing import NamedTuple, cast

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfResult(NamedTuple):
    """Region-document TF-IDF artifact: scores matrix + vocab lookups + region row ordering.

    Fields:
        matrix:       (n_regions, vocab_size) ndarray of TF-IDF scores; rows align with region_order.
        vocab:        terms ordered by column index.
        word_to_idx:  term → column index lookup.
        region_order: region names ordered by row index.

    Tuple-unpackable: `matrix, vocab, word_to_idx, region_order = tfidf_matrix_pairs(...)` still works.
    """
    matrix: np.ndarray
    vocab: list[str]
    word_to_idx: dict[str, int]
    region_order: list[str]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "domain"))

from word_similarity_calculator import (  # noqa: E402, F401
    WordSimilarityCalculator,
    cost_for_word_pair_by_similarity,
    similarity_for_word_pair_by_cost,
)

ALIGN_DIR = PROJECT_ROOT / "experiments" / "analysis"
DAT_PARQUET = ALIGN_DIR / "train_all_alignments_dialect-aware.parquet"
DIT_PARQUET = ALIGN_DIR / "train_all_alignments_dialect-ignorant.parquet"
DAT_TSV = PROJECT_ROOT / "transcripts" / "dialect-aware" / "fhnw" / "stt4sg" / "train_all_transcribed.tsv"
DIT_TSV = PROJECT_ROOT / "transcripts" / "dialect-ignorant" / "whisper-large-v2" / "stt4sg" / "train_all_enriched_transcribed_praet.tsv"
BALANCED_TSV = PROJECT_ROOT / "datasets" / "STT4SG-350 v2.1" / "train_balanced.tsv"

REGIONS = ["Wallis", "Zürich", "Bern", "Basel", "Graubünden", "Innerschweiz", "Ostschweiz"]

# Calibrated epsilon penalty: must match build_alignment_table.LAMBDA_.
LAMBDA = 0.45

DAT_COLOR = "#4c72b0"
DIT_COLOR = "#dd8452"

REGION_COLORS = {
    "Wallis": "#e41a1c",
    "Zürich": "#377eb8",
    "Bern": "#4daf4a",
    "Basel": "#984ea3",
    "Graubünden": "#ff7f00",
    "Innerschweiz": "#dbb500",
    "Ostschweiz": "#a65628",
}


def epsilon_cost() -> float:
    """Cost the bipartite solver charges for routing an unmatched word through an ε edge (= λ).

    Computed lazily so calculator/lambda changes propagate across Streamlit reruns without restart.
    """
    return WordSimilarityCalculator(sent_len=1, lambda_=LAMBDA).cost_for_epsilon_by_penalty()


def deletion_similarity() -> float:
    """Similarity equivalent of the bipartite solver's ε-edge cost (1 − λ).

    Use this to impute deletions when computing per-edge similarity means: a deletion contributes
    the same similarity it would have contributed in cost-domain (`1 − λ`), keeping the analysis
    consistent with the matcher's ε-edge cost.
    """
    return similarity_for_word_pair_by_cost(epsilon_cost())


@st.cache_data
def load_alignments() -> pd.DataFrame:
    """Concat both alignment parquets and tag each row with its model."""
    dat = pd.read_parquet(DAT_PARQUET).assign(model="dialect-aware")
    dit = pd.read_parquet(DIT_PARQUET).assign(model="dialect-ignorant")
    return pd.concat([dat, dit], ignore_index=True)


@st.cache_data
def load_metadata() -> pd.DataFrame:
    """Slim per-sentence frame: region/speaker metadata + reference + both hypotheses + tense flag."""
    dat_cols = ["path", "dialect_region", "client_id", "gender", "age", "canton", "sentence", "fhnw_transcript"]
    dat = pd.read_csv(DAT_TSV, sep="\t", encoding="utf-8-sig")[dat_cols]
    dit = pd.read_csv(DIT_TSV, sep="\t", encoding="utf-8-sig")[
        ["path", "whisper_large_v2_transcript", "is_praeteritum"]
    ]
    return dat.merge(dit, on="path", how="left").rename(columns={
        "sentence": "reference",
        "fhnw_transcript": "dat_hypothesis",
        "whisper_large_v2_transcript": "dit_hypothesis",
    })


@st.cache_data
def joined_view(regions: tuple[str, ...]) -> pd.DataFrame:
    """Alignments & metadata, filtered to the selected regions."""
    alignments = load_alignments()
    metadata = load_metadata()
    df = alignments.merge(metadata, on="path", how="left")
    return df[df["dialect_region"].isin(regions)].reset_index(drop=True)


@st.cache_data
def load_balanced_paths() -> pd.DataFrame:
    """train_balanced.tsv joined with the praeteritum flag (path, region, is_praeteritum)."""
    balanced_data = pd.read_csv(BALANCED_TSV, sep="\t", encoding="utf-8-sig")[["path", "dialect_region"]]
    is_praet = pd.read_csv(DIT_TSV, sep="\t", encoding="utf-8-sig", usecols=["path", "is_praeteritum"])
    return balanced_data.merge(is_praet, on="path", how="left")


@st.cache_data
def tfidf_matrix_pairs(include_preterite: bool) -> TfidfResult:
    """TF-IDF over (ref, DIT-hyp) pairs across the 7 dialect regions.
    Each ref, hyp pair is one term (encoded as ref+hyp for the vectorizer);
    a region = document (bag of its pairs).
    Resulting matrix shape: (7 regions × pairs vocab).

    Vectorizer config:
        * `sublinear_tf=True`: `1 + log(count)` so hapaxes don't dominate.
        * `smooth_idf=False`: no IDF +1 smoothing; universal-term contribution stays minimal.
        * L2 row-norm (sklearn default) — cross-region comparability.
    """
    df = joined_view(tuple(REGIONS))
    if not include_preterite:
        df = df[~df["is_praeteritum"].fillna(False).astype(bool)]
    df = df[
        (df["model"] == "dialect-ignorant")
        & df["reference_word"].notna()
        & df["hypothesis_word"].notna()
        & (df["reference_word"] != df["hypothesis_word"]) # filter out matches where ref and hyp are the same word
    ]
    pairs = df["reference_word"] + "+" + df["hypothesis_word"]
    docs_per_region = [" ".join(pairs[df["dialect_region"] == r]) for r in REGIONS]

    vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False,
                          sublinear_tf=True, smooth_idf=False)
    matrix = vec.fit_transform(docs_per_region).toarray()

    return TfidfResult(
        matrix=matrix,
        vocab=vec.get_feature_names_out().tolist(),
        word_to_idx=cast(dict[str, int], dict(vec.vocabulary_)), # for column lookup from detail view (word selected)
        region_order=list(REGIONS),
    )