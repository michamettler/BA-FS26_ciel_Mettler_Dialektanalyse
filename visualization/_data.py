"""Shared data layer for the Streamlit visualization pages."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

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