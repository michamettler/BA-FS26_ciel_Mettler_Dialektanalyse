"""Shared data layer for the Streamlit visualization pages."""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple, cast

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "domain"))

from word_similarity_calculator import (  # noqa: E402, F401
    WordSimilarityCalculator,
    cost_for_word_pair_by_similarity,
    similarity_for_word_pair_by_cost,
)


class TfidfResult(NamedTuple):
    """Region-document TF-IDF artifact: scores matrix + vocab lookups + region row ordering.

    Fields:
        matrix: (n_regions, vocab_size) ndarray of TF-IDF scores; rows align with region_order.
        vocab: terms ordered by column index.
        word_to_idx: term → column index lookup.
        region_order: region names ordered by row index.

    Tuple-unpackable: `matrix, vocab, word_to_idx, region_order = tfidf_matrix_pairs(...)` still works.
    """
    matrix: np.ndarray
    vocab: list[str]
    word_to_idx: dict[str, int]
    region_order: list[str]


@dataclass(frozen=True)
class DatasetConfig:
    """Per-dataset file paths, audio roots, and DAT/DIT join key."""
    name: str
    dat_parquet: Path
    dit_parquet: Path
    dat_dit_parquet: Path
    dat_tsv: Path
    dit_tsv: Path
    balanced_tsv: Path | None         # None → "no balanced subset"; Regional Distance drops the filter.
    audio_roots: tuple[Path, ...]
    metadata_join_key: str            # join column between DAT and DIT TSVs ("path" or "clip_id")


CloudMode = Literal["ref_dit", "dat_dit"]

ALIGN_DIR = PROJECT_ROOT / "experiments" / "analysis"

STT4SG = DatasetConfig(
    name="STT4SG-350",
    dat_parquet=ALIGN_DIR / "stt4sg" / "train_all_alignments_dialect-aware.parquet",
    dit_parquet=ALIGN_DIR / "stt4sg" / "train_all_alignments_dialect-ignorant.parquet",
    dat_dit_parquet=ALIGN_DIR / "stt4sg" / "train_all_alignments_dat-dit.parquet",
    dat_tsv=PROJECT_ROOT / "transcripts" / "dialect-aware" / "fhnw" / "stt4sg" / "train_all_transcribed.tsv",
    dit_tsv=PROJECT_ROOT / "transcripts" / "dialect-ignorant" / "whisper-large-v2" / "stt4sg" / "train_all_enriched_transcribed_praet.tsv",
    balanced_tsv=PROJECT_ROOT / "datasets" / "STT4SG-350 v2.1" / "train_balanced.tsv",
    audio_roots=(
        PROJECT_ROOT / "datasets" / "STT4SG-350 v2.1" / "clips__train_valid-001",
        PROJECT_ROOT / "datasets" / "STT4SG-350 v2.1" / "clips__test",
    ),
    metadata_join_key="path",
)

SDS200 = DatasetConfig(
    name="SDS-200",
    dat_parquet=ALIGN_DIR / "sds-200" / "train_all_alignments_dialect-aware.parquet",
    dit_parquet=ALIGN_DIR / "sds-200" / "train_all_alignments_dialect-ignorant.parquet",
    dat_dit_parquet=ALIGN_DIR / "sds-200" / "train_all_alignments_dat-dit.parquet",
    dat_tsv=PROJECT_ROOT / "transcripts" / "dialect-aware" / "fhnw" / "sds-200" / "train_all_transcribed.tsv",
    dit_tsv=PROJECT_ROOT / "transcripts" / "dialect-ignorant" / "whisper-large-v2" / "sds-200" / "export_20211220_enriched_transcribed_praet.tsv",
    balanced_tsv=None,
    audio_roots=(
        PROJECT_ROOT / "datasets" / "SDS-200 Corpus" / "export_20211220_clips-001",
    ),
    metadata_join_key="clip_id",
)

DATASETS: dict[str, DatasetConfig] = {STT4SG.name: STT4SG, SDS200.name: SDS200}
DATASET_CHOICES: tuple[str, ...] = (STT4SG.name, SDS200.name)
DEFAULT_DATASET = STT4SG.name

REGIONS = ["Wallis", "Zürich", "Bern", "Basel", "Graubünden", "Innerschweiz", "Ostschweiz"]

# Calibrated hyperparameters: must match the ones in build_alignment_table
ALPHA = 0.85
LAMBDA = 0.45
USE_GLOBAL_LEXICAL_NORMALIZATION = False
USE_SQUARED_POSITIONAL = True

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

MODE_TO_MODEL: dict[CloudMode, str] = {
    "ref_dit": "dialect-ignorant",
    "dat_dit": "dat-dit",
}


def epsilon_cost() -> float:
    """Cost the bipartite solver charges for routing an unmatched word through an ε edge (= λ).
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
def load_alignments(dataset: str, include_dat_dit: bool = False) -> pd.DataFrame:
    """Concat the REF-anchored alignment parquets for the given dataset; opt-in to also include DAT-DIT."""
    cfg = DATASETS[dataset]
    dat = pd.read_parquet(cfg.dat_parquet).assign(model="dialect-aware")
    dit = pd.read_parquet(cfg.dit_parquet).assign(model="dialect-ignorant")
    frames = [dat, dit]
    if include_dat_dit:
        dat_dit = pd.read_parquet(cfg.dat_dit_parquet).assign(model="dat-dit")
        frames.append(dat_dit)
    out = pd.concat(frames, ignore_index=True)
    # `model` is low-cardinality and heavily filtered/grouped on — category cuts memory and speeds groupbys.
    out["model"] = out["model"].astype("category")
    return out


def _load_metadata_path_joined(cfg: DatasetConfig) -> pd.DataFrame:
    """STT4SG-shape metadata: DAT TSV joined to DIT TSV on `path`."""
    dat_cols = ["path", "dialect_region", "client_id", "gender", "age", "canton", "sentence", "fhnw_transcript"]
    dat = pd.read_csv(cfg.dat_tsv, sep="\t", encoding="utf-8-sig")[dat_cols]
    dit = pd.read_csv(cfg.dit_tsv, sep="\t", encoding="utf-8-sig")[
        ["path", "whisper_large_v2_transcript", "is_praeteritum"]
    ]
    return dat.merge(dit, on="path", how="left").rename(columns={
        "sentence": "reference",
        "fhnw_transcript": "dat_hypothesis",
        "whisper_large_v2_transcript": "dit_hypothesis",
    })


def _load_metadata_clip_id_joined(cfg: DatasetConfig) -> pd.DataFrame:
    """SDS-200-shape metadata: DAT TSV (has no `path`) joined to DIT TSV on `clip_id`; emit
    DIT's `path` so downstream merges with the alignment parquet match (alignment stores DIT's path)."""
    dat = pd.read_csv(cfg.dat_tsv, sep="\t", encoding="utf-8-sig")[
        ["clip_id", "sentence", "fhnw_transcript", "client_id", "gender", "age", "canton"]
    ]
    dit = pd.read_csv(cfg.dit_tsv, sep="\t", encoding="utf-8-sig")[
        ["clip_id", "path", "dialect_region", "whisper_large_v2_transcript", "is_praeteritum"]
    ]
    merged = dat.merge(dit, on="clip_id", how="inner").rename(columns={
        "sentence": "reference",
        "fhnw_transcript": "dat_hypothesis",
        "whisper_large_v2_transcript": "dit_hypothesis",
    })
    return merged[[
        "path", "dialect_region", "client_id", "gender", "age", "canton",
        "reference", "dat_hypothesis", "dit_hypothesis", "is_praeteritum",
    ]]


@st.cache_data
def load_metadata(dataset: str) -> pd.DataFrame:
    """Per-sentence metadata for the given dataset."""
    cfg = DATASETS[dataset]
    if cfg.metadata_join_key == "path":
        df = _load_metadata_path_joined(cfg)
    elif cfg.metadata_join_key == "clip_id":
        df = _load_metadata_clip_id_joined(cfg)
    else:
        raise ValueError(f"Unknown metadata_join_key: {cfg.metadata_join_key}")
    # `dialect_region` is the dominant groupby/filter key — category speeds those and cuts memory.
    df["dialect_region"] = df["dialect_region"].astype("category")
    return df


@st.cache_data
def joined_view(regions: tuple[str, ...], dataset: str, include_dat_dit: bool = False) -> pd.DataFrame:
    """Alignments & metadata for the given dataset, filtered to the selected regions."""
    alignments = load_alignments(dataset, include_dat_dit)
    metadata = load_metadata(dataset)
    df = alignments.merge(metadata, on="path", how="left")
    return df[df["dialect_region"].isin(regions)].reset_index(drop=True)


@st.cache_data
def load_balanced_paths(dataset: str) -> pd.DataFrame | None:
    """train_balanced.tsv joined with the praeteritum flag (path, region, is_praeteritum).

    Returns `None` when the dataset has no curated balanced subset (SDS-200).
    """
    cfg = DATASETS[dataset]
    if cfg.balanced_tsv is None:
        return None
    balanced = pd.read_csv(cfg.balanced_tsv, sep="\t", encoding="utf-8-sig")[["path", "dialect_region"]]
    is_praet = pd.read_csv(cfg.dit_tsv, sep="\t", encoding="utf-8-sig", usecols=["path", "is_praeteritum"])
    return balanced.merge(is_praet, on="path", how="left")


@st.cache_data
def tfidf_matrix_pairs(include_preterite: bool, mode: CloudMode, dataset: str) -> TfidfResult:
    """TF-IDF over alignment pairs across the 7 dialect regions, for the given dataset.

    Vectorizer config:
        * `sublinear_tf=True`: to damp very frequent pairs.
        * `smooth_idf=False`: use unsmoothed IDF; terms present in all regions get IDF 1 -> TF-IDF weight 0 after IDF shift.
        * `norm='l2'` (sklearn default): making comparisons across regions meaningful.
    """
    df = joined_view(tuple(REGIONS), dataset, include_dat_dit=(mode == "dat_dit"))
    if not include_preterite:
        df = df[~df["is_praeteritum"].fillna(False).astype(bool)]
    df = df[
        (df["model"] == MODE_TO_MODEL[mode])
        & df["reference_word"].notna()
        & df["hypothesis_word"].notna()
        & (df["reference_word"] != df["hypothesis_word"])  # filter out matches where ref and hyp are the same word
        ]
    pairs = df["reference_word"] + "+" + df["hypothesis_word"]
    docs_per_region = [" ".join(pairs[df["dialect_region"] == r]) for r in REGIONS]

    vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False,
                          sublinear_tf=True, smooth_idf=False)
    matrix = vec.fit_transform(docs_per_region).toarray()

    return TfidfResult(
        matrix=matrix,
        vocab=vec.get_feature_names_out().tolist(),
        word_to_idx=cast(dict[str, int], dict(vec.vocabulary_)),  # for column lookup from detail view (word selected)
        region_order=list(REGIONS),
    )


@st.cache_data
def lexicon_search_index(dataset: str, regions: tuple[str, ...], include_preterite: bool):
    """Page-1 search aggregates: ref-word frequency table + sidebar metric counts, cached per filter combo so they
    don't recompute on every rerun."""
    df = joined_view(regions, dataset)  # include_dat_dit=False
    if not include_preterite:
        df = df[~df["is_praeteritum"].fillna(False).astype(bool)]
    ref_counts = (
        df[df["reference_word"].notna()]
        .groupby("reference_word").size().sort_values(ascending=False)
    )
    return ref_counts, int(len(df)), int(df["path"].nunique())


@st.cache_data
def pair_region_counts(dataset: str, regions: tuple[str, ...], include_preterite: bool, mode: CloudMode) -> pd.DataFrame:
    """(ref+hyp) pair x region occurrence counts behind the word cloud, cached per filter combo so the
    overview groupby doesn't rerun on every interaction."""
    df = joined_view(regions, dataset, include_dat_dit=(mode == "dat_dit"))
    if not include_preterite:
        df = df[~df["is_praeteritum"].fillna(False).astype(bool)]
    df = df[df["model"] == MODE_TO_MODEL[mode]].dropna(subset=["hypothesis_word", "reference_word"])
    df = df[df["reference_word"] != df["hypothesis_word"]]
    pair = df["reference_word"] + "+" + df["hypothesis_word"]
    return (
        pd.DataFrame({"pair": pair, "dialect_region": df["dialect_region"]})
        .groupby(["pair", "dialect_region"], observed=True).size().unstack(fill_value=0)
    )
