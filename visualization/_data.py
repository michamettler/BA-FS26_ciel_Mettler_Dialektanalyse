"""Shared data layer for the Streamlit visualization pages."""
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALIGN_DIR = PROJECT_ROOT / "experiments" / "analysis"
DAT_PARQUET = ALIGN_DIR / "train_all_alignments_dialect-aware.parquet"
DIT_PARQUET = ALIGN_DIR / "train_all_alignments_dialect-ignorant.parquet"
DAT_TSV = PROJECT_ROOT / "transcripts" / "dialect-aware" / "fhnw" / "stt4sg" / "train_all_transcribed.tsv"
DIT_TSV = PROJECT_ROOT / "transcripts" / "dialect-ignorant" / "whisper-large-v2" / "stt4sg" / "train_all_enriched_transcribed_praet.tsv"
BALANCED_TSV = PROJECT_ROOT / "datasets" / "STT4SG-350 v2.1" / "train_balanced.tsv"

REGIONS = ["Wallis", "Zürich", "Bern", "Basel", "Graubünden", "Innerschweiz", "Ostschweiz"]

# Calibrated epsilon penalty — must match build_alignment_table.LAMBDA_.
LAMBDA = 0.45

DAT_COLOR = "#4c72b0"
DIT_COLOR = "#dd8452"


@st.cache_data
def load_alignments() -> pd.DataFrame:
    """Concat both alignment parquets and tag each row with its model."""
    dat = pd.read_parquet(DAT_PARQUET).assign(model="dialect-aware")
    dit = pd.read_parquet(DIT_PARQUET).assign(model="dialect-ignorant")
    return pd.concat([dat, dit], ignore_index=True)


@st.cache_data
def load_metadata() -> pd.DataFrame:
    """Slim per-sentence frame: region/speaker metadata + reference + both hypotheses."""
    dat_cols = ["path", "dialect_region", "client_id", "gender", "age", "canton", "sentence", "fhnw_transcript"]
    dat = pd.read_csv(DAT_TSV, sep="\t", encoding="utf-8-sig")[dat_cols]
    dit = pd.read_csv(DIT_TSV, sep="\t", encoding="utf-8-sig")[["path", "whisper_large_v2_transcript"]]
    return dat.merge(dit, on="path", how="left").rename(columns={
        "sentence": "reference",
        "fhnw_transcript": "dat_hypothesis",
        "whisper_large_v2_transcript": "dit_hypothesis",
    })


@st.cache_data
def joined_view(regions: tuple[str, ...]) -> pd.DataFrame:
    """Alignments + metadata, filtered to the selected regions."""
    align = load_alignments()
    meta = load_metadata()
    df = align.merge(meta, on="path", how="left")
    return df[df["dialect_region"].isin(regions)].reset_index(drop=True)


@st.cache_data
def load_balanced_paths() -> pd.DataFrame:
    """train_balanced.tsv: per-region count-balanced subset of train_all (path + region only)."""
    return pd.read_csv(BALANCED_TSV, sep="\t", encoding="utf-8-sig")[["path", "dialect_region"]]