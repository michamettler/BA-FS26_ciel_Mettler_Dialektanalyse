"""Build paired Wallis/Zürich metadata for hyperparameter-tuning ground truth.

For each sampled sentence, picks one Wallis recording and one Zürich recording (both ASR models attached) so the
alignment-quality signal is not influenced by sentence-level variation.

Inputs (test split - the only split where every sentence is recorded in all 7 dialects):
- transcripts/dialect-ignorant/whisper-large-v2/stt4sg/test_enriched_transcribed_praet.tsv
  (provides sentence, clip_is_usable, drop_reason, is_praeteritum, dit transcript)
- transcripts/dialect-aware/fhnw/stt4sg/test_transcribed.tsv
  (provides dat transcript, joined on ``path``)

Output:
- samples_metadata.tsv with one row per sentence and columns for both regions and both ASR models.
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DIT_TSV = PROJECT_ROOT / "transcripts" / "dialect-ignorant" / "whisper-large-v2" / "stt4sg" / "test_enriched_transcribed_praet.tsv"
DAT_TSV = PROJECT_ROOT / "transcripts" / "dialect-aware" / "fhnw" / "stt4sg" / "test_transcribed.tsv"

STT_DIR = PROJECT_ROOT / "datasets" / "STT4SG-350 v2.1"
AUDIO_SOURCES = [STT_DIR / "clips__test"]

TARGET_DIR = Path(__file__).resolve().parent
OUTPUT_TSV = TARGET_DIR / "samples_metadata.tsv"

REGIONS = ["Wallis", "Zürich"]
REGION_PREFIX = {"Wallis": "wallis", "Zürich": "zurich"}
N_SENTENCES = 100
SEED = 43  # because 42 contained hallucinations


def _resolve_audio_path(file_name: str) -> str | None:
    for source_folder in AUDIO_SOURCES:
        if (source_folder / file_name).exists():
            return str((source_folder / file_name).relative_to(PROJECT_ROOT))
    return None


def _path_stem(path: str) -> str:
    """Drop the extension so dit (.flac) and dat (.mp3) paths join."""
    return path.rsplit(".", 1)[0]


def _load_joined() -> pd.DataFrame:
    """Join dit + dat on the path stem (dit stores .flac, dat stores .mp3 — only
    the dat extension exists on disk, so the dat path is kept as canonical)."""
    dit = pd.read_csv(DIT_TSV, sep="\t", encoding="utf-8-sig")
    dat = pd.read_csv(DAT_TSV, sep="\t", encoding="utf-8-sig")[["path", "fhnw_transcript"]]
    dit["_stem"] = dit["path"].map(_path_stem)
    dat["_stem"] = dat["path"].map(_path_stem)
    dit = dit.drop(columns=["path"])
    dat = dat.rename(columns={"fhnw_transcript": "dat_transcript"})
    df = dit.merge(dat, on="_stem", how="left").drop(columns=["_stem"])
    return df[
        (df["clip_is_usable"] == True)
        & (df["drop_reason"].fillna("").str.strip() == "")
        & df["dat_transcript"].notna()
        & (~df["dat_transcript"].astype(str).str.startswith("ERROR"))
        ].reset_index(drop=True)


def main() -> None:
    df = _load_joined()
    pools = {region: df[df["dialect_region"] == region] for region in REGIONS}
    for region, pool in pools.items():
        print(f"  {region}: {len(pool):,} eligible recordings, "
              f"{pool['sentence'].nunique():,} distinct sentences")

    common = set.intersection(*(set(pool["sentence"].unique()) for pool in pools.values()))
    print(f"  Sentences eligible in all {len(REGIONS)} regions: {len(common):,}")
    if len(common) < N_SENTENCES:
        raise ValueError(f"Only {len(common)} eligible sentences, need {N_SENTENCES}")

    sampled_sentences = pd.Series(sorted(common)).sample(n=N_SENTENCES, random_state=SEED).tolist()

    picks: dict[str, pd.DataFrame] = {}
    for region, pool in pools.items():
        picked = (
            pool[pool["sentence"].isin(sampled_sentences)]
            .groupby("sentence", group_keys=False)
            .sample(n=1, random_state=SEED)
            .set_index("sentence")
        )
        picks[region] = picked
        print(f"  {region}: {len(picked)} recordings paired, "
              f"{picked['client_id'].nunique()} distinct speakers")

    rows: list[dict] = []
    missing = {region: 0 for region in REGIONS}
    for sentence in sampled_sentences:
        wallis = picks["Wallis"].loc[sentence]
        zurich = picks["Zürich"].loc[sentence]
        entry = {
            "sentence": sentence,
            "sentence_source": wallis["sentence_source"],
            "is_praeteritum": wallis["is_praeteritum"],
        }
        for region, row in (("Wallis", wallis), ("Zürich", zurich)):
            prefix = REGION_PREFIX[region]
            rel_path = _resolve_audio_path(row["path"])
            if rel_path is None:
                missing[region] += 1
                rel_path = ""
            entry[f"{prefix}_path"] = rel_path
            entry[f"{prefix}_client_id"] = row["client_id"]
            entry[f"{prefix}_dit_transcript"] = row["whisper_large_v2_transcript"]
            entry[f"{prefix}_dat_transcript"] = row["dat_transcript"]
        rows.append(entry)

    column_order = ["sentence", "sentence_source", "is_praeteritum"]
    for region in REGIONS:
        prefix = REGION_PREFIX[region]
        column_order += [f"{prefix}_path", f"{prefix}_client_id",
                         f"{prefix}_dit_transcript", f"{prefix}_dat_transcript"]
    pd.DataFrame(rows)[column_order].to_csv(
        OUTPUT_TSV, sep="\t", index=False, encoding="utf-8-sig"
    )

    for region, n_missing in missing.items():
        if n_missing:
            print(f"  Warning: {n_missing} {region} audio files not found.")
    print(f"Done: {len(rows)} sentences written to {OUTPUT_TSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
