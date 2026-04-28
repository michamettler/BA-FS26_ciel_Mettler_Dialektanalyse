"""Extract per-region subsets from the already-transcribed train_all set.

Reads the existing transcribed TSV (sentence + whisper_large_v2_transcript) and
writes one metadata TSV per region. Audio files are not copied; the ``path``
column stores the location relative to the project root.
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_TSV = PROJECT_ROOT / "transcripts" / "dialect-ignorant" / "whisper-large-v2" / "stt4sg" / "train_all_enriched_transcribed_praet.tsv"

STT_DIR = PROJECT_ROOT / "datasets" / "STT4SG-350 v2.1"
AUDIO_SOURCES = [STT_DIR / "clips__test", STT_DIR / "clips__train_valid-001"]

TARGET_DIR = Path(__file__).resolve().parent


def _resolve_audio_path(file_name: str) -> str | None:
    """Return the audio path relative to PROJECT_ROOT, or None if missing."""
    for source_folder in AUDIO_SOURCES:
        if (source_folder / file_name).exists():
            return str((source_folder / file_name).relative_to(PROJECT_ROOT))
    return None


def run_extraction(region: str, count: int = 100, seed: int = 42):
    print(f"--- Extracting {count} samples for region: {region} ---")

    df = pd.read_csv(SOURCE_TSV, sep="\t", encoding="utf-8-sig")

    usable = df[
        (df["dialect_region"] == region)
        & (df["clip_is_usable"] == True)
        & (df["drop_reason"].fillna("").str.strip() == "")
    ]
    print(f"Usable pool size: {len(usable)}")

    if len(usable) < count:
        raise ValueError(f"Only {len(usable)} usable rows for {region}, requested {count}")

    sampled = usable.sample(n=count, random_state=seed)

    resolved_rows: list[dict] = []
    missing = 0
    for _, row in sampled.iterrows():
        rel_path = _resolve_audio_path(row["path"])
        if rel_path is not None:
            resolved_rows.append({**row, "path": rel_path})
        else:
            missing += 1
            print(f"Warning: audio file {row['path']} not found.")

    target_tsv = TARGET_DIR / f"{region.lower().replace('ü', 'u')}_metadata.tsv"
    pd.DataFrame(resolved_rows).to_csv(target_tsv, sep="\t", index=False, encoding="utf-8-sig")

    print(f"Done: {len(resolved_rows)} entries written to {target_tsv.relative_to(PROJECT_ROOT)}")
    if missing:
        print(f"  ({missing} files not found and skipped)")


if __name__ == "__main__":
    for region in ["Zürich", "Wallis"]:
        run_extraction(region)