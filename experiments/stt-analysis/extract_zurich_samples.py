"""Extract a Zurich dialect subset from the STT4SG-350 dataset.

Only creates a metadata TSV - audio files stay in the original dataset directory.
The ``path`` column stores each clip's location relative to the project root,
e.g. ``datasets/STT4SG-350 v2.1/clips__train_valid-001/uuid/hash.flac``.
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STT_DIR = PROJECT_ROOT / "datasets" / "STT4SG-350 v2.1"
SOURCE_TSV = STT_DIR / "train_balanced.tsv"

TARGET_TSV = Path(__file__).resolve().parent / "zurich_subset_metadata.tsv"

AUDIO_SOURCES = [STT_DIR / "clips__test", STT_DIR / "clips__train_valid-001"]


def _resolve_audio_path(file_name: str) -> str | None:
    """Return the path relative to PROJECT_ROOT for *file_name*, or None."""
    for source_folder in AUDIO_SOURCES:
        if (source_folder / file_name).exists():
            return str((source_folder / file_name).relative_to(PROJECT_ROOT))
    return None


def run_extraction(count: int = 100, region: str = "Zürich"):
    """Filter *count* experiments for *region* and write a metadata TSV."""
    print(f"--- Starting Extraction: {region} ---")

    if not SOURCE_TSV.exists():
        print(f"Error: Could not find {SOURCE_TSV}")
        return

    df = pd.read_csv(SOURCE_TSV, sep="\t", encoding="utf-8-sig")
    zh_df = df[df["dialect_region"] == region].sample(n=count, random_state=42)

    resolved_rows: list[dict] = []
    missing = 0

    for _, row in zh_df.iterrows():
        rel_path = _resolve_audio_path(row["path"])
        if rel_path is not None:
            resolved_rows.append({**row, "path": rel_path})
        else:
            missing += 1
            print(f"Warning: Audio file {row['path']} not found in dataset.")

    output_df = pd.DataFrame(resolved_rows)
    output_df.to_csv(TARGET_TSV, sep="\t", index=False, encoding="utf-8-sig")

    print(f"\nDone! {len(resolved_rows)} entries written to: {TARGET_TSV}")
    if missing:
        print(f"  ({missing} files not found and skipped)")


if __name__ == "__main__":
    run_extraction(100, "Zürich")

