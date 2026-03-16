import pandas as pd
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STT_DIR = PROJECT_ROOT / "datasets" / "STT4SG-350 v2.1"
SOURCE_TSV = STT_DIR / "train_balanced.tsv"

TARGET_DIR = PROJECT_ROOT / "samples" / "zurich_subset"
TARGET_CSV = TARGET_DIR / "subset_metadata.tsv"
CLIPS_DIR = TARGET_DIR / "clips"
CLIPS_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_SOURCES = [STT_DIR / "clips__test", STT_DIR / "clips__train_valid-001"]


def run_extraction(count=100, region="Zürich"):
    print(f"--- Starting Extraction: {region} ---")

    if not SOURCE_TSV.exists():
        print(f"Error: Could not find {SOURCE_TSV}")
        return

    df = pd.read_csv(SOURCE_TSV, sep="\t", encoding="utf-8-sig")
    zh_df = df[df["dialect_region"] == region].sample(n=count, random_state=42)

    successful_samples = []

    print(f"Copying files to {CLIPS_DIR}...")
    for _, row in zh_df.iterrows():
        file_name = row["path"]
        file_found = False

        for source_folder in AUDIO_SOURCES:
            source_file = source_folder / file_name
            if source_file.exists():
                target_file = CLIPS_DIR / file_name
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, target_file)
                # Store path relative to CLIPS_DIR so TSV stays portable
                successful_samples.append(
                    {**row, "path": str(Path("clips") / file_name)}
                )
                file_found = True
                break

        if not file_found:
            print(f"Warning: Audio file {file_name} not found.")

    output_df = pd.DataFrame(successful_samples)
    output_tsv = TARGET_CSV
    output_df.to_csv(output_tsv, sep="\t", index=False, encoding="utf-8-sig")

    print(
        f"\nDone! {len(successful_samples)} files copied, metadata saved to: {output_tsv}"
    )


if __name__ == "__main__":
    run_extraction(100, "Zürich")
