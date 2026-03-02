import pandas as pd
from pathlib import Path
import os

# Path(__file__) is this script;
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "datasets"

def analyze_dataset(file_path, label):
    """Loads a TSV file and prints a structural overview."""
    print(f"\n{'#'*60}")
    print(f" ANALYZING DATASET: {label}")
    print(f"{'#'*60}")

    if not file_path.exists():
        print(f"ERROR: File not found at {file_path}")
        return None

    # Load data (TSV uses tab separators)
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')

    # General Info
    print(f"\n[1] BASIC STATS")
    print(f"- Total number of entries: {len(df)}")
    print(f"- Columns found: {list(df.columns)}")

    # Checking for missing data
    missing = df.isnull().sum().sum()
    print(f"- Missing values in total: {missing}")

    # Dialect analysis
    dialect_col = [c for c in df.columns if 'dialect' in c.lower() or 'region' in c.lower()]
    if dialect_col:
        print(f"\n[2] DIALECT DISTRIBUTION ({dialect_col[0]})")
        print(df[dialect_col[0]].value_counts().head(10))
    
    # Text length analysis
    text_col = [c for c in df.columns if c in ['sentence', 'transcription', 'text']]
    if text_col:
        df['char_length'] = df[text_col[0]].str.len()
        print(f"\n[3] TEXT STATS (Character Length)")
        print(f"- Average length: {df['char_length'].mean():.2f}")
        print(f"- Shortest: {df['char_length'].min()}")
        print(f"- Longest: {df['char_length'].max()}")

    print("\n[4] DATA PREVIEW (First 5 rows):")
    print(df.head())
    
    return df

# Define the specific TSV paths based on your structure
stt4sg_path = DATA_DIR / "STT4SG-350 v2.1" / "train_balanced.tsv"
sds200_path = DATA_DIR / "SDS-200 Corpus" / "export_20211220.tsv"

# Run analysis
if __name__ == "__main__":
    stt_df = analyze_dataset(stt4sg_path, "STT4SG-350 (Swiss German)")
    sds_df = analyze_dataset(sds200_path, "SDS-200 (Swiss German)")