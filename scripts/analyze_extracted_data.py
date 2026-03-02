import pandas as pd
from pathlib import Path

# SETUP PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "02-analysis-position"
SUBSET_TSV = ANALYSIS_DIR / "subset_metadata.tsv"

def analyze_extracted_data():
    print(f"{'='*20} ZÜRICH SUBSET ANALYSIS {'='*20}")
    
    if not SUBSET_TSV.exists():
        print(f"Error: {SUBSET_TSV} not found. Did you run the extraction script?")
        return

    # Load the subset
    df = pd.read_csv(SUBSET_TSV, sep='\t', encoding='utf-8')

    # 1. Basic Stats
    print(f"\n[1] Sample Overview")
    print(f"- Total Samples: {len(df)}")
    print(f"- Unique Speakers (client_id): {df['client_id'].nunique()}")

    # 2. Gender Distribution
    if 'gender' in df.columns:
        print(f"\n[2] Gender Distribution:")
        print(df['gender'].value_counts(dropna=False))

    # 3. Age Groups
    if 'age' in df.columns:
        print(f"\n[3] Age Group Distribution:")
        print(df['age'].value_counts(dropna=False))

    # 4. Audio Duration
    if 'duration' in df.columns:
        total_sec = df['duration'].sum()
        print(f"\n[4] Audio Stats:")
        print(f"- Total Duration: {total_sec:.2f} seconds (~{total_sec/60:.2f} minutes)")
        print(f"- Average Clip Length: {df['duration'].mean():.2f} seconds")

    # 5. Text Complexity
    df['word_count'] = df['sentence'].apply(lambda x: len(str(x).split()))
    print(f"\n[5] Sentence Complexity:")
    print(f"- Average Word Count: {df['word_count'].mean():.1f} words")
    print(f"- Max Word Count: {df['word_count'].max()} words")

if __name__ == "__main__":
    analyze_extracted_data()