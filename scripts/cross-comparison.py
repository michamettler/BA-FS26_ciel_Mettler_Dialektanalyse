import pandas as pd
from pathlib import Path
from utils import (
    clean,
    build_word_comparison_df,
)

# paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "02-analysis-position"
DAT_TSV = ANALYSIS_DIR / "dialect-aware-transcript.tsv"
DIT_TSV = ANALYSIS_DIR / "dialect-ignorant-transcript.tsv"
OUTPUT_CSV_MAX_WORD_LEN = ANALYSIS_DIR / "cross-comparison_max-word-len.csv"
OUTPUT_CSV_NED = ANALYSIS_DIR / "cross-comparison_ned.csv"

def generate_csv():
    print("--- Generating Word-Level CSV for SQL/Analysis ---")

    df_dat = pd.read_csv(DAT_TSV, sep='\t', encoding='utf-8-sig')
    df_dit = pd.read_csv(DIT_TSV, sep='\t', encoding='utf-8-sig')

    df_dit = df_dit.rename(columns={'sentence': 'sentence_dit'})  # avoid collision
    df = pd.merge(df_dat, df_dit[['path', 'DIT']], on='path', how='inner')

    missing = len(df_dat) - len(df)
    if missing > 0:
        print(f"  Warning: {missing} DAT rows had no matching DIT entry and were dropped.")

    # Determine the longest word across all data for Levenshtein normalisation
    all_words = (
        df['sentence'].dropna().str.split().explode().tolist()
        + df['DAT'].dropna().str.split().explode().tolist()
        + df['DIT'].dropna().str.split().explode().tolist()
    )
    max_word_len = max(len(clean(w)) for w in all_words)
    print(f"  Longest word length (normalizer): {max_word_len}")

    word_df = build_word_comparison_df(df, max_word_len=max_word_len)
    word_df.to_csv(OUTPUT_CSV_MAX_WORD_LEN, index=False, encoding='utf-8-sig')
    
    word_df_ned = build_word_comparison_df(df, max_word_len=None)
    word_df_ned.to_csv(OUTPUT_CSV_NED, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    generate_csv()