import pandas as pd
import re
from pathlib import Path

# SETUP PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "02-analysis-position"
DAT_TSV = ANALYSIS_DIR / "dialect-aware-transcript.tsv"
DIT_TSV = ANALYSIS_DIR / "dialect-ignorant-transcript.tsv"
OUTPUT_CSV = ANALYSIS_DIR / "word_comparison_analysis.csv"


def clean(text):
    """Normalizes text for comparison by removing punctuation and lowercasing."""
    if pd.isna(text):
        return ""
    return re.sub(r'[^\w\s]', '', str(text)).lower()


def calculate_word_possibility(src_word_clean, target_words_clean, pos_index):
    """
    Calculates possibility score:
      1.0 = Exact match at the same position index.
      0.0 = No match at that position.
    """
    if pos_index < len(target_words_clean):
        if src_word_clean == target_words_clean[pos_index]:
            return 1.0
    return 0.0


def generate_csv():
    print("--- Generating Word-Level CSV for SQL/Analysis ---")

    df_dat = pd.read_csv(DAT_TSV, sep='\t', encoding='utf-8')
    df_dit = pd.read_csv(DIT_TSV, sep='\t', encoding='utf-8')

    df_dit = df_dit.rename(columns={'sentence': 'sentence_dit'})  # avoid collision
    df = pd.merge(df_dat, df_dit[['path', 'DIT']], on='path', how='inner')

    missing = len(df_dat) - len(df)
    if missing > 0:
        print(f"  Warning: {missing} DAT rows had no matching DIT entry and were dropped.")

    word_rows = []

    for _, row in df.iterrows():
        clip_id = row['path']

        src_words = str(row['sentence']).split()
        src_cleaned = [clean(w) for w in src_words]

        dat_words = str(row['DAT']).split()
        dat_cleaned = [clean(w) for w in dat_words]

        dit_words = str(row['DIT']).split()
        dit_cleaned = [clean(w) for w in dit_words]

        dat_len = len(dat_words)
        dit_len = len(dit_words)
        src_len = len(src_words)

        for i, original_word in enumerate(src_words):
            clean_src = src_cleaned[i]

            dat_poss = calculate_word_possibility(clean_src, dat_cleaned, i)
            dit_poss = calculate_word_possibility(clean_src, dit_cleaned, i)

            actual_dat = dat_words[i] if i < dat_len else None
            actual_dit = dit_words[i] if i < dit_len else None

            dat_advantage = round(dat_poss - dit_poss, 1)  # +1.0, 0, or -1.0

            word_rows.append({
                'clip_id':         clip_id,
                'word_index':      i,
                'src_word':        original_word,
                'dat_at_pos':      actual_dat,
                'dit_at_pos':      actual_dit,
                'dat_possibility': dat_poss,
                'dit_possibility': dit_poss,
                'dat_advantage':   dat_advantage,  # positive = DAT did better
                'is_discrepancy':  1 if dat_advantage != 0 else 0,
                'src_len':         src_len,
                'dat_len':         dat_len,
                'dit_len':         dit_len,
                'dat_len_ratio':   round(dat_len / src_len, 3) if src_len else None,
                'dit_len_ratio':   round(dit_len / src_len, 3) if src_len else None,
            })

    word_df = pd.DataFrame(word_rows)
    word_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    print(f"\nSuccess! Total words processed: {len(word_df)}")
    print(f"  Discrepancies:     {word_df['is_discrepancy'].sum()} "
          f"({100 * word_df['is_discrepancy'].mean():.1f}%)")
    print(f"  DAT better:        {(word_df['dat_advantage'] > 0).sum()}")
    print(f"  DIT better:        {(word_df['dat_advantage'] < 0).sum()}")
    print(f"  Clips merged:      {word_df['clip_id'].nunique()}")
    print(f"\nFile saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    generate_csv()