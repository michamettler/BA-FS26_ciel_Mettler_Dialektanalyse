"""
Shared helper functions for dialect analysis scripts.
"""

import re
import pandas as pd


def clean(text):
    """Cleans and normalizes text for comparison.
    - Lowercase (Groß-/Kleinschreibung)
    - ß → ss (Doppel-S, Swiss German convention)
    - Remove punctuation and special characters (Satz-/Sonderzeichen)
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = text.replace('ß', 'ss')
    text = re.sub(r'[^\w\s]', '', text)
    return text


def calculate_levenshtein_distance(src_word, target_word):
    """Wagner-Fischer DP for Levenshtein edit distance."""
    if src_word == target_word:
        return 0
    if len(src_word) == 0:
        return len(target_word)
    if len(target_word) == 0:
        return len(src_word)

    matrix = [[0] * (len(target_word) + 1) for _ in range(len(src_word) + 1)]
    for i in range(len(src_word) + 1):
        matrix[i][0] = i
    for j in range(len(target_word) + 1):
        matrix[0][j] = j

    for i in range(1, len(src_word) + 1):
        for j in range(1, len(target_word) + 1):
            cost = 0 if src_word[i - 1] == target_word[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,           # deletion
                matrix[i][j - 1] + 1,           # insertion
                matrix[i - 1][j - 1] + cost     # substitution
            )
    return matrix[-1][-1]


def normalized_levenshtein_score(distance, max_len):
    """Normalised Levenshtein similarity in [0.0, 1.0]. Higher = more similar."""
    return round(1 - (distance / max_len), 3) if max_len > 0 else 1.0


def calculate_position_score(src_index, target_index, max_len):
    """Normalised position gap in [0.0, 1.0]. 0 = same position, 1 = max gap."""
    if max_len <= 1:
        return 0.0
    return round(abs(src_index - target_index) / (max_len - 1), 3)


def combined_score_weighted(lev_sim, position_score, alpha=0.7):
    """Weighted average: α · lev_sim + (1-α) · pos_sim.
    α=0.7 weights lexical similarity higher than positional similarity.
    Higher = better match."""
    pos_sim = 1.0 - position_score
    return round(alpha * lev_sim + (1 - alpha) * pos_sim, 3)


def combined_score_harmonic(lev_sim, position_score):
    """Harmonic mean (F1-style) of lev_sim and pos_sim.
    Penalises pairs where either score is low. Higher = better match.
    Based on: van Rijsbergen (1979). Information Retrieval. Butterworths."""
    pos_sim = 1.0 - position_score
    if lev_sim + pos_sim == 0:
        return 0.0
    return round(2 * lev_sim * pos_sim / (lev_sim + pos_sim), 3)


def build_word_comparison_df(df, max_word_len=None):
    """Build the word-level cross-comparison DataFrame.
    For each source word, compares it to each target word in DAT and DIT,
    """
    word_rows = []

    for _, row in df.iterrows():
        clip_id = row['path']

        src_words   = str(row['sentence']).split()
        src_cleaned = [clean(w) for w in src_words]

        dat_words   = str(row['DAT']).split()
        dat_cleaned = [clean(w) for w in dat_words]

        dit_words   = str(row['DIT']).split()
        dit_cleaned = [clean(w) for w in dit_words]

        src_len         = len(src_words)
        dat_len         = len(dat_words)
        dit_len         = len(dit_words)
        max_target_len  = max(dat_len, dit_len)
        max_len_for_pos = max(src_len, max_target_len)

        for i, src_word in enumerate(src_words):
            clean_src = src_cleaned[i]

            for j in range(max_target_len):
                dat_word  = dat_words[j]   if j < dat_len else None
                dit_word  = dit_words[j]   if j < dit_len else None
                dat_clean = dat_cleaned[j] if j < dat_len else ""
                dit_clean = dit_cleaned[j] if j < dit_len else ""

                dat_lev_dist = calculate_levenshtein_distance(clean_src, dat_clean)
                dit_lev_dist = calculate_levenshtein_distance(clean_src, dit_clean)

                if max_word_len is not None:
                    norm_dat = max_word_len
                    norm_dit = max_word_len
                else:
                    norm_dat = max(len(clean_src), len(dat_clean))
                    norm_dit = max(len(clean_src), len(dit_clean))

                dat_lev_sim = normalized_levenshtein_score(dat_lev_dist, norm_dat)
                dit_lev_sim = normalized_levenshtein_score(dit_lev_dist, norm_dit)

                position_score = calculate_position_score(i, j, max_len_for_pos)

                word_rows.append({
                    'clip_id':               clip_id,
                    'src_word_index':        i,
                    'target_word_index':     j,
                    'is_same_position':      1 if i == j else 0,
                    'position_score':        position_score,
                    'src_word':              src_word,
                    'dat_word':              dat_word,
                    'dit_word':              dit_word,
                    'dat_lev_sim':           dat_lev_sim,
                    'dit_lev_sim':           dit_lev_sim,
                    'dat_combined_weighted': combined_score_weighted(dat_lev_sim, position_score),
                    'dat_combined_harmonic': combined_score_harmonic(dat_lev_sim, position_score),
                    'dit_combined_weighted': combined_score_weighted(dit_lev_sim, position_score),
                    'dit_combined_harmonic': combined_score_harmonic(dit_lev_sim, position_score),
                    'dat_advantage':         round(dat_lev_sim - dit_lev_sim, 3),
                    'src_len':               src_len,
                    'dat_len':               dat_len,
                    'dit_len':               dit_len,
                })

    return pd.DataFrame(word_rows)
