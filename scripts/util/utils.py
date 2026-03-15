"""
Shared helper functions for dialect analysis scripts.
"""

import re
import pandas as pd


def clean(text):
    """Cleans and normalizes text for comparison.
    - Lowercase
    - Doppel-S, Swiss German convention
    - Remove punctuation and special characters
    - Only allow a-z, digits, and German umlauts (äöü) — strips any non-Latin/non-German Unicode
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = text.replace("ß", "ss")
    text = re.sub(r"[^a-z0-9äöü\s]", "", text)
    return text.strip()


def calculate_levenshtein_distance(src_word, target_word):
    """Wagner-Fischer DP for Levenshtein edit distance.
    Returns the number of edits (insertions, deletions, substitutions) needed to transform src_word into target_word.
    """
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
                matrix[i - 1][j] + 1,  # deletion
                matrix[i][j - 1] + 1,  # insertion
                matrix[i - 1][j - 1] + cost,  # substitution
            )
    return matrix[-1][-1]


def normalize_levenshtein_distance(distance, global_max_word_length):
    """Normalised Levenshtein similarity in [0.0, 1.0]. Higher = more similar."""
    if global_max_word_length > 0:
        return round(1 - (distance / global_max_word_length), 3)
    else:
        return 1.0


def calculate_position_score(src_index, target_index, global_max_sentence_length):
    """Normalised position similarity in [0.0, 1.0]. 1 = same position, 0 = max gap.
    Calculates how close the target word's position is to the source word's position,
    normalised by the maximum sentence length to account for different sentence lengths.
    """
    if global_max_sentence_length > 1:
        distance_to_target_word = abs(src_index - target_index)
        max_possible_distance = (
                global_max_sentence_length - 1
        )  # -1 because max index gap in a sequence of length n is n-1
        return round(1.0 - (distance_to_target_word / max_possible_distance), 3)
    else:
        return 1.0  # sentence only has 0 or 1 word, so position is irrelevant - treat as perfect match


def calculate_score_weighted(word_score, position_score, alpha=0.5):
    """Weighted average: alpha * word_score + (1-alpha) * position_score.
    Default alpha=0.5 weights lexical and positional similarity equally.
    Higher = better match."""
    return round(alpha * word_score + (1 - alpha) * position_score, 3)


def calculate_score_harmonic(word_score, position_score):
    """Harmonic mean (F1-style) of word_score and position_score.
    Penalises pairs where either score is low. Higher = better match.
    Based on: van Rijsbergen (1979). Information Retrieval. Butterworths."""
    if word_score + position_score == 0:
        return 0.0
    return round(2 * word_score * position_score / (word_score + position_score), 3)


def evaluate_scores(alpha: float, target_word: str, src_word: str, global_max_word_length, position_score: float) -> \
        tuple[float, float, float]:
    if target_word is not None:
        lev_distance = calculate_levenshtein_distance(src_word, target_word)
        word_score = normalize_levenshtein_distance(lev_distance, global_max_word_length)

        score_weighted = calculate_score_weighted(
            word_score, position_score, alpha
        )
        score_harmonic = calculate_score_harmonic(
            word_score, position_score
        )
    else:
        word_score = 0.0
        score_weighted = 0.0
        score_harmonic = 0.0
    return score_harmonic, score_weighted, word_score


def build_word_comparison_df(
        df, global_max_word_length, global_max_sentence_length, alpha=0.5
):
    """Build the word-level cross-comparison DataFrame.
    For each source word, compares it to each target word position in both DAT and DIT transcriptions.
    Returns a DataFrame with one row per (clip, src_word_index, target_word_index) triple.
    """
    word_rows = []

    for _, row in df.iterrows():
        clip_id = row["path"]

        # create list of words for source, DIT and DAT sentence & clean them
        src_words = [clean(w) for w in str(row["sentence"]).split()]
        dit_words = [clean(w) for w in str(row["DIT"]).split()]
        dat_words = [clean(w) for w in str(row["DAT"]).split()]

        n_src_words = len(src_words)  # number of words in source sentence
        n_dit_words = len(dit_words)  # number of words in DIT target sentence
        n_dat_words = len(dat_words)  # number of words in DAT target sentence

        for i, src_word in enumerate(src_words):
            for j in range(
                    max(
                        n_dit_words, n_dat_words
                    )  # take max to cover all words in both targets
            ):
                dit_word = dit_words[j] if j < n_dit_words else None
                dat_word = dat_words[j] if j < n_dat_words else None

                position_score = calculate_position_score(
                    i, j, global_max_sentence_length
                )  # position accounts for both DIT and DAT since they are aligned by index

                (dit_score_harmonic,
                 dit_score_weighted,
                 dit_word_score) = evaluate_scores(alpha, dit_word, src_word, global_max_word_length, position_score)

                (dat_score_harmonic,
                 dat_score_weighted,
                 dat_word_score) = evaluate_scores(alpha, dat_word, src_word, global_max_word_length, position_score)

                word_rows.append(
                    {
                        "clip_id": clip_id,
                        "src_word_index": i,
                        "target_word_index": j,
                        "is_same_position": 1 if i == j else 0,
                        "src_word": src_word,
                        "dit_word": dit_word,
                        "dat_word": dat_word,
                        "dit_word_score": dit_word_score,
                        "dat_word_score": dat_word_score,
                        "position_score": position_score,
                        "dit_score_weighted": dit_score_weighted,
                        "dit_score_harmonic": dit_score_harmonic,
                        "dat_score_weighted": dat_score_weighted,
                        "dat_score_harmonic": dat_score_harmonic,
                        "src_len": n_src_words,
                        "dit_len": n_dit_words,
                        "dat_len": n_dat_words,
                    }
                )

    return pd.DataFrame(word_rows)
