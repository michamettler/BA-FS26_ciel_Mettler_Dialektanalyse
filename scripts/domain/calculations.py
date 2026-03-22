"""
Shared helper functions for dialect analysis scripts.
"""

import re
import pandas as pd
import Levenshtein


def clean(text):
    """Cleans and normalizes text for comparison.
    - Lowercase
    - Doppel-S, Swiss German convention
    - Remove punctuation and special characters
    - Only allow a-z, digits, and German umlauts (äöü) - strips any non-Latin/non-German Unicode
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = text.replace("ß", "ss")
    text = re.sub(r"[^a-z0-9äöü\s]", "", text)
    return text.strip()


def calculate_levenshtein_distance(src_word, target_word):
    """Levenshtein edit distance using python-Levenshtein library.
    Returns the number of edits (insertions, deletions, substitutions) needed to transform src_word into target_word.
    """
    return Levenshtein.distance(src_word, target_word)


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
