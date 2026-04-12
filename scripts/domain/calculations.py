"""
Shared helper functions for dialect analysis scripts.
"""

import re
import pandas as pd
import Levenshtein


def clean(text: str) -> str:
    """Clean and normalize text for comparison.

    Lowercases, replaces ß with ss (Swiss German convention), and removes
    all characters except a-z, digits, and German umlauts (äöü).

    Args:
        text: Raw input text to clean.

    Returns:
        Cleaned and normalized string.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = text.replace("ß", "ss")
    text = re.sub(r"[^a-z0-9äöü\s]", "", text)
    return text.strip()


def calculate_word_similarity_global(
        src_word: str, target_word: str, global_max_word_length: int
) -> float:
    """Calculate normalized Levenshtein similarity using a global maximum word length.

    Normalization is based on global_max_word_length to achieve comparable similarities
    across the dataset.

    Args:
        src_word: Source word.
        target_word: Target word to compare against.
        global_max_word_length: Globally known maximum word length for normalization.

    Returns:
        Similarity in [0.0, 1.0]. Higher means more similar.
    """
    distance = Levenshtein.distance(src_word, target_word)

    if global_max_word_length > 0:
        return 1 - (distance / global_max_word_length)
    else:
        return 1.0


def calculate_word_similarity_local(src_word: str, target_word: str) -> float:
    """Calculate normalized Levenshtein similarity using local (per-pair) normalization.

    Delegates to python-Levenshtein's ratio:
    similarity = (len(src) + len(target) - distance) / (len(src) + len(target)).

    Args:
        src_word: Source word.
        target_word: Target word to compare against.

    Returns:
        Similarity in [0.0, 1.0]. Higher means more similar.
    """
    return Levenshtein.ratio(src_word, target_word)


def calculate_position_score(
        src_index: int, target_index: int, global_max_sentence_length: int
) -> float:
    """Calculate normalized position similarity between two word positions.

    Returns 1.0 if positions are identical, 0.0 if the gap is maximal relative
    to the global maximum sentence length.

    Args:
        src_index: Position of the source word in its sentence.
        target_index: Position of the target word in its sentence.
        global_max_sentence_length: Globally known maximum sentence length for normalization.

    Returns:
        Position similarity in [0.0, 1.0]. Higher means closer positions.
    """
    if global_max_sentence_length > 1:
        distance_to_target_word = abs(src_index - target_index)
        max_possible_distance = (
                global_max_sentence_length - 1
        )  # -1 because max index gap in a sequence of length n is n-1, needed for worst case to be 0
        return 1.0 - (distance_to_target_word / max_possible_distance)
    else:
        return 1.0  # sentence only has 0 or 1 word, so position is irrelevant - treat as perfect match


def calculate_score_weighted(
        word_score: float, position_score: float, alpha=0.5
) -> float:
    """Compute weighted average of lexical and positional similarity scores.

    Args:
        word_score: Lexical similarity score in [0.0, 1.0].
        position_score: Positional similarity score in [0.0, 1.0].
        alpha: Weight for lexical similarity; (1 - alpha) is applied to positional. Defaults to 0.5.

    Returns:
        Combined score in [0.0, 1.0]. Higher means better match.
    """
    return alpha * word_score + (1 - alpha) * position_score


def calculate_score_harmonic(word_score: float, position_score: float) -> float:
    """Compute harmonic mean (F1-style) of lexical and positional similarity scores.

    Penalizes pairs where either score is low.
    Based on: van Rijsbergen (1979). Information Retrieval. Butterworths.

    Args:
        word_score: Lexical similarity score in [0.0, 1.0].
        position_score: Positional similarity score in [0.0, 1.0].

    Returns:
        Harmonic mean in [0.0, 1.0]. Higher means better match.
    """
    if word_score + position_score == 0:
        return 0.0
    return 2 * word_score * position_score / (word_score + position_score)
