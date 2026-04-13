"""
Cross-comparison logic for word-level dialect analysis.

Naive baseline approach: compares every reference word against every hypothesis word
(exhaustive O(n x m) cross-comparison without 1:1 alignment constraint).
"""

import pandas as pd

from models import CalculationParameters, WordSimilarity
from preprocessing import clean_word
from calculations import (
    calculate_similarities_for_word_pair
)


def generate_cross_comparison_df(
        ref_words: list[str],
        hyp_words: list[str],
        calculation_parameters: CalculationParameters,
) -> pd.DataFrame:
    """Cross-compare every reference word against every hypothesis word.

    For each (ref_word, hyp_word) pair, computes lexical similarity (Levenshtein),
    positional similarity, and weighted similarity.

    Args:
        ref_words: list of reference words.
        hyp_words: list of hypothesis words.
        calculation_parameters: Matching configuration (alpha, normalization mode, max lengths).

    Returns:
        DataFrame with one row per (ref, hyp) word pair and their similarities.
    """
    similarities = _evaluate_all_pairs(ref_words, hyp_words, calculation_parameters)

    rows = []
    for i, ref_word in enumerate(ref_words):
        ref_word_cleaned = clean_word(ref_word)
        for sim in similarities[i]:
            rows.append(
                {
                    "ref_index": i,
                    "hyp_index": sim.target_index,
                    "ref_word": ref_word_cleaned,
                    "hyp_word": sim.target_word,
                    "word_similarity": sim.word_similarity,
                    "position_similarity": sim.position_similarity,
                    "similarity_weighted": sim.similarity_weighted,
                }
            )

    return pd.DataFrame(rows)


def _evaluate_all_pairs(
        ref_words: list[str],
        hyp_words: list[str],
        calculation_parameters: CalculationParameters,
) -> list[list[WordSimilarity]]:
    """Compare every ref word against every hyp word.
    """
    results = []
    for i, ref_word in enumerate(ref_words):
        ref_word_cleaned = clean_word(ref_word)
        pair_results = []
        for j, hyp_word in enumerate(hyp_words):
            hyp_word_cleaned = clean_word(hyp_word)

            similarity_weighted, word_similarity, position_similarity = (
                calculate_similarities_for_word_pair(
                    ref_word=ref_word_cleaned,
                    ref_position=i,
                    hyp_word=hyp_word_cleaned,
                    hyp_position=j,
                    calculation_parameters=calculation_parameters,
                ))

            pair_results.append(
                WordSimilarity(
                    target_index=j,
                    target_word=hyp_word_cleaned,
                    word_similarity=word_similarity,
                    position_similarity=position_similarity,
                    similarity_weighted=similarity_weighted,
                )
            )
        results.append(pair_results)
    return results
