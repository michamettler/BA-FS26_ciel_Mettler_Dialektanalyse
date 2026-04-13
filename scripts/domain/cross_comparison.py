"""
Cross-comparison logic for word-level dialect analysis.

Naive baseline approach: compares every reference word against every hypothesis word
(exhaustive O(n x m) cross-comparison without 1:1 alignment constraint).
"""

import pandas as pd

from models import WordSimilarity
from preprocessing import clean_word
from word_similarity_calculator import WordSimilarityCalculator


def generate_cross_comparison_df(
        ref_words: list[str],
        hyp_words: list[str],
        calculator: WordSimilarityCalculator,
) -> pd.DataFrame:
    """Cross-compare every reference word against every hypothesis word.

    Args:
        ref_words: list of reference words.
        hyp_words: list of hypothesis words.
        calculator: WordSimilarityCalculator instance for computing similarities.

    Returns:
        DataFrame with one row per (ref, hyp) word pair and their similarities.
    """
    similarities = _evaluate_all_pairs(ref_words, hyp_words, calculator)

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
                    "word_similarity": sim.lexical_similarity,
                    "position_similarity": sim.positional_similarity,
                    "similarity_weighted": sim.similarity_weighted,
                }
            )

    return pd.DataFrame(rows)


def _evaluate_all_pairs(
        ref_words: list[str],
        hyp_words: list[str],
        calculator: WordSimilarityCalculator,
) -> list[list[WordSimilarity]]:
    """Compare every ref word against every hyp word.
    """
    results = []
    for i, ref_word in enumerate(ref_words):
        ref_word_cleaned = clean_word(ref_word)
        pair_results = []
        for j, hyp_word in enumerate(hyp_words):
            hyp_word_cleaned = clean_word(hyp_word)

            similarity_weighted, lexical_similarity, positional_similarity = (
                calculator.combined_weighted_similarities(
                    ref_word=ref_word_cleaned,
                    ref_position=i,
                    hyp_word=hyp_word_cleaned,
                    hyp_position=j,
                ))

            pair_results.append(
                WordSimilarity(
                    target_index=j,
                    target_word=hyp_word_cleaned,
                    lexical_similarity=lexical_similarity,
                    positional_similarity=positional_similarity,
                    similarity_weighted=similarity_weighted,
                )
            )
        results.append(pair_results)
    return results
