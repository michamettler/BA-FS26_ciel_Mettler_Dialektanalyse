"""
Shared helper functions for dialect analysis scripts.
"""

from models import CalculationParameters

import Levenshtein

# Scale factor: network simplex requires integer weights, so float costs are scaled to ints.
_COST_SCALE = 1000


def calculate_similarity_for_word_pair_by_weighted_lexical_and_positional_similarities(
        ref_word: str,
        ref_position: int,
        hyp_word: str,
        hyp_position: int,
        calculation_parameters: CalculationParameters,
) -> float:
    """Calculate the similarity score between two words based on lexical and positional similarity.

    Lexical similarity is computed via Levenshtein distance (normalized globally or locally).
    Positional similarity is based on the distance of word positions in their respective sentences,
    normalized by the global maximum sentence length.
    The two are combined into a single similarity using a weighted average (alpha).

    Args:
        ref_word: The reference word for comparison.
        ref_position: The position of the reference word in the sentence.
        hyp_word: The hypothesis word for comparison.
        hyp_position: The position of the hypothesis word in the sentence.
        calculation_parameters: Matching configuration (alpha, normalization mode, max lengths).

    Returns:
        Similarity in [0.0, 1.0]. Higher means a better match.
    """
    # Calculate word similarity (lexical similarity) using Levenshtein distance, normalized either globally or locally. TODO also levels?
    if calculation_parameters.use_global_levenshtein_normalization:
        word_similarity = _calculate_word_similarity_global(
            ref_word=ref_word,
            hyp_word=hyp_word,
            global_max_word_length=calculation_parameters.max_word_len,
        )
    else:
        word_similarity = _calculate_word_similarity_local(
            ref_word=ref_word, hyp_word=hyp_word
        )

    # Calculate position similarity based on distance in sentence, normalized by global max sentence length.
    position_similarity = _calculate_position_similarity(  # TODO maybe in levels (based on neighbourhood)
        ref_index=ref_position,
        hyp_index=hyp_position,
        global_max_sentence_length=calculation_parameters.max_sent_len,
    )

    # Combine lexical and positional similarity into a single similarity using a weighted average.
    return calculate_weighted_similarity(word_similarity, position_similarity, alpha=calculation_parameters.alpha)


def calculate_cost_for_word_pair_by_similarity(similarity: float) -> int:
    """Convert a similarity to an integer cost for the network flow algorithm.

    Cost = (1 - similarity) * _COST_SCALE, rounded to the nearest integer.

    Args:
        similarity: Similarity in [0.0, 1.0].

    Returns:
        Integer cost (0 = perfect match, _COST_SCALE = no similarity).
    """
    return round((1 - similarity) * _COST_SCALE)


def calculate_cost_for_epsilon_by_penalty(penalty: float) -> int:
    """Convert an epsilon penalty to an integer cost for the network flow algorithm.

    Cost = penalty * _COST_SCALE, rounded to the nearest integer.

    Args:
        penalty: Penalty in [0.0, 1.0]. Higher means more costly to leave a word unmatched.

    Returns:
        Integer cost (0 = free epsilon routing, _COST_SCALE = maximum penalty).
    """
    return round(penalty * _COST_SCALE)


def calculate_weighted_similarity(
        word_similarity: float, position_similarity: float, alpha=0.5
) -> float:
    """Compute weighted average of lexical and positional similarity.

    Args:
        word_similarity: Lexical similarity in [0.0, 1.0].
        position_similarity: Positional similarity in [0.0, 1.0].
        alpha: Weight for lexical similarity; (1 - alpha) is applied to positional. Defaults to 0.5.

    Returns:
        Combined similarity in [0.0, 1.0]. Higher means a better match.
    """
    return alpha * word_similarity + (1 - alpha) * position_similarity


def calculate_harmonic_similarity(word_similarity: float, position_similarity: float) -> float:
    """Compute harmonic mean (F1-style) of lexical and positional similarity.

    Penalizes pairs where either similarity is low.
    Based on: van Rijsbergen (1979). Information Retrieval. Butterworths.

    Args:
        word_similarity: Lexical similarity in [0.0, 1.0].
        position_similarity: Positional similarity in [0.0, 1.0].

    Returns:
        Harmonic mean in [0.0, 1.0]. Higher means a better match.
    """
    if word_similarity + position_similarity == 0:
        return 0.0
    return 2 * word_similarity * position_similarity / (word_similarity + position_similarity)


# -- Helper functions --

def _calculate_word_similarity_global(
        ref_word: str, hyp_word: str, global_max_word_length: int | None
) -> float:
    """Calculate normalized Levenshtein similarity using a global maximum word length.

    Normalization is based on global_max_word_length to achieve comparable similarities
    across the dataset.

    Args:
        ref_word: Reference word.
        hyp_word: Hypothesis word to compare against.
        global_max_word_length: Globally known maximum word length for normalization.

    Returns:
        Similarity in [0.0, 1.0]. Higher means more similar.
    """
    distance = Levenshtein.distance(ref_word, hyp_word)

    if global_max_word_length is None:
        raise ValueError(
            "global_max_word_length must be set when use_global_levenshtein_normalization is True"
        )

    if global_max_word_length > 0:
        return max(0.0, 1 - (distance / global_max_word_length))
    else:
        return 1.0


def _calculate_word_similarity_local(ref_word: str, hyp_word: str) -> float:
    """Calculate normalized Levenshtein similarity using local (per-pair) normalization.

    Delegates to python-Levenshtein's ratio:
    similarity = (len(ref) + len(hyp) - distance) / (len(ref) + len(hyp)).

    Args:
        ref_word: Reference word.
        hyp_word: Hypothesis word to compare against.

    Returns:
        Similarity in [0.0, 1.0]. Higher means more similar.
    """
    return Levenshtein.ratio(ref_word, hyp_word)


def _calculate_position_similarity(
        ref_index: int, hyp_index: int, global_max_sentence_length: int
) -> float:
    """Calculate normalized position similarity between two word positions.

    Returns 1.0 if positions are identical, 0.0 if the gap is maximal relative
    to the global maximum sentence length.

    Args:
        ref_index: Position of the reference word in its sentence.
        hyp_index: Position of the hypothesis word in its sentence.
        global_max_sentence_length: Globally known maximum sentence length for normalization.

    Returns:
        Position similarity in [0.0, 1.0]. Higher means closer positions.
    """
    if global_max_sentence_length > 1:
        distance_to_hyp_word = abs(ref_index - hyp_index)
        max_possible_distance = (
                global_max_sentence_length - 1
        )  # -1 because max index gap in a sequence of length n is n-1, needed for worst case to be 0
        return max(0.0, 1.0 - (distance_to_hyp_word / max_possible_distance))
    else:
        return 1.0  # sentence only has 0 or 1 word, so position is irrelevant - treat as perfect match
