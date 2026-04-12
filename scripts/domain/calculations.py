"""
Shared helper functions for dialect analysis scripts.
"""

from models import CalculationParameters

import Levenshtein


def calculate_cost_for_word_pair_by_lexical_and_positional_score(
        src_word: str,
        src_position: int,
        target_word: str,
        target_position: int,
        calculation_parameters: CalculationParameters,
) -> float:
    """Calculate the cost between two words based on their lexical and positional similarity.

    This function determines the cost between two words using the Levenshtein distance, which
    can be normalized either globally or locally. It also considers the positional difference
    of the words in their respective sentences, normalized by the global maximum sentence length.
    The final cost is computed as one minus the weighted combination of lexical and positional
    similarity scores.

    Args:
        src_word: The source word for comparison.
        src_position: The position of the source word in the sentence.
        target_word: The target word for comparison.
        target_position: The position of the target word in the sentence.
        calculation_parameters: Matching configuration (alpha, normalization mode, max lengths).

    Returns:
        A float representing the cost calculated as one minus the combined similarity score.
    """
    # Calculate word similarity (lexical similarity) using Levenshtein distance, normalized either globally or locally. TODO also levels?
    if calculation_parameters.use_global_levenshtein_normalization:
        word_similarity = _calculate_word_similarity_global(
            src_word=src_word,
            target_word=target_word,
            global_max_word_length=calculation_parameters.max_word_len,
        )
    else:
        word_similarity = _calculate_word_similarity_local(
            src_word=src_word, target_word=target_word
        )

    # Calculate position similarity (positional similarity) based on distance in sentence, normalized by global max sentence length.
    position_score = _calculate_position_score(  # TODO maybe in levels (based on neighbourhood)
        src_index=src_position,
        target_index=target_position,
        global_max_sentence_length=calculation_parameters.max_sent_len,
    )

    # Combine lexical and positional similarity into a single score using a weighted average.
    score = calculate_score_weighted(word_similarity, position_score, alpha=calculation_parameters.alpha)

    return 1 - score  # Convert similarity to cost


def calculate_score_weighted(
        word_score: float, position_score: float, alpha=0.5
) -> float:
    """Compute weighted average of lexical and positional similarity scores.

    Args:
        word_score: Lexical similarity score in [0.0, 1.0].
        position_score: Positional similarity score in [0.0, 1.0].
        alpha: Weight for lexical similarity; (1 - alpha) is applied to positional. Defaults to 0.5.

    Returns:
        Combined score in [0.0, 1.0]. Higher means a better match.
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
        Harmonic mean in [0.0, 1.0]. Higher means a better match.
    """
    if word_score + position_score == 0:
        return 0.0
    return 2 * word_score * position_score / (word_score + position_score)


# -- Helper functions --

def _calculate_word_similarity_global(
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
        return max(0.0, 1 - (distance / global_max_word_length))
    else:
        return 1.0


def _calculate_word_similarity_local(src_word: str, target_word: str) -> float:
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


def _calculate_position_score(
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
        return max(0.0, 1.0 - (distance_to_target_word / max_possible_distance))
    else:
        return 1.0  # sentence only has 0 or 1 word, so position is irrelevant - treat as perfect match
