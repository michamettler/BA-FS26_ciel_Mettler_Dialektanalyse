"""
Similarity calculation for dialect analysis.
"""

import Levenshtein

# Scale factor: network simplex requires integer weights, so float costs are scaled to ints.
_COST_SCALE = 1000


def cost_for_word_pair_by_similarity(similarity: float) -> int:
    """Convert a similarity to an integer cost for the network flow algorithm.
    Cost = (1 - similarity) * _COST_SCALE, rounded to the nearest integer.
    """
    return round((1 - similarity) * _COST_SCALE)


class WordSimilarityCalculator:
    """Calculates word-level similarities and costs based on configurable parameters

    Args:
        sent_len: sentence length used for positional-similarity normalization (either global over a whole dataset or
            local to a single reference/hypothesis sentence).
        alpha: Weight for lexical vs. positional similarity (1 = lexical only, 0 = positional only, default 0.7).
        lambda_: Penalty for unmatched words routed through epsilon-nodes, in [0, 1].
        use_global_lexical_normalization: Normalize lexical similarity globally or locally.
        max_word_len: Longest word length for global Levenshtein normalization (required if global).
    """

    def __init__(
            self,
            sent_len: int,
            alpha: float = 0.7,
            lambda_: float = 0.3,
            use_global_lexical_normalization: bool = False,
            max_word_len: int | None = None,
    ):
        if sent_len is None or sent_len <= 0:
            raise ValueError(f"sent_len must be > 0; got {sent_len}")
        if use_global_lexical_normalization:
            if max_word_len is None or max_word_len <= 0:
                raise ValueError(f"max_word_len must be > 0 when global normalization is enabled; got {max_word_len}")
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be between 0 and 1; got {alpha}")
        if not 0 <= lambda_ <= 1:
            raise ValueError(f"lambda_ must be between 0 and 1; got {lambda_}")

        self.sent_len = sent_len
        self.alpha = alpha
        self.lambda_ = lambda_
        self.use_global_lexical_normalization = use_global_lexical_normalization
        self.max_word_len = max_word_len

    def combined_weighted_lexical_positional_similarity(
            self,
            ref_word: str,
            ref_position: int,
            hyp_word: str,
            hyp_position: int,
    ) -> float:
        """Calculate the combined similarity for a word pair as a weighted average of lexical and positional similarity.
        Alpha controls the weight (1 = lexical only, 0 = positional only).

        Weighted_similarity = alpha * lexical_similarity + (1 - alpha) * positional_similarity

        Args:
            ref_word: Reference word.
            ref_position: Reference word position in the sentence.
            hyp_word: Hypothesis word.
            hyp_position: Hypothesis word position in the sentence.
        Returns:
            The weighted combined similarity in [0.0, 1.0].
        """
        lexical_similarity = self.lexical_similarity(ref_word, hyp_word)
        positional_similarity = self.positional_similarity(ref_position, hyp_position)

        weighted_similarity = self.alpha * lexical_similarity + (1 - self.alpha) * positional_similarity
        return weighted_similarity

    def lexical_similarity(self, ref_word: str, hyp_word: str) -> float:
        """Lexical similarity via Levenshtein distance, normalized globally or locally.

        Global normalization (uses Levenshtein.distance): 1 - (distance / max_word_len)
        Local normalization (uses Levenshtein.ratio): 1 - (distance / (len1 + len2))
        """
        if self.use_global_lexical_normalization:
            distance = Levenshtein.distance(ref_word, hyp_word)

            if self.max_word_len is None:
                raise ValueError(
                    "max_word_len must be set when use_global_lexical_normalization is True"
                )
            return max(0.0, 1 - (distance / self.max_word_len))  # max to make sure similarity is never negative
        else:
            return Levenshtein.ratio(ref_word, hyp_word)

    def positional_similarity(self, ref_index: int, hyp_index: int) -> float:
        """Positional similarity via the gap between word positions, normalized by sentence length.

        Returns 1.0 if positions are identical, 0.0 if the gap is maximal relative to the sentence length.
        """
        if self.sent_len > 1:
            distance_to_hyp_word = abs(ref_index - hyp_index)
            max_possible_distance = (
                    self.sent_len - 1
            )  # -1 because max index gap in a sequence of length n is n-1, needed for worst case to be 0
            return max(0.0,
                       1.0 - (distance_to_hyp_word / max_possible_distance)
                       )  # max to make sure similarity is never negative
        else:
            return 1.0  # sentence only has 0 or 1 words, so position is irrelevant - treat as perfect match

    def cost_for_epsilon_by_penalty(self) -> int:
        """Convert an epsilon penalty to an integer cost for the network flow algorithm.

        Cost = penalty (lambda_) * _COST_SCALE, rounded to the nearest integer."""
        return round(self.lambda_ * _COST_SCALE)
