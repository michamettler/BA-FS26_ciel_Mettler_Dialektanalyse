"""
Similarity calculation for dialect analysis.
"""

import Levenshtein

# Scale factor: network simplex requires integer weights, so float costs are scaled to ints.
COST_SCALE = 1000


def cost_for_word_pair_by_similarity(similarity: float) -> float:
    """Convert a similarity to a cost.
    Cost = (1 - similarity).
    """
    return 1 - similarity


def similarity_for_word_pair_by_cost(cost: float) -> float:
    """Convert a cost to a similarity (inverse of `cost_for_word_pair_by_similarity`).
    Similarity = (1 - cost).
    """
    return 1 - cost


def scale_cost_for_networkx(cost: float) -> int:
    """Convert the theoretical cost to an integer cost for the network flow algorithm.
    Cost = cost * COST_SCALE, rounded to the nearest integer.
    """
    return round(cost * COST_SCALE)


class WordSimilarityCalculator:
    """Calculates word-level similarities and costs based on configurable parameters

    Args:
        sent_len: sentence length used for positional-similarity normalization (either global over a whole dataset or
            local to a single reference/hypothesis sentence).
        alpha: Weight for lexical vs. positional similarity (1 = lexical only, 0 = positional only, default 0.7).
        lambda_: Penalty for unmatched words routed through epsilon-nodes, in [0, 1].
        use_global_lexical_normalization: Normalize lexical similarity globally or locally.
        max_word_len: Longest word length for global Levenshtein normalization (required if global).
        use_squared_positional: If True, square the positional similarity to penalize larger gaps more steeply.
    """

    def __init__(
            self,
            sent_len: int,
            alpha: float = 0.7,
            lambda_: float = 0.3,
            use_global_lexical_normalization: bool = False,
            max_word_len: int | None = None,
            use_squared_positional: bool = False,
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
        self.use_squared_positional = use_squared_positional

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
            return 1 - (distance / self.max_word_len)
        else:
            return Levenshtein.ratio(ref_word, hyp_word)

    def positional_similarity(self, ref_index: int, hyp_index: int) -> float:
        """Positional similarity via the gap between word positions, normalized by sentence length.

        For N > 1:  sim = 1 - |i - j| / (N - 1)
        For N = 1:  sim = 1.0 (single-word sentence — position is irrelevant)

        When use_squared_positional is True, the linear similarity is squared;
        small gaps stay close to 1, mid/large gaps are penalized more.
        """
        if self.sent_len > 1:
            linear_similarity = 1.0 - abs(ref_index - hyp_index) / (self.sent_len - 1)
            return linear_similarity ** 2 if self.use_squared_positional else linear_similarity
        return 1.0

    def cost_for_epsilon_by_penalty(self) -> float:
        return self.lambda_
