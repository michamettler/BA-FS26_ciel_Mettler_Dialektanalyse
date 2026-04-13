from dataclasses import dataclass


@dataclass
class CalculationParameters:
    """Configuration for bipartite word-matching.

    Tuning parameters:
        alpha: Weight for lexical vs. positional similarity (default 0.7 = 70 % lexical).
        lambda_: Penalty cost for unmatched words routed through epsilon-nodes, in [0, 1].
        use_global_levenshtein_normalization: Whether to normalize Levenshtein distance
            globally (by max_word_len) or locally (per word pair).
        max_word_len: Optional longest word length used for global Levenshtein
            normalization. Required only when global normalization is enabled.
        max_sent_len: Longest sentence length used for positional-score normalization.
    """
    max_sent_len: int
    alpha: float = 0.7
    lambda_: float = 0.7
    use_global_levenshtein_normalization: bool = False
    max_word_len: int | None = None

    def __post_init__(self):
        if self.max_sent_len is None:
            raise ValueError("max_sent_len must be set")
        if self.use_global_levenshtein_normalization:
            if self.max_word_len is None:
                raise ValueError(
                    "max_word_len must be set when use_global_levenshtein_normalization is enabled"
                )
            if self.max_word_len <= 0:
                raise ValueError(
                    f"max_word_len must be greater than 0 for normalization; got {self.max_word_len}"
                )
        if self.max_sent_len <= 0:
            raise ValueError(
                f"max_sent_len must be greater than 0 for normalization; got {self.max_sent_len}"
            )
        if not 0 <= self.alpha <= 1:
            raise ValueError(f"alpha must be between 0 and 1 inclusive; got {self.alpha}")
        if not 0 <= self.lambda_ <= 1:
            raise ValueError(f"lambda_ must be between 0 and 1 inclusive; got {self.lambda_}")


@dataclass
class WordSimilarity:
    """Similarity scores for comparing a source word against a single target word."""
    target_index: int
    target_word: str
    word_similarity: float
    position_similarity: float
    similarity_weighted: float
