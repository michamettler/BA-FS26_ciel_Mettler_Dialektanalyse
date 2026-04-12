from dataclasses import dataclass


@dataclass
class CalculationParameters:
    """Configuration for bipartite word-matching.

    Tuning parameters:
        alpha: Weight for lexical vs. positional similarity (default 0.7 = 70 % lexical).
        lambda_: Penalty cost for unmatched words routed through ε-nodes, in [0, 1].
        use_global_levenshtein_normalization: Whether to normalize Levenshtein distance
            globally (by max_word_len) or locally (per word pair).
        max_word_len: Longest word length used for global Levenshtein normalization.
        max_sent_len: Longest sentence length used for positional-score normalization.
    """
    max_word_len: int
    max_sent_len: int
    alpha: float = 0.7
    lambda_: float = 0.3
    use_global_levenshtein_normalization: bool = False

    def __post_init__(self):
        if self.max_word_len is None:
            raise ValueError(f"max_word_len must be set")
        if self.max_sent_len is None:
            raise ValueError(f"max_sent_len must be set")


@dataclass
class WordScore:
    """Scores for comparing a source word against a single target word."""
    target_index: int
    target_word: str
    word_score: float
    position_score: float
    score_weighted: float
    score_harmonic: float
