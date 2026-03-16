from dataclasses import dataclass


@dataclass
class WordScore:
    """Scores for comparing a source word against a single target word."""
    target_index: int
    target_word: str
    word_score: float
    position_score: float
    score_weighted: float
    score_harmonic: float
