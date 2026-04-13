from dataclasses import dataclass


@dataclass
class WordSimilarity:
    """Similarity scores for comparing a source word against a single target word."""
    target_index: int
    target_word: str
    lexical_similarity: float
    positional_similarity: float
    similarity_weighted: float
