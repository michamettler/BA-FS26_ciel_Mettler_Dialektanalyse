"""
Cross-comparison logic for word-level dialect analysis.
"""

import pandas as pd

from domain.models import WordScore
from scripts.domain.calculations import (
    clean,
    calculate_levenshtein_distance,
    normalize_levenshtein_distance,
    calculate_position_score,
    calculate_score_weighted,
    calculate_score_harmonic,
)


def generate_cross_comparison_df(
    df, global_max_word_length, global_max_sentence_length, alpha=0.5
):
    """Generates a DataFrame where each row corresponds to a comparison between a source word and a target word (DIT and DAT).
    We compare each source word to every word in the target sentences (DIT and DAT) to capture all possible alignments,
    not just same-index pairs, since word order can differ.
    The calculated scores (Levenshtein similarity, position similarity, weighted and harmonic) allow us to analyze how closely
    the DIT and DAT sentences match the source sentence at the word level, accounting for both lexical similarity and
    positional alignment.
    """
    word_rows = []

    for _, row in df.iterrows():
        clip_id = row["path"]

        # create list of words for source, DIT and DAT sentence & clean them
        src_words = [clean(w) for w in str(row["sentence"]).split()]
        dit_words = [clean(w) for w in str(row["DIT"]).split()]
        dat_words = [clean(w) for w in str(row["DAT"]).split()]

        n_src_words = len(src_words)  # number of words in source sentence
        n_dit_words = len(dit_words)  # number of words in DIT target sentence
        n_dat_words = len(dat_words)  # number of words in DAT target sentence

        for i, src_word in enumerate(src_words):
            dit_scores: list[WordScore] = evaluate_scores(
                alpha,
                dit_words,
                src_word,
                i,
                global_max_word_length,
                global_max_sentence_length,
            )
            dat_scores: list[WordScore] = evaluate_scores(
                alpha,
                dat_words,
                src_word,
                i,
                global_max_word_length,
                global_max_sentence_length,
            )

            for dit, dat in zip(dit_scores, dat_scores):
                word_rows.append(
                    {
                        "clip_id": clip_id,
                        "src_word_index": i,
                        "dit_word_index": dit.target_index,
                        "dat_word_index": dat.target_index,
                        "src_word": src_word,
                        "dit_word": dit.target_word,
                        "dat_word": dat.target_word,
                        "dit_word_score": dit.word_score,
                        "dat_word_score": dat.word_score,
                        "position_score": dit.position_score,
                        "dit_score_weighted": dit.score_weighted,
                        "dit_score_harmonic": dit.score_harmonic,
                        "dat_score_weighted": dat.score_weighted,
                        "dat_score_harmonic": dat.score_harmonic,
                        "src_len": n_src_words,
                        "dit_len": n_dit_words,
                        "dat_len": n_dat_words,
                    }
                )

    return pd.DataFrame(word_rows)


def evaluate_scores(
    alpha, target_words, src_word, i, global_max_word_length, global_max_sentence_length
):
    """Compares src_word against every word in target_words.
    Returns a list of WordScore objects, one per target word.
    i = source word index
    j = target word index
    """
    results = []
    for j, target_word in enumerate(target_words):
        lev_distance = calculate_levenshtein_distance(src_word, target_word)
        word_similarity = normalize_levenshtein_distance(
            lev_distance, global_max_word_length
        )
        position_score = calculate_position_score(i, j, global_max_sentence_length)
        score_weighted = calculate_score_weighted(
            word_similarity, position_score, alpha
        )
        score_harmonic = calculate_score_harmonic(word_similarity, position_score)

        word_score = WordScore(
            j,
            target_word,
            word_similarity,
            position_score,
            score_weighted,
            score_harmonic,
        )

        results.append(word_score)
    return results
