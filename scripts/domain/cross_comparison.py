"""
Cross-comparison logic for word-level dialect analysis.
"""

import pandas as pd

from domain.models import WordSimilarity
from domain.calculations import (
    clean,
    calculate_levenshtein_distance,
    normalize_levenshtein_distance,
    calculate_position_similarity,
    calculate_weighted_similarity,
    calculate_harmonic_similarity,
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
            dit_scores: list[WordSimilarity] = evaluate_scores(
                alpha,
                dit_words,
                src_word,
                i,
                global_max_word_length,
                global_max_sentence_length,
            )
            dat_scores: list[WordSimilarity] = evaluate_scores(
                alpha,
                dat_words,
                src_word,
                i,
                global_max_word_length,
                global_max_sentence_length,
            )

            ## TODO this currently only works for sentences with same amount of words. Think about solution with epsilon's.
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
                        "dit_word_similarity": dit.word_similarity,
                        "dat_word_similarity": dat.word_similarity,
                        "position_similarity": dit.position_similarity,
                        "dit_similarity_weighted": dit.similarity_weighted,
                        "dit_similarity_harmonic": dit.similarity_harmonic,
                        "dat_similarity_weighted": dat.similarity_weighted,
                        "dat_similarity_harmonic": dat.similarity_harmonic,
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
    Returns a list of WordSimilarity objects, one per target word.
    i = source word index
    j = target word index
    """
    results = []
    for j, target_word in enumerate(target_words):
        lev_distance = calculate_levenshtein_distance(src_word, target_word)
        word_similarity = normalize_levenshtein_distance(
            lev_distance, global_max_word_length
        )
        position_similarity = calculate_position_similarity(i, j, global_max_sentence_length)
        similarity_weighted = calculate_weighted_similarity(
            word_similarity, position_similarity, alpha
        )
        similarity_harmonic = calculate_harmonic_similarity(word_similarity, position_similarity)

        word_sim = WordSimilarity(
            j,
            target_word,
            word_similarity,
            position_similarity,
            similarity_weighted,
            similarity_harmonic,
        )

        results.append(word_sim)
    return results
