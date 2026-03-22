def best_matchings_df(word_comparison_results_df, transcript, score_type="weighted"):
    """Naive matching based on highest score.
    Extended matching logic will be added to domain later on."
    """
    score_col = f"{transcript.lower()}_score_{score_type}"
    word_col = f"{transcript.lower()}_word"
    return word_comparison_results_df.loc[
        word_comparison_results_df.groupby(["clip_id", "src_word_index"], sort=False)[
            score_col
        ].idxmax()
    ][["clip_id", "src_word_index", "src_word", word_col, score_col]].reset_index(
        drop=True
    )