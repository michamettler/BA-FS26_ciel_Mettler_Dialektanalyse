"""Word-cloud overview for the Dialect Word Lexicon page.

Implementation methods for the dialect-candidate word cloud and its supporting
top-table: reference words where DAT outperforms DIT (Δ > 0) across the active
region filter.
"""
import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts

from _data import deletion_similarity

_TABLE_COLUMNS = ["word", "count", "DAT mean sim per ref word", "DIT mean sim per ref word",
                  "sim delta per ref word (DAT − DIT)"]


def render_caption() -> None:
    """Section header and explanatory text above the word cloud."""
    st.markdown("### Word Cloud for Dialect Specific Candidates")
    st.markdown(
        "Reference words ranked by **sim delta per ref word (DAT − DIT)** across the selected "
        "regions, restricted to words above the minimum-occurrences threshold (sidebar). "
        "Means are computed as the average over all DAT/DIT alignments for the reference "
        "word. Larger Delta means that DIT struggles more than DAT on this word, which indicates a "
        "dialect-specific word. Only deltas > 0 are displayed."
    )


def compute_top_table(df_view: pd.DataFrame, min_count_threshold: int) -> pd.DataFrame:
    """Top-200 reference words ranked by sim delta per ref word (DAT − DIT).

    Filters: insertions (ref_word = NaN) excluded; reference words must appear at least
    `min_count_threshold` times in the selected regions; only positive deltas (DAT outperforms DIT).
    Deletions count toward the mean with similarity = 1 − λ, matching the solver's ε cost.
    Returns an empty DataFrame (with the expected columns) if no candidates pass the filters.
    """
    rows = df_view[df_view["reference_word"].notna()].copy()
    if rows.empty:
        return pd.DataFrame(columns=_TABLE_COLUMNS)
    rows["similarity"] = rows["similarity"].fillna(deletion_similarity())

    sim = rows.pivot_table(
        index="reference_word", columns="model", values="similarity", aggfunc="mean"
    )
    counts = rows[rows["model"] == "dialect-aware"].groupby("reference_word").size()
    delta = (sim.get("dialect-aware") - sim.get("dialect-ignorant")).dropna()
    delta = delta[counts >= min_count_threshold]
    delta = delta[delta > 0].sort_values(ascending=False).head(200)

    if delta.empty:
        return pd.DataFrame(columns=_TABLE_COLUMNS)

    return pd.DataFrame({
        "word": delta.index,
        "count": [int(counts[w]) for w in delta.index],
        "DAT mean sim per ref word": [round(sim.loc[w, "dialect-aware"], 3) for w in delta.index],
        "DIT mean sim per ref word": [round(sim.loc[w, "dialect-ignorant"], 3) for w in delta.index],
        "sim delta per ref word (DAT − DIT)": delta.round(3).values,
    })


def render_word_cloud(table: pd.DataFrame) -> None:
    """Echarts word cloud where token size encodes delta."""
    cloud_data = [
        {"name": str(row["word"]), "value": float(row["sim delta per ref word (DAT − DIT)"])}
        for _, row in table.iterrows()
    ]
    option = {
        "tooltip": {"show": True},
        "series": [{
            "type": "wordCloud",
            "shape": "circle",
            "left": "center",
            "top": "center",
            "width": "100%",
            "height": "100%",
            "sizeRange": [14, 80],
            "rotationRange": [-30, 30],
            "rotationStep": 15,
            "gridSize": 8,
            "drawOutOfBound": False,
            "emphasis": {
                "focus": "self",
                "textStyle": {"shadowBlur": 8, "shadowColor": "#999"},
            },
            "data": cloud_data,
        }],
    }
    st_echarts(options=option, height="520px")


def render_top_candidates_expander(table: pd.DataFrame) -> None:
    """Collapsible table listing the top dialect-candidate words with per-model similarities."""
    deletion_sim = deletion_similarity()
    column_config = {
        "word": st.column_config.TextColumn(
            help="Standard German reference word.",
        ),
        "count": st.column_config.NumberColumn(
            help="Number of times this reference word appears in the selected regions "
                 "(the minimum-occurrences threshold).",
        ),
        "DAT mean sim per ref word": st.column_config.NumberColumn(
            help="Mean DAT similarity per reference word: average similarity across all alignments for this reference "
                 "word in the selected regions.",
        ),
        "DIT mean sim per ref word": st.column_config.NumberColumn(
            help="Mean DIT similarity per reference word: average similarity "
                 "across all alignments for this reference word in the selected regions.",
        ),
        "sim delta per ref word (DAT − DIT)": st.column_config.NumberColumn(
            help="Similarity delta per reference word: DAT mean similarity per reference word − "
                 "DIT mean similarity per reference word.",
        ),
    }
    with st.expander("Top Dialect Specific Candidates"):
        st.caption(
            "Top reference words where DAT outperforms DIT across the selected regions, which indicates a "
            "dialect-specific candidate.")
        st.dataframe(table, use_container_width=True, hide_index=True, column_config=column_config)
        st.caption("Default ordering: descending by **sim delta per ref word (DAT − DIT)**.")
        st.caption(
            f"Deletions count with similarity = 1 − λ = {deletion_sim:.2f}, matching the solver's ε cost."
        )
