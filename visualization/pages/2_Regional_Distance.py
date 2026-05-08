"""
Regional Distance from Standard German: Page 2.

Aggregate dialect distance per region, computed as mean per-sentence alignment
cost (substitution rows: 1 − sim; epsilon rows: λ = 0.45) divided by reference
word count. Restricted to STT4SG-350 train_balanced (~25k sentences per region;
sample-size-balanced subset of train_all) so per-region samples are comparable.
"""
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _data import (  # noqa: E402
    DAT_COLOR, DIT_COLOR, LAMBDA,
    cost_for_word_pair_by_similarity,
    load_alignments, load_balanced_paths,
)


@st.cache_data
def per_sentence_cost() -> pd.DataFrame:
    """Per-sentence alignment cost for both models, restricted to train_balanced paths."""
    balanced = load_balanced_paths()
    balanced_paths = set(balanced["path"])

    align = load_alignments()
    align = align[align["path"].isin(balanced_paths)].copy()
    align["is_ref"] = align["reference_word"].notna()
    align["cost"] = cost_for_word_pair_by_similarity(align["similarity"]).fillna(LAMBDA)

    grouped = (
        align.groupby(["path", "model"], observed=True)
        .agg(total_cost=("cost", "sum"), n_ref_words=("is_ref", "sum"))
        .reset_index()
    )
    grouped = grouped[grouped["n_ref_words"] > 0]
    grouped = grouped.merge(balanced, on="path", how="inner")
    grouped["cost_per_ref_word"] = grouped["total_cost"] / grouped["n_ref_words"]
    return grouped


@st.cache_data
def regional_summary(include_praet: bool) -> pd.DataFrame:
    """Per-region mean cost (DAT, DIT), DAT−DIT similarity delta, n. Sorted by delta desc."""
    df = per_sentence_cost()
    if not include_praet:
        df = df[~df["is_praeteritum"].fillna(False).astype(bool)]

    summary = (
        df.groupby(["dialect_region", "model"], observed=True)
        .agg(mean_cost=("cost_per_ref_word", "mean"),
             n_sentences=("path", "size"))
        .reset_index()
    )
    pivoted = summary.pivot(
        index="dialect_region", columns="model",
        values=["mean_cost", "n_sentences"],
    )
    out = pd.DataFrame({
        "DAT cost": pivoted[("mean_cost", "dialect-aware")],
        "DIT cost": pivoted[("mean_cost", "dialect-ignorant")],
        "delta (DAT sim − DIT sim)": pivoted[("mean_cost", "dialect-ignorant")] - pivoted[("mean_cost", "dialect-aware")],
        "n sentences": pivoted[("n_sentences", "dialect-aware")].astype(int),
    })
    # Default ordering: dialect signal strongest first.
    return out.sort_values("delta (DAT sim − DIT sim)", ascending=False).reset_index()


# --- Page ---
st.set_page_config(page_title="Regional Distance", layout="wide")
st.title("Regional Distance from Standard German")

st.markdown(
    "Aggregate dialect distance per region, measured as mean per-sentence alignment cost "
    f"(substitution: 1 − sim; ε rows: λ = {LAMBDA}) divided by reference word count. "
    "Computed on the **train_balanced** subset (~25k sentences per region; subset of train_all). "
    "Lower cost = closer to Standard German; higher delta (DAT sim − DIT sim) = stronger dialect signal. "
    "Regions sorted by delta (descending)."
)

include_praet = st.sidebar.toggle("Include Preterite sentences", value=False,
                                  help="Off by default: Preterite avoidance is a Swiss-German-wide "
                                       "feature, so it lifts every region's delta similarly and "
                                       "confounds the regional ranking. Toggle on for a 'total dialect "
                                       "distance' view.")

with st.spinner("Computing per-sentence alignment costs…"):
    summary = regional_summary(include_praet)
    per_sentence = per_sentence_cost()
    if not include_praet:
        per_sentence = per_sentence[~per_sentence["is_praeteritum"].fillna(False).astype(bool)]

# Sidebar at-a-glance counts (alignment rows behind the per-sentence aggregates + sentence count).
align_in_view = load_alignments()
align_in_view = align_in_view[align_in_view["path"].isin(set(per_sentence["path"]))]
st.sidebar.metric("Rows in view", f"{len(align_in_view):,}")
st.sidebar.metric("Unique sentences", f"{per_sentence['path'].nunique():,}")

regions_sorted = summary["dialect_region"].tolist()
model_scale = alt.Scale(
    domain=["dialect-aware", "dialect-ignorant"],
    range=[DAT_COLOR, DIT_COLOR],
)

# --- Headline plots: paired bars + delta bar (Altair, vertically stacked) ---
long_summary = summary.melt(
    id_vars=["dialect_region"],
    value_vars=["DAT cost", "DIT cost"],
    var_name="model_label",
    value_name="cost",
)
long_summary["model"] = long_summary["model_label"].map(
    {"DAT cost": "dialect-aware", "DIT cost": "dialect-ignorant"}
)

paired_chart = (
    alt.Chart(long_summary)
    .mark_bar(opacity=0.9)
    .encode(
        x=alt.X("dialect_region:N", sort=regions_sorted, title=None,
                axis=alt.Axis(labelAngle=0, labelOverlap=False)),
        y=alt.Y("cost:Q", title="Mean cost per ref word"),
        color=alt.Color("model:N", scale=model_scale, legend=alt.Legend(title=None)),
        xOffset=alt.XOffset("model:N"),
        tooltip=[
            alt.Tooltip("dialect_region:N", title="Region"),
            alt.Tooltip("model:N", title="Model"),
            alt.Tooltip("cost:Q", format=".4f", title="Cost"),
        ],
    )
    .properties(height=300, title="Mean alignment cost per region")
)

delta_chart = (
    alt.Chart(summary)
    .mark_bar(opacity=0.9)
    .encode(
        x=alt.X("dialect_region:N", sort=regions_sorted, title=None,
                axis=alt.Axis(labelAngle=0, labelOverlap=False)),
        y=alt.Y("delta (DAT sim − DIT sim):Q", title="Delta (DAT sim − DIT sim)"),
        color=alt.condition(
            alt.datum["delta (DAT sim − DIT sim)"] >= 0,
            alt.value(DIT_COLOR),
            alt.value(DAT_COLOR),
        ),
        tooltip=[
            alt.Tooltip("dialect_region:N", title="Region"),
            alt.Tooltip("delta (DAT sim − DIT sim):Q", format=".4f", title="Delta"),
            alt.Tooltip("n sentences:Q", title="N sentences"),
        ],
    )
    .properties(height=220, title="Dialect-specific distance per region (positive = DAT outperforms DIT)")
)
zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeWidth=0.6, color="black").encode(y="y:Q")

st.altair_chart(alt.vconcat(paired_chart, delta_chart + zero), use_container_width=True)

# --- Table ---
st.markdown("### Per-region summary")
display_table = summary.copy()
for col in ("DAT cost", "DIT cost", "delta (DAT sim − DIT sim)"):
    display_table[col] = display_table[col].round(4)
st.dataframe(display_table, hide_index=True, use_container_width=True)

# --- Distribution ---
st.markdown("### Per-sentence cost distribution")
st.caption(
    "Each box: within-region distribution of per-sentence alignment cost. "
    "Whiskers span 5th–95th percentile; box is q1–q3; tick is the median. "
    "Wide IQR = mixed dialect compliance across speakers."
)

# Aggregate to box stats server-side
def _box_stats(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (region, model), grp in df.groupby(["dialect_region", "model"], observed=True):
        v = grp["cost_per_ref_word"]
        out.append({
            "dialect_region": region, "model": model,
            "min": v.quantile(0.05), "q1": v.quantile(0.25),
            "median": v.median(),
            "q3": v.quantile(0.75), "max": v.quantile(0.95),
        })
    return pd.DataFrame(out)


box_stats = _box_stats(per_sentence)

base = alt.Chart(box_stats).encode(
    x=alt.X("dialect_region:N", sort=regions_sorted, title=None,
            axis=alt.Axis(labelAngle=0, labelOverlap=False)),
    color=alt.Color("model:N", scale=model_scale, legend=alt.Legend(title=None)),
    xOffset=alt.XOffset("model:N"),
)
whisker = base.mark_rule(strokeWidth=1.5).encode(
    y=alt.Y("min:Q", title="Cost per ref word"),
    y2="max:Q",
)
box = base.mark_bar(size=22, opacity=0.9).encode(y="q1:Q", y2="q3:Q")
median_tick = base.mark_tick(thickness=2.5, color="black", size=22).encode(y="median:Q")

st.altair_chart((whisker + box + median_tick).properties(height=380), use_container_width=True)