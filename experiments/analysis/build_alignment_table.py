"""
Build word-alignment parquets for STT4SG-350 train_all.

Aligns each sentence's reference (canonical Hochdeutsch) against both ASR outputs
using the calibrated bipartite solver and writes one parquet per model.
Per-sentence metadata (region, speaker, etc.) lives in the source transcript TSVs
and is joined back at analysis time on `path`.

Run: python experiments/analysis/build_alignment_table.py
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "domain"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "utils"))

from bipartite_matching import (  # noqa: E402
    ATTR_SIMILARITY,
    ATTR_WORD,
    build_full_bipartite_graph,
    build_reduced_graph_by_matching,
    extract_index_from_node_name,
    get_bipartite_edges,
    is_eps_node,
    solve_matching,
)
from word_similarity_calculator import WordSimilarityCalculator  # noqa: E402

# Paths
DAT_TSV = PROJECT_ROOT / "transcripts" / "dialect-aware" / "fhnw" / "stt4sg" / "train_all_transcribed.tsv"
DIT_TSV = PROJECT_ROOT / "transcripts" / "dialect-ignorant" / "whisper-large-v2" / "stt4sg" / "train_all_enriched_transcribed_praet.tsv"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "analysis"
DAT_OUT = OUTPUT_DIR / "train_all_alignments_dialect-aware.parquet"
DIT_OUT = OUTPUT_DIR / "train_all_alignments_dialect-ignorant.parquet"

# Fixed Hyperparameters, based on analysis in scripts/bipartite-matching-hyperparameters.ipynb, param-selection cell.
ALPHA = 0.85
LAMBDA_ = 0.45
USE_GLOBAL_LEXICAL_NORM = False
USE_SQUARED_POSITIONAL = True

# Other constants
DAT_HYP_COL = "fhnw_transcript"
DIT_HYP_COL = "whisper_large_v2_transcript"

OUTPUT_COLUMNS = ["path", "reference_word", "hypothesis_word", "reference_index", "hypothesis_index", "similarity"]


def load_and_filter() -> pd.DataFrame:
    """Read DAT and DIT TSVs, inner-join on `path`, apply quality filters.
    Needed because annotations live in dialect-ignorant TSV."""
    dat = pd.read_csv(DAT_TSV, sep="\t", encoding="utf-8-sig")[
        ["path", "sentence", DAT_HYP_COL, "dialect_region"]
    ]
    dit = pd.read_csv(DIT_TSV, sep="\t", encoding="utf-8-sig")[
        ["path", DIT_HYP_COL, "clip_is_usable", "drop_reason"]
    ]
    df = dat.merge(dit, on="path", how="inner", validate="one_to_one")

    df = df[df["dialect_region"].notna() & (df["dialect_region"].astype(str).str.strip() != "")]
    df = df[(df["clip_is_usable"] == True) & (df["drop_reason"].fillna("").str.strip() == "")]
    df = df[df[DAT_HYP_COL].notna() & ~df[DAT_HYP_COL].astype(str).str.startswith("ERROR")]
    df = df[df[DIT_HYP_COL].notna() & ~df[DIT_HYP_COL].astype(str).str.startswith("ERROR")]
    return df.reset_index(drop=True)


def align_pair(reference: str, hypothesis: str) -> list[dict]:
    """Run the calibrated solver on one ref/hyp pair, return one alignment per real-word edge.

    Skips epsilon-to-epsilon edges; NA on the absent side for deletions/insertions.
    """
    ref_words = str(reference).split()
    hyp_words = str(hypothesis).split()
    if not ref_words or not hyp_words:
        return []

    calc = WordSimilarityCalculator(
        sent_len=max(len(ref_words), len(hyp_words)),
        alpha=ALPHA, lambda_=LAMBDA_,
        use_global_lexical_normalization=USE_GLOBAL_LEXICAL_NORM,
        use_squared_positional=USE_SQUARED_POSITIONAL,
    )
    G = build_full_bipartite_graph(ref_words, hyp_words, calc)
    matching = solve_matching(G)
    M = build_reduced_graph_by_matching(G, matching)  # epsilon-to-epsilon pairs already removed

    alignments: list[dict] = []
    for u, v in get_bipartite_edges(M):
        u_is_eps = is_eps_node(M.nodes[u])
        v_is_eps = is_eps_node(M.nodes[v])

        if u_is_eps:
            j = extract_index_from_node_name(v)
            alignments.append(_alignment_row(None, M.nodes[v][ATTR_WORD], None, j, None))
        elif v_is_eps:
            i = extract_index_from_node_name(u)
            alignments.append(_alignment_row(M.nodes[u][ATTR_WORD], None, i, None, None))
        else:
            i = extract_index_from_node_name(u)
            j = extract_index_from_node_name(v)
            similarity = M.edges[u, v][ATTR_SIMILARITY]

            alignments.append(_alignment_row(
                M.nodes[u][ATTR_WORD], M.nodes[v][ATTR_WORD], i, j, similarity
            ))
    return alignments


def _alignment_row(reference_word, hypothesis_word, reference_index, hypothesis_index, similarity) -> dict:
    return {
        "reference_word": reference_word, "hypothesis_word": hypothesis_word,
        "reference_index": reference_index, "hypothesis_index": hypothesis_index,
        "similarity": similarity,
    }


def write_alignments(alignments: list[dict], out_path: Path) -> None:
    """Write alignments to zstd-compressed parquet with deterministic row order."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # `columns=` ensures the schema is set even when alignments is empty (no KeyError on sort).
    df = (
        pd.DataFrame(alignments, columns=OUTPUT_COLUMNS)
        .sort_values(["path", "reference_index", "hypothesis_index"])
        .reset_index(drop=True)
    )
    # Nullable Int32 keeps insertion/deletion rows as proper missing values, not float NaN.
    df["reference_index"] = df["reference_index"].astype("Int32")
    df["hypothesis_index"] = df["hypothesis_index"].astype("Int32")
    df.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def main() -> None:
    print(f"[{_now()}] start building alignment tables for train_all")
    t0 = time.perf_counter()
    n_workers = os.cpu_count() or 1

    df = load_and_filter()
    print(f"Filtered to {len(df):,} sentence pairs")

    for hyp_col, out_path, label in [
        (DAT_HYP_COL, DAT_OUT, "dialect-aware"),
        (DIT_HYP_COL, DIT_OUT, "dialect-ignorant"),
    ]:
        tasks = list(zip(df["path"], df["sentence"], df[hyp_col]))
        alignments = run_alignments_in_parallel(tasks, n_workers)
        write_alignments(alignments, out_path)
        size_mb = out_path.stat().st_size / 1e6
        print(f"Wrote {out_path.name}: {len(alignments):,} {label} alignments ({size_mb:.0f} MB)")

    runtime = time.perf_counter() - t0
    print(f"[{_now()}] done in {runtime:.1f}s on {n_workers} workers")


# --- Parallelization Code ---

def chunk_worklist(tasks: list[tuple], n_chunks: int) -> list[list[tuple]]:
    chunk_size = max(1, (len(tasks) + n_chunks - 1) // n_chunks)
    return [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]


def process_batch(tasks: list[tuple]) -> list[dict]:
    """Worker entry point: align a batch of (path, ref, hyp) tasks."""
    out: list[dict] = []
    for path, ref, hyp in tasks:
        for row in align_pair(ref, hyp):
            row["path"] = path
            out.append(row)
    return out


def run_alignments_in_parallel(tasks: list[tuple], n_workers: int) -> list[dict]:
    """Distribute tasks across workers; aggregate alignments with periodic progress."""
    chunks = chunk_worklist(tasks, n_workers * 8)
    print(f"  dispatching {len(tasks):,} sentence pairs across {len(chunks)} batches × {n_workers} workers")

    out: list[dict] = []
    log_interval = max(1, len(chunks) // 20)
    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(process_batch, c) for c in chunks]
        done = 0
        for fut in as_completed(futures):
            out.extend(fut.result())
            done += 1
            if done % log_interval == 0 or done == len(chunks):
                elapsed = time.perf_counter() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(chunks) - done) / rate if rate > 0 else 0
                print(f"    {done}/{len(chunks)} batches  ({rate:.1f}/s, ETA {eta:.0f}s)")
    return out


if __name__ == "__main__":
    main()