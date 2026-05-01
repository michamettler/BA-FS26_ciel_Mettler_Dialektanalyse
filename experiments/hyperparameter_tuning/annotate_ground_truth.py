"""Interactive terminal CLI for ground-truth alignment annotation.

Reads a metadata TSV (with `path`, `sentence`, and a hypothesis column),
pre-fills an alignment per sample using the bipartite solver, lets the user
correct it, and writes a JSON file matching the synthetic GT schema.

Usage:
    python annotate_ground_truth.py <metadata.tsv> [--alpha 0.7] [--lambda 0.5]
                                     [--use-global-norm]
                                     [--path-column path]
                                     [--hyp-column whisper_large_v2_transcript]
                                     [--hyp-suffix dit]
                                     [--output <path>]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.text import Text

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "domain"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "utils"))

from bipartite_matching import (  # noqa: E402
    HYPOTHESIS_PARTITION, REFERENCE_PARTITION,
    build_full_bipartite_graph, extract_index_from_node_name,
    get_node_name, is_eps_node, solve_matching,
)
from preprocessing import clean_word  # noqa: E402
from word_similarity_calculator import WordSimilarityCalculator  # noqa: E402

DATASET_SOURCE = "STT4SG-350 v2.1 (test)"

console = Console()


def tokenize(text: str) -> list[str]:
    return [w for w in clean_word(text).split() if w]


def solver_prefill(ref: list[str], hyp: list[str], alpha: float, lambda_: float,
                   use_global_norm: bool, global_max_word_len: int | None,
                   ) -> tuple[dict[int, int | None], dict[int, float | None]]:
    """Run the bipartite solver and return (alignment, similarities) keyed by ref_idx.

    similarities[i] is the edge similarity for ref[i] (None if matched to epsilon).
    """
    if not ref:
        return {}, {}
    sent_len = max(len(ref), len(hyp), 1)
    calc = WordSimilarityCalculator(
        alpha=alpha, lambda_=lambda_, sent_len=sent_len,
        use_global_lexical_normalization=use_global_norm,
        max_word_len=global_max_word_len if use_global_norm else None,
    )
    G = build_full_bipartite_graph(ref, hyp, calc)
    matching = solve_matching(G)

    alignment: dict[int, int | None] = {}
    similarities: dict[int, float | None] = {}
    for ref_idx in range(len(ref)):
        ref_node = get_node_name(REFERENCE_PARTITION, ref_idx)
        hyp_node = matching[ref_node]
        if is_eps_node(G.nodes[hyp_node]):
            alignment[ref_idx] = None
            similarities[ref_idx] = None
        else:
            hyp_idx = extract_index_from_node_name(hyp_node)
            alignment[ref_idx] = hyp_idx
            similarities[ref_idx] = G.edges[ref_node, hyp_node]["similarity"]
    return alignment, similarities


def color_for_similarity(sim: float | None) -> str:
    if sim is None:
        return "dim"
    if sim >= 0.9:
        return "green"
    if sim >= 0.5:
        return "yellow"
    return "red"


def render_sample(progress: str, path: str, ref: list[str], hyp: list[str],
                  alignment: dict[int, int | None], similarities: dict[int, float | None],
                  edited: set[int], alpha: float, lambda_: float, use_global_norm: bool) -> None:
    console.rule(f"[bold cyan]{progress}[/bold cyan]")
    console.print(f"[dim]path:[/dim] {path}")
    console.print(
        f"[dim]α={alpha} λ={lambda_} use_global_lexical_normalization={use_global_norm}[/dim]\n"
    )

    console.print(_words_with_indices_panel("REF", ref))
    console.print(_words_with_indices_panel("HYP", hyp))
    console.print()

    table = Table(show_header=True, header_style="bold")
    table.add_column("ref")
    table.add_column("→", justify="center")
    table.add_column("hyp")
    table.add_column("similarity", justify="right")
    table.add_column("", width=2)

    for i, ref_word in enumerate(ref):
        hyp_idx = alignment.get(i)
        sim = similarities.get(i)
        sim_color = color_for_similarity(sim)
        ref_cell = Text(f"[{i}] {ref_word}")
        if hyp_idx is None:
            hyp_cell = Text("ε (deletion)", style="dim")
            sim_cell = Text("—", style="dim")
        else:
            hyp_cell = Text(f"[{hyp_idx}] {hyp[hyp_idx]}", style=sim_color)
            sim_cell = Text(f"{sim:.3f}" if sim is not None else "—", style=sim_color)
        edited_marker = Text("*", style="bold magenta") if i in edited else Text("")
        table.add_row(ref_cell, Text("→", style="dim"), hyp_cell, sim_cell, edited_marker)

    console.print(table)


def _words_with_indices_panel(label: str, words: list[str]) -> Text:
    line = Text(f"{label}: ", style="bold")
    for i, w in enumerate(words):
        line.append(f"[{i}]", style="dim cyan")
        line.append(f"{w} ")
    return line


def parse_edit_command(cmd: str, n_ref: int, n_hyp: int) -> tuple[int, int | None] | str | None:
    """Parse an edit line. Returns:
        (ref_idx, hyp_idx_or_None) for a successful edit
        a string error message if invalid
        None if the input is empty (caller treats as 'accept')
    """
    cmd = cmd.strip()
    if not cmd:
        return None
    parts = cmd.split()
    if len(parts) != 2:
        return "Format: '<ref_idx> <hyp_idx>' or '<ref_idx> -' for deletion."
    try:
        ref_idx = int(parts[0])
    except ValueError:
        return f"Ref index must be an integer, got '{parts[0]}'."
    if not (0 <= ref_idx < n_ref):
        return f"Ref index {ref_idx} out of range [0, {n_ref - 1}]."
    if parts[1] == "-":
        return ref_idx, None
    try:
        hyp_idx = int(parts[1])
    except ValueError:
        return f"Hyp index must be an integer or '-', got '{parts[1]}'."
    if not (0 <= hyp_idx < n_hyp):
        return f"Hyp index {hyp_idx} out of range [0, {n_hyp - 1}]."
    return ref_idx, hyp_idx


def annotate_sample(progress: str, path: str, ref: list[str], hyp: list[str],
                    alignment: dict[int, int | None], similarities: dict[int, float | None],
                    alpha: float, lambda_: float, use_global_norm: bool) -> str:
    """Returns one of: 'accept', 'back', 'quit'. Mutates `alignment` in place."""
    edited: set[int] = set()
    while True:
        console.clear()
        render_sample(progress, path, ref, hyp, alignment, similarities, edited,
                      alpha, lambda_, use_global_norm)
        console.print(
            "\n[dim]enter[/dim]=accept  "
            "[dim]'<ref> <hyp>'[/dim]=reassign  "
            "[dim]'<ref> -'[/dim]=deletion  "
            "[dim]b[/dim]=back  "
            "[dim]q[/dim]=save+quit"
        )
        cmd = console.input("> ").strip()
        if cmd == "":
            return "accept"
        if cmd == "b":
            return "back"
        if cmd == "q":
            return "quit"
        result = parse_edit_command(cmd, len(ref), len(hyp))
        if isinstance(result, str):
            console.print(f"[red]{result}[/red]")
            console.input("[dim]press enter to continue[/dim]")
            continue
        ref_idx, hyp_idx = result
        if hyp_idx is not None:
            conflicting = [r for r, h in alignment.items() if h == hyp_idx and r != ref_idx]
            for r in conflicting:
                alignment[r] = None
                similarities[r] = None
                edited.add(r)
            if conflicting:
                conflicts_str = ", ".join(f"ref[{r}]" for r in conflicting)
                console.print(
                    f"[yellow]Auto-unassigned {conflicts_str} (was pointing to hyp[{hyp_idx}]); "
                    f"now ε. Reassign if needed.[/yellow]"
                )
                console.input("[dim]press enter to continue[/dim]")
        alignment[ref_idx] = hyp_idx
        similarities[ref_idx] = None  # mark as user-set; no solver similarity
        edited.add(ref_idx)


def load_existing(output_path: Path) -> list[dict]:
    if not output_path.exists():
        return []
    with output_path.open(encoding="utf-8-sig") as f:
        return json.load(f)


def save_atomic(output_path: Path, entries: list[dict]) -> None:
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    tmp.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata_tsv", type=Path, help="Path to the metadata TSV.")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.5)
    parser.add_argument("--use-global-norm", action="store_true",
                        help="Enable global lexical normalization (default: off).")
    parser.add_argument("--path-column", default="path",
                        help="Metadata TSV column holding the audio path (default: path).")
    parser.add_argument("--hyp-column", default="whisper_large_v2_transcript",
                        help="Metadata TSV column to use as the hypothesis (default: whisper_large_v2_transcript).")
    parser.add_argument("--hyp-suffix", default="dit",
                        help="Tag identifying the hypothesis source; appended to the default output filename (default: dit).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON. Default: <metadata_stem>_ground_truth_alignments_<hyp-suffix>.json next to the input.")
    args = parser.parse_args()

    df = pd.read_csv(args.metadata_tsv, sep="\t", encoding="utf-8-sig")
    output_path = args.output or args.metadata_tsv.parent / args.metadata_tsv.name.replace(
        "_metadata.tsv", f"_ground_truth_alignments_{args.hyp_suffix}.json"
    )

    global_max_word_len = None
    if args.use_global_norm:
        global_max_word_len = max(
            len(w)
            for _, row in df.iterrows()
            for w in tokenize(row["sentence"]) + tokenize(row[args.hyp_column])
        )

    existing = load_existing(output_path)
    done_indices = {e["index"] for e in existing}
    console.print(f"[bold]Loaded {len(existing)} existing entries from {output_path.name}.[/bold]")

    total = len(df)
    i = 0
    while i < total:
        if i in done_indices:
            i += 1
            continue
        row = df.iloc[i]
        ref = tokenize(row["sentence"])
        hyp = tokenize(row[args.hyp_column])
        alignment, similarities = solver_prefill(
            ref, hyp, args.alpha, args.lambda_, args.use_global_norm, global_max_word_len
        )

        progress = f"[{len(existing) + 1}/{total}]  index={i}  region={row.get('dialect_region', '?')}"
        action = annotate_sample(
            progress, row[args.path_column], ref, hyp, alignment, similarities,
            args.alpha, args.lambda_, args.use_global_norm,
        )

        if action == "quit":
            console.print(f"[bold]Saved {len(existing)} entries. Bye.[/bold]")
            return
        if action == "back":
            if existing:
                last = existing.pop()
                done_indices.discard(last["index"])
                save_atomic(output_path, existing)
                i = last["index"]
                console.print(f"[yellow]Reverted entry index={last['index']}.[/yellow]")
                console.input("[dim]press enter to continue[/dim]")
            else:
                console.print("[yellow]Nothing to go back to.[/yellow]")
                console.input("[dim]press enter to continue[/dim]")
            continue

        entry = {
            "index": i,
            "source": DATASET_SOURCE,
            "path": row[args.path_column],
            "reference": ref,
            "hypothesis": hyp,
            "alignment": {str(k): v for k, v in alignment.items()},
        }
        existing.append(entry)
        done_indices.add(i)
        save_atomic(output_path, existing)
        i += 1

    console.print(f"[bold green]Done. {len(existing)} entries written to {output_path}.[/bold green]")


if __name__ == "__main__":
    main()