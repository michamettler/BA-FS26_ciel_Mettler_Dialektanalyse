#!/usr/bin/env python3
"""
Transcribe STT4SG-350 or SDS-200 audio with the FHNW STT4SG Gradio API.

Features:
  - Checkpoint/resume: saves progress to JSON every N files; resumes automatically
  - Concurrent workers: ThreadPoolExecutor for parallel I/O-bound API calls
  - Progress bar with ETA, error count, and completion stats
  - Graceful shutdown on Ctrl+C / SIGTERM — checkpoint is saved before exit
  - Designed for Linux server execution

Usage (STT4SG-350, default corpus, split required):
    python transcribe_fhnw_full.py --split test
    python transcribe_fhnw_full.py --split train_all
    python transcribe_fhnw_full.py --split valid

Usage (SDS-200, single all-in-one mode — split defaults to "full"):
    python transcribe_fhnw_full.py --corpus sds-200

Common flags:
    --workers 4 --checkpoint-every 100
    --resume   # explicit resume (default)
    --restart  # discard checkpoint, start fresh
"""

import re
import sys
import json
import time
import signal
import argparse
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from gradio_client import Client, handle_file

# ── thread-local Gradio clients (reused across files) ───────────────────────
_thread_local = threading.local()

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_ROOT = PROJECT_ROOT / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "transcriptions" / "fhnw"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# ── FHNW API ─────────────────────────────────────────────────────────────────
UPLOAD_URL = "https://stt4sg.fhnw.ch/long_v3/"
STATUS_URL = "https://stt4sg.fhnw.ch/long_v3/status/"

# ── corpus / split configuration ────────────────────────────────────────────
# `path_column` is the column in the source TSV that holds the audio path
# (relative to `clips_dir`). STT4SG uses `path`; SDS-200 uses `clip_path`.
CORPUS_CONFIG = {
    "stt4sg": {
        "dataset_dir": "STT4SG-350 v2.1",
        "path_column": "path",
        "splits": {
            "test": {"tsv": "test.tsv", "clips_dir": "clips__test"},
            "train_balanced": {"tsv": "train_balanced.tsv", "clips_dir": "clips__train_valid-001"},
            "train_all": {"tsv": "train_all.tsv", "clips_dir": "clips__train_valid-001"},
            "valid": {"tsv": "valid.tsv", "clips_dir": "clips__train_valid-001"},
        },
    },
    "sds-200": {
        "dataset_dir": "SDS-200 Corpus",
        "path_column": "clip_path",
        # SDS-200 is transcribed as one batch (mirroring the whisper-large-v2
        # reference, which also covers the full export rather than per split).
        "splits": {
            "full": {"tsv": "export_20211220.tsv", "clips_dir": "export_20211220_clips-001"},
        },
    },
}

# ── global shutdown flag ─────────────────────────────────────────────────────
_shutdown = threading.Event()


def _handle_signal(signum, _frame):
    print(f"\nReceived signal {signum} — shutting down gracefully …")
    _shutdown.set()


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ── helpers ──────────────────────────────────────────────────────────────────
def _extract_uuid(result_text: str) -> str | None:
    match = re.search(r"uuid=([^\s\)\]]+)", result_text)
    return match.group(1) if match else None


PER_FILE_TIMEOUT = 120  # hard timeout per attempt in seconds
POLL_INTERVAL = 5  # seconds between status checks


def _get_upload_client() -> Client:
    """Return a per-thread reusable upload client (avoids repeated handshakes)."""
    client = getattr(_thread_local, "upload_client", None)
    if client is None:
        client = Client(
            UPLOAD_URL, verbose=False, httpx_kwargs={"timeout": 30.0}
        )
        _thread_local.upload_client = client
    return client


def _reset_upload_client():
    """Discard the cached client so the next call creates a fresh one."""
    _thread_local.upload_client = None


def _do_one_attempt(audio_path: Path, upload_client: Client) -> str:
    """Single upload+poll attempt with a wall-clock deadline.
    The upload_client is passed in from the outer worker thread (where it's cached)."""
    deadline = time.monotonic() + PER_FILE_TIMEOUT

    upload_result = upload_client.predict(
        file_path=handle_file(str(audio_path)),
        api_name="/handle_upload",
    )

    uuid = _extract_uuid(str(upload_result))
    if not uuid:
        return "ERROR: NO UUID"

    status_client = Client(
        STATUS_URL,
        httpx_kwargs={"params": {"uuid": uuid}, "timeout": 30.0},
        verbose=False,
    )

    max_polls = int(PER_FILE_TIMEOUT / POLL_INTERVAL)
    for poll in range(max_polls):
        if _shutdown.is_set():
            return "ERROR: SHUTDOWN"
        if time.monotonic() >= deadline:
            return "ERROR: POLL TIMEOUT"
        result = status_client.predict(api_name="/check_file_status")
        txt_path = (
            result[4].get("value") if isinstance(result[4], dict) else result[4]
        )
        if txt_path is not None:
            break
        time.sleep(POLL_INTERVAL)
    else:
        return "ERROR: POLL TIMEOUT"

    return Path(txt_path).read_text(encoding="utf-8").strip()


def transcribe_single(audio_path: Path, max_retries: int = 3) -> str:
    """Transcribe one audio file via the FHNW Gradio API (thread-safe).
    Each attempt is wrapped with a hard timeout to prevent indefinite blocking.
    The upload client is managed on this (outer worker) thread and passed into the
    timeout wrapper so thread-local reuse works across files."""
    for attempt in range(max_retries):
        if _shutdown.is_set():
            return "ERROR: SHUTDOWN"
        if attempt > 0:
            time.sleep(2 ** attempt)
        upload_client = _get_upload_client()
        mini = ThreadPoolExecutor(max_workers=1)
        future = mini.submit(_do_one_attempt, audio_path, upload_client)
        try:
            result = future.result(timeout=PER_FILE_TIMEOUT)
            mini.shutdown(wait=False)
            if result == "ERROR: NO UUID" and attempt < max_retries - 1:
                _reset_upload_client()
                continue
            return result
        except TimeoutError:
            # Don't wait for the hung thread — let it die on its own via
            # httpx timeout (30s) or the poll deadline. Discard the stale client.
            mini.shutdown(wait=False, cancel_futures=True)
            _reset_upload_client()
            if attempt < max_retries - 1:
                continue
            return "ERROR: ATTEMPT TIMEOUT"
        except Exception as e:
            mini.shutdown(wait=False)
            _reset_upload_client()
            if attempt < max_retries - 1:
                continue
            return f"ERROR: {e}"

    return "ERROR: MAX RETRIES EXCEEDED"


# ── checkpoint / output naming ───────────────────────────────────────────────
# STT4SG keeps its legacy filenames so in-flight checkpoints stay valid; every
# other corpus is prefixed by name. When a corpus has only one canonical mode
# the split is dropped from the stem for cleaner filenames.
def _name_stem(corpus: str, split: str) -> str:
    if corpus == "stt4sg":
        return split
    if len(CORPUS_CONFIG[corpus]["splits"]) == 1:
        return corpus
    return f"{corpus}_{split}"


def _ckpt_path(corpus: str, split: str) -> Path:
    return CHECKPOINT_DIR / f"{_name_stem(corpus, split)}_checkpoint.json"


def _output_path(corpus: str, split: str) -> Path:
    return OUTPUT_DIR / f"fhnw_{_name_stem(corpus, split)}.tsv"


# ── checkpoint I/O ───────────────────────────────────────────────────────────
def load_checkpoint(corpus: str, split: str) -> dict[str, str]:
    path = _ckpt_path(corpus, split)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Checkpoint loaded: {len(data)} files already done.")
        return data
    return {}


def save_checkpoint(corpus: str, split: str, results: dict[str, str]):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _ckpt_path(corpus, split)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    tmp.rename(path)  # atomic on POSIX


def delete_checkpoint(corpus: str, split: str):
    path = _ckpt_path(corpus, split)
    if path.exists():
        path.unlink()
        print(f"  Deleted old checkpoint for '{_name_stem(corpus, split)}'.")


# ── final output ─────────────────────────────────────────────────────────────
def save_final(
    df: pd.DataFrame,
    completed: dict[str, str],
    corpus: str,
    split: str,
    path_column: str,
):
    df = df.copy()
    df["fhnw_transcript"] = df[path_column].map(completed).fillna("ERROR: NOT TRANSCRIBED")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _output_path(corpus, split)
    df.to_csv(output_path, sep="\t", index=False, encoding="utf-8-sig")

    errors = int(df["fhnw_transcript"].str.startswith("ERROR").sum())
    ok = len(df) - errors
    print(f"  Saved: {output_path}")
    print(f"  Total: {len(df)}  |  Success: {ok}  |  Errors: {errors}")


# ── main loop ────────────────────────────────────────────────────────────────
def run(corpus: str, split: str, workers: int, checkpoint_every: int, restart: bool):
    corpus_cfg = CORPUS_CONFIG[corpus]
    split_cfg = corpus_cfg["splits"][split]
    path_column = corpus_cfg["path_column"]

    dataset_dir = DATASETS_ROOT / corpus_cfg["dataset_dir"]
    tsv_path = dataset_dir / split_cfg["tsv"]
    clips_dir = dataset_dir / split_cfg["clips_dir"]

    if not tsv_path.exists():
        sys.exit(f"Error: {tsv_path} not found.")
    if not clips_dir.exists():
        sys.exit(f"Error: {clips_dir} not found.")

    df = pd.read_csv(tsv_path, sep="\t", encoding="utf-8-sig")
    total = len(df)
    print(f"  Corpus '{corpus}', split '{split}': {total:,} files")

    # checkpoint handling
    if restart:
        delete_checkpoint(corpus, split)
        completed: dict[str, str] = {}
    else:
        completed = load_checkpoint(corpus, split)

    pending = [i for i, row in df.iterrows() if row[path_column] not in completed]
    print(f"  Completed: {len(completed):,}  |  Remaining: {len(pending):,}")

    if not pending:
        print("  Nothing to do — all files already transcribed.")
        save_final(df, completed, corpus, split, path_column)
        return

    lock = threading.Lock()
    new_since_ckpt = 0
    errors = 0

    pbar = tqdm(
        total=len(pending),
        desc=_name_stem(corpus, split),
        unit="file",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    def process(idx: int) -> tuple[str, str]:
        """Transcribe one file and return (path, result)."""
        row = df.iloc[idx]
        rel_path = row[path_column]
        audio_path = clips_dir / rel_path

        try:
            if not audio_path.exists():
                result = "ERROR: FILE NOT FOUND"
            else:
                result = transcribe_single(audio_path)
        except Exception as e:
            result = f"ERROR: {e}"
            tqdm.write(f"  [WORKER ERROR] {rel_path[:60]}: {e}")

        return rel_path, result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Process in batches to avoid 160k+ futures in memory at once,
        # while keeping workers saturated within each batch.
        batch_size = max(workers * 50, 500)
        for batch_start in range(0, len(pending), batch_size):
            if _shutdown.is_set():
                break
            batch = pending[batch_start:batch_start + batch_size]
            futures = {pool.submit(process, idx): idx for idx in batch}

            for future in as_completed(futures):
                if _shutdown.is_set():
                    for f in futures:
                        f.cancel()
                    break

                try:
                    path, result = future.result()
                except Exception as exc:
                    tqdm.write(f"Unexpected worker error: {exc}")
                    continue

                do_checkpoint = False
                with lock:
                    completed[path] = result
                    if result.startswith("ERROR"):
                        errors += 1
                    new_since_ckpt += 1
                    pbar.update(1)
                    pbar.set_postfix(ok=len(completed) - errors, err=errors, ordered=False)

                    if new_since_ckpt >= checkpoint_every:
                        do_checkpoint = True
                        new_since_ckpt = 0

                if do_checkpoint:
                    with lock:
                        snapshot = dict(completed)
                    save_checkpoint(corpus, split, snapshot)

    pbar.close()

    # always save on exit (no lock needed — pool is shut down)
    save_checkpoint(corpus, split, completed)
    print(f"  Checkpoint saved ({len(completed):,} files).")

    if _shutdown.is_set():
        print("  Shutdown requested — partial results saved. Re-run to resume.")
    else:
        save_final(df, completed, corpus, split, path_column)


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Transcribe STT4SG-350 or SDS-200 with the FHNW API",
    )
    parser.add_argument(
        "--corpus",
        default="stt4sg",
        choices=list(CORPUS_CONFIG.keys()),
        help="Source corpus to transcribe (default: stt4sg)",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split to transcribe (defaults to the only available split when the corpus has just one)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of concurrent API workers (default: 2)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Save checkpoint every N completed files (default: 50)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint (default)",
    )
    group.add_argument(
        "--restart",
        action="store_true",
        help="Discard checkpoint and start fresh",
    )
    args = parser.parse_args()

    valid_splits = list(CORPUS_CONFIG[args.corpus]["splits"].keys())
    if args.split is None:
        if len(valid_splits) == 1:
            args.split = valid_splits[0]
        else:
            parser.error(
                f"--split required for corpus '{args.corpus}'. "
                f"Choose one of: {', '.join(valid_splits)}"
            )
    elif args.split not in valid_splits:
        parser.error(
            f"--split '{args.split}' invalid for corpus '{args.corpus}'. "
            f"Choose one of: {', '.join(valid_splits)}"
        )

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] transcribe_fhnw_full")
    print(f"  corpus={args.corpus}, split={args.split}")
    print(f"  workers={args.workers}, checkpoint_every={args.checkpoint_every}")
    run(args.corpus, args.split, args.workers, args.checkpoint_every, args.restart)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Done.")


if __name__ == "__main__":
    main()
