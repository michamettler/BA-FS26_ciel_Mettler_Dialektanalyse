#!/usr/bin/env python3
"""
Transcribe the full STT4SG-350 dataset using the FHNW STT4SG Gradio API.

Features:
  - Checkpoint/resume: saves progress to JSON every N files; resumes automatically
  - Concurrent workers: ThreadPoolExecutor for parallel I/O-bound API calls
  - Progress bar with ETA, error count, and completion stats
  - Graceful shutdown on Ctrl+C / SIGTERM — checkpoint is saved before exit
  - Designed for Linux server execution

Usage:
    python transcribe_fhnw_full.py --split test
    python transcribe_fhnw_full.py --split train_balanced
    python transcribe_fhnw_full.py --split test --workers 8 --checkpoint-every 100
    python transcribe_fhnw_full.py --split test --resume   # explicit resume (default)
    python transcribe_fhnw_full.py --split test --restart   # discard checkpoint, start fresh
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

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "STT4SG-350 v2.1"
OUTPUT_DIR = PROJECT_ROOT / "transcriptions" / "fhnw"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# ── FHNW API ─────────────────────────────────────────────────────────────────
UPLOAD_URL = "https://stt4sg.fhnw.ch/long_v3/"
STATUS_URL = "https://stt4sg.fhnw.ch/long_v3/status/"

# ── split configuration ──────────────────────────────────────────────────────
SPLIT_CONFIG = {
    "test": {
        "tsv": "test.tsv",
        "clips_dir": "clips__test",
    },
    "train_balanced": {
        "tsv": "train_balanced.tsv",
        "clips_dir": "clips__train_valid-001",
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


def transcribe_single(audio_path: Path, max_retries: int = 3, delay: float = 0.5) -> str:
    """Transcribe one audio file via the FHNW Gradio API (thread-safe)."""
    for attempt in range(max_retries):
        if _shutdown.is_set():
            return "ERROR: SHUTDOWN"
        time.sleep(delay)
        try:
            upload_client = Client(UPLOAD_URL, verbose=False)
            upload_result = upload_client.predict(
                file_path=handle_file(str(audio_path)),
                api_name="/handle_upload",
            )
            uuid = _extract_uuid(str(upload_result))
            if not uuid:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return "ERROR: NO UUID"

            status_client = Client(
                STATUS_URL,
                httpx_kwargs={"params": {"uuid": uuid}},
                verbose=False,
            )

            # Poll until transcription is ready (max ~5 min per file)
            for poll in range(60):
                if _shutdown.is_set():
                    return "ERROR: SHUTDOWN"
                result = status_client.predict(api_name="/check_file_status")
                txt_path = (
                    result[4].get("value") if isinstance(result[4], dict) else result[4]
                )
                if txt_path is not None:
                    break
                time.sleep(5)
            else:
                return "ERROR: POLL TIMEOUT"

            transcription = Path(txt_path).read_text(encoding="utf-8").strip()
            return transcription

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return f"ERROR: {e}"

    return "ERROR: MAX RETRIES EXCEEDED"


# ── checkpoint I/O ───────────────────────────────────────────────────────────
def _ckpt_path(split: str) -> Path:
    return CHECKPOINT_DIR / f"{split}_checkpoint.json"


def load_checkpoint(split: str) -> dict[str, str]:
    path = _ckpt_path(split)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Checkpoint loaded: {len(data)} files already done.")
        return data
    return {}


def save_checkpoint(split: str, results: dict[str, str]):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _ckpt_path(split)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    tmp.rename(path)  # atomic on POSIX


def delete_checkpoint(split: str):
    path = _ckpt_path(split)
    if path.exists():
        path.unlink()
        print(f"  Deleted old checkpoint for '{split}'.")


# ── final output ─────────────────────────────────────────────────────────────
def save_final(df: pd.DataFrame, completed: dict[str, str], split: str):
    df = df.copy()
    df["transcript"] = df["path"].map(completed).fillna("ERROR: NOT TRANSCRIBED")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"fhnw_{split}.tsv"
    df.to_csv(output_path, sep="\t", index=False, encoding="utf-8-sig")

    errors = int(df["transcript"].str.startswith("ERROR").sum())
    ok = len(df) - errors
    print(f"  Saved: {output_path}")
    print(f"  Total: {len(df)}  |  Success: {ok}  |  Errors: {errors}")


# ── main loop ────────────────────────────────────────────────────────────────
def run(split: str, workers: int, checkpoint_every: int, restart: bool):
    config = SPLIT_CONFIG[split]
    tsv_path = DATASET_DIR / config["tsv"]
    clips_dir = DATASET_DIR / config["clips_dir"]

    if not tsv_path.exists():
        sys.exit(f"Error: {tsv_path} not found.")
    if not clips_dir.exists():
        sys.exit(f"Error: {clips_dir} not found.")

    df = pd.read_csv(tsv_path, sep="\t", encoding="utf-8-sig")
    total = len(df)
    print(f"  Split '{split}': {total:,} files")

    # checkpoint handling
    if restart:
        delete_checkpoint(split)
        completed: dict[str, str] = {}
    else:
        completed = load_checkpoint(split)

    pending = [i for i, row in df.iterrows() if row["path"] not in completed]
    print(f"  Completed: {len(completed):,}  |  Remaining: {len(pending):,}")

    if not pending:
        print("  Nothing to do — all files already transcribed.")
        save_final(df, completed, split)
        return

    lock = threading.Lock()
    new_since_ckpt = 0
    errors = 0

    pbar = tqdm(
        total=len(pending),
        desc=f"{split}",
        unit="file",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    def process(idx: int):
        nonlocal new_since_ckpt, errors
        if _shutdown.is_set():
            return

        row = df.iloc[idx]
        audio_path = clips_dir / row["path"]

        if not audio_path.exists():
            result = "ERROR: FILE NOT FOUND"
        else:
            result = transcribe_single(audio_path)

        with lock:
            completed[row["path"]] = result
            if result.startswith("ERROR"):
                errors += 1
            new_since_ckpt += 1
            pbar.update(1)
            pbar.set_postfix(ok=len(completed) - errors, err=errors, ordered=False)

            if new_since_ckpt >= checkpoint_every:
                save_checkpoint(split, completed)
                new_since_ckpt = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process, idx): idx for idx in pending}
        for future in as_completed(futures):
            if _shutdown.is_set():
                # cancel remaining futures
                for f in futures:
                    f.cancel()
                break
            try:
                future.result()
            except Exception as exc:
                tqdm.write(f"Unexpected worker error: {exc}")

    pbar.close()

    # always save on exit
    save_checkpoint(split, completed)
    print(f"  Checkpoint saved ({len(completed):,} files).")

    if _shutdown.is_set():
        print("  Shutdown requested — partial results saved. Re-run to resume.")
    else:
        save_final(df, completed, split)


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Transcribe STT4SG-350 with the FHNW API (full dataset)",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=list(SPLIT_CONFIG.keys()),
        help="Dataset split to transcribe",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent API workers (default: 4)",
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

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] transcribe_fhnw_full")
    print(f"  workers={args.workers}, checkpoint_every={args.checkpoint_every}")
    run(args.split, args.workers, args.checkpoint_every, args.restart)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Done.")


if __name__ == "__main__":
    main()
