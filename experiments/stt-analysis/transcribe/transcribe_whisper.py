import torch
import pandas as pd
import whisper
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SUBSET_DIR = Path(__file__).resolve().parent.parent
SOURCE_TSV = SUBSET_DIR / "zurich_subset_metadata.tsv"
OUTPUT_DIR = SUBSET_DIR / "stt-transcript-analysis"


def transcribe(model_name, output_tsv):
    """
    Partly AI generated code for file mgmt, prompts based on folder c4c03e6f-50a2-4d24-ae88-caf3032798fa
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"--- Loading Model ('{model_name}') on {device} ---")
    model = whisper.load_model(model_name, device=device)

    if not SOURCE_TSV.exists():
        print(f"Error: {SOURCE_TSV} not found.")
        return

    df = pd.read_csv(SOURCE_TSV, sep="\t", encoding="utf-8-sig")
    transcriptions = []

    print(f"Transcribing {len(df)} files ...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = PROJECT_ROOT / row["path"]
        if audio_path.exists():
            result = model.transcribe(
                str(audio_path),
                language="de",
                fp16=False
            )
            transcriptions.append(result["text"].strip())
        else:
            transcriptions.append("ERROR: FILE NOT FOUND")

    df['transcript'] = transcriptions
    output_path = OUTPUT_DIR / output_tsv
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, sep="\t", index=False, encoding="utf-8-sig")
    print(f"\nSuccess! File saved at: {output_path}")
