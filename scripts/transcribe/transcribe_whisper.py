import torch
import pandas as pd
import whisper
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def transcribe(model_name, target_type, output_tsv, subset):
    analysis_dir = PROJECT_ROOT / "samples" / subset
    source_tsv = analysis_dir / "subset_metadata.tsv"
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"--- Loading {target_type} Model ('{model_name}') on {device} ---")
    model = whisper.load_model(model_name, device=device)

    if not source_tsv.exists():
        print(f"Error: {source_tsv} not found.")
        return

    df = pd.read_csv(source_tsv, sep="\t", encoding="utf-8-sig")
    transcriptions = []

    print(f"Transcribing {len(df)} files for {target_type}...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = analysis_dir / row["path"]
        if audio_path.exists():
            result = model.transcribe(str(audio_path), language="de", fp16=False)
            transcriptions.append(result["text"].strip())
        else:
            transcriptions.append("ERROR: FILE NOT FOUND")

    df[target_type] = transcriptions
    output_path = analysis_dir / "whisper" / output_tsv
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, sep="\t", index=False, encoding="utf-8-sig")
    print(f"\nSuccess! {target_type} file saved at: {output_path}")



