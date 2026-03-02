import pandas as pd
import whisper
from pathlib import Path
from tqdm import tqdm

# PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "02-analysis-position"
SOURCE_TSV = ANALYSIS_DIR / "subset_metadata.tsv"
OUTPUT_TSV = ANALYSIS_DIR / "dialect-ignorant-transcript.tsv"

def run_dit():
    print("--- Loading DIT Model ('tiny') ---")
    # 'tiny' is limited in its ability to process complex dialect structures
    model = whisper.load_model("tiny")

    if not SOURCE_TSV.exists():
        print(f"Error: {SOURCE_TSV} not found.")
        return

    df = pd.read_csv(SOURCE_TSV, sep='\t', encoding='utf-8')
    transcriptions = []

    print(f"Transcribing {len(df)} files for DIT...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = ANALYSIS_DIR / row['path']
        
        if audio_path.exists():
            # Force 'de' (German) as smaller models often guess the wrong language
            result = model.transcribe(str(audio_path), language="de")
            transcriptions.append(result['text'].strip())
        else:
            transcriptions.append("ERROR: FILE NOT FOUND")

    df['DIT'] = transcriptions
    df.to_csv(OUTPUT_TSV, sep='\t', index=False, encoding='utf-8-sig')
    print(f"\nSuccess! DIT file saved at: {OUTPUT_TSV}")

if __name__ == "__main__":
    run_dit()