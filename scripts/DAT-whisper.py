import pandas as pd
import whisper
from pathlib import Path
from tqdm import tqdm

# PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "02-analysis-position"
SOURCE_TSV = ANALYSIS_DIR / "subset_metadata.tsv"
OUTPUT_TSV = ANALYSIS_DIR / "dialect-aware-transcript.tsv"

def run_dat():
    print("--- Loading DAT Model ('base') ---")
    # 'base' (74M parameters) offers a balance for dialect normalization
    model = whisper.load_model("base")

    if not SOURCE_TSV.exists():
        print(f"Error: {SOURCE_TSV} not found.")
        return

    df = pd.read_csv(SOURCE_TSV, sep='\t', encoding='utf-8')
    transcriptions = []

    print(f"Transcribing {len(df)} files for DAT...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Path is relative to the analysis directory
        audio_path = ANALYSIS_DIR / row['path']
        
        if audio_path.exists():
            # Force 'de' (German) to guide the model
            result = model.transcribe(str(audio_path), language="de")
            transcriptions.append(result['text'].strip())
        else:
            transcriptions.append("ERROR: FILE NOT FOUND")

    df['DAT'] = transcriptions
    # Save with utf-8-sig to ensure German umlauts display correctly in Excel
    df.to_csv(OUTPUT_TSV, sep='\t', index=False, encoding='utf-8-sig')
    print(f"\nSuccess! DAT file saved at: {OUTPUT_TSV}")

if __name__ == "__main__":
    run_dat()