import pandas as pd
import shutil
from pathlib import Path

# 1. SETUP PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STT_DIR = PROJECT_ROOT / "datasets" / "STT4SG-350 v2.1"
SOURCE_TSV = STT_DIR / "train_balanced.tsv"

# Target directory
TARGET_DIR = PROJECT_ROOT / "02-analysis-position"
TARGET_DIR.mkdir(exist_ok=True)

# Audio source folders
AUDIO_SOURCES = [
    STT_DIR / "clips__test",
    STT_DIR / "clips__train_valid-001"
]

def run_extraction(count=100, region="Zürich"):
    print(f"--- Starting Extraction: {region} ---")
    
    if not SOURCE_TSV.exists():
        print(f"Error: Could not find {SOURCE_TSV}")
        return

    df = pd.read_csv(SOURCE_TSV, sep='\t', encoding='utf-8')
    
    # Filter for region and take a random sample
    zh_df = df[df['dialect_region'] == region].sample(n=count, random_state=42)
    
    successful_samples = []

    print(f"Copying files to {TARGET_DIR}...")
    
    for _, row in zh_df.iterrows():
        file_name = row['path']
        file_found = False
        
        for source_folder in AUDIO_SOURCES:
            source_file = source_folder / file_name
            
            if source_file.exists():
                # Define exactly where the file should go
                target_file = TARGET_DIR / file_name
                
                # FIX: Create the subfolder (e.g., '300bb931...') if it doesn't exist
                target_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy the file
                shutil.copy2(source_file, target_file)
                
                successful_samples.append(row)
                file_found = True
                break
        
        if not file_found:
            print(f"Warning: Audio file {file_name} not found.")

    # 2. CREATE THE SUBSET TSV
    output_df = pd.DataFrame(successful_samples)
    output_tsv = TARGET_DIR / "subset_metadata.tsv"
    output_df.to_csv(output_tsv, sep='\t', index=False, encoding='utf-8')

    print(f"\nDONE!")
    print(f"- {len(successful_samples)} audio files copied successfully.")
    print(f"- Metadata saved to: {output_tsv}")

if __name__ == "__main__":
    run_extraction(100, "Zürich")