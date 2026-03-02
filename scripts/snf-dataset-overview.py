import pandas as pd
from pathlib import Path

# 1. SETUP PATHS
# Navigating from /scripts up to root, then into datasets
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STT_DIR = PROJECT_ROOT / "datasets" / "STT4SG-350 v2.1"
TSV_PATH = STT_DIR / "train_balanced.tsv"

# The folders where the actual .wav/.mp3 files live
AUDIO_FOLDERS = [
    STT_DIR / "clips__test",
    STT_DIR / "clips__train_valid-001"
]

def perform_deep_dive():
    print(f"--- Deep Dive: STT4SG-350 ---")
    
    # Load the balanced training set
    df = pd.read_csv(TSV_PATH, sep='\t', encoding='utf-8')
    
    # 2. CALCULATE HOURS PER DIALECT
    # Grouping by dialect_region and summing the 'duration' column
    dialect_stats = df.groupby('dialect_region')['duration'].sum() / 3600
    print("\n[1] Audio Hours per Dialect Region:")
    print(dialect_stats.map(lambda x: f"{x:.2f} hours"))
    print(f"\nTotal Dataset Duration: {(df['duration'].sum() / 3600):.2f} hours")

    # 3. PATH VALIDATION
    # Let's check if the first 100 files exist in your folders
    print("\n[2] Verifying Audio File Connections...")
    sample_size = 100
    found_count = 0
    
    for _, row in df.head(sample_size).iterrows():
        file_name = row['path']
        # Check both potential clip folders
        file_exists = any((folder / file_name).exists() for folder in AUDIO_FOLDERS)
        if file_exists:
            found_count += 1
            
    print(f"Verification Result: {found_count}/{sample_size} sample files located on disk.")
    if found_count == 0:
        print("CRITICAL: No audio files found. Check your folder names (e.g., clips__test vs clips).")

    # 4. GENDER RATIO PER DIALECT
    print("\n[3] Gender Distribution by Region:")
    gender_pivot = pd.crosstab(df['dialect_region'], df['gender'], normalize='index') * 100
    print(gender_pivot.round(1).astype(str) + '%')

if __name__ == "__main__":
    perform_deep_dive()