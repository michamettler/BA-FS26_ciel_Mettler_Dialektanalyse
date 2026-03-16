from transcribe_whisper import transcribe

if __name__ == "__main__":
    transcribe("medium", "DIT", "dialect-ignorant-transcript.tsv", "zurich_subset")
    transcribe("large", "DAT", "dialect-aware-transcript.tsv", "zurich_subset")