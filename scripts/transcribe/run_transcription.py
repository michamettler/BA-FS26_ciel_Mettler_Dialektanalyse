from transcribe_whisper import transcribe as transcribe_whisper
from transcribe_fhnw import transcribe as transcribe_fhnw

if __name__ == "__main__":
    # DIT
    #transcribe_whisper("small", "DIT", "dialect-ignorant-transcript-small.tsv", "zurich_subset")
    #transcribe_whisper("medium", "DIT", "dialect-ignorant-transcript-medium.tsv", "zurich_subset")
    #transcribe_whisper("large-v1", "DIT", "dialect-ignorant-transcript-v1.tsv", "zurich_subset")
    
    # DAT
    #transcribe_whisper("large-v2", "DAT", "dialect-aware-transcript-v2.tsv", "zurich_subset")
    #transcribe_whisper("large-v3", "DAT", "dialect-aware-transcript-v3.tsv", "zurich_subset")
    #transcribe_fhnw("DAT", "dialect-aware-transcript-fhnw.tsv", "zurich_subset")
    #transcribe_whisper("large-v1", "DIT", "dialect-ignorant-transcript-v1-prompted.tsv", "zurich_subset")
    transcribe_whisper("large-v2", "DAT", "dialect-ignorant-transcript-v2-prompted.tsv", "zurich_subset")
    transcribe_whisper("large-v3", "DAT", "dialect-ignorant-transcript-v3-prompted.tsv", "zurich_subset")