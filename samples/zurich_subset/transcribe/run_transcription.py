from transcribe_whisper import transcribe as transcribe_whisper
from transcribe_fhnw import transcribe as transcribe_fhnw

if __name__ == "__main__":
    transcribe_whisper("small", "whisper-small.tsv")
    #transcribe_whisper("medium", "whisper-medium.tsv")
    #transcribe_whisper("large-v1", "whisper-large-v1.tsv")
    #transcribe_whisper("large-v2", "whisper-large-v2.tsv")
    #transcribe_whisper("large-v3", "whisper-large-v3.tsv")
    #transcribe_fhnw("fhnw.tsv")
