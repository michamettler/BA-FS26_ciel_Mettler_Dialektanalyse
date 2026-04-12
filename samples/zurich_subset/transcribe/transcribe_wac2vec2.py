import torch
import torchaudio
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SUBSET_DIR = Path(__file__).resolve().parent.parent
SOURCE_TSV = SUBSET_DIR / "zurich_subset_metadata.tsv"
OUTPUT_DIR = SUBSET_DIR / "stt-transcript-analysis"

MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-german"
TARGET_SAMPLE_RATE = 16_000


def transcribe(output_tsv: str):
    """
    Dialect-ignorant baseline transcription using a Standard German wav2vec2 model.
    Trained on Common Voice DE only — no Swiss German exposure.
    NOT compatible with macos, have to use ZHAW Linux Server.
    """
    if not SOURCE_TSV.exists():
        print(f"Error: {SOURCE_TSV} not found.")
        return

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"--- Loading model: {MODEL_ID} ---")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    model.eval()
    print(f"Model loaded on {device}.")

    df = pd.read_csv(SOURCE_TSV, sep="\t", encoding="utf-8-sig")
    transcriptions = []

    print(f"Transcribing {len(df)} files ...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = PROJECT_ROOT / row["path"]

        if not audio_path.exists():
            transcriptions.append("ERROR: FILE NOT FOUND")
            continue

        try:
            waveform, sample_rate = torchaudio.load(str(audio_path))

            if sample_rate != TARGET_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE
                )
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            audio_array = waveform.squeeze().numpy()

            inputs = processor(
                audio_array,
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )

            input_values = inputs.input_values.to(device, dtype=dtype)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            with torch.no_grad():
                logits = model(input_values, attention_mask=attention_mask).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]
            transcriptions.append(transcription)

        except Exception as e:
            print(f"\nError on {audio_path.name}: {e}")
            transcriptions.append("ERROR: TRANSCRIPTION FAILED")

    df["transcript"] = transcriptions
    output_path = OUTPUT_DIR / output_tsv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False, encoding="utf-8-sig")
    print(f"\nDone! Saved to: {output_path}")