import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from gradio_client import Client, handle_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

UPLOAD_URL = "https://stt4sg.fhnw.ch/long_v3/"
STATUS_URL = "https://stt4sg.fhnw.ch/long_v3/status/"


def _extract_uuid(result_text: str) -> str | None:
    """Extract the uuid= value from the upload status markdown."""
    match = re.search(r"uuid=([^\s\)\]]+)", result_text)
    return match.group(1) if match else None


def transcribe(target_type, output_tsv, subset):
    analysis_dir = PROJECT_ROOT / "samples" / subset
    source_tsv = analysis_dir / "subset_metadata.tsv"

    print("--- Connecting to FHNW STT4SG Upload Endpoint ---")
    upload_client = Client(UPLOAD_URL)

    if not source_tsv.exists():
        print(f"Error: {source_tsv} not found.")
        return

    df = pd.read_csv(source_tsv, sep="\t", encoding="utf-8-sig")
    transcriptions = []

    print(f"Transcribing {len(df)} files for {target_type}...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = analysis_dir / row["path"]
        if not audio_path.exists():
            transcriptions.append("ERROR: FILE NOT FOUND")
            continue
        try:
            # Step 1: upload & get UUID
            upload_result = upload_client.predict(
                file_path=handle_file(str(audio_path)),
                api_name="/handle_upload"
            )
            uuid = _extract_uuid(str(upload_result))
            if not uuid:
                print(f"\n  Could not parse UUID from: {upload_result}")
                transcriptions.append("ERROR: NO UUID")
                continue

            # Step 2: connect to status app with UUID, fetch transcription
            status_client = Client(
                STATUS_URL,
                httpx_kwargs={"params": {"uuid": uuid}},
                verbose=False  # suppress "Loaded as API" noise
            )
            result = status_client.predict(api_name="/check_file_status")
            txt_path = result[4].get("value") if isinstance(result[4], dict) else result[4]
            transcription = Path(txt_path).read_text(encoding="utf-8").strip()
            transcriptions.append(transcription)

        except Exception as e:
            print(f"\nError on {audio_path.name}: {e}")
            transcriptions.append("ERROR: TRANSCRIPTION FAILED")

    df[target_type] = transcriptions
    output_path = analysis_dir / "fhnw" / output_tsv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False, encoding="utf-8-sig")
    print(f"\nSuccess! {target_type} file saved at: {output_path}")