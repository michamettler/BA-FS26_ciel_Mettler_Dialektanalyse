# BA-FS26_ciel_Mettler_Dialektanalyse

## Project Setup

### Mac

```bash
# ffmpeg
brew install ffmpeg

# virtuelle Umgebung
python3 -m venv venv
source venv/bin/activate

# dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Windows

```bash
# ffmpeg
winget install ffmpeg

# virtuelle Umgebung
python -m venv venv
venv\Scripts\activate

# dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> **NVIDIA GPU?** Torch mit CUDA statt der CPU-Version:
> ```bash
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```
> Diese Zeile **vor** `pip install -r requirements.txt` ausführen.
>
> Falls `pip install -r` bereits gelaufen ist, torch nachträglich überschreiben:
> ```bash
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
> ```
>
> CUDA-Version prüfen (oben rechts in der Ausgabe):
> ```bash
> nvidia-smi
> ```