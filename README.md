# BA-FS26_ciel_Mettler_Dialektanalyse

## Project Setup

### Mac

```bash
# ffmpeg
brew install ffmpeg

# virtual environment
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

# virtual environment
python -m venv venv
venv\Scripts\activate

# dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> **NVIDIA GPU?** Install torch with CUDA instead of the CPU version:
> ```bash
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```
> Run this line **before** `pip install -r requirements.txt`.
>
> If `pip install -r` has already been run, overwrite torch afterwards:
> ```bash
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
> ```
>
> Check your CUDA version (top right of the output):
> ```bash
> nvidia-smi
> ```
>
> Verify CUDA is detected:
> ```python
> import torch
> print(torch.cuda.is_available())  # should print True
> ```