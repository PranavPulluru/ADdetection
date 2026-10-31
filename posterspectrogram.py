import os
import re
import csv
import torch
import torchaudio
import numpy as np
from typing import List, Tuple
from datetime import datetime
import soundfile as sf

# NEW: for raster image output
import matplotlib
matplotlib.use("Agg")  # ensures it saves to file (no GUI)
import matplotlib.pyplot as plt

try:
    import audioread  # type: ignore
    HAS_AUDIOREAD = True
except Exception:
    HAS_AUDIOREAD = False

# =========================
# CONFIGURATION
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DIARIZED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned")

# NEW: separate image output folder (keeps things tidy)
SPECTROGRAM_IMG_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "logmel_images")
os.makedirs(SPECTROGRAM_IMG_DIR, exist_ok=True)

SAMPLE_RATE = 16000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

SUPPORTED_AUDIO_EXTENSIONS = [
    ".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".mp4", ".m4v"
]

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
)

# =========================
# HELPERS
# =========================
def parse_subject_id(name: str) -> str:
    m = re.search(r"(S\d{3,})", name, re.IGNORECASE)
    return m.group(1).upper() if m else name.split("_")[0]

# =========================
# AUDIO LOADING
# =========================
def load_audio_any(path: str):
    try:
        data, sr = sf.read(path, always_2d=True)
        data = torch.from_numpy(data.T.astype(np.float32))
        return data, sr
    except Exception:
        if not HAS_AUDIOREAD:
            raise
        with audioread.audio_open(path) as f:
            sr = f.samplerate
            audio = np.concatenate([np.frombuffer(b, dtype=np.int16) for b in f]).astype(np.float32) / 32768.0
        return torch.from_numpy(audio).unsqueeze(0), sr

# =========================
# MAIN (PNG spectrogram for FIRST file only)
# =========================
def generate_spectrogram_png_for_file(filename: str):
    path = os.path.join(DIARIZED_DIR, filename)
    subject_id = parse_subject_id(os.path.splitext(filename)[0])

    waveform, sr = load_audio_any(path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    # mono mixdown: [channels, time] -> [time]
    waveform = waveform.mean(dim=0)

    # log-mel: [1, n_mels, time]
    logmel = torch.log10(mel_spectrogram(waveform.unsqueeze(0)) + 1e-10).squeeze(0)  # [n_mels, time]
    logmel_np = logmel.cpu().numpy()

    # Save raster PNG (not vector)
    out_path = os.path.join(SPECTROGRAM_IMG_DIR, f"{subject_id}_logmel.png")

    plt.figure(figsize=(12, 4))
    plt.imshow(logmel_np, origin="lower", aspect="auto")
    plt.title(f"Log-Mel Spectrogram: {filename}")
    plt.xlabel("Time (frames)")
    plt.ylabel("Mel bins")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)  # DPI makes it clearly raster
    plt.close()

    print(f"[OK] Saved spectrogram image: {out_path}")

if __name__ == "__main__":
    print("Log-mel PNG export | FIRST file only")

    files = sorted([
        f for f in os.listdir(DIARIZED_DIR)
        if any(f.lower().endswith(e) for e in SUPPORTED_AUDIO_EXTENSIONS)
    ])

    if not files:
        raise FileNotFoundError(f"No supported audio files found in: {DIARIZED_DIR}")

    first_file = files[0]
    print(f"Processing ONLY: {first_file}")
    generate_spectrogram_png_for_file(first_file)
