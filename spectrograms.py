import os
import re
import csv
import torch
import torchaudio
import numpy as np
from typing import List, Tuple
from datetime import datetime
import soundfile as sf

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
LOGMEL_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "logmel")
LOG_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "logs")

SAMPLE_RATE = 16000
CHUNK_DURATION = 10.0
HOP_DURATION = 3.0

CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
HOP_SAMPLES = int(SAMPLE_RATE * HOP_DURATION)

OVERLAP_RATIO = 1.0 - (HOP_DURATION / CHUNK_DURATION)

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

SUPPORTED_AUDIO_EXTENSIONS = [
    ".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".mp4", ".m4v"
]

MANIFEST_PATH = os.path.join(LOG_DIR, "chunk_manifest.csv")
os.makedirs(LOGMEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

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

def make_chunk_id(subject_id: str, idx: int, start: int) -> str:
    return f"{subject_id}_chunk_{idx:04d}_start_{start:010d}"

def append_manifest_row(row: dict):
    exists = os.path.exists(MANIFEST_PATH)
    with open(MANIFEST_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            w.writeheader()
        w.writerow(row)

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
            audio = np.concatenate([
                np.frombuffer(b, dtype=np.int16) for b in f
            ]).astype(np.float32) / 32768.0
        return torch.from_numpy(audio).unsqueeze(0), sr

# =========================
# CHUNKING
# =========================
def segment_audio(waveform: torch.Tensor):
    total = waveform.numel()
    chunks = []
    start = 0
    while start < total:
        end = start + CHUNK_SAMPLES
        chunk = waveform[start:min(end, total)]
        if chunk.numel() < CHUNK_SAMPLES:
            chunk = torch.nn.functional.pad(chunk, (0, CHUNK_SAMPLES - chunk.numel()))
        chunks.append((start, chunk))
        start += HOP_SAMPLES
    return chunks

# =========================
# MAIN
# =========================
def process_audio_file(filename: str):
    path = os.path.join(DIARIZED_DIR, filename)
    subject_id = parse_subject_id(os.path.splitext(filename)[0])

    waveform, sr = load_audio_any(path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    waveform = waveform.mean(dim=0)

    chunks = segment_audio(waveform)

    print(f"  Found {len(chunks)} chunks | hop={HOP_DURATION}s | overlap={int(OVERLAP_RATIO*100)}%")

    for i, (start, chunk) in enumerate(chunks):
        chunk_id = make_chunk_id(subject_id, i, start)
        logmel = torch.log10(mel_spectrogram(chunk.unsqueeze(0)) + 1e-10)
        out = os.path.join(LOGMEL_DIR, f"{chunk_id}.npy")
        np.save(out, logmel.squeeze(0).numpy())

        append_manifest_row({
            "timestamp": datetime.now().isoformat(),
            "source_filename": filename,
            "subject_id": subject_id,
            "chunk_id": chunk_id,
            "chunk_idx": i,
            "start_sample": start,
            "modality": "logmel",
            "output_path": out,
            "status": "ok",
            "error": "",
        })

if __name__ == "__main__":
    print("Log-mel extraction | 10s chunks | 3s hop")
    for f in sorted(os.listdir(DIARIZED_DIR)):
        if any(f.lower().endswith(e) for e in SUPPORTED_AUDIO_EXTENSIONS):
            process_audio_file(f)
