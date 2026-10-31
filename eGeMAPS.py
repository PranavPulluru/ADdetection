import os
import re
import csv
import numpy as np
from typing import Tuple, List
from datetime import datetime
import torch
import torch.nn.functional as F
import soundfile as sf
import torchaudio
import opensmile

try:
    import audioread
    HAS_AUDIOREAD = True
except Exception:
    HAS_AUDIOREAD = False

# =========================
# CONFIGURATION
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DIARIZED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned")
EGEMAPS_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "eGeMAPS")
LOG_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "logs")

SAMPLE_RATE = 16000
CHUNK_DURATION = 10.0
HOP_DURATION = 3.0

CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
HOP_SAMPLES = int(SAMPLE_RATE * HOP_DURATION)
OVERLAP_RATIO = 1.0 - (HOP_DURATION / CHUNK_DURATION)

os.makedirs(EGEMAPS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

MANIFEST_PATH = os.path.join(LOG_DIR, "chunk_manifest.csv")

# =========================
# HELPERS
# =========================
def parse_subject_id(name: str) -> str:
    m = re.search(r"(S\d{3,})", name, re.IGNORECASE)
    return m.group(1).upper() if m else name.split("_")[0]

def make_chunk_id(subject_id: str, idx: int, start: int) -> str:
    return f"{subject_id}_chunk_{idx:04d}_start_{start:010d}"

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

def segment_audio(waveform: torch.Tensor):
    chunks = []
    start = 0
    total = waveform.numel()
    while start < total:
        end = start + CHUNK_SAMPLES
        chunk = waveform[start:min(end, total)]
        if chunk.numel() < CHUNK_SAMPLES:
            chunk = F.pad(chunk, (0, CHUNK_SAMPLES - chunk.numel()))
        chunks.append((start, chunk))
        start += HOP_SAMPLES
    return chunks

# =========================
# MAIN
# =========================
def process_audio_file(filename: str, smile):
    path = os.path.join(DIARIZED_DIR, filename)
    subject_id = parse_subject_id(os.path.splitext(filename)[0])

    waveform, sr = load_audio_any(path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    waveform = waveform.mean(dim=0)

    chunks = segment_audio(waveform)

    print(f"  Found {len(chunks)} chunks | hop={HOP_DURATION}s | overlap={int(OVERLAP_RATIO*100)}%")

    tmp_dir = os.path.join(LOG_DIR, "_tmp_chunks")
    os.makedirs(tmp_dir, exist_ok=True)

    for i, (start, chunk) in enumerate(chunks):
        chunk_id = make_chunk_id(subject_id, i, start)
        tmp_wav = os.path.join(tmp_dir, f"{chunk_id}.wav")

        sf.write(tmp_wav, chunk.numpy(), SAMPLE_RATE)
        feats = smile.process_file(tmp_wav).values.flatten().astype(np.float32)
        np.save(os.path.join(EGEMAPS_DIR, f"{chunk_id}.npy"), feats)

        with open(MANIFEST_PATH, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([datetime.now().isoformat(), filename, subject_id, chunk_id, "egemaps"])

        os.remove(tmp_wav)

if __name__ == "__main__":
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    print("eGeMAPS extraction | 10s chunks | 3s hop")

    for f in sorted(os.listdir(DIARIZED_DIR)):
        if f.lower().endswith(".wav"):
            process_audio_file(f, smile)
