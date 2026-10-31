import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
import torchaudio

# ✅ soundfile handles decoding for wav/flac without torchcodec
try:
    import soundfile as sf
except Exception as e:
    sf = None


# NOTE: soundfile reliably supports wav/flac. (mp3/m4a often won't.)
SUPPORTED_AUDIO_EXTS = {".wav", ".flac"}


def find_audio_files(in_dir: Path) -> List[Path]:
    files: List[Path] = []
    for p in in_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTS:
            files.append(p)
    files.sort()
    return files


def load_audio_mono_resample(path: Path, target_sr: int) -> Tuple[torch.Tensor, int]:
    """
    Returns waveform [1, N] float32 and sample_rate.
    Uses soundfile for decoding to avoid torchcodec.
    """
    if sf is None:
        raise ImportError(
            "Missing dependency: soundfile\n"
            "Install it with: pip install soundfile\n"
            "Then re-run this script."
        )

    wav_np, sr = sf.read(str(path), always_2d=True)  # shape [N, C]
    # Convert to mono by averaging channels
    wav_np = wav_np.mean(axis=1)  # [N]

    wav = torch.from_numpy(wav_np).to(torch.float32).unsqueeze(0)  # [1, N]

    # Resample if needed (torchaudio.functional.resample doesn't require torchcodec)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
        sr = target_sr

    # Safety: remove NaNs/Infs
    wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)

    return wav, sr


def compute_logmel(
    wav: torch.Tensor,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """
    Output shape: [n_mels, T] (float32)
    """
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=fmin,
        f_max=fmax,
        power=2.0,
        center=True,
        pad_mode="reflect",
    )(wav)  # [1, n_mels, T]

    mel_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)(mel)  # [1, n_mels, T]
    mel_db = mel_db.squeeze(0)  # [n_mels, T]

    arr = mel_db.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.ascontiguousarray(arr)


def main() -> int:
    PROJECT_ROOT = Path(r"C:\Users\adipu\OneDrive\Desktop\ADmci_proj")

    parser = argparse.ArgumentParser(description="Generate FULL-CLIP log-mel spectrograms (no chunking, no torchcodec).")
    parser.add_argument("--in_dir", type=str, default=str(PROJECT_ROOT / "data" / "processed" / "cleaned"))
    parser.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "data" / "processed" / "logmel"))

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--fmin", type=float, default=0.0)
    parser.add_argument("--fmax", type=float, default=8000.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    files = find_audio_files(in_dir)
    print("=" * 90)
    print("FULL-CLIP LOGMEL GENERATION (no chunking, no torchcodec)")
    print("=" * 90)
    print(f"Time:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input dir:   {in_dir}")
    print(f"Output dir:  {out_dir}")
    print(f"Files found: {len(files)}")
    print(f"Device:      {device}")
    print(
        f"SR={args.sample_rate} n_fft={args.n_fft} hop={args.hop_length} "
        f"n_mels={args.n_mels} fmin={args.fmin} fmax={args.fmax}"
    )
    print("=" * 90)

    ok = 0
    skipped = 0
    failed = 0

    for i, audio_path in enumerate(files, start=1):
        file_id = audio_path.stem
        out_path = out_dir / f"{file_id}.npy"

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            wav, sr = load_audio_mono_resample(audio_path, args.sample_rate)
            wav = wav.to(device)

            logmel = compute_logmel(
                wav=wav,
                sr=sr,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                n_mels=args.n_mels,
                fmin=args.fmin,
                fmax=args.fmax,
            )

            np.save(str(out_path), logmel)
            ok += 1

        except Exception as e:
            failed += 1
            print(f"[FAIL] {audio_path.name}: {e}")

        if i % 50 == 0 or i == len(files):
            print(f"Progress {i}/{len(files)} | ok={ok} skipped={skipped} failed={failed}")

    print("\nDone.")
    print(f"ok={ok} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
