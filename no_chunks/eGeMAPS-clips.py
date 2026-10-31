import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List

import numpy as np

SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}


def find_audio_files(in_dir: Path) -> List[Path]:
    files: List[Path] = []
    for p in in_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTS:
            files.append(p)
    files.sort()
    return files


def main() -> int:
    # PROJECT_ROOT = one level above scripts/
    PROJECT_ROOT = Path(r"C:\Users\adipu\OneDrive\Desktop\ADmci_proj")

    parser = argparse.ArgumentParser(description="Generate FULL-CLIP eGeMAPS (no chunking).")
    parser.add_argument("--in_dir", type=str, default=str(PROJECT_ROOT / "data" / "processed" / "cleaned"))
    parser.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "data" / "processed" / "eGeMAPS"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import opensmile  # type: ignore
    except Exception as e:
        print("ERROR: Could not import 'opensmile'.")
        print("Install with:")
        print("  pip install opensmile")
        print("If you're on Windows and it still fails, tell me the exact error and I’ll adapt the script.")
        return 2

    # eGeMAPSv02 Functionals is the standard 88-dim vector (most setups)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    files = find_audio_files(in_dir)
    print("=" * 90)
    print("FULL-CLIP eGeMAPS GENERATION (no chunking)")
    print("=" * 90)
    print(f"Time:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input dir:   {in_dir}")
    print(f"Output dir:  {out_dir}")
    print(f"Files found: {len(files)}")
    print("=" * 90)

    ok = 0
    skipped = 0
    failed = 0

    for i, audio_path in enumerate(files, start=1):
        file_id = audio_path.stem  # keep cleaned suffix in the ID
        out_path = out_dir / f"{file_id}.npy"

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            df = smile.process_file(str(audio_path))  # pandas DataFrame, 1 row
            vec = df.iloc[0].to_numpy(dtype=np.float32, copy=True)  # shape [D]
            vec = np.ascontiguousarray(vec)

            np.save(str(out_path), vec)
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
