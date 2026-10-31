"""
Build chunk-level labels from subject-level labels.

UPDATED (Jan 2026):
- eGeMAPS is now CHUNK-LEVEL, so we match eGeMAPS by chunk basename:
    <chunk_base_name>.npy  (preferred)
  (optional fallback to .csv if you saved those too)

Still robustly handles label CSV filenames like:
  - S001.wav
  - S001.interviewee_only.wav
  - S001.interviewee-only.wav
without renaming any files on disk.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import re

# Handle csv module shadowing
_current_dir = os.path.dirname(os.path.abspath(__file__))
_removed_from_path = False
if _current_dir in sys.path:
    sys.path.remove(_current_dir)
    _removed_from_path = True

import csv  # stdlib

if _removed_from_path:
    sys.path.insert(0, _current_dir)


# =========================
# Robust subject-id parsing
# =========================
_CLEANED_SUFFIX_PATTERNS = [
    r"\.interviewee[_-]?only$",
    r"\.intervieweeonly$",
    r"\.interviewee$",
]

_AUDIO_EXT_PATTERN = r"\.(wav|mp3|flac|m4a|aac)$"


def subject_id_from_audio_filename(filename: str) -> str:
    """
    Extract subject_id from an audio filename robustly.

    Examples:
      S001.wav -> S001
      S001.interviewee_only.wav -> S001
      S001.interviewee-only.wav -> S001
    """
    name = (filename or "").strip()
    if not name:
        return ""

    name_no_ext = re.sub(_AUDIO_EXT_PATTERN, "", name, flags=re.IGNORECASE)

    for pat in _CLEANED_SUFFIX_PATTERNS:
        name_no_ext = re.sub(pat, "", name_no_ext, flags=re.IGNORECASE)

    return name_no_ext.strip()


def parse_labels_csv(labels_csv_path: Path) -> Dict[str, int]:
    """
    Read labels.csv and create mapping: subject_id -> label.
    Expects columns: filename,label
    """
    labels_map: Dict[str, int] = {}

    if not labels_csv_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv_path}")

    with open(labels_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames or "filename" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(
                f"CSV must contain 'filename' and 'label' columns. Found columns: {reader.fieldnames}"
            )

        for row_num, row in enumerate(reader, start=2):
            filename = (row.get("filename") or "").strip()
            label_str = (row.get("label") or "").strip()

            if not filename or not label_str:
                continue

            subject_id = subject_id_from_audio_filename(filename)
            if not subject_id:
                print(f"Warning: Row {row_num}: Could not parse subject_id from '{filename}', skipping")
                continue

            try:
                label = int(label_str)
                if label not in (0, 1):
                    print(f"Warning: Row {row_num}: Label '{label}' not in {{0,1}}, skipping")
                    continue
            except ValueError:
                print(f"Warning: Row {row_num}: Invalid label '{label_str}', skipping")
                continue

            if subject_id in labels_map:
                print(f"Warning: Subject '{subject_id}' appears multiple times, using last occurrence")

            labels_map[subject_id] = label

    return labels_map


def extract_subject_id_from_chunk(chunk_base_name: str) -> Optional[str]:
    """
    Extract subject_id from chunk base name like:
      S001_chunk_0000
      S001.interviewee_only_chunk_0000
      S001.interviewee-only_chunk_0000
    """
    s = (chunk_base_name or "").strip()
    if not s:
        return None

    token = "_chunk_"
    if token not in s:
        return None

    prefix = s.split(token, 1)[0]  # e.g., "S154.interviewee_only"
    subject_id = subject_id_from_audio_filename(prefix + ".wav")
    return subject_id if subject_id else None


# =========================
# NEW: chunk-level eGeMAPS matching
# =========================
def find_chunk_egemaps_file(egemaps_dir: Path, chunk_base_name: str) -> Tuple[Optional[Path], bool]:
    """
    Find eGeMAPS for this chunk basename.

    Expected (preferred):
      eGeMAPS/<chunk_base_name>.npy

    Optional fallback:
      eGeMAPS/<chunk_base_name>.csv
    """
    npy_path = egemaps_dir / f"{chunk_base_name}.npy"
    if npy_path.exists():
        return npy_path, False

    csv_path = egemaps_dir / f"{chunk_base_name}.csv"
    if csv_path.exists():
        return csv_path, False

    return None, True


def scan_chunk_files(logmel_dir: Path) -> list[Tuple[str, Path]]:
    """
    Scan logmel directory for all .npy chunk files.
    Returns list of (chunk_base_name, full_path)
    """
    chunks: list[Tuple[str, Path]] = []

    if not logmel_dir.exists():
        print(f"Warning: Logmel directory does not exist: {logmel_dir}")
        return chunks

    for npy_file in logmel_dir.glob("*.npy"):
        chunks.append((npy_file.stem, npy_file))

    return chunks


def build_chunk_labels(
    labels_map: Dict[str, int],
    chunks: list[Tuple[str, Path]],
    egemaps_dir: Path,
) -> list[Dict[str, Any]]:
    """
    Build chunk labels by matching:
      - subject_id derived from chunk filename -> label
      - chunk basename -> chunk-level eGeMAPS path
    """
    chunk_data: list[Dict[str, Any]] = []

    for chunk_base_name, logmel_path in chunks:
        subject_id = extract_subject_id_from_chunk(chunk_base_name)
        if subject_id is None:
            print(f"Warning: Could not extract subject_id from '{chunk_base_name}', skipping")
            continue

        if subject_id not in labels_map:
            print(f"Warning: Subject '{subject_id}' not found in labels, skipping '{chunk_base_name}'")
            continue

        label = labels_map[subject_id]

        egemaps_path, eg_missing = find_chunk_egemaps_file(egemaps_dir, chunk_base_name)

        chunk_data.append({
            "file_id": chunk_base_name,
            "subject_id": subject_id,
            "label": label,
            "logmel_path": str(logmel_path.resolve()),
            "egemaps_path": str(egemaps_path.resolve()) if egemaps_path else "",
            "egemaps_missing": bool(eg_missing),
        })

    return chunk_data


def write_output_csv(chunk_data: list[Dict[str, Any]], output_path: Path) -> None:
    if not chunk_data:
        print("Warning: No chunk data to write!")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["file_id", "subject_id", "label", "logmel_path", "egemaps_path", "egemaps_missing"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(chunk_data)

    print(f"Successfully wrote {len(chunk_data)} rows to {output_path}")


def print_summary(total_chunks: int, chunks_written: int, missing_egemaps: int) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total logmel chunks found:        {total_chunks}")
    print(f"Rows written (matched labels):    {chunks_written}")
    print(f"Rows missing chunk eGeMAPS:       {missing_egemaps}")
    if total_chunks > 0:
        print(f"Write rate:                       {(chunks_written / total_chunks) * 100:.1f}%")
    print("=" * 60)


def main() -> int:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(
        description="Build chunk-level labels from subject-level labels (chunked eGeMAPS compatible)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    default_labels_csv = os.path.join(PROJECT_ROOT, "labels.csv")
    default_logmel_dir = os.path.join(PROJECT_ROOT, "data", "processed", "logmel")
    default_egemaps_dir = os.path.join(PROJECT_ROOT, "data", "processed", "eGeMAPS")
    default_out_csv = os.path.join(PROJECT_ROOT, "chunk_labels.csv")

    parser.add_argument("--labels_csv", type=str, default=default_labels_csv)
    parser.add_argument("--logmel_dir", type=str, default=default_logmel_dir)
    parser.add_argument("--egemaps_dir", type=str, default=default_egemaps_dir)
    parser.add_argument("--out_csv", type=str, default=default_out_csv)

    args = parser.parse_args()

    labels_csv_path = Path(args.labels_csv).expanduser().resolve()
    logmel_dir = Path(args.logmel_dir).expanduser().resolve()
    egemaps_dir = Path(args.egemaps_dir).expanduser().resolve()
    output_path = Path(args.out_csv).expanduser().resolve()

    print("Building chunk labels (chunk-level eGeMAPS)...")
    print(f"Labels CSV:  {labels_csv_path}")
    print(f"Logmel dir:  {logmel_dir}")
    print(f"eGeMAPS dir: {egemaps_dir}")
    print(f"Out CSV:     {output_path}\n")

    labels_map = parse_labels_csv(labels_csv_path)
    print(f"Loaded {len(labels_map)} subject labels")

    chunks = scan_chunk_files(logmel_dir)
    total_chunks = len(chunks)
    print(f"Found {total_chunks} logmel chunk files")

    chunk_data = build_chunk_labels(labels_map, chunks, egemaps_dir)

    missing_eg = sum(1 for r in chunk_data if r["egemaps_missing"])
    write_output_csv(chunk_data, output_path)
    print_summary(total_chunks, len(chunk_data), missing_eg)

    if missing_eg > 0:
        print("\n⚠️  Note: Some rows are missing chunk-level eGeMAPS.")
        print("   That usually means you haven’t run chunked eGeMAPS extraction on all audio yet,")
        print("   or your output folder/casing doesn’t match (eGeMAPS vs egemaps).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
