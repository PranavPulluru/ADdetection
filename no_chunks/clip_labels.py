"""
Build CLIP-level labels from subject-level labels.

UPDATED (Jan 2026):
- NO CHUNKING: each row corresponds to one full cleaned clip (one file_id).
- We match features by file stem (file_id):
    logmel/<file_id>.npy
    eGeMAPS/<file_id>.npy (preferred) or <file_id>.csv (fallback)

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
from typing import Dict, Optional, Tuple, Any, List
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

_AUDIO_EXT_PATTERN = r"\.(wav|mp3|flac|m4a|aac|ogg)$"


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


# =========================
# CLIP-level eGeMAPS matching
# =========================
def find_clip_egemaps_file(egemaps_dir: Path, file_id: str) -> Tuple[Optional[Path], bool]:
    """
    Find eGeMAPS for this clip file_id.

    Preferred:
      eGeMAPS/<file_id>.npy
    Fallback:
      eGeMAPS/<file_id>.csv
    """
    npy_path = egemaps_dir / f"{file_id}.npy"
    if npy_path.exists():
        return npy_path, False

    csv_path = egemaps_dir / f"{file_id}.csv"
    if csv_path.exists():
        return csv_path, False

    return None, True


def subject_id_from_file_id(file_id: str) -> Optional[str]:
    """
    Convert file_id (stem of feature files) into subject_id.

    Examples of file_id:
      S001
      S001.interviewee_only
      S001.interviewee-only

    We reuse the robust parser by pretending it's an audio filename.
    """
    s = (file_id or "").strip()
    if not s:
        return None
    sid = subject_id_from_audio_filename(s + ".wav")
    return sid if sid else None


def scan_clip_logmel_files(logmel_dir: Path) -> List[Tuple[str, Path]]:
    """
    Scan logmel directory for all .npy FULL-CLIP files.
    Returns list of (file_id, full_path)
    """
    clips: List[Tuple[str, Path]] = []

    if not logmel_dir.exists():
        print(f"Warning: Logmel directory does not exist: {logmel_dir}")
        return clips

    for npy_file in logmel_dir.glob("*.npy"):
        file_id = npy_file.stem
        clips.append((file_id, npy_file))

    return clips


def build_clip_labels(
    labels_map: Dict[str, int],
    clips: List[Tuple[str, Path]],
    egemaps_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Build clip labels by matching:
      - subject_id derived from file_id -> label
      - file_id -> clip-level eGeMAPS path
    """
    clip_data: List[Dict[str, Any]] = []

    for file_id, logmel_path in clips:
        subject_id = subject_id_from_file_id(file_id)
        if subject_id is None:
            print(f"Warning: Could not extract subject_id from '{file_id}', skipping")
            continue

        if subject_id not in labels_map:
            print(f"Warning: Subject '{subject_id}' not found in labels, skipping '{file_id}'")
            continue

        label = labels_map[subject_id]

        egemaps_path, eg_missing = find_clip_egemaps_file(egemaps_dir, file_id)

        clip_data.append({
            "file_id": file_id,
            "subject_id": subject_id,
            "label": label,
            "logmel_path": str(logmel_path.resolve()),
            "egemaps_path": str(egemaps_path.resolve()) if egemaps_path else "",
            "egemaps_missing": bool(eg_missing),
        })

    return clip_data


def write_output_csv(clip_data: List[Dict[str, Any]], output_path: Path) -> None:
    if not clip_data:
        print("Warning: No clip data to write!")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["file_id", "subject_id", "label", "logmel_path", "egemaps_path", "egemaps_missing"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(clip_data)

    print(f"Successfully wrote {len(clip_data)} rows to {output_path}")


def print_summary(total_clips: int, clips_written: int, missing_egemaps: int) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY (CLIP-LEVEL)")
    print("=" * 60)
    print(f"Total logmel clips found:         {total_clips}")
    print(f"Rows written (matched labels):    {clips_written}")
    print(f"Rows missing clip eGeMAPS:        {missing_egemaps}")
    if total_clips > 0:
        print(f"Write rate:                       {(clips_written / total_clips) * 100:.1f}%")
    print("=" * 60)


def main() -> int:
    PROJECT_ROOT = Path(r"C:\Users\adipu\OneDrive\Desktop\ADmci_proj")

    parser = argparse.ArgumentParser(
        description="Build CLIP-level labels from subject-level labels (no chunking)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    default_labels_csv = os.path.join(PROJECT_ROOT, "labels.csv")
    default_logmel_dir = os.path.join(PROJECT_ROOT, "data", "processed", "logmel")
    default_egemaps_dir = os.path.join(PROJECT_ROOT, "data", "processed", "eGeMAPS")

    # ✅ keep same default name if your training expects it,
    # but this is now CLIP labels, not chunk labels.
    default_out_csv = os.path.join(PROJECT_ROOT, "clip_labels.csv")

    parser.add_argument("--labels_csv", type=str, default=default_labels_csv)
    parser.add_argument("--logmel_dir", type=str, default=default_logmel_dir)
    parser.add_argument("--egemaps_dir", type=str, default=default_egemaps_dir)
    parser.add_argument("--out_csv", type=str, default=default_out_csv)

    args = parser.parse_args()

    labels_csv_path = Path(args.labels_csv).expanduser().resolve()
    logmel_dir = Path(args.logmel_dir).expanduser().resolve()
    egemaps_dir = Path(args.egemaps_dir).expanduser().resolve()
    output_path = Path(args.out_csv).expanduser().resolve()

    print("Building CLIP labels (no chunking)...")
    print(f"Labels CSV:  {labels_csv_path}")
    print(f"Logmel dir:  {logmel_dir}")
    print(f"eGeMAPS dir: {egemaps_dir}")
    print(f"Out CSV:     {output_path}\n")

    labels_map = parse_labels_csv(labels_csv_path)
    print(f"Loaded {len(labels_map)} subject labels")

    clips = scan_clip_logmel_files(logmel_dir)
    total_clips = len(clips)
    print(f"Found {total_clips} logmel clip files")

    clip_data = build_clip_labels(labels_map, clips, egemaps_dir)

    missing_eg = sum(1 for r in clip_data if r["egemaps_missing"])
    write_output_csv(clip_data, output_path)
    print_summary(total_clips, len(clip_data), missing_eg)

    if missing_eg > 0:
        print("\n⚠️  Note: Some rows are missing clip-level eGeMAPS.")
        print("   Run full-clip eGeMAPS extraction for those files or set require_egemaps=False in the dataset.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
