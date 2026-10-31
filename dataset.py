import os
import sys
import re
from typing import Optional, List, Tuple, Dict, Any

# Handle csv module shadowing (similar to csv.py and build_chunk_labels.py)
# Temporarily remove current directory from path to import the real csv module
# This is needed because pandas uses csv.QUOTE_NONNUMERIC internally
_current_dir = os.path.dirname(os.path.abspath(__file__))
_removed_from_path = False
if _current_dir in sys.path:
    sys.path.remove(_current_dir)
    _removed_from_path = True

# Import the real standard library csv module (pandas uses it internally)
import csv  # noqa: F401

import numpy as np
import pandas as pd

# Restore the path AFTER pandas is imported (so pandas uses the stdlib csv module)
if _removed_from_path:
    sys.path.insert(0, _current_dir)

import torch
from torch.utils.data import Dataset


# =========================
# Robust subject-id parsing
# =========================
_CLEANED_SUFFIX_PATTERNS = [
    r"\.interviewee[_-]?only$",   # .interviewee_only or .interviewee-only
    r"\.intervieweeonly$",        # .intervieweeonly
    r"\.interviewee$",            # .interviewee
]
_AUDIO_EXT_PATTERN = r"\.(wav|mp3|flac|m4a|aac)$"


def _subject_id_from_audio_like(name: str) -> str:
    """
    Normalize a filename-ish token to a clean subject_id.

    Examples:
      "S001.wav" -> "S001"
      "S001.interviewee_only.wav" -> "S001"
      "S001.interviewee-only" -> "S001"
      "S154.interviewee_only" -> "S154"
    """
    s = (name or "").strip()
    if not s:
        return ""

    # Remove final audio extension if present
    s = re.sub(_AUDIO_EXT_PATTERN, "", s, flags=re.IGNORECASE)

    # Strip known cleaned suffixes
    for pat in _CLEANED_SUFFIX_PATTERNS:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)

    return s.strip()


def _parse_subject_id(file_id: str) -> str:
    """
    file_id can be:
      - "S001_chunk_0007" -> "S001"
      - "S154.interviewee_only_chunk_0002" -> "S154"
      - fallback: "S001" -> "S001"
    """
    s = str(file_id or "").strip()
    if not s:
        return ""

    if "_chunk_" in s:
        prefix = s.split("_chunk_", 1)[0]  # may include ".interviewee_only"
        # Convert prefix into a clean subject id without renaming anything on disk
        return _subject_id_from_audio_like(prefix + ".wav")  # add extension so parser path is consistent

    # If someone already stored subject_id as file_id
    # Also strip any cleaned suffix that might appear
    return _subject_id_from_audio_like(s + ".wav")


# =========================
# eGeMAPS loaders
# =========================
def _load_egemaps_csv(csv_path: str) -> torch.Tensor:
    df = pd.read_csv(csv_path)
    # Most common: a single row of feature columns
    if len(df) == 1:
        vec = df.iloc[0].to_numpy(dtype=np.float32)
        return torch.from_numpy(vec)
    # If it isn't 1-row, flatten everything (covers odd export formats)
    vec = df.to_numpy(dtype=np.float32).reshape(-1)
    return torch.from_numpy(vec)


def _load_egemaps_npy(npy_path: str) -> torch.Tensor:
    arr = np.load(npy_path).astype(np.float32)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 1:
        raise ValueError(f"eGeMAPS NPY expected (D,) or (1,D), got {arr.shape}: {npy_path}")
    return torch.from_numpy(arr)


def _load_egemaps_chunk(egemaps_dir: str, file_id: str, row: Optional[pd.Series] = None) -> torch.Tensor:
    """
    Chunk-level eGeMAPS loader.

    Priority:
      1) If chunk_labels.csv contains an explicit path column (egemaps_path/egemaps_file/eg_path),
         use it (absolute or relative to egemaps_dir).
      2) Else, try {egemaps_dir}/{file_id}.npy then .csv
      3) Else, raise FileNotFoundError
    """
    # 1) explicit path column support (if present)
    if row is not None:
        for col in ("egemaps_path", "egemaps_file", "eg_path", "egemaps"):
            if col in row.index and pd.notna(row[col]):
                p = str(row[col]).strip()
                if p:
                    # If relative, resolve against egemaps_dir
                    if not os.path.isabs(p):
                        p = os.path.join(egemaps_dir, p)
                    if os.path.exists(p):
                        if p.lower().endswith(".npy"):
                            return _load_egemaps_npy(p)
                        if p.lower().endswith(".csv"):
                            return _load_egemaps_csv(p)
                        # If extension unknown, try best-effort by probing
                        if os.path.exists(p + ".npy"):
                            return _load_egemaps_npy(p + ".npy")
                        if os.path.exists(p + ".csv"):
                            return _load_egemaps_csv(p + ".csv")

    # 2) standard naming: {file_id}.npy/.csv
    tried = [
        os.path.join(egemaps_dir, f"{file_id}.npy"),
        os.path.join(egemaps_dir, f"{file_id}.csv"),
    ]
    for p in tried:
        if os.path.exists(p):
            if p.lower().endswith(".npy"):
                return _load_egemaps_npy(p)
            return _load_egemaps_csv(p)

    raise FileNotFoundError(f"Missing chunk-level eGeMAPS for {file_id}. Tried: {tried}")


def _load_egemaps_subject(egemaps_dir: str, subject_id: str) -> torch.Tensor:
    """
    Loads per-subject eGeMAPS features.
    Prefers CSV, falls back to NPY.

    Supports naming conventions:
      - {subject_id}.csv / {subject_id}.npy
      - {subject_id}.interviewee_only.csv / .npy
      - {subject_id}.interviewee-only.csv / .npy

    Expected:
      - CSV : 1 row, D columns
      - NPY : shape (D,) or (1,D)
    """
    # Candidates in priority order (no renaming, just lookup)
    csv_candidates = [
        os.path.join(egemaps_dir, f"{subject_id}.csv"),
        os.path.join(egemaps_dir, f"{subject_id}.interviewee_only.csv"),
        os.path.join(egemaps_dir, f"{subject_id}.interviewee-only.csv"),
    ]
    npy_candidates = [
        os.path.join(egemaps_dir, f"{subject_id}.npy"),
        os.path.join(egemaps_dir, f"{subject_id}.interviewee_only.npy"),
        os.path.join(egemaps_dir, f"{subject_id}.interviewee-only.npy"),
    ]

    # 1) Prefer CSV if present
    for csv_path in csv_candidates:
        if os.path.exists(csv_path):
            return _load_egemaps_csv(csv_path)

    # 2) Then try NPY
    for npy_path in npy_candidates:
        if os.path.exists(npy_path):
            return _load_egemaps_npy(npy_path)

    # 3) Fallback: any file that starts with "{subject_id}." and ends with .csv or .npy
    try:
        if os.path.isdir(egemaps_dir):
            prefix = f"{subject_id}."
            for name in os.listdir(egemaps_dir):
                if not name.startswith(prefix):
                    continue
                full = os.path.join(egemaps_dir, name)
                if name.lower().endswith(".csv") and os.path.isfile(full):
                    return _load_egemaps_csv(full)
                if name.lower().endswith(".npy") and os.path.isfile(full):
                    return _load_egemaps_npy(full)
    except Exception:
        pass

    raise FileNotFoundError(
        f"Missing eGeMAPS for subject {subject_id}. "
        f"Tried: {csv_candidates + npy_candidates} (and fallback prefix scan)."
    )


class ADChunkDataset(Dataset):
    """
    chunk_labels.csv must contain:
      - file_id (e.g., S001_chunk_0000 OR S001.interviewee_only_chunk_0000)
      - label (0/1)
    Optional:
      - subject_id (recommended). If missing, we parse from file_id.
      - egemaps_path / egemaps_file / eg_path / egemaps (optional explicit per-chunk path)

    Files:
      - logmel_dir/{file_id}.npy        (shape [mel, time])
      - egemaps_dir/{file_id}.npy/.csv  (chunk-level, preferred)
        fallback:
      - egemaps_dir/{subject_id}.csv or {subject_id}.npy   (subject-level)
    """

    def __init__(self, chunk_labels_csv: str, logmel_dir: str, egemaps_dir: str, cache_egemaps: bool = True):
        self.df = pd.read_csv(chunk_labels_csv)
        required = {"file_id", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"chunk_labels.csv missing columns: {missing}")

        self.df["file_id"] = self.df["file_id"].astype(str)
        self.df["label"] = self.df["label"].astype(int)

        # Ensure binary labels
        bad = self.df[~self.df["label"].isin([0, 1])]
        if len(bad) > 0:
            raise ValueError(f"Non-binary labels found. Example:\n{bad.head()}")

        self.logmel_dir = logmel_dir
        self.egemaps_dir = egemaps_dir
        self.cache_egemaps = cache_egemaps

        # Cache is now by *file_id* for chunk-level, and optionally by subject_id for fallback
        self._eg_cache_by_file: Dict[str, torch.Tensor] = {}
        self._eg_cache_by_subject: Dict[str, torch.Tensor] = {}

        # Ensure subject_id exists in df (either provided or derived)
        if "subject_id" not in self.df.columns:
            self.df["subject_id"] = self.df["file_id"].apply(_parse_subject_id)
        else:
            # Normalize it in case it accidentally contains suffixes
            self.df["subject_id"] = (
                self.df["subject_id"].astype(str)
                .apply(lambda s: _subject_id_from_audio_like(s + ".wav"))
            )

        # Optional: sanity check to catch empty subject_ids early
        empties = self.df[self.df["subject_id"].astype(str).str.len() == 0]
        if len(empties) > 0:
            ex = empties.head(5)[["file_id", "subject_id"]]
            raise ValueError(f"Some rows have empty subject_id after parsing. Examples:\n{ex}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        file_id = row["file_id"]
        subject_id = row["subject_id"]
        label = int(row["label"])

        # -------------------------
        # logmel chunk (by file_id)
        # -------------------------
        logmel_path = os.path.join(self.logmel_dir, file_id + ".npy")
        if not os.path.exists(logmel_path):
            raise FileNotFoundError(f"Missing log-mel chunk: {logmel_path}")

        logmel = np.load(logmel_path).astype(np.float32)
        if logmel.ndim != 2:
            raise ValueError(f"log-mel expected [mel,time], got {logmel.shape}: {logmel_path}")
        logmel_t = torch.from_numpy(logmel).unsqueeze(0)  # [1, mel, time]

        # -----------------------------------------
        # eGeMAPS (chunk-level preferred, by file_id)
        # -----------------------------------------
        if self.cache_egemaps and file_id in self._eg_cache_by_file:
            eg_t = self._eg_cache_by_file[file_id]
        else:
            try:
                eg_t = _load_egemaps_chunk(self.egemaps_dir, file_id, row=row)
                if self.cache_egemaps:
                    self._eg_cache_by_file[file_id] = eg_t
            except FileNotFoundError:
                # Fallback to subject-level if chunk-level isn't present
                if self.cache_egemaps and subject_id in self._eg_cache_by_subject:
                    eg_t = self._eg_cache_by_subject[subject_id]
                else:
                    eg_t = _load_egemaps_subject(self.egemaps_dir, subject_id)
                    if self.cache_egemaps:
                        self._eg_cache_by_subject[subject_id] = eg_t

        y = torch.tensor(label, dtype=torch.long)

        return logmel_t, eg_t, y, file_id, subject_id


def pad_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]]) -> Dict[str, Any]:
    """
    Pads log-mel along time dimension to max time in batch.
    Input logmel: [1, mel, time]
    Output logmel: [B, 1, mel, max_time]
    """
    logmels, egemaps, labels, file_ids, subject_ids = zip(*batch)

    # Original (unpadded) time lengths, used for packing/masking in the model.
    lengths = torch.tensor([int(x.shape[-1]) for x in logmels], dtype=torch.long)

    max_t = max(x.shape[-1] for x in logmels)
    padded = []
    for x in logmels:
        t = x.shape[-1]
        if t < max_t:
            x = torch.nn.functional.pad(x, (0, max_t - t))
        padded.append(x)

    return {
        "logmel": torch.stack(padded, dim=0),
        "egemaps": torch.stack(egemaps, dim=0),
        "label": torch.stack(labels, dim=0),
        "lengths": lengths,
        "file_id": list(file_ids),
        "subject_id": list(subject_ids),
    }



print("LOADED dataset.py OK; exports:", [name for name in globals().keys() if "Dataset" in name])
