"""
dataset.py (Jan 2026) — FULL-CLIP (no chunking)

Loads one sample per cleaned audio clip, using precomputed:
- log-mel spectrogram:  <file_id>.npy
- eGeMAPS vector:       <file_id>.npy  (preferred) OR <file_id>.csv (fallback)

Expects a CSV (clip-level) with columns:
  file_id,subject_id,label,logmel_path,egemaps_path,egemaps_missing

This replaces the chunk-based dataset. No chunk logic exists here.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Handle csv module shadowing (if a local csv.py exists)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_removed_from_path = False
if _current_dir in sys.path:
    sys.path.remove(_current_dir)
    _removed_from_path = True

import csv  # stdlib

if _removed_from_path:
    sys.path.insert(0, _current_dir)

import numpy as np
import torch
from torch.utils.data import Dataset


# =========================================================
# Helpers
# =========================================================
def _safe_float_array(x: np.ndarray) -> np.ndarray:
    """Ensure float32 contiguous array."""
    x = np.asarray(x)
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return np.ascontiguousarray(x)


def _load_egemaps(path: str) -> np.ndarray:
    """
    Load eGeMAPS from .npy (preferred) or .csv (fallback).
    Returns float32 vector of shape [D].
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"eGeMAPS not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        vec = np.load(path)
        vec = np.asarray(vec).reshape(-1)
        return _safe_float_array(vec)

    if ext == ".csv":
        # Accept either:
        # - a single row of numbers
        # - a header row then a single row
        rows: List[List[str]] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for r in reader:
                if r:
                    rows.append(r)

        if not rows:
            raise ValueError(f"Empty eGeMAPS CSV: {path}")

        # If first row contains any non-numeric tokens, treat as header and use next row.
        def _is_number(s: str) -> bool:
            try:
                float(s)
                return True
            except Exception:
                return False

        data_row = rows[0]
        if not all(_is_number(v) for v in data_row):
            if len(rows) < 2:
                raise ValueError(f"eGeMAPS CSV has header but no data row: {path}")
            data_row = rows[1]

        vec = np.array([float(v) for v in data_row], dtype=np.float32)
        return _safe_float_array(vec)

    raise ValueError(f"Unsupported eGeMAPS extension '{ext}' for file: {path}")


def _load_logmel(path: str) -> np.ndarray:
    """
    Load log-mel spectrogram from .npy.
    Returns float32 array shaped [1, n_mels, T] (adds channel dim if missing).
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"logmel not found: {path}")

    x = np.load(path)
    x = np.asarray(x)

    # Accept common shapes:
    # [n_mels, T]  -> [1, n_mels, T]
    # [1, n_mels, T] -> keep
    if x.ndim == 2:
        x = x[None, ...]
    elif x.ndim == 3:
        pass
    else:
        raise ValueError(f"Unexpected logmel shape {x.shape} from {path}")

    return _safe_float_array(x)


# =========================================================
# Dataset
# =========================================================
class ADClipDataset(Dataset):
    """
    Full-clip dataset: each index corresponds to ONE cleaned clip (one file_id).
    """

    def __init__(
        self,
        clip_labels_csv: str,
        require_egemaps: bool = True,
    ):
        super().__init__()

        self.clip_labels_csv = os.path.abspath(clip_labels_csv)
        self.require_egemaps = bool(require_egemaps)

        if not os.path.exists(self.clip_labels_csv):
            raise FileNotFoundError(f"clip labels CSV not found: {self.clip_labels_csv}")

        self.rows: List[Dict[str, Any]] = []
        with open(self.clip_labels_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {self.clip_labels_csv}")

            needed = {"file_id", "subject_id", "label", "logmel_path", "egemaps_path", "egemaps_missing"}
            missing_cols = needed - set(reader.fieldnames)
            if missing_cols:
                raise ValueError(
                    f"CSV missing required columns {sorted(missing_cols)}. "
                    f"Found: {reader.fieldnames}"
                )

            for row in reader:
                file_id = (row.get("file_id") or "").strip()
                subject_id = (row.get("subject_id") or "").strip()
                label_str = (row.get("label") or "").strip()
                logmel_path = (row.get("logmel_path") or "").strip()
                egemaps_path = (row.get("egemaps_path") or "").strip()
                eg_missing_str = (row.get("egemaps_missing") or "").strip()

                if not file_id or not subject_id or label_str == "" or not logmel_path:
                    continue

                try:
                    label = int(label_str)
                except Exception:
                    continue

                eg_missing = eg_missing_str.lower() in ("1", "true", "yes", "y", "t")

                # If eGeMAPS is required, drop missing rows now
                if self.require_egemaps and (eg_missing or not egemaps_path):
                    continue

                self.rows.append(
                    {
                        "file_id": file_id,
                        "subject_id": subject_id,
                        "label": label,
                        "logmel_path": logmel_path,
                        "egemaps_path": egemaps_path,
                        "egemaps_missing": eg_missing,
                    }
                )

        if len(self.rows) == 0:
            raise RuntimeError(
                "No valid rows loaded. If you're missing eGeMAPS for some clips, "
                "either generate them or pass require_egemaps=False."
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]

        logmel = _load_logmel(r["logmel_path"])
        egemaps = None
        if r["egemaps_path"] and (not r["egemaps_missing"]):
            egemaps = _load_egemaps(r["egemaps_path"])
        elif self.require_egemaps:
            raise FileNotFoundError(f"Missing eGeMAPS for file_id={r['file_id']}")

        # Convert to torch
        logmel_t = torch.from_numpy(logmel)            # [1, n_mels, T]
        egemaps_t = torch.from_numpy(egemaps) if egemaps is not None else torch.zeros(0, dtype=torch.float32)
        label_t = torch.tensor(int(r["label"]), dtype=torch.long)

        return {
            "file_id": r["file_id"],
            "subject_id": r["subject_id"],
            "logmel": logmel_t,
            "egemaps": egemaps_t,
            "label": label_t,
        }


# =========================================================
# Collate (pads full clips to max T in batch)
# =========================================================
def pad_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pads logmel along time dim to max length in batch.
    logmel: [1, n_mels, T]
    """
    file_ids = [b["file_id"] for b in batch]
    subject_ids = [b["subject_id"] for b in batch]

    labels = torch.stack([b["label"] for b in batch], dim=0)  # [B]
    egemaps = torch.stack([b["egemaps"] for b in batch], dim=0)  # [B, D]

    # Pad logmel time dimension
    specs = [b["logmel"] for b in batch]  # each [1, n_mels, T]
    T_max = max(int(s.shape[-1]) for s in specs)

    padded = []
    lengths = []
    for s in specs:
        T = int(s.shape[-1])
        lengths.append(T)
        if T < T_max:
            pad_amt = T_max - T
            s = torch.nn.functional.pad(s, (0, pad_amt), mode="constant", value=0.0)
        padded.append(s)

    logmel = torch.stack(padded, dim=0)  # [B, 1, n_mels, T_max]
    lengths = torch.tensor(lengths, dtype=torch.long)

    return {
        "file_id": file_ids,
        "subject_id": subject_ids,
        "logmel": logmel,
        "lengths": lengths,
        "egemaps": egemaps,
        "label": labels,
    }
