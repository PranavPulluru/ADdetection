import os
import sys
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
import csv

import numpy as np
import pandas as pd

# Restore the path AFTER pandas is imported (so pandas uses the stdlib csv module)
if _removed_from_path:
    sys.path.insert(0, _current_dir)
import torch
from torch.utils.data import Dataset


def _parse_subject_id(file_id: str) -> str:
    # file_id like "S001_chunk_0007" -> "S001"
    if "_chunk_" in file_id:
        return file_id.split("_chunk_")[0]
    # fallback: if someone already stored subject_id as file_id
    return file_id


def _load_egemaps_subject(egemaps_dir: str, subject_id: str) -> torch.Tensor:
    """
    Loads per-subject eGeMAPS features.
    Prefers CSV, falls back to NPY.

    Expected:
      - {subject_id}.csv : 1 row, 88 columns
      - {subject_id}.npy : shape (88,) or (1,88)
    """
    csv_path = os.path.join(egemaps_dir, subject_id + ".csv")
    npy_path = os.path.join(egemaps_dir, subject_id + ".npy")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if len(df) != 1:
            raise ValueError(f"eGeMAPS CSV expected 1 row, got {len(df)}: {csv_path}")
        vec = df.iloc[0].to_numpy(dtype=np.float32)
        return torch.from_numpy(vec)

    if os.path.exists(npy_path):
        arr = np.load(npy_path).astype(np.float32)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 1:
            raise ValueError(f"eGeMAPS NPY expected (88,) or (1,88), got {arr.shape}: {npy_path}")
        return torch.from_numpy(arr)

    raise FileNotFoundError(f"Missing eGeMAPS for subject {subject_id} (.csv/.npy not found)")


class ADChunkDataset(Dataset):
    """
    chunk_labels.csv must contain:
      - file_id (e.g., S001_chunk_0000)
      - label (0/1)
    Optional:
      - subject_id (recommended). If missing, we parse from file_id.

    Files:
      - logmel_dir/{file_id}.npy        (shape [mel, time])
      - egemaps_dir/{subject_id}.csv or {subject_id}.npy   (per subject)
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
        self._eg_cache: Dict[str, torch.Tensor] = {}

        # Ensure subject_id exists in df (either provided or derived)
        if "subject_id" not in self.df.columns:
            self.df["subject_id"] = self.df["file_id"].apply(_parse_subject_id)
        else:
            self.df["subject_id"] = self.df["subject_id"].astype(str)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        file_id = row["file_id"]
        subject_id = row["subject_id"]
        label = int(row["label"])

        # logmel chunk
        logmel_path = os.path.join(self.logmel_dir, file_id + ".npy")
        if not os.path.exists(logmel_path):
            raise FileNotFoundError(f"Missing log-mel chunk: {logmel_path}")

        logmel = np.load(logmel_path).astype(np.float32)
        if logmel.ndim != 2:
            raise ValueError(f"log-mel expected [mel,time], got {logmel.shape}: {logmel_path}")
        logmel_t = torch.from_numpy(logmel).unsqueeze(0)  # [1, mel, time]

        # subject-level egemaps (cached)
        if self.cache_egemaps and subject_id in self._eg_cache:
            eg_t = self._eg_cache[subject_id]
        else:
            eg_t = _load_egemaps_subject(self.egemaps_dir, subject_id)
            if self.cache_egemaps:
                self._eg_cache[subject_id] = eg_t

        y = torch.tensor(label, dtype=torch.long)

        return logmel_t, eg_t, y, file_id, subject_id


def pad_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]]) -> Dict[str, Any]:
    """
    Pads log-mel along time dimension to max time in batch.
    Input logmel: [1, mel, time]
    Output logmel: [B, 1, mel, max_time]
    """
    logmels, egemaps, labels, file_ids, subject_ids = zip(*batch)

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
        "file_id": list(file_ids),
        "subject_id": list(subject_ids),
    }

print("LOADED dataset.py OK; exports:", [name for name in globals().keys() if "Dataset" in name])

