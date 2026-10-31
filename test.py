import os
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from dataset import ADChunkDataset, pad_collate
from model import HybridCRNNModel

# =========================================================
# PROJECT ROOT
# =========================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =========================================================
# CONFIGURATION (EDIT THESE)
# =========================================================

# --- Model ---
CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT, "scripts", "models", "best_binary.pth"
)

DEVICE = "cuda"          # "cuda", "cuda:0", or "cpu"
BATCH_SIZE = 32
NUM_WORKERS = 2

# --- Data ---
# Option 1: explicit test split CSV
TEST_SPLIT_CSV = None   # e.g. os.path.join(PROJECT_ROOT, "test_split.csv")

# Option 2: use chunk_labels.csv as test set
CHUNK_LABELS_CSV = os.path.join(PROJECT_ROOT, "chunk_labels.csv")

LOGMEL_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "logmel")
EGEMAPS_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "eGeMAPS")

# --- Output ---
OUT_CSV = os.path.join(PROJECT_ROOT, "scripts", "test_predictions.csv")
SAVE_PREDICTIONS = True


# =========================================================
# CHECKPOINT LOADING
# =========================================================

def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[HybridCRNNModel, Dict]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        config = checkpoint.get("config", {
            "mel_bins": 128,
            "egemaps_input_dim": 88,
            "num_classes": 2,
        })

        model = HybridCRNNModel(**config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return model, checkpoint.get("additional_info", {})

    # state_dict only
    model = HybridCRNNModel(
        mel_bins=128,
        egemaps_input_dim=88,
        num_classes=2,
    )
    model.load_state_dict(checkpoint, strict=True)
    return model, {}


# =========================================================
# METRICS
# =========================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    num_classes: int = 2,
) -> Dict[str, float]:

    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    metrics["confusion_matrix"] = cm

    if num_classes == 2:
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        tn, fp, fn, tp = cm.ravel()

        metrics.update({
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "sensitivity": tp / max(1, tp + fn),
            "specificity": tn / max(1, tn + fp),
        })

        if y_probs is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_probs[:, 1])

    return metrics


# =========================================================
# EVALUATION LOOP
# =========================================================

def evaluate_model(
    model: HybridCRNNModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 2,
):

    model.eval()
    all_logits, all_labels = [], []
    all_file_ids, all_subject_ids = [], []
    total_loss, total_n = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            logmel = batch["logmel"].to(device)
            egemaps = batch["egemaps"].to(device)
            labels = batch["label"].to(device)

            logits = model(logmel, egemaps)
            loss = criterion(logits, labels)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_file_ids.extend(batch["file_id"])
            all_subject_ids.extend(batch["subject_id"])

            total_loss += loss.item() * labels.size(0)
            total_n += labels.size(0)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    probs = torch.softmax(logits, dim=1).numpy()
    preds = logits.argmax(dim=1).numpy()
    y_true = labels.numpy()

    metrics = compute_metrics(y_true, preds, probs, num_classes)
    metrics["loss"] = total_loss / max(1, total_n)

    predictions = []
    for i in range(len(y_true)):
        predictions.append({
            "sample_id": all_file_ids[i],
            "subject_id": all_subject_ids[i],
            "y_true": int(y_true[i]),
            "y_pred": int(preds[i]),
            "p_positive": float(probs[i, 1]),
            "p_negative": float(probs[i, 0]),
        })

    return metrics, predictions


# =========================================================
# MAIN
# =========================================================

def main():

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    model, _ = load_checkpoint(CHECKPOINT_PATH, device)
    model.to(device)

    num_classes = model.num_classes

    # Dataset selection
    if TEST_SPLIT_CSV:
        test_csv = TEST_SPLIT_CSV
    else:
        test_csv = CHUNK_LABELS_CSV

    dataset = ADChunkDataset(
        test_csv,
        LOGMEL_DIR,
        EGEMAPS_DIR,
        cache_egemaps=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        collate_fn=pad_collate,
    )

    criterion = nn.CrossEntropyLoss()

    print("Evaluating...")
    metrics, predictions = evaluate_model(
        model, loader, criterion, device, num_classes
    )

    print("\nResults:")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k}: {v}")

    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])

    if SAVE_PREDICTIONS:
        pd.DataFrame(predictions).to_csv(OUT_CSV, index=False)
        print(f"Predictions saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
