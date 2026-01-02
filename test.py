import os
import argparse
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


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[HybridCRNNModel, Dict]:
    """
    Load model checkpoint, supporting both formats:
    a) torch.save(model.state_dict(), path) -> OrderedDict
    b) torch.save({"model_state_dict": ..., "config": ..., ...}, path) -> dict
    
    Tries to use model.load() first (for format b), falls back to manual loading for format a).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Case 1: Full checkpoint dict with config (from model.save())
    # Try using the model's load method first
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint and "config" in checkpoint:
        try:
            model, additional_info = HybridCRNNModel.load(checkpoint_path, device=device, strict=True)
            return model, additional_info
        except Exception as e:
            warnings.warn(f"Failed to use model.load(), falling back to manual loading: {e}")

    # Case 2: Checkpoint dict with model_state_dict but no config (try to find config JSON)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        config = checkpoint.get("config")
        if config is None:
            # Try to infer from checkpoint path
            config_path = checkpoint_path.replace(".pth", "_config.json").replace(".pt", "_config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, "r") as f:
                    config = json.load(f)
            else:
                # Use defaults matching train.py
                config = {
                    "mel_bins": 128,
                    "egemaps_input_dim": 88,
                    "num_classes": 2,
                }
                warnings.warn(
                    f"No config found in checkpoint or JSON file. Using defaults: {config}. "
                    "This may fail if model architecture differs."
                )

        model = HybridCRNNModel(**config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        additional_info = checkpoint.get("additional_info", {})
        return model, additional_info

    # Case 3: Only state_dict (OrderedDict from torch.save(model.state_dict(), path))
    # We need config to create the model - try to load from JSON or use defaults
    config_path = checkpoint_path.replace(".pth", "_config.json").replace(".pt", "_config.json")
    if os.path.exists(config_path):
        import json
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        # Use defaults matching train.py
        config = {
            "mel_bins": 128,
            "egemaps_input_dim": 88,
            "num_classes": 2,
        }
        warnings.warn(
            f"Checkpoint appears to be state_dict only, and no config JSON found. "
            f"Using defaults: {config}. This may fail if model architecture differs."
        )

    model = HybridCRNNModel(**config)
    model.load_state_dict(checkpoint, strict=True)
    return model, {}


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    num_classes: int = 2,
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    For binary classification:
    - accuracy, precision, recall, F1 (macro = weighted for binary)
    - sensitivity (recall for positive class), specificity
    - ROC-AUC (if probabilities provided)
    
    For multi-class:
    - accuracy, precision/recall/F1 (macro + weighted)
    - per-class metrics
    """
    metrics = {}

    # Accuracy
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    metrics["confusion_matrix"] = cm

    # Precision, Recall, F1
    if num_classes == 2:
        # Binary classification
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        metrics["precision"] = prec
        metrics["recall"] = rec
        metrics["f1"] = f1

        # Sensitivity (TPR) and Specificity (TNR)
        tn, fp, fn, tp = cm.ravel()
        metrics["sensitivity"] = tp / max(1, tp + fn)  # Recall for positive class
        metrics["specificity"] = tn / max(1, tn + fp)  # Recall for negative class
        metrics["tp"] = int(tp)
        metrics["tn"] = int(tn)
        metrics["fp"] = int(fp)
        metrics["fn"] = int(fn)

        # ROC-AUC (if probabilities available)
        if y_probs is not None:
            try:
                # For binary, use probabilities for positive class
                if y_probs.ndim == 2 and y_probs.shape[1] == 2:
                    y_probs_positive = y_probs[:, 1]
                else:
                    y_probs_positive = y_probs
                metrics["roc_auc"] = roc_auc_score(y_true, y_probs_positive)
            except ValueError as e:
                warnings.warn(f"Could not compute ROC-AUC: {e}")
                metrics["roc_auc"] = None
    else:
        # Multi-class classification
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        metrics["precision_macro"] = prec_macro
        metrics["recall_macro"] = rec_macro
        metrics["f1_macro"] = f1_macro
        metrics["precision_weighted"] = prec_weighted
        metrics["recall_weighted"] = rec_weighted
        metrics["f1_weighted"] = f1_weighted

        # Per-class metrics
        prec_per_class, rec_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0, labels=list(range(num_classes))
        )
        for i in range(num_classes):
            metrics[f"precision_class_{i}"] = prec_per_class[i]
            metrics[f"recall_class_{i}"] = rec_per_class[i]
            metrics[f"f1_class_{i}"] = f1_per_class[i]

        # ROC-AUC (if probabilities available and binary-like)
        if y_probs is not None and num_classes == 2:
            try:
                if y_probs.ndim == 2 and y_probs.shape[1] == 2:
                    y_probs_positive = y_probs[:, 1]
                else:
                    y_probs_positive = y_probs
                metrics["roc_auc"] = roc_auc_score(y_true, y_probs_positive)
            except ValueError as e:
                warnings.warn(f"Could not compute ROC-AUC: {e}")
                metrics["roc_auc"] = None

    return metrics


def evaluate_model(
    model: HybridCRNNModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 2,
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Evaluate model on test set and collect predictions.
    
    Returns:
        - metrics: Dictionary of evaluation metrics
        - predictions: List of dicts with per-sample predictions
    """
    model.eval()
    all_logits = []
    all_labels = []
    all_file_ids = []
    all_subject_ids = []
    total_loss = 0.0
    total_n = 0

    with torch.no_grad():
        for batch in loader:
            logmel = batch["logmel"].to(device)
            egemaps = batch["egemaps"].to(device)
            labels = batch["label"].to(device)
            file_ids = batch["file_id"]
            subject_ids = batch["subject_id"]

            # Forward pass
            logits = model(logmel, egemaps)
            loss = criterion(logits, labels)

            # Accumulate
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_file_ids.extend(file_ids)
            all_subject_ids.extend(subject_ids)

            total_loss += loss.item() * labels.size(0)
            total_n += labels.size(0)

    # Concatenate all predictions
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Convert to numpy
    y_true = all_labels.numpy()
    y_pred = all_logits.argmax(dim=1).numpy()
    y_probs = torch.softmax(all_logits, dim=1).numpy()

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_probs, num_classes=num_classes)
    metrics["loss"] = total_loss / max(1, total_n)

    # Collect per-sample predictions
    predictions = []
    for i in range(len(y_true)):
        pred_dict = {
            "sample_id": all_file_ids[i],
            "subject_id": all_subject_ids[i],
            "y_true": int(y_true[i]),
            "y_pred": int(y_pred[i]),
        }
        # Add probability columns
        if num_classes == 2:
            pred_dict["p_positive"] = float(y_probs[i, 1])
            pred_dict["p_negative"] = float(y_probs[i, 0])
        else:
            for c in range(num_classes):
                pred_dict[f"p{c}"] = float(y_probs[i, c])
        predictions.append(pred_dict)

    return metrics, predictions


def print_metrics_summary(metrics: Dict[str, float], num_classes: int = 2):
    """Print a clean summary of evaluation metrics."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(f"\nLoss: {metrics['loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")

    if num_classes == 2:
        print(f"\nBinary Classification Metrics:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  Sensitivity (TPR): {metrics['sensitivity']:.4f}")
        print(f"  Specificity (TNR): {metrics['specificity']:.4f}")

        if metrics.get("roc_auc") is not None:
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

        print(f"\nConfusion Matrix:")
        cm = metrics["confusion_matrix"]
        print(f"                Predicted")
        print(f"              Negative  Positive")
        print(f"  Actual Neg    {cm[0,0]:5d}     {cm[0,1]:5d}")
        print(f"  Actual Pos    {cm[1,0]:5d}     {cm[1,1]:5d}")
        print(f"\n  TP: {metrics['tp']}, TN: {metrics['tn']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    else:
        print(f"\nMulti-Class Classification Metrics:")
        print(f"  Precision (macro):   {metrics['precision_macro']:.4f}")
        print(f"  Recall (macro):      {metrics['recall_macro']:.4f}")
        print(f"  F1-Score (macro):    {metrics['f1_macro']:.4f}")
        print(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
        print(f"  Recall (weighted):    {metrics['recall_weighted']:.4f}")
        print(f"  F1-Score (weighted):  {metrics['f1_weighted']:.4f}")

        if metrics.get("roc_auc") is not None:
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

        print(f"\nConfusion Matrix:")
        cm = metrics["confusion_matrix"]
        print("     ", end="")
        for j in range(num_classes):
            print(f"  C{j}", end="")
        print()
        for i in range(num_classes):
            print(f"  C{i}  ", end="")
            for j in range(num_classes):
                print(f"{cm[i,j]:5d}", end="")
            print()

        print(f"\nPer-Class Metrics:")
        for i in range(num_classes):
            print(f"  Class {i}:")
            print(f"    Precision: {metrics[f'precision_class_{i}']:.4f}")
            print(f"    Recall:    {metrics[f'recall_class_{i}']:.4f}")
            print(f"    F1-Score:  {metrics[f'f1_class_{i}']:.4f}")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on a test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth or .pt file)",
    )

    # Data arguments - support both CSV and folder-based approaches
    parser.add_argument(
        "--split_csv",
        type=str,
        default=None,
        help="Path to test split CSV file (must contain file_id and label columns)",
    )
    parser.add_argument(
        "--chunk_labels_csv",
        type=str,
        default=None,
        help="Alternative: path to chunk_labels CSV (if using folder-based approach)",
    )
    parser.add_argument(
        "--logmel_dir",
        type=str,
        default=None,
        help="Directory containing log-mel spectrogram .npy files",
    )
    parser.add_argument(
        "--egemaps_dir",
        type=str,
        default=None,
        help="Directory containing eGeMAPS feature files (.csv or .npy per subject)",
    )

    # Evaluation arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda, cpu, or cuda:0, cuda:1, etc.)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of data loader workers",
    )

    # Output arguments
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Optional: save per-sample predictions to CSV file",
    )

    args = parser.parse_args()

    # Device setup
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("⚠️  CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(args.device)
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, checkpoint_info = load_checkpoint(args.checkpoint, device)
    model = model.to(device)
    model.eval()

    num_classes = model.num_classes
    print(f"Model loaded: {num_classes} classes")

    if checkpoint_info:
        print(f"Checkpoint info: {list(checkpoint_info.keys())}")

    # Dataset setup
    # Support two modes:
    # 1. --split_csv: test split CSV with file_id and label
    # 2. --chunk_labels_csv + --logmel_dir + --egemaps_dir: full dataset paths
    if args.split_csv:
        if not os.path.exists(args.split_csv):
            raise FileNotFoundError(f"Test split CSV not found: {args.split_csv}")
        if not args.logmel_dir or not args.egemaps_dir:
            raise ValueError(
                "When using --split_csv, you must also provide --logmel_dir and --egemaps_dir"
            )
        test_csv = args.split_csv
        logmel_dir = args.logmel_dir
        egemaps_dir = args.egemaps_dir
    elif args.chunk_labels_csv and args.logmel_dir and args.egemaps_dir:
        test_csv = args.chunk_labels_csv
        logmel_dir = args.logmel_dir
        egemaps_dir = args.egemaps_dir
    else:
        raise ValueError(
            "Either provide --split_csv (with --logmel_dir and --egemaps_dir) "
            "or --chunk_labels_csv with --logmel_dir and --egemaps_dir"
        )

    print(f"\nLoading test dataset from: {test_csv}")
    print(f"  logmel_dir: {logmel_dir}")
    print(f"  egemaps_dir: {egemaps_dir}")

    dataset = ADChunkDataset(
        test_csv,
        logmel_dir,
        egemaps_dir,
        cache_egemaps=True,
    )

    print(f"Test set size: {len(dataset)} samples")

    # Check for class imbalance
    labels = dataset.df["label"].values
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < num_classes:
        missing = set(range(num_classes)) - set(unique_labels)
        warnings.warn(
            f"⚠️  Warning: Some classes are missing in test set. "
            f"Present: {unique_labels.tolist()}, Missing: {list(missing)}"
        )
    print(f"Class distribution: {dict(zip(unique_labels, counts))}")

    # DataLoader
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Important: no shuffling for evaluation
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=pad_collate,
    )

    # Loss function (matching training setup)
    # Use unweighted CrossEntropyLoss for test evaluation
    criterion = nn.CrossEntropyLoss()

    # Evaluate
    print("\nEvaluating model...")
    metrics, predictions = evaluate_model(model, test_loader, criterion, device, num_classes=num_classes)

    # Print summary
    print_metrics_summary(metrics, num_classes=num_classes)

    # Save predictions CSV if requested
    if args.out_csv:
        df_pred = pd.DataFrame(predictions)
        df_pred.to_csv(args.out_csv, index=False)
        print(f"✅ Predictions saved to: {args.out_csv}")

    print("✅ Evaluation complete!")


if __name__ == "__main__":
    main()

