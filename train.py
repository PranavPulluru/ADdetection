import os
import argparse
import random
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import ADChunkDataset, pad_collate
from model import HybridCRNNModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_by_subject(df, val_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    subjects = df["subject_id"].astype(str).unique().tolist()
    rng = random.Random(seed)
    rng.shuffle(subjects)

    n_val = max(1, int(len(subjects) * val_frac))
    val_subjects = set(subjects[:n_val])

    train_idx, val_idx = [], []
    for i, row in df.iterrows():
        sid = str(row["subject_id"])
        (val_idx if sid in val_subjects else train_idx).append(i)
    return train_idx, val_idx


def compute_class_weights(labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=2).float()
    total = counts.sum().clamp(min=1.0)
    weights = total / (2.0 * counts.clamp(min=1.0))
    return weights.to(device)


def metrics_from_logits(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    p1 = torch.softmax(logits, dim=1)[:, 1]
    print("THRESH=", 0.55, "p1 min/mean/max:", float(p1.min()), float(p1.mean()), float(p1.max()))
    print("p1>=thr:", int((p1 >= 0.85).sum()), "/", p1.numel())

    pred = (p1 >= 0.55).long() #TODO: Change this as necessary
    tp = ((pred == 1) & (y == 1)).sum().item() 
    tn = ((pred == 0) & (y == 0)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = (2 * prec * rec) / max(1e-12, (prec + rec))
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    all_logits, all_y = [], []
    for batch in loader:
        logmel = batch["logmel"].to(device)
        eg = batch["egemaps"].to(device)
        y = batch["label"].to(device)

        logits = model(logmel, eg)
        all_logits.append(logits)
        all_y.append(y)

    logits = torch.cat(all_logits, dim=0)
    y = torch.cat(all_y, dim=0)
    return metrics_from_logits(logits, y)


def train_one_epoch(model, loader, optimizer, criterion, device, amp: bool) -> float:
    model.train()
    total_loss, total_n = 0.0, 0
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    for batch in loader:
        logmel = batch["logmel"].to(device)
        eg = batch["egemaps"].to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(logmel, eg)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * y.size(0)
        total_n += y.size(0)

    return total_loss / max(1, total_n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_labels_csv", type=str, required=True)
    parser.add_argument("--logmel_dir", type=str, required=True)
    parser.add_argument("--egemaps_dir", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--save_path", type=str, default="models/best_binary.pth")
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--no_egemaps_cache", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.use_amp and (device.type == "cuda")

    ds = ADChunkDataset(
        args.chunk_labels_csv,
        args.logmel_dir,
        args.egemaps_dir,
        cache_egemaps=(not args.no_egemaps_cache),
    )

    train_idx, val_idx = split_by_subject(ds.df, args.val_frac, args.seed)
    print(f"✅ Subject-safe split: train chunks={len(train_idx)} | val chunks={len(val_idx)} "
          f"| subjects={ds.df['subject_id'].nunique()}")

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=pad_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=pad_collate,
    )

    model = HybridCRNNModel(mel_bins=128, egemaps_input_dim=88, num_classes=2).to(device)

    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss()
        print("Using unweighted CrossEntropyLoss.")
    else:
        train_labels = torch.tensor([ds.df.iloc[i]["label"] for i in train_idx], dtype=torch.long)
        weights = compute_class_weights(train_labels, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print(f"Using class weights: w0={weights[0].item():.3f}, w1={weights[1].item():.3f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f2 = -1.0
    os.makedirs(os.path.dirname(args.save_path) if os.path.dirname(args.save_path) else ".", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, amp)
        val_metrics = evaluate(model, val_loader, device)

        beta = 2.0
        prec = val_metrics["prec"]
        rec = val_metrics["rec"]
        val_f2 = (1 + beta**2) * (prec * rec) / (beta**2 * prec + rec + 1e-12)

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_acc={val_metrics['acc']:.3f} val_f1={val_metrics['f1']:.3f} val_f2={val_f2:.3f} "
            f"prec={val_metrics['prec']:.3f} rec={val_metrics['rec']:.3f} "
            f"(tp={val_metrics['tp']} fp={val_metrics['fp']} tn={val_metrics['tn']} fn={val_metrics['fn']})"
        )

        if val_f2 > best_f2:
            best_f2 = val_f2
            model.save(
                args.save_path,
                additional_info={
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "train_loss": train_loss,
                    "binary_labels": {"0": "Control", "1": "AD_or_MCI"},
                    "egemaps_level": "subject",
                    "val_f2": val_f2,
                    "beta": beta,
                },
            )
            print(f"💾 Saved new best model (val_f2={best_f2:.3f}) -> {args.save_path}")

    print(f"\nDone. Best val_f2={best_f2:.3f}")


if __name__ == "__main__":
    main()
