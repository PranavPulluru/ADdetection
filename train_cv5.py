import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset  # PyTorch DataLoader utilities (framework QoL)

from dataset import ADChunkDataset, pad_collate
from model import HybridCRNNModel


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ -> project root

CLIP_LABELS_CSV = os.path.join(PROJECT_ROOT, "clip_labels.csv")

LOGMEL_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "logmel")
EGEMAPS_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "eGeMAPS")

MODELS_DIR = os.path.join(PROJECT_ROOT, "scripts", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


N_FOLDS = 5
SEED = 42
SHUFFLE_SUBJECTS = True

EPOCHS = 30
BATCH_SIZE = 8
LR = 5e-4
WEIGHT_DECAY = 0.0
NUM_WORKERS = 2
USE_AMP = True
GRAD_CLIP_NORM = 1.0

GRAD_ACCUM_STEPS = 1

EARLY_STOP_PATIENCE = 8
EARLY_STOP_METRIC = "f2"
MIN_DELTA = 1e-4

ACC_THRESHOLD = 0.75

USE_FOCAL_LOSS = True
FOCAL_ALPHA = 0.5
FOCAL_GAMMA = 4.0

REPORT_THR = 0.50
THRESHOLDS = np.linspace(0.05, 0.95, 181)
BETA_REPORT = 1.5


# Sets all random seeds for reproducibility
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Returns the correct PyTorch autocast context for AMP
def _autocast_ctx(device: torch.device, enabled: bool):
    try:
        return torch.amp.autocast(device_type=device.type, enabled=enabled)  # PyTorch AMP utility
    except Exception:
        return torch.cuda.amp.autocast(enabled=enabled)  # PyTorch AMP utility


# Safe division helper to avoid divide-by-zero
def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


# Computes confusion matrix counts from probabilities and threshold
def _confusion_from_probs(y_true: np.ndarray, y_prob: np.ndarray, thr: float):
    y_pred = (y_prob >= thr).astype(np.int64)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, fp, tn, fn


# Computes standard metrics from confusion matrix values
def _metrics_from_conf(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    sens = _safe_div(tp, tp + fn)
    spec = _safe_div(tn, tn + fp)
    prec = _safe_div(tp, tp + fp)
    return {"acc": acc, "sens": sens, "spec": spec, "prec": prec, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


# Computes F-beta score directly from confusion matrix values
def fbeta_from_conf(tp: int, fp: int, fn: int, beta: float) -> float:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    b2 = beta * beta
    denom = (b2 * prec + rec)
    return (1.0 + b2) * prec * rec / denom if denom > 0 else 0.0


# Finds the best decision threshold for reporting purposes
def best_threshold_report(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray, beta: float):
    best = None
    for thr in thresholds:
        tp, fp, tn, fn = _confusion_from_probs(y_true, y_prob, float(thr))
        m = _metrics_from_conf(tp, fp, tn, fn)
        fb = fbeta_from_conf(tp, fp, fn, beta)
        key = (-m["acc"], fn, fp, -fb)
        if best is None or key < best["key"]:
            best = {"thr": float(thr), "key": key, "acc": m["acc"], "fbeta": fb, "tp": tp, "fp": fp, "tn": tn, "fn": fn}
    return best


# Extracts eGeMAPS tensor from a dataset sample
def _get_egemaps_from_sample(sample) -> torch.Tensor:
    if isinstance(sample, dict):
        return sample["egemaps"]
    if isinstance(sample, (tuple, list)):
        return sample[1]
    raise TypeError(f"Unsupported sample type: {type(sample)}")


# Computes mean and std of eGeMAPS features for normalization
def compute_egemaps_stats(dataset, indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    all_vecs = []
    for idx in indices:
        eg = _get_egemaps_from_sample(dataset[idx]).detach().cpu().numpy()
        all_vecs.append(eg)
    all_vecs = np.stack(all_vecs, axis=0)
    mean = torch.from_numpy(all_vecs.mean(axis=0)).float()
    std = torch.from_numpy(all_vecs.std(axis=0)).float().clamp(min=1e-6)
    return mean, std


# Creates subject-level stratified cross-validation folds
def _subject_stratified_folds(df, n_folds: int, seed: int, shuffle: bool = True) -> List[Tuple[List[str], List[str]]]:
    subj_to_label: Dict[str, int] = {}
    for sid, g in df.groupby("subject_id"):
        labs = g["label"].unique().tolist()
        if len(labs) != 1:
            raise ValueError(f"Subject {sid} has multiple labels: {labs}")
        subj_to_label[str(sid)] = int(labs[0])

    neg = [sid for sid, lab in subj_to_label.items() if lab == 0]
    pos = [sid for sid, lab in subj_to_label.items() if lab == 1]

    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(neg)
        rng.shuffle(pos)

    fold_neg = [[] for _ in range(n_folds)]
    fold_pos = [[] for _ in range(n_folds)]
    for i, sid in enumerate(neg):
        fold_neg[i % n_folds].append(sid)
    for i, sid in enumerate(pos):
        fold_pos[i % n_folds].append(sid)

    folds = []
    for k in range(n_folds):
        val_sids = fold_neg[k] + fold_pos[k]
        train_sids = []
        for j in range(n_folds):
            if j == k:
                continue
            train_sids += (fold_neg[j] + fold_pos[j])
        folds.append((train_sids, val_sids))
    return folds


# Maps subject IDs to dataset row indices
def _indices_for_subjects(df, subject_ids: List[str]) -> List[int]:
    s = set(str(x) for x in subject_ids)
    return [i for i, row in df.iterrows() if str(row["subject_id"]) in s]


# Custom focal loss implementation for classification
class FocalLossCE(nn.Module):
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = torch.log_softmax(logits, dim=1)  # PyTorch loss utilities
        p = torch.exp(logp)

        tgt = target.long().view(-1, 1)
        logpt = logp.gather(1, tgt).squeeze(1)
        pt = p.gather(1, tgt).squeeze(1)

        alpha_t = torch.where(
            target == 1,
            torch.tensor(self.alpha, device=logits.device),
            torch.tensor(1.0 - self.alpha, device=logits.device),
        )

        loss = -alpha_t * ((1.0 - pt) ** self.gamma) * logpt

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# Builds and moves the model to the correct device
def build_model(device: torch.device) -> HybridCRNNModel:
    return HybridCRNNModel(num_classes=2).to(device)  # PyTorch nn.Module usage


# Runs one training epoch and returns average loss
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler,
    amp: bool,
    grad_accum_steps: int = 1,
) -> float:
    model.train()  # PyTorch training mode
    total_loss, total_n = 0.0, 0

    optimizer.zero_grad(set_to_none=True)  # PyTorch optimizer pattern

    for step, batch in enumerate(loader, start=1):
        logmel = batch["logmel"].to(device)
        eg = batch["egemaps"].to(device)
        y = batch["label"].to(device)
        lengths = batch.get("lengths")
        if lengths is not None:
            lengths = lengths.to(device)

        with _autocast_ctx(device, amp):  # PyTorch AMP context
            logits = model(logmel, eg, lengths=lengths)
            loss = criterion(logits, y)
            loss = loss / max(1, grad_accum_steps)

        if amp and device.type == "cuda":
            scaler.scale(loss).backward()  # PyTorch AMP scaling
        else:
            loss.backward()

        bs = int(y.size(0))
        total_loss += float(loss.item()) * bs * max(1, grad_accum_steps)
        total_n += bs

        do_step = (step % grad_accum_steps == 0) or (step == len(loader))
        if do_step:
            if GRAD_CLIP_NORM is not None:
                if amp and device.type == "cuda":
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)  # PyTorch utility

            if amp and device.type == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

    return total_loss / max(1, total_n)


# Evaluates model on validation set and returns loss and probabilities
@torch.no_grad()  # PyTorch inference decorator
def eval_probs_and_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()  # PyTorch eval mode
    total_loss, total_n = 0.0, 0
    all_p1, all_y = [], []

    for batch in loader:
        logmel = batch["logmel"].to(device)
        eg = batch["egemaps"].to(device)
        y = batch["label"].to(device)
        lengths = batch.get("lengths")
        if lengths is not None:
            lengths = lengths.to(device)

        with _autocast_ctx(device, amp):
            logits = model(logmel, eg, lengths=lengths)
            loss = criterion(logits, y)
            p1 = torch.softmax(logits, dim=1)[:, 1]  # PyTorch softmax

        bs = int(y.size(0))
        total_loss += float(loss.item()) * bs
        total_n += bs

        all_p1.append(p1.detach().float().cpu())
        all_y.append(y.detach().cpu())

    avg_loss = total_loss / max(1, total_n)
    y_true = torch.cat(all_y).numpy().astype(np.int64)
    p1 = torch.cat(all_p1).numpy().astype(np.float32)
    return avg_loss, y_true, p1


# Main training entry point for cross-validation
def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch device selection
    amp = USE_AMP and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp) if device.type == "cuda" else None  # PyTorch AMP scaler

    print("=" * 92)
    print("CV5 TRAINING (metric-based early stopping) | subject-level folds, FULL-CLIP training (no chunking)")
    print("=" * 92)
    print(f"Device: {device} | AMP: {amp}")
    print(f"Folds: {N_FOLDS} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR} | WD: {WEIGHT_DECAY}")
    print(f"GradAccum: {GRAD_ACCUM_STEPS} | Loss: {'Focal' if USE_FOCAL_LOSS else 'CrossEntropy'} (alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA})")
    print(f"Early-stop metric: {EARLY_STOP_METRIC} (val_loss/acc/f2)")
    print(f"Labels CSV: {CLIP_LABELS_CSV}")
    print("=" * 92)

    ds = ADChunkDataset(CLIP_LABELS_CSV, LOGMEL_DIR, EGEMAPS_DIR, cache_egemaps=True)

    folds = _subject_stratified_folds(ds.df, n_folds=N_FOLDS, seed=SEED, shuffle=SHUFFLE_SUBJECTS)
    subj_to_label = {str(sid): int(g["label"].iloc[0]) for sid, g in ds.df.groupby("subject_id")}

    fold_summaries = []

    for fold_id, (train_sids, val_sids) in enumerate(folds, start=1):
        train_idx = _indices_for_subjects(ds.df, train_sids)
        val_idx = _indices_for_subjects(ds.df, val_sids)

        train_loader = DataLoader(
            Subset(ds, train_idx),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=(device.type == "cuda"),
            collate_fn=pad_collate,
        )
        val_loader = DataLoader(
            Subset(ds, val_idx),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(device.type == "cuda"),
            collate_fn=pad_collate,
        )

        model = build_model(device)

        eg_mean, eg_std = compute_egemaps_stats(ds, train_idx)
        model.set_egemaps_stats(eg_mean.to(device), eg_std.to(device))

        if USE_FOCAL_LOSS:
            criterion = FocalLossCE(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, reduction="mean")
        else:
            criterion = nn.CrossEntropyLoss()  # PyTorch loss function

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)  # PyTorch optimizer

        best_val_loss = float("inf")
        best_epoch = 0
        bad = 0

        ckpt_path = os.path.join(MODELS_DIR, f"cv5_fullclip_fold{fold_id}_best.pth")

        tr_pos = sum(subj_to_label[sid] == 1 for sid in train_sids)
        tr_neg = sum(subj_to_label[sid] == 0 for sid in train_sids)
        va_pos = sum(subj_to_label[sid] == 1 for sid in val_sids)
        va_neg = sum(subj_to_label[sid] == 0 for sid in val_sids)

        print("\n" + "-" * 92)
        print(
            f"FOLD {fold_id}/{N_FOLDS} | train subj={len(train_sids)} (pos={tr_pos}, neg={tr_neg}) | "
            f"val subj={len(val_sids)} (pos={va_pos}, neg={va_neg})"
        )
        print("-" * 92)

        best_report = {"acc": None, "fn": None, "fp": None, "f2": None}

        for epoch in range(1, EPOCHS + 1):
            tr_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                scaler=scaler,
                amp=amp,
                grad_accum_steps=GRAD_ACCUM_STEPS,
            )

            va_loss, y_true, p1 = eval_probs_and_loss(model, val_loader, criterion, device, amp)

            tp, fp, tn, fn = _confusion_from_probs(y_true, p1, REPORT_THR)
            m = _metrics_from_conf(tp, fp, tn, fn)
            f2_at_thr = fbeta_from_conf(tp, fp, fn, beta=2.0)

            best_thr = best_threshold_report(y_true, p1, THRESHOLDS, beta=BETA_REPORT)

            print(
                f"Fold {fold_id} | Ep {epoch:02d}/{EPOCHS} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f}\n"
                f"  VAL@{REPORT_THR:.2f}: acc={m['acc']:.3f} sens={m['sens']:.3f} spec={m['spec']:.3f} "
                f"(FP={m['fp']} FN={m['fn']}) | F2={f2_at_thr:.3f}\n"
                f"  VAL best thr (report): thr={best_thr['thr']:.2f} acc={best_thr['acc']:.3f} "
                f"F{BETA_REPORT:g}={best_thr['fbeta']:.3f} (FP={best_thr['fp']} FN={best_thr['fn']})"
            )

            if EARLY_STOP_METRIC == "val_loss":
                current = float(va_loss)
                best = float(best_val_loss)
                improved = (best - current) > MIN_DELTA
            elif EARLY_STOP_METRIC == "acc":
                current = float(m["acc"])
                best = float(best_report["acc"]) if best_report["acc"] is not None else -1.0
                improved = (current - best) > MIN_DELTA
            elif EARLY_STOP_METRIC == "f2":
                current = float(f2_at_thr)
                best = float(best_report["f2"]) if best_report["f2"] is not None else -1.0
                improved = (current - best) > MIN_DELTA
            else:
                raise ValueError(f"Unknown EARLY_STOP_METRIC: {EARLY_STOP_METRIC}")

            improved = improved and (float(m["acc"]) >= ACC_THRESHOLD)

            if improved:
                if va_loss < best_val_loss:
                    best_val_loss = float(va_loss)

                best_epoch = epoch
                bad = 0

                model.save(
                    ckpt_path,
                    additional_info={
                        "fold": fold_id,
                        "epoch": epoch,
                        "val_loss": float(va_loss),
                        "val_metrics_at_thr": {"thr": REPORT_THR, **m, "f2": f2_at_thr},
                        "best_thr_report": best_thr,
                        "egemaps_mean": eg_mean.tolist(),
                        "egemaps_std": eg_std.tolist(),
                        "notes": "FULL-CLIP training",
                    },
                )
                best_report = {"acc": m["acc"], "fn": m["fn"], "fp": m["fp"], "f2": f2_at_thr}
                print(f"Saved BEST fold-{fold_id}: {EARLY_STOP_METRIC}={current:.4f} (epoch {best_epoch})")
            else:
                bad += 1
                print(
                    f"No improve ({bad}/{EARLY_STOP_PATIENCE}) | best_epoch={best_epoch} | "
                    f"best_acc={best_report['acc'] if best_report['acc'] is not None else 'NA'} | "
                    f"best_f2={best_report['f2'] if best_report['f2'] is not None else 'NA'} | "
                    f"best_val_loss={best_val_loss:.4f}"
                )

            if bad >= EARLY_STOP_PATIENCE:
                print(f"Early stop fold {fold_id} at epoch {epoch} | best epoch {best_epoch}")
                break

        fold_summaries.append(
            {
                "fold": fold_id,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_acc": best_report["acc"],
                "best_f2": best_report["f2"],
                "best_fn": best_report["fn"],
                "best_fp": best_report["fp"],
                "ckpt": ckpt_path,
            }
        )

    print("\n" + "=" * 92)
    print("CV SUMMARY (metric-based early stopping) — FULL CLIP")
    print("=" * 92)

    losses = np.array([x["best_val_loss"] for x in fold_summaries], dtype=np.float32)
    accs = np.array([x["best_acc"] if x["best_acc"] is not None else np.nan for x in fold_summaries], dtype=np.float32)
    f2s = np.array([x["best_f2"] if x["best_f2"] is not None else np.nan for x in fold_summaries], dtype=np.float32)
    fns = np.array([x["best_fn"] if x["best_fn"] is not None else np.nan for x in fold_summaries], dtype=np.float32)
    fps = np.array([x["best_fp"] if x["best_fp"] is not None else np.nan for x in fold_summaries], dtype=np.float32)

    def _nanstd(a): return float(np.nanstd(a, ddof=1)) if np.isfinite(a).sum() > 1 else 0.0

    for x in fold_summaries:
        print(
            f"Fold {x['fold']}: best_epoch={x['best_epoch']:02d} val_loss={x['best_val_loss']:.4f} "
            f"| acc@{REPORT_THR:.2f}={x['best_acc'] if x['best_acc'] is not None else 'NA'} "
            f"f2@{REPORT_THR:.2f}={x['best_f2'] if x['best_f2'] is not None else 'NA'} "
            f"FN={x['best_fn']} FP={x['best_fp']} | {os.path.basename(x['ckpt'])}"
        )

    print("-" * 92)
    print(f"VAL_LOSS mean={float(np.mean(losses)):.4f} std={float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0:.4f}")
    print(f"ACC@{REPORT_THR:.2f} mean={float(np.nanmean(accs)):.4f} std={_nanstd(accs):.4f}")
    print(f"F2 @{REPORT_THR:.2f} mean={float(np.nanmean(f2s)):.4f} std={_nanstd(f2s):.4f}")
    print(f"FN mean={float(np.nanmean(fns)):.2f} std={_nanstd(fns):.2f}")
    print(f"FP mean={float(np.nanmean(fps)):.2f} std={_nanstd(fps):.2f}")
    print("=" * 92)


if __name__ == "__main__":
    main()
