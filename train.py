import os
import random
from typing import List, Tuple, Dict
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import ADChunkDataset, pad_collate
from model import SimplifiedHybridModel  # alias -> HybridCRNNModel in your new model.py

# =========================================================
# CONFIGURATION
# =========================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ -> project root

CHUNK_LABELS_CSV = os.path.join(PROJECT_ROOT, "chunk_labels.csv")
LOGMEL_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "logmel")
EGEMAPS_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "eGeMAPS")

MODELS_DIR = os.path.join(PROJECT_ROOT, "scripts", "models")
os.makedirs(MODELS_DIR, exist_ok=True)
SAVE_PATH = os.path.join(MODELS_DIR, "best_hybrid_crnn_block64_bilstm1_mlp2_accfirst.pth")

# Training
EPOCHS = 30
BATCH_SIZE = 16
LR = 5e-4
WEIGHT_DECAY = 5e-3
SEED = 42

NUM_WORKERS = 2
USE_AMP = True  # will auto-disable on CPU

# validation split
VAL_N_SUBJECTS = 40

# Subject aggregation for reporting
AGG_MODE = "topk_mean"    # mean / max / topk_mean
TOPK = 2
THRESHOLDS = np.linspace(0.05, 0.95, 181)

# Still compute / print F-beta (bias toward recall), but we SELECT by ACCURACY.
BETA = 1.5

# Early stopping: based on val subject-level ACCURACY first (then FN, then FP, then F-beta)
EARLY_STOP_PATIENCE = 6
MIN_DELTA_ACC = 1e-4

# SpecAugment (mild) -- applied to log-mel input
USE_SPEC_AUGMENT = True
SPEC_AUG_FREQ_MASK = 10
SPEC_AUG_TIME_MASK = 20
SPEC_AUG_FREQ_MASKS = 1
SPEC_AUG_TIME_MASKS = 2

# Loss config
USE_CLASS_WEIGHTS = True


# =========================================================
# Utilities
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def spec_augment(spec: torch.Tensor, training: bool = True) -> torch.Tensor:
    if not training or not USE_SPEC_AUGMENT:
        return spec

    B, C, F, T = spec.shape
    spec = spec.clone()

    # clamp masks to spec shape
    freq_max = min(SPEC_AUG_FREQ_MASK, max(1, F // 4))
    time_max = min(SPEC_AUG_TIME_MASK, max(1, T // 4))

    for b in range(B):
        for _ in range(SPEC_AUG_FREQ_MASKS):
            f = random.randint(0, freq_max)
            f0 = random.randint(0, max(0, F - f))
            spec[b, :, f0:f0 + f, :] = 0

        for _ in range(SPEC_AUG_TIME_MASKS):
            t = random.randint(0, time_max)
            t0 = random.randint(0, max(0, T - t))
            spec[b, :, :, t0:t0 + t] = 0

    return spec


def split_by_subject_fixed_n_stratified(df, seed: int, val_n_subjects: int) -> Tuple[List[int], List[int], List[str]]:
    rng = random.Random(seed)

    subj_to_label: Dict[str, int] = {}
    for sid, g in df.groupby("subject_id"):
        labels = g["label"].unique().tolist()
        if len(labels) > 1:
            raise ValueError(f"Subject {sid} has multiple labels: {labels}")
        subj_to_label[str(sid)] = int(labels[0])

    neg_subjs = [sid for sid, lab in subj_to_label.items() if lab == 0]
    pos_subjs = [sid for sid, lab in subj_to_label.items() if lab == 1]
    rng.shuffle(neg_subjs)
    rng.shuffle(pos_subjs)

    all_subjects = list(subj_to_label.keys())
    n_val = max(1, min(val_n_subjects, len(all_subjects) - 1))

    val_subjects: List[str] = []
    if len(neg_subjs) > 0 and len(pos_subjs) > 0 and n_val >= 2:
        val_subjects.append(neg_subjs.pop())
        val_subjects.append(pos_subjs.pop())

    remaining_pool = neg_subjs + pos_subjs
    rng.shuffle(remaining_pool)
    for sid in remaining_pool:
        if len(val_subjects) >= n_val:
            break
        val_subjects.append(sid)

    val_set = set(val_subjects)
    train_idx = [i for i, row in df.iterrows() if str(row["subject_id"]) not in val_set]
    val_idx = [i for i, row in df.iterrows() if str(row["subject_id"]) in val_set]
    return train_idx, val_idx, val_subjects


def compute_class_weights(labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=2).float()
    total = counts.sum().clamp(min=1.0)
    weights = total / (2.0 * counts.clamp(min=1.0))
    return weights.to(device)


def _get_egemaps_from_sample(sample) -> torch.Tensor:
    """
    Supports dataset items that are either:
      - dict with key 'egemaps'
      - tuple/list where egemaps is 2nd element: (logmel, egemaps, label, ...)
    """
    if isinstance(sample, dict):
        return sample["egemaps"]
    if isinstance(sample, (tuple, list)):
        if len(sample) < 2:
            raise ValueError(f"Sample tuple too short to contain egemaps: len={len(sample)}")
        return sample[1]
    raise TypeError(f"Unsupported sample type: {type(sample)}")


def compute_egemaps_stats(dataset, indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    print("\n📊 Computing eGeMAPS normalization statistics on TRAIN chunks...")
    all_vecs = []
    for idx in indices:
        sample = dataset[idx]
        eg = _get_egemaps_from_sample(sample)
        all_vecs.append(eg.detach().cpu().numpy())
    all_vecs = np.stack(all_vecs, axis=0)
    mean = torch.from_numpy(all_vecs.mean(axis=0)).float()
    std = torch.from_numpy(all_vecs.std(axis=0)).float()
    print(f"   mean range: [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"   std  range: [{std.min():.3f}, {std.max():.3f}]")
    return mean, std


# =========================================================
# Metrics
# =========================================================
def _confusion_from_probs(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(np.int64)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, fp, tn, fn


def _safe_div(a, b):
    return float(a) / float(b) if b != 0 else 0.0


def _metrics_from_confusion(tp, fp, tn, fn):
    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    prec1 = _safe_div(tp, tp + fp)
    rec1 = _safe_div(tp, tp + fn)
    prec0 = _safe_div(tn, tn + fn)
    rec0 = _safe_div(tn, tn + fp)
    return {
        "acc": acc,
        "prec": prec1,
        "rec": rec1,
        "sensitivity": rec1,
        "specificity": rec0,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def fbeta_score_from_confusion(tp: int, fp: int, tn: int, fn: int, beta: float) -> float:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    b2 = beta * beta
    denom = (b2 * prec + rec)
    return (1.0 + b2) * prec * rec / denom if denom > 0 else 0.0


def _aggregate_subject_level(subject_ids, y_true, y_prob, reducer="mean", topk=2):
    subj_probs = defaultdict(list)
    subj_true = {}

    for sid, yt, yp in zip(subject_ids, y_true, y_prob):
        sid_str = str(sid)
        subj_probs[sid_str].append(float(yp))
        subj_true[sid_str] = int(yt)

    out_ids, out_true, out_prob = [], [], []
    for sid_str, probs_list in subj_probs.items():
        probs = np.array(probs_list, dtype=np.float32)

        if reducer == "mean":
            p = float(probs.mean())
        elif reducer == "max":
            p = float(probs.max())
        elif reducer == "topk_mean":
            k = max(1, min(int(topk), probs.size))
            top = np.sort(probs)[-k:]
            p = float(top.mean())
        else:
            raise ValueError("reducer must be 'mean', 'max', or 'topk_mean'")

        out_ids.append(sid_str)
        out_true.append(subj_true[sid_str])
        out_prob.append(p)

    return (
        np.array(out_ids),
        np.array(out_true, dtype=np.int64),
        np.array(out_prob, dtype=np.float32),
    )


def best_threshold_by_fbeta(y_true, y_prob, thresholds, beta: float):
    """
    Pick threshold that maximizes SUBJECT ACCURACY first, then:
      1) max accuracy
      2) minimize FN
      3) minimize FP
      4) maximize F-beta (tie-break)
    """
    best = None
    for thr in thresholds:
        tp, fp, tn, fn = _confusion_from_probs(y_true, y_prob, float(thr))
        m = _metrics_from_confusion(tp, fp, tn, fn)
        fbeta = fbeta_score_from_confusion(tp, fp, tn, fn, beta)

        key = (-m["acc"], fn, fp, -fbeta)

        if best is None or key < best["key"]:
            best = {"thr": float(thr), "key": key, "fbeta": float(fbeta), **m}
    return best


def _extract_subject_ids_from_batch(batch):
    for key in ["subject_id", "subject_ids", "sid", "sids"]:
        if key in batch:
            return batch[key]
    raise KeyError("Batch doesn't contain subject IDs")


@torch.no_grad()
def evaluate_with_subject_agg(model, loader, device, beta: float):
    model.eval()
    all_p1, all_y, all_sids = [], [], []

    for batch in loader:
        logmel = batch["logmel"].to(device)
        eg = batch["egemaps"].to(device)
        y = batch["label"].to(device)
        sids = _extract_subject_ids_from_batch(batch)

        logits = model(logmel, eg)
        probs = torch.softmax(logits, dim=1)

        all_p1.append(probs[:, 1].cpu())
        all_y.append(y.cpu())

        if torch.is_tensor(sids):
            all_sids.extend([str(x.item()) for x in sids.cpu()])
        else:
            all_sids.extend([str(x) for x in sids])

    p1_chunk = torch.cat(all_p1).numpy()
    y_true = torch.cat(all_y).numpy().astype(np.int64)

    subj_ids, subj_true, subj_prob = _aggregate_subject_level(
        all_sids, y_true, p1_chunk, reducer=AGG_MODE, topk=TOPK
    )

    best = best_threshold_by_fbeta(subj_true, subj_prob, THRESHOLDS, beta=beta)
    best["n_subjects"] = len(subj_ids)

    tp, fp, tn, fn = _confusion_from_probs(y_true, p1_chunk, 0.5)
    chunk_m = _metrics_from_confusion(tp, fp, tn, fn)
    chunk_m["fbeta"] = float(fbeta_score_from_confusion(tp, fp, tn, fn, beta))

    return {"chunk": chunk_m, "subject": best}


def _autocast_ctx(device: torch.device, enabled: bool):
    """
    Avoid FutureWarning from torch.cuda.amp.autocast by using torch.amp.autocast.
    On CPU this is a no-op.
    """
    try:
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    except Exception:
        # very old torch fallback
        return torch.cuda.amp.autocast(enabled=enabled)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, amp: bool):
    model.train()
    total_loss, total_n = 0.0, 0

    for batch in loader:
        logmel = batch["logmel"].to(device)
        eg = batch["egemaps"].to(device)
        y = batch["label"].to(device)

        logmel = spec_augment(logmel, training=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_ctx(device, amp):
            logits = model(logmel, eg)
            loss = criterion(logits, y)

        if amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * y.size(0)
        total_n += int(y.size(0))

    return total_loss / max(1, total_n)


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = USE_AMP and (device.type == "cuda")

    scaler = torch.cuda.amp.GradScaler(enabled=amp) if device.type == "cuda" else None

    print("=" * 70)
    print(f"HYBRID TRAINING (CRNN: 3×(2conv64+pool+dropout0.4) → BiLSTM(1) → meanpool → dense) + eGeMAPS(2-layer MLP)")
    print("=" * 70)
    print(f"Device: {device} | AMP: {amp}")
    print(f"SpecAugment: {USE_SPEC_AUGMENT}")
    print(f"Batch size: {BATCH_SIZE} | LR: {LR} | WD: {WEIGHT_DECAY}")
    print(f"Val subjects: {VAL_N_SUBJECTS} | Agg: {AGG_MODE} topk={TOPK}")
    print(f"Early stop patience: {EARLY_STOP_PATIENCE} (max ACC, then min FN, then min FP, then max F{BETA:g})")
    print("=" * 70)

    ds = ADChunkDataset(CHUNK_LABELS_CSV, LOGMEL_DIR, EGEMAPS_DIR, cache_egemaps=True)

    train_idx, val_idx, val_subjects = split_by_subject_fixed_n_stratified(ds.df, SEED, VAL_N_SUBJECTS)
    print(f"✅ Split: train={len(train_idx)} chunks, val={len(val_idx)} chunks (val subjects={len(val_subjects)})")

    eg_mean, eg_std = compute_egemaps_stats(ds, train_idx)

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        collate_fn=pad_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        collate_fn=pad_collate,
    )

    # ✅ IMPORTANT: use HYBRID mode by default now
    model = SimplifiedHybridModel(
        mode="hybrid",
        mel_bins=128,
        egemaps_input_dim=88,
        num_classes=2,
        # You can tune these if needed; match your model defaults
        lstm_hidden_size=128,
        cnn_dropout=0.4,
        crnn_embed_dim=128,
        eg_hidden1=128,
        eg_hidden2=64,
        eg_embed_dim=64,
        eg_dropout=0.2,
        fusion_hidden=128,
        fusion_dropout=0.2,
    ).to(device)

    # eGeMAPS normalization stats (even if you switch modes later)
    model.set_egemaps_stats(eg_mean.to(device), eg_std.to(device))

    # Param count (works even if model has no get_model_info)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📌 Model params: {total_params:,}")

    if USE_CLASS_WEIGHTS:
        train_labels = torch.tensor([ds.df.iloc[i]["label"] for i in train_idx], dtype=torch.long)
        weights = compute_class_weights(train_labels, device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print(f"Using class weights: {weights.detach().cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using unweighted CrossEntropyLoss")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR / 10)

    best_val_acc = -1.0
    best_val_fn = 10**9
    best_val_fp = 10**9
    best_val_f = -1.0
    best_epoch = 0
    patience = 0

    print("\n" + "=" * 70)
    print("TRAINING START")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, amp)
        train_eval = evaluate_with_subject_agg(model, train_loader, device, beta=BETA)
        val_eval = evaluate_with_subject_agg(model, val_loader, device, beta=BETA)

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        tr = train_eval["subject"]
        va = val_eval["subject"]

        train_chunk_acc = float(train_eval["chunk"]["acc"])
        val_chunk_acc = float(val_eval["chunk"]["acc"])

        print(
            f"\nEpoch {epoch:02d}/{EPOCHS} | loss={train_loss:.4f} | lr={lr_now:.6f}\n"
            f"  TRAIN subject: acc={tr['acc']:.3f} F{BETA:g}={tr['fbeta']:.3f} sens={tr['sensitivity']:.3f} spec={tr['specificity']:.3f} "
            f"(FP={tr['fp']:.1f} FN={tr['fn']:.1f}) @thr={tr['thr']:.2f} | chunk_acc={train_chunk_acc:.3f}\n"
            f"  VAL   subject: acc={va['acc']:.3f} F{BETA:g}={va['fbeta']:.3f} sens={va['sensitivity']:.3f} spec={va['specificity']:.3f} "
            f"(FP={va['fp']:.1f} FN={va['fn']:.1f}) @thr={va['thr']:.2f} | chunk_acc={val_chunk_acc:.3f}"
        )

        val_acc = float(va["acc"])
        val_fn = int(va["fn"])
        val_fp = int(va["fp"])
        val_f = float(va["fbeta"])

        improved = (
            (val_acc > best_val_acc + MIN_DELTA_ACC)
            or (
                abs(val_acc - best_val_acc) <= MIN_DELTA_ACC
                and (
                    (val_fn < best_val_fn)
                    or (val_fn == best_val_fn and val_fp < best_val_fp)
                    or (val_fn == best_val_fn and val_fp == best_val_fp and val_f > best_val_f)
                )
            )
        )

        if improved:
            best_val_acc = val_acc
            best_val_fn = val_fn
            best_val_fp = val_fp
            best_val_f = val_f
            best_epoch = epoch
            patience = 0

            model.save(
                SAVE_PATH,
                additional_info={
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_subject_metrics": va,
                    "train_subject_metrics": tr,
                    "beta": BETA,
                    "egemaps_mean": eg_mean.tolist(),
                    "egemaps_std": eg_std.tolist(),
                    "mode": "hybrid",
                },
            )
            print(
                f"  💾 BEST model saved (val acc={best_val_acc:.4f}, "
                f"FN={best_val_fn}, FP={best_val_fp}, F{BETA:g}={best_val_f:.4f})"
            )
        else:
            patience += 1
            print(
                f"  📉 No improvement ({patience}/{EARLY_STOP_PATIENCE}) | best: "
                f"acc={best_val_acc:.4f}, FN={best_val_fn}, FP={best_val_fp}, F{BETA:g}={best_val_f:.4f}"
            )

        if patience >= EARLY_STOP_PATIENCE:
            print(
                f"\n🛑 Early stopping at epoch {epoch} "
                f"(best epoch {best_epoch}, best val acc={best_val_acc:.4f}, "
                f"FN={best_val_fn}, FP={best_val_fp}, F{BETA:g}={best_val_f:.4f})"
            )
            break

    print("\n" + "=" * 70)
    print(
        f"✅ Done. Best epoch: {best_epoch} | best val acc={best_val_acc:.4f} | "
        f"best FN={best_val_fn} | best FP={best_val_fp} | best val F{BETA:g}={best_val_f:.4f}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
