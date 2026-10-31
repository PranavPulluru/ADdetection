# eval_poster_no_edit.py
# ============================================================
# NO-EDIT EVAL + POSTER VISUALS for your UP-TO-DATE pipeline
#
# Works with:
#   - dataset_no_chunks.py (ADClipDataset + pad_collate)
#   - model.py (HybridCRNNModel with mode: hybrid/logmel_only/egemaps_only)
#
# Assumes the SAME project structure style as train_cv5.py:
#   project_root/
#     clip_labels.csv
#     scripts/
#       eval_poster_no_edit.py   <-- put this file here and run it from scripts/
#       models/
#         cv5_fullclip_fold1_best.pth
#         ...
#
# Run:
#   python eval_poster_no_edit.py
#
# Output:
#   scripts/poster_outputs/
# ============================================================

import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# --- Your files (NO EDIT) ---
from dataset_no_chunks import ADClipDataset, pad_collate
from model import HybridCRNNModel


# ============================================================
# CONFIG (no edits needed if you match train_cv5 structure)
# ============================================================

@dataclass
class CFG:
    SEED: int = 42
    N_FOLDS: int = 5

    # Paths (match train_cv5 style)
    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ -> root
    CLIP_LABELS_CSV: str = os.path.join(PROJECT_ROOT, "clip_labels.csv")

    # Your training saves here in train_cv5.py:
    MODELS_DIR: str = os.path.join(PROJECT_ROOT, "scripts", "models")

    # Output
    OUT_DIR: str = os.path.join(PROJECT_ROOT, "scripts", "poster_outputs")

    # Eval params
    BATCH_SIZE: int = 8
    NUM_WORKERS: int = 2
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Threshold sweep resolution
    THRESH_GRID: int = 201  # 0..1 inclusive


C = CFG()


# ============================================================
# Repro
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Metrics
# ============================================================

def fbeta_from_pr(precision: float, recall: float, beta: float) -> float:
    if precision <= 0 and recall <= 0:
        return 0.0
    b2 = beta * beta
    denom = (b2 * precision + recall)
    if denom <= 0:
        return 0.0
    return (1 + b2) * precision * recall / denom

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_from_pr(prec, rec, beta=2.0)
    return {"acc": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1), "f2": float(f2)}

def mean_std(vals: List[float]) -> Tuple[float, float]:
    a = np.array(vals, dtype=np.float64)
    if len(a) == 0:
        return float("nan"), float("nan")
    if len(a) == 1:
        return float(a.mean()), 0.0
    return float(a.mean()), float(a.std(ddof=1))


# ============================================================
# Subject-stratified folds (no leakage)
#   Built directly from clip_labels.csv via ADClipDataset rows
# ============================================================

def build_subject_map(ds: ADClipDataset) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    subj_to_indices: Dict[str, List[int]] = {}
    subj_to_label: Dict[str, int] = {}

    for i, r in enumerate(ds.rows):
        sid = str(r["subject_id"])
        lab = int(r["label"])
        subj_to_indices.setdefault(sid, []).append(i)

        if sid in subj_to_label and subj_to_label[sid] != lab:
            raise ValueError(f"Subject {sid} has multiple labels in CSV: {subj_to_label[sid]} vs {lab}")
        subj_to_label[sid] = lab

    return subj_to_indices, subj_to_label

def subject_stratified_folds(subj_to_label: Dict[str, int], n_folds: int, seed: int, shuffle: bool = True):
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

def indices_for_subjects(subj_to_indices: Dict[str, List[int]], subject_ids: List[str]) -> List[int]:
    out: List[int] = []
    for sid in subject_ids:
        out.extend(subj_to_indices[str(sid)])
    return out


# ============================================================
# Plot helpers
# ============================================================

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def save_fig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def normalize_confmat(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = cm / row_sums
        out[np.isnan(out)] = 0.0
    return out

def plot_confmat(cm: np.ndarray, title: str, is_float: bool) -> None:
    plt.figure(figsize=(5.3, 4.6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["Control", "AD"])
    plt.yticks([0, 1], ["Control", "AD"])
    plt.colorbar(fraction=0.046, pad=0.04)

    thresh = cm.max() * 0.5 if cm.size else 0.0
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            txt = f"{v:.2f}" if is_float else f"{int(v)}"
            plt.text(j, i, txt, ha="center", va="center", color="white" if v > thresh else "black")

def plot_foldwise_bars(fold_metrics: List[Dict[str, float]], key: str, title: str, out_path: str) -> None:
    xs = [f"F{i+1}" for i in range(len(fold_metrics))]
    ys = [m[key] for m in fold_metrics]
    plt.figure(figsize=(6.6, 4.6))
    plt.bar(xs, ys)
    plt.ylim(0, 1.0)
    plt.ylabel(key.upper())
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.25)
    save_fig(out_path)

def plot_ablation_bar(vals: Dict[str, float], out_path: str) -> None:
    keys = list(vals.keys())
    ys = [vals[k] for k in keys]
    plt.figure(figsize=(6.2, 4.6))
    plt.bar(keys, ys)
    plt.ylim(0, 1.0)
    plt.ylabel("F2")
    plt.title("Ablation (F2 @ threshold=0.5, subject-level)")
    plt.grid(True, axis="y", alpha=0.25)
    save_fig(out_path)

def plot_threshold_sweep(thresholds: np.ndarray, f2s: np.ndarray, out_path: str) -> None:
    best_i = int(np.argmax(f2s))
    best_t = float(thresholds[best_i])
    best_f2 = float(f2s[best_i])

    plt.figure(figsize=(6.3, 4.6))
    plt.plot(thresholds, f2s)
    plt.axvline(best_t, linestyle="--")
    plt.xlabel("Decision Threshold")
    plt.ylabel("F2")
    plt.title(f"Threshold Sweep (Best F2={best_f2:.3f} @ t={best_t:.3f})")
    plt.grid(True, alpha=0.25)
    save_fig(out_path)


# ============================================================
# Model inference (supports your three modes)
# ============================================================

@torch.no_grad()
def probs_for_mode(model: HybridCRNNModel, loader: DataLoader, device: torch.device, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    old_mode = model.mode
    model.mode = mode
    model.eval()

    all_p1: List[torch.Tensor] = []
    all_y: List[torch.Tensor] = []

    for batch in loader:
        logmel = batch["logmel"].to(device)
        eg = batch["egemaps"].to(device)
        y = batch["label"].to(device)
        lengths = batch.get("lengths")
        if lengths is not None:
            lengths = lengths.to(device)

        logits = model(logmel, eg, lengths=lengths)
        p1 = torch.softmax(logits, dim=1)[:, 1]

        all_p1.append(p1.detach().float().cpu())
        all_y.append(y.detach().cpu())

    model.mode = old_mode

    y_true = torch.cat(all_y).numpy().astype(np.int64)
    p1 = torch.cat(all_p1).numpy().astype(np.float32)
    return y_true, p1


# ============================================================
# Main
# ============================================================

def main():
    set_seed(C.SEED)
    ensure_dir(C.OUT_DIR)

    device = torch.device(C.DEVICE)
    print(f"Device: {device}")
    print(f"CSV: {C.CLIP_LABELS_CSV}")
    print(f"Models: {C.MODELS_DIR}")
    print(f"Out: {C.OUT_DIR}")

    # Dataset (full-clip)
    ds = ADClipDataset(C.CLIP_LABELS_CSV, require_egemaps=True)
    subj_to_indices, subj_to_label = build_subject_map(ds)
    folds = subject_stratified_folds(subj_to_label, n_folds=C.N_FOLDS, seed=C.SEED, shuffle=True)

    # Per-fold results
    fold_metrics_hybrid: List[Dict[str, float]] = []
    fold_metrics_logmel: List[Dict[str, float]] = []
    fold_metrics_eg: List[Dict[str, float]] = []

    # Global arrays (aggregated across folds, subject-level is already ensured by fold splits)
    Y_all: List[np.ndarray] = []
    P_h_all: List[np.ndarray] = []
    P_l_all: List[np.ndarray] = []
    P_e_all: List[np.ndarray] = []

    # Evaluate each fold on its validation subjects using the saved fold checkpoint
    for fold_id, (train_sids, val_sids) in enumerate(folds, start=1):
        ckpt = os.path.join(C.MODELS_DIR, f"cv5_fullclip_fold{fold_id}_best.pth")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

        # Load model exactly as saved
        model, _info = HybridCRNNModel.load(ckpt, device=device, strict=False)

        val_idx = indices_for_subjects(subj_to_indices, val_sids)
        val_loader = DataLoader(
            Subset(ds, val_idx),
            batch_size=C.BATCH_SIZE,
            shuffle=False,
            num_workers=C.NUM_WORKERS,
            pin_memory=(device.type == "cuda"),
            collate_fn=pad_collate,
        )

        # probs for each mode
        y, p_h = probs_for_mode(model, val_loader, device, "hybrid")
        _, p_l = probs_for_mode(model, val_loader, device, "logmel_only")
        _, p_e = probs_for_mode(model, val_loader, device, "egemaps_only")

        # Metrics @ 0.5
        pred_h = (p_h >= 0.5).astype(np.int64)
        pred_l = (p_l >= 0.5).astype(np.int64)
        pred_e = (p_e >= 0.5).astype(np.int64)

        mh = compute_metrics(y, pred_h)
        ml = compute_metrics(y, pred_l)
        me = compute_metrics(y, pred_e)

        fold_metrics_hybrid.append(mh)
        fold_metrics_logmel.append(ml)
        fold_metrics_eg.append(me)

        Y_all.append(y)
        P_h_all.append(p_h)
        P_l_all.append(p_l)
        P_e_all.append(p_e)

        print(
            f"Fold {fold_id}: "
            f"Hybrid(acc={mh['acc']:.3f}, recall={mh['recall']:.3f}, f2={mh['f2']:.3f}) | "
            f"LogMel(f2={ml['f2']:.3f}) | eGeMAPS(f2={me['f2']:.3f})"
        )

    # Aggregate across folds
    y_all = np.concatenate(Y_all, axis=0)
    ph_all = np.concatenate(P_h_all, axis=0)
    pl_all = np.concatenate(P_l_all, axis=0)
    pe_all = np.concatenate(P_e_all, axis=0)

    # ------------------------------------------------------------
    # Save metric tables
    # ------------------------------------------------------------
    def write_fold_csv(name: str, fold_metrics: List[Dict[str, float]]):
        path = os.path.join(C.OUT_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write("fold,acc,precision,recall,f1,f2\n")
            for i, m in enumerate(fold_metrics, start=1):
                f.write(f"{i},{m['acc']:.6f},{m['precision']:.6f},{m['recall']:.6f},{m['f1']:.6f},{m['f2']:.6f}\n")

    write_fold_csv("metrics_foldwise_hybrid.csv", fold_metrics_hybrid)
    write_fold_csv("metrics_foldwise_logmel.csv", fold_metrics_logmel)
    write_fold_csv("metrics_foldwise_egemaps.csv", fold_metrics_eg)

    # mean±std summary (poster-friendly)
    def summarize(fold_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        out = {}
        for k in ["acc", "precision", "recall", "f1", "f2"]:
            vals = [m[k] for m in fold_metrics]
            mu, sd = mean_std(vals)
            out[k] = {"mean": mu, "std": sd}
        return out

    summary = {
        "hybrid": summarize(fold_metrics_hybrid),
        "logmel_only": summarize(fold_metrics_logmel),
        "egemaps_only": summarize(fold_metrics_eg),
    }
    with open(os.path.join(C.OUT_DIR, "metrics_mean_std.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # also write a compact poster CSV for hybrid
    with open(os.path.join(C.OUT_DIR, "poster_metrics_table_hybrid.csv"), "w", encoding="utf-8") as f:
        f.write("metric,mean,std\n")
        for k, v in summary["hybrid"].items():
            f.write(f"{k},{v['mean']:.6f},{v['std']:.6f}\n")

    # ------------------------------------------------------------
    # 1) Confusion matrices (Hybrid @ 0.5)
    # ------------------------------------------------------------
    pred_all = (ph_all >= 0.5).astype(np.int64)
    cm = confusion_matrix(y_all, pred_all, labels=[0, 1])
    cm_norm = normalize_confmat(cm)

    plot_confmat(cm, "Confusion Matrix (Counts) — Hybrid @0.5", is_float=False)
    save_fig(os.path.join(C.OUT_DIR, "confusion_matrix_counts.png"))

    plot_confmat(cm_norm, "Confusion Matrix (Row-Normalized) — Hybrid @0.5", is_float=True)
    save_fig(os.path.join(C.OUT_DIR, "confusion_matrix_normalized.png"))

    # ------------------------------------------------------------
    # 2) ROC curve (Hybrid vs LogMel vs eGeMAPS)
    # ------------------------------------------------------------
    fpr_h, tpr_h, _ = roc_curve(y_all, ph_all)
    fpr_l, tpr_l, _ = roc_curve(y_all, pl_all)
    fpr_e, tpr_e, _ = roc_curve(y_all, pe_all)

    auc_h = auc(fpr_h, tpr_h)
    auc_l = auc(fpr_l, tpr_l)
    auc_e = auc(fpr_e, tpr_e)

    plt.figure(figsize=(5.9, 4.7))
    plt.plot(fpr_h, tpr_h, label=f"Hybrid (AUC={auc_h:.3f})")
    plt.plot(fpr_l, tpr_l, label=f"Log-Mel (AUC={auc_l:.3f})")
    plt.plot(fpr_e, tpr_e, label=f"eGeMAPS (AUC={auc_e:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Validation across 5 folds)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.25)
    save_fig(os.path.join(C.OUT_DIR, "roc_curve.png"))

    # ------------------------------------------------------------
    # 3) Precision–Recall curve
    # ------------------------------------------------------------
    prec_h, rec_h, _ = precision_recall_curve(y_all, ph_all)
    prec_l, rec_l, _ = precision_recall_curve(y_all, pl_all)
    prec_e, rec_e, _ = precision_recall_curve(y_all, pe_all)

    ap_h = average_precision_score(y_all, ph_all)
    ap_l = average_precision_score(y_all, pl_all)
    ap_e = average_precision_score(y_all, pe_all)

    plt.figure(figsize=(5.9, 4.7))
    plt.plot(rec_h, prec_h, label=f"Hybrid (AP={ap_h:.3f})")
    plt.plot(rec_l, prec_l, label=f"Log-Mel (AP={ap_l:.3f})")
    plt.plot(rec_e, prec_e, label=f"eGeMAPS (AP={ap_e:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (Validation across 5 folds)")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.25)
    save_fig(os.path.join(C.OUT_DIR, "pr_curve.png"))

    # ------------------------------------------------------------
    # 4) Fold-wise bars (Hybrid)
    # ------------------------------------------------------------
    plot_foldwise_bars(
        fold_metrics_hybrid, "acc",
        "Fold-wise ACC (Hybrid @0.5)",
        os.path.join(C.OUT_DIR, "foldwise_acc.png")
    )
    plot_foldwise_bars(
        fold_metrics_hybrid, "recall",
        "Fold-wise RECALL (Hybrid @0.5)",
        os.path.join(C.OUT_DIR, "foldwise_recall.png")
    )
    plot_foldwise_bars(
        fold_metrics_hybrid, "f2",
        "Fold-wise F2 (Hybrid @0.5)",
        os.path.join(C.OUT_DIR, "foldwise_f2.png")
    )

    # ------------------------------------------------------------
    # 5) Ablation bar chart (F2 @0.5)
    # ------------------------------------------------------------
    overall_h = compute_metrics(y_all, (ph_all >= 0.5).astype(np.int64))
    overall_l = compute_metrics(y_all, (pl_all >= 0.5).astype(np.int64))
    overall_e = compute_metrics(y_all, (pe_all >= 0.5).astype(np.int64))

    ablation = {
        "Hybrid": overall_h["f2"],
        "Log-Mel": overall_l["f2"],
        "eGeMAPS": overall_e["f2"],
    }
    plot_ablation_bar(ablation, os.path.join(C.OUT_DIR, "ablation_f2.png"))

    # ------------------------------------------------------------
    # 6) Threshold sweep (Hybrid) — F2 vs threshold
    # ------------------------------------------------------------
    thresholds = np.linspace(0.0, 1.0, C.THRESH_GRID)
    f2s = []
    for t in thresholds:
        pred_t = (ph_all >= t).astype(np.int64)
        f2s.append(compute_metrics(y_all, pred_t)["f2"])
    f2s = np.array(f2s, dtype=np.float64)

    plot_threshold_sweep(thresholds, f2s, os.path.join(C.OUT_DIR, "threshold_sweep_f2.png"))

    # ------------------------------------------------------------
    # Save overall metrics JSON (poster easy)
    # ------------------------------------------------------------
    with open(os.path.join(C.OUT_DIR, "metrics_overall.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "hybrid@0.5": overall_h,
                "logmel_only@0.5": overall_l,
                "egemaps_only@0.5": overall_e,
                "roc_auc": {"hybrid": float(auc_h), "logmel": float(auc_l), "egemaps": float(auc_e)},
                "pr_ap": {"hybrid": float(ap_h), "logmel": float(ap_l), "egemaps": float(ap_e)},
            },
            f,
            indent=2,
        )

    print("\n✅ DONE. Poster outputs saved to:")
    print(f"   {C.OUT_DIR}")
    print("\nKey PNGs:")
    for name in [
        "confusion_matrix_counts.png",
        "confusion_matrix_normalized.png",
        "roc_curve.png",
        "pr_curve.png",
        "foldwise_acc.png",
        "foldwise_recall.png",
        "foldwise_f2.png",
        "ablation_f2.png",
        "threshold_sweep_f2.png",
    ]:
        print("  -", name)

    print("\nKey tables:")
    for name in [
        "metrics_foldwise_hybrid.csv",
        "metrics_foldwise_logmel.csv",
        "metrics_foldwise_egemaps.csv",
        "metrics_mean_std.json",
        "poster_metrics_table_hybrid.csv",
        "metrics_overall.json",
    ]:
        print("  -", name)


if __name__ == "__main__":
    main()
