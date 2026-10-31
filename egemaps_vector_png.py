import os
import csv
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from egemaps_viz import C, CLASS_NAMES, FEATURE_NAMES, pretty_name, load_data

# Feature families in eGeMAPSv02 order: (label, start index, end index exclusive)
GROUPS = [
    ("Pitch (F0)", 0, 10),
    ("Loudness", 10, 20),
    ("Flux", 20, 22),
    ("MFCC 1-4", 22, 30),
    ("Voice quality", 30, 40),
    ("Formants F1-F3", 40, 58),
    ("Spectral (voiced only)", 58, 76),
    ("Unvoiced", 76, 81),
    ("Timing + level", 81, 88),
]


def fmt_value(v: float) -> str:
    # Compact number formatting that fits inside a narrow cell
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.2f}"
    if a >= 0.01 or v == 0:
        return f"{v:.3f}"
    return f"{v:.1e}"


def load_one(file_id: str) -> np.ndarray:
    return np.load(os.path.join(C.EGEMAPS_DIR, f"{file_id}.npy")).astype(np.float64).flatten()


def label_for(file_id: str) -> str:
    with open(C.CLIP_LABELS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["file_id"] == file_id:
                return CLASS_NAMES[int(row["label"])]
    return "unlabeled"


def plot_vector(vec: np.ndarray, z: np.ndarray, n_clips: int, title: str, path: str) -> None:
    n = len(vec)
    cmap = plt.get_cmap("RdBu_r")
    norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)

    fig, ax = plt.subplots(figsize=(20, 5.4))
    ax.set_xlim(-1.2, n + 1.2)
    ax.set_ylim(-2.5, 1.4)
    ax.axis("off")

    # One cell per feature: color = z-score, text = raw value
    for i in range(n):
        zi = float(np.clip(z[i], -3, 3))
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=cmap(norm(zi)), edgecolor="white", linewidth=0.6))
        ax.text(i + 0.5, 0.5, fmt_value(vec[i]), rotation=90, ha="center", va="center", fontsize=6.5,
                color="white" if abs(zi) > 1.8 else "black")
        ax.text(i + 0.5, -0.08, f"{i}", ha="center", va="top", fontsize=5, color="0.4")
        ax.text(i + 0.5, -0.35, pretty_name(FEATURE_NAMES[i]), rotation=90, ha="center", va="top", fontsize=6.5)

    # Group dividers + labels above the strip
    for name, a, b in GROUPS:
        if a > 0:
            ax.plot([a, a], [-0.05, 1.05], color="black", linewidth=1.4)
        ax.plot([a + 0.1, b - 0.1], [1.08, 1.08], color="0.3", linewidth=1.0)
        ax.text((a + b) / 2, 1.15, name, ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Vector brackets
    for x0, dx in ((-0.4, 0.3), (n + 0.4, -0.3)):
        ax.plot([x0 + dx, x0, x0, x0 + dx], [1.02, 1.02, -0.02, -0.02], color="black", linewidth=2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, aspect=50, shrink=0.35)
    cbar.set_label(f"Color: z-score vs. all {n_clips} recordings (blue = low, red = high).  Text: raw value.",
                   fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title(title, fontsize=13, pad=4)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one recording's 88-dim eGeMAPS vector as a PNG.")
    parser.add_argument("--file_id", type=str, default="S001.interviewee_only")
    args = parser.parse_args()

    X, _ = load_data()
    mu, sd = X.mean(axis=0), X.std(axis=0).clip(min=1e-8)

    vec = load_one(args.file_id)
    z = (vec - mu) / sd
    subject = args.file_id.split(".")[0]
    title = f"eGeMAPS feature vector (88 values): {subject} ({label_for(args.file_id)})"

    os.makedirs(C.OUT_DIR, exist_ok=True)
    plot_vector(vec, z, X.shape[0], title, os.path.join(C.OUT_DIR, f"egemaps_vector_{subject}.png"))


if __name__ == "__main__":
    main()
