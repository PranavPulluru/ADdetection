import os
import re
import csv
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


@dataclass
class CFG:
    SEED: int = 42

    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ -> root
    CLIP_LABELS_CSV: str = os.path.join(PROJECT_ROOT, "clip_labels.csv")
    EGEMAPS_DIR: str = os.path.join(PROJECT_ROOT, "data", "processed", "eGeMAPS")

    OUT_DIR: str = os.path.join(PROJECT_ROOT, "scripts", "poster_outputs")

    TOP_K_BARS: int = 20  # features shown in effect-size chart
    TOP_K_BOX: int = 8  # features shown in per-class box plots


C = CFG()

CLASS_NAMES = ["Control", "AD"]
CLASS_COLORS = ["#4C72B0", "#DD8452"]


# eGeMAPSv02 Functionals feature order (opensmile.FeatureSet.eGeMAPSv02, FeatureLevel.Functionals)
def _egemaps_feature_names() -> List[str]:
    names: List[str] = []
    lld_stats = [
        "amean", "stddevNorm", "percentile20.0", "percentile50.0", "percentile80.0", "pctlrange0-2",
        "meanRisingSlope", "stddevRisingSlope", "meanFallingSlope", "stddevFallingSlope",
    ]
    names += [f"F0semitoneFrom27.5Hz_sma3nz_{s}" for s in lld_stats]
    names += [f"loudness_sma3_{s}" for s in lld_stats]
    ms = ["amean", "stddevNorm"]
    names += [f"spectralFlux_sma3_{s}" for s in ms]
    for k in range(1, 5):
        names += [f"mfcc{k}_sma3_{s}" for s in ms]
    for f in ["jitterLocal", "shimmerLocaldB", "HNRdBACF", "logRelF0-H1-H2", "logRelF0-H1-A3"]:
        names += [f"{f}_sma3nz_{s}" for s in ms]
    for k in range(1, 4):
        for f in ["frequency", "bandwidth", "amplitudeLogRelF0"]:
            names += [f"F{k}{f}_sma3nz_{s}" for s in ms]
    for f in ["alphaRatioV", "hammarbergIndexV", "slopeV0-500", "slopeV500-1500", "spectralFluxV"]:
        names += [f"{f}_sma3nz_{s}" for s in ms]
    for k in range(1, 5):
        names += [f"mfcc{k}V_sma3nz_{s}" for s in ms]
    for f in ["alphaRatioUV", "hammarbergIndexUV", "slopeUV0-500", "slopeUV500-1500", "spectralFluxUV"]:
        names.append(f"{f}_sma3nz_amean")
    names += [
        "loudnessPeaksPerSec", "VoicedSegmentsPerSec", "MeanVoicedSegmentLengthSec",
        "StddevVoicedSegmentLengthSec", "MeanUnvoicedSegmentLength", "StddevUnvoicedSegmentLength",
        "equivalentSoundLevel_dBp",
    ]
    assert len(names) == 88, len(names)
    return names


FEATURE_NAMES = _egemaps_feature_names()


def pretty_name(name: str) -> str:
    # Shortens raw openSMILE names for plot labels
    s = name.replace("F0semitoneFrom27.5Hz", "F0").replace("_sma3nz", "").replace("_sma3", "")
    s = s.replace("_amean", " mean").replace("_stddevNorm", " CV")
    s = s.replace("_percentile20.0", " p20").replace("_percentile50.0", " median").replace("_percentile80.0", " p80")
    s = s.replace("_pctlrange0-2", " range p20-p80")
    s = s.replace("_meanRisingSlope", " rising slope").replace("_stddevRisingSlope", " rising slope SD")
    s = s.replace("_meanFallingSlope", " falling slope").replace("_stddevFallingSlope", " falling slope SD")
    s = s.replace("HNRdBACF", "HNR").replace("shimmerLocaldB", "shimmer").replace("jitterLocal", "jitter")
    s = s.replace("logRelF0-", "").replace("amplitudeLogRelF0", " amplitude")
    s = s.replace("equivalentSoundLevel_dBp", "equivalent sound level")
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s) if " " not in s and "_" not in s else s
    return s


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    # Loads one 88-dim eGeMAPS vector + label per clip listed in clip_labels.csv
    X_list, y_list = [], []
    with open(C.CLIP_LABELS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("egemaps_missing", "False")).strip().lower() == "true":
                continue
            path = os.path.join(C.EGEMAPS_DIR, f"{row['file_id']}.npy")
            vec = np.load(path).astype(np.float64).flatten()
            if vec.size != 88:
                raise ValueError(f"{path}: expected 88 features, got {vec.size}")
            X_list.append(vec)
            y_list.append(int(row["label"]))
    return np.stack(X_list), np.array(y_list)


def zscore(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(axis=0)) / X.std(axis=0).clip(min=1e-8)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    # Effect size of b relative to a, using pooled SD
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((b.mean() - a.mean()) / pooled) if pooled > 0 else 0.0


def fdr_bh(p: np.ndarray) -> np.ndarray:
    # Benjamini-Hochberg adjusted p-values
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    out = np.empty(n)
    out[order] = q
    return out


def save(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_effect_sizes(d: np.ndarray, q: np.ndarray, path: str) -> np.ndarray:
    top = np.argsort(-np.abs(d))[: C.TOP_K_BARS][::-1]
    labels = [pretty_name(FEATURE_NAMES[i]) + (" *" if q[i] < 0.05 else "") for i in top]
    colors = [CLASS_COLORS[1] if d[i] > 0 else CLASS_COLORS[0] for i in top]

    plt.figure(figsize=(7.0, 0.32 * len(top) + 1.4))
    plt.barh(range(len(top)), d[top], color=colors)
    plt.yticks(range(len(top)), labels, fontsize=8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Cohen's d (AD − Control)")
    plt.title(f"Top {len(top)} eGeMAPS features by effect size")
    plt.grid(True, axis="x", alpha=0.25)
    plt.figtext(0.01, -0.01, "Orange: higher in AD, blue: lower in AD.  * FDR-adjusted Welch t-test q < 0.05",
                fontsize=7, ha="left")
    save(path)
    return top[::-1]


def plot_top_boxplots(X: np.ndarray, y: np.ndarray, d: np.ndarray, path: str) -> None:
    top = np.argsort(-np.abs(d))[: C.TOP_K_BOX]
    ncols = 4
    nrows = int(np.ceil(len(top) / ncols))
    rng = np.random.default_rng(C.SEED)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.8 * nrows))
    for ax, i in zip(axes.flat, top):
        groups = [X[y == c, i] for c in (0, 1)]
        bp = ax.boxplot(groups, widths=0.55, patch_artist=True, showfliers=False)
        for patch, col in zip(bp["boxes"], CLASS_COLORS):
            patch.set_facecolor(col)
            patch.set_alpha(0.45)
        for k, (g, col) in enumerate(zip(groups, CLASS_COLORS), start=1):
            ax.scatter(k + rng.uniform(-0.15, 0.15, len(g)), g, s=5, color=col, alpha=0.6, linewidths=0)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_title(f"{pretty_name(FEATURE_NAMES[i])}\n(d = {d[i]:+.2f})", fontsize=9)
        ax.grid(True, axis="y", alpha=0.25)
    for ax in list(axes.flat)[len(top):]:
        ax.axis("off")
    fig.suptitle("Most class-separating eGeMAPS features (raw units)")
    save(path)


def plot_embedding(Z2: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str, path: str) -> None:
    plt.figure(figsize=(5.9, 4.7))
    for c in (0, 1):
        m = y == c
        plt.scatter(Z2[m, 0], Z2[m, 1], s=14, alpha=0.7, color=CLASS_COLORS[c],
                    label=f"{CLASS_NAMES[c]} (n={m.sum()})", linewidths=0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.25)
    save(path)


def main() -> None:
    os.makedirs(C.OUT_DIR, exist_ok=True)
    X, y = load_data()
    print(f"Loaded {X.shape[0]} clips x {X.shape[1]} features "
          f"({(y == 0).sum()} {CLASS_NAMES[0]}, {(y == 1).sum()} {CLASS_NAMES[1]})")

    d = np.array([cohens_d(X[y == 0, i], X[y == 1, i]) for i in range(X.shape[1])])
    p = np.array([stats.ttest_ind(X[y == 1, i], X[y == 0, i], equal_var=False).pvalue for i in range(X.shape[1])])
    q = fdr_bh(np.nan_to_num(p, nan=1.0))

    with open(os.path.join(C.OUT_DIR, "egemaps_feature_stats.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "control_mean", "ad_mean", "cohens_d", "p_welch", "q_fdr"])
        for i in np.argsort(-np.abs(d)):
            w.writerow([FEATURE_NAMES[i], X[y == 0, i].mean(), X[y == 1, i].mean(), d[i], p[i], q[i]])
    print(f"{(q < 0.05).sum()} / {len(q)} features significant at FDR q < 0.05")

    plot_effect_sizes(d, q, os.path.join(C.OUT_DIR, "egemaps_effect_sizes.png"))
    plot_top_boxplots(X, y, d, os.path.join(C.OUT_DIR, "egemaps_top_features_boxplots.png"))

    Z = zscore(X)
    pca = PCA(n_components=2, random_state=C.SEED)
    Zp = pca.fit_transform(Z)
    ev = pca.explained_variance_ratio_ * 100
    plot_embedding(Zp, y, "PCA of eGeMAPS feature vectors",
                   f"PC1 ({ev[0]:.1f}% var)", f"PC2 ({ev[1]:.1f}% var)",
                   os.path.join(C.OUT_DIR, "egemaps_pca.png"))

    Zt = TSNE(n_components=2, perplexity=30, init="pca", random_state=C.SEED).fit_transform(Z)
    plot_embedding(Zt, y, "t-SNE of eGeMAPS feature vectors", "t-SNE 1", "t-SNE 2",
                   os.path.join(C.OUT_DIR, "egemaps_tsne.png"))


if __name__ == "__main__":
    main()
