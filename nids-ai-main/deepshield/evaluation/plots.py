"""Standard evaluation figures (shared palette from deepshield.viz)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from deepshield import viz
from deepshield.data.schema import CLASS_ORDER

plt = viz.plt


def confusion_matrix_plot(cm, title: str, out: Path, classes=CLASS_ORDER) -> Path:
    viz.apply_style()
    cm = np.asarray(cm, dtype=float)
    rows = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(norm, cmap=viz.SEQUENTIAL, vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] > 0:
                ax.text(j, i, f"{norm[i, j]:.2f}\n{int(cm[i, j]):,}" if i == j else f"{norm[i, j]:.2f}",
                        ha="center", va="center", fontsize=6.3,
                        color="#ffffff" if norm[i, j] > 0.55 else viz.INK_2)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("share of true class (row-normalised)")
    cb.outline.set_visible(False)
    ax.set_title(title)
    return viz.save(fig, out)


def roc_pr_plot(curves: dict[str, tuple[np.ndarray, np.ndarray]], title: str, out: Path) -> Path:
    """curves: model name -> (y_true binary, attack score). Up to 8 models."""
    viz.apply_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for (name, (y, s)), colour in zip(curves.items(), viz.CATEGORICAL):
        fpr, tpr, _ = roc_curve(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        a1.plot(fpr, tpr, color=colour, label=name)
        a2.plot(rec, prec, color=colour, label=name)
    a1.plot([0, 1], [0, 1], color=viz.AXIS, lw=1, ls="--")
    a1.set(xlabel="false-positive rate", ylabel="true-positive rate (detection rate)",
           title="ROC (attack vs normal)")
    a2.set(xlabel="recall", ylabel="precision", title="Precision-recall (attack vs normal)")
    for a in (a1, a2):
        a.set_xlim(0, 1)
        a.set_ylim(0, 1.01)
    a2.legend(loc="lower left", fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    return viz.save(fig, out)


def per_class_f1_plot(results: dict[str, dict], title: str, out: Path,
                      classes=CLASS_ORDER) -> Path:
    """results: model -> multiclass_report. Grouped horizontal bars."""
    viz.apply_style()
    models = list(results)[:8]
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(classes) * max(1, len(models) / 3) + 1.2))
    h = 0.8 / len(models)
    ys = np.arange(len(classes))[::-1]
    for m, (name, colour) in enumerate(zip(models, viz.CATEGORICAL)):
        f1 = [results[name]["per_class"][c].get("f1", np.nan) for c in classes]
        ax.barh(ys + 0.4 - h * (m + 0.5), f1, height=h * 0.9, color=colour, label=name)
    ax.set_yticks(ys, classes)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1-score on test set")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8, ncols=min(4, len(models)))
    ax.set_title(title)
    return viz.save(fig, out)


def history_plot(histories: dict[str, dict], title: str, out: Path) -> Path:
    """histories: model -> {"train_loss": [...], "val_loss": [...], "best_epoch": int}."""
    viz.apply_style()
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3), squeeze=False)
    for ax, (name, h) in zip(axes[0], histories.items()):
        ep = np.arange(1, len(h["train_loss"]) + 1)
        ax.plot(ep, h["train_loss"], color=viz.CATEGORICAL[0], label="train")
        ax.plot(ep, h["val_loss"], color=viz.CATEGORICAL[1], label="validation")
        if h.get("best_epoch"):
            ax.axvline(h["best_epoch"], color=viz.MUTED, lw=1, ls="--")
        ax.set_title(name, fontsize=9.5)
        ax.set_xlabel("epoch")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    return viz.save(fig, out)


def score_hist_plot(scores_normal, scores_attack, threshold: float, title: str,
                    xlabel: str, out: Path) -> Path:
    viz.apply_style()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    lo = min(np.quantile(scores_normal, 0.001), np.quantile(scores_attack, 0.001))
    hi = max(np.quantile(scores_normal, 0.999), np.quantile(scores_attack, 0.999))
    bins = np.linspace(lo, hi, 80)
    ax.hist(scores_normal, bins=bins, density=True, histtype="step", lw=1.6,
            color=viz.NORMAL_COLOUR, label="Normal")
    ax.hist(scores_attack, bins=bins, density=True, histtype="step", lw=1.6,
            color=viz.ATTACK_COLOUR, label="Attack")
    ax.axvline(threshold, color=viz.INK_2, lw=1, ls="--")
    ax.annotate("threshold", (threshold, ax.get_ylim()[1] * 0.92), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color=viz.INK_2)
    ax.set_xlabel(xlabel)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8)
    ax.set_title(title)
    return viz.save(fig, out)
