"""Evaluation metrics shared by every model.

Multiclass: accuracy, macro/weighted precision-recall-F1, per-class metrics,
confusion matrix, one-vs-rest ROC-AUC and PR-AUC (average precision).
Binary (attack vs normal): the same plus false-positive and false-negative
rates. A multiclass model's attack score is 1 - P(Normal).

Uncertainty: stratified bootstrap confidence intervals over test records,
paired bootstrap for differences between two models on the same records,
and McNemar's exact test on paired errors with Holm correction.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

from deepshield.data.schema import CLASS_ORDER


def _macro_f1_from_cm(cm: np.ndarray, present: np.ndarray) -> float:
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1[present].mean())


def multiclass_report(y_true: np.ndarray, y_pred: np.ndarray,
                      proba: np.ndarray | None = None,
                      classes: list[str] = CLASS_ORDER) -> dict:
    k = len(classes)
    labels = np.arange(k)
    present = np.bincount(y_true, minlength=k) > 0
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    out = {
        "n": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        # macro averages over classes PRESENT in y_true (absent classes have
        # undefined recall; they are listed as unavailable instead)
        "macro_precision": float(p[present].mean()),
        "macro_recall": float(r[present].mean()),
        "macro_f1": float(f[present].mean()),
        "weighted_f1": float(np.average(f[present], weights=s[present])),
        "per_class": {c: ({"precision": float(p[i]), "recall": float(r[i]),
                           "f1": float(f[i]), "support": int(s[i])} if present[i]
                          else {"support": 0, "status": "unavailable (no test support)"})
                      for i, c in enumerate(classes)},
        "confusion_matrix": cm.tolist(),
        "classes": list(classes),
    }
    if proba is not None:
        aucs, aps = {}, {}
        for i, c in enumerate(classes):
            yi = (y_true == i)
            if 0 < yi.sum() < len(yi):
                aucs[c] = float(roc_auc_score(yi, proba[:, i]))
                aps[c] = float(average_precision_score(yi, proba[:, i]))
        out["roc_auc_ovr_per_class"], out["pr_auc_per_class"] = aucs, aps
        out["roc_auc_ovr_macro"] = float(np.mean(list(aucs.values())))
        out["pr_auc_macro"] = float(np.mean(list(aps.values())))
    # binary view of a multiclass model
    yb = (y_true != 0).astype(int)
    out["binary_view"] = binary_report(
        yb, (y_pred != 0).astype(int),
        None if proba is None else 1.0 - proba[:, 0])
    return out


def binary_report(y_true: np.ndarray, y_pred: np.ndarray,
                  score: np.ndarray | None = None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    out = {
        "n": int(len(y_true)), "accuracy": float((tp + tn) / len(y_true)),
        "precision": float(prec), "recall_detection_rate": float(rec),
        "f1": float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    if score is not None and 0 < y_true.sum() < len(y_true):
        out["roc_auc"] = float(roc_auc_score(y_true, score))
        out["pr_auc"] = float(average_precision_score(y_true, score))
    return out


def detection_by_category(y_multi: np.ndarray, flagged: np.ndarray,
                          classes: list[str] = CLASS_ORDER) -> dict:
    """Share of each category flagged as attack (for Normal: false-positive rate)."""
    return {c: {"flagged_fraction": float(flagged[y_multi == i].mean()),
                "support": int((y_multi == i).sum())}
            for i, c in enumerate(classes) if (y_multi == i).any()}


# ── uncertainty ───────────────────────────────────────────────────────────────

def _strata_resample(rng, strata_idx: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in strata_idx])


def bootstrap_ci(y_true, y_pred, metric: str = "macro_f1", n_boot: int = 1000,
                 seed: int = 0, k: int = len(CLASS_ORDER), alpha: float = 0.05,
                 blocks: np.ndarray | None = None) -> dict:
    """Percentile CI. Stratified by class (Track A) or by time block (Track B)."""
    rng = np.random.default_rng(seed)
    strata = blocks if blocks is not None else y_true
    strata_idx = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    present = np.bincount(y_true, minlength=k) > 0
    vals = np.empty(n_boot)
    for b in range(n_boot):
        ix = _strata_resample(rng, strata_idx)
        yt, yp = y_true[ix], y_pred[ix]
        if metric == "macro_f1":
            cm = np.bincount(yt * k + yp, minlength=k * k).reshape(k, k)
            vals[b] = _macro_f1_from_cm(cm, present)
        elif metric == "accuracy":
            vals[b] = (yt == yp).mean()
        else:
            raise ValueError(metric)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"metric": metric, "low": float(lo), "high": float(hi), "n_boot": n_boot}


def paired_bootstrap_diff(y_true, pred_a, pred_b, n_boot: int = 1000, seed: int = 0,
                          k: int = len(CLASS_ORDER), blocks: np.ndarray | None = None) -> dict:
    """CI for macro-F1(A) - macro-F1(B) resampling the SAME records for both."""
    rng = np.random.default_rng(seed)
    strata = blocks if blocks is not None else y_true
    strata_idx = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    present = np.bincount(y_true, minlength=k) > 0

    def mf1(yt, yp):
        return _macro_f1_from_cm(np.bincount(yt * k + yp, minlength=k * k).reshape(k, k), present)

    obs = mf1(y_true, pred_a) - mf1(y_true, pred_b)
    d = np.empty(n_boot)
    for b in range(n_boot):
        ix = _strata_resample(rng, strata_idx)
        d[b] = mf1(y_true[ix], pred_a[ix]) - mf1(y_true[ix], pred_b[ix])
    lo, hi = np.quantile(d, [0.025, 0.975])
    return {"diff_macro_f1": float(obs), "low": float(lo), "high": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """Exact McNemar test on records where exactly one model is wrong."""
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    n01, n10 = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"a_right_b_wrong": n01, "a_wrong_b_right": n10, "p_value": float(p)}


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[name] = {"p": p, "p_holm": adj, "significant": adj < alpha}
    return out
