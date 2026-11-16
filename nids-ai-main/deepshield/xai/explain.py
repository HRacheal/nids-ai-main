"""Explanations for DeepShield models.

* Random Forest: SHAP TreeExplainer, interventional, with a fixed background
  sample of training records. (The path-dependent variant failed SHAP's own
  additivity check on our class-weighted forest, so it is not used.) Every
  explanation satisfies additivity: base value + contributions = the
  model's predicted probability.
* PyTorch classifiers (MLP, 1D-CNN): Captum Integrated Gradients for the
  predicted class's logit, from a baseline equal to the mean training
  record. Completeness: attributions sum to f(x) - f(baseline), and the
  reported convergence delta measures the numerical error.
* Autoencoder: per-feature share of the squared reconstruction error.
* Model-agnostic: grouped permutation importance (drop in validation
  macro-F1 when one ORIGINAL feature's columns are shuffled together).

One-hot and missing-flag columns are summed back to their original feature
(e.g. all `proto=*` columns -> `proto`), so every method reports the same
42 (Track A) features. Attributions describe what the MODEL relies on;
they are not evidence that a feature causes an attack.
"""

from __future__ import annotations

import numpy as np
import torch


def base_feature(name: str) -> str:
    if "=" in name:
        return name.split("=", 1)[0]
    if name.endswith("_missing"):
        return name[: -len("_missing")]
    return name


def grouping(feature_names: list[str]) -> tuple[list[str], np.ndarray]:
    """(original feature names, matrix G with G[col, feature] = 1)."""
    bases = list(dict.fromkeys(base_feature(n) for n in feature_names))
    G = np.zeros((len(feature_names), len(bases)), dtype=np.float32)
    for i, n in enumerate(feature_names):
        G[i, bases.index(base_feature(n))] = 1.0
    return bases, G


def aggregate(attr: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Sum attributions over the columns of each original feature (last axis = columns)."""
    if attr.ndim == 3:            # (n, columns, classes) -> (n, features, classes)
        return np.einsum("ncK,cf->nfK", attr, G)
    return attr @ G


# ── Random Forest ─────────────────────────────────────────────────────────────

class RFExplainer:
    def __init__(self, model, background: np.ndarray):
        import shap
        self.model = model
        self.explainer = shap.TreeExplainer(model, data=background,
                                            feature_perturbation="interventional")
        self.expected = np.asarray(self.explainer.expected_value)

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        sv = self.explainer.shap_values(X)
        return np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)   # (n, F, K)


# ── PyTorch ───────────────────────────────────────────────────────────────────

def integrated_gradients(model: torch.nn.Module, X: np.ndarray, target: np.ndarray,
                         baseline: np.ndarray, steps: int = 64, batch: int = 64):
    """IG attributions (n, F) for `target` classes, plus convergence deltas (n,)."""
    from captum.attr import IntegratedGradients
    model.eval()
    ig = IntegratedGradients(model)
    outs, deltas = [], []
    base = torch.from_numpy(baseline.astype(np.float32))[None, :]
    for s in range(0, len(X), batch):
        xb = torch.from_numpy(np.ascontiguousarray(X[s:s + batch], dtype=np.float32))
        tb = torch.from_numpy(np.asarray(target[s:s + batch], dtype=np.int64))
        a, dl = ig.attribute(xb, baselines=base.expand_as(xb), target=tb, n_steps=steps,
                             internal_batch_size=batch * 8, return_convergence_delta=True)
        outs.append(a.detach().numpy())
        deltas.append(dl.detach().numpy())
    return np.concatenate(outs), np.concatenate(deltas)


def autoencoder_contributions(model, X: np.ndarray) -> np.ndarray:
    """Squared reconstruction error per column (n, F); rows sum to F * MSE."""
    with torch.no_grad():
        xb = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        return ((model(xb) - xb) ** 2).numpy()


# ── Model-agnostic ────────────────────────────────────────────────────────────

def grouped_permutation_importance(predict, X: np.ndarray, y: np.ndarray, G: np.ndarray,
                                   names: list[str], repeats: int = 3, seed: int = 0) -> dict:
    """Mean drop in macro-F1 when each original feature's columns are permuted together."""
    from deepshield.evaluation.metrics import multiclass_report
    rng = np.random.default_rng(seed)
    base = multiclass_report(y, predict(X))["macro_f1"]
    out = {}
    for j, name in enumerate(names):
        cols = np.flatnonzero(G[:, j])
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, cols] = X[rng.permutation(len(X))][:, cols]
            drops.append(base - multiclass_report(y, predict(Xp))["macro_f1"])
        out[name] = {"mean": float(np.mean(drops)), "std": float(np.std(drops))}
    return {"baseline_macro_f1": base, "importance": out}


def top_contributions(values: np.ndarray, names: list[str], raw: dict | None = None,
                      k: int = 8) -> list[dict]:
    """Top-|k| signed contributions of one record, for display."""
    order = np.argsort(-np.abs(values))[:k]
    return [{"feature": names[i], "contribution": float(values[i]),
             **({"value": raw.get(names[i])} if raw else {})} for i in order]
