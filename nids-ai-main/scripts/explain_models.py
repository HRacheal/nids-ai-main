#!/usr/bin/env python3
"""Phase 6: global and local explanations for the Track A models.

Outputs artifacts/metrics/track_a/explanations.json and
reports/figures/xai/*.png. Explanation samples are drawn from the TEST set
(stratified by class) because they describe the deployed models; nothing
here is used to train or select a model.

Usage:  .venv\\Scripts\\python.exe scripts\\explain_models.py [--quick]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepshield import viz  # noqa: E402
from deepshield.config import PROJECT_ROOT, ensure_dir, path  # noqa: E402
from deepshield.data.schema import CLASS_ORDER  # noqa: E402
from deepshield.training import store  # noqa: E402
from deepshield.training.trainer import load_checkpoint, predict  # noqa: E402
from deepshield.xai import explain as E  # noqa: E402

FIG = PROJECT_ROOT / "reports" / "figures" / "xai"
plt = viz.plt


def stratified(y, per_class, seed=0):
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), min(per_class, (y == c).sum()),
                                              replace=False) for c in np.unique(y)]))


def bar_panels(globals_: dict[str, dict[str, float]], out: Path, k: int = 12):
    viz.apply_style()
    n = len(globals_)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 4.2))
    for ax, (method, imp) in zip(np.atleast_1d(axes), globals_.items()):
        items = sorted(imp.items(), key=lambda kv: -kv[1])[:k][::-1]
        vals = np.array([v for _, v in items])
        ax.barh([n_ for n_, _ in items], vals / vals.max(), height=0.6, color=viz.SINGLE)
        ax.set_title(method, fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.set_xticks([0, 0.5, 1])
        ax.tick_params(axis="y", labelsize=7.5)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Global feature importance by method (each scaled to its own maximum)",
                 x=0.01, ha="left", fontweight="semibold")
    fig.supxlabel("relative importance", fontsize=8.5, color=viz.INK_2)
    fig.tight_layout()
    return viz.save(fig, out)


def class_heatmap(mat: np.ndarray, feats: list[str], out: Path, title: str):
    viz.apply_style()
    lim = np.abs(mat).max()
    fig, ax = plt.subplots(figsize=(9, 4.4))
    im = ax.imshow(mat, cmap=viz.DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(feats)), feats, rotation=60, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("mean SHAP value toward the true class")
    cb.outline.set_visible(False)
    ax.set_title(title)
    return viz.save(fig, out)


def local_plot(panels: list[tuple[str, list[dict]]], out: Path, title: str):
    viz.apply_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.4))
    for ax, (name, rows) in zip(np.atleast_1d(axes), panels):
        rows = rows[::-1]
        vals = [r["contribution"] for r in rows]
        colours = [viz.ATTACK_COLOUR if v > 0 else viz.NORMAL_COLOUR for v in vals]
        ax.barh([r["feature"] for r in rows], vals, color=colours, height=0.6)
        ax.axvline(0, color=viz.AXIS, lw=1)
        ax.set_title(name, fontsize=9)
        ax.grid(axis="y", visible=False)
    fig.suptitle(title, x=0.01, ha="left", fontweight="semibold")
    fig.supxlabel("contribution (orange: pushes toward the predicted class)", fontsize=8.5, color=viz.INK_2)
    fig.tight_layout()
    return viz.save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    ensure_dir(FIG)
    d = store.load_track_a_arrays()
    names, G = E.grouping(d["feature_names"])
    baseline = d["X_train"].mean(axis=0)
    rng = np.random.default_rng(0)
    background = d["X_train"][rng.choice(len(d["X_train"]), 50, replace=False)]
    s_shap = stratified(d["y_test"], 3 if args.quick else 20)
    s_ig = stratified(d["y_test"], 5 if args.quick else 50)
    s_perm = stratified(d["y_val"], 100 if args.quick else 900)
    out: dict = {"features": names, "samples": {"shap": len(s_shap), "ig": len(s_ig),
                                                 "permutation_val": len(s_perm)}}
    glob: dict[str, dict[str, float]] = {}

    # Random Forest: SHAP
    rf = joblib.load(path("models") / "track_a" / "rf_multiclass" / "model.joblib")
    rfe = E.RFExplainer(rf, background)
    X = d["X_test"][s_shap]
    sv = rfe.shap_values(X)                                     # (n, F, K)
    prob = rf.predict_proba(X)
    add_err = float(np.abs(sv.sum(1) + rfe.expected - prob).max())
    pred = prob.argmax(1)
    agg = E.aggregate(sv, G)                                     # (n, 42, K)
    sv_pred = agg[np.arange(len(X)), :, pred]                    # toward predicted class
    glob["RF - SHAP"] = dict(zip(names, np.abs(sv_pred).mean(0).tolist()))
    y = d["y_test"][s_shap]
    heat = np.stack([agg[y == c, :, c].mean(0) for c in range(len(CLASS_ORDER))])
    top = np.argsort(-np.abs(heat).sum(0))[:20]
    class_heatmap(heat[:, top], [names[i] for i in top], FIG / "rf_shap_by_class.png",
                  "Random Forest: features that push records toward their true class (SHAP)")
    out["rf_shap"] = {"additivity_max_error": add_err, "background_size": 50,
                      "global_mean_abs": glob["RF - SHAP"],
                      "per_class_mean_toward_true_class": {CLASS_ORDER[c]: dict(zip(names, heat[c].tolist()))
                                                           for c in range(len(CLASS_ORDER))}}
    print(f"RF SHAP done ({time.time() - t0:.0f}s), additivity error {add_err:.2e}", flush=True)

    # Neural networks: Integrated Gradients
    nets = {}
    for name in ("mlp", "cnn"):
        model, _ = load_checkpoint(path("models") / "track_a" / name / "model.pt")
        nets[name] = model
        Xi = d["X_test"][s_ig]
        tgt = predict(model, Xi).argmax(1)
        attr, delta = E.integrated_gradients(model, Xi, tgt, baseline, steps=32 if args.quick else 64)
        with __import__("torch").no_grad():
            import torch
            fx = model(torch.from_numpy(Xi)).gather(1, torch.from_numpy(tgt)[:, None]).squeeze(1)
            fb = model(torch.from_numpy(baseline[None, :].astype(np.float32)))[0, torch.from_numpy(tgt)]
        rel = np.abs(delta) / np.maximum(np.abs((fx - fb).numpy()), 1e-6)
        a = E.aggregate(attr, G)
        label = {"mlp": "MLP - Integrated Gradients", "cnn": "1D-CNN - Integrated Gradients"}[name]
        glob[label] = dict(zip(names, np.abs(a).mean(0).tolist()))
        out[f"{name}_ig"] = {"steps": 32 if args.quick else 64, "baseline": "mean training record",
                             "completeness_relative_error_median": float(np.median(rel)),
                             "completeness_relative_error_p95": float(np.quantile(rel, 0.95)),
                             "global_mean_abs": glob[label]}
        print(f"{name} IG done ({time.time() - t0:.0f}s), median completeness error {np.median(rel):.4f}",
              flush=True)

    # Autoencoder: which features are badly reconstructed for attacks
    ae, meta = load_checkpoint(path("models") / "track_a" / "autoencoder" / "model.pt")
    att = d["X_test"][s_ig][d["y_test"][s_ig] != 0]
    contrib = E.aggregate(E.autoencoder_contributions(ae, att), G)
    share = contrib / contrib.sum(1, keepdims=True)
    glob["Autoencoder - error share"] = dict(zip(names, share.mean(0).tolist()))
    out["autoencoder_error_share_attacks"] = glob["Autoencoder - error share"]

    # Model-agnostic: grouped permutation importance on validation (all three classifiers)
    Xv, yv = d["X_val"][s_perm], d["y_val"][s_perm]
    perm = {}
    for label, fn in (("RF", lambda X_: rf.predict(X_)),
                      ("MLP", lambda X_: predict(nets["mlp"], X_).argmax(1)),
                      ("1D-CNN", lambda X_: predict(nets["cnn"], X_).argmax(1))):
        perm[label] = E.grouped_permutation_importance(fn, Xv, yv, G, names, repeats=1 if args.quick else 3)
        print(f"permutation {label} done ({time.time() - t0:.0f}s)", flush=True)
    out["permutation_importance_val"] = perm
    glob["RF - permutation"] = {k: max(v["mean"], 0) for k, v in perm["RF"]["importance"].items()}
    glob["MLP - permutation"] = {k: max(v["mean"], 0) for k, v in perm["MLP"]["importance"].items()}

    # agreement between methods (Spearman over the 42 features) and top-10 overlap
    methods = list(glob)
    vec = {m: np.array([glob[m][f] for f in names]) for m in methods}
    out["agreement_spearman"] = {f"{a} | {b}": float(spearmanr(vec[a], vec[b]).statistic)
                                 for i, a in enumerate(methods) for b in methods[i + 1:]}
    top10 = {m: set(np.array(names)[np.argsort(-vec[m])[:10]]) for m in methods}
    out["top10"] = {m: sorted(top10[m], key=lambda f: -glob[m][f]) for m in methods}
    out["top10_overlap"] = {f"{a} | {b}": len(top10[a] & top10[b])
                            for i, a in enumerate(methods) for b in methods[i + 1:]}
    bar_panels({m: glob[m] for m in ["RF - SHAP", "RF - permutation", "MLP - Integrated Gradients",
                                     "1D-CNN - Integrated Gradients", "Autoencoder - error share"]},
               FIG / "global_importance.png")

    # local explanations: one correctly classified record per showcase class
    out["local_examples"] = {}
    for cls in ("DoS", "Exploits", "Reconnaissance"):
        c = CLASS_ORDER.index(cls)
        cand = [i for i, j in enumerate(s_shap) if d["y_test"][j] == c and pred[i] == c]
        if not cand:
            continue
        i = cand[0]
        j = s_shap[i]
        rf_top = E.top_contributions(agg[i, :, c], names)
        attr, _ = E.integrated_gradients(nets["mlp"], d["X_test"][j:j + 1], np.array([c]), baseline)
        mlp_top = E.top_contributions(E.aggregate(attr, G)[0], names)
        out["local_examples"][cls] = {"test_index": int(j), "rf_probability": float(prob[i, c]),
                                      "rf_shap_top": rf_top, "mlp_ig_top": mlp_top}
        local_plot([("Random Forest (SHAP, probability units)", rf_top),
                    ("MLP (Integrated Gradients, logit units)", mlp_top)],
                   FIG / f"local_{cls.lower()}.png",
                   f"Why was test record {j} classified as {cls}?")
    out["seconds"] = round(time.time() - t0)
    store.save_metrics("A", "explanations", out)
    print(f"done in {out['seconds']}s")


if __name__ == "__main__":
    main()
