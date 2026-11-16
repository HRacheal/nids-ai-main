#!/usr/bin/env python3
"""Phase 5: leave-one-attack-class-out (LOCO) experiment, Track A.

This is NOT ordinary multiclass evaluation. For each attack category c:

1. Remove every record of c from the training AND validation data before
   ANY fitted step. The preprocessor (imputation medians, means/standard
   deviations, one-hot levels) is refitted for every fold on the fold's
   training records only, so c cannot leak in through preprocessing either.
2. Train a closed-set classifier (RF, MLP) on the remaining 9 classes (Normal
   + 8 attacks) with the configuration selected in Phases 3-4.

   Reduced training data (docs/sampling.md): the training records are a
   minority-preserving stratified sample of the Track A training split
   (fraction chosen on validation data by scripts/sampling_study.py, seed 0,
   the same sample for every fold; classes under 2,000 records kept whole).
   Validation data and the official test set are complete.
3. Evaluate on the official test set. For the held-out category's test
   records (which the model has never seen) we report:
     - "as Normal"         : missed entirely (false negatives)
     - "as another attack" : still raised as an attack, wrong category
     - "as unknown"        : max softmax probability below a threshold set
                             on VALIDATION data of the known classes only
                             (5 % of known validation records fall below it)
   and, for comparison, how often the same threshold rejects test records
   of KNOWN classes.
4. The anomaly detectors (Autoencoder, Isolation Forest) are trained on
   normal traffic only, so every attack class is already unseen by them;
   their detection rate per category is reported alongside.

Usage:  .venv\\Scripts\\python.exe scripts\\leave_one_class_out.py [--quick] [--models rf mlp]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepshield.config import ensure_dir  # noqa: E402
from deepshield.data import sampling  # noqa: E402
from deepshield.data.loaders import load_track_a  # noqa: E402
from deepshield.data.preprocess import Preprocessor, encode_targets  # noqa: E402
from deepshield.data.splits import track_a_validation_split  # noqa: E402
from deepshield.data.imbalance import class_weights, random_oversample  # noqa: E402
from deepshield.data.schema import ATTACK_CLASSES, CLASS_ORDER  # noqa: E402
from deepshield.evaluation import metrics as M  # noqa: E402
from deepshield.training import store  # noqa: E402
from deepshield.training.trainer import TrainConfig, array_batches, fit, predict  # noqa: E402
from deepshield import viz  # noqa: E402

REJECT_QUANTILE = 0.05


def train_rf(Xtr, ytr, cfg, n_est):
    from sklearn.ensemble import RandomForestClassifier
    idx = random_oversample(ytr, 2000) if cfg["imbalance"] == "oversample" else np.arange(len(ytr))
    rf = RandomForestClassifier(n_estimators=n_est, min_samples_leaf=cfg["min_samples_leaf"],
                                max_features=cfg["max_features"], n_jobs=4, random_state=0,
                                class_weight="balanced_subsample" if cfg["imbalance"] == "balanced_subsample" else None)
    rf.fit(Xtr[idx], ytr[idx])
    return lambda X: rf.predict_proba(X)


def train_mlp(Xtr, ytr, Xva, yva, m, k):
    params = {**m["params"], "n_out": k, "n_in": Xtr.shape[1]}
    tc = {kk: v for kk, v in m["train_config"].items() if kk in TrainConfig.__dataclass_fields__}
    cfg = TrainConfig(**{**tc, "seed": 0})
    idx = np.arange(len(ytr))
    w = np.ones(k)
    if m["imbalance"] == "oversample":
        idx = random_oversample(ytr, 2000)
    else:
        w = class_weights(ytr, k, m["imbalance"])
    loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    res = fit("MLP", params, cfg, array_batches(Xtr[idx], ytr[idx], cfg.batch_size, True),
              array_batches(Xva, yva, 4096, False), loss, log=lambda *_: None)
    return lambda X: predict(res.model, X)


def fold_arrays(raw, sample, c):
    """Preprocessor fitted on the fold's training records (class c excluded); transformed splits."""
    tr_rows = sample[raw["y_train"][sample] != c]
    va_rows = np.flatnonzero(raw["y_val"] != c)
    fit_df = raw["train"].iloc[tr_rows]
    assert c not in encode_targets(fit_df)[0]                         # never seen by the preprocessor
    pre = Preprocessor.for_track("A").fit(fit_df)
    return {"X_train": pre.transform(fit_df), "y_train": raw["y_train"][tr_rows],
            "X_val": pre.transform(raw["val"].iloc[va_rows]), "y_val": raw["y_val"][va_rows],
            "X_test": pre.transform(raw["test"]), "y_test": raw["y_test"],
            "n_features": pre.n_features, "preprocessor_fit_records": int(len(tr_rows))}


def run(raw, sample, held: str, model: str, rf_cfg, rf_n, mlp_meta):
    c = CLASS_ORDER.index(held)
    known = [i for i in range(len(CLASS_ORDER)) if i != c]
    remap = -np.ones(len(CLASS_ORDER), dtype=np.int64)
    remap[known] = np.arange(len(known))
    d = fold_arrays(raw, sample, c)
    Xtr, ytr = d["X_train"], remap[d["y_train"]]
    Xva, yva = d["X_val"], remap[d["y_val"]]
    assert c not in d["y_train"] and c not in d["y_val"]              # held-out class fully removed
    assert (ytr >= 0).all() and (yva >= 0).all()
    t0 = time.time()
    f = train_rf(Xtr, ytr, rf_cfg, rf_n) if model == "rf" else train_mlp(Xtr, ytr, Xva, yva, mlp_meta, len(known))
    secs = time.time() - t0
    pva = f(Xva)
    tau = float(np.quantile(pva.max(1), REJECT_QUANTILE))          # known-class validation only
    pte = f(d["X_test"])
    pred_full = np.array(known)[pte.argmax(1)]                       # back to original class ids
    reject = pte.max(1) < tau
    y = d["y_test"]
    h = y == c
    others = {CLASS_ORDER[k]: float((pred_full[h] == k).mean()) for k in known}
    known_mask = ~h
    rep_known = M.multiclass_report(remap[y[known_mask]], pte[known_mask].argmax(1),
                                    classes=[CLASS_ORDER[k] for k in known])
    return {
        "held_out": held, "model": model, "train_seconds": round(secs, 1),
        "held_out_test_support": int(h.sum()),
        "held_out_predicted_as_normal": float((pred_full[h] == 0).mean()),
        "held_out_predicted_as_other_attack": float((pred_full[h] != 0).mean()),
        "held_out_prediction_distribution": others,
        "most_common_wrong_attack": max(((k, v) for k, v in others.items() if k != "Normal"),
                                        key=lambda kv: kv[1])[0],
        "rejection_threshold_max_prob": tau,
        "held_out_rejected_as_unknown": float(reject[h].mean()),
        "held_out_flagged_attack_or_unknown": float(((pred_full[h] != 0) | reject[h]).mean()),
        "known_classes_rejected_as_unknown": float(reject[known_mask].mean()),
        "known_classes_macro_f1": rep_known["macro_f1"],
        "n_features_fold": d["n_features"], "preprocessor_fit_records": d["preprocessor_fit_records"],
        "train_counts": sampling.class_counts(d["y_train"]), "peak_memory_mb": sampling.peak_memory_mb(),
    }


def plot(results, anomaly, out):
    viz.apply_style()
    plt = viz.plt
    models = sorted({r["model"] for r in results})
    fig, axes = plt.subplots(1, len(models), figsize=(5.4 * len(models), 4.2), sharey=True, squeeze=False)
    names = ATTACK_CLASSES[::-1]
    for ax, m in zip(axes[0], models):
        rows = {r["held_out"]: r for r in results if r["model"] == m}
        y = np.arange(len(names))
        a = np.array([rows[n]["held_out_predicted_as_other_attack"] for n in names])
        nrm = np.array([rows[n]["held_out_predicted_as_normal"] for n in names])
        ax.barh(y, a, color=viz.ATTACK_COLOUR, height=0.6, label="flagged as another attack")
        ax.barh(y, nrm, left=a, color=viz.NORMAL_COLOUR, height=0.6, label="classified Normal (missed)")
        for yy, n in zip(y, names):
            ax.plot(anomaly["autoencoder"][n], yy, marker="D", ms=6, color=viz.INK, ls="none",
                    label="Autoencoder detection rate" if yy == 0 else None)
        ax.set_yticks(y, names)
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of the unseen class's test records")
        ax.set_title({"rf": "Random Forest", "mlp": "MLP"}[m] + ": class removed from training")
        ax.grid(axis="y", visible=False)
    axes[0, -1].legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    return viz.save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--models", nargs="*", default=["rf", "mlp"])
    args = ap.parse_args()
    study_path = store.metrics_path("A", "sampling_study")
    if not study_path.exists() and not args.quick:
        subprocess.run([sys.executable, "-u", str(Path(__file__).with_name("sampling_study.py"))], check=True)
    frac = 1.0 if args.quick else store.load_metrics("A", "sampling_study")["chosen_fraction"]
    # raw records with exactly the split used everywhere else (checked against the saved labels)
    full, test = load_track_a("train"), load_track_a("test")
    tr_idx, va_idx = track_a_validation_split(full, seed=0)
    raw = {"train": full.iloc[tr_idx].reset_index(drop=True), "val": full.iloc[va_idx].reset_index(drop=True),
           "test": test}
    del full
    saved = store.load_track_a_arrays()
    for s_ in ("train", "val", "test"):
        raw[f"y_{s_}"] = encode_targets(raw[s_])[0].astype(np.int64)
        assert np.array_equal(raw[f"y_{s_}"], saved[f"y_{s_}"]), f"{s_} split differs from the saved arrays"
    del saved
    sample = sampling.stratified_subsample(raw["y_train"], frac, seed=0)
    training_data = {"track": "A", "full_or_reduced": "reduced" if frac < 1 else "full",
                     "train_fraction": frac, "seed": 0, "floor_per_class": sampling.TRACK_A_FLOOR,
                     "strategy": "minority-preserving stratified sample of the training split, the same "
                                 "for every fold; validation and official test sets complete",
                     "distribution": sampling.distribution_table(raw["y_train"],
                                                                 {"sample": raw["y_train"][sample]})}
    print(f"training sample: {len(sample):,} of {len(raw['y_train']):,} records ({frac:.0%})", flush=True)
    rf_m = store.load_metrics("A", "rf_multiclass")
    rf_cfg = {k: rf_m["config"][k] for k in ("min_samples_leaf", "max_features", "imbalance")}
    rf_n = 40 if args.quick else rf_m["config"]["n_estimators"]
    mlp_m = store.load_metrics("A", "mlp")
    if args.quick:
        mlp_m = {**mlp_m, "train_config": {**mlp_m["train_config"], "max_epochs": 2}}
    classes = ATTACK_CLASSES[:2] if args.quick else ATTACK_CLASSES
    results = []
    t0 = time.time()
    for held in classes:
        for model in args.models:
            key = store.run_key(held=held, model=model, rf=[rf_cfg, rf_n],
                                mlp=[mlp_m["params"], mlp_m["train_config"], mlp_m["imbalance"]],
                                reject=REJECT_QUANTILE, train_fraction=frac, floor=sampling.TRACK_A_FLOOR,
                                preprocessing="refit per fold without the held-out class")
            r = None if args.quick else store.load_run("A", "loco", key)
            if r is None:
                r = run(raw, sample, held, model, rf_cfg, rf_n, mlp_m)
                if not args.quick:
                    store.save_run("A", "loco", key, r)      # resumable: one file per (class, model)
                store.release_memory()
            r = {k: v for k, v in r.items() if k != "saved_utc"}
            results.append(r)
            print(f"  {held:15s} {model:4s} as-normal {r['held_out_predicted_as_normal']:.2f} "
                  f"as-other-attack {r['held_out_predicted_as_other_attack']:.2f} "
                  f"unknown {r['held_out_rejected_as_unknown']:.2f} (known rejected "
                  f"{r['known_classes_rejected_as_unknown']:.2f}) -> {r['most_common_wrong_attack']}",
                  flush=True)
    # anomaly detectors never saw any attack: their per-category detection rates (seed 0)
    anomaly = {}
    for name in ("autoencoder", "iforest"):
        m = store.load_metrics("A", name)["test"]["by_category"]
        anomaly[name] = {k: v["flagged_fraction"] for k, v in m.items() if k != "Normal"}
    store.save_metrics("A", "leave_one_class_out", {
        "experiment": "leave one attack class out of training (Track A)",
        "rejection_rule": f"max softmax prob below the {REJECT_QUANTILE:.0%} quantile on known-class validation data",
        "preprocessing": "Track A preprocessor refitted for every fold on that fold's training records "
                         "(held-out class excluded)",
        "training_data": training_data,
        "anomaly_detectors_note": "autoencoder and Isolation Forest rates are from the full-data models "
                                  "(trained on normal traffic only, so every attack class is unseen)",
        "results": results, "anomaly_detector_detection_rate_by_category": anomaly,
        "seconds": round(time.time() - t0)})
    if not args.quick:
        plot(results, anomaly, ensure_dir(store.FIG_DIR) / "leave_one_class_out.png")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
