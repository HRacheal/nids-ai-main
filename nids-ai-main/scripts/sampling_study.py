#!/usr/bin/env python3
"""Choose the Track A training-data fraction for the remaining experiments (validation only).

For each candidate fraction (70 %, 50 %, 30 %) a minority-preserving
stratified sample of the Track A TRAINING split is drawn
(deepshield.data.sampling.stratified_subsample, seed 0, classes with fewer
than 2,000 training records kept whole). The already selected Random Forest
and MLP configurations are trained on it with seed 0 and scored on the
unchanged VALIDATION split. The reference is the same configuration trained
on all training data (seed 0), read from the saved metrics.

Decision rule (fixed before the study was run; the test set is never used):
  use 50 % (the project owner's preferred setting) if BOTH models lose at most
  MAX_DROP validation macro-F1 against full data; otherwise 70 % under the same
  rule; otherwise full data. The 30 % row is reported for information.

The fraction chosen here is used by scripts/leave_one_class_out.py.
Output: artifacts/metrics/track_a/sampling_study.json (runs cached in runs/sampling_study).

Usage:  .venv\\Scripts\\python.exe scripts\\sampling_study.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepshield.data import sampling  # noqa: E402
from deepshield.data.imbalance import class_weights, random_oversample  # noqa: E402
from deepshield.data.schema import CLASS_ORDER  # noqa: E402
from deepshield.evaluation import metrics as M  # noqa: E402
from deepshield.training import store  # noqa: E402
from deepshield.training.trainer import TrainConfig, array_batches, fit, predict  # noqa: E402

FRACTIONS = [0.7, 0.5, 0.3]
PREFERRED = 0.5
MAX_DROP = 0.02
K = len(CLASS_ORDER)


def train_rf(X, y, cfg, seed=0):
    from sklearn.ensemble import RandomForestClassifier
    idx = random_oversample(y, 2000, seed=seed) if cfg["imbalance"] == "oversample" else np.arange(len(y))
    rf = RandomForestClassifier(n_estimators=cfg["n_estimators"], min_samples_leaf=cfg["min_samples_leaf"],
                                max_features=cfg["max_features"], n_jobs=4, random_state=seed,
                                class_weight="balanced_subsample"
                                if cfg["imbalance"] == "balanced_subsample" else None)
    rf.fit(X[idx], y[idx])
    return rf.predict_proba


def train_mlp(X, y, Xv, yv, m, resume):
    tc = {k: v for k, v in m["train_config"].items() if k in TrainConfig.__dataclass_fields__}
    cfg = TrainConfig(**{**tc, "seed": 0})
    idx, w = np.arange(len(y)), np.ones(K)
    if m["imbalance"] == "oversample":
        idx = random_oversample(y, 2000, seed=0)
    else:
        w = class_weights(y, K, m["imbalance"])
    loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    res = fit("MLP", m["params"], cfg, array_batches(X[idx], y[idx], cfg.batch_size, True),
              array_batches(Xv, yv, 4096, False), loss, resume_path=resume)
    return lambda Z: predict(res.model, Z), res.history


def main():
    t_all = time.time()
    d = store.load_track_a_arrays()
    rf_m, mlp_m = store.load_metrics("A", "rf_multiclass"), store.load_metrics("A", "mlp")
    ref = {"rf": next(r["val_macro_f1"] for r in rf_m["seed_summary"]["per_seed"] if r["seed"] == 0),
           "mlp": next(r["val_macro_f1"] for r in mlp_m["seed_summary"]["per_seed"] if r["seed"] == 0)}
    samples = {f: sampling.stratified_subsample(d["y_train"], f, seed=0) for f in FRACTIONS}
    dist = sampling.distribution_table(d["y_train"], {f"{f:.0%}": d["y_train"][i] for f, i in samples.items()})
    rows = []
    for f in FRACTIONS:
        idx = samples[f]
        X, y = d["X_train"][idx], d["y_train"][idx]
        for model in ("rf", "mlp"):
            key = store.run_key(study="sampling", model=model, fraction=f, floor=sampling.TRACK_A_FLOOR,
                                rf=rf_m["config"], mlp=[mlp_m["params"], mlp_m["train_config"], mlp_m["imbalance"]])
            r = store.load_run("A", "sampling_study", key)
            print(f"  {model} {f:.0%} ({len(idx):,} records)" + ("  [cached]" if r else ""), flush=True)
            if r is None:
                t0 = time.time()
                hist = None
                if model == "rf":
                    f_pred = train_rf(X, y, rf_m["config"])
                else:
                    f_pred, hist = train_mlp(X, y, d["X_val"], d["y_val"], mlp_m,
                                             store.epoch_checkpoint("A", "sampling_study", key))
                secs = time.time() - t0
                pv = f_pred(d["X_val"])
                rep = M.multiclass_report(d["y_val"], pv.argmax(1), pv)
                r = {"model": model, "fraction": f, "n_train": int(len(idx)),
                     "val_macro_f1": rep["macro_f1"], "val_accuracy": rep["accuracy"],
                     "val_macro_precision": rep.get("macro_precision"), "val_macro_recall": rep.get("macro_recall"),
                     "val_per_class_f1": {c: rep["per_class"][c].get("f1") for c in CLASS_ORDER},
                     "fit_seconds": round(secs, 1), "peak_memory_mb": sampling.peak_memory_mb(),
                     "epochs_run": hist["epochs_run"] if hist else None}
                store.save_run("A", "sampling_study", key, r)
                store.clear_epoch_checkpoint("A", "sampling_study", key)
                store.release_memory()
            r = {k: v for k, v in r.items() if k != "saved_utc"}
            r["drop_vs_full"] = ref[model] - r["val_macro_f1"]
            rows.append(r)
            print(f"    val macro-F1 {r['val_macro_f1']:.4f} (full {ref[model]:.4f}, drop {r['drop_vs_full']:+.4f}), "
                  f"{r['fit_seconds']:.0f}s", flush=True)

    def ok(f):
        return all(r["drop_vs_full"] <= MAX_DROP for r in rows if r["fraction"] == f)
    chosen = PREFERRED if ok(PREFERRED) else (0.7 if ok(0.7) else 1.0)
    store.save_metrics("A", "sampling_study", {
        "purpose": "choose the Track A training fraction for leave-one-class-out (validation data only)",
        "strategy": "per-class random sample of the training split, seed 0; classes with fewer than "
                    f"{sampling.TRACK_A_FLOOR} training records kept whole",
        "decision_rule": f"use {PREFERRED:.0%} if RF and MLP both lose <= {MAX_DROP} validation macro-F1 "
                         "versus full training data (same configuration, seed 0); else 70 % under the same "
                         "rule; else full data. Test data not used.",
        "reference_full_data_val_macro_f1": ref, "results": rows, "chosen_fraction": chosen,
        "class_distribution": dist, "seconds": round(time.time() - t_all)})
    print(f"chosen training fraction: {chosen:.0%}")


if __name__ == "__main__":
    main()
