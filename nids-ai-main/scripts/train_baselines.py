#!/usr/bin/env python3
"""Phase 3: traditional machine-learning baselines on Track A.

1. Random Forest (multiclass): hyperparameters and imbalance strategy are
   chosen on the VALIDATION fold by macro-F1; the final configuration is
   trained with seeds 0, 1, 2 and evaluated once on the official test set.
2. Random Forest (binary attack/normal): same configuration, binary target.
3. Isolation Forest (anomaly detection): trained on NORMAL training
   records only, so it never sees an attack. Its anomaly score is evaluated
   against the labels; the alert threshold is the 95th percentile of the
   validation NORMAL scores (a 5 % target false-positive rate), so no attack
   label is used to set it. Isolation Forest is not a classifier and is not
   used for multiclass prediction.

Usage:  .venv\\Scripts\\python.exe scripts\\train_baselines.py [--quick]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepshield.config import PROJECT_ROOT, ensure_dir, load_config  # noqa: E402
from deepshield.data.imbalance import random_oversample  # noqa: E402
from deepshield.data.schema import CLASS_ORDER  # noqa: E402
from deepshield.evaluation import metrics as M  # noqa: E402
from deepshield.evaluation import plots  # noqa: E402
from deepshield.training import store  # noqa: E402

N_JOBS = 4
IF_FPR_TARGET = 0.05


def rf_grid(quick: bool):
    grid = {"min_samples_leaf": [1, 3], "max_features": ["sqrt", 0.3],
            "imbalance": ["none", "balanced_subsample", "oversample"]}
    if quick:
        grid = {"min_samples_leaf": [3], "max_features": ["sqrt"], "imbalance": ["none"]}
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*grid.values())]


def fit_rf(X, y, cfg, seed, n_estimators):
    idx = np.arange(len(y))
    if cfg["imbalance"] == "oversample":
        idx = random_oversample(y, min_count=2000, seed=seed)
    rf = RandomForestClassifier(
        n_estimators=n_estimators, min_samples_leaf=cfg["min_samples_leaf"],
        max_features=cfg["max_features"], n_jobs=N_JOBS, random_state=seed,
        class_weight="balanced_subsample" if cfg["imbalance"] == "balanced_subsample" else None)
    t0 = time.time()
    rf.fit(X[idx], y[idx])
    return rf, time.time() - t0


def timed_predict(model, X):
    t0 = time.time()
    p = model.predict_proba(X)
    return p, (time.time() - t0) / len(X) * 1e6     # microseconds per record


def random_forest(d, quick, seeds, figs):
    n_est = 60 if quick else 150
    study = []
    for cfg in rf_grid(quick):
        rf, secs = fit_rf(d["X_train"], d["y_train"], cfg, 0, n_est)
        pv = rf.predict_proba(d["X_val"])
        rep = M.multiclass_report(d["y_val"], pv.argmax(1), pv)
        study.append({**cfg, "val_macro_f1": rep["macro_f1"], "val_accuracy": rep["accuracy"],
                      "fit_seconds": round(secs, 1)})
        print(f"  RF {cfg} -> val macro-F1 {rep['macro_f1']:.4f} ({secs:.0f}s)", flush=True)
        del rf
    best = max(study, key=lambda r: r["val_macro_f1"])
    cfg = {k: best[k] for k in ("min_samples_leaf", "max_features", "imbalance")}
    print("  RF selected:", cfg)

    runs = []
    for seed in seeds:
        rf, secs = fit_rf(d["X_train"], d["y_train"], cfg, seed, n_est)
        pv, _ = timed_predict(rf, d["X_val"])
        pt, us = timed_predict(rf, d["X_test"])
        yp = pt.argmax(1)
        rep = M.multiclass_report(d["y_test"], yp, pt)
        rep_val = M.multiclass_report(d["y_val"], pv.argmax(1), pv)
        runs.append({"seed": seed, "fit_seconds": secs, "predict_us_per_record": us,
                     "test": rep, "val_macro_f1": rep_val["macro_f1"]})
        store.save_predictions("A", f"rf_multiclass_s{seed}", "test", proba=pt.astype(np.float32))
        store.save_predictions("A", f"rf_multiclass_s{seed}", "val", proba=pv.astype(np.float32))
        if seed == seeds[0]:
            joblib.dump(rf, store.model_dir("A", "rf_multiclass") / "model.joblib", compress=3)
            main_rep, main_pred = rep, yp
            ov = d["test_overlap"]
            main_rep["non_overlapping_test"] = M.multiclass_report(
                d["y_test"][~ov], yp[~ov], pt[~ov])
            main_rep["bootstrap_macro_f1"] = M.bootstrap_ci(d["y_test"], yp)
            imp = sorted(zip(d["feature_names"], rf.feature_importances_), key=lambda t: -t[1])[:25]
            plots.confusion_matrix_plot(rep["confusion_matrix"], "Random Forest, Track A test (seed 0)",
                                        figs / "cm_rf_multiclass.png")
        del rf
        print(f"  RF seed {seed}: test macro-F1 {rep['macro_f1']:.4f}, acc {rep['accuracy']:.4f}", flush=True)
    model_size = (store.model_dir("A", "rf_multiclass") / "model.joblib").stat().st_size
    store.save_metrics("A", "rf_multiclass", {
        "model": "Random Forest", "task": "multiclass", "track": "A",
        "config": {**cfg, "n_estimators": n_est, "criterion": "gini"},
        "selection": "validation macro-F1 over grid", "grid_results": study,
        "seeds": seeds, "test": main_rep,
        "seed_summary": _seed_summary(runs),
        "fit_seconds": float(np.mean([r["fit_seconds"] for r in runs])),
        "predict_us_per_record": float(np.mean([r["predict_us_per_record"] for r in runs])),
        "model_bytes": model_size, "top_feature_importance_mdi": imp,
    })
    return cfg, n_est, study, main_pred


def random_forest_binary(d, cfg, n_est, seeds):
    runs = []
    bcfg = dict(cfg)
    for seed in seeds:
        rf, secs = fit_rf(d["X_train"], d["yb_train"], bcfg, seed, n_est)
        pv = rf.predict_proba(d["X_val"])[:, 1]
        pt, us = timed_predict(rf, d["X_test"])
        s = pt[:, 1]
        rep = M.binary_report(d["yb_test"], (s >= 0.5).astype(int), s)
        rep["by_category"] = M.detection_by_category(d["y_test"], s >= 0.5)
        runs.append({"seed": seed, "fit_seconds": secs, "predict_us_per_record": us, "test": rep})
        store.save_predictions("A", f"rf_binary_s{seed}", "test", score=s.astype(np.float32))
        store.save_predictions("A", f"rf_binary_s{seed}", "val", score=pv.astype(np.float32))
        if seed == seeds[0]:
            joblib.dump(rf, store.model_dir("A", "rf_binary") / "model.joblib", compress=3)
            main = rep
        del rf
        print(f"  RF-binary seed {seed}: F1 {rep['f1']:.4f} ROC-AUC {rep['roc_auc']:.4f}", flush=True)
    store.save_metrics("A", "rf_binary", {
        "model": "Random Forest", "task": "binary", "track": "A",
        "config": {**bcfg, "n_estimators": n_est, "threshold": 0.5},
        "seeds": seeds, "test": main,
        "seed_f1": {"mean": float(np.mean([r["test"]["f1"] for r in runs])),
                    "std": float(np.std([r["test"]["f1"] for r in runs]))},
        "fit_seconds": float(np.mean([r["fit_seconds"] for r in runs])),
        "predict_us_per_record": float(np.mean([r["predict_us_per_record"] for r in runs])),
    })


def isolation_forest(d, quick, seeds, figs):
    normal_tr = d["X_train"][d["y_train"] == 0]
    grid = [{"n_estimators": n, "max_samples": m, "max_features": f}
            for n in ([100] if quick else [100, 300])
            for m in ([256] if quick else [256, 2048])
            for f in ([1.0] if quick else [1.0, 0.5])]
    study = []
    for cfg in grid:
        iso = IsolationForest(**cfg, random_state=0, n_jobs=N_JOBS).fit(normal_tr)
        sv = -iso.score_samples(d["X_val"])             # higher = more anomalous
        auc = roc_auc_score(d["yb_val"], sv)
        study.append({**cfg, "val_roc_auc": auc})
        print(f"  IF {cfg} -> val ROC-AUC {auc:.4f}", flush=True)
    best = max(study, key=lambda r: r["val_roc_auc"])
    cfg = {k: best[k] for k in ("n_estimators", "max_samples", "max_features")}
    runs = []
    for seed in seeds:
        t0 = time.time()
        iso = IsolationForest(**cfg, random_state=seed, n_jobs=N_JOBS).fit(normal_tr)
        secs = time.time() - t0
        sv = -iso.score_samples(d["X_val"])
        thr = float(np.quantile(sv[d["yb_val"] == 0], 1 - IF_FPR_TARGET))
        t0 = time.time()
        st = -iso.score_samples(d["X_test"])
        us = (time.time() - t0) / len(st) * 1e6
        flag = st >= thr
        rep = M.binary_report(d["yb_test"], flag.astype(int), st)
        rep["by_category"] = M.detection_by_category(d["y_test"], flag)
        rep["threshold"] = thr
        runs.append({"seed": seed, "fit_seconds": secs, "predict_us_per_record": us, "test": rep})
        store.save_predictions("A", f"iforest_s{seed}", "test", score=st.astype(np.float32))
        store.save_predictions("A", f"iforest_s{seed}", "val", score=sv.astype(np.float32))
        if seed == seeds[0]:
            joblib.dump({"model": iso, "threshold": thr}, store.model_dir("A", "iforest") / "model.joblib",
                        compress=3)
            main = rep
            plots.score_hist_plot(st[d["yb_test"] == 0], st[d["yb_test"] == 1], thr,
                                  "Isolation Forest anomaly scores, Track A test",
                                  "anomaly score (-score_samples)", figs / "iforest_scores.png")
        print(f"  IF seed {seed}: ROC-AUC {rep['roc_auc']:.4f} F1 {rep['f1']:.4f} "
              f"FPR {rep['false_positive_rate']:.3f}", flush=True)
    store.save_metrics("A", "iforest", {
        "model": "Isolation Forest", "task": "binary anomaly detection", "track": "A",
        "trained_on": "normal training records only",
        "config": cfg, "threshold_rule": f"{1 - IF_FPR_TARGET:.0%} quantile of validation normal scores",
        "selection": "validation ROC-AUC", "grid_results": study, "seeds": seeds, "test": main,
        "seed_roc_auc": {"mean": float(np.mean([r["test"]["roc_auc"] for r in runs])),
                         "std": float(np.std([r["test"]["roc_auc"] for r in runs]))},
        "fit_seconds": float(np.mean([r["fit_seconds"] for r in runs])),
        "predict_us_per_record": float(np.mean([r["predict_us_per_record"] for r in runs])),
    })


def _seed_summary(runs):
    f = [r["test"]["macro_f1"] for r in runs]
    a = [r["test"]["accuracy"] for r in runs]
    return {"macro_f1_mean": float(np.mean(f)), "macro_f1_std": float(np.std(f)),
            "accuracy_mean": float(np.mean(a)), "accuracy_std": float(np.std(a)),
            "per_seed": [{"seed": r["seed"], "macro_f1": r["test"]["macro_f1"],
                          "val_macro_f1": r["val_macro_f1"]} for r in runs]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny grid, for smoke tests")
    ap.add_argument("--only", choices=["rf", "rf_binary", "iforest"], nargs="*")
    args = ap.parse_args()
    seeds = load_config()["project"]["seeds"][: (1 if args.quick else 3)]
    figs = ensure_dir(store.FIG_DIR)
    d = store.load_track_a_arrays()
    only = set(args.only or ["rf", "rf_binary", "iforest"])
    t0 = time.time()
    study_path = ensure_dir(PROJECT_ROOT / "reports" / "preprocessing") / "imbalance_study.json"
    study_all = json.loads(study_path.read_text(encoding="utf-8")) if study_path.exists() else {}
    cfg = n_est = None
    if "rf" in only:
        print("Random Forest (multiclass)")
        cfg, n_est, study, _ = random_forest(d, args.quick, seeds, figs)
        study_all["random_forest"] = study
        study_path.write_text(json.dumps(study_all, indent=2), encoding="utf-8")
    if "rf_binary" in only:
        if cfg is None:
            m = store.load_metrics("A", "rf_multiclass")
            cfg = {k: m["config"][k] for k in ("min_samples_leaf", "max_features", "imbalance")}
            n_est = m["config"]["n_estimators"]
        print("Random Forest (binary)")
        random_forest_binary(d, cfg, n_est, seeds)
    if "iforest" in only:
        print("Isolation Forest")
        isolation_forest(d, args.quick, seeds, figs)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
