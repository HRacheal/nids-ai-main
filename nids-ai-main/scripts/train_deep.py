#!/usr/bin/env python3
"""Phase 4 (Track A): MLP, 1D-CNN (+ random-order ablation) and Autoencoder.

Selection protocol (identical for every network):
  1. hyperparameter search with seed 0, scored on the VALIDATION fold
     (classifiers: macro-F1; autoencoder: ROC-AUC of reconstruction error);
  2. the chosen configuration is retrained with seeds 0, 1, 2;
  3. each seed is evaluated ONCE on the official test set.
Early stopping always uses validation loss. The test set never influences
any choice.

Reduced search (memory/compute constraint, documented in docs/models.md):
the laptop used has 7.7 GB RAM shared with other applications, and a first
full-grid run was killed for low system memory. The grids below keep small
and medium configurations and drop the largest (3-layer 512-unit MLP,
batch-size study, the 64/128-channel kernel-5 CNN). Every finished run is
cached in artifacts/metrics/track_a/runs/ immediately, so an interrupted job
resumes where it stopped.

Usage:  .venv\\Scripts\\python.exe scripts\\train_deep.py [--only mlp cnn ae] [--quick]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepshield.config import PROJECT_ROOT, ensure_dir, load_config  # noqa: E402
from deepshield.data.imbalance import class_weights, random_oversample  # noqa: E402
from deepshield.data.schema import CLASS_ORDER  # noqa: E402
from deepshield.evaluation import metrics as M  # noqa: E402
from deepshield.evaluation import plots  # noqa: E402
from deepshield.models.networks import count_parameters  # noqa: E402
from deepshield.training import store  # noqa: E402
from deepshield.training.trainer import (TrainConfig, array_batches, fit,  # noqa: E402
                                         predict, save_checkpoint)

K = len(CLASS_ORDER)
AE_FPR_TARGET = 0.05
REDUCED_NOTE = ("reduced search because of limited memory (7.7 GB shared); "
                "see docs/models.md for the dropped configurations")
STUDY = PROJECT_ROOT / "reports" / "preprocessing" / "imbalance_study.json"


def classifier_run(d, arch, params, cfg: TrainConfig, imbalance: str, resume=None):
    idx = np.arange(len(d["y_train"]))
    if imbalance == "oversample":
        idx = random_oversample(d["y_train"], 2000, seed=cfg.seed)
        w = np.ones(K)
    else:
        w = class_weights(d["y_train"], K, imbalance)
    Xtr, ytr = d["X_train"][idx], d["y_train"][idx]
    loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    res = fit(arch, params, cfg, array_batches(Xtr, ytr, cfg.batch_size, True),
              array_batches(d["X_val"], d["y_val"], 4096, False), loss, resume_path=resume)
    pv = predict(res.model, d["X_val"])
    return res, pv, M.multiclass_report(d["y_val"], pv.argmax(1), pv)


def search(d, arch, candidates, base_cfg, label, study_rows):
    best = None
    for params, cfg_over, imb in candidates:
        cfg = TrainConfig(**{**base_cfg.__dict__, **cfg_over})
        key = store.run_key(kind="search", arch=arch, params=params, cfg=cfg.__dict__, imbalance=imb)
        row = store.load_run("A", f"search_{arch.lower()}", key)
        if row is None:
            print(f"  {label} {params} {cfg_over} imbalance={imb}", flush=True)
            res, _, rep = classifier_run(d, arch, params, cfg, imb,
                                         store.epoch_checkpoint("A", f"search_{arch.lower()}", key))
            row = {"params": params, "train_config": cfg_over, "imbalance": imb,
                   "val_macro_f1": rep["macro_f1"], "val_accuracy": rep["accuracy"],
                   "best_epoch": res.history["best_epoch"], "seconds": round(res.seconds, 1),
                   "seed": cfg.seed, "history": res.history}
            store.save_run("A", f"search_{arch.lower()}", key, row)
            store.clear_epoch_checkpoint("A", f"search_{arch.lower()}", key)
            del res
            store.release_memory()
        else:
            print(f"  {label} {params} {cfg_over} imbalance={imb}  [cached]", flush=True)
        row = {k: v for k, v in row.items() if k not in ("history", "saved_utc")}
        study_rows.append(row)
        print(f"    -> val macro-F1 {row['val_macro_f1']:.4f}", flush=True)
        if best is None or row["val_macro_f1"] > best["val_macro_f1"]:
            best = row
    return best


def final_classifier(d, name, title, arch, params, cfg_over, imb, base_cfg, seeds,
                     search_rows, figs, extra=None):
    runs, histories = [], {}
    for seed in seeds:
        cfg = TrainConfig(**{**base_cfg.__dict__, **cfg_over, "seed": seed})
        key = store.run_key(kind="final", name=name, arch=arch, params=params, cfg=cfg.__dict__,
                            imbalance=imb, first=seed == seeds[0])
        run = store.load_run("A", name, key)
        if run is None:
            res, pv, rep_val = classifier_run(d, arch, params, cfg, imb,
                                              store.epoch_checkpoint("A", name, key))
            t0 = time.time()
            pt = predict(res.model, d["X_test"])
            us = (time.time() - t0) / len(pt) * 1e6
            store.save_predictions("A", f"{name}_s{seed}", "test", proba=pt.astype(np.float32))
            store.save_predictions("A", f"{name}_s{seed}", "val", proba=pv.astype(np.float32))
            if seed == seeds[0]:
                save_checkpoint(store.model_dir("A", name) / "model.pt", res.model, arch, params, cfg,
                                {"imbalance": imb, "task": "multiclass", "classes": CLASS_ORDER})
            run = {"seed": seed, "val_macro_f1": rep_val["macro_f1"], "seconds": res.seconds, "us": us,
                   "history": res.history, "n_parameters": count_parameters(res.model)}
            store.save_run("A", name, key, run)          # predictions + model are saved first
            store.clear_epoch_checkpoint("A", name, key)
            del res
            store.release_memory()
        else:
            print(f"  {name} seed {seed} [cached]", flush=True)
            pt = store.load_predictions("A", f"{name}_s{seed}", "test")["proba"]
        yp = pt.argmax(1)
        rep = M.multiclass_report(d["y_test"], yp, pt)
        runs.append({**run, "test": rep})
        histories[f"seed {seed}"] = run["history"]
        if seed == seeds[0]:
            main = rep
            ov = d["test_overlap"]
            main["non_overlapping_test"] = M.multiclass_report(d["y_test"][~ov], yp[~ov], pt[~ov])
            main["bootstrap_macro_f1"] = M.bootstrap_ci(d["y_test"], yp)
            n_params = run["n_parameters"]
            plots.confusion_matrix_plot(rep["confusion_matrix"], f"{title}, Track A test (seed {seed})",
                                        figs / f"cm_{name}.png")
        print(f"  {name} seed {seed}: test macro-F1 {rep['macro_f1']:.4f} acc {rep['accuracy']:.4f}",
              flush=True)
    plots.history_plot(histories, f"{title}: training and validation loss", figs / f"history_{name}.png")
    f1s = [r["test"]["macro_f1"] for r in runs]
    store.save_metrics("A", name, {
        "model": title, "task": "multiclass", "track": "A", "arch": arch,
        "params": params, "train_config": {**base_cfg.__dict__, **cfg_over}, "imbalance": imb,
        "selection": "validation macro-F1 (seed 0)", "search": search_rows,
        "seeds": seeds, "test": main, "n_parameters": n_params,
        "seed_summary": {"macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
                         "accuracy_mean": float(np.mean([r["test"]["accuracy"] for r in runs])),
                         "per_seed": [{"seed": r["seed"], "macro_f1": r["test"]["macro_f1"],
                                       "val_macro_f1": r["val_macro_f1"]} for r in runs]},
        "histories": {f"seed {r['seed']}": r["history"] for r in runs},
        "fit_seconds": float(np.mean([r["seconds"] for r in runs])),
        "predict_us_per_record": float(np.mean([r["us"] for r in runs])),
        "search_note": REDUCED_NOTE,
        **(extra or {}),
    })


def run_mlp(d, quick, seeds, figs, base):
    rows = []
    n_in = d["X_train"].shape[1]
    # reduced grid: small and medium sizes; the 3-layer 512-unit network is dropped
    grid = ([([256, 128], 0.3, 1e-3)] if quick else
            [([128, 64], 0.2, 1e-3), ([256, 128], 0.2, 1e-3),
             ([256, 128], 0.2, 3e-4), ([256, 128], 0.4, 1e-3)])
    cands = [({"n_in": n_in, "n_out": K, "hidden": h, "dropout": p}, {"lr": lr}, "sqrt")
             for h, p, lr in grid]
    best = search(d, "MLP", cands, base, "MLP", rows)
    imb_rows = []
    if not quick:   # imbalance strategy study with the best architecture
        for imb in ["none", "balanced", "oversample"]:
            search(d, "MLP", [(best["params"], best["train_config"], imb)], base, "MLP", imb_rows)
        imb_rows.append(best)
        best = max(imb_rows, key=lambda r: r["val_macro_f1"])
        rows += imb_rows[:-1]       # (batch-size study dropped in the reduced search)
        study = json.loads(STUDY.read_text(encoding="utf-8")) if STUDY.exists() else {}
        study["mlp"] = imb_rows
        STUDY.write_text(json.dumps(study, indent=2), encoding="utf-8")
    final_classifier(d, "mlp", "MLP", "MLP", best["params"], best["train_config"],
                     best["imbalance"], base, seeds, rows, figs)
    return best


def run_cnn(d, quick, seeds, figs, base, mlp_best):
    rows = []
    n_num = sum(1 for f in d["feature_names"] if "=" not in f)
    n_cat = d["X_train"].shape[1] - n_num
    common = {"n_numeric": n_num, "n_categorical": n_cat, "n_out": K, "dropout": 0.3}
    # reduced grid: the 64/128-channel kernel-5 network is dropped
    grid = [([32, 64], 3)] if quick else [([32, 64], 3), ([32, 64], 5), ([64, 128], 3)]
    cfg_over = dict(mlp_best["train_config"])   # same lr / batch size as the selected MLP
    cands = [({**common, "channels": c, "kernel": k}, cfg_over, mlp_best["imbalance"])
             for c, k in grid]
    best = search(d, "CNN1D", cands, base, "CNN", rows)
    final_classifier(d, "cnn", "1D-CNN", "CNN1D", best["params"], best["train_config"],
                     best["imbalance"], base, seeds, rows, figs)
    # ablation: same architecture and training, numeric features in random order
    perm = np.random.default_rng(1234).permutation(n_num).tolist()
    params = {**best["params"], "feature_perm": perm}
    final_classifier(d, "cnn_random_order", "1D-CNN (random feature order)", "CNN1D", params,
                     best["train_config"], best["imbalance"], base, seeds, [], figs,
                     extra={"ablation_of": "cnn", "permutation_seed": 1234})


def run_autoencoder(d, quick, seeds, figs, base):
    n_in = d["X_train"].shape[1]
    Xn_tr = d["X_train"][d["y_train"] == 0]
    Xn_val = d["X_val"][d["y_val"] == 0]
    mse = torch.nn.MSELoss()
    fwd = lambda m, b: (m(b[0]), b[0])  # noqa: E731
    score_fn = lambda m, xb: m.score(xb)  # noqa: E731
    study = []
    best = None
    for bott in ([16] if quick else [8, 16, 32]):
        params = {"n_in": n_in, "hidden": [128, 64], "bottleneck": bott}
        cfg = TrainConfig(**{**base.__dict__, "seed": 0})
        key = store.run_key(kind="search", arch="Autoencoder", params=params, cfg=cfg.__dict__)
        row = store.load_run("A", "search_autoencoder", key)
        print(f"  AE bottleneck {bott}" + ("  [cached]" if row else ""), flush=True)
        if row is None:
            res = fit("Autoencoder", params, cfg, array_batches(Xn_tr, None, cfg.batch_size, True),
                      array_batches(Xn_val, None, 4096, False), mse, forward=fwd,
                      resume_path=store.epoch_checkpoint("A", "search_autoencoder", key))
            sv = predict(res.model, d["X_val"], fn=score_fn)
            row = {"params": params, "val_roc_auc": float(roc_auc_score(d["yb_val"], sv)),
                   "best_epoch": res.history["best_epoch"], "val_normal_mse": res.history["best_val_loss"],
                   "seconds": round(res.seconds, 1), "seed": 0}
            store.save_run("A", "search_autoencoder", key, row)
            store.clear_epoch_checkpoint("A", "search_autoencoder", key)
            del res
            store.release_memory()
        row = {k: v for k, v in row.items() if k != "saved_utc"}
        study.append(row)
        auc = row["val_roc_auc"]
        print(f"    -> val ROC-AUC {auc:.4f}", flush=True)
        if best is None or auc > best["val_roc_auc"]:
            best = study[-1]
    runs, histories = [], {}
    for seed in seeds:
        cfg = TrainConfig(**{**base.__dict__, "seed": seed})
        key = store.run_key(kind="final", name="autoencoder", params=best["params"], cfg=cfg.__dict__,
                            first=seed == seeds[0])
        run = store.load_run("A", "autoencoder", key)
        if run is None:
            res = fit("Autoencoder", best["params"], cfg, array_batches(Xn_tr, None, cfg.batch_size, True),
                      array_batches(Xn_val, None, 4096, False), mse, forward=fwd,
                      resume_path=store.epoch_checkpoint("A", "autoencoder", key))
            sv = predict(res.model, d["X_val"], fn=score_fn)
            thr = float(np.quantile(sv[d["yb_val"] == 0], 1 - AE_FPR_TARGET))
            t0 = time.time()
            st = predict(res.model, d["X_test"], fn=score_fn)
            us = (time.time() - t0) / len(st) * 1e6
            store.save_predictions("A", f"autoencoder_s{seed}", "test", score=st.astype(np.float32))
            store.save_predictions("A", f"autoencoder_s{seed}", "val", score=sv.astype(np.float32))
            if seed == seeds[0]:
                save_checkpoint(store.model_dir("A", "autoencoder") / "model.pt", res.model, "Autoencoder",
                                best["params"], cfg, {"threshold": thr, "task": "anomaly",
                                                      "threshold_rule": "95th percentile of validation normal MSE"})
            run = {"seed": seed, "threshold": thr, "seconds": res.seconds, "us": us, "history": res.history,
                   "n_parameters": count_parameters(res.model)}
            store.save_run("A", "autoencoder", key, run)
            store.clear_epoch_checkpoint("A", "autoencoder", key)
            del res
            store.release_memory()
        else:
            print(f"  autoencoder seed {seed} [cached]", flush=True)
            st = store.load_predictions("A", f"autoencoder_s{seed}", "test")["score"]
        thr = run["threshold"]
        flag = st >= thr
        rep = M.binary_report(d["yb_test"], flag.astype(int), st)
        rep["by_category"] = M.detection_by_category(d["y_test"], flag)
        rep["threshold"] = thr
        runs.append({**run, "test": rep})
        histories[f"seed {seed}"] = run["history"]
        if seed == seeds[0]:
            main = rep
            n_params = run["n_parameters"]
            plots.score_hist_plot(np.log10(st[d["yb_test"] == 0]), np.log10(st[d["yb_test"] == 1]),
                                  np.log10(thr), "Autoencoder reconstruction error, Track A test",
                                  "log10(mean squared reconstruction error)", figs / "autoencoder_scores.png")
        print(f"  autoencoder seed {seed}: ROC-AUC {rep['roc_auc']:.4f} F1 {rep['f1']:.4f} "
              f"FPR {rep['false_positive_rate']:.3f}", flush=True)
    plots.history_plot(histories, "Autoencoder: reconstruction loss (normal traffic)",
                       figs / "history_autoencoder.png")
    store.save_metrics("A", "autoencoder", {
        "model": "Autoencoder", "task": "binary anomaly detection", "track": "A",
        "trained_on": "normal training records only", "arch": "Autoencoder",
        "params": best["params"], "train_config": base.__dict__, "loss": "MSE",
        "selection": "validation ROC-AUC (seed 0)", "search": study,
        "threshold_rule": f"{1 - AE_FPR_TARGET:.0%} quantile of validation normal reconstruction error",
        "seeds": seeds, "test": main, "n_parameters": n_params,
        "seed_roc_auc": {"mean": float(np.mean([r["test"]["roc_auc"] for r in runs])),
                         "std": float(np.std([r["test"]["roc_auc"] for r in runs]))},
        "histories": {f"seed {r['seed']}": r["history"] for r in runs},
        "fit_seconds": float(np.mean([r["seconds"] for r in runs])),
        "predict_us_per_record": float(np.mean([r["us"] for r in runs])),
        "search_note": REDUCED_NOTE,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=["mlp", "cnn", "ae"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="recompute final metrics even if the model's metrics file exists")
    args = ap.parse_args()
    only = set(args.only or ["mlp", "cnn", "ae"])
    seeds = load_config()["project"]["seeds"][: (1 if args.quick else 3)]
    base = TrainConfig(max_epochs=3 if args.quick else 30, patience=4, batch_size=512)
    figs = ensure_dir(store.FIG_DIR)
    d = store.load_track_a_arrays()
    t0 = time.time()
    mlp_best = None
    if "mlp" in only and store.metrics_path("A", "mlp").exists() and not (args.force or args.quick):
        print("MLP: artifacts/metrics/track_a/mlp.json exists, skipped (use --force to recompute)")
        only.discard("mlp")
    if ("cnn" in only and store.metrics_path("A", "cnn").exists()
            and store.metrics_path("A", "cnn_random_order").exists() and not (args.force or args.quick)):
        print("1D-CNN: cnn.json and cnn_random_order.json exist, skipped (use --force to recompute)")
        only.discard("cnn")
    if "mlp" in only:
        print("MLP")
        mlp_best = run_mlp(d, args.quick, seeds, figs, base)
    if "cnn" in only:
        if mlp_best is None:
            m = store.load_metrics("A", "mlp")
            mlp_best = {"train_config": {k: v for k, v in m["train_config"].items() if k in ("lr", "batch_size")},
                        "imbalance": m["imbalance"]}
        print("1D-CNN")
        run_cnn(d, args.quick, seeds, figs, base, mlp_best)
    if "ae" in only:
        print("Autoencoder")
        run_autoencoder(d, args.quick, seeds, figs, base)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
