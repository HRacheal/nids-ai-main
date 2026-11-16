#!/usr/bin/env python3
"""Phase 4 (Track B): LSTM over per-host flow windows, with comparison models.

All models predict the category of every flow in the chronological
validation and test blocks (Track B split, docs/preprocessing.md):

* FlowLSTM - window = the target flow plus up to L-1 earlier flows from the
  same source host in the same block. L chosen on validation.
* per-flow MLP - same features, no history (does order/context help?).
* per-flow Random Forest - non-neural, no history.
* context baseline - per-flow MLP probabilities plus the MEAN of the MLP's
  predicted probabilities for the host's previous L-1 flows, combined by a
  logistic regression fitted on the validation block (it never sees the test
  block). It uses context but no learned ordering.
* oracle context - the same, but with the TRUE labels of the previous flows.
  Not deployable (true labels are unknown at detection time); it is an upper
  bound showing how much host history could help.

Training samples (CPU/memory budget): every epoch draws a fresh random sample
of at most EPOCH_CAP target flows per class from the training block (all
flows of the small classes). Each class's loss is multiplied by
n_class / n_sampled so the objective equals the full-data objective, then by
1/sqrt(frequency) class weights. Windows still use every earlier flow of the
host, sampled or not. Validation loss (early stopping) uses the same
per-class sampling fractions on the validation block with a fixed seed;
validation and test PREDICTIONS use every flow.

Reduced search (documented in docs/models.md): L in {4, 16} (L = 32 dropped),
at most 15 epochs with patience 3. Every finished run is cached in
artifacts/metrics/track_b/runs/ so an interrupted job resumes, and every
epoch is checkpointed in artifacts/cache/resume/.

Reduced training data (docs/sampling.md): every Track B model is trained on
the most recent TRAIN_FRACTION of each session's training period
(deepshield.data.sampling.recent_fraction), with the per-class epoch caps
scaled by the same fraction. Flows are never shuffled before windows are
built; the validation and test blocks are complete and unchanged. They are
read from the memory-mapped feature file rather than copied into RAM.

Usage:  .venv\\Scripts\\python.exe scripts\\train_sequence.py [--only lstm mlp rf context]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepshield.config import PROJECT_ROOT, ensure_dir, load_config, path  # noqa: E402
from deepshield.data import sampling  # noqa: E402
from deepshield.data.imbalance import class_weights  # noqa: E402
from deepshield.data.splits import SESSION_CUTOFF  # noqa: E402
from deepshield.data.schema import CLASS_ORDER  # noqa: E402
from deepshield.data.sequences import WindowIndex  # noqa: E402
from deepshield.evaluation import metrics as M  # noqa: E402
from deepshield.evaluation import plots  # noqa: E402
from deepshield.models.networks import count_parameters  # noqa: E402
from deepshield.training import store  # noqa: E402
from deepshield.training.trainer import TrainConfig, fit, save_checkpoint  # noqa: E402

K = len(CLASS_ORDER)
TRAIN_FRACTION = 0.5          # share of each session's training period used (most recent part)
FULL_EPOCH_CAP = np.array([60_000] + [30_000] * (K - 1))
EPOCH_CAP = (FULL_EPOCH_CAP * TRAIN_FRACTION).astype(int)   # max sampled targets per class and epoch
SEQ_LENGTHS = [4, 16]
REDUCED_NOTE = ("reduced search because of limited memory/CPU (7.7 GB RAM shared, 4 cores): "
                "L in {4, 16}, <= 15 epochs, per-class epoch sampling; see docs/models.md. "
                f"Reduced training data: most recent {TRAIN_FRACTION:.0%} of each session's training "
                "period; see docs/sampling.md")
TRAINING_DATA: dict = {}      # filled in main(); stored with every Track B metrics file


class LazyRows:
    """Rows of a block read on demand from the memory-mapped feature file (no RAM copy).

    Values are rounded to float16 exactly like the in-RAM training block, so
    every block is treated identically."""

    def __init__(self, X_all, gidx):
        self.X_all, self.gidx = X_all, gidx
        self.shape = (len(gidx), X_all.shape[1])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, i):
        if isinstance(i, slice):
            i = np.arange(self.shape[0])[i]
        return np.asarray(self.X_all[self.gidx[np.asarray(i)]]).astype(np.float16)


class Block:
    """One chronological block: features in RAM (float16), labels, window index."""

    def __init__(self, X_all, y_all, meta, name, keep=None, in_ram=True):
        mask = meta["block"].to_numpy() == name
        if keep is not None:
            mask &= keep
        idx = np.flatnonzero(mask)
        self.name = name
        if in_ram:
            self.X = np.empty((len(idx), X_all.shape[1]), dtype=np.float16)
            for s in range(0, len(idx), 200_000):      # chunked: avoids a float32 copy of the block
                self.X[s:s + 200_000] = X_all[idx[s:s + 200_000]]
        else:
            self.X = LazyRows(X_all, idx)
        self.y = y_all[idx]
        bm = meta.iloc[idx]                       # used only here: not kept (srcip strings are large)
        self.wi = WindowIndex.build(bm["srcip"].to_numpy(), bm["ltime"].to_numpy())
        # 10-minute time bins: bootstrap blocks for confidence intervals
        self.time_bin = (bm["ltime"].to_numpy() // 600).astype(np.int64)

    def windows(self, records, L):
        w, lengths = self.wi.gather(self.X, records, L)
        return torch.from_numpy(w.astype(np.float32)), torch.from_numpy(lengths.astype(np.int64))


def sample_fractions(y_train) -> np.ndarray:
    n = np.bincount(y_train, minlength=K)
    return np.minimum(1.0, EPOCH_CAP / np.maximum(n, 1))


def sampled_targets(y, rng, frac):
    keep = rng.random(len(y)) < frac[y]
    return rng.permutation(np.flatnonzero(keep))


def loss_weights(y_train, frac):
    w = class_weights(y_train, K, "sqrt") / frac          # undo the per-class sampling
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def make_batches(block: Block, L: int | None, bs: int, frac, fixed_seed: int | None = None):
    """Batch generator; L=None -> single-flow batches (per-flow models)."""
    def gen(rng):
        r = np.random.default_rng(fixed_seed) if fixed_seed is not None else rng
        targets = sampled_targets(block.y, r, frac)
        for s in range(0, len(targets), bs):
            t = targets[s:s + bs]
            yb = torch.from_numpy(block.y[t])
            if L is None:
                yield torch.from_numpy(block.X[t].astype(np.float32)), yb
            else:
                w, lengths = block.windows(t, L)
                yield w, lengths, yb
    return gen


lstm_forward = lambda m, b: (m(b[0], b[1]), b[2])  # noqa: E731


@torch.no_grad()
def predict_block(model, block: Block, L: int | None, bs: int = 2048) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(block.y), bs):
        r = np.arange(s, min(s + bs, len(block.y)))
        if L is None:
            logits = model(torch.from_numpy(block.X[r].astype(np.float32)))
        else:
            logits = model(*block.windows(r, L))
        out.append(torch.softmax(logits, 1).numpy().astype(np.float32))
    return np.concatenate(out)


def report(block: Block, proba: np.ndarray) -> dict:
    yp = proba.argmax(1)
    rep = M.multiclass_report(block.y, yp, proba)
    rep["bootstrap_macro_f1_time_blocks"] = M.bootstrap_ci(block.y, yp, blocks=block.time_bin, n_boot=500)
    return rep


def train_nn(arch, params, train: Block, val: Block, L, cfg: TrainConfig, resume=None):
    frac = sample_fractions(train.y)
    loss = torch.nn.CrossEntropyLoss(weight=loss_weights(train.y, frac))
    fwd = lstm_forward if L is not None else None
    return fit(arch, params, cfg, make_batches(train, L, cfg.batch_size, frac),
               make_batches(val, L, 2048, frac, fixed_seed=0), loss, forward=fwd, resume_path=resume)


def nn_final(name, arch, params, L, blocks, seeds, base, extra_ckpt):
    """Train/evaluate one configuration for every seed (resumable). Returns runs."""
    train, val, test = blocks
    runs = []
    for seed in seeds:
        cfg = TrainConfig(**{**base.__dict__, "seed": seed})
        key = store.run_key(kind="final", name=name, arch=arch, params=params, L=L, cfg=cfg.__dict__,
                            cap=EPOCH_CAP.tolist(), first=seed == seeds[0], train_fraction=TRAIN_FRACTION)
        run = store.load_run("B", name, key)
        if run is None:
            res = train_nn(arch, params, train, val, L, cfg, store.epoch_checkpoint("B", name, key))
            pv = predict_block(res.model, val, L)
            t0 = time.time()
            pt = predict_block(res.model, test, L)
            us = (time.time() - t0) / len(pt) * 1e6
            store.save_predictions("B", f"{name}_s{seed}", "test", proba=pt)
            store.save_predictions("B", f"{name}_s{seed}", "val", proba=pv)
            if seed == seeds[0]:
                save_checkpoint(store.model_dir("B", name) / "model.pt", res.model, arch, params, cfg,
                                {**extra_ckpt, "task": "multiclass", "classes": CLASS_ORDER})
            run = {"seed": seed, "val_macro_f1": M.multiclass_report(val.y, pv.argmax(1))["macro_f1"],
                   "seconds": res.seconds, "us": us, "history": res.history,
                   "peak_memory_mb": sampling.peak_memory_mb(),
                   "n_parameters": count_parameters(res.model)}
            store.save_run("B", name, key, run)
            store.clear_epoch_checkpoint("B", name, key)
            del res, pv
            store.release_memory()
        else:
            print(f"  {name} seed {seed} [cached]", flush=True)
            pt = store.load_predictions("B", f"{name}_s{seed}", "test")["proba"]
        rep = report(test, pt)
        runs.append({**run, "test": rep})
        print(f"  {name} seed {seed}: test macro-F1 {rep['macro_f1']:.4f}", flush=True)
    return runs


def run_lstm(blocks, seeds, base, figs, n_in):
    train, val, _ = blocks
    study = []
    params = {"n_in": n_in, "n_out": K, "proj": 64, "hidden": 64, "layers": 1, "dropout": 0.2}
    for L in SEQ_LENGTHS:
        cfg = TrainConfig(**{**base.__dict__, "seed": 0})
        key = store.run_key(kind="search", arch="FlowLSTM", params=params, L=L, cfg=cfg.__dict__,
                            cap=EPOCH_CAP.tolist(), train_fraction=TRAIN_FRACTION)
        row = store.load_run("B", "search_lstm", key)
        print(f"  LSTM L={L}" + ("  [cached]" if row else ""), flush=True)
        if row is None:
            res = train_nn("FlowLSTM", params, train, val, L, cfg,
                           store.epoch_checkpoint("B", "search_lstm", key))
            pv = predict_block(res.model, val, L)
            rep = M.multiclass_report(val.y, pv.argmax(1), pv)
            row = {"seq_len": L, "params": params, "val_macro_f1": rep["macro_f1"],
                   "best_epoch": res.history["best_epoch"], "epochs_run": res.history["epochs_run"],
                   "seconds": round(res.seconds), "seed": 0}
            store.save_run("B", "search_lstm", key, row)
            store.clear_epoch_checkpoint("B", "search_lstm", key)
            del res, pv
            store.release_memory()
        row = {k: v for k, v in row.items() if k != "saved_utc"}
        study.append(row)
        print(f"    -> val macro-F1 {row['val_macro_f1']:.4f} ({row['seconds']}s)", flush=True)
    best = max(study, key=lambda r: r["val_macro_f1"])
    L = best["seq_len"]
    runs = nn_final("lstm", "FlowLSTM", best["params"], L, blocks, seeds, base,
                    {"seq_len": L, "entity": "srcip"})
    plots.confusion_matrix_plot(runs[0]["test"]["confusion_matrix"], f"LSTM (L={L}), Track B test",
                                figs / "cm_lstm.png")
    plots.history_plot({f"seed {r['seed']}": r["history"] for r in runs},
                       f"LSTM (L={L}): training and validation loss", figs / "history_lstm.png")
    _save("lstm", "LSTM", runs, {"arch": "FlowLSTM", "params": best["params"], "seq_len": L,
                                 "entity": "source IP", "search": study,
                                 "n_parameters": runs[0]["n_parameters"], "train_config": base.__dict__})
    return L


def run_mlp(blocks, seeds, base, figs, n_in):
    a = store.load_metrics("A", "mlp")
    params = {**a["params"], "n_in": n_in}
    runs = nn_final("mlp_flow", "MLP", params, None, blocks, seeds, base, {})
    plots.confusion_matrix_plot(runs[0]["test"]["confusion_matrix"], "Per-flow MLP, Track B test",
                                figs / "cm_mlp_flow_track_b.png")
    _save("mlp_flow", "Per-flow MLP", runs, {"arch": "MLP", "params": params,
                                             "n_parameters": runs[0]["n_parameters"],
                                             "train_config": base.__dict__,
                                             "architecture_from": "Track A MLP selection"})


def run_rf(blocks, seeds):
    from sklearn.ensemble import RandomForestClassifier
    train, val, test = blocks
    a = store.load_metrics("A", "rf_multiclass")["config"]
    frac = sample_fractions(train.y)
    runs = []
    for seed in seeds:
        key = store.run_key(kind="final", name="rf_flow", config=a, seed=seed, cap=EPOCH_CAP.tolist(),
                            train_fraction=TRAIN_FRACTION)
        run = store.load_run("B", "rf_flow", key)
        if run is None:
            t = np.sort(sampled_targets(train.y, np.random.default_rng(seed), frac))
            sw = (1.0 / frac)[train.y[t]]                    # weight the sample back to the full block
            rf = RandomForestClassifier(n_estimators=a["n_estimators"],
                                        min_samples_leaf=a["min_samples_leaf"], max_features=a["max_features"],
                                        class_weight="balanced_subsample"
                                        if a["imbalance"] == "balanced_subsample" else None,
                                        n_jobs=4, random_state=seed)
            t0 = time.time()
            rf.fit(train.X[t].astype(np.float32), train.y[t], sample_weight=sw)
            secs = time.time() - t0

            def proba(X):
                return np.concatenate([rf.predict_proba(X[s:s + 50_000].astype(np.float32))
                                       for s in range(0, len(X), 50_000)]).astype(np.float32)
            pv = proba(val.X)
            t0 = time.time()
            pt = proba(test.X)
            us = (time.time() - t0) / len(pt) * 1e6
            store.save_predictions("B", f"rf_flow_s{seed}", "test", proba=pt)
            store.save_predictions("B", f"rf_flow_s{seed}", "val", proba=pv)
            if seed == seeds[0]:
                joblib.dump(rf, store.model_dir("B", "rf_flow") / "model.joblib", compress=3)
            run = {"seed": seed, "val_macro_f1": M.multiclass_report(val.y, pv.argmax(1))["macro_f1"],
                   "seconds": secs, "us": us, "n_train_samples": int(len(t))}
            store.save_run("B", "rf_flow", key, run)
            del rf, pv
            store.release_memory()
        else:
            print(f"  rf_flow seed {seed} [cached]", flush=True)
            pt = store.load_predictions("B", f"rf_flow_s{seed}", "test")["proba"]
        rep = report(test, pt)
        runs.append({**run, "test": rep})
        print(f"  per-flow RF seed {seed}: test macro-F1 {rep['macro_f1']:.4f}", flush=True)
    _save("rf_flow", "Per-flow Random Forest", runs,
          {"config": a, "training_sample": "per-class sample (EPOCH_CAP), weighted back to the full block"})


def run_context(blocks, L, seeds):
    """Context baseline and oracle on top of the per-flow MLP (seed by seed)."""
    from sklearn.linear_model import LogisticRegression
    _, val, test = blocks

    def feats(block, p, variant):
        src = p if variant == "context_baseline" else np.eye(K, dtype=np.float32)[block.y]
        out = np.empty((len(p), 2 * K + 1), np.float32)
        for s in range(0, len(p), 50_000):                  # chunked window gathering
            r = np.arange(s, min(s + 50_000, len(p)))
            w, lengths = block.wi.gather(src, r, L)
            prev = w[:, :-1, :]                              # exclude the target flow itself
            n_prev = np.maximum(lengths - 1, 0)[:, None]
            ctx = np.divide(prev.sum(1), n_prev, out=np.zeros((len(r), K), np.float32), where=n_prev > 0)
            out[r] = np.hstack([np.log(p[r] + 1e-6), ctx, (n_prev > 0).astype(np.float32)])
        return out

    for variant in ("context_baseline", "oracle_context"):
        runs = []
        for seed in seeds:
            pv = store.load_predictions("B", f"mlp_flow_s{seed}", "val")["proba"]
            pt = store.load_predictions("B", f"mlp_flow_s{seed}", "test")["proba"]
            lr = LogisticRegression(max_iter=300, C=1.0)
            lr.fit(feats(val, pv, variant), val.y)
            prob = lr.predict_proba(feats(test, pt, variant))
            full = np.zeros((len(prob), K), np.float32)
            full[:, lr.classes_] = prob
            rep = report(test, full)
            runs.append({"seed": seed, "test": rep, "val_macro_f1": None, "seconds": 0, "us": None})
            store.save_predictions("B", f"{variant}_s{seed}", "test", proba=full)
            print(f"  {variant} seed {seed}: test macro-F1 {rep['macro_f1']:.4f}", flush=True)
        _save(variant, "Context baseline (previous predictions)" if variant == "context_baseline"
              else "Oracle context (previous TRUE labels; upper bound, not deployable)", runs,
              {"seq_len": L, "combiner": "multinomial logistic regression fitted on the validation block",
               "base_model": "per-flow MLP"})


def _save(name, title, runs, extra):
    f1 = [r["test"]["macro_f1"] for r in runs]
    store.save_metrics("B", name, {
        "model": title, "task": "multiclass", "track": "B", "test": runs[0]["test"],
        "seeds": [r["seed"] for r in runs],
        "seed_summary": {"macro_f1_mean": float(np.mean(f1)), "macro_f1_std": float(np.std(f1)),
                         "per_seed": [{"seed": r["seed"], "macro_f1": r["test"]["macro_f1"],
                                       "val_macro_f1": r["val_macro_f1"]} for r in runs]},
        "histories": {f"seed {r['seed']}": r["history"] for r in runs if "history" in r},
        "fit_seconds": float(np.mean([r["seconds"] for r in runs])),
        "predict_us_per_record": (float(np.mean([r["us"] for r in runs]))
                                  if runs[0]["us"] is not None else None),
        "epoch_sampling_cap_per_class": EPOCH_CAP.tolist(), "search_note": REDUCED_NOTE,
        "training_data": TRAINING_DATA, "peak_memory_mb_process": sampling.peak_memory_mb(),
        **extra})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=["lstm", "mlp", "rf", "context"])
    args = ap.parse_args()
    only = set(args.only or ["lstm", "mlp", "rf", "context"])
    seeds = load_config()["project"]["seeds"]
    base = TrainConfig(max_epochs=15, patience=3, batch_size=512, lr=1e-3)
    figs = ensure_dir(store.FIG_DIR)
    d = path("processed") / "track_b"
    X_all = np.load(d / "X.npy", mmap_mode="r")
    y_all = np.load(d / "y.npy")
    meta = pd.read_parquet(d / "meta.parquet", columns=["srcip", "stime", "ltime", "block"])
    t0 = time.time()
    in_train = (meta["block"] == "train").to_numpy()
    session = np.where(meta["stime"].to_numpy() < SESSION_CUTOFF, "2015-01-22", "2015-02-18")
    lt = meta["ltime"].to_numpy()
    keep = sampling.recent_fraction(in_train, session, lt, TRAIN_FRACTION)
    TRAINING_DATA.update({
        "track": "B", "full_or_reduced": "reduced" if TRAIN_FRACTION < 1 else "full",
        "strategy": "most recent fraction of each capture session's training period (chronological; "
                    "no shuffling before window construction; validation/test blocks unchanged)",
        "train_fraction_of_time_span": TRAIN_FRACTION,
        "epoch_cap_full": FULL_EPOCH_CAP.tolist(), "epoch_cap_used": EPOCH_CAP.tolist(),
        "distribution": sampling.distribution_table(y_all[in_train], {"reduced": y_all[keep]}),
    })
    study = {"chosen_fraction": TRAIN_FRACTION, "strategy": TRAINING_DATA["strategy"],
             "full": {"n": int(in_train.sum()), "counts": sampling.class_counts(y_all[in_train])},
             "candidates": {}}
    for f in (0.7, 0.5, 0.3):
        kf = sampling.recent_fraction(in_train, session, lt, f)
        study["candidates"][f"{f:.0%}"] = sampling.distribution_table(
            y_all[in_train], {"sample": y_all[kf]})["sample"]
    out_json = ensure_dir(PROJECT_ROOT / "reports" / "preprocessing") / "sampling_track_b.json"
    out_json.write_text(json.dumps(study, indent=2), encoding="utf-8")
    blocks = [Block(X_all, y_all, meta, "train", keep=keep, in_ram=True),
              Block(X_all, y_all, meta, "val", in_ram=False),
              Block(X_all, y_all, meta, "test", in_ram=False)]
    del meta, in_train, session, lt, keep
    store.release_memory()
    n_in = blocks[0].X.shape[1]
    per_epoch = int((sample_fractions(blocks[0].y) * np.bincount(blocks[0].y, minlength=K)).sum())
    print(f"blocks loaded in {time.time() - t0:.0f}s: " + ", ".join(f"{b.name} {len(b.y)}" for b in blocks)
          + f"; ~{per_epoch:,} training targets per epoch", flush=True)
    L = None
    if "lstm" in only:
        print("LSTM", flush=True)
        L = run_lstm(blocks, seeds, base, figs, n_in)
    if "mlp" in only:
        print("Per-flow MLP", flush=True)
        run_mlp(blocks, seeds, base, figs, n_in)
    if "rf" in only:
        print("Per-flow Random Forest", flush=True)
        run_rf(blocks, seeds)
    if "context" in only:
        L = L or store.load_metrics("B", "lstm")["seq_len"]
        print("Context baselines", flush=True)
        run_context(blocks, L, seeds)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
