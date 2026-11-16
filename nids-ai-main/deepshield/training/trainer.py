"""Generic PyTorch training loop with early stopping (CPU).

* Adam optimiser, (weighted) cross-entropy for classifiers, MSE for the
  autoencoder.
* Validation loss is computed after every epoch; the best state is kept and
  training stops after `patience` epochs without improvement.
* Every run records the architecture, hyperparameters, per-epoch losses,
  best epoch, wall-clock time and parameter count, and saves a checkpoint
  from which the model can be rebuilt (`load_checkpoint`).
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch
from torch import nn

from deepshield.config import set_seed, torch_threads
from deepshield.models.networks import build, count_parameters
from deepshield.training.store import atomic_replace


@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 512
    max_epochs: int = 40
    patience: int = 5
    weight_decay: float = 0.0
    seed: int = 0


@dataclass
class TrainResult:
    model: nn.Module
    history: dict = field(default_factory=dict)
    seconds: float = 0.0


Batches = Callable[[np.random.Generator], Iterator[tuple]]


def _array_batches(X: np.ndarray, y: np.ndarray | None, bs: int, shuffle: bool):
    def gen(rng: np.random.Generator):
        idx = rng.permutation(len(X)) if shuffle else np.arange(len(X))
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            xb = torch.from_numpy(np.ascontiguousarray(X[b]))
            yield (xb,) if y is None else (xb, torch.from_numpy(y[b]))
    return gen


def fit(arch: str, params: dict, cfg: TrainConfig, train_batches: Batches,
        val_batches: Batches, loss_fn: nn.Module,
        forward: Callable | None = None, log: Callable = print,
        resume_path: Path | None = None) -> TrainResult:
    """Train with early stopping on validation loss.

    If `resume_path` is given, the complete training state (model, optimiser,
    best weights, history and both RNG states) is written there after every
    epoch, and an existing file is loaded first. An interrupted run therefore
    continues from its last finished epoch and produces the same result as an
    uninterrupted one. The caller deletes the file once the run is stored.
    """
    torch.set_num_threads(torch_threads())
    set_seed(cfg.seed)
    model = build(arch, params)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    fwd = forward or (lambda m, b: (m(b[0]), b[-1]))
    rng = np.random.default_rng(cfg.seed)
    best, best_loss, best_epoch, bad = None, float("inf"), 0, 0
    hist = {"train_loss": [], "val_loss": [], "epoch_seconds": []}
    t_start = time.time()
    first_epoch, prior_seconds = 1, 0.0
    if resume_path is not None and Path(resume_path).exists():
        st = torch.load(resume_path, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        best, best_loss, best_epoch, bad = st["best"], st["best_loss"], st["best_epoch"], st["bad"]
        hist, prior_seconds = st["hist"], st["seconds"]
        rng.bit_generator.state = st["np_rng"]
        torch.set_rng_state(st["torch_rng"])
        first_epoch = st["epoch"] + 1
        log(f"    resumed after epoch {st['epoch']} from {Path(resume_path).name}")
    epoch = first_epoch - 1
    for epoch in range(first_epoch, cfg.max_epochs + 1):
        if bad >= cfg.patience:          # resumed a run that had already stopped
            epoch -= 1
            break
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for batch in train_batches(rng):
            opt.zero_grad()
            out, target = fwd(model, batch)
            loss = loss_fn(out, target)
            loss.backward()
            opt.step()
            tot += loss.item() * len(target)
            n += len(target)
        model.eval()
        vt, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_batches(np.random.default_rng(0)):
                out, target = fwd(model, batch)
                vt += loss_fn(out, target).item() * len(target)
                vn += len(target)
        tl, vl = tot / n, vt / vn
        hist["train_loss"].append(tl)
        hist["val_loss"].append(vl)
        hist["epoch_seconds"].append(time.time() - t0)
        improved = vl < best_loss - 1e-5
        if improved:
            best, best_loss, best_epoch, bad = copy.deepcopy(model.state_dict()), vl, epoch, 0
        else:
            bad += 1
        log(f"    epoch {epoch:2d}  train {tl:.4f}  val {vl:.4f}{'  *' if improved else ''}"
            f"  ({hist['epoch_seconds'][-1]:.0f}s)")
        if resume_path is not None:
            Path(resume_path).parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(resume_path).with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "best": best,
                        "best_loss": best_loss, "best_epoch": best_epoch, "bad": bad, "hist": hist,
                        "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                        "epoch": epoch, "seconds": prior_seconds + time.time() - t_start}, tmp)
            atomic_replace(tmp, Path(resume_path))
        if bad >= cfg.patience:
            break
    model.load_state_dict(best)
    model.eval()
    hist.update(best_epoch=best_epoch, best_val_loss=best_loss, epochs_run=epoch,
                stopped_early=bad >= cfg.patience)
    return TrainResult(model, hist, prior_seconds + time.time() - t_start)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, batch_size: int = 4096,
            fn: Callable | None = None) -> np.ndarray:
    """Softmax probabilities (or fn(model, xb) outputs) for every row of X."""
    model.eval()
    outs = []
    for s in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(X[s:s + batch_size]))
        outs.append((fn(model, xb) if fn else torch.softmax(model(xb), dim=1)).numpy())
    return np.concatenate(outs)


def save_checkpoint(path: Path, model: nn.Module, arch: str, params: dict,
                    cfg: TrainConfig, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    meta = {"arch": arch, "params": params, "train_config": asdict(cfg),
            "n_parameters": count_parameters(model), **(extra or {})}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def load_checkpoint(path: Path) -> tuple[nn.Module, dict]:
    meta = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
    model = build(meta["arch"], meta["params"])
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model, meta


array_batches = _array_batches
