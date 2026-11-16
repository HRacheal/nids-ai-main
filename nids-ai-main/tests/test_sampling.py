"""Reduced-training-data helpers: they must only ever shrink TRAINING data."""

import importlib.util
from pathlib import Path

import numpy as np

from deepshield.data import sampling
from deepshield.data.sequences import WindowIndex

ROOT = Path(__file__).resolve().parents[1]


def test_stratified_subsample_keeps_every_class_and_small_classes_whole():
    y = np.repeat(np.arange(4), [10_000, 4_000, 1_500, 40])
    idx = sampling.stratified_subsample(y, 0.5, seed=0, floor=2000)
    c = np.bincount(y[idx], minlength=4)
    assert c.tolist() == [5000, 2000, 1500, 40]          # floor 2000; classes below it kept whole
    assert np.all(np.diff(idx) > 0)                       # sorted, no duplicates
    assert np.array_equal(idx, sampling.stratified_subsample(y, 0.5, seed=0, floor=2000))   # reproducible
    assert len(sampling.stratified_subsample(y, 1.0)) == len(y)


def test_recent_fraction_is_chronological_and_training_only():
    rng = np.random.default_rng(0)
    n = 5000
    session = np.where(np.arange(n) < 2500, "s1", "s2")
    ltime = np.r_[np.sort(rng.integers(0, 1000, 2500)), np.sort(rng.integers(5000, 6000, 2500))]
    block = np.where((ltime % 1000) < 700, "train", np.where((ltime % 1000) < 850, "val", "test"))
    in_train = block == "train"
    keep = sampling.recent_fraction(in_train, session, ltime, 0.5)
    assert not keep[~in_train].any()                       # never validation / test / gap records
    for s in ("s1", "s2"):                                  # a contiguous, most recent span per session
        kt, dt = ltime[keep & (session == s)], ltime[in_train & ~keep & (session == s)]
        assert len(kt) and kt.min() >= dt.max()


def _load_train_sequence():
    spec = importlib.util.spec_from_file_location("train_sequence", ROOT / "scripts" / "train_sequence.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_lazy_rows_give_identical_windows(tmp_path):
    ts = _load_train_sequence()
    rng = np.random.default_rng(1)
    X_all = rng.normal(size=(300, 6)).astype(np.float32)
    np.save(tmp_path / "X.npy", X_all)
    mm = np.load(tmp_path / "X.npy", mmap_mode="r")
    gidx = np.sort(rng.choice(300, 120, replace=False))
    ram = X_all[gidx].astype(np.float16)
    lazy = ts.LazyRows(mm, gidx)
    host = rng.integers(0, 7, 120)
    wi = WindowIndex.build(host, np.arange(120))
    rec = rng.choice(120, 30, replace=False)
    a, la = wi.gather(ram, rec, 5)
    b, lb = wi.gather(lazy, rec, 5)
    assert np.array_equal(a, b) and np.array_equal(la, lb)
    assert np.array_equal(lazy[10:20], ram[10:20]) and lazy.shape == ram.shape


def test_peak_memory_is_reported():
    import sys
    v = sampling.peak_memory_mb()
    if sys.platform == "win32":
        assert v is not None and v > 10      # regression: the Windows query used to fail silently
    assert v is None or v > 0
