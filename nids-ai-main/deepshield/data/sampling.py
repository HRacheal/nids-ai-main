"""Reduced training data (documented in docs/sampling.md).

Only TRAINING data is ever reduced. Validation and test data, the official
UNSW-NB15 test set and the Track B validation/test blocks are never touched.

* Track A: `stratified_subsample` keeps a fraction of each class's training
  records, chosen at random with a fixed seed, but never fewer than `floor`
  records of a class (or all of them if the class is smaller). Every class
  therefore stays present and the rarest classes are kept whole.
* Track B: `recent_fraction` keeps the most recent fraction of each capture
  session's training period. The flows are not shuffled; windows are built
  afterwards from the kept flows in time order, the 300 s gaps before the
  validation and test blocks are unchanged, so no window crosses a block
  boundary and nothing from the future enters training.
"""

from __future__ import annotations

import numpy as np

from deepshield.data.schema import CLASS_ORDER

TRACK_A_FLOOR = 2000          # classes with fewer training records are kept whole


def stratified_subsample(y: np.ndarray, fraction: float, seed: int = 0,
                         floor: int = TRACK_A_FLOOR) -> np.ndarray:
    """Sorted indices of a per-class random sample (fraction, with a minimum per class)."""
    if fraction >= 1.0:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        n = min(len(idx), max(int(round(fraction * len(idx))), floor))
        keep.append(rng.choice(idx, n, replace=False))
    return np.sort(np.concatenate(keep))


def recent_fraction(in_train: np.ndarray, session: np.ndarray, ltime: np.ndarray,
                    fraction: float) -> np.ndarray:
    """Boolean mask: training records in the most recent `fraction` of each session's training period."""
    if fraction >= 1.0:
        return in_train.copy()
    keep = np.zeros(len(in_train), dtype=bool)
    for s in np.unique(session[in_train]):
        m = in_train & (session == s)
        t = ltime[m]
        cut = t.min() + (1.0 - fraction) * (t.max() - t.min())
        keep |= m & (ltime >= cut)
    return keep


def class_counts(y: np.ndarray, k: int = len(CLASS_ORDER)) -> dict[str, int]:
    return dict(zip(CLASS_ORDER[:k], np.bincount(y, minlength=k).astype(int).tolist()))


def distribution_table(y_full: np.ndarray, samples: dict[str, np.ndarray]) -> dict:
    """Counts, share of the sample, and share of the class retained, per class and sample."""
    full = np.bincount(y_full, minlength=len(CLASS_ORDER))
    out = {"full": {"n": int(len(y_full)), "counts": class_counts(y_full),
                    "class_share": dict(zip(CLASS_ORDER, np.round(full / full.sum(), 5).tolist()))}}
    for name, ys in samples.items():
        c = np.bincount(ys, minlength=len(CLASS_ORDER))
        out[name] = {"n": int(len(ys)), "fraction_of_full": round(len(ys) / len(y_full), 4),
                     "counts": class_counts(ys),
                     "class_share": dict(zip(CLASS_ORDER, np.round(c / c.sum(), 5).tolist())),
                     "class_retained": dict(zip(CLASS_ORDER, np.round(c / np.maximum(full, 1), 4).tolist()))}
    return out


def peak_memory_mb() -> float | None:
    """Peak resident memory of this process (MB), where the OS reports it."""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE   # 64-bit pseudo-handle; default int overflows
        query = k32.K32GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        if query(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return round(pmc.PeakWorkingSetSize / 2**20, 1)
    except Exception:
        pass
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except Exception:
        return None
