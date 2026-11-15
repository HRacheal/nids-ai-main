"""Class-imbalance strategies, applied to TRAINING data only.

Three options are compared on the validation set (Phases 3 and 4); the
results are in reports/preprocessing/imbalance_study.json:

* "none"        - plain empirical risk.
* "sqrt"        - loss weights proportional to 1/sqrt(class frequency),
                  normalised to mean 1. A damped form of inverse-frequency
                  weighting: full inverse weights (431x for Worms vs Normal)
                  make neural-network training unstable and inflate false
                  positives on the minority classes.
* "balanced"    - weights proportional to 1/frequency (sklearn "balanced").
* "oversample"  - random oversampling (with replacement) of each minority
                  class up to `min_count` records. SMOTE is deliberately not
                  used: it interpolates between records of a class, but here
                  38 % of records are exact duplicates and many features are
                  discrete counters or one-hot categories, where
                  interpolated points are not valid flows.
"""

from __future__ import annotations

import numpy as np


def class_weights(y: np.ndarray, n_classes: int, mode: str = "sqrt") -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    present = counts > 0
    w = np.zeros(n_classes)
    if mode == "none":
        w[present] = 1.0
    elif mode == "sqrt":
        w[present] = 1.0 / np.sqrt(counts[present])
    elif mode == "balanced":
        w[present] = 1.0 / counts[present]
    else:
        raise ValueError(mode)
    # normalise so the average weight per training record is 1
    w *= counts.sum() / (w * counts).sum()
    return w


def random_oversample(y: np.ndarray, min_count: int, seed: int = 0) -> np.ndarray:
    """Indices into y where every class has at least `min_count` records."""
    rng = np.random.default_rng(seed)
    idx = [np.arange(len(y))]
    for c in np.unique(y):
        members = np.flatnonzero(y == c)
        if len(members) < min_count:
            idx.append(rng.choice(members, min_count - len(members), replace=True))
    return np.concatenate(idx)
