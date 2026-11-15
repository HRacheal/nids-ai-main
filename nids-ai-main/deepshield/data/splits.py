"""Train/validation/test splits for both tracks.

Track A: the official test set is untouched. A validation set (for early
stopping, hyperparameters and thresholds) is carved out of the official
training set with StratifiedGroupKFold: stratified by attack category, and
grouped by the full feature vector so that identical records (38.6 % of the
training set are exact duplicates) never sit on both sides of the split.

Track B: chronological. Each capture session (22 Jan and 18 Feb 2015) is cut
by end time into train (first 70 % of its records), validation (next 15 %)
and test (last 15 %). Records ending within GAP_SECONDS after a cut are
dropped. Because the ct_* counters partly count later connections and their
exact window could not be reproduced (dataset profile, section 7), the gap
reduces but cannot be proven to remove leakage through those features.
Classes are not forced into every block: a class absent from a block is
reported with zero support, never fixed by a random split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from deepshield.data import schema

VAL_FOLDS = 7            # 1/7 = 14.3 % of the training set for validation
GAP_SECONDS = 300
TRACK_B_FRACTIONS = (0.70, 0.85)
SESSION_CUTOFF = pd.Timestamp("2015-02-01", tz="UTC").timestamp()


def feature_groups(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return pd.util.hash_pandas_object(df[features], index=False).to_numpy()


def track_a_validation_split(df: pd.DataFrame, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Indices (train, validation) within the official Track A training set."""
    groups = feature_groups(df, schema.TRACK_A_FEATURES)
    sgkf = StratifiedGroupKFold(n_splits=VAL_FOLDS, shuffle=True, random_state=seed)
    tr, va = next(sgkf.split(np.zeros(len(df)), df["attack_cat"], groups))
    return np.sort(tr), np.sort(va)


def track_b_blocks(stime: np.ndarray, ltime: np.ndarray,
                   fractions=TRACK_B_FRACTIONS, gap: int = GAP_SECONDS) -> tuple[np.ndarray, dict]:
    """Assign each Track B record to 'train', 'val', 'test' or 'gap'.

    Returns the block label per record and the cut times per session.
    """
    block = np.full(len(ltime), "gap", dtype=object)
    session = np.where(stime < SESSION_CUTOFF, "2015-01-22", "2015-02-18")
    cuts = {}
    for s in np.unique(session):
        m = session == s
        lt = ltime[m]
        c1, c2 = np.quantile(lt, fractions[0]), np.quantile(lt, fractions[1])
        b = np.full(m.sum(), "gap", dtype=object)
        b[lt <= c1] = "train"
        b[(lt > c1 + gap) & (lt <= c2)] = "val"
        b[lt > c2 + gap] = "test"
        block[m] = b
        cuts[s] = {"train_end": int(c1), "val_start": int(c1 + gap),
                   "val_end": int(c2), "test_start": int(c2 + gap),
                   "session_end": int(lt.max())}
    return block, cuts
