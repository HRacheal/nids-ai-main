"""Phase 2 tests: preprocessing, leakage-free splits, windows, imbalance."""

import numpy as np
import pandas as pd
import pytest

from deepshield.data import schema
from deepshield.data.imbalance import class_weights, random_oversample
from deepshield.data.preprocess import CLIP, Preprocessor, encode_targets
from deepshield.data.sequences import WindowIndex
from deepshield.data.splits import track_a_validation_split, track_b_blocks


def _frame(n=400, seed=0, with_missing=False):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.exponential(100, n) for c in schema.TRACK_A_NUMERIC})
    df["proto"] = rng.choice(["tcp", "udp", "arp"], n, p=[0.6, 0.39, 0.01])
    df["service"] = rng.choice(["-", "dns", "http"], n)
    df["state"] = rng.choice(["FIN", "INT"], n)
    df["attack_cat"] = rng.choice(schema.CLASS_ORDER, n)
    df["label"] = (df["attack_cat"] != "Normal").astype(int)
    if with_missing:
        df.loc[df.index[:50], "ct_ftp_cmd"] = np.nan
    return df


def test_preprocessor_uses_training_statistics_only():
    train, test = _frame(seed=0), _frame(seed=1)
    test["sbytes"] *= 1000                         # shifted test distribution
    pre = Preprocessor.for_track("A").fit(train)
    Xtr, Xte = pre.transform(train), pre.transform(test)
    j = pre.feature_names_.index("sbytes")
    assert abs(Xtr[:, j].mean()) < 1e-5              # standardised on train
    assert Xte[:, j].mean() > 1                      # test NOT re-standardised
    assert Xtr.dtype == np.float32 and np.isfinite(Xte).all()
    assert np.abs(Xte[:, :len(pre.numeric)]).max() <= CLIP


def test_rare_and_unseen_categories_go_to_other():
    train, test = _frame(), _frame(seed=2)
    test.loc[test.index[0], "state"] = "ACC"          # never seen in training
    pre = Preprocessor.for_track("A", min_level_count=20).fit(train)
    assert "proto=arp" not in pre.feature_names_       # rare -> other
    X = pre.transform(test)
    assert X[0, pre.feature_names_.index("state=other")] == 1.0
    for c in schema.CATEGORICAL:                       # exactly one hot per column
        cols = [i for i, n in enumerate(pre.feature_names_) if n.startswith(c + "=")]
        assert (X[:, cols].sum(axis=1) == 1).all()


def test_missing_values_imputed_with_indicator():
    train = _frame(with_missing=True)
    pre = Preprocessor.for_track("A").fit(train)
    X = pre.transform(train)
    assert pre.missing_cols_ == ["ct_ftp_cmd"]
    ind = X[:, pre.feature_names_.index("ct_ftp_cmd_missing")]
    assert ind[:50].sum() == 50 and ind[50:].sum() == 0
    assert np.isfinite(X).all()


def test_transform_rejects_missing_columns():
    pre = Preprocessor.for_track("A").fit(_frame())
    with pytest.raises(ValueError, match="lacks required columns"):
        pre.transform(_frame().drop(columns=["sbytes"]))


def test_encode_targets():
    df = pd.DataFrame({"attack_cat": ["Normal", "Worms"], "label": [0, 1]})
    y, yb = encode_targets(df)
    assert y.tolist() == [0, 9] and yb.tolist() == [0, 1]


def test_validation_split_keeps_identical_records_together():
    df = _frame(n=700)
    df = pd.concat([df, df.iloc[:200]], ignore_index=True)    # exact duplicates
    tr, va = track_a_validation_split(df)
    assert len(set(tr) & set(va)) == 0
    key = pd.util.hash_pandas_object(df[schema.TRACK_A_FEATURES], index=False)
    assert not set(key.iloc[tr]) & set(key.iloc[va])


def test_track_b_blocks_are_chronological_with_gap():
    lt = np.arange(0, 10_000, 1, dtype=np.int64) + 1_424_000_000   # 18 Feb session
    block, cuts = track_b_blocks(lt, lt, gap=100)
    c = cuts["2015-02-18"]
    assert lt[block == "train"].max() <= c["train_end"]
    assert lt[block == "val"].min() > c["train_end"] + 100
    assert lt[block == "test"].min() > c["val_end"] + 100
    assert lt[block == "train"].max() < lt[block == "val"].min() < lt[block == "test"].min()
    assert (block == "gap").sum() == 200


def test_windows_stay_within_host_and_order():
    host = np.array(["a", "b", "a", "a", "b", "a"])
    ltime = np.array([1, 1, 2, 3, 4, 5])
    X = np.arange(6, dtype=np.float32)[:, None] + 1          # record i has value i+1
    wi = WindowIndex.build(host, ltime)
    win, lengths = wi.gather(X, np.array([5, 4, 0]), length=3)
    assert win[0, :, 0].tolist() == [3, 4, 6]                 # a: records 2, 3, 5
    assert win[1, :, 0].tolist() == [0, 2, 5] and lengths[1] == 2   # b: records 1, 4
    assert win[2, :, 0].tolist() == [0, 0, 1] and lengths[2] == 1


def test_class_weights_and_oversampling():
    y = np.array([0] * 900 + [1] * 90 + [2] * 10)
    w = class_weights(y, 3, "sqrt")
    assert w[2] > w[1] > w[0]
    assert np.isclose((w[y]).mean(), 1.0)
    assert np.isclose(class_weights(y, 3, "balanced")[2] / class_weights(y, 3, "balanced")[0], 90)
    idx = random_oversample(y, min_count=100)
    assert np.bincount(y[idx]).tolist() == [900, 100, 100]
