#!/usr/bin/env python3
"""Phase 2: build the processed datasets and fitted preprocessors.

Track A -> data/processed/track_a/   (X_*.npy, y_*.npy, meta)
Track B -> data/processed/track_b/   (X.npy for all non-gap records, meta.parquet)
Preprocessors -> artifacts/models/preprocessor_track_{a,b}.joblib
Summary -> reports/preprocessing/split_summary.json

Every preprocessing statistic is fitted on the training portion only
(Track A: official training set minus the validation fold; Track B: the
chronological training block), then applied unchanged to the other splits.

Usage:  .venv\\Scripts\\python.exe scripts\\prepare_data.py [--track A|B|both]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepshield.config import PROJECT_ROOT, ensure_dir, path  # noqa: E402
from deepshield.data import schema  # noqa: E402
from deepshield.data.loaders import load_track_a, load_track_b  # noqa: E402
from deepshield.data.preprocess import Preprocessor, encode_targets  # noqa: E402
from deepshield.data.splits import (GAP_SECONDS, feature_groups,  # noqa: E402
                                    track_a_validation_split, track_b_blocks)

SUMMARY = ensure_dir(PROJECT_ROOT / "reports" / "preprocessing") / "split_summary.json"
CHUNK = 250_000


def _counts(cats: pd.Series) -> dict:
    return {c: int((cats == c).sum()) for c in schema.CLASS_ORDER}


def _transform_chunked(pre: Preprocessor, df: pd.DataFrame) -> np.ndarray:
    out = np.empty((len(df), pre.n_features), dtype=np.float32)
    for s in range(0, len(df), CHUNK):
        out[s:s + CHUNK] = pre.transform(df.iloc[s:s + CHUNK])
    return out


def prepare_track_a(summary: dict) -> None:
    out = ensure_dir(path("processed") / "track_a")
    full = load_track_a("train")
    test = load_track_a("test")
    tr_idx, va_idx = track_a_validation_split(full, seed=0)
    train, val = full.iloc[tr_idx].reset_index(drop=True), full.iloc[va_idx].reset_index(drop=True)

    pre = Preprocessor.for_track("A").fit(train)
    pre.save(ensure_dir(path("models")) / "preprocessor_track_a.joblib")

    # test records whose exact feature vector occurs in the (whole) training set
    train_hashes = set(feature_groups(full, schema.TRACK_A_FEATURES))
    overlap = np.fromiter((h in train_hashes for h in feature_groups(test, schema.TRACK_A_FEATURES)),
                          bool, len(test))
    for name, df in (("train", train), ("val", val), ("test", test)):
        np.save(out / f"X_{name}.npy", pre.transform(df))
        y_multi, y_bin = encode_targets(df)
        np.save(out / f"y_{name}.npy", y_multi)
        np.save(out / f"ybin_{name}.npy", y_bin)
        np.save(out / f"id_{name}.npy", df["id"].to_numpy())
    np.save(out / "test_overlaps_train.npy", overlap)
    # raw test rows (features + labels) for the dashboard's "analyse records" demo
    (out / "feature_names.json").write_text(json.dumps(pre.feature_names_), encoding="utf-8")

    vh, th = set(feature_groups(val, schema.TRACK_A_FEATURES)), set(feature_groups(train, schema.TRACK_A_FEATURES))
    summary["track_a"] = {
        "train_rows": len(train), "val_rows": len(val), "test_rows": len(test),
        "val_fraction_of_official_train": round(len(val) / len(full), 4),
        "feature_vectors_shared_train_val": len(vh & th),
        "train_counts": _counts(train["attack_cat"]), "val_counts": _counts(val["attack_cat"]),
        "test_counts": _counts(test["attack_cat"]),
        "test_rows_overlapping_train": int(overlap.sum()),
        "n_features": pre.n_features,
        "feature_groups": {g: sum(1 for v in pre.group_of_.values() if v == g)
                           for g in dict.fromkeys(pre.group_of_.values())},
        "categorical_levels_kept": {c: len(v) for c, v in pre.levels_.items()},
    }
    print(f"Track A: {pre.n_features} features; train {len(train)}, val {len(val)}, test {len(test)}")


def prepare_track_b(summary: dict) -> None:
    out = ensure_dir(path("processed") / "track_b")
    meta_cols = ["source_file", "source_row", "srcip", "sport", "dstip", "dsport",
                 "stime", "ltime", "proto", "service", "state", "sbytes", "dbytes",
                 "dur", "attack_cat", "label"]
    feats = schema.TRACK_B_FEATURES
    df = load_track_b(list(dict.fromkeys(meta_cols + feats)))
    for c in feats:
        if c not in schema.CATEGORICAL and df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    block, cuts = track_b_blocks(df["stime"].to_numpy(), df["ltime"].to_numpy())
    df["block"] = block
    gap_rows = int((block == "gap").sum())
    df = df[df["block"] != "gap"].reset_index(drop=True)

    train_mask = (df["block"] == "train").to_numpy()
    pre = Preprocessor.for_track("B").fit(df.loc[train_mask])
    pre.save(ensure_dir(path("models")) / "preprocessor_track_b.joblib")

    X = _transform_chunked(pre, df)
    np.save(out / "X.npy", X)
    y_multi, y_bin = encode_targets(df)
    np.save(out / "y.npy", y_multi)
    np.save(out / "ybin.npy", y_bin)
    df[meta_cols + ["block"]].to_parquet(out / "meta.parquet", index=False)
    (out / "feature_names.json").write_text(json.dumps(pre.feature_names_), encoding="utf-8")

    blocks = {}
    for b in ("train", "val", "test"):
        m = df["block"] == b
        blocks[b] = {"rows": int(m.sum()), "counts": _counts(df.loc[m, "attack_cat"]),
                     "hosts": int(df.loc[m, "srcip"].nunique())}
    summary["track_b"] = {"gap_seconds": GAP_SECONDS, "cuts_utc_epoch": cuts,
                          "rows_dropped_in_gaps": gap_rows, "blocks": blocks,
                          "n_features": pre.n_features,
                          "missing_indicator_columns": pre.missing_cols_,
                          "imputed_medians": pre.medians_}
    print(f"Track B: {pre.n_features} features; " +
          ", ".join(f"{b} {v['rows']}" for b, v in blocks.items()) + f"; gap {gap_rows}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["A", "B", "both"], default="both")
    args = ap.parse_args()
    summary = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else {}
    t0 = time.time()
    if args.track in ("A", "both"):
        prepare_track_a(summary)
    if args.track in ("B", "both"):
        prepare_track_b(summary)
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {SUMMARY.relative_to(PROJECT_ROOT)} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
