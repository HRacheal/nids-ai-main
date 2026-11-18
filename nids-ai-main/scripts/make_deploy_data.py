r"""Build the small data files the hosted (Render) dashboard needs.

Why: the full data (meta.parquet 45 MB, part-1..4.parquet 195 MB,
X_train.npy 107 MB, track_b/X.npy 1.9 GB) is too big for GitHub and for a
512 MB server. This script extracts only what the dashboard uses:

  artifacts/deploy/replay_test_block.pkl.gz  N consecutive flows of the Track B
                                             test block, in replay order. By
                                             default the window with the most
                                             attack types is chosen (the start of
                                             the test block is almost all normal
                                             traffic, so "first N" shows no attacks)
  artifacts/deploy/track_a_baseline.npz      the 50 background rows + baseline
                                             mean used for explanations
                                             (identical to the local values)

Run from the inner nids-ai-main folder, with the venv active:
  python scripts/make_deploy_data.py                    (50,000 flows)
  python scripts/make_deploy_data.py --max-records 0    (all 366,522 flows; needs a 2 GB server)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from deepshield.config import path  # noqa: E402
from deepshield.inference.engine import (  # noqa: E402
    BASELINE_FILE,
    DEPLOY_DIR,
    REPLAY_FILE,
    ReplayEngine,
)


def build_baseline() -> None:
    src = path("processed") / "track_a" / "X_train.npy"
    X = np.load(src, mmap_mode="r")
    rng = np.random.default_rng(0)                      # same as TrackAEngine._base
    background = np.asarray(X[rng.choice(len(X), 50, replace=False)])
    baseline = np.asarray(X[::10]).mean(axis=0)
    np.savez_compressed(BASELINE_FILE, background=background, baseline=baseline)
    print(f"  {BASELINE_FILE.name}: {BASELINE_FILE.stat().st_size / 1e6:.2f} MB")


def pick_window(truth: pd.DataFrame, n: int, step: int = 2_000) -> int:
    """Start of the n-flow window with the most attack categories, and an attack
    share closest to the whole test block (so it looks like the full replay)."""
    total = len(truth)
    if n >= total:
        return 0
    lab = truth["label"].to_numpy()
    cats = truth["attack_cat"].to_numpy()
    overall = lab.mean()
    starts = list(range(0, total - n + 1, step)) + [total - n]
    best_score, best_start = None, 0
    for s in starts:
        wl, wc = lab[s:s + n], cats[s:s + n]
        score = (len(set(wc[wl == 1])), -abs(wl.mean() - overall))
        if best_score is None or score > best_score:
            best_score, best_start = score, s
    return best_start


def describe(name: str, truth: pd.DataFrame, records: pd.DataFrame) -> None:
    t0 = pd.to_datetime(records["ltime"].min(), unit="s")
    t1 = pd.to_datetime(records["ltime"].max(), unit="s")
    attacks = int(truth["label"].sum())
    print(f"  {name}: {len(truth):,} flows, {t0} -> {t1}, "
          f"{attacks:,} attacks ({attacks / max(len(truth), 1):.1%})")
    vc = truth.loc[truth["label"] == 1, "attack_cat"].value_counts()
    if len(vc):
        print("    attack types: " + ", ".join(f"{k} {v:,}" for k, v in vc.items()))


def build_replay(max_records: int | None, mode: str) -> None:
    records, truth = ReplayEngine._load_test_block(None)       # full test block, replay order
    records, truth = records.reset_index(drop=True), truth.reset_index(drop=True)
    describe("full test block", truth, records)
    if max_records and max_records < len(records):
        start = 0 if mode == "first" else pick_window(truth, max_records)
        records = records.iloc[start:start + max_records]
        truth = truth.iloc[start:start + max_records]
        describe(f"chosen window (flows {start:,}-{start + max_records:,})", truth, records)
    clash = {"attack_cat", "label"} & set(records.columns)
    if clash:
        sys.exit(f"Unexpected label columns among features: {clash}")
    df = pd.concat([records.reset_index(drop=True), truth.reset_index(drop=True)], axis=1)
    df.to_pickle(REPLAY_FILE)                           # gzip, keeps exact dtypes
    mb = REPLAY_FILE.stat().st_size / 1e6
    print(f"  {REPLAY_FILE.name}: {len(df):,} flows, {mb:.1f} MB")
    if mb > 95:
        print("  WARNING: over GitHub's 100 MB limit, use a smaller --max-records")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-records", type=int, default=50_000,
                    help="flows to keep for the hosted replay (0 = all). Default 50000")
    ap.add_argument("--mode", choices=["auto", "first"], default="auto",
                    help="auto: window with the most attack types (default); first: first N flows")
    args = ap.parse_args()

    meta = path("processed") / "track_b" / "meta.parquet"
    if not meta.exists():
        sys.exit(f"{meta} not found: run this on your own computer, not on the server.")

    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {DEPLOY_DIR}")
    build_baseline()
    build_replay(args.max_records or None, args.mode)
    print("Done. Commit with:  git add -f artifacts/deploy")


if __name__ == "__main__":
    main()
