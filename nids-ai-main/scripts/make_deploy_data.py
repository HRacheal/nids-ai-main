r"""Build the small data files the hosted (Render) dashboard needs.

Why: the full data (meta.parquet 45 MB, part-1..4.parquet 195 MB,
X_train.npy 107 MB, track_b/X.npy 1.9 GB) is too big for GitHub and for a
512 MB server. This script extracts only what the dashboard uses:

  artifacts/deploy/replay_test_block.pkl.gz  first N flows of the Track B test
                                             block, in replay order (same rows
                                             the local replay uses)
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


def build_replay(max_records: int | None) -> None:
    records, truth = ReplayEngine._load_test_block(max_records)
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
    args = ap.parse_args()

    meta = path("processed") / "track_b" / "meta.parquet"
    if not meta.exists():
        sys.exit(f"{meta} not found: run this on your own computer, not on the server.")

    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {DEPLOY_DIR}")
    build_baseline()
    build_replay(args.max_records or None)
    print("Done. Commit with:  git add -f artifacts/deploy")


if __name__ == "__main__":
    main()
