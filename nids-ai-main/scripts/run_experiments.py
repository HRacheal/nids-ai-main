#!/usr/bin/env python3
"""Run the experiment pipeline in order, one job at a time.

Each step is resumable (finished runs are cached), so after an interruption
simply run this script again. Steps whose outputs already exist are skipped
unless --force is given.

  1  scripts/train_baselines.py        RF, Isolation Forest        (Track A)
  2  scripts/train_deep.py             MLP, 1D-CNN, Autoencoder    (Track A)
  3  scripts/train_sequence.py         LSTM + comparisons          (Track B)
  4  scripts/sampling_study.py         training fraction for LOCO  (Track A, validation only)
  5  scripts/leave_one_class_out.py    unseen-attack experiment    (Track A)
  6  scripts/explain_models.py         SHAP / Integrated Gradients (Track A)
  7  scripts/compare_models.py         tables, paired tests, figures

Usage:  .venv\\Scripts\\python.exe scripts\\run_experiments.py [--from N] [--force]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "artifacts" / "metrics"
STEPS = [
    ("train_baselines.py", M / "track_a" / "rf_multiclass.json"),
    ("train_deep.py", M / "track_a" / "autoencoder.json"),
    ("train_sequence.py", M / "track_b" / "oracle_context.json"),
    ("sampling_study.py", M / "track_a" / "sampling_study.json"),
    ("leave_one_class_out.py", M / "track_a" / "leave_one_class_out.json"),
    ("explain_models.py", M / "track_a" / "explanations.json"),
    ("compare_models.py", M / "comparison.json"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONWARNINGS": "ignore"}
    for i, (script, output) in enumerate(STEPS, start=1):
        if i < args.start:
            continue
        if output.exists() and not args.force and script != "compare_models.py":
            print(f"[{i}] {script}: output exists, skipped", flush=True)
            continue
        print(f"[{i}] {script} ...", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / script)], cwd=ROOT, env=env)
        print(f"[{i}] {script}: exit {r.returncode} after {time.time() - t0:.0f} s", flush=True)
        if r.returncode != 0:
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
