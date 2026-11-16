#!/usr/bin/env python3
"""Build the submission package: dist/deepshield-ai-<date>[-with-models].zip

Contents = every file git tracks or would track (`git ls-files --cached
--others --exclude-standard`), so .gitignore decides what is left out: the
UNSW-NB15 dataset (data/, not redistributable, see docs/dataset_setup.md),
the virtual environment, predictions, caches and runtime databases.

--with-models also adds artifacts/models/ (trained models and preprocessors,
~130 MB) so the dashboard runs without retraining. The dataset is never
included in either variant.

A MANIFEST.sha256 (SHA-256 of every packaged file) is written into the zip.

Usage:  .venv\\Scripts\\python.exe scripts\\make_package.py [--with-models]
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEVER = ("data/", ".venv/", "dist/")          # belt and braces on top of .gitignore


def files(with_models: bool) -> list[str]:
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                         cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8")
    names = {n for n in out.split("\0") if n and (ROOT / n).is_file()}
    if with_models:
        names |= {p.relative_to(ROOT).as_posix() for p in (ROOT / "artifacts" / "models").rglob("*")
                  if p.is_file()}
    return sorted(n for n in names if not n.startswith(NEVER))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-models", action="store_true")
    args = ap.parse_args()
    names = files(args.with_models)
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    out = dist / f"deepshield-ai-{time.strftime('%Y%m%d')}{'-with-models' if args.with_models else ''}.zip"
    manifest = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for n in names:
            data = (ROOT / n).read_bytes()
            manifest.append(f"{hashlib.sha256(data).hexdigest()}  {n}")
            z.writestr(f"deepshield-ai/{n}", data)
        z.writestr("deepshield-ai/MANIFEST.sha256", "\n".join(manifest) + "\n")
    assert not any(n.startswith("data/") for n in names), "dataset must never be packaged"
    print(f"{out.relative_to(ROOT)}: {len(names)} files, {out.stat().st_size / 2**20:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
