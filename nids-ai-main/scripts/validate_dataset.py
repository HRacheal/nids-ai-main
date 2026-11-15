#!/usr/bin/env python3
"""Check that every UNSW-NB15 file is present, unmodified and well-formed.

For each registered file: presence, size, SHA-256, and (for CSVs) the record
count and field count of every record. Exits non-zero on any problem and
prints where to get missing files.

Usage:  .venv\\Scripts\\python.exe scripts\\validate_dataset.py [--quick]
        --quick  checksums only (skips the record-by-record parse)
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepshield.data.files import FILES, file_path, raw_dir, sha256  # noqa: E402

EXPECTED_FIELDS = {"track_a_train": 45, "track_a_test": 45, "track_b_1": 49,
                   "track_b_2": 49, "track_b_3": 49, "track_b_4": 49,
                   "features": 4, "list_events": 3, "ground_truth": 12}
HAS_HEADER = {"track_a_train", "track_a_test", "features", "list_events",
              "ground_truth"}


def count_records(p: Path, has_header: bool) -> tuple[int, Counter]:
    widths: Counter = Counter()
    n = 0
    with open(p, newline="", encoding="latin-1") as fh:
        reader = csv.reader(fh)
        if has_header:
            next(reader)
        for row in reader:
            n += 1
            widths[len(row)] += 1
    return n, widths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    print(f"Dataset folder: {raw_dir()}")
    problems = 0
    for f in FILES:
        p = file_path(f.role)
        tag = "required" if f.required else "recommended"
        if not p.exists():
            print(f"  MISSING  {f.filename} ({tag}, role {f.role})")
            problems += f.required
            continue
        ok_hash = sha256(p) == f.sha256
        line = f"  {'OK ' if ok_hash else 'BAD'}      {f.filename:32} {p.stat().st_size:>12,} bytes  sha256 {'ok' if ok_hash else 'MISMATCH'}"
        if not ok_hash:
            problems += 1
        if not args.quick and f.records is not None:
            n, widths = count_records(p, f.role in HAS_HEADER)
            exp_w = EXPECTED_FIELDS[f.role]
            ok_n = n == f.records or (f.role == "list_events" and n == f.records)
            ok_w = set(widths) == {exp_w}
            line += f"  records {n:,} ({'ok' if ok_n else 'expected ' + format(f.records, ',')})"
            line += f"  fields {'ok' if ok_w else dict(widths)}"
            problems += (not ok_n) + (not ok_w)
        print(line)
    if problems:
        print(f"\n{problems} problem(s). See docs/dataset_setup.md to obtain the files.")
        return 1
    print("\nAll dataset files verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
