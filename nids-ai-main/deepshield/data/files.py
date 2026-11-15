"""Registry of the acquired UNSW-NB15 files, identified by SHA-256.

Files are resolved by checksum, not by filename. This matters because in the
acquired copies the official training/testing filenames are swapped relative
to the official documentation (see docs/dataset_profile.md, section 2):

  * official documentation: training set 175,341 records, testing set 82,332
  * file named UNSW_NB15_training-set.csv: 82,332 records
  * file named UNSW_NB15_testing-set.csv:  175,341 records

Project decision (approved by the project owner): follow the official
documentation by content. The 175,341-record file is the Track A training
set and the 82,332-record file is the Track A test set. Filenames on disk
are left unchanged.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from deepshield.config import path


@dataclass(frozen=True)
class DataFile:
    role: str
    filename: str           # name as delivered (never renamed)
    sha256: str
    records: int | None     # data records (excluding any header) verified in Phase 1
    required: bool
    subdir: str = ""


FILES = [
    DataFile("track_a_train", "UNSW_NB15_testing-set.csv",
             "bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa",
             175_341, True),
    DataFile("track_a_test", "UNSW_NB15_training-set.csv",
             "734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559",
             82_332, True),
    DataFile("track_b_1", "UNSW-NB15_1.csv",
             "7d851bbeabd27894ce39c8e78835c73341fc946652fb7743b9eff193b55eb511",
             700_001, True),
    DataFile("track_b_2", "UNSW-NB15_2.csv",
             "6130ad02873cc6069ae695cf2844f2e8c2e9a9a1b7532dd82ab8f202757cacf8",
             700_001, True),
    DataFile("track_b_3", "UNSW-NB15_3.csv",
             "ae990a96c3dfcd425ce2801aadb1727a34d5e0ae6d8215dbcdae60dedfaef640",
             700_001, True),
    DataFile("track_b_4", "UNSW-NB15_4.csv",
             "cdf563692d51d405541dd659ddcdad9fa01f001f05fe9fc4b67f00ca12fbc96a",
             440_044, True),
    DataFile("features", "NUSW-NB15_features.csv",
             "c55f19cceebb6360dc50f44f8a5f246ccefbcf8a6c604ac1ad46e643869cafce",
             49, True),
    DataFile("list_events", "UNSW-NB15_LIST_EVENTS.csv",
             "5b40f8128e2c87e691157c76debde04c328b0eb66bab97f1e7e17625f84497f2",
             208, False),
    DataFile("ground_truth", "NUSW-NB15_GT.csv",
             "6d27542cb6457db599e0a78274ac141fec0296531f00ac5569551a276c5ab1a8",
             174_347, False, subdir="_mirror2_alextamboli"),
]
BY_ROLE = {f.role: f for f in FILES}
TRACK_B_ROLES = ["track_b_1", "track_b_2", "track_b_3", "track_b_4"]


class DatasetFileError(RuntimeError):
    pass


def raw_dir() -> Path:
    return path("raw_unsw_nb15")


def file_path(role: str) -> Path:
    f = BY_ROLE[role]
    return raw_dir() / f.subdir / f.filename


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(role: str, verify: bool = False) -> Path:
    """Path of a dataset file, optionally re-verifying its checksum."""
    p = file_path(role)
    if not p.exists():
        raise DatasetFileError(
            f"Missing {BY_ROLE[role].filename} (role {role}) in {p.parent}. "
            "See docs/dataset_profile.md for download instructions.")
    if verify and sha256(p) != BY_ROLE[role].sha256:
        raise DatasetFileError(f"Checksum mismatch for {p}")
    return p


def verify_all() -> list[dict]:
    """Check presence, size and SHA-256 of every registered file."""
    rows = []
    for f in FILES:
        p = file_path(f.role)
        row = {"role": f.role, "filename": f.filename, "required": f.required,
               "present": p.exists(), "bytes": None, "sha256_ok": None}
        if p.exists():
            row["bytes"] = p.stat().st_size
            row["sha256_ok"] = sha256(p) == f.sha256
        rows.append(row)
    return rows
