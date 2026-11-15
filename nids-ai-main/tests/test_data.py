"""Phase 1 tests: dataset registry, parsing repairs and schema consistency.

Tests marked `dataset` need the real files in data/raw/unsw_nb15 and are
skipped automatically when they are absent (e.g. on a fresh clone).
"""

from pathlib import Path

import pandas as pd
import pytest

from deepshield.data import schema
from deepshield.data.files import BY_ROLE, FILES, file_path, sha256
from deepshield.data.loaders import _read_raw_file

HAVE_DATA = all(file_path(f.role).exists() for f in FILES if f.required)
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="UNSW-NB15 files not present")


def test_schema_lists_are_consistent():
    assert len(schema.RAW_COLUMNS) == 49
    assert len(schema.TRACK_A_FEATURES) == 42
    assert len(schema.TRACK_A_NUMERIC) == 39
    renamed = {schema.RAW_TO_TRACK_A.get(c, c) for c in schema.RAW_COLUMNS}
    # every Track B feature exists in the renamed raw columns
    assert set(schema.TRACK_B_FEATURES) <= renamed
    assert set(schema.TRACK_A_FEATURES) - renamed == {"rate"}


def test_normalise_attack_cat():
    s = pd.Series([" Fuzzers ", "Backdoors", "Backdoor", "", None, "Worms"])
    assert schema.normalise_attack_cat(s).tolist() == [
        "Fuzzers", "Backdoor", "Backdoor", "Normal", "Normal", "Worms"]


def test_track_a_roles_follow_content_not_filename():
    # The swap documented in docs/dataset_profile.md section 2.3.
    assert BY_ROLE["track_a_train"].filename == "UNSW_NB15_testing-set.csv"
    assert BY_ROLE["track_a_train"].records == 175_341
    assert BY_ROLE["track_a_test"].records == 82_332


def test_raw_parser_repairs_known_quirks(tmp_path: Path):
    """A synthetic two-row main file containing every quirk seen in Phase 1."""
    base = ["0"] * 49
    r1 = list(base)
    r1[0], r1[1], r1[2], r1[3] = "59.166.0.0", "0x000b", "149.171.126.6", "-"
    r1[4], r1[5], r1[13] = "tcp", "FIN", "-"
    r1[37], r1[38], r1[39] = " ", "", " "      # blank counters
    r1[47], r1[48] = " Fuzzers ", "1"
    r2 = list(base)
    r2[0], r2[1], r2[2], r2[3] = "175.45.176.1", "1043", "149.171.126.9", "80"
    r2[4], r2[5], r2[13] = "udp", "INT", "dns"
    r2[47], r2[48] = "Backdoors", "1"
    p = tmp_path / "mini.csv"
    p.write_text("﻿" + ",".join(r1) + "\n" + ",".join(r2) + "\n", encoding="utf-8")
    df, stats = _read_raw_file(p, 1)
    assert df.loc[0, "srcip"] == "59.166.0.0"          # BOM removed
    assert df.loc[0, "sport"] == 11 and df.loc[0, "dsport"] == -1
    assert df["attack_cat"].tolist() == ["Fuzzers", "Backdoor"]
    assert pd.isna(df.loc[0, "ct_flw_http_mthd"]) and pd.isna(df.loc[0, "is_ftp_login"])
    assert stats["sport_hex"] == 1 and stats["dsport_missing_dash"] == 1
    assert "smean" in df.columns and "smeansz" not in df.columns


@needs_data
def test_dataset_checksums_match_registry():
    for f in FILES:
        p = file_path(f.role)
        if p.exists():
            assert sha256(p) == f.sha256, f.filename


@needs_data
def test_track_a_loads_with_documented_counts():
    from deepshield.data.loaders import load_track_a
    tr, te = load_track_a("train"), load_track_a("test")
    assert tr.shape == (175_341, 45) and te.shape == (82_332, 45)
    assert tr["attack_cat"].value_counts()["Worms"] == 130
    assert set(tr["attack_cat"]) == set(schema.CLASS_ORDER)
    assert ((tr["attack_cat"] == "Normal") == (tr["label"] == 0)).all()


@needs_data
def test_track_b_attack_counts_match_official_list_events():
    from deepshield.data.loaders import load_track_b
    b = load_track_b(["attack_cat", "label"])
    le = pd.read_csv(file_path("list_events"), encoding="latin-1")
    le = le[le["Attack category"].notna() & (le["Attack category"].str.strip() != "normal")]
    official = le.assign(c=schema.normalise_attack_cat(le["Attack category"])) \
                 .groupby("c")["Number of events"].sum()
    ours = b.loc[b["label"] == 1, "attack_cat"].value_counts()
    assert official.to_dict() == ours.to_dict()
    assert len(b) == 2_540_047
