"""Loaders for both tracks. Raw files are only ever read, never written.

Track B parsing handles the quirks verified in Phase 1 (see
docs/dataset_profile.md, section 7):
  * UTF-8 byte-order mark at the start of UNSW-NB15_1.csv;
  * ports given as hexadecimal ("0x000b") or "-" (no port, e.g. ICMP/ARP);
  * blank or whitespace-only values in ct_flw_http_mthd, is_ftp_login and
    ct_ftp_cmd (parsed as missing);
  * attack_cat padded with spaces, "Backdoors" vs "Backdoor", and empty for
    normal traffic.
Every repair is counted and saved next to the cached parquet, so the
profile reports exactly what was changed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from deepshield.config import ensure_dir, path
from deepshield.data import schema
from deepshield.data.files import TRACK_B_ROLES, resolve

log = logging.getLogger(__name__)

_RAW_STRING_COLS = ["srcip", "sport", "dstip", "dsport", "proto", "state",
                    "service", "ct_flw_http_mthd", "is_ftp_login",
                    "ct_ftp_cmd", "attack_cat"]
_RAW_BLANKABLE = ["ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd"]
_INT_COLS = ["sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "spkts",
             "dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz",
             "trans_depth", "res_bdy_len", "stime", "ltime",
             "is_sm_ips_ports", "ct_state_ttl", "ct_srv_src", "ct_srv_dst",
             "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
             "ct_dst_sport_ltm", "ct_dst_src_ltm", "label"]


# ── Track A ───────────────────────────────────────────────────────────────────

def load_track_a(split: str, cache: bool = True) -> pd.DataFrame:
    """Official training ("train") or testing ("test") partition.

    The split is resolved by checksum (see deepshield.data.files): "train" is
    the 175,341-record file, "test" the 82,332-record file.
    """
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    cached = path("interim") / f"track_a_{split}.parquet"
    if cache and cached.exists():
        return pd.read_parquet(cached)
    df = pd.read_csv(resolve(f"track_a_{split}"), low_memory=False)
    expected = ["id"] + schema.TRACK_A_FEATURES + ["attack_cat", "label"]
    if list(df.columns) != expected:
        raise ValueError(f"Unexpected Track A columns: {list(df.columns)}")
    df["attack_cat"] = schema.normalise_attack_cat(df["attack_cat"])
    for c in schema.CATEGORICAL:
        df[c] = df[c].astype(str)
    if cache:
        df.to_parquet(ensure_dir(cached.parent) / cached.name, index=False)
    return df


# ── Track B ───────────────────────────────────────────────────────────────────

def _parse_port(s: pd.Series, stats: dict, name: str) -> pd.Series:
    s = s.str.strip()
    dash = s.isin(["-", ""]) | s.isna()
    hexm = s.str.startswith("0x", na=False)
    out = pd.Series(-1, index=s.index, dtype="int64")
    dec = ~dash & ~hexm
    out[dec] = pd.to_numeric(s[dec], errors="coerce").fillna(-1).astype("int64")
    out[hexm] = s[hexm].map(lambda v: int(v, 16)).astype("int64")
    stats[f"{name}_missing_dash"] = int(dash.sum())
    stats[f"{name}_hex"] = int(hexm.sum())
    stats[f"{name}_out_of_range"] = int((out > 65535).sum())
    return out


def _read_raw_file(p: Path, file_no: int) -> tuple[pd.DataFrame, dict]:
    dtypes = {c: str for c in _RAW_STRING_COLS}
    dtypes.update({c: "float64" for c in schema.RAW_COLUMNS
                   if c not in _RAW_STRING_COLS})
    try:
        df = pd.read_csv(p, header=None, names=schema.RAW_COLUMNS,
                         dtype=dtypes, encoding="utf-8-sig",
                         keep_default_na=False, na_values=[])
    except UnicodeDecodeError:  # pragma: no cover - not observed in Phase 1
        df = pd.read_csv(p, header=None, names=schema.RAW_COLUMNS,
                         dtype=dtypes, encoding="latin-1",
                         keep_default_na=False, na_values=[])
    stats: dict = {"file": p.name, "records": len(df)}

    df["sport"] = _parse_port(df["sport"], stats, "sport")
    df["dsport"] = _parse_port(df["dsport"], stats, "dsport")
    for c in _RAW_BLANKABLE:
        stripped = df[c].str.strip()
        blank = stripped == ""
        stats[f"{c}_blank"] = int(blank.sum())
        df[c] = pd.to_numeric(stripped.where(~blank), errors="raise")
    stats["is_ftp_login_values"] = {
        str(k): int(v) for k, v in df["is_ftp_login"].value_counts(dropna=False).items()}

    raw_cat = df["attack_cat"]
    stats["attack_cat_raw_values"] = {
        repr(k): int(v) for k, v in raw_cat.value_counts().items()}
    df["attack_cat"] = schema.normalise_attack_cat(raw_cat)

    for c in _INT_COLS:
        if df[c].isna().any() or (df[c] % 1 != 0).any():
            raise ValueError(f"{p.name}: non-integer values in {c}")
        df[c] = df[c].astype("int64")
    for c in ["proto", "state", "service", "srcip", "dstip"]:
        df[c] = df[c].str.strip()

    df = df.rename(columns=schema.RAW_TO_TRACK_A)
    df.insert(0, "source_row", np.arange(len(df), dtype=np.int32))
    df.insert(0, "source_file", np.int8(file_no))
    return df, stats


def build_track_b_cache(force: bool = False) -> Path:
    """Parse the four main files into data/interim/track_b/part-N.parquet."""
    out_dir = ensure_dir(path("interim") / "track_b")
    stats_path = out_dir / "parse_stats.json"
    if stats_path.exists() and not force:
        return out_dir
    all_stats = []
    for i, role in enumerate(TRACK_B_ROLES, start=1):
        log.info("parsing %s", role)
        df, stats = _read_raw_file(resolve(role), i)
        df.to_parquet(out_dir / f"part-{i}.parquet", index=False)
        all_stats.append(stats)
        del df
    stats_path.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
    return out_dir


def load_track_b(columns: list[str] | None = None) -> pd.DataFrame:
    """All Track B records in original file order (file 1..4, row order)."""
    out_dir = build_track_b_cache()
    parts = [pd.read_parquet(out_dir / f"part-{i}.parquet", columns=columns)
             for i in range(1, 5)]
    return pd.concat(parts, ignore_index=True)


def track_b_parse_stats() -> list[dict]:
    return json.loads((build_track_b_cache() / "parse_stats.json")
                      .read_text(encoding="utf-8"))
