#!/usr/bin/env python3
"""Phase 1: validate and profile the acquired UNSW-NB15 files.

Reads the raw files (read-only) and writes every measured number to
reports/dataset/profile.json, which docs/dataset_profile.md and the
figure script use. Nothing is assumed from online descriptions.

Usage:  .venv\\Scripts\\python.exe scripts\\profile_dataset.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepshield.config import PROJECT_ROOT, ensure_dir  # noqa: E402
from deepshield.data import schema  # noqa: E402
from deepshield.data.files import FILES, resolve, verify_all  # noqa: E402
from deepshield.data.loaders import (load_track_a, load_track_b,  # noqa: E402
                                     track_b_parse_stats)

OUT = ensure_dir(PROJECT_ROOT / "reports" / "dataset") / "profile.json"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC")


def _row_hash(df: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(df, index=False).to_numpy()


def file_checks() -> list[dict]:
    rows = verify_all()
    for r, f in zip(rows, FILES):
        r["role_resolved_by_checksum"] = f.role
        r["expected_records"] = f.records
    return rows


def profile_track_a() -> dict:
    tr, te = load_track_a("train"), load_track_a("test")
    out: dict = {"columns": list(tr.columns),
                 "dtypes": {c: str(t) for c, t in tr.dtypes.items()},
                 "numeric_features": schema.TRACK_A_NUMERIC,
                 "categorical_features": schema.CATEGORICAL}
    feats = schema.TRACK_A_FEATURES
    for name, df in (("train", tr), ("test", te)):
        num = df[schema.TRACK_A_NUMERIC]
        out[name] = {
            "rows": len(df), "columns": df.shape[1],
            "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().items()},
            "attack_cat_counts": {k: int(df["attack_cat"].eq(k).sum()) for k in schema.CLASS_ORDER},
            "missing_total": int(df.isna().sum().sum()),
            "missing_by_column": {c: int(v) for c, v in df.isna().sum().items() if v},
            "infinite_total": int(np.isinf(num.to_numpy()).sum()),
            "negative_by_column": {c: int(v) for c, v in (num < 0).sum().items() if v},
            "id_unique": bool(df["id"].is_unique),
            "id_range": [int(df["id"].min()), int(df["id"].max())],
            "label_consistent_with_attack_cat": bool(
                ((df["attack_cat"] == "Normal") == (df["label"] == 0)).all()),
            "exact_duplicates_excl_id": int(df.drop(columns="id").duplicated().sum()),
            "feature_duplicates": int(df[feats].duplicated().sum()),
            "categorical_levels": {c: int(df[c].nunique()) for c in schema.CATEGORICAL},
            "binary_like_values": {c: sorted(map(int, df[c].unique()))
                                   for c in ["is_ftp_login", "is_sm_ips_ports"]},
        }
        g = df.groupby(feats, sort=False, dropna=False)["attack_cat"].nunique()
        out[name]["conflicting_feature_groups"] = int((g > 1).sum())
        out[name]["rows_in_conflicting_groups"] = int(
            df.merge(g[g > 1].rename("n").reset_index(), on=feats, how="inner").shape[0])
        dup_rows = df[df[feats].duplicated(keep=False)]
        out[name]["duplicate_rows_by_class"] = {
            k: int(v) for k, v in dup_rows["attack_cat"].value_counts().items()}
    # imbalance
    c = out["train"]["attack_cat_counts"]
    out["imbalance_ratio_majority_to_minority_train"] = round(max(c.values()) / min(c.values()), 1)
    # cross-split overlap (test feature vectors that also occur in training)
    h_tr = set(_row_hash(tr[feats]))
    in_train = np.fromiter((h in h_tr for h in _row_hash(te[feats])), bool, len(te))
    out["test_rows_with_feature_vector_in_train"] = int(in_train.sum())
    out["test_rows_with_feature_vector_in_train_by_class"] = {
        k: int(v) for k, v in te.loc[in_train, "attack_cat"].value_counts().items()}
    # categorical levels
    out["categorical"] = {}
    for col in schema.CATEGORICAL:
        trv, tev = set(tr[col]), set(te[col])
        out["categorical"][col] = {
            "train_levels": len(trv), "test_levels": len(tev),
            "test_only_levels": sorted(tev - trv),
            "test_rows_with_unseen_level": int(te[col].isin(tev - trv).sum()),
            "top_train": {k: int(v) for k, v in tr[col].value_counts().head(8).items()},
        }
    # numeric summaries (training data only)
    desc = {}
    for col in schema.TRACK_A_NUMERIC:
        s = tr[col]
        desc[col] = {"min": float(s.min()), "median": float(s.median()),
                     "mean": float(s.mean()), "max": float(s.max()),
                     "zero_fraction": round(float((s == 0).mean()), 4),
                     "skew": round(float(s.skew()), 2), "unique": int(s.nunique())}
    out["numeric_summary_train"] = desc
    # Spearman correlation on a fixed training sample (rank-based: robust to the
    # heavy tails); only pairs with |rho| >= 0.9 are listed.
    sample = tr[schema.TRACK_A_NUMERIC].sample(50_000, random_state=0)
    const = [c for c in sample if sample[c].nunique() <= 1]
    corr = sample.drop(columns=const).corr(method="spearman")
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iat[i, j]
            if abs(r) >= 0.9:
                pairs.append({"a": cols[i], "b": cols[j], "rho": round(float(r), 3)})
    out["spearman_high_pairs_train_sample"] = sorted(pairs, key=lambda p: -abs(p["rho"]))
    out["spearman_constant_columns_in_sample"] = const
    return out


def verify_ct_window(df: pd.DataFrame) -> dict:
    """Test how the ct_* counters were computed (section 7 of the profile).

    The features file says e.g. ct_src_ltm = "No. of connections of the same
    source address in 100 connections according to the last time". We
    recompute that count under several window definitions (file order, which
    is ltime order) and report the exact-match rate. We also count records
    whose value is LARGER than the number of earlier records with the same
    key (+1 for itself): such values are impossible for any backward-looking
    window, which proves the released counters also count later records.
    """
    res: dict = {"records_tested": len(df), "match_rate": {}}
    df = df.reset_index(drop=True)
    n = len(df)
    for col, keys in [("ct_src_ltm", ["srcip"]), ("ct_dst_ltm", ["dstip"]),
                      ("ct_dst_src_ltm", ["srcip", "dstip"])]:
        key = pd.util.hash_pandas_object(df[keys], index=False).to_numpy()
        target = df[col].to_numpy()
        back = np.ones(n, dtype=np.int64)
        fwd = np.zeros(n, dtype=np.int64)
        for lag in range(1, 100):
            same = key[lag:] == key[:-lag]
            back[lag:] += same
            fwd[:-lag] += same
        blk = pd.Series(1, index=pd.MultiIndex.from_arrays(
            [np.arange(n) // 100, key])).groupby(level=[0, 1]).transform("size").to_numpy()
        res["match_rate"][col] = {
            "previous_100_incl_self": round(float((back == target).mean()), 4),
            "next_100_incl_self": round(float((fwd + 1 == target).mean()), 4),
            "fixed_blocks_of_100": round(float((blk == target).mean()), 4),
        }
        cum_prev = pd.Series(key).groupby(key).cumcount().to_numpy() + 1
        res.setdefault("values_exceeding_all_earlier_same_key", {})[col] = int(
            (target > cum_prev).sum())
    res["examples_first_rows"] = df.head(6)[["srcip", "dstip", "ltime",
                                              "ct_src_ltm"]].astype(str).to_dict("records")
    return res


def profile_track_b(list_events: pd.DataFrame) -> dict:
    out: dict = {"parse_stats": track_b_parse_stats()}
    cols_needed = ["source_file", "source_row", "srcip", "dstip", "stime",
                   "ltime", "attack_cat", "label"]
    b = load_track_b(cols_needed)
    out["rows"] = len(b)
    out["rows_by_file"] = {int(k): int(v) for k, v in b["source_file"].value_counts().sort_index().items()}
    out["label_counts"] = {str(k): int(v) for k, v in b["label"].value_counts().items()}
    out["attack_cat_counts"] = {k: int(b["attack_cat"].eq(k).sum()) for k in schema.CLASS_ORDER}
    out["label_consistent_with_attack_cat"] = bool(
        ((b["attack_cat"] == "Normal") == (b["label"] == 0)).all())
    out["label_inconsistent_rows"] = int(((b["attack_cat"] == "Normal") != (b["label"] == 0)).sum())
    # time structure
    days = pd.to_datetime(b["stime"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    out["time"] = {
        "stime_min": _iso(b["stime"].min()), "stime_max": _iso(b["stime"].max()),
        "by_file": {int(f): {"stime_min": _iso(g["stime"].min()), "stime_max": _iso(g["stime"].max()),
                             "fraction_rows_where_stime_decreases": round(float((g["stime"].diff() < 0).mean()), 4)}
                    for f, g in b.groupby("source_file")},
        "records_by_utc_day": {k: int(v) for k, v in days.value_counts().sort_index().items()},
        "attacks_by_utc_day": {k: int(v) for k, v in days[b["label"] == 1].value_counts().sort_index().items()},
        "ltime_minus_stime_negative": int((b["ltime"] < b["stime"]).sum()),
        "fraction_rows_where_ltime_decreases_within_file": round(float(
            b.groupby("source_file")["ltime"].diff().lt(0).mean()), 4),
    }
    hourly = b.assign(hour=(b["stime"] // 3600) * 3600).groupby(["hour", "label"]).size().unstack(fill_value=0)
    out["hourly_counts"] = {"hour_epoch": [int(h) for h in hourly.index],
                            "normal": hourly.get(0, pd.Series(0, hourly.index)).astype(int).tolist(),
                            "attack": hourly.get(1, pd.Series(0, hourly.index)).astype(int).tolist()}
    # entities
    out["unique_srcip"] = int(b["srcip"].nunique())
    out["unique_dstip"] = int(b["dstip"].nunique())
    per_ip = b.groupby("srcip")["label"].agg(["size", "mean"])
    mixed = per_ip[(per_ip["mean"] > 0) & (per_ip["mean"] < 1)]
    out["srcip_label_purity"] = {
        "ips_all_normal": int((per_ip["mean"] == 0).sum()),
        "ips_all_attack": int((per_ip["mean"] == 1).sum()),
        "ips_mixed": int(len(mixed)),
        "records_from_mixed_ips": int(mixed["size"].sum()),
        "mixed_ips_detail": {ip: {"records": int(r["size"]), "attack_fraction": round(float(r["mean"]), 4)}
                             for ip, r in mixed.sort_values("size", ascending=False).head(15).iterrows()},
    }
    att = b[b["label"] == 1]
    out["attack_source_prefixes"] = {k: int(v) for k, v in
                                     att["srcip"].str.rsplit(".", n=1).str[0].value_counts().head(8).items()}
    out["normal_source_prefixes"] = {k: int(v) for k, v in
                                     b.loc[b["label"] == 0, "srcip"].str.rsplit(".", n=1).str[0].value_counts().head(8).items()}
    del b, att, days

    # duplicates, processed one file at a time to bound memory
    feats = schema.TRACK_B_FEATURES
    full_cols = [c for c in pd.read_parquet(
        PROJECT_ROOT / "data/interim/track_b/part-1.parquet").columns
        if c not in ("source_file", "source_row")]
    h_full, h_feat, cats = [], [], []
    missing = {}
    for i in range(1, 5):
        part = pd.read_parquet(PROJECT_ROOT / f"data/interim/track_b/part-{i}.parquet")
        h_full.append(_row_hash(part[full_cols]))
        h_feat.append(_row_hash(part[feats]))
        cats.append(part["attack_cat"].to_numpy())
        for c, v in part.isna().sum().items():
            if v:
                missing[c] = missing.get(c, 0) + int(v)
        if i == 1:
            out["ct_window_verification_first_100k_of_file1"] = verify_ct_window(
                part.iloc[:100_000][["srcip", "dstip", "ltime", "ct_src_ltm",
                                     "ct_dst_ltm", "ct_dst_src_ltm"]])
        del part
    h_full, h_feat, cats = np.concatenate(h_full), np.concatenate(h_feat), np.concatenate(cats)
    out["missing_by_column"] = missing
    out["exact_duplicates"] = int(pd.Series(h_full).duplicated().sum())
    fd = pd.DataFrame({"h": h_feat, "cat": cats})
    out["feature_duplicates"] = int(fd["h"].duplicated().sum())
    out["feature_duplicates_by_class"] = {
        k: int(v) for k, v in fd.loc[fd["h"].duplicated(), "cat"].value_counts().items()}
    g = fd.groupby("h")["cat"].nunique()
    out["conflicting_feature_groups"] = int((g > 1).sum())
    out["rows_in_conflicting_groups"] = int(fd["h"].isin(g[g > 1].index).sum())

    # comparison with the official event list
    le = list_events[list_events["Attack category"].notna()].copy()
    out["list_events_total_row"] = int(
        list_events.loc[list_events["Attack category"].isna(), "Number of events"].sum())
    le["cat"] = schema.normalise_attack_cat(le["Attack category"].replace({"normal": ""}))
    out["list_events_totals"] = {k: int(v) for k, v in le.groupby("cat")["Number of events"].sum().items()}
    out["list_events_subcategories"] = int(le["Attack subcategory"].notna().sum())
    return out


def profile_ground_truth() -> dict:
    gt = pd.read_csv(resolve("ground_truth"), encoding="latin-1", low_memory=False)
    cat = schema.normalise_attack_cat(gt["Attack category"])
    return {"rows": len(gt), "columns": list(gt.columns),
            "attack_category_counts": {k: int(v) for k, v in cat.value_counts().items()},
            "unique_attack_names": int(gt["Attack Name"].nunique())}


def main() -> None:
    t0 = time.time()
    feats = pd.read_csv(resolve("features"), encoding="latin-1")
    le = pd.read_csv(resolve("list_events"), encoding="latin-1")
    raw_names = [schema.normalise_feature_name(n) for n in feats["Name"]]
    profile = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "files": file_checks(),
        "features_file": {
            "rows": len(feats), "columns": list(feats.columns),
            "names_match_schema": raw_names == schema.RAW_COLUMNS,
            "raw_names_as_written": list(feats["Name"]),
            "types": {schema.normalise_feature_name(n): t.strip()
                      for n, t in zip(feats["Name"], feats["Type "])},
            "descriptions": {schema.normalise_feature_name(n): str(d).strip()
                             for n, d in zip(feats["Name"], feats["Description"])},
        },
        "column_differences": {
            "track_a_only": sorted(set(schema.TRACK_A_FEATURES) - set(schema.TRACK_B_FEATURES)) + ["id"],
            "track_b_only": schema.TRACK_B_ID_COLUMNS,
            "renamed_raw_to_track_a": schema.RAW_TO_TRACK_A,
        },
        "track_a": profile_track_a(),
    }
    print("Track A done", round(time.time() - t0), "s", flush=True)
    profile["track_b"] = profile_track_b(le)
    print("Track B done", round(time.time() - t0), "s", flush=True)
    profile["ground_truth"] = profile_ground_truth()
    OUT.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    ok = all(r["present"] and r["sha256_ok"] for r in profile["files"])
    print(f"wrote {OUT.relative_to(PROJECT_ROOT)}; all files verified: {ok}; "
          f"{round(time.time() - t0)} s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
