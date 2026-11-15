"""Feature preprocessing, fitted on training data only.

One `Preprocessor` turns a table of flow records into a float32 matrix:

1. Missing values (Track B only, three counter columns): imputed with the
   training median, plus a 0/1 "was missing" indicator per column, because
   blankness depends on the source file rather than being random.
2. Heavy-tailed non-negative numeric features: log1p. Bounded features
   (TTLs, TCP windows, small counters, flags) are left on their own scale.
3. Standardisation (training mean/std), then clipping to +-CLIP so a single
   extreme test value cannot dominate a neural network's activations.
4. Categorical features: one-hot for levels seen at least `min_level_count`
   times in training; every other level, including levels never seen in
   training, goes to "<col>=other".

Columns are emitted in a fixed order grouped by meaning (basic flow, load
and content, TCP, timing, connection counters, flags, missing indicators,
categorical), which the 1D-CNN relies on (see docs/preprocessing.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from deepshield.data import schema

CLIP = 10.0

FEATURE_GROUPS: dict[str, list[str]] = {
    "basic": ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl",
              "dttl", "sloss", "dloss"],
    "load_content": ["sload", "dload", "smean", "dmean", "trans_depth",
                     "response_body_len"],
    "tcp": ["swin", "dwin", "stcpb", "dtcpb", "tcprtt", "synack", "ackdat"],
    "timing": ["sinpkt", "dinpkt", "sjit", "djit"],
    "counters": ["ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
                 "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
                 "ct_state_ttl", "ct_flw_http_mthd", "ct_ftp_cmd"],
    "flags": ["is_ftp_login", "is_sm_ips_ports"],
}
# Not log-transformed: bounded ranges where the raw scale is meaningful.
NO_LOG = {"sttl", "dttl", "swin", "dwin", "ct_state_ttl", "is_ftp_login",
          "is_sm_ips_ports"}
assert sorted(sum(FEATURE_GROUPS.values(), [])) == sorted(schema.TRACK_A_NUMERIC)


@dataclass
class Preprocessor:
    numeric: list[str]
    categorical: list[str] = field(default_factory=lambda: list(schema.CATEGORICAL))
    min_level_count: int = 50
    # fitted state
    medians_: dict = field(default_factory=dict)
    missing_cols_: list = field(default_factory=list)
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    levels_: dict = field(default_factory=dict)
    feature_names_: list = field(default_factory=list)
    group_of_: dict = field(default_factory=dict)

    @classmethod
    def for_track(cls, track: str, **kw) -> "Preprocessor":
        order = sum(FEATURE_GROUPS.values(), [])
        if track == "A":
            numeric = order
        elif track == "B":
            numeric = [c for c in order if c != "rate"]
        else:
            raise ValueError(track)
        return cls(numeric=numeric, **kw)

    # ── fitting ───────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        num = df[self.numeric]
        self.missing_cols_ = [c for c in self.numeric if num[c].isna().any()]
        self.medians_ = {c: float(num[c].median()) for c in self.missing_cols_}
        z = self._numeric_raw(df)
        self.mean_ = z.mean(axis=0)
        std = z.std(axis=0)
        self.std_ = np.where(std > 0, std, 1.0)
        self.levels_ = {}
        for c in self.categorical:
            vc = df[c].astype(str).value_counts()
            self.levels_[c] = sorted(vc[vc >= self.min_level_count].index)
        names, groups = [], {}
        for g, cols in FEATURE_GROUPS.items():
            for col in cols:
                if col in self.numeric:
                    names.append(col)
                    groups[col] = g
        for col in self.missing_cols_:
            names.append(f"{col}_missing")
            groups[names[-1]] = "missing"
        for c in self.categorical:
            for lv in self.levels_[c] + ["other"]:
                names.append(f"{c}={lv}")
                groups[names[-1]] = c
        self.feature_names_, self.group_of_ = names, groups
        return self

    def _numeric_raw(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in self.numeric:
            v = df[c].to_numpy(dtype=np.float64)
            if c in self.medians_:
                v = np.where(np.isnan(v), self.medians_[c], v)
            if np.isnan(v).any():
                raise ValueError(f"unexpected missing values in {c}")
            if c not in NO_LOG:
                v = np.log1p(np.clip(v, 0, None))
            cols.append(v)
        return np.stack(cols, axis=1)

    # ── transforming ──────────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Preprocessor is not fitted")
        missing = [c for c in self.numeric + self.categorical if c not in df.columns]
        if missing:
            raise ValueError(f"input lacks required columns: {missing}")
        z = (self._numeric_raw(df) - self.mean_) / self.std_
        parts = [np.clip(z, -CLIP, CLIP)]
        if self.missing_cols_:
            parts.append(np.stack([df[c].isna().to_numpy() for c in self.missing_cols_],
                                  axis=1).astype(np.float64))
        for c in self.categorical:
            vals = df[c].astype(str).to_numpy()
            levels = self.levels_[c]
            idx = {lv: i for i, lv in enumerate(levels)}
            pos = np.array([idx.get(v, len(levels)) for v in vals])
            oh = np.zeros((len(df), len(levels) + 1))
            oh[np.arange(len(df)), pos] = 1.0
            parts.append(oh)
        return np.concatenate(parts, axis=1).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    @property
    def n_features(self) -> int:
        return len(self.feature_names_)

    def save(self, p) -> None:
        joblib.dump(self, p)

    @staticmethod
    def load(p) -> "Preprocessor":
        return joblib.load(p)


def encode_targets(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(multiclass index in schema.CLASS_ORDER, binary label)."""
    idx = {c: i for i, c in enumerate(schema.CLASS_ORDER)}
    y_multi = df["attack_cat"].map(idx)
    if y_multi.isna().any():
        raise ValueError(f"unknown categories: {set(df['attack_cat']) - set(idx)}")
    return y_multi.to_numpy(np.int64), df["label"].to_numpy(np.int64)
