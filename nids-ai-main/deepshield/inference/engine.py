"""Inference for the dashboard: raw flow records in, predictions out.

`TrackAEngine` analyses records in the Track A (training/testing-set)
format with any trained Track A model. `ReplayEngine` replays the Track B
chronological TEST block in time order through the Track B per-flow model
and the LSTM, keeping a per-host history exactly like the LSTM's windows.

Ground-truth isolation (replay): records are split into a `FlowRecord`
(features and network metadata, what a sensor would see) and a
`GroundTruth` (labels). Models only ever receive FlowRecords; ground truth
goes solely to the `EvaluationMonitor`, which scores the predictions for the
demo. `tests/test_inference.py` checks that shuffling the labels leaves the
predictions unchanged.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

import os

from deepshield.config import PROJECT_ROOT, path
from deepshield.data import schema
from deepshield.data.preprocess import Preprocessor
from deepshield.data.schema import CLASS_ORDER
from deepshield.training.trainer import load_checkpoint, predict
from deepshield.xai import explain as E

MAX_UPLOAD_ROWS = 50_000

# Small files for hosting (Render). Built by scripts/make_deploy_data.py from the
# full local data, and committed to Git. The full data (meta.parquet, part-N,
# X_train.npy) is too big for GitHub and for a small server.
DEPLOY_DIR = PROJECT_ROOT / "artifacts" / "deploy"
REPLAY_FILE = DEPLOY_DIR / "replay_test_block.pkl.gz"
BASELINE_FILE = DEPLOY_DIR / "track_a_baseline.npz"


def use_deploy_data() -> bool:
    """True on the server (full data absent) or when forced with DEEPSHIELD_DEPLOY_DATA=1."""
    return os.environ.get("DEEPSHIELD_DEPLOY_DATA") == "1"


class InputError(ValueError):
    """Raised when uploaded records cannot be analysed (message is user-facing)."""


def _models_dir(track: str) -> Path:
    return path("models") / f"track_{track.lower()}"


def validate_records(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    """Coerce and check uploaded records. Returns (valid rows, per-row problems)."""
    if len(df) == 0:
        raise InputError("No records supplied.")
    if len(df) > MAX_UPLOAD_ROWS:
        raise InputError(f"Too many records ({len(df):,}); the limit is {MAX_UPLOAD_ROWS:,}.")
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise InputError(f"Missing required columns: {', '.join(missing[:10])}"
                         + (" ..." if len(missing) > 10 else ""))
    problems = []
    bad = np.zeros(len(df), dtype=bool)
    for c in features:
        if c in schema.CATEGORICAL:
            df[c] = df[c].astype(str).str.strip()
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        invalid = v.isna() | ~np.isfinite(v.fillna(0)) | (v < 0)
        for i in np.flatnonzero(invalid.to_numpy())[:20]:
            problems.append({"row": int(i), "column": c, "value": str(df[c].iloc[i]),
                             "problem": "not a finite non-negative number"})
        bad |= invalid.to_numpy()
        df[c] = v
    return df[~bad].reset_index(drop=True), problems + (
        [{"rows_rejected": int(bad.sum())}] if bad.any() else [])


# ── Track A ───────────────────────────────────────────────────────────────────

MODEL_INFO = {
    "rf_multiclass": ("Random Forest", "multiclass"),
    "mlp": ("MLP", "multiclass"),
    "cnn": ("1D-CNN", "multiclass"),
    "rf_binary": ("Random Forest (binary)", "binary"),
    "autoencoder": ("Autoencoder", "anomaly"),
    "iforest": ("Isolation Forest", "anomaly"),
}


class TrackAEngine:
    def __init__(self):
        d = _models_dir("A")
        self.pre: Preprocessor = Preprocessor.load(path("models") / "preprocessor_track_a.joblib")
        self.models: dict = {}
        self.meta: dict = {}
        for name, (title, task) in MODEL_INFO.items():
            p_pt, p_jl = d / name / "model.pt", d / name / "model.joblib"
            if p_pt.exists():
                self.models[name], self.meta[name] = load_checkpoint(p_pt)
            elif p_jl.exists():
                obj = joblib.load(p_jl)
                if isinstance(obj, dict):          # isolation forest + threshold
                    self.models[name], self.meta[name] = obj["model"], {"threshold": obj["threshold"]}
                else:
                    self.models[name], self.meta[name] = obj, {}
        self.names, self.G = E.grouping(self.pre.feature_names_)
        self._baseline = None
        self._rf_explainer = None
        self._lock = threading.Lock()

    def available(self) -> list[dict]:
        return [{"id": n, "name": MODEL_INFO[n][0], "task": MODEL_INFO[n][1]} for n in self.models]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.pre.transform(df)

    def predict(self, df: pd.DataFrame, model: str) -> list[dict]:
        if model not in self.models:
            raise InputError(f"Model '{model}' is not available.")
        X = self.transform(df)
        m, task = self.models[model], MODEL_INFO[model][1]
        out = []
        if task == "multiclass":
            proba = m.predict_proba(X) if hasattr(m, "predict_proba") else predict(m, X)
            for p in proba:
                k = int(p.argmax())
                out.append({"category": CLASS_ORDER[k], "confidence": float(p[k]),
                            "is_attack": k != 0, "probabilities": dict(zip(CLASS_ORDER, map(float, p)))})
        elif task == "binary":
            s = m.predict_proba(X)[:, 1]
            out = [{"category": "Attack" if v >= 0.5 else "Normal", "confidence": float(max(v, 1 - v)),
                    "is_attack": bool(v >= 0.5), "attack_probability": float(v)} for v in s]
        else:
            if model == "autoencoder":
                s = predict(m, X, fn=lambda mm, xb: mm.score(xb))
            else:
                s = -m.score_samples(X)
            thr = self.meta[model]["threshold"]
            out = [{"category": "Anomalous" if v >= thr else "Normal", "is_attack": bool(v >= thr),
                    "anomaly_score": float(v), "threshold": float(thr),
                    "confidence": None} for v in s]
        return out

    def _base(self):
        if self._baseline is None:
            full = path("processed") / "track_a" / "X_train.npy"
            if full.exists() and not use_deploy_data():
                X = np.load(full, mmap_mode="r")
                rng = np.random.default_rng(0)
                self._background = np.asarray(X[rng.choice(len(X), 50, replace=False)])
                self._baseline = np.asarray(X[::10]).mean(axis=0)
            elif BASELINE_FILE.exists():
                # Same 50 background rows and baseline, precomputed from X_train.npy
                z = np.load(BASELINE_FILE)
                self._background, self._baseline = z["background"], z["baseline"]
            else:
                raise InputError("Explanation baseline missing: run scripts/make_deploy_data.py "
                                 "and commit artifacts/deploy/.")
        return self._baseline

    def explain(self, row: pd.DataFrame, model: str, k: int = 8) -> dict:
        """Local explanation of ONE record for the given model."""
        X = self.transform(row)
        raw = {c: (row[c].iloc[0].item() if hasattr(row[c].iloc[0], "item") else row[c].iloc[0])
               for c in schema.TRACK_A_FEATURES if c in row}
        pred = self.predict(row, model)[0]
        with self._lock:
            if model == "rf_multiclass":
                self._base()
                if self._rf_explainer is None:
                    self._rf_explainer = E.RFExplainer(self.models[model], self._background)
                c = CLASS_ORDER.index(pred["category"])
                sv = E.aggregate(self._rf_explainer.shap_values(X), self.G)[0, :, c]
                method, unit = "SHAP (TreeExplainer, interventional)", "probability"
                base_value = float(self._rf_explainer.expected[c])
            elif model in ("mlp", "cnn"):
                c = CLASS_ORDER.index(pred["category"])
                a, _ = E.integrated_gradients(self.models[model], X, np.array([c]), self._base(), steps=64)
                sv = E.aggregate(a, self.G)[0]
                method, unit, base_value = "Integrated Gradients (Captum)", "logit", None
            elif model == "autoencoder":
                contrib = E.aggregate(E.autoencoder_contributions(self.models[model], X), self.G)[0]
                sv = contrib / max(contrib.sum(), 1e-12)
                method, unit, base_value = "Share of reconstruction error", "share", None
            else:
                raise InputError("Local explanations are available for Random Forest, MLP, "
                                 "1D-CNN and Autoencoder.")
        return {"model": model, "method": method, "unit": unit, "base_value": base_value,
                "prediction": pred, "contributions": E.top_contributions(sv, self.names, raw, k),
                "note": "Attributions show what the model relied on, not what caused the traffic."}


# ── Track B replay ────────────────────────────────────────────────────────────

@dataclass
class FlowRecord:
    """What a sensor would observe: features and network metadata, no labels."""
    seq: int
    features: dict
    srcip: str
    dstip: str
    sport: int
    dsport: int
    proto: str
    service: str
    stime: int
    ltime: int


@dataclass
class GroundTruth:
    seq: int
    attack_cat: str
    label: int


class EvaluationMonitor:
    """Running demo accuracy; the ONLY component that sees ground truth."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.correct = 0
        self.tp = self.fp = self.fn = self.tn = 0
        self.cm = np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=np.int64)

    def update(self, truths: list[GroundTruth], preds: list[dict]) -> None:
        for t, p in zip(truths, preds):
            assert t.seq == p["seq"]
            i, j = CLASS_ORDER.index(t.attack_cat), CLASS_ORDER.index(p["category"])
            self.cm[i, j] += 1
            self.n += 1
            self.correct += i == j
            a, b = t.label == 1, p["is_attack"]
            self.tp += a and b
            self.fp += (not a) and b
            self.fn += a and not b
            self.tn += (not a) and not b

    def summary(self) -> dict:
        n = max(self.n, 1)
        return {"evaluated": self.n, "accuracy": self.correct / n,
                "detection_rate": self.tp / max(self.tp + self.fn, 1),
                "false_positive_rate": self.fp / max(self.fp + self.tn, 1),
                "note": "Available only because this is a labelled dataset replay."}


class ReplayEngine:
    """Replays the Track B chronological TEST block in end-time order."""

    REPLAY_MODELS = {"lstm": "LSTM (per-host sequence)", "mlp_flow": "Per-flow MLP",
                     "rf_flow": "Per-flow Random Forest"}

    def __init__(self, max_records: int | None = None):
        d = _models_dir("B")
        self.pre: Preprocessor = Preprocessor.load(path("models") / "preprocessor_track_b.joblib")
        self.models, self.meta = {}, {}
        for name in self.REPLAY_MODELS:
            if (d / name / "model.pt").exists():
                self.models[name], self.meta[name] = load_checkpoint(d / name / "model.pt")
            elif (d / name / "model.joblib").exists():
                self.models[name], self.meta[name] = joblib.load(d / name / "model.joblib"), {}
        self.seq_len = int(self.meta.get("lstm", {}).get("seq_len", 1))
        self.records, self.truth = self._load_test_block(max_records)
        self.history: dict[str, deque] = {}
        self.position = 0

    def available(self) -> list[dict]:
        return [{"id": n, "name": self.REPLAY_MODELS[n]} for n in self.models]

    @staticmethod
    def _load_test_block(max_records):
        meta_path = path("processed") / "track_b" / "meta.parquet"
        if (use_deploy_data() or not meta_path.exists()) and REPLAY_FILE.exists():
            # Hosted mode: prebuilt replay block (same rows, same order, same columns)
            df = pd.read_pickle(REPLAY_FILE)
            if max_records:
                df = df.iloc[:max_records]
            df = df.reset_index(drop=True)
            truth = df[["attack_cat", "label"]]
            records = df.drop(columns=["attack_cat", "label"])
            return records, truth
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found and no {REPLAY_FILE.name}: run "
                "scripts/make_deploy_data.py locally and commit artifacts/deploy/.")
        meta = pd.read_parquet(path("processed") / "track_b" / "meta.parquet")
        test = meta[meta["block"] == "test"].sort_values(["ltime", "source_file", "source_row"],
                                                          kind="stable")
        if max_records:
            test = test.iloc[:max_records]
        feats = schema.TRACK_B_FEATURES
        parts = []
        for f, g in test.groupby("source_file", sort=False):
            part = pd.read_parquet(path("interim") / "track_b" / f"part-{f}.parquet",
                                   columns=["source_row"] + [c for c in feats if c not in
                                                            ("proto", "service", "state", "sbytes", "dbytes", "dur")])
            parts.append(g.merge(part, on="source_row", how="left"))
        df = pd.concat(parts).sort_values(["ltime", "source_file", "source_row"], kind="stable")
        df = df.reset_index(drop=True)
        cols = ["srcip", "dstip", "sport", "dsport", "proto", "service", "stime", "ltime"] + feats
        records = df[list(dict.fromkeys(cols))]      # proto/service are both metadata and features
        truth = df[["attack_cat", "label"]]
        return records, truth

    def reset(self):
        self.history.clear()
        self.position = 0

    def next_batch(self, n: int) -> tuple[list[FlowRecord], list[GroundTruth]]:
        s, e = self.position, min(self.position + n, len(self.records))
        self.position = e
        flows, truths = [], []
        for i in range(s, e):
            r = self.records.iloc[i]
            flows.append(FlowRecord(i, {c: r[c] for c in schema.TRACK_B_FEATURES}, r["srcip"], r["dstip"],
                                    int(r["sport"]), int(r["dsport"]), r["proto"], r["service"],
                                    int(r["stime"]), int(r["ltime"])))
            t = self.truth.iloc[i]
            truths.append(GroundTruth(i, t["attack_cat"], int(t["label"])))
        return flows, truths

    @property
    def finished(self) -> bool:
        return self.position >= len(self.records)

    def predict(self, flows: list[FlowRecord], model: str) -> list[dict]:
        """Predict a batch of FlowRecords (labels are never passed in)."""
        if not flows:
            return []
        X = self.pre.transform(pd.DataFrame([f.features for f in flows]))
        if model == "lstm":
            L = self.seq_len
            wins = np.zeros((len(flows), L, X.shape[1]), dtype=np.float32)
            lengths = np.zeros(len(flows), dtype=np.int64)
            for i, (f, x) in enumerate(zip(flows, X)):
                h = self.history.setdefault(f.srcip, deque(maxlen=L - 1))
                seq = list(h) + [x]
                wins[i, L - len(seq):] = np.stack(seq)
                lengths[i] = len(seq)
                h.append(x)
            with torch.no_grad():
                proba = torch.softmax(self.models["lstm"](torch.from_numpy(wins),
                                                          torch.from_numpy(lengths)), 1).numpy()
        elif model == "mlp_flow":
            proba = predict(self.models[model], X)
        elif model == "rf_flow":
            proba = self.models[model].predict_proba(X)
        else:
            raise InputError(f"Unknown replay model {model}")
        out = []
        for f, p in zip(flows, proba):
            k = int(p.argmax())
            out.append({"seq": f.seq, "category": CLASS_ORDER[k], "confidence": float(p[k]),
                        "is_attack": k != 0, "probabilities": dict(zip(CLASS_ORDER, map(float, p)))})
        return out


def metrics_catalog() -> dict:
    """All saved evaluation results (actual experiment outputs)."""
    out = {}
    for track in ("a", "b"):
        d = path("metrics") / f"track_{track}"
        if d.exists():
            out[track] = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))}
    return out
