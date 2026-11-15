"""Where experiment inputs and outputs live.

* model files            artifacts/models/<track>/<name>/           (git-ignored)
* metrics + configs      artifacts/metrics/<track>/<name>.json      (committed)
* predictions            artifacts/predictions/<track>/<name>_<split>.npz (git-ignored)
* figures                reports/figures/models/                    (committed)
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

from deepshield.config import PROJECT_ROOT, ensure_dir, path

FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "models"


def model_dir(track: str, name: str) -> Path:
    return ensure_dir(path("models") / f"track_{track.lower()}" / name)


def metrics_path(track: str, name: str) -> Path:
    return ensure_dir(path("metrics") / f"track_{track.lower()}") / f"{name}.json"


def predictions_path(track: str, name: str, split: str) -> Path:
    return ensure_dir(PROJECT_ROOT / "artifacts" / "predictions" / f"track_{track.lower()}") \
        / f"{name}_{split}.npz"


def save_metrics(track: str, name: str, record: dict) -> Path:
    record = dict(record)
    record.setdefault("saved_utc", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    record.setdefault("machine", {"cpu_count_logical": __import__("os").cpu_count(),
                                  "platform": platform.platform()})
    p = metrics_path(track, name)
    p.write_text(json.dumps(record, indent=2, default=_json_default), encoding="utf-8")
    return p


def load_metrics(track: str, name: str) -> dict:
    return json.loads(metrics_path(track, name).read_text(encoding="utf-8"))


def save_predictions(track: str, name: str, split: str, **arrays) -> Path:
    p = predictions_path(track, name, split)
    np.savez_compressed(p, **arrays)
    return p


def load_predictions(track: str, name: str, split: str) -> dict:
    with np.load(predictions_path(track, name, split)) as z:
        return {k: z[k] for k in z.files}


def load_track_a_arrays() -> dict:
    d = path("processed") / "track_a"
    out = {}
    for s in ("train", "val", "test"):
        out[f"X_{s}"] = np.load(d / f"X_{s}.npy")
        out[f"y_{s}"] = np.load(d / f"y_{s}.npy")
        out[f"yb_{s}"] = np.load(d / f"ybin_{s}.npy")
    out["test_overlap"] = np.load(d / "test_overlaps_train.npy")
    out["feature_names"] = json.loads((d / "feature_names.json").read_text(encoding="utf-8"))
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
