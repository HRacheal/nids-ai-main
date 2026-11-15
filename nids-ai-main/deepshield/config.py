"""Project configuration and paths.

All paths in ``config/default.yaml`` are relative to the project root, so the
project runs from any checkout location without machine-specific paths.
"""

from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | os.PathLike | None = None) -> dict:
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """Absolute path for an entry of the ``paths`` section."""
    return PROJECT_ROOT / load_config()["paths"][key]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if imported) PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # torch is optional for the data-only scripts
        pass


def torch_threads() -> int:
    return int(load_config()["runtime"].get("torch_num_threads", 4))
