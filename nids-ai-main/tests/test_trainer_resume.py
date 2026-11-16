"""Per-epoch resume: an interrupted run must end exactly like an uninterrupted one."""

import numpy as np
import torch

from deepshield.training.trainer import TrainConfig, array_batches, fit


def _data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 12)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    return X, y


def _fit(cfg, resume=None):
    X, y = _data()
    return fit("MLP", {"n_in": 12, "n_out": 2, "hidden": [16], "dropout": 0.3}, cfg,
               array_batches(X, y, 64, True), array_batches(X, y, 256, False),
               torch.nn.CrossEntropyLoss(), log=lambda *_: None, resume_path=resume)


def test_resume_matches_uninterrupted(tmp_path):
    full = _fit(TrainConfig(max_epochs=4, patience=10))
    ck = tmp_path / "run.pt"
    _fit(TrainConfig(max_epochs=2, patience=10), ck)          # "interrupted" after epoch 2
    resumed = _fit(TrainConfig(max_epochs=4, patience=10), ck)
    assert resumed.history["epochs_run"] == 4
    assert np.allclose(full.history["val_loss"], resumed.history["val_loss"])
    for a, b in zip(full.model.state_dict().values(), resumed.model.state_dict().values()):
        assert torch.equal(a, b)
