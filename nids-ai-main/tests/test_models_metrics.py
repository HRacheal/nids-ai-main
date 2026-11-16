"""Phase 3-5 tests: network shapes, checkpoints, LSTM masking, metrics, statistics, MITRE."""

import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score

from deepshield.evaluation import metrics as M
from deepshield.models.networks import CNN1D, MLP, Autoencoder, FlowLSTM
from deepshield.training.trainer import TrainConfig, array_batches, fit, load_checkpoint, save_checkpoint


def test_network_output_shapes():
    x = torch.randn(5, 50)
    assert MLP(50, 10, [32, 16])(x).shape == (5, 10)
    assert CNN1D(20, 30, 10, [8, 16], 3)(x).shape == (5, 10)
    perm = list(np.random.default_rng(0).permutation(20))
    assert CNN1D(20, 30, 10, [8, 16], 3, feature_perm=perm)(x).shape == (5, 10)
    ae = Autoencoder(50, [16], 4)
    assert ae(x).shape == x.shape and ae.score(x).shape == (5,)


def test_cnn_random_order_permutes_only_numeric_block():
    torch.manual_seed(0)
    a = CNN1D(4, 3, 2, [4, 4], 3).eval()
    b = CNN1D(4, 3, 2, [4, 4], 3, feature_perm=[3, 2, 1, 0]).eval()
    b.load_state_dict(a.state_dict(), strict=False)
    x = torch.randn(2, 7)
    x_rev = torch.cat([x[:, :4].flip(1), x[:, 4:]], 1)
    assert torch.allclose(a(x_rev), b(x), atol=1e-6)     # b reverses internally


def test_lstm_ignores_padding():
    torch.manual_seed(0)
    m = FlowLSTM(6, 3, proj=8, hidden=8).eval()
    seq = torch.randn(1, 3, 6)
    short = torch.cat([torch.zeros(1, 2, 6), seq], 1)       # left-padded to length 5
    long_noise = torch.cat([torch.randn(1, 2, 6) * 50, seq], 1)
    a = m(short, torch.tensor([3]))
    b = m(long_noise, torch.tensor([3]))                     # padded positions must not matter
    c = m(seq, torch.tensor([3]))
    assert torch.allclose(a, b, atol=1e-6) and torch.allclose(a, c, atol=1e-6)


def test_fit_early_stopping_and_checkpoint_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 8)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    cfg = TrainConfig(max_epochs=30, patience=2, batch_size=64, seed=0)
    params = {"n_in": 8, "n_out": 2, "hidden": [16], "dropout": 0.0}
    res = fit("MLP", params, cfg, array_batches(X, y, 64, True), array_batches(X, y, 400, False),
              torch.nn.CrossEntropyLoss(), log=lambda *_: None)
    h = res.history
    assert h["best_val_loss"] == min(h["val_loss"]) and 1 <= h["best_epoch"] <= h["epochs_run"]
    save_checkpoint(tmp_path / "m.pt", res.model, "MLP", params, cfg)
    m2, meta = load_checkpoint(tmp_path / "m.pt")
    xt = torch.from_numpy(X[:10])
    assert torch.equal(res.model(xt), m2(xt)) and meta["n_parameters"] > 0


def test_multiclass_report_matches_sklearn_and_marks_absent_classes():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 9, 500)                              # class 9 (Worms) absent
    p = np.where(rng.random(500) < 0.7, y, rng.integers(0, 10, 500))
    rep = M.multiclass_report(y, p)
    assert np.isclose(rep["macro_f1"], f1_score(y, p, labels=np.arange(9), average="macro"))
    assert rep["per_class"]["Worms"]["support"] == 0 and "unavailable" in rep["per_class"]["Worms"]["status"]
    assert np.isclose(rep["binary_view"]["accuracy"], ((y != 0) == (p != 0)).mean())


def test_bootstrap_and_paired_difference():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 3000)
    good = np.where(rng.random(3000) < 0.9, y, (y + 1) % 3)
    bad = np.where(rng.random(3000) < 0.6, y, (y + 1) % 3)
    ci = M.bootstrap_ci(y, good, n_boot=200, k=3)
    obs = M.multiclass_report(y, good, classes=["a", "b", "c"])["macro_f1"]
    assert ci["low"] < obs < ci["high"]
    d = M.paired_bootstrap_diff(y, good, bad, n_boot=200, k=3)
    assert d["diff_macro_f1"] > 0 and d["excludes_zero"]


def test_mcnemar_and_holm():
    y = np.zeros(100, dtype=int)
    a = np.zeros(100, dtype=int)
    b = np.r_[np.ones(30, dtype=int), np.zeros(70, dtype=int)]
    r = M.mcnemar(y, a, b)
    assert r["a_right_b_wrong"] == 30 and r["p_value"] < 1e-6
    assert M.mcnemar(y, a, a)["p_value"] == 1.0
    h = M.holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert h["x"]["p_holm"] == pytest.approx(0.03) and h["z"]["p_holm"] == pytest.approx(0.06)
    assert h["y"]["p_holm"] == pytest.approx(0.06) and not h["y"]["significant"]


def test_mitre_candidates_validated_against_official_stix():
    from deepshield.intel.mitre import CANDIDATES, enrich, load_validated
    v = load_validated()
    assert v["attack_version"] and v["all_valid"]
    assert set(v["categories"]) == set(CANDIDATES)
    e = enrich("DoS")
    assert {t["id"] for t in e["techniques"]} == {"T1498", "T1499"} and "not confirmed" in e["disclaimer"]
    assert enrich("Generic")["techniques"] == []
