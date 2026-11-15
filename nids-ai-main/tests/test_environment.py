"""Phase 0 dependency compatibility checks for DeepShield AI.

Each test exercises the specific library feature DeepShield will rely on in
later phases, not merely that the package imports.
"""

import sys

import numpy as np
import pytest

pytestmark = pytest.mark.environment

N_FEATURES = 8
N_CLASSES = 3


def _toy_data(n: int = 300, seed: int = 0):
    """Small, linearly separable multiclass data set."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, N_CLASSES, n)
    centres = rng.normal(0, 3, (N_CLASSES, N_FEATURES))
    X = centres[y] + rng.normal(0, 1, (n, N_FEATURES))
    return X.astype(np.float32), y.astype(np.int64)


def _torch_mlp(seed: int = 0):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(N_FEATURES, 16), nn.ReLU(), nn.Linear(16, N_CLASSES)
    )


def _train(model, X, y, steps: int = 150):
    import torch

    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


# ── Python / imports ──────────────────────────────────────────────────────────

def test_python_version():
    assert sys.version_info[:2] == (3, 12)


def test_running_inside_project_venv():
    assert sys.prefix != sys.base_prefix, "tests must run inside .venv"


def test_torch_imports_after_sklearn_and_shap():
    """Regression check for a Windows DLL conflict found in Phase 0.

    scikit-learn 1.5.0 bundled msvcp140.dll 14.32; once loaded, torch 2.13's
    c10.dll failed to initialise (WinError 1114). Runs in a fresh process
    so the import order of this test session cannot mask the problem.
    """
    import subprocess

    code = "import sklearn, shap, torch; torch.zeros(1)"
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-800:]


@pytest.mark.parametrize("module", [
    "numpy", "pandas", "sklearn", "scipy", "torch", "captum", "shap",
    "matplotlib", "pyarrow", "joblib", "flask", "flask_socketio",
    "reportlab", "rich", "yaml",
])
def test_import(module):
    __import__(module)


# ── PyTorch ───────────────────────────────────────────────────────────────────

def test_torch_trains_saves_loads(tmp_path):
    import torch

    torch.set_num_threads(2)
    X, y = _toy_data()
    model = _torch_mlp()
    losses = _train(model, X, y)
    assert losses[-1] < 0.5 * losses[0]

    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    clone = _torch_mlp(seed=123)
    clone.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        xt = torch.from_numpy(X[:20])
        assert torch.equal(model(xt), clone(xt))


def test_torch_seeded_training_is_reproducible():
    import torch

    X, y = _toy_data()
    a, b = _torch_mlp(seed=7), _torch_mlp(seed=7)
    torch.manual_seed(0)
    _train(a, X, y, steps=20)
    torch.manual_seed(0)
    _train(b, X, y, steps=20)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_torch_conv1d_and_lstm_forward_backward():
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence

    conv = nn.Sequential(
        nn.Conv1d(1, 8, kernel_size=3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(8, N_CLASSES),
    )
    x = torch.randn(4, 1, 40, requires_grad=True)
    out = conv(x)
    out.sum().backward()
    assert out.shape == (4, N_CLASSES) and x.grad is not None

    lstm = nn.LSTM(input_size=12, hidden_size=16, num_layers=2, batch_first=True)
    seq = torch.randn(3, 10, 12)
    lengths = torch.tensor([10, 6, 1])          # padded / masked windows
    packed = pack_padded_sequence(seq, lengths, batch_first=True,
                                  enforce_sorted=False)
    _, (h, _) = lstm(packed)
    assert h.shape == (2, 3, 16)
    h[-1].sum().backward()


# ── Captum ────────────────────────────────────────────────────────────────────

def test_captum_integrated_gradients_completeness():
    import torch
    from captum.attr import IntegratedGradients

    X, y = _toy_data()
    model = _torch_mlp()
    _train(model, X, y)
    model = model.double().eval()

    x = torch.from_numpy(X[:5]).double()
    baseline = torch.from_numpy(X.mean(axis=0, keepdims=True)).double()
    baseline = baseline.expand_as(x)
    target = model(x).argmax(dim=1)

    ig = IntegratedGradients(model)
    attr, delta = ig.attribute(x, baselines=baseline, target=target,
                               n_steps=512, return_convergence_delta=True)
    assert attr.shape == x.shape
    # Completeness: attributions sum to f(x) - f(baseline) for the target, up
    # to numerical-integration error (non-zero for ReLU networks because of
    # their kinks, so a relative tolerance is the meaningful criterion).
    with torch.no_grad():
        fx = model(x).gather(1, target[:, None]).squeeze(1)
        fb = model(baseline).gather(1, target[:, None]).squeeze(1)
    gap = attr.sum(dim=1) - (fx - fb)
    assert (gap.abs() / (fx - fb).abs()).max().item() < 0.01
    # Captum's reported convergence delta is exactly this completeness gap.
    assert torch.allclose(delta, gap, atol=1e-9)


# ── SHAP ──────────────────────────────────────────────────────────────────────

def test_shap_tree_explainer_multiclass_shape_and_additivity():
    import shap
    from sklearn.ensemble import RandomForestClassifier

    X, y = _toy_data()
    rf = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    explainer = shap.TreeExplainer(rf, data=X[:50],
                                   feature_perturbation="interventional")
    sv = explainer.shap_values(X[:5])
    # SHAP 0.50 returns (n_samples, n_features, n_classes). Older versions
    # returned a list of per-class arrays; DeepShield must normalise both.
    arr = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)
    assert arr.shape == (5, N_FEATURES, N_CLASSES)

    reconstructed = arr.sum(axis=1) + np.asarray(explainer.expected_value)
    np.testing.assert_allclose(reconstructed, rf.predict_proba(X[:5]),
                               atol=1e-6)


def test_shap_gradient_explainer_on_torch_model():
    import shap
    import torch

    X, y = _toy_data()
    model = _torch_mlp()
    _train(model, X, y)
    model.eval()
    explainer = shap.GradientExplainer(model, torch.from_numpy(X[:50]))
    sv = explainer.shap_values(torch.from_numpy(X[:3]))
    arr = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)
    assert arr.shape == (3, N_FEATURES, N_CLASSES)
    assert np.isfinite(arr).all()


# ── scikit-learn / joblib ─────────────────────────────────────────────────────

def test_sklearn_baselines_and_joblib_roundtrip(tmp_path):
    import joblib
    from sklearn.ensemble import IsolationForest, RandomForestClassifier

    X, y = _toy_data()
    rf = RandomForestClassifier(n_estimators=20, class_weight="balanced",
                                random_state=0).fit(X, y)
    iso = IsolationForest(n_estimators=50, random_state=0).fit(X[y == 0])
    joblib.dump((rf, iso), tmp_path / "baselines.joblib")
    rf2, iso2 = joblib.load(tmp_path / "baselines.joblib")
    np.testing.assert_array_equal(rf.predict_proba(X), rf2.predict_proba(X))
    np.testing.assert_array_equal(iso.score_samples(X), iso2.score_samples(X))


# ── Dashboard stack ───────────────────────────────────────────────────────────

def test_flask_socketio_roundtrip():
    from flask import Flask, jsonify
    from flask_socketio import SocketIO, emit

    app = Flask(__name__)
    sio = SocketIO(app, async_mode="threading")

    @app.route("/api/ping")
    def ping():
        return jsonify({"ok": True})

    @sio.on("ping_test")
    def on_ping(data):
        emit("pong_test", {"echo": data["x"]})

    assert app.test_client().get("/api/ping").get_json() == {"ok": True}

    client = sio.test_client(app)
    assert client.is_connected()
    client.emit("ping_test", {"x": 42})
    received = client.get_received()
    assert {"name": "pong_test", "args": [{"echo": 42}], "namespace": "/"} in received
    client.disconnect()


# ── Reporting / data / plotting ───────────────────────────────────────────────

def test_reportlab_builds_pdf(tmp_path):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Table

    path = tmp_path / "report.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4)
    doc.build([Paragraph("DeepShield AI", getSampleStyleSheet()["Title"]),
               Table([["Model", "Macro-F1"], ["CNN", "0.00"]])])
    assert path.read_bytes()[:5] == b"%PDF-"


def test_parquet_roundtrip_preserves_dtypes(tmp_path):
    import pandas as pd

    df = pd.DataFrame({
        "sbytes": np.arange(5, dtype=np.float32),
        "proto": pd.Categorical(["tcp", "udp", "tcp", "arp", "udp"]),
        "attack_cat": ["Normal", "DoS", "Normal", "Worms", "Exploits"],
    })
    path = tmp_path / "flows.parquet"
    df.to_parquet(path)
    pd.testing.assert_frame_equal(df, pd.read_parquet(path))


def test_matplotlib_headless_png(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.bar(["Normal", "Attack"], [56000, 119341])
    path = tmp_path / "plot.png"
    fig.savefig(path)
    plt.close(fig)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_scapy_builds_packet_without_capture():
    # Live capture is optional; only packet construction/parsing is checked.
    from scapy.layers.inet import IP, TCP

    raw = bytes(IP(src="10.0.0.1", dst="10.0.0.2") / TCP(dport=80, flags="S"))
    parsed = IP(raw)
    assert parsed[TCP].dport == 80 and parsed[TCP].flags == "S"
