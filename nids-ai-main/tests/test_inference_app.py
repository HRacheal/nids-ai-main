"""Phase 7-8 tests: inference, ground-truth isolation, history, dashboard API.

Need the trained artifacts (artifacts/models) and processed data; skipped
automatically when they are absent (e.g. fresh clone without the dataset).
"""

import io

import numpy as np
import pandas as pd
import pytest

from deepshield.config import path
from deepshield.data import schema
from deepshield.inference.engine import InputError, validate_records

HAVE_A = (path("models") / "track_a" / "rf_multiclass" / "model.joblib").exists() and \
         (path("interim") / "track_a_test.parquet").exists()
HAVE_B = (path("models") / "track_b" / "lstm" / "model.pt").exists() and \
         (path("processed") / "track_b" / "meta.parquet").exists()
needs_a = pytest.mark.skipif(not HAVE_A, reason="Track A artifacts not present")
needs_b = pytest.mark.skipif(not HAVE_B, reason="Track B artifacts not present")


def _records(n=6):
    df = pd.read_parquet(path("interim") / "track_a_test.parquet")
    return df.groupby("attack_cat").head(1).head(n).reset_index(drop=True)


def test_validate_records_reports_problems():
    df = pd.DataFrame({c: [1.0, 2.0] for c in schema.TRACK_A_FEATURES})
    df["proto"], df["service"], df["state"] = "tcp", "-", "FIN"
    df.loc[1, "sbytes"] = -5
    valid, problems = validate_records(df, schema.TRACK_A_FEATURES)
    assert len(valid) == 1 and problems[-1] == {"rows_rejected": 1}
    with pytest.raises(InputError, match="Missing required columns"):
        validate_records(df.drop(columns=["dur"]), schema.TRACK_A_FEATURES)
    with pytest.raises(InputError, match="No records"):
        validate_records(df.iloc[:0], schema.TRACK_A_FEATURES)


@needs_a
def test_track_a_engine_predicts_with_every_model_and_explains():
    from deepshield.inference.engine import TrackAEngine
    eng = TrackAEngine()
    df = _records()
    assert eng.transform(df).shape[1] == len(eng.pre.feature_names_)
    for m in eng.models:
        out = eng.predict(df, m)
        assert len(out) == len(df) and all("is_attack" in o for o in out)
    ex = eng.explain(df.iloc[[1]], "rf_multiclass")
    p = ex["prediction"]["probabilities"][ex["prediction"]["category"]]
    # SHAP additivity: base value + all contributions = predicted probability
    import deepshield.xai.explain as E
    X = eng.transform(df.iloc[[1]])
    c = schema.CLASS_ORDER.index(ex["prediction"]["category"])
    full = E.aggregate(eng._rf_explainer.shap_values(X), eng.G)[0, :, c]
    assert abs(ex["base_value"] + full.sum() - p) < 1e-5
    ig = eng.explain(df.iloc[[1]], "mlp")
    assert ig["method"].startswith("Integrated Gradients") and len(ig["contributions"]) == 8


@needs_a
def test_models_receive_identical_features_regardless_of_extra_columns():
    from deepshield.inference.engine import TrackAEngine
    eng = TrackAEngine()
    df = _records()
    noisy = df.assign(attack_cat="Normal", label=0, extra="x")   # labels must not be used
    a = eng.predict(df[schema.TRACK_A_FEATURES], "mlp")
    b = eng.predict(noisy, "mlp")
    assert [x["probabilities"] for x in a] == [x["probabilities"] for x in b]


@needs_b
def test_replay_predictions_do_not_depend_on_ground_truth():
    """Label-shuffle test: shuffling the ground truth must not change any prediction."""
    from deepshield.inference.engine import ReplayEngine
    eng = ReplayEngine(max_records=600)
    flows, truths = eng.next_batch(600)
    p1 = eng.predict(flows, "lstm")
    eng.reset()
    eng.truth = eng.truth.sample(frac=1, random_state=0).reset_index(drop=True)
    flows2, truths2 = eng.next_batch(600)
    p2 = eng.predict(flows2, "lstm")
    assert [t.attack_cat for t in truths] != [t.attack_cat for t in truths2] or len(set(t.attack_cat for t in truths)) == 1
    assert [p["probabilities"] for p in p1] == [p["probabilities"] for p in p2]


@needs_b
def test_replay_lstm_history_matches_offline_windows():
    """Streaming per-host history gives the same LSTM output as the offline window index."""
    import torch
    from deepshield.data.sequences import WindowIndex
    from deepshield.inference.engine import ReplayEngine
    eng = ReplayEngine(max_records=400)
    flows, _ = eng.next_batch(400)
    online = np.array([p["probabilities"]["Normal"] for p in eng.predict(flows, "lstm")])
    X = eng.pre.transform(pd.DataFrame([f.features for f in flows]))
    wi = WindowIndex.build(np.array([f.srcip for f in flows]), np.array([f.ltime for f in flows]))
    w, lengths = wi.gather(X, np.arange(len(flows)), eng.seq_len)
    with torch.no_grad():
        off = torch.softmax(eng.models["lstm"](torch.from_numpy(w), torch.from_numpy(lengths)), 1)[:, 0].numpy()
    assert np.allclose(online, off, atol=1e-5)


@pytest.fixture
def client(tmp_path):
    from deepshield.app.server import create_app
    app, sio = create_app(db_path=tmp_path / "t.db", replay_max_records=300)
    app.testing = True
    return app, sio, app.test_client()


@needs_a
def test_dashboard_api_end_to_end(client):
    app, sio, c = client
    assert c.get("/").status_code == 200
    st = c.get("/api/status").get_json()
    assert {m["id"] for m in st["track_a_models"]} >= {"rf_multiclass", "mlp", "cnn", "autoencoder"}
    s = c.get("/api/sample_records?n=30&seed=1").get_json()
    r = c.post("/api/analyze", json={"model": "rf_multiclass", "records": s["records"]}).get_json()
    assert r["n"] == 30 and sum(r["summary"].values()) == 30
    ov = c.get("/api/overview").get_json()
    assert ov["total"] == 30 and ov["malicious"] == sum(p["is_attack"] for p in r["predictions"])
    al = c.get("/api/alerts?limit=5").get_json()
    assert al["total"] == r["alerts_created"]
    if al["rows"]:
        aid = al["rows"][0]["id"]
        d = c.get(f"/api/alerts/{aid}").get_json()
        assert d["record"] and "techniques" in d["mitre"]
        ex = c.get(f"/api/alerts/{aid}/explain").get_json()
        assert ex["contributions"]
        assert c.post(f"/api/alerts/{aid}/ack").get_json()["ok"]
    assert c.get("/api/predictions?search=tcp").status_code == 200
    m = c.get("/api/models").get_json()
    assert "rf_multiclass" in m["models"]["a"] and m["models"]["a"]["rf_multiclass"]["macro_f1"] > 0
    pdf = c.get("/api/report.pdf")
    assert pdf.status_code == 200 and pdf.data[:5] == b"%PDF-"
    csv = c.get("/api/report.csv").data.decode()
    assert csv.count("\n") == 31            # header + 30 predictions
    assert c.get("/api/mitre").get_json()["all_valid"]


@needs_a
def test_dashboard_rejects_bad_input(client):
    _, _, c = client
    r = c.post("/api/analyze", data={"file": (io.BytesIO(b"a,b\n1,2\n"), "x.csv"), "model": "mlp"},
               content_type="multipart/form-data")
    assert r.status_code == 400 and "Missing required columns" in r.get_json()["error"]
    s = c.get("/api/sample_records?n=2&seed=1").get_json()
    rec = s["records"]
    rec[0]["sbytes"] = "abc"
    r = c.post("/api/analyze", json={"model": "rf_multiclass", "records": rec}).get_json()
    assert r["n"] == 1 and r["problems"]
    assert c.post("/api/analyze", json={"model": "nope", "records": s["records"]}).status_code == 400
    assert c.get("/api/alerts/doesnotexist").status_code == 404


@needs_b
def test_replay_over_socketio(client):
    import time
    app, sio, _ = client
    sc = sio.test_client(app)
    ack = sc.emit("replay_start", {"model": "lstm", "speed": 1600, "restart": True}, callback=True)
    assert ack["ok"]
    S = app.extensions["deepshield_state"]
    for _ in range(100):
        if not S.replay_status["running"]:
            break
        time.sleep(0.2)
    assert S.replay_status["processed"] == 300
    events = [e["name"] for e in sc.get_received()]
    assert "replay_batch" in events and "replay_status" in events
    assert S.history.overview()["total"] == 300
    assert S.monitor.summary()["evaluated"] == 300


@needs_b
def test_replay_stop_with_page_payload(client):
    """The Stop button emits replay_stop WITH a payload ({}); it must stop the replay (regression)."""
    import time
    app, sio, _ = client
    sc = sio.test_client(app)
    S = app.extensions["deepshield_state"]
    assert sc.emit("replay_start", {"model": "mlp_flow", "speed": 20, "restart": True}, callback=True)["ok"]
    time.sleep(0.5)
    ack = sc.emit("replay_stop", {}, callback=True)
    assert ack == {"ok": True}
    for _ in range(50):
        if not S.replay_status["running"]:
            break
        time.sleep(0.1)
    assert not S.replay_status["running"]
    assert S.replay_status["processed"] < 300          # stopped before the end of the (300-record) replay
