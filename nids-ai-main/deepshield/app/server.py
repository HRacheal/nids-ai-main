
r"""
DeepShield AI dashboard: Flask + Socket.IO (same stack as the original NIDS-AI).

Pages (single-page app, templates/index.html):
  Overview · Traffic monitor (dataset replay) · Threat detection (analyse
  records) · Alerts & history · Model comparison · Explainability · Reports

Everything shown comes from the trained models, the saved experiment
metrics (artifacts/metrics) and the prediction history database; nothing
is hard-coded. Replay is labelled as a dataset demonstration throughout.

Run locally:
  .venv\Scripts\python.exe -m deepshield.app
  http://127.0.0.1:5000

Render:
  Uses the PORT environment variable and listens on 0.0.0.0.
"""

from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, send_file, send_from_directory
from flask_socketio import SocketIO

from deepshield.config import PROJECT_ROOT, path
from deepshield.data import schema
from deepshield.inference.engine import (
    EvaluationMonitor,
    InputError,
    ReplayEngine,
    TrackAEngine,
    metrics_catalog,
    validate_records,
)
from deepshield.intel.mitre import load_validated
from deepshield.monitoring.history import History

log = logging.getLogger("deepshield.app")
FIGURES = PROJECT_ROOT / "reports" / "figures"


class State:
    """Lazily loaded engines (the app starts even if some artifacts are missing)."""

    def __init__(self, db_path=None, replay_max_records=None):
        self.history = History(db_path) if db_path else History()
        self.replay_max_records = replay_max_records
        self._a = self._b = None
        self.errors: dict[str, str] = {}
        self.monitor = EvaluationMonitor()
        self.replay_thread: threading.Thread | None = None
        self.replay_stop = threading.Event()
        self.replay_status = {
            "running": False,
            "model": None,
            "speed": 0,
            "processed": 0,
            "total": 0,
            "session": None,
        }
        self._lock = threading.Lock()

    @property
    def a(self) -> TrackAEngine | None:
        with self._lock:
            if self._a is None and "track_a" not in self.errors:
                try:
                    self._a = TrackAEngine()
                except Exception as e:
                    self.errors["track_a"] = f"{type(e).__name__}: {e}"
                    log.exception("Track A engine failed to load")
            return self._a

    @property
    def b(self) -> ReplayEngine | None:
        with self._lock:
            if self._b is None and "track_b" not in self.errors:
                try:
                    self._b = ReplayEngine(self.replay_max_records)
                except Exception as e:
                    self.errors["track_b"] = f"{type(e).__name__}: {e}"
                    log.exception("Replay engine failed to load")
            return self._b


def create_app(db_path=None, replay_max_records=None) -> tuple[Flask, SocketIO]:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
    sio = SocketIO(app, async_mode="threading")
    S = State(db_path, replay_max_records)
    app.extensions["deepshield_state"] = S

    def err(msg, code=400):
        return jsonify({"error": msg}), code

    # Pages / static
    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/figures/<path:p>")
    def figures(p):
        return send_from_directory(FIGURES, p)

    # Status / overview
    @app.route("/api/status")
    def status():
        a, b = S.a, S.b
        return jsonify({
            "track_a_models": a.available() if a else [],
            "replay_models": b.available() if b else [],
            "replay_records": len(b.records) if b else 0,
            "lstm_sequence_length": b.seq_len if b else None,
            "errors": S.errors,
            "replay": S.replay_status,
            "attack_version": load_validated().get("attack_version"),
        })

    @app.route("/api/overview")
    def overview():
        ov = S.history.overview()
        ov["recent_alerts"] = S.history.alerts(limit=8)["rows"]
        ov["replay"] = S.replay_status
        ov["evaluation"] = S.monitor.summary()
        return jsonify(ov)

    # History / alerts
    @app.route("/api/predictions")
    def predictions():
        q = request.args
        return jsonify(
            S.history.predictions(
                q.get("search", ""),
                q.get("category", ""),
                q.get("severity", ""),
                q.get("source", ""),
                q.get("only_attacks") == "1",
                int(q.get("limit", 100)),
                int(q.get("offset", 0)),
            )
        )

    @app.route("/api/alerts")
    def alerts():
        q = request.args
        return jsonify(
            S.history.alerts(
                q.get("search", ""),
                q.get("category", ""),
                q.get("severity", ""),
                int(q.get("limit", 100)),
                int(q.get("offset", 0)),
            )
        )

    @app.route("/api/alerts/<alert_id>")
    def alert(alert_id):
        a = S.history.alert(alert_id)
        return jsonify(a) if a else err("alert not found", 404)

    @app.route("/api/alerts/<alert_id>/ack", methods=["POST"])
    def ack(alert_id):
        return jsonify({"ok": S.history.acknowledge(alert_id)})

    @app.route("/api/alerts/<alert_id>/explain")
    def alert_explain(alert_id):
        a = S.history.alert(alert_id)
        if not a or not a.get("record"):
            return err("no stored record for this alert", 404)
        if a["source"] == "replay":
            return err(
                "Replay alerts come from Track B sequence/per-flow models; "
                "local explanations are provided for Track A models (Threat detection page).",
                400,
            )
        if not S.a:
            return err("Track A models unavailable", 503)
        try:
            return jsonify(S.a.explain(pd.DataFrame([a["record"]]), a["model"]))
        except InputError as e:
            return err(str(e))

    @app.route("/api/history/clear", methods=["POST"])
    def clear():
        if S.replay_status["running"]:
            return err("stop the replay first")
        S.history.clear()
        S.monitor.reset()
        return jsonify({"ok": True})

    # Threat detection on uploaded / sample records
    @app.route("/api/sample_records")
    def sample_records():
        n = min(int(request.args.get("n", 20)), 200)
        cat = request.args.get("category", "")
        p = path("interim") / "track_a_test.parquet"
        if not p.exists():
            return err("Track A test data not prepared (run scripts/prepare_data.py)", 503)
        df = pd.read_parquet(p)
        if cat:
            df = df[df["attack_cat"] == cat]
        rng = np.random.default_rng(int(request.args.get("seed", int(time.time()))))
        df = df.iloc[rng.choice(len(df), min(n, len(df)), replace=False)]
        return jsonify({
            "records": df[schema.TRACK_A_FEATURES].to_dict("records"),
            "true_labels": df["attack_cat"].tolist(),
            "ids": df["id"].tolist(),
        })

    @app.route("/api/analyze", methods=["POST"])
    def analyze():
        if not S.a:
            return err("Track A models unavailable: " + S.errors.get("track_a", ""), 503)
        model = (request.json or {}).get("model") if request.is_json else request.form.get("model")
        try:
            if "file" in request.files:
                df = pd.read_csv(io.BytesIO(request.files["file"].read()), low_memory=False)
            elif request.is_json:
                df = pd.DataFrame(request.json.get("records", []))
            else:
                return err("send a CSV file (field 'file') or JSON {'records': [...], 'model': ...}")
            model = model or "rf_multiclass"
            valid, problems = validate_records(df, schema.TRACK_A_FEATURES)
            if len(valid) == 0:
                return jsonify({"error": "no valid records", "problems": problems}), 400
            preds = S.a.predict(valid, model)
        except InputError as e:
            return err(str(e))
        except (pd.errors.ParserError, UnicodeDecodeError) as e:
            return err(f"could not read CSV: {e}")

        rows = [
            {
                **p,
                "proto": r["proto"],
                "service": r["service"],
                "record": {
                    c: (r[c] if c in schema.CATEGORICAL else float(r[c]))
                    for c in schema.TRACK_A_FEATURES
                },
            }
            for p, (_, r) in zip(preds, valid.iterrows())
        ]
        alerts_ = S.history.add(rows, "upload", uuid.uuid4().hex[:8], model)
        for a in alerts_:
            sio.emit("alert", a)
        summary = pd.Series([p["category"] for p in preds]).value_counts().to_dict()
        return jsonify({
            "model": model,
            "n": len(preds),
            "summary": summary,
            "problems": problems,
            "predictions": [
                {k: v for k, v in r.items() if k != "record"}
                for r in rows
            ][:500],
            "alerts_created": len(alerts_),
        })

    @app.route("/api/explain", methods=["POST"])
    def explain():
        if not S.a:
            return err("Track A models unavailable", 503)
        body = request.json or {}
        try:
            valid, problems = validate_records(
                pd.DataFrame([body.get("record", {})]),
                schema.TRACK_A_FEATURES,
            )
            if len(valid) != 1:
                return jsonify({"error": "invalid record", "problems": problems}), 400
            return jsonify(S.a.explain(valid, body.get("model", "rf_multiclass")))
        except InputError as e:
            return err(str(e))

    # Models / explainability
    @app.route("/api/models")
    def models():
        cat = metrics_catalog()
        out = {}
        for track, ms in cat.items():
            out[track] = {}
            for name, m in ms.items():
                if name in ("explanations", "leave_one_class_out"):
                    continue
                t = m.get("test", {})
                row = {
                    "name": m.get("model", name),
                    "task": m.get("task"),
                    "fit_seconds": m.get("fit_seconds"),
                    "predict_us_per_record": m.get("predict_us_per_record"),
                    "n_parameters": m.get("n_parameters"),
                    "seeds": m.get("seeds"),
                    "seed_summary": m.get("seed_summary"),
                }
                if "macro_f1" in t:
                    row.update({
                        k: t.get(k)
                        for k in (
                            "accuracy", "macro_precision", "macro_recall", "macro_f1",
                            "weighted_f1", "roc_auc_ovr_macro", "pr_auc_macro",
                            "confusion_matrix", "classes", "per_class",
                        )
                    })
                    row["bootstrap"] = (
                        t.get("bootstrap_macro_f1")
                        or t.get("bootstrap_macro_f1_time_blocks")
                    )
                    bv = t.get("binary_view", {})
                    row["binary"] = {
                        k: bv.get(k)
                        for k in (
                            "f1", "roc_auc", "pr_auc",
                            "false_positive_rate", "false_negative_rate",
                        )
                    }
                else:
                    row.update({
                        k: t.get(k)
                        for k in (
                            "accuracy", "precision", "recall_detection_rate", "f1",
                            "roc_auc", "pr_auc", "false_positive_rate",
                            "false_negative_rate", "confusion_matrix", "threshold",
                        )
                    })
                    row["by_category"] = t.get("by_category")
                out[track][name] = row

        figs = sorted(
            str(p.relative_to(FIGURES)).replace("\\", "/")
            for p in (FIGURES / "models").glob("*.png")
        )
        loco = cat.get("a", {}).get("leave_one_class_out")
        comp = PROJECT_ROOT / "artifacts" / "metrics" / "comparison.json"
        return jsonify({
            "models": out,
            "figures": figs,
            "leave_one_class_out": loco,
            "comparison": json.loads(comp.read_text(encoding="utf-8")) if comp.exists() else None,
        })

    @app.route("/api/explanations")
    def explanations():
        p = path("metrics") / "track_a" / "explanations.json"
        figs = sorted(
            str(f.relative_to(FIGURES)).replace("\\", "/")
            for f in (FIGURES / "xai").glob("*.png")
        )
        return jsonify({
            "global": json.loads(p.read_text(encoding="utf-8")) if p.exists() else None,
            "figures": figs,
        })

    @app.route("/api/mitre")
    def mitre():
        return jsonify(load_validated())

    # Reports
    @app.route("/api/report.pdf")
    def report_pdf():
        expl_p = path("metrics") / "track_a" / "explanations.json"
        expl = json.loads(expl_p.read_text(encoding="utf-8")) if expl_p.exists() else None
        local = None
        if S.a:
            for sev in ("CRITICAL", "HIGH", "MEDIUM"):
                rows = S.history.alerts(severity=sev, limit=50)["rows"]
                rows = [
                    r for r in rows
                    if r["source"] == "upload"
                    and r["model"] in ("rf_multiclass", "mlp", "cnn")
                ]
                if rows:
                    a = S.history.alert(rows[0]["id"])
                    try:
                        local = S.a.explain(pd.DataFrame([a["record"]]), a["model"])
                    except Exception as e:
                        log.warning("explanation for report failed: %s", e)
                    break
        from deepshield.monitoring.reports import build_pdf   # heavy (PDF/plots): load on demand
        pdf = build_pdf(S.history, metrics_catalog(), expl, local)
        return send_file(
            io.BytesIO(pdf),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"deepshield_report_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
        )

    @app.route("/api/report.csv")
    def report_csv():
        from deepshield.monitoring.reports import build_csv
        return send_file(
            io.BytesIO(build_csv(S.history).encode("utf-8")),
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"deepshield_predictions_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        )

    # Replay (Socket.IO)
    def replay_loop(model: str, speed: int, session: str):
        b = S.b
        batch = max(1, min(speed // 4, 500))
        st = S.replay_status
        try:
            while not S.replay_stop.is_set() and not b.finished:
                t0 = time.time()
                flows, truths = b.next_batch(batch)
                preds = b.predict(flows, model)
                S.monitor.update(truths, preds)
                rows = [
                    {
                        **p,
                        "flow_time": f.ltime,
                        "srcip": f.srcip,
                        "dstip": f.dstip,
                        "sport": f.sport,
                        "dsport": f.dsport,
                        "proto": f.proto,
                        "service": f.service,
                    }
                    for p, f in zip(preds, flows)
                ]
                alerts_ = S.history.add(rows, "replay", session, model)
                st["processed"] = b.position
                sio.emit("replay_batch", {
                    "flows": [
                        {
                            "seq": r["seq"],
                            "time": r["flow_time"],
                            "src": f"{r['srcip']}:{r['sport']}",
                            "dst": f"{r['dstip']}:{r['dsport']}",
                            "proto": r["proto"],
                            "service": r["service"],
                            "category": r["category"],
                            "confidence": r["confidence"],
                            "is_attack": r["is_attack"],
                        }
                        for r in rows[-40:]
                    ],
                    "alerts": alerts_[-20:],
                    "status": st,
                    "evaluation": S.monitor.summary(),
                })
                time.sleep(max(0.0, 0.25 - (time.time() - t0)))
        except Exception as e:
            log.exception("replay failed")
            sio.emit("replay_error", {"error": str(e)})
        finally:
            st["running"] = False
            sio.emit("replay_status", st)

    @sio.on("replay_start")
    def replay_start(data):
        if S.replay_status["running"]:
            return {"error": "replay already running"}
        b = S.b
        if not b:
            return {"error": "Replay unavailable: " + S.errors.get("track_b", "")}
        model = (data or {}).get("model", "lstm")
        if model not in b.models:
            return {"error": f"model {model} not available"}
        speed = int((data or {}).get("speed", 400))
        if (data or {}).get("restart") or b.finished:
            b.reset()
            S.monitor.reset()
        S.replay_stop.clear()
        session = uuid.uuid4().hex[:8]
        S.replay_status.update(
            running=True,
            model=model,
            speed=speed,
            total=len(b.records),
            processed=b.position,
            session=session,
        )
        S.replay_thread = threading.Thread(
            target=replay_loop,
            args=(model, speed, session),
            daemon=True,
        )
        S.replay_thread.start()
        return {"ok": True, "status": S.replay_status}

    @sio.on("replay_stop")
    def replay_stop(data=None):
        S.replay_stop.set()
        return {"ok": True}

    return app, sio


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="DeepShield AI dashboard")

    # Render supplies PORT automatically. Local use defaults to port 5000.
    ap.add_argument(
        "--host",
        default=os.environ.get("HOST", "0.0.0.0"),
        help="interface to bind (default: 0.0.0.0)",
    )
    ap.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", 5000)),
        help="port to bind (default: PORT environment variable or 5000)",
    )
    ap.add_argument(
        "--replay-max-records",
        type=int,
        default=None,
        help="limit the replay to the first N test-block flows (faster start-up)",
    )

    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    app, sio = create_app(replay_max_records=args.replay_max_records)

    print(
        f"DeepShield AI dashboard on http://{args.host}:{args.port} "
        "(Ctrl+C to stop)"
    )

    sio.run(
        app,
        host=args.host,
        port=args.port,
        allow_unsafe_werkzeug=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
