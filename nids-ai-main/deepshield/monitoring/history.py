"""Persistent prediction history and alerts (SQLite, standard library).

Every prediction the dashboard makes (replay or uploaded records) is stored
with its source, model, network metadata, confidence and timestamp; attack
predictions also create an alert with a severity level and candidate MITRE
ATT&CK context. The database lives in artifacts/runtime/ (git-ignored).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from deepshield.config import ensure_dir, PROJECT_ROOT
from deepshield.intel.mitre import SEVERITY, enrich

DB_PATH = PROJECT_ROOT / "artifacts" / "runtime" / "deepshield.db"
SEVERITY_ORDER = ["CLEAN", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created REAL NOT NULL,            -- wall-clock time of the prediction
    source TEXT NOT NULL,             -- 'replay' | 'upload' | 'sample'
    session TEXT NOT NULL,
    model TEXT NOT NULL,
    flow_time INTEGER,                -- flow end time from the dataset (epoch s), if known
    srcip TEXT, dstip TEXT, sport INTEGER, dsport INTEGER, proto TEXT, service TEXT,
    category TEXT NOT NULL,
    is_attack INTEGER NOT NULL,
    confidence REAL,
    anomaly_score REAL,
    severity TEXT NOT NULL,
    record_json TEXT                  -- input features, for explanations
);
CREATE INDEX IF NOT EXISTS ix_pred_created ON predictions(created);
CREATE INDEX IF NOT EXISTS ix_pred_cat ON predictions(category);
CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    prediction_id INTEGER NOT NULL REFERENCES predictions(id),
    created REAL NOT NULL,
    severity TEXT NOT NULL,
    category TEXT NOT NULL,
    model TEXT NOT NULL,
    srcip TEXT, dstip TEXT,
    confidence REAL,
    mitre_json TEXT,
    acknowledged INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_alert_created ON alerts(created);
"""


def severity_for(category: str, is_attack: bool, confidence: float | None) -> str:
    """Category default severity, lowered one level when the model is unsure."""
    if not is_attack:
        return "CLEAN"
    sev = SEVERITY.get(category, "MEDIUM")
    if confidence is not None and confidence < 0.6:
        sev = SEVERITY_ORDER[max(1, SEVERITY_ORDER.index(sev) - 1)]
    return sev


class History:
    def __init__(self, db_path: Path = DB_PATH):
        ensure_dir(Path(db_path).parent)
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)

    # ── writing ───────────────────────────────────────────────────────────────
    def add(self, rows: list[dict], source: str, session: str, model: str) -> list[dict]:
        """Store predictions; returns the alerts created."""
        now = time.time()
        alerts = []
        with self._lock, self._conn:
            for r in rows:
                cat = r["category"]
                # binary/anomaly models output "Attack"/"Anomalous" (no category): MEDIUM
                sev = ("MEDIUM" if r["is_attack"] and cat not in SEVERITY
                       else severity_for(cat, r["is_attack"], r.get("confidence")))
                cur = self._conn.execute(
                    "INSERT INTO predictions (created, source, session, model, flow_time, srcip, dstip,"
                    " sport, dsport, proto, service, category, is_attack, confidence, anomaly_score,"
                    " severity, record_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (now, source, session, model, r.get("flow_time"), r.get("srcip"), r.get("dstip"),
                     r.get("sport"), r.get("dsport"), r.get("proto"), r.get("service"), cat,
                     int(r["is_attack"]), r.get("confidence"), r.get("anomaly_score"), sev,
                     json.dumps(r.get("record"), default=float) if r.get("record") else None))
                if r["is_attack"]:
                    a = {"id": uuid.uuid4().hex[:10], "prediction_id": cur.lastrowid, "created": now,
                         "severity": sev, "category": cat, "model": model, "srcip": r.get("srcip"),
                         "dstip": r.get("dstip"), "confidence": r.get("confidence"),
                         "mitre": enrich(cat) if cat in SEVERITY else None}
                    self._conn.execute(
                        "INSERT INTO alerts (id, prediction_id, created, severity, category, model, srcip,"
                        " dstip, confidence, mitre_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (a["id"], a["prediction_id"], now, sev, cat, model, a["srcip"], a["dstip"],
                         a["confidence"], json.dumps(a["mitre"])))
                    alerts.append(a)
        return alerts

    def clear(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM alerts")
            self._conn.execute("DELETE FROM predictions")

    def acknowledge(self, alert_id: str) -> bool:
        with self._lock, self._conn:
            return self._conn.execute("UPDATE alerts SET acknowledged=1 WHERE id=?",
                                      (alert_id,)).rowcount == 1

    # ── reading ───────────────────────────────────────────────────────────────
    def _q(self, sql, args=()):
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    def overview(self) -> dict:
        tot = self._q("SELECT COUNT(*) n, SUM(is_attack) a, MIN(created) t0, MAX(created) t1,"
                      " MIN(flow_time) f0, MAX(flow_time) f1 FROM predictions")[0]
        cats = self._q("SELECT category, COUNT(*) n FROM predictions WHERE is_attack=1"
                       " GROUP BY category ORDER BY n DESC")
        sev = self._q("SELECT severity, COUNT(*) n FROM alerts GROUP BY severity")
        by_model = self._q("SELECT model, source, COUNT(*) n, SUM(is_attack) a FROM predictions"
                           " GROUP BY model, source")
        top_src = self._q("SELECT srcip, COUNT(*) n FROM alerts WHERE srcip IS NOT NULL"
                          " GROUP BY srcip ORDER BY n DESC LIMIT 5")
        n, a = tot["n"] or 0, tot["a"] or 0
        return {"total": n, "malicious": a, "normal": n - a,
                "attack_share": a / n if n else 0.0,
                "first_prediction": tot["t0"], "last_prediction": tot["t1"],
                "flow_time_start": tot["f0"], "flow_time_end": tot["f1"],
                "attack_distribution": {r["category"]: r["n"] for r in cats},
                "severity": {s: next((r["n"] for r in sev if r["severity"] == s), 0)
                             for s in SEVERITY_ORDER[1:]},
                "by_model": by_model, "top_sources": top_src}

    def predictions(self, search: str = "", category: str = "", severity: str = "",
                    source: str = "", only_attacks: bool = False, limit: int = 100,
                    offset: int = 0) -> dict:
        where, args = ["1=1"], []
        if search:
            where.append("(srcip LIKE ? OR dstip LIKE ? OR proto LIKE ? OR service LIKE ? OR category LIKE ?)")
            args += [f"%{search}%"] * 5
        for col, val in (("category", category), ("severity", severity), ("source", source)):
            if val:
                where.append(f"{col}=?")
                args.append(val)
        if only_attacks:
            where.append("is_attack=1")
        w = " AND ".join(where)
        total = self._q(f"SELECT COUNT(*) n FROM predictions WHERE {w}", args)[0]["n"]
        rows = self._q(f"SELECT id, created, source, model, flow_time, srcip, dstip, sport, dsport, proto,"
                       f" service, category, is_attack, confidence, anomaly_score, severity"
                       f" FROM predictions WHERE {w} ORDER BY id DESC LIMIT ? OFFSET ?",
                       args + [min(int(limit), 1000), int(offset)])
        return {"total": total, "rows": rows}

    def alerts(self, search: str = "", category: str = "", severity: str = "",
               limit: int = 100, offset: int = 0) -> dict:
        where, args = ["1=1"], []
        if search:
            where.append("(a.srcip LIKE ? OR a.dstip LIKE ? OR a.category LIKE ? OR a.id LIKE ?)")
            args += [f"%{search}%"] * 4
        for col, val in (("a.category", category), ("a.severity", severity)):
            if val:
                where.append(f"{col}=?")
                args.append(val)
        w = " AND ".join(where)
        total = self._q(f"SELECT COUNT(*) n FROM alerts a WHERE {w}", args)[0]["n"]
        rows = self._q(f"SELECT a.*, p.flow_time, p.source, p.proto, p.service, p.sport, p.dsport"
                       f" FROM alerts a JOIN predictions p ON p.id=a.prediction_id WHERE {w}"
                       f" ORDER BY a.created DESC, a.rowid DESC LIMIT ? OFFSET ?",
                       args + [min(int(limit), 1000), int(offset)])
        for r in rows:
            r["mitre"] = json.loads(r.pop("mitre_json") or "null")
        return {"total": total, "rows": rows}

    def alert(self, alert_id: str) -> dict | None:
        rows = self._q("SELECT a.*, p.record_json, p.flow_time, p.source, p.proto, p.service, p.sport,"
                       " p.dsport, p.anomaly_score FROM alerts a JOIN predictions p ON p.id=a.prediction_id"
                       " WHERE a.id=?", (alert_id,))
        if not rows:
            return None
        r = rows[0]
        r["mitre"] = json.loads(r.pop("mitre_json") or "null")
        r["record"] = json.loads(r.pop("record_json") or "null")
        return r

    def export_rows(self, limit: int = 100_000) -> list[dict]:
        return self._q("SELECT id, created, source, model, flow_time, srcip, sport, dstip, dsport, proto,"
                       " service, category, is_attack, confidence, anomaly_score, severity"
                       " FROM predictions ORDER BY id LIMIT ?", (limit,))
