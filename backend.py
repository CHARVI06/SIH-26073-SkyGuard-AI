"""SkyGuard AI local backend and deterministic AWS stream simulator.

The simulator is the single data source for this local demo when no real AWS/MQTT
source is configured. It writes observations to SQLite and exposes them via REST.
"""
from __future__ import annotations

import math
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    from sklearn.ensemble import IsolationForest
except Exception:
    IsolationForest = None
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from urllib.parse import parse_qs, urlparse

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "skyguard.db"
STATION_ID = "AWS-001"
LOCATION = "Greater Noida, Uttar Pradesh, India"
TZ = ZoneInfo("Asia/Kolkata")
UTC = timezone.utc
INTERVAL_SECONDS = 60

_db_lock = threading.Lock()
_server = None
_server_thread = None
_stream_thread = None


def connect():
    conn = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with connect() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            temperature REAL NOT NULL,
            humidity REAL NOT NULL,
            pressure REAL NOT NULL,
            anomaly INTEGER NOT NULL DEFAULT 0,
            anomaly_score REAL NOT NULL DEFAULT 0,
            confidence REAL NOT NULL DEFAULT 0,
            health_score REAL NOT NULL DEFAULT 100,
            severity TEXT NOT NULL DEFAULT 'Normal',
            root_cause TEXT,
            explanation TEXT
        )""")
        db.commit()
        count = db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        if count == 0:
            seed_history(db)


def deterministic_reading(at: datetime):
    """Produce plausible deterministic weather observations for local simulation."""
    local = at.astimezone(TZ)
    hour = local.hour + local.minute / 60.0
    day_index = local.toordinal() % 29
    temp = 25.0 + 4.2 * math.sin((hour - 6) * math.pi / 12) + 0.18 * math.sin(day_index)
    humidity = 66.0 - 10.0 * math.sin((hour - 6) * math.pi / 12) + 0.5 * math.cos(day_index)
    pressure = 1012.0 + 2.8 * math.sin((hour + 2) * math.pi / 12) + 0.4 * math.sin(day_index / 3)
    return round(temp, 2), round(humidity, 2), round(pressure, 2)


def analyze(temp, humidity, pressure, history):
    # Isolation Forest is the primary local ML detector. The statistical path is
    # retained as a deterministic fallback if sklearn is unavailable.
    if len(history) < 12:
        return False, 0.02, 98.0, 99.0, "Normal", None, "No abnormal sensor behavior detected."
    vals = [(r[0], r[1], r[2]) for r in history[-128:]]
    score = 0.0
    anomaly = False
    z = [0.0, 0.0, 0.0]
    if IsolationForest is not None:
        model = IsolationForest(n_estimators=80, contamination=0.05, random_state=42)
        model.fit(vals)
        decision = float(model.decision_function([[temp, humidity, pressure]])[0])
        anomaly = int(model.predict([[temp, humidity, pressure]])[0]) == -1
        score = min(1.0, max(0.0, 0.5 - decision))
        means = [sum(v[i] for v in vals) / len(vals) for i in range(3)]
        stds = []
        for i in range(3):
            var = sum((v[i] - means[i]) ** 2 for v in vals) / max(1, len(vals) - 1)
            stds.append(max(math.sqrt(var), 0.25 if i == 0 else 0.8))
        z = [abs((x - m) / s) for x, m, s in zip((temp, humidity, pressure), means, stds)]
    else:
        means = [sum(v[i] for v in vals) / len(vals) for i in range(3)]
        stds = []
        for i in range(3):
            var = sum((v[i] - means[i]) ** 2 for v in vals) / max(1, len(vals) - 1)
            stds.append(max(math.sqrt(var), 0.25 if i == 0 else 0.8))
        z = [abs((x - m) / s) for x, m, s in zip((temp, humidity, pressure), means, stds)]
        score = min(1.0, max(z) / 4.0)
        anomaly = score >= 0.75
    confidence = min(99.0, 70.0 + score * 29.0)
    health = max(0.0, 100.0 - score * 35.0)
    cause = None
    explanation = "No abnormal sensor behavior detected."
    severity = "Normal"
    if anomaly:
        idx = z.index(max(z))
        cause = ["temperature_spike", "humidity_deviation", "pressure_deviation"][idx]
        severity = "High" if score >= 0.9 else "Medium"
        explanation = f"Multivariate deviation detected in {cause.replace('_', ' ')}."
    return anomaly, round(score, 4), round(confidence, 1), round(health, 1), severity, cause, explanation


def insert_observation(at=None, forced=None):
    at = at or datetime.now(UTC)
    temp, humidity, pressure = deterministic_reading(at)
    if forced:
        temp, humidity, pressure = forced
    with _db_lock, connect() as db:
        history = db.execute("SELECT temperature, humidity, pressure FROM observations WHERE station_id=? ORDER BY timestamp", (STATION_ID,)).fetchall()
        anomaly, score, confidence, health, severity, cause, explanation = analyze(temp, humidity, pressure, history)
        db.execute("""INSERT INTO observations
            (station_id,timestamp,temperature,humidity,pressure,anomaly,anomaly_score,confidence,health_score,severity,root_cause,explanation)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (STATION_ID, at.astimezone(UTC).isoformat(), temp, humidity, pressure, int(anomaly), score, confidence, health, severity, cause, explanation))
        db.commit()


def seed_history(db):
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    # One observation every 6 hours for the previous 30 days: enough real backend
    # history to exercise 24H/7D/30D without creating frontend-only data.
    rows = []
    for i in range(120, -1, -1):
        at = now - timedelta(hours=6 * i)
        t, h, p = deterministic_reading(at)
        rows.append((STATION_ID, at.isoformat(), t, h, p, 0, 0.02, 98.0, 99.0, "Normal", None, "No abnormal sensor behavior detected."))
    db.executemany("""INSERT INTO observations
        (station_id,timestamp,temperature,humidity,pressure,anomaly,anomaly_score,confidence,health_score,severity,root_cause,explanation)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
    db.commit()


def stream_loop():
    last_slot = None
    while True:
        now = datetime.now(UTC).replace(second=0, microsecond=0)
        slot = now.replace(second=(now.second // INTERVAL_SECONDS) * INTERVAL_SECONDS)
        if slot != last_slot:
            insert_observation(slot)
            last_slot = slot
        time.sleep(1)


def rows(limit=500, hours=None, anomalies_only=False):
    sql = "SELECT * FROM observations WHERE station_id=?"
    params = [STATION_ID]
    if hours is not None:
        cutoff = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
        sql += " AND timestamp >= ?"
        params.append(cutoff)
    if anomalies_only:
        sql += " AND anomaly=1"
    sql += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    with _db_lock, connect() as db:
        return [dict(r) for r in db.execute(sql, params).fetchall()]


def serialize(row):
    if not row:
        return None
    row = dict(row)
    row["is_anomaly"] = bool(row.pop("anomaly", 0))
    return row


def current_health():
    try:
        latest = rows(limit=1)[0]
        age = (datetime.now(UTC) - datetime.fromisoformat(latest["timestamp"])).total_seconds()
        freshness = "LIVE" if age <= 90 else "STALE" if age <= 300 else "OFFLINE"
        return {"backend": "ONLINE", "database": "ONLINE", "sensor_stream": "ACTIVE" if freshness == "LIVE" else "INACTIVE", "ai_detection": "ACTIVE", "station": freshness, "data_pipeline": "RUNNING", "freshness": freshness, "latest_age_seconds": round(age, 1)}
    except Exception:
        return {"backend": "ONLINE", "database": "ERROR", "sensor_stream": "INACTIVE", "ai_detection": "INACTIVE", "station": "OFFLINE", "data_pipeline": "STOPPED", "freshness": "OFFLINE"}


def response_json(handler, data, status=200):
    raw = json.dumps(data).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.end_headers()
    handler.wfile.write(raw)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        return
    def do_OPTIONS(self):
        response_json(self, {})
    def do_GET(self):
        path = urlparse(self.path).path
        q = parse_qs(urlparse(self.path).query)
        try:
            if path == "/api/live":
                response_json(self, serialize(rows(limit=1)[0]))
            elif path == "/api/history":
                hours = float(q.get("hours", [24])[0])
                response_json(self, {"readings": [serialize(r) for r in rows(hours=hours)]})
            elif path == "/api/alerts":
                response_json(self, {"alerts": [serialize(r) for r in rows(limit=100, anomalies_only=True)]})
            elif path == "/api/stats":
                all_rows = rows(limit=10000)
                anomalies = sum(bool(r["anomaly"]) for r in all_rows)
                avg_health = sum(r["health_score"] for r in all_rows) / len(all_rows) if all_rows else 0
                high = sum(r["severity"] == "High" for r in all_rows)
                response_json(self, {"total_observations": len(all_rows), "total_anomalies": anomalies, "high_severity": high, "avg_health": round(avg_health, 1), "uptime": "100%"})
            elif path == "/api/health":
                response_json(self, current_health())
            elif path == "/api/station":
                response_json(self, {"station_id": STATION_ID, "location": LOCATION, "timezone": "Asia/Kolkata", "sensor_interval_seconds": INTERVAL_SECONDS})
            else:
                response_json(self, {"error": "Not found"}, 404)
        except Exception as exc:
            response_json(self, {"error": str(exc)}, 500)
    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/inject-anomaly":
            # Controlled demo/testing input; it is stored in the same authoritative DB.
            insert_observation(datetime.now(UTC), forced=(42.0, 18.0, 1080.0))
            response_json(self, {"ok": True, "message": "Controlled anomaly observation inserted."}, 201)
        else:
            response_json(self, {"error": "Not found"}, 404)


def start_backend(host="127.0.0.1", port=5000):
    global _server, _server_thread, _stream_thread
    init_db()
    if _server is None:
        try:
            _server = ThreadingHTTPServer((host, port), Handler)
        except OSError:
            # Another already-running instance is fine for Streamlit reruns.
            _server = False
        if _server:
            _server_thread = threading.Thread(target=_server.serve_forever, daemon=True)
            _server_thread.start()
    if _stream_thread is None:
        _stream_thread = threading.Thread(target=stream_loop, daemon=True)
        _stream_thread.start()


if __name__ == "__main__":
    start_backend()
    print("SkyGuard backend running on http://127.0.0.1:5000")
    while True:
        time.sleep(3600)
