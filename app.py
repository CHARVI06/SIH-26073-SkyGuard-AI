import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components

from backend import start_backend

st.set_page_config(
    page_title="SKYGUARD AI - Station Overview",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: #090812;
    }

    [data-testid="stMainBlockContainer"],
    .block-container {
    max-width: 100% !important;
    padding: 4rem 0 0 !important;
}

    [data-testid="stVerticalBlock"] {
        gap: 0 !important;
    }

    iframe {
        display: block !important;
        width: 100% !important;
        border: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
HTML_FILE = BASE_DIR / "index.html"
CSS_FILE = BASE_DIR / "skyguard-cosmic-aurora.css"

STATION_ID = "AWS-001"
DEFAULT_API = "http://127.0.0.1:5000/api"


def get_api_url():
    try:
        return st.secrets.get("API_URL", DEFAULT_API).rstrip("/")
    except Exception:
        return DEFAULT_API


API_URL = get_api_url()

# Start the single local backend/data source once per process.
start_backend()


def get_json(url):
    try:
        response = requests.get(
            url,
            params={"station_id": STATION_ID},
            timeout=5,
        )
        response.raise_for_status()
        return response.json(), True
    except Exception:
        return None, False


def find_readings(payload):
    if payload is None:
        return []

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in (
            "readings", "data", "history", "results",
            "items", "records", "observations"
        ):
            value = payload.get(key)
            if isinstance(value, list):
                return value

        if any(k in payload for k in ("temperature", "temp", "humidity", "pressure")):
            return [payload]

    return []


def number(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_reading(row):
    if not isinstance(row, dict):
        return None

    temp = number(row.get("temperature", row.get("temp")))
    humidity = number(row.get("humidity", row.get("hum")))
    pressure = number(row.get("pressure", row.get("pres")))

    if temp is None and humidity is None and pressure is None:
        return None

    anomaly_raw = row.get(
        "anomaly",
        row.get("is_anomaly", row.get("anomaly_detected", False))
    )

    if isinstance(anomaly_raw, str):
        anomaly = anomaly_raw.lower() in {
            "true", "1", "yes", "anomaly", "abnormal"
        }
    else:
        anomaly = bool(anomaly_raw)

    score = number(row.get("anomaly_score", row.get("score")))
    confidence = number(row.get("confidence", row.get("model_confidence")))
    health = number(
        row.get(
            "health_score",
            row.get("sensor_health", row.get("health"))
        )
    )

    if score is not None and 0 <= score <= 1:
        score *= 100
    if confidence is not None and 0 <= confidence <= 1:
        confidence *= 100
    if health is not None and 0 <= health <= 1:
        health *= 100

    timestamp = (
        row.get("timestamp")
        or row.get("time")
        or row.get("datetime")
        or row.get("created_at")
    )

    return {
        "timestamp": timestamp,
        "temperature": temp,
        "humidity": humidity,
        "pressure": pressure,
        "anomaly": anomaly,
        "anomaly_score": score,
        "confidence": confidence,
        "health_score": health,
        "severity": row.get("severity", row.get("risk_level")),
        "root_cause": row.get("root_cause", row.get("cause")),
        "explanation": row.get(
            "explanation",
            row.get("message", row.get("reason"))
        ),
    }


def fetch_backend_data():
    endpoints = [
        f"{API_URL}/stations/{STATION_ID}/readings",
        f"{API_URL}/readings",
        f"{API_URL}/history",
        f"{API_URL}/observations",
    ]

    for endpoint in endpoints:
        payload, ok = get_json(endpoint)
        if not ok:
            continue

        rows = [normalize_reading(x) for x in find_readings(payload)]
        rows = [x for x in rows if x is not None]

        if rows:
            return rows, True, endpoint

    return [], False, None


def build_initial_html(rows, backend_ok):
    page = HTML_FILE.read_text(encoding="utf-8")

    if CSS_FILE.is_file():
        redesign_css = CSS_FILE.read_text(encoding="utf-8")
        page = page.replace(
            "</head>",
            f'<style id="atmospheric-observatory-skin">{redesign_css}</style></head>',
            1,
        )

    # Replace only the existing chart data block in memory so the first
    # browser render uses backend data when the backend is reachable.
    chart_rows = rows[-10:]
    temps = [round(float(r["temperature"]), 3) if r["temperature"] is not None else 0 for r in chart_rows]
    hums = [round(float(r["humidity"]), 3) if r["humidity"] is not None else 0 for r in chart_rows]
    pressures = [round(float(r["pressure"]), 3) if r["pressure"] is not None else 0 for r in chart_rows]

    anomaly_indices = [i for i, r in enumerate(chart_rows) if r.get("anomaly")]
    anomaly_index = anomaly_indices[-1] if anomaly_indices else -1

    raw_data = f"""var rawData = {{
            temp:     {{ values: {json.dumps(temps)}, min: 15, max: 35, color: '#ff4370', anomalyIndex: {anomaly_index} }},
            humidity: {{ values: {json.dumps(hums)}, min: 40, max: 100, color: '#00f2ff', anomalyIndex: {anomaly_index} }},
            pressure: {{ values: {json.dumps(pressures)}, min: 980, max: 1040, color: '#8b5cf6', anomalyIndex: {anomaly_index} }}
        }};
        window.rawData = rawData;"""

    page, count = re.subn(
        r"(?:const|let|var) rawData = \{.*?\n\s*\};",
        raw_data,
        page,
        count=1,
        flags=re.DOTALL,
    )

    if count != 1:
        raise RuntimeError("Could not locate the existing chart data block.")

    latest = rows[-1] if rows else {}
    page = page.replace(
        "<body>",
        f"<body data-api-url=\"{API_URL}\" data-backend-connected=\"{'true' if backend_ok else 'false'}\" data-initial-temp=\"{latest.get('temperature', '')}\" data-initial-humidity=\"{latest.get('humidity', '')}\" data-initial-pressure=\"{latest.get('pressure', '')}\" data-initial-health=\"{latest.get('health_score', '')}\" data-initial-confidence=\"{latest.get('confidence', '')}\" data-initial-timestamp=\"{latest.get('timestamp', '')}\">",
        1,
    )

    return page


rows, backend_ok, _ = fetch_backend_data()

html = build_initial_html(rows, backend_ok)

components.html(
    html,
    height=900,
    scrolling=False,
)
