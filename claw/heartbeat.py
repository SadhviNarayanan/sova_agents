"""
heartbeat.py — continuous WHOOP-data polling loop, runs as a daemon thread.

Each tick fetches the Whoop row identified by the composite key (user_id, timestamp),
where timestamp = anchor + tick * interval.  Before each fetch, fix_timestamps()
rewrites the DB so row N is always exactly at anchor + N * interval, keeping the
two sides in sync.  severity and stage come from patientProfile (via query.py).

Entry points:
  start(user_id) — launch as a non-blocking background daemon thread (use this)
  run(user_id)   — blocking loop, called internally by start()
"""

from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

import requests

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from anomaly import infer_anomaly_level
from call_twilio import call_911, call_caregiver, text_caregiver
from db import log_anomaly_to_db
from big_query import fix_timestamps, get_anchor, fetch_by_timestamp
from query import get_patient_profile

DEBATE_BASE_URL = "http://localhost:8000"

_thread: threading.Thread | None = None
_stop_flag: bool = False

# Polling frequency weights (minutes). Higher severity = shorter interval.
_SEVERITY_FREQ = {0: 90, 1: 60, 2: 30}
_STAGE_LAG     = {0: 10, 1: 10, 2: 20, 3: 30, 4: 60, 5: 120}

_MIN_INTERVAL_MINUTES = 30


def calculate_polling_freq(severity: int, stage: int) -> float:
    """severity (0–2) and stage (0–5) → poll interval in minutes (minimum 30)."""
    raw = (_SEVERITY_FREQ[severity] * 0.45) + (_STAGE_LAG[stage] * 0.55)
    return max(raw, _MIN_INTERVAL_MINUTES)


def _trigger_debate(data: dict) -> None:
    """POST to /start-debate, stream SSE results, then re-route using escalation_level."""
    patient_id = data.get("user_id", "unknown")
    profile = get_patient_profile(patient_id) or {}

    payload = {
        "patient_id":       patient_id,
        "age":              profile.get("age"),
        "gender":           profile.get("gender"),
        "diagnosis":        profile.get("conditions", []),
        "heart_rate":       data.get("resting_heart_rate"),
        "hrv":              data.get("hrv"),
        "recovery_score":   data.get("recovery_score"),
        "day_strain":       data.get("day_strain"),
        "sleep_performance":data.get("sleep_performance"),
        "respiratory_rate": data.get("respiratory_rate"),
        "blood_pressure":   data.get("blood_pressure"),
        "temperature":      data.get("skin_temp_deviation"),
        "triggered_signals":data.get("triggered_signals", []),
        "anomaly_level":    data.get("anomaly_level"),
        "severity":         profile.get("severity"),
        "stage":            profile.get("stage"),
    }

    try:
        requests.post(f"{DEBATE_BASE_URL}/start-debate/{patient_id}", json=payload, timeout=10)
    except Exception as exc:
        print(f"[debate] POST /start-debate failed: {exc}")
        return

    try:
        with requests.get(f"{DEBATE_BASE_URL}/stream/{patient_id}", stream=True, timeout=120) as resp:
            event_type = None
            for raw in resp.iter_lines():
                if not raw:
                    event_type = None
                    continue
                line = raw.decode() if isinstance(raw, bytes) else raw
                if line.startswith("event:"):
                    event_type = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    payload_str = line[len("data:"):].strip()
                    if event_type == "agent_speak":
                        try:
                            msg = json.loads(payload_str)
                            print(f"[debate] {msg.get('agent')}: {msg.get('statement')}")
                        except json.JSONDecodeError:
                            pass
                    elif event_type == "decision":
                        try:
                            decision = json.loads(payload_str)
                            level = int(decision.get("escalation_level", 0))
                            print(f"[debate] decision → escalation_level={level}")
                            if level == 4:
                                _fire(call_911, data)
                            elif level == 3:
                                _fire(call_caregiver, data)
                            elif level == 2:
                                _fire(text_caregiver, data)
                            elif level == 1:
                                _fire(_log_anomaly, data, level)
                        except (json.JSONDecodeError, ValueError):
                            pass
                    elif event_type == "error":
                        print(f"[debate] stream error: {payload_str}")
                        return
    except Exception as exc:
        print(f"[debate] stream error: {exc}")


def _log_anomaly(data: dict, level: int) -> None:
    log_anomaly_to_db(data, level)


def _fire(fn, *args) -> None:
    """Spawn fn(*args) in a detached thread so the heartbeat loop never blocks on it."""
    threading.Thread(target=fn, args=args, daemon=False).start()


def process(data: dict, anomaly: int) -> None:
    uid      = data.get("user_id", "unknown")
    ts       = data.get("date", datetime.now(timezone.utc).isoformat())
    recovery = data.get("recovery_score")
    hrv      = data.get("hrv")
    rhr      = data.get("resting_heart_rate")
    strain   = data.get("day_strain")
    sleep_p  = data.get("sleep_performance")

    print(f"[{ts}] {uid}  recovery={recovery}  hrv={hrv}  rhr={rhr}  strain={strain}  sleep={sleep_p}%")

    if anomaly == 4:
        _fire(call_911, data)
    elif anomaly == 3:
        _fire(call_caregiver, data)
    elif anomaly == 2:
        _fire(text_caregiver, data)
    elif anomaly == 1:
        _fire(_log_anomaly, data, anomaly)

    if anomaly >= 2:
        _fire(_trigger_debate, data)


def run(user_id: str, max_ticks: int | None = None) -> None:
    """Blocking poll loop — call start() instead to run this in the background."""
    global _stop_flag
    _stop_flag = False
    print(f"Heartbeat monitor started for user={user_id}")

    anchor: datetime | None = None
    tick = 0

    while (max_ticks is None or tick < max_ticks) and not _stop_flag:
        # 1. Patient profile drives severity/stage → interval
        profile = get_patient_profile(user_id)
        if not profile:
            print(f"No patient profile found for {user_id} — stopping heartbeat")
            break
        severity = profile["severity"]
        stage    = profile["stage"]
        interval = calculate_polling_freq(severity, stage)

        
        if anchor is None:
            anchor = get_anchor(user_id) #base timestamp
            if anchor is None:
                print(f"No Whoop data found for {user_id} — stopping heartbeat")
                break

        # 3. Derive the exact timestamp for this tick and fetch that row
        expected_ts = anchor + timedelta(minutes=tick * interval)
        data = fetch_by_timestamp(user_id, expected_ts)
        if not data:
            print(f"No Whoop row at tick={tick} ts={expected_ts.isoformat()} — stopping heartbeat")
            break

        # 4. Score anomaly and route
        anomaly = infer_anomaly_level(data)
        data["anomaly_level"] = anomaly

        process(data, anomaly)
        print(f"   next poll in {interval:.0f}min  (severity={severity}, stage={stage}, anomaly={anomaly})\n")

        tick += 1
        if (max_ticks is None or tick < max_ticks) and not _stop_flag:
            time.sleep(interval * 60)

    print("Heartbeat monitor stopped")


def start(user_id: str) -> None:
    """Launch the heartbeat loop as a background daemon thread (non-blocking)."""
    global _thread
    if _thread and _thread.is_alive():
        return
    _thread = threading.Thread(target=run, args=(user_id,), daemon=True, name="heartbeat")
    _thread.start()


def stop() -> None:
    """Signal the heartbeat thread to exit after its current sleep interval."""
    global _stop_flag
    _stop_flag = True


if __name__ == "__main__":
    import sys as _sys
    _pid = _sys.argv[1] if len(_sys.argv) > 1 else "PATIENT_001"
    start(_pid)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Heartbeat shutting down")
