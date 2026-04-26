"""
heartbeat.py — continuous WHOOP-data polling loop, runs as a daemon thread.

Each tick fetches the Whoop row identified by the composite key (user_id, timestamp),
where timestamp = anchor + tick * interval.  severity and stage come from
patientProfile (via query.py).

Entry points:
  start(user_id) — launch as a non-blocking background daemon thread (use this)
  run(user_id)   — blocking loop, called internally by start()
"""

from __future__ import annotations

import sys
import threading
import time
from datetime import datetime, timedelta

import requests

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from anomaly import infer_anomaly_level
from call_twilio import call_911, call_caregiver, text_caregiver
from db import log_anomaly_to_db
from big_query import get_anchor, fetch_by_timestamp
from query import get_patient_profile

DEBATE_BASE_URL = "https://sova-agents.onrender.com"

_thread: threading.Thread | None = None
_stop_flag: bool = False

# Polling frequency weights (minutes). Higher severity = shorter interval.
_SEVERITY_FREQ = {0: 5, 1: 3, 2: 1}
_STAGE_LAG     = {0: 1, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

_MIN_INTERVAL_MINUTES = 0.25


def calculate_polling_freq(severity: int, stage: int) -> float:
    """severity (0–2) and stage (0–5) → poll interval in minutes (minimum 30s)."""
    raw = (_SEVERITY_FREQ[severity] * 0.45) + (_STAGE_LAG[stage] * 0.55)
    return max(raw, _MIN_INTERVAL_MINUTES)


def _trigger_debate(data: dict, interval: float = 0) -> None:
    """POST to /analyze (blocking), then route using escalation_level from the response."""
    patient_id = data.get("user_id", "unknown")
    profile = get_patient_profile(patient_id) or {}

    payload = {
        "patientId":             patient_id,
        "Age":                   profile.get("Age") or profile.get("age"),
        "Gender":                profile.get("Gender") or profile.get("gender"),
        "Surgery":               profile.get("Surgery") or profile.get("conditions", ""),
        "DischargeDate":         str(profile.get("DischargeDate", "")),
        "RiskLevel":             profile.get("RiskLevel"),
        "BloodPressure":         profile.get("BloodPressure"),
        "HeartRate":             profile.get("HeartRate"),
        "Allergies":             profile.get("Allergies", "None"),
        "CurrentMedications":    profile.get("CurrentMedications", ""),
        "EmergencyContactName":  profile.get("EmergencyContactName", ""),
        "EmergencyContactPhone": profile.get("EmergencyContactPhone", ""),
        "severity":              profile.get("severity"),
        "stage":                 profile.get("stage"),
        "anomaly_level":         data.get("anomaly_level"),
        "interval":              interval,
        "vitals": {
            "HeartRate":     data.get("resting_heart_rate"),
            "BloodPressure": data.get("blood_pressure"),
            "Temperature":   data.get("skin_temp_deviation"),
            "TimeStamp":     data.get("date"),
        },
    }

    try:
        resp = requests.post(f"{DEBATE_BASE_URL}/analyze", json=payload, timeout=300)
        resp.raise_for_status()
        result = resp.json()
    except Exception as exc:
        print(f"[debate] POST /analyze failed: {exc}")
        return

    level = result.get("escalation_level", 1)
    print(f"[debate] decision → urgency={result.get('urgency_level')} action={result.get('immediate_action')} level={level}")

    if level == 4:
        _fire(call_911, data)
    elif level == 3:
        _fire(call_caregiver, data)
    elif level == 2:
        _fire(text_caregiver, data)


def _log_anomaly(data: dict, level: int) -> None:
    log_anomaly_to_db(data, level)


def _fire(fn, *args) -> None:
    """Spawn fn(*args) in a detached thread so the heartbeat loop never blocks on it."""
    threading.Thread(target=fn, args=args, daemon=False).start()


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
            anchor = get_anchor(user_id)
            if anchor is None:
                print(f"No Whoop data found for {user_id} — stopping heartbeat")
                break

        # 2. Derive the exact timestamp for this tick and fetch that row
        expected_ts = anchor + timedelta(minutes=tick * interval)
        data = fetch_by_timestamp(user_id, expected_ts)
        if not data:
            print(f"No Whoop row at tick={tick} ts={expected_ts.isoformat()} — stopping heartbeat")
            break

        # 3. Score anomaly and trigger debate
        anomaly = infer_anomaly_level(data)
        data["anomaly_level"] = anomaly

        ts = data.get("date", datetime.now().isoformat())
        print(f"[{ts}] {user_id}  recovery={data.get('recovery_score')}  hrv={data.get('hrv')}  rhr={data.get('resting_heart_rate')}  strain={data.get('day_strain')}  sleep={data.get('sleep_performance')}%")

        #copilot
        if anomaly == 4:
            _fire(call_caregiver, data)
        elif anomaly > 1:
            _fire(_trigger_debate, data, interval)
        else:
            _fire(_log_anomaly, data, anomaly)
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
