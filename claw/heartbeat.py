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

import sys
import threading
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from anomaly import infer_anomaly_level
from call_twilio import call_911, call_caregiver, text_caregiver
from db import log_anomaly_to_db
from big_query import fix_timestamps, get_anchor, fetch_by_timestamp
from query import get_patient_profile

_thread: threading.Thread | None = None
_stop_flag: bool = False

# Polling frequency weights (seconds)
_SEVERITY_FREQ = {0: 60, 1: 30, 2: 15}
_STAGE_LAG     = {0: 15, 1: 15, 2: 30, 3: 60, 4: 120, 5: 300}


def calculate_polling_freq(severity: int, stage: int) -> float:
    """severity (0–2) and stage (0–5) → poll interval in seconds."""
    return (_SEVERITY_FREQ[severity] * 0.45) + (_STAGE_LAG[stage] * 0.55)


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

        # 2. Sync DB timestamps to current interval; capture anchor on first tick
        fixed = fix_timestamps(user_id, interval)
        if fixed:
            print(f"   [timestamps] corrected {fixed} row(s) at freq={interval:.0f}s")

        if anchor is None:
            anchor = get_anchor(user_id)
            if anchor is None:
                print(f"No Whoop data found for {user_id} — stopping heartbeat")
                break

        # 3. Derive the exact timestamp for this tick and fetch that row
        expected_ts = anchor + timedelta(seconds=tick * interval)
        data = fetch_by_timestamp(user_id, expected_ts)
        if not data:
            print(f"No Whoop row at tick={tick} ts={expected_ts.isoformat()} — stopping heartbeat")
            break

        # 4. Score anomaly and route
        anomaly = infer_anomaly_level(data)
        data["anomaly_level"] = anomaly

        process(data, anomaly)
        print(f"   next poll in {interval:.0f}s  (severity={severity}, stage={stage}, anomaly={anomaly})\n")

        tick += 1
        if (max_ticks is None or tick < max_ticks) and not _stop_flag:
            time.sleep(interval)

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
