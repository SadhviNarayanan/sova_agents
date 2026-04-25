"""
heartbeat.py — continuous WHOOP-data polling loop, runs as a daemon thread.

Mirrors openclaw's orchestration pattern: fetch → process → sleep,
where sleep interval is driven by severity and stage passed in the data payload,
and escalation is driven by anomaly_level (1–4) also received from the data source.

Entry points:
  start() — launch as a non-blocking background daemon thread (use this)
  run()   — blocking loop, called internally by start()
Data source: synthetic_data.get_data()  (swap for real DB/CSV read when ready)
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

from call_twilio import call_911, call_caregiver, text_caregiver
from db import log_anomaly_to_db
from synthetic_data import get_data

_thread: threading.Thread | None = None
_stop_flag: bool = False

# Polling frequency weights (seconds)
_SEVERITY_FREQ = {0: 60, 1: 30, 2: 15}
_STAGE_LAG     = {0: 15, 1: 15, 2: 30, 3: 60, 4: 120, 5: 300}


def calculate_polling_freq(severity: int, stage: int) -> float:
    """severity (0–2) and stage (0–5) → poll interval in seconds."""
    return (_SEVERITY_FREQ[severity] * 0.45) + (_STAGE_LAG[stage] * 0.55)


def _log_anomaly(data: dict, level: int) -> None:
    """Input: snapshot + anomaly level → writes directly to the GCP database."""
    log_anomaly_to_db(data, level)


def _fire(fn, *args) -> None:
    """Spawn fn(*args) in a detached thread — heartbeat loop does not block on it."""
    t = threading.Thread(target=fn, args=args, daemon=False)
    t.start()


def process(data: dict, anomaly: int) -> None:
    """Input: WHOOP snapshot + anomaly level received from data source → log and route."""
    uid      = data.get("user_id", "unknown")
    ts       = data.get("date", datetime.now(timezone.utc).isoformat())
    recovery = data.get("recovery_score")
    hrv      = data.get("hrv")
    rhr      = data.get("resting_heart_rate")
    strain   = data.get("day_strain")
    sleep_p  = data.get("sleep_performance")

    print(f"[{ts}] {uid}  recovery={recovery}  hrv={hrv}  rhr={rhr}  strain={strain}  sleep={sleep_p}%")

    if anomaly == 4:
        _fire(call_911)
    elif anomaly == 3:
        _fire(call_caregiver)
    elif anomaly == 2:
        _fire(text_caregiver)
    elif anomaly == 1:
        _fire(_log_anomaly, data, anomaly)


def run(max_ticks: int | None = None) -> None:
    """Blocking poll loop — call start() instead to run this in the background."""
    global _stop_flag
    _stop_flag = False
    print("Heartbeat monitor started")
    tick = 0

    while (max_ticks is None or tick < max_ticks) and not _stop_flag:
        data     = get_data()
        severity = data.get("severity", 1)
        stage    = data.get("stage", 2)
        anomaly  = data.get("anomaly_level", 0)
        interval = calculate_polling_freq(severity, stage)

        process(data, anomaly)
        print(f"   next poll in {interval:.0f}s  (severity={severity}, stage={stage}, anomaly={anomaly})\n")

        tick += 1
        if (max_ticks is None or tick < max_ticks) and not _stop_flag:
            time.sleep(interval)

    print("Heartbeat monitor stopped")


def start() -> None:
    """Launch the heartbeat loop as a background daemon thread (non-blocking)."""
    global _thread
    if _thread and _thread.is_alive():
        return
    _thread = threading.Thread(target=run, daemon=True, name="heartbeat")
    _thread.start()


def stop() -> None:
    """Signal the heartbeat thread to exit after its current sleep interval."""
    global _stop_flag
    _stop_flag = True


if __name__ == "__main__":
    start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Heartbeat shutting down")
