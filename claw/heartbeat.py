"""
heartbeat.py — continuous patient-data polling loop, runs as a daemon thread.

Mirrors openclaw's orchestration pattern: fetch → process → sleep,
where the sleep interval is determined dynamically by calculate_polling_freq()
based on the patient's current severity and post-discharge stage.

Entry points:
  start() — launch as a non-blocking background daemon thread (use this)
  run()   — blocking loop, called internally by start()
Data source: synthetic_data.get_data()  (swap for real source when ready)
"""

import threading
import time
from datetime import datetime, timezone

from synthetic_data import get_data

_thread: threading.Thread | None = None
_stop_flag: bool = False

# Polling frequency weights (seconds)
_SEVERITY_FREQ = {0: 60, 1: 30, 2: 15}
_STAGE_LAG     = {0: 15, 1: 15, 2: 30, 3: 60, 4: 120, 5: 300}


def calculate_polling_freq(severity: int, stage: int) -> float:
    """Input: severity (0–2) and post-discharge stage (0–5) → poll interval in seconds."""
    return (_SEVERITY_FREQ[severity] * 0.45) + (_STAGE_LAG[stage] * 0.55)


def process(data: dict) -> None:
    """Input: patient snapshot dict → side-effects (alerts, logging, escalation hooks)."""
    ts  = data.get("timestamp", datetime.now(timezone.utc).isoformat())
    pid = data.get("patient_id")
    hr  = data.get("heart_rate")
    spo2 = data.get("spo2")

    print(f"[{ts}] {pid}  HR={hr}  SpO2={spo2}%")

    # ── escalation hooks (placeholders) ─────────────────────────────────────
    if hr and hr > 100:
        print(f"  ⚠  Elevated HR ({hr}) — trigger council review")

    if spo2 and spo2 < 94:
        print(f"  ⚠  Low SpO2 ({spo2}%) — trigger council review")


def run(max_ticks: int | None = None) -> None:
    """Blocking poll loop — call start() instead to run this in the background."""
    global _stop_flag
    _stop_flag = False
    print("Heartbeat monitor started")
    tick = 0

    while (max_ticks is None or tick < max_ticks) and not _stop_flag:
        data = get_data()
        severity = data["severity"]
        stage    = data["stage"]
        interval = calculate_polling_freq(severity, stage)

        process(data)
        print(f"   next poll in {interval:.0f}s  (severity={severity}, stage={stage})\n")

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
    # Keep main thread alive so the daemon has something to run against
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Heartbeat shutting down")
