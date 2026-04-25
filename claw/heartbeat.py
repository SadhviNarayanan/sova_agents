"""
heartbeat.py — continuous patient-data polling loop.

Mirrors openclaw's orchestration pattern: fetch → process → sleep,
where the sleep interval is determined dynamically by calculate_polling_freq()
based on the patient's current severity and post-discharge stage.

Entry point: run()
Data source:  synthetic_data.get_data()  (swap for real source when ready)
"""

import time
from datetime import datetime, timezone

from synthetic_data import get_data

# Polling frequency weights (seconds)
_SEVERITY_FREQ = {0: 60, 1: 30, 2: 15}
_STAGE_LAG     = {0: 15, 1: 15, 2: 30, 3: 60, 4: 120, 5: 300}


def calculate_polling_freq(severity: int, stage: int) -> float:
    """Input: severity (0–2) and post-discharge stage (0–5) → poll interval in seconds."""
    return (_SEVERITY_FREQ[severity] * 0.45) + (_STAGE_LAG[stage] * 0.55)


def process(data: dict) -> None:
    """Input: patient snapshot dict → side-effects (alerts, logging, escalation hooks)."""
    ts  = data.get("timestamp", datetime.now(timezone.utc).isoformat())
    pid = data.get("patient_id", "unknown")
    hr  = data.get("heart_rate")
    spo2 = data.get("spo2")

    print(f"[{ts}] {pid}  HR={hr}  SpO2={spo2}%")

    # ── escalation hooks (placeholders) ─────────────────────────────────────
    if hr and hr > 100:
        print(f"  ⚠  Elevated HR ({hr}) — trigger council review")

    if spo2 and spo2 < 94:
        print(f"  ⚠  Low SpO2 ({spo2}%) — trigger council review")


def run(max_ticks: int | None = None) -> None:
    """
    Main polling loop — mirrors openclaw's orchestrate_debate round structure.

    Runs indefinitely (or for max_ticks iterations if set).
    Each tick: fetch data → process → sleep for the dynamically computed interval.
    """
    print("💓 Heartbeat monitor started")
    tick = 0

    while max_ticks is None or tick < max_ticks:
        data = get_data()

        severity = data.get("severity", 1)
        stage    = data.get("stage", 2)
        interval = calculate_polling_freq(severity, stage)

        process(data)
        print(f"   next poll in {interval:.0f}s  (severity={severity}, stage={stage})\n")

        tick += 1
        if max_ticks is None or tick < max_ticks:
            time.sleep(interval)

    print("💓 Heartbeat monitor stopped")


if __name__ == "__main__":
    run()
