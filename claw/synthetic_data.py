"""
synthetic_data.py — placeholder data source for heartbeat polling.

get_data() is the single entry point. Swap the body for a real DB/API
call when the data layer is ready; heartbeat.py only ever calls this.
"""

from datetime import datetime, timezone


def get_data() -> dict:
    """Returns the latest patient snapshot. Placeholder — replace with real source."""
    return {
        "patient_id": "patient_001",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "heart_rate": 88,
        "spo2": 96,
        "blood_pressure": "128/82",
        "severity": 1,   # 0=low, 1=medium, 2=high  (drives polling freq)
        "stage": 2,      # 0–5 post-discharge stage   (drives polling freq)
        "symptoms": [],
    }
