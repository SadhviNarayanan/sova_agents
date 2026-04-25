"""Temporary one-shot script — delete once timestamps are clean."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from query import get_all_patients, get_patient_profile, fix_vitals_timestamps
from heartbeat import calculate_polling_freq

print("Syncing vitals timestamps for all patients ...\n")
for _p in get_all_patients():
    _pid      = _p["patientId"]
    _profile  = get_patient_profile(_pid)
    _interval = calculate_polling_freq(_profile["severity"], _profile["stage"])
    _fixed    = fix_vitals_timestamps(_pid, _interval)
    print(f"  {_pid}: {_fixed} row(s) corrected at freq={_interval:.0f}s")

print("\nTimestamp sync complete")
