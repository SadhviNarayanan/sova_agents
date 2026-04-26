"""
query.py — BigQuery reads/writes for CareRelay patient data.

Tables (all in automaticbalancetransfer.sova):
  patientProfile  — static patient demographics + medical context
  vitals          — real-time vitals stream (patientId, TimeStamp, HeartRate, BloodPressure, Temperature)
  Whoop_Data      — wearable sensor history (used by anomaly model)
  agentSummary    — LLM council decision log
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

from google.cloud import bigquery

_PROJECT = "automaticbalancetransfer"
_DATASET = "sova"

_PROFILE_TABLE = f"`{_PROJECT}.{_DATASET}.patientProfile`"
_VITALS_TABLE  = f"`{_PROJECT}.{_DATASET}.vitals`"

_RISK_TO_SEVERITY: dict[str, int] = {
    "low":      0,
    "medium":   1,
    "moderate": 1,
    "high":     2,
    "critical": 2,
}

_STAGE_THRESHOLDS = [2, 7, 14, 30, 60]  # days since discharge → stage 0-5


def _client() -> bigquery.Client:
    return bigquery.Client(project=_PROJECT)


def _parse_dt(value) -> datetime:
    dt = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


def _risk_to_severity(risk_level: str | None) -> int:
    if not risk_level:
        return 1
    return _RISK_TO_SEVERITY.get(risk_level.lower().strip(), 1)


def _discharge_to_stage(discharge_date: date | str | None) -> int:
    if discharge_date is None:
        return 2
    if isinstance(discharge_date, str):
        discharge_date = date.fromisoformat(discharge_date)
    days = (date.today() - discharge_date).days
    for stage, threshold in enumerate(_STAGE_THRESHOLDS):
        if days < threshold:
            return stage
    return 5


def get_patient_profile(patient_id: str) -> dict:
    """
    Fetch a patient's row from patientProfile by patientId.
    Returns a dict with all schema fields plus derived `severity` and `stage`.
    Returns an empty dict if not found.
    """
    rows = list(_client().query(f"""
        SELECT *
        FROM {_PROFILE_TABLE}
        WHERE patientId = '{patient_id}'
        LIMIT 1
    """))
    if not rows:
        return {}

    row = dict(rows[0])
    row["severity"] = _risk_to_severity(row.get("RiskLevel"))
    row["stage"]    = _discharge_to_stage(row.get("DischargeDate"))
    return row


def get_all_patients() -> list[dict]:
    """Return all patientProfile rows ordered by patientId."""
    rows = list(_client().query(f"""
        SELECT * FROM {_PROFILE_TABLE}
        ORDER BY patientId ASC
    """))
    return [dict(r) for r in rows]


def fix_vitals_timestamps(patient_id: str, freq_seconds: float) -> int:
    """
    Rewrite vitals TimeStamps so row N sits at anchor + N * freq_seconds,
    where anchor is the earliest existing TimeStamp for this patient.
    Uses (patientId, TimeStamp) as the unique key for each UPDATE.
    Returns the number of rows corrected.
    """
    client = _client()

    rows = list(client.query(f"""
        SELECT TimeStamp FROM {_VITALS_TABLE}
        WHERE patientId = '{patient_id}'
        ORDER BY TimeStamp ASC
    """))
    if not rows:
        return 0

    anchor = _parse_dt(rows[0]["TimeStamp"])
    fixed  = 0

    for i, row in enumerate(rows):
        expected = anchor + timedelta(seconds=i * freq_seconds)
        actual   = _parse_dt(row["TimeStamp"])

        if abs((actual - expected).total_seconds()) < 1:
            continue

        client.query(f"""
            UPDATE {_VITALS_TABLE}
            SET TimeStamp = TIMESTAMP('{expected.isoformat()}')
            WHERE patientId = '{patient_id}'
              AND TimeStamp = TIMESTAMP('{actual.isoformat()}')
        """).result()
        fixed += 1

    return fixed


def insert_vitals() -> None:
    """Insert a new row into vitals."""
    client = _client()
    patient_id = "startID"  
    timestamp = datetime.now()
    ts = timestamp.isoformat()
    client.query(f"""
                    INSERT INTO {_PROFILE_TABLE} (patientId, Name, Age, Conditions, RiskLevel, DischargeDate)
                    VALUES ('{patient_id}', 'John Doe', 65, 'Hypertension, Diabetes', 'High', TIMESTAMP('{ts}'))
                """).result()
    heart_rate, blood_pressure, temperature = 80, "120/80", 98.6
    for i in range(3):  # retry up to 3 times in case of transient errors
        try:
            if i < 2:

                client.query(f"""
                    INSERT INTO {_VITALS_TABLE} (patientId, TimeStamp, HeartRate, BloodPressure, Temperature)
                    VALUES ('{patient_id}', TIMESTAMP('{ts}'), {heart_rate}, '{blood_pressure}', {temperature})
                """).result()
            else:
                heart_rate = "150/92"
                client.query(f"""
                    INSERT INTO {_VITALS_TABLE} (patientId, TimeStamp, HeartRate, BloodPressure, Temperature)
                    VALUES ('{patient_id}', TIMESTAMP('{ts}'), {heart_rate}, '{blood_pressure}', {temperature})
                """).result()
            ts += 30 # increment timestamp for next row to avoid duplicates
            break                   
        except Exception as e:
            print(f"Error occurred while inserting vitals for patient {patient_id}: {e}")
            time.sleep(1)  # Wait before retrying   

def get_latest_vitals(patient_id: str) -> dict:
    """Fetch the most-recent vitals row for a patient."""
    rows = list(_client().query(f"""
        SELECT *
        FROM {_VITALS_TABLE}
        WHERE patientId = '{patient_id}'
        ORDER BY TimeStamp DESC
        LIMIT 1
    """))
    return dict(rows[0]) if rows else {}


if __name__ == "__main__":
    insert_vitals()