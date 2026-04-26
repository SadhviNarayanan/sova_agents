"""
query.py — BigQuery reads/writes for CareRelay patient data.

Tables (all in automaticbalancetransfer.sova):
  patientProfile  — static patient demographics + medical context
  vitals          — real-time vitals stream (patientId, TimeStamp, HeartRate, BloodPressure, Temperature)
  Whoop_Data      — wearable sensor history (used by anomaly model)
  agentSummary    — LLM council decision log
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta

from google.cloud import bigquery
from google.oauth2 import service_account

_PROJECT = "automaticbalancetransfer"
_DATASET = "sova"

_PROFILE_TABLE = f"`{_PROJECT}.{_DATASET}.patientProfile`"
_VITALS_TABLE  = f"`{_PROJECT}.{_DATASET}.vitals`"
_COLUMN_CACHE: dict[str, dict[str, str]] = {}

_RISK_TO_SEVERITY: dict[str, int] = {
    "low":      0,
    "medium":   1,
    "moderate": 1,
    "high":     2,
    "critical": 2,
}

_STAGE_THRESHOLDS = [2, 7, 14, 30, 60]  # days since discharge → stage 0-5


def _client() -> bigquery.Client:
    credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if credentials_json:
        credentials_info = json.loads(credentials_json)
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        return bigquery.Client(project=_PROJECT, credentials=credentials)
    return bigquery.Client(project=_PROJECT)


def _patient_id_job_config(patient_id: str) -> bigquery.QueryJobConfig:
    return bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("patient_id", "STRING", patient_id),
        ]
    )


def _columns_for_table(table_name: str) -> dict[str, str]:
    if table_name in _COLUMN_CACHE:
        return _COLUMN_CACHE[table_name]

    rows = list(_client().query(
        f"""
        SELECT column_name
        FROM `{_PROJECT}.{_DATASET}.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = @table_name
        """,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("table_name", "STRING", table_name),
            ]
        ),
    ))
    columns = {str(row["column_name"]).lower(): str(row["column_name"]) for row in rows}
    _COLUMN_CACHE[table_name] = columns
    return columns


def _column_for(table_name: str, *candidates: str) -> str:
    columns = _columns_for_table(table_name)
    for candidate in candidates:
        actual = columns.get(candidate.lower())
        if actual:
            return actual
    raise KeyError(
        f"None of {candidates} exist in {_PROJECT}.{_DATASET}.{table_name}. "
        f"Available columns: {sorted(columns.values())}"
    )


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
        WHERE patientId = @patient_id
        LIMIT 1
    """, job_config=_patient_id_job_config(patient_id)))
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
    patient_col = _column_for("vitals", "patientId", "PatientId", "PatientID", "patientID", "user_id")
    timestamp_col = _column_for("vitals", "TimeStamp", "Timestamp", "timestamp")

    rows = list(client.query(f"""
        SELECT `{timestamp_col}` AS TimeStamp FROM {_VITALS_TABLE}
        WHERE `{patient_col}` = @patient_id
        ORDER BY `{timestamp_col}` ASC
    """, job_config=_patient_id_job_config(patient_id)))
    if not rows:
        return 0

    anchor = _parse_dt(rows[0]["TimeStamp"])
    fixed  = 0

    for i, row in enumerate(rows):
        expected = anchor + timedelta(seconds=i * freq_seconds)
        actual   = _parse_dt(row["TimeStamp"])

        if abs((actual - expected).total_seconds()) < 1:
            continue

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("expected_ts", "TIMESTAMP", expected),
                bigquery.ScalarQueryParameter("patient_id", "STRING", patient_id),
                bigquery.ScalarQueryParameter("actual_ts", "TIMESTAMP", actual),
            ]
        )
        client.query(f"""
            UPDATE {_VITALS_TABLE}
            SET `{timestamp_col}` = @expected_ts
            WHERE `{patient_col}` = @patient_id
              AND `{timestamp_col}` = @actual_ts
        """, job_config=job_config).result()
        fixed += 1

    return fixed


def insert_vitals() -> None:
    """Seed patientProfile + vitals for the demo patient."""
    client = _client()

    profile = {
        "patientId": "startID",
        "Age": 68, "Gender": "Female",
        "Surgery": "Post-cardiac event", "DischargeDate": "2026-04-10",
        "RiskLevel": "high", "BloodPressure": "138/86", "HeartRate": 74,
        "Allergies": "None", "CurrentMedications": "Metoprolol, Aspirin",
        "EmergencyContactName": "Family Contact", "EmergencyContactPhone": "+15125550101",
    }

    try:
        client.query(f"""
            INSERT INTO {_PROFILE_TABLE}
              (patientId, Age, Gender, Surgery, DischargeDate, RiskLevel,
               BloodPressure, HeartRate, Allergies, CurrentMedications,
               EmergencyContactName, EmergencyContactPhone)
            VALUES (
              '{profile["patientId"]}', {profile["Age"]}, '{profile["Gender"]}',
              '{profile["Surgery"]}', DATE('{profile["DischargeDate"]}'), '{profile["RiskLevel"]}',
              '{profile["BloodPressure"]}', {profile["HeartRate"]}, '{profile["Allergies"]}',
              '{profile["CurrentMedications"]}', '{profile["EmergencyContactName"]}',
              '{profile["EmergencyContactPhone"]}'
            )
        """).result()
        print(f"Inserted profile for {profile['patientId']}")
    except Exception as e:
        print(f"Error inserting profile for {profile['patientId']}: {e}")

    now = datetime.now()
    # (offset_seconds, heart_rate, blood_pressure, temperature)
    # historical rows near rhr_baseline=74; final row is current abnormal
    vitals_rows = [
        (0,  73, "136/84", 37.1),
        (10, 75, "137/85", 37.0),
        (20, 74, "138/86", 37.0),
        (30, 99, "148/94", 38.4),
    ]

    pid = profile["patientId"]
    for offset_seconds, heart_rate, blood_pressure, temperature in vitals_rows:
        ts = now + timedelta(seconds=offset_seconds)
        try:
            client.query(f"""
                INSERT INTO {_VITALS_TABLE}
                  (patientId, TimeStamp, HeartRate, BloodPressure, Temperature)
                VALUES (
                  '{pid}', TIMESTAMP('{ts.isoformat()}'),
                  {heart_rate}, '{blood_pressure}', {temperature}
                )
            """).result()
            print(f"Inserted vitals for {pid} at {ts}")
        except Exception as e:
            print(f"Error inserting vitals for {pid} at {ts}: {e}")

def get_latest_vitals(patient_id: str) -> dict:
    """Fetch the most-recent vitals row for a patient."""
    patient_col = _column_for("vitals", "patientId", "PatientId", "PatientID", "patientID", "user_id")
    timestamp_col = _column_for("vitals", "TimeStamp", "Timestamp", "timestamp")
    rows = list(_client().query(f"""
        SELECT *
        FROM {_VITALS_TABLE}
        WHERE `{patient_col}` = @patient_id
        ORDER BY `{timestamp_col}` DESC
        LIMIT 1
    """, job_config=_patient_id_job_config(patient_id)))
    return dict(rows[0]) if rows else {}


if __name__ == "__main__":
    insert_vitals()
