from google.cloud import bigquery
from datetime import datetime, timedelta

_TABLE   = "`automaticbalancetransfer.sova.Whoop_Data`"
_PROJECT = "automaticbalancetransfer"


def _client() -> bigquery.Client:
    return bigquery.Client(project=_PROJECT)


def _parse_dt(value) -> datetime:
    dt = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


def fetch_all_rows(user_id: str) -> list[dict]:
    """All rows for a user, oldest first."""
    rows = list(_client().query(f"""
        SELECT * FROM {_TABLE}
        WHERE user_id = '{user_id}'
        ORDER BY date ASC
    """))
    return [dict(r) for r in rows]


def get_anchor(user_id: str) -> datetime | None:
    """Timestamp of the first (oldest) row — the fixed origin for tick arithmetic."""
    rows = list(_client().query(f"""
        SELECT date FROM {_TABLE}
        WHERE user_id = '{user_id}'
        ORDER BY date ASC
        LIMIT 1
    """))
    return _parse_dt(rows[0]["date"]) if rows else None


def fix_timestamps(user_id: str, freq_seconds: float) -> int:
    """
    Rewrite every row's timestamp so row N sits at anchor + N * freq_seconds.
    (user_id, date) is the unique key; drifted rows are corrected via DML UPDATE.
    Returns the number of rows that were changed.
    """
    rows = fetch_all_rows(user_id)
    if not rows or rows[0].get("date") is None:
        return 0

    anchor = _parse_dt(rows[0]["date"])
    client = _client()
    fixed  = 0

    for i, row in enumerate(rows):
        expected = anchor + timedelta(seconds=i * freq_seconds)
        actual   = _parse_dt(row["date"])

        if abs((actual - expected).total_seconds()) < 1:
            continue

        client.query(f"""
            UPDATE {_TABLE}
            SET date = '{expected.isoformat()}'
            WHERE user_id = '{user_id}' AND date = '{actual.isoformat()}'
        """).result()
        fixed += 1

    return fixed


def fetch_by_timestamp(user_id: str, ts: datetime) -> dict:
    """
    Fetch the Whoop row at exactly (user_id, ts).
    Returns an empty dict if no row exists at that timestamp.
    Field names are normalized to match what the anomaly model expects.
    """
    rows = list(_client().query(f"""
        SELECT * FROM {_TABLE}
        WHERE user_id = '{user_id}' AND date = '{ts.isoformat()}'
        LIMIT 1
    """))
    if not rows:
        return {}

    p = dict(rows[0])
    return {
        "user_id":                  user_id,
        "date":                     p.get("date"),
        "day_of_week":              p.get("day_of_week"),
        "age":                      p.get("age"),
        "gender":                   p.get("gender"),
        "weight_kg":                p.get("weight_kg"),
        "height_cm":                p.get("height_cm"),
        "fitness_level":            p.get("fitness_level"),
        "primary_sport":            p.get("primary_sport"),
        "profile_score":            p.get("profile_score"),
        "hrv":                      p.get("hrv"),
        "hrv_baseline":             p.get("hrv_baseline"),
        "resting_heart_rate":       p.get("rhr"),
        "rhr_baseline":             p.get("rhr_baseline"),
        "respiratory_rate":         p.get("resp_rate"),
        "skin_temp_deviation":      p.get("temp_dev"),
        "sleep_hours":              p.get("sleep_hours"),
        "sleep_efficiency":         p.get("efficiency"),
        "sleep_performance":        p.get("performance"),
        "light_sleep_hours":        p.get("light"),
        "rem_sleep_hours":          p.get("rem"),
        "deep_sleep_hours":         p.get("deep"),
        "wake_ups":                 p.get("wakeups"),
        "time_to_fall_asleep_min":  p.get("sleep_latency"),
        "day_strain":               p.get("day_strain"),
        "calories_burned":          p.get("calories"),
        "workout_completed":        p.get("worked_out"),
        "activity_type":            p.get("type"),
        "activity_duration_min":    p.get("duration"),
        "activity_strain":          p.get("strain"),
        "avg_heart_rate":           p.get("avg_hr"),
        "max_heart_rate":           p.get("max_hr"),
        "activity_calories":        p.get("activity_calories"),
        "hr_zone_1_min":            p.get("zone1"),
        "hr_zone_2_min":            p.get("zone2"),
        "hr_zone_3_min":            p.get("zone3"),
        "hr_zone_4_min":            p.get("zone4"),
        "hr_zone_5_min":            p.get("zone5"),
        "workout_time_of_day":      p.get("time_of_day"),
    }
