from google.cloud import bigquery
from requests import get
from datetime import datetime

def fetch_profile(user_id):
    client = bigquery.Client(project="automaticbalancetransfer")

    query = f"""
    SELECT *
    FROM `automaticbalancetransfer.sova.Whoop_Data`
    WHERE user_id = '{user_id}'
    LIMIT 1
    """
    rows = list(client.query(query))
    return dict(rows[0]) if rows else {}


def get_data(user_id: str) -> dict:
    profile = fetch_profile(user_id)

    return {
        # identity
        "user_id": user_id,
        "date": profile.get("date"), # now.isoformat(),
        "day_of_week": profile.get("day_of_week"), # now.strftime("%A"),
        "age": profile.get("age"),
        "gender": profile.get("gender"),
        "weight_kg": profile.get("weight_kg"),
        "height_cm": profile.get("height_cm"),
        "fitness_level": profile.get("fitness_level"),
        "primary_sport": profile.get("primary_sport"),

        # profile
        "profile_score": profile.get("profile_score"),
        "hrv": profile.get("hrv"),
        "hrv_baseline": profile.get("hrv_baseline"),
        "resting_heart_rate": profile.get("rhr"),
        "rhr_baseline": profile.get("rhr_baseline"),
        "respiratory_rate": profile.get("resp_rate"),
        "skin_temp_deviation": profile.get("temp_dev"),

        # profile
        "sleep_hours": profile.get("sleep_hours"),
        "sleep_efficiency": profile.get("efficiency"),
        "sleep_performance": profile.get("performance"),
        "light_sleep_hours": profile.get("light"),
        "rem_sleep_hours": profile.get("rem"),
        "deep_sleep_hours": profile.get("deep"),
        "wake_ups": profile.get("wakeups"),
        "time_to_fall_asleep_min": profile.get("sleep_latency"),

        # activity
        "day_strain": profile.get("day_strain"),
        "calories_burned": profile.get("calories"),
        "workout_completed": profile.get("worked_out"),
        "activity_type": profile.get("type"),
        "activity_duration_min": profile.get("duration"),
        "activity_strain": profile.get("strain"),
        "avg_heart_rate": profile.get("avg_hr"),
        "max_heart_rate": profile.get("max_hr"),
        "activity_calories": profile.get("activity_calories"),
        "hr_zone_1_min": profile.get("zone1"),
        "hr_zone_2_min": profile.get("zone2"),
        "hr_zone_3_min": profile.get("zone3"),
        "hr_zone_4_min": profile.get("zone4"),
        "hr_zone_5_min": profile.get("zone5"),
        "workout_time_of_day": profile.get("time_of_day"),

        # backend-computed (you fill later)
        "severity": 1,
        "stage": 2,
        "anomaly_level": 0,
    }


print(get_data("USER_00001"))
