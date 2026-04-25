"""
synthetic_data.py — placeholder data source for heartbeat polling.

get_data() returns one patient snapshot using the exact field names from
whoop_fitness_dataset_100k.csv. Swap the body for a real DB/CSV read when ready.
"""

from datetime import datetime, timezone


def get_data() -> dict:
    """Returns the latest WHOOP patient snapshot. Placeholder — replace with real source."""
    return {
        # identity
        "user_id":                  "user_001",
        "date":                     datetime.now(timezone.utc).isoformat(),
        "day_of_week":              "Friday",
        "age":                      34,
        "gender":                   "female",
        "weight_kg":                68.0,
        "height_cm":                165.0,
        "fitness_level":            "intermediate",
        "primary_sport":            "running",

        # recovery
        "recovery_score":           58,     # 0–100; <34 red, 34–66 yellow, 67+ green
        "hrv":                      52.0,   # ms
        "hrv_baseline":             60.0,   # ms — patient's personal baseline
        "resting_heart_rate":       62,     # bpm
        "rhr_baseline":             58,     # bpm — patient's personal baseline
        "respiratory_rate":         15.2,   # breaths/min
        "skin_temp_deviation":      0.1,    # °C from baseline

        # sleep
        "sleep_hours":              7.2,
        "sleep_efficiency":         88.0,   # %
        "sleep_performance":        82.0,   # %
        "light_sleep_hours":        3.1,
        "rem_sleep_hours":          1.8,
        "deep_sleep_hours":         2.3,
        "wake_ups":                 2,
        "time_to_fall_asleep_min":  11.0,

        # strain / activity
        "day_strain":               10.4,   # 0–21 WHOOP strain scale
        "calories_burned":          2340,
        "workout_completed":        True,
        "activity_type":            "run",
        "activity_duration_min":    45,
        "activity_strain":          8.2,
        "avg_heart_rate":           142,
        "max_heart_rate":           171,
        "activity_calories":        480,
        "hr_zone_1_min":            5,
        "hr_zone_2_min":            10,
        "hr_zone_3_min":            15,
        "hr_zone_4_min":            12,
        "hr_zone_5_min":            3,
        "workout_time_of_day":      "morning",

        # received from backend — heartbeat does not calculate these
        "severity":                 1,      # 0=low, 1=medium, 2=high
        "stage":                    2,      # 0–5 post-discharge stage
        "anomaly_level":            0,      # 0=normal, 1=log, 2=text, 3=call, 4=911
    }
