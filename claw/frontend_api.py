from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from query import get_latest_vitals
from anomaly import infer_anomaly_level


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VitalInformation(BaseModel):
    # Identity
    user_id:                  Optional[str]      = None
    date:                     Optional[date]     = None
    day_of_week:              Optional[str]      = None
    TimeStamp:                Optional[datetime] = None

    # Demographics
    age:                      Optional[int]      = None
    gender:                   Optional[str]      = None
    weight_kg:                Optional[float]    = None
    height_cm:                Optional[float]    = None
    fitness_level:            Optional[str]      = None
    primary_sport:            Optional[str]      = None

    # Recovery & strain
    recovery_score:           Optional[float]    = None
    day_strain:               Optional[float]    = None

    # Sleep
    sleep_hours:              Optional[float]    = None
    sleep_efficiency:         Optional[float]    = None
    sleep_performance:        Optional[float]    = None
    light_sleep_hours:        Optional[float]    = None
    rem_sleep_hours:          Optional[float]    = None
    deep_sleep_hours:         Optional[float]    = None
    wake_ups:                 Optional[int]      = None
    time_to_fall_asleep_min:  Optional[float]    = None

    # Vitals
    hrv:                      Optional[float]    = None
    resting_heart_rate:       Optional[float]    = None
    hrv_baseline:             Optional[int]      = None
    rhr_baseline:             Optional[int]      = None
    respiratory_rate:         Optional[float]    = None
    skin_temp_deviation:      Optional[float]    = None
    calories_burned:          Optional[float]    = None

    # Activity
    workout_completed:        Optional[int]      = None
    activity_type:            Optional[str]      = None
    activity_duration_min:    Optional[float]    = None
    activity_strain:          Optional[float]    = None
    avg_heart_rate:           Optional[float]    = None
    max_heart_rate:           Optional[float]    = None
    activity_calories:        Optional[float]    = None
    hr_zone_1_min:            Optional[float]    = None
    hr_zone_2_min:            Optional[float]    = None
    hr_zone_3_min:            Optional[float]    = None
    hr_zone_4_min:            Optional[float]    = None
    hr_zone_5_min:            Optional[float]    = None
    workout_time_of_day:      Optional[str]      = None

    # Derived — computed by anomaly.infer_anomaly_level, not stored in BQ
    anomaly_level:            Optional[int]      = None


@app.get("/vitals/{patient_id}", response_model=VitalInformation)
def vitals_to_frontend(patient_id: str) -> VitalInformation:
    data = get_latest_vitals(patient_id)
    data["anomaly_level"] = infer_anomaly_level(data)
    return VitalInformation(**data)