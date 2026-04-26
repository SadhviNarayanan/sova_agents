from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import base64
import io
import json
import math
import os
import sys
import asyncio
import threading
import time
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date, timedelta, timezone

from carerelay_backend.agents import AGENT_PROMPTS
import anthropic  # For Claude API integration

def load_local_dotenv(path: str = ".env") -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(path)
        return
    except Exception:
        pass

    env_path = os.path.join(os.getcwd(), path)
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_dotenv()

# Add parent directory to path to import agentic_convo and langgraph_council
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agentic_convo import MedicalCouncilOrchestrator
from langgraph_council import AGENT_CONFIGS, LangGraphMedicalCouncil
from claw.call_twilio import call_caregiver
from claw.query import get_latest_vitals, get_patient_profile

app = FastAPI(title="CareRelay API", description="AI-powered post-hospital care monitoring system")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Pydantic models
class PatientVitals(BaseModel):
    heart_rate: float
    spo2: float
    blood_pressure: Optional[str] = None
    timestamp: datetime

class PatientState(BaseModel):
    patient_id: str
    vitals_history: List[PatientVitals]
    symptoms: List[str]
    medications: List[str]
    last_checkin: Optional[datetime] = None

class CouncilDecision(BaseModel):
    decision: str
    doctor_report: str
    confidence: float
    actions: List[str]
    escalation_level: int  # 1-5, 5 being highest

class CheckinResponse(BaseModel):
    transcript: str
    patient_state: PatientState
    needs_council: bool

class Vitals(BaseModel):
    HeartRate: Optional[int] = None
    BloodPressure: Optional[str] = None
    Temperature: Optional[float] = None
    TimeStamp: Optional[datetime] = None

class StatusVitals(BaseModel):
    heartRate: Optional[int] = None
    hrv: Optional[int] = None
    spo2: Optional[int] = None
    sleepHours: Optional[float] = None
    bloodPressure: Optional[str] = None
    temperature: Optional[float] = None
    timestamp: Optional[str] = None

class StatusEscalation(BaseModel):
    caregiverCallTriggered: bool = False
    reason: Optional[str] = None

class StatusDeliberation(BaseModel):
    triggered: bool = False
    status: str = "idle"
    streamUrl: Optional[str] = None

class StatusTrajectoryPoint(BaseModel):
    label: str
    hoursFromNow: float
    riskLevel: str
    riskScore: int

class StatusNotification(BaseModel):
    type: str
    title: str
    message: str
    requiresResponse: bool = False
    actions: List[str] = []

class PatientStatusResponse(BaseModel):
    patientId: str
    vitals: StatusVitals
    anomalyLevel: int
    riskLevel: str
    recommendedAction: str
    escalation: StatusEscalation
    deliberation: StatusDeliberation
    trajectory: List[StatusTrajectoryPoint] = []
    notification: Optional[StatusNotification] = None

class SimulationModeRequest(BaseModel):
    mode: str

class SimulationModeResponse(BaseModel):
    patientId: str
    mode: str
    statusUrl: str

class SpecialistResponse(BaseModel):
    id: str
    name: str
    specialty: str

class SpecialistCallStartRequest(BaseModel):
    specialistId: str
    clientSessionId: str
    patientContext: Optional[Dict[str, Any]] = None

class SpecialistCallStartResponse(BaseModel):
    sessionId: str
    websocketUrl: str

class AnalyzeRequest(BaseModel):
    # Patient profile
    patientId: str
    Age: int
    Gender: str
    DateOfBirth: Optional[date] = None
    Address: Optional[str] = None
    Surgery: str
    DischargeDate: date
    RiskLevel: str                        # Low / Medium / High
    BloodPressure: Optional[str] = None  # baseline
    HeartRate: Optional[int] = None      # baseline
    Allergies: Optional[str] = None
    CurrentMedications: Optional[str] = None
    DoctorPhoneNumber: Optional[str] = None
    EmergencyContactName: Optional[str] = None
    EmergencyContactPhone: Optional[str] = None
    # Derived fields
    severity: Optional[int] = None       # 0=low, 1=medium, 2=high
    stage: Optional[int] = None          # 0-5, days-since-discharge band
    # Live vitals
    vitals: Optional[Vitals] = None
    # ML heartbeat output
    anomaly_level: Optional[int] = None  # 0-4, from ML model
    interval: Optional[int] = None       # current polling freq in seconds
    # Routing
    webhook_url: Optional[str] = None

    @property
    def patient_id(self) -> str:
        return self.patientId

# In-memory storage (in production, use database)
patients: Dict[str, PatientState] = {}

# Per-patient in-memory SSE state.
# History lets a frontend connect a few seconds late and still see the full debate.
debate_histories: Dict[str, List[dict]] = {}
debate_subscribers: Dict[str, List[asyncio.Queue]] = {}
debate_status: Dict[str, str] = {}
high_risk_episodes: Dict[str, dict] = {}
simulation_modes: Dict[str, str] = {}
specialist_call_sessions: Dict[str, dict] = {}
high_risk_lock = threading.Lock()
HIGH_RISK_COOLDOWN = timedelta(minutes=15)
DEFAULT_SIMULATION_PATIENT_ID = "default"
executor = ThreadPoolExecutor()
USE_BIGQUERY_VITALS = os.getenv("SOVA_USE_BIGQUERY_VITALS", "false").lower() in {"1", "true", "yes"}


def masked_env_value(name: str) -> str:
    value = os.getenv(name)
    if not value:
        return "unset"
    if len(value) <= 8:
        return f"set(len={len(value)})"
    return f"set(len={len(value)}, value={value[:4]}...{value[-4:]})"


@app.on_event("startup")
async def log_elevenlabs_env_status():
    print(
        "ElevenLabs env status: "
        f"ELEVENLABS_API_KEY={masked_env_value('ELEVENLABS_API_KEY')}, "
        f"ELEVENLABS_AGENT_ID={masked_env_value('ELEVENLABS_AGENT_ID')}"
    )


@app.get("/")
async def root():
    return {"message": "CareRelay API is running"}


@app.get("/debug/env-check")
async def env_check():
    return {
        "elevenlabsApiKeyPresent": bool(os.getenv("ELEVENLABS_API_KEY")),
        "elevenlabsAgentIdPresent": bool(os.getenv("ELEVENLABS_AGENT_ID")),
    }


@app.get("/v1/specialists", response_model=List[SpecialistResponse])
async def specialists():
    return [
        SpecialistResponse(id=key, name=config["name"], specialty=config["specialty"])
        for key, config in AGENT_CONFIGS.items()
    ]


def specialist_by_id(specialist_id: str) -> dict:
    config = AGENT_CONFIGS.get(specialist_id)
    if not config:
        raise HTTPException(status_code=404, detail="Unknown specialist")
    return {
        "id": specialist_id,
        "name": config["name"],
        "specialty": config["specialty"],
        "system_prompt": config.get("system_prompt", ""),
    }

SEVERITY_LABELS = {0: "low", 1: "medium", 2: "high"}
STAGE_LABELS = {
    0: "day of discharge",
    1: "days 1-3 post-discharge",
    2: "days 4-7 post-discharge",
    3: "week 2 post-discharge",
    4: "weeks 3-4 post-discharge",
    5: "month 2+ post-discharge",
}
ANOMALY_LABELS = {0: "normal", 1: "mild", 2: "moderate", 3: "significant", 4: "critical"}
SIMULATION_MODE_ALIASES = {
    "ok": "low",
    "normal": "low",
    "low": "low",
    "med": "medium",
    "medium": "medium",
    "moderate": "medium",
    "high": "high",
    "critical": "high",
}

def build_patient_dict(request: AnalyzeRequest) -> dict:
    days_since_discharge = (date.today() - request.DischargeDate).days if request.DischargeDate else None
    return {
        "patient_id": request.patientId,
        "age": request.Age,
        "gender": request.Gender,
        "surgery": request.Surgery,
        "discharge_date": request.DischargeDate.isoformat() if request.DischargeDate else None,
        "days_since_discharge": days_since_discharge,
        "recovery_stage": STAGE_LABELS.get(request.stage, f"stage {request.stage}") if request.stage is not None else None,
        "risk_level": request.RiskLevel,
        "severity": SEVERITY_LABELS.get(request.severity, "unknown") if request.severity is not None else None,
        "baseline_heart_rate": request.HeartRate,
        "baseline_blood_pressure": request.BloodPressure,
        "allergies": request.Allergies,
        "current_medications": request.CurrentMedications,
        "vitals": {
            "heart_rate": request.vitals.HeartRate if request.vitals else None,
            "blood_pressure": request.vitals.BloodPressure if request.vitals else None,
            "temperature": request.vitals.Temperature if request.vitals else None,
            "timestamp": request.vitals.TimeStamp.isoformat() if request.vitals and request.vitals.TimeStamp else None,
        } if request.vitals else None,
        "anomaly_level": request.anomaly_level,
        "anomaly_severity": ANOMALY_LABELS.get(request.anomaly_level, "unknown") if request.anomaly_level is not None else None,
        "polling_interval_seconds": request.interval,
        "emergency_contact": {
            "name": request.EmergencyContactName,
            "phone": request.EmergencyContactPhone,
        },
    }


def risk_level_for_anomaly(anomaly_level: int) -> str:
    if anomaly_level >= 3:
        return "high"
    if anomaly_level == 2:
        return "medium"
    return "low"


def recommended_action_for_risk(risk_level: str) -> str:
    if risk_level == "high":
        return "Sova is contacting your caregiver"
    if risk_level == "medium":
        return "Talk with an AI specialist"
    return "Continue monitoring"


def _int_or_none(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _float_or_none(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _string_or_none(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _value(raw: dict, *keys: str):
    for key in keys:
        if key in raw and raw[key] is not None:
            return raw[key]
    return None


def normalized_simulation_mode(mode: str) -> str:
    normalized = SIMULATION_MODE_ALIASES.get(mode.strip().lower())
    if not normalized:
        raise HTTPException(status_code=400, detail="Mode must be one of: low, med, medium, high")
    return normalized


def simulated_vitals_row(patient_id: str, mode: str = "low") -> dict:
    """
    Deterministic live demo vitals.
    This keeps the KMP dashboard useful even when BigQuery/Render credentials are unavailable.
    Set SOVA_USE_BIGQUERY_VITALS=true to use database reads again.
    """
    now = datetime.now(timezone.utc)
    seed = sum(ord(ch) for ch in patient_id)
    t = now.timestamp() / 2.0
    wave = math.sin(t + seed)
    slow_wave = math.sin((t / 3.0) + seed)
    oxygen_wave = math.sin((t / 2.0) + seed)

    if mode == "high":
        heart_rate = round(126 + wave * 6)
        hrv = round(25 + slow_wave * 4)
        spo2 = round(88 + oxygen_wave)
        sleep_hours = 4.3
        systolic = round(164 + wave * 8)
        diastolic = round(102 + slow_wave * 5)
        temperature = round(102.1 + slow_wave * 0.4, 1)
    elif mode == "medium":
        heart_rate = round(106 + wave * 4)
        hrv = round(42 + slow_wave * 4)
        spo2 = round(94 + max(-1, min(1, oxygen_wave)))
        sleep_hours = 5.8
        systolic = round(142 + wave * 6)
        diastolic = round(90 + slow_wave * 4)
        temperature = round(100.5 + slow_wave * 0.2, 1)
    else:
        heart_rate = round(76 + wave * 5 + ((seed % 5) - 2))
        hrv = round(60 + slow_wave * 6)
        spo2 = round(98 + max(-1, min(1, oxygen_wave)))
        sleep_hours = 7.3
        systolic = round(118 + wave * 5)
        diastolic = round(76 + slow_wave * 3)
        temperature = round(98.5 + slow_wave * 0.2, 1)

    return {
        "user_id": patient_id,
        "TimeStamp": now.isoformat(),
        "resting_heart_rate": heart_rate,
        "rhr_baseline": 76,
        "hrv": hrv,
        "hrv_baseline": 60,
        "spo2": spo2,
        "sleep_hours": sleep_hours,
        "sleep_performance": round((sleep_hours / 8.0) * 100),
        "BloodPressure": f"{systolic}/{diastolic}",
        "Temperature": temperature,
        "simulation_mode": mode,
        "simulated": True,
    }


def simulated_trajectory(mode: str, wave_seed: str) -> List[StatusTrajectoryPoint]:
    now = datetime.now(timezone.utc)
    seed = sum(ord(ch) for ch in wave_seed)
    drift = math.sin((now.timestamp() / 12.0) + seed)

    if mode == "high":
        points = [
            ("-4h", -4.0, "medium", 52 + round(drift * 2)),
            ("-2h", -2.0, "medium", 63 + round(drift * 2)),
            ("Now", 0.0, "high", 84 + round(drift * 3)),
            ("+2h", 2.0, "high", 92 + round(drift * 2)),
            ("+4h", 4.0, "high", 95 + round(drift * 2)),
            ("+6h", 6.0, "high", 97 + round(drift * 2)),
        ]
    elif mode == "medium":
        points = [
            ("-4h", -4.0, "low", 22 + round(drift * 2)),
            ("-2h", -2.0, "medium", 38 + round(drift * 2)),
            ("Now", 0.0, "medium", 47 + round(drift * 3)),
            ("+2h", 2.0, "medium", 58 + round(drift * 2)),
            ("+4h", 4.0, "medium", 63 + round(drift * 2)),
            ("+6h", 6.0, "high", 69 + round(drift * 3)),
        ]
    else:
        points = [
            ("-4h", -4.0, "low", 18 + round(drift * 2)),
            ("-2h", -2.0, "low", 16 + round(drift * 2)),
            ("Now", 0.0, "low", 14 + round(drift * 2)),
            ("+2h", 2.0, "low", 11 + round(drift * 2)),
            ("+4h", 4.0, "low", 10 + round(drift * 2)),
            ("+6h", 6.0, "low", 8 + round(drift * 2)),
        ]

    return [
        StatusTrajectoryPoint(
            label=label,
            hoursFromNow=hours,
            riskLevel=risk,
            riskScore=max(0, min(100, score)),
        )
        for label, hours, risk, score in points
    ]


def status_vitals_from_row(patient_id: str, raw: dict) -> StatusVitals:
    return StatusVitals(
        heartRate=_int_or_none(_value(raw, "heartRate", "HeartRate", "resting_heart_rate", "rhr")),
        hrv=_int_or_none(_value(raw, "hrv", "HRV")),
        spo2=_int_or_none(_value(raw, "spo2", "SpO2", "SPO2", "oxygen", "Oxygen")),
        sleepHours=_float_or_none(_value(raw, "sleepHours", "sleep_hours", "SleepHours")),
        bloodPressure=_string_or_none(_value(raw, "bloodPressure", "BloodPressure", "blood_pressure")),
        temperature=_float_or_none(_value(raw, "temperature", "Temperature")),
        timestamp=_string_or_none(_value(raw, "timestamp", "TimeStamp", "Timestamp", "date")),
    )


def anomaly_snapshot(patient_id: str, raw: dict, vitals: StatusVitals) -> dict:
    snapshot = dict(raw)
    snapshot.setdefault("user_id", patient_id)
    snapshot.setdefault("date", vitals.timestamp)
    if vitals.heartRate is not None:
        snapshot.setdefault("resting_heart_rate", vitals.heartRate)
        snapshot.setdefault("rhr_baseline", _int_or_none(_value(raw, "rhr_baseline", "RhrBaseline")) or vitals.heartRate)
    if vitals.hrv is not None:
        snapshot.setdefault("hrv", vitals.hrv)
        snapshot.setdefault("hrv_baseline", _int_or_none(_value(raw, "hrv_baseline", "HrvBaseline")) or vitals.hrv)
    if vitals.sleepHours is not None:
        snapshot.setdefault("sleep_performance", _float_or_none(_value(raw, "sleep_performance", "performance")) or 75)
    return snapshot


def simple_vitals_anomaly(vitals: StatusVitals) -> int:
    level = 0
    heart_rate = vitals.heartRate
    if heart_rate is not None:
        if heart_rate >= 130 or heart_rate < 40:
            level = max(level, 4)
        elif heart_rate >= 120 or heart_rate < 45:
            level = max(level, 3)
        elif heart_rate >= 105 or heart_rate < 50:
            level = max(level, 2)

    temperature = vitals.temperature
    if temperature is not None:
        if temperature >= 103:
            level = max(level, 4)
        elif temperature >= 101.5:
            level = max(level, 3)
        elif temperature >= 100.4:
            level = max(level, 2)

    pressure = vitals.bloodPressure
    if pressure:
        parts = pressure.replace(" ", "").split("/")
        if len(parts) == 2:
            systolic = _int_or_none(parts[0])
            diastolic = _int_or_none(parts[1])
            if systolic is not None and diastolic is not None:
                if systolic >= 180 or diastolic >= 120:
                    level = max(level, 4)
                elif systolic >= 160 or diastolic >= 100:
                    level = max(level, 3)
                elif systolic >= 140 or diastolic >= 90:
                    level = max(level, 2)
    if vitals.spo2 is not None:
        if vitals.spo2 < 90:
            level = max(level, 4)
        elif vitals.spo2 < 92:
            level = max(level, 3)
        elif vitals.spo2 < 95:
            level = max(level, 2)
    return level


def infer_patient_anomaly(patient_id: str, raw: dict, vitals: StatusVitals) -> int:
    model_level = 0
    try:
        from claw.anomaly import infer_anomaly_level
        model_level = infer_anomaly_level(anomaly_snapshot(patient_id, raw, vitals))
    except Exception as exc:
        print(f"Unable to run ML anomaly scorer for patientId={patient_id}; using vitals rules only. {exc}")
    return max(model_level, simple_vitals_anomaly(vitals))


def parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _date_value(value, fallback: date) -> date:
    if isinstance(value, date):
        return value
    if value:
        return date.fromisoformat(str(value))
    return fallback


def analyze_request_for_status(patient_id: str, profile: dict, vitals: StatusVitals, anomaly_level: int, risk_level: str) -> AnalyzeRequest:
    today = date.today()
    discharge = _date_value(profile.get("DischargeDate"), today)
    dob = profile.get("DateOfBirth")
    return AnalyzeRequest(
        patientId=patient_id,
        Age=_int_or_none(profile.get("Age")) or 0,
        Gender=str(profile.get("Gender") or "Unknown"),
        DateOfBirth=_date_value(dob, today) if dob else None,
        Address=_string_or_none(profile.get("Address")),
        Surgery=str(profile.get("Surgery") or "Post-discharge recovery"),
        DischargeDate=discharge,
        RiskLevel=risk_level.capitalize(),
        BloodPressure=vitals.bloodPressure,
        HeartRate=vitals.heartRate,
        Allergies=_string_or_none(profile.get("Allergies")) or "None",
        CurrentMedications=_string_or_none(profile.get("CurrentMedications")) or "None",
        DoctorPhoneNumber=_string_or_none(profile.get("DoctorPhoneNumber")),
        EmergencyContactName=_string_or_none(profile.get("EmergencyContactName")),
        EmergencyContactPhone=_string_or_none(profile.get("EmergencyContactPhone")),
        severity=2 if risk_level == "high" else 1 if risk_level == "medium" else 0,
        stage=_int_or_none(profile.get("stage")),
        anomaly_level=anomaly_level,
        interval=30 if risk_level == "high" else 60 if risk_level == "medium" else 300,
        vitals=Vitals(
            HeartRate=vitals.heartRate,
            BloodPressure=vitals.bloodPressure,
            Temperature=vitals.temperature,
            TimeStamp=parse_timestamp(vitals.timestamp),
        ),
    )


def start_debate_session(patient_id: str, request: AnalyzeRequest, loop: asyncio.AbstractEventLoop) -> dict:
    debate_histories[patient_id] = []
    debate_subscribers[patient_id] = debate_subscribers.get(patient_id, [])
    debate_status[patient_id] = "running"

    def publish(event: dict):
        payload = dict(event)
        payload.setdefault("patient_id", patient_id)
        if payload.get("type") == "done":
            payload.setdefault("event", "done")
            debate_status[patient_id] = "done"
        elif payload.get("type") == "error":
            payload.setdefault("event", "error")
            debate_status[patient_id] = "error"

        debate_histories.setdefault(patient_id, []).append(payload)
        for subscriber in list(debate_subscribers.get(patient_id, [])):
            subscriber.put_nowait(payload)

    def on_event(event):
        loop.call_soon_threadsafe(publish, event)

    publish({
        "type": "started",
        "event": "started",
        "patient_id": patient_id,
        "message": "Sova specialist council is joining.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    def run():
        try:
            council = LangGraphMedicalCouncil(max_utterances=8, event_callback=on_event)
            council.orchestrate_debate(build_patient_dict(request), webhook_url=request.webhook_url)
        except Exception as exc:
            message = str(exc)
            if "GRAPH_RECURSION_LIMIT" in message or "recursion limit" in message.lower():
                message = "The specialist council needed more time. Please retry."
            on_event({
                "type": "error",
                "event": "error",
                "patient_id": patient_id,
                "message": message,
            })

    loop.run_in_executor(executor, run)
    return {
        "status": "debate started",
        "patient_id": patient_id,
        "stream_url": f"/stream/{patient_id}",
        "note": "Final decision will be POSTed to webhook_url when complete"
    }


def current_deliberation(patient_id: str) -> StatusDeliberation:
    status = debate_status.get(patient_id)
    has_history = patient_id in debate_histories
    if status or has_history:
        return StatusDeliberation(
            triggered=True,
            status=status or "running",
            streamUrl=f"/stream/{patient_id}",
        )
    return StatusDeliberation()


def clear_deliberation_state(patient_id: str):
    debate_histories.pop(patient_id, None)
    debate_status.pop(patient_id, None)
    for subscriber in list(debate_subscribers.get(patient_id, [])):
        subscriber.put_nowait({
            "type": "done",
            "event": "done",
            "patient_id": patient_id,
        })
    debate_subscribers.pop(patient_id, None)


def clear_all_deliberation_state():
    for patient_id in list(debate_histories.keys() | debate_status.keys() | debate_subscribers.keys()):
        clear_deliberation_state(patient_id)


def safe_call_caregiver(payload: dict):
    try:
        call_caregiver(payload)
    except Exception as exc:
        print(f"Caregiver call failed for patientId={payload.get('patientId') or payload.get('patient_id')}: {exc}")


def trigger_high_risk_once(patient_id: str, request: AnalyzeRequest, raw_vitals: dict, loop: asyncio.AbstractEventLoop) -> StatusEscalation:
    now = datetime.now(timezone.utc)
    with high_risk_lock:
        episode = high_risk_episodes.get(patient_id)
        if episode and now - episode["started_at"] < HIGH_RISK_COOLDOWN:
            if not current_deliberation(patient_id).triggered:
                start_debate_session(patient_id, request, loop)
            return StatusEscalation(
                caregiverCallTriggered=True,
                reason="Caregiver escalation already started for this high-risk episode.",
            )
        high_risk_episodes[patient_id] = {"started_at": now}

    caregiver_payload = dict(raw_vitals)
    caregiver_payload.setdefault("patientId", patient_id)
    caregiver_payload.setdefault("patient_id", patient_id)
    loop.run_in_executor(executor, safe_call_caregiver, caregiver_payload)
    start_debate_session(patient_id, request, loop)
    return StatusEscalation(
        caregiverCallTriggered=True,
        reason="High risk detected. Sova started caregiver escalation.",
    )


def notification_for_status(risk_level: str, anomaly_level: int, escalation: StatusEscalation) -> Optional[StatusNotification]:
    if escalation.caregiverCallTriggered:
        return StatusNotification(
            type="caregiver_escalation",
            title="Sova is contacting your caregiver",
            message=escalation.reason or "A high-risk change was detected. We are notifying your caregiver now.",
            requiresResponse=False,
        )
    if risk_level in {"medium", "high"} or anomaly_level >= 2:
        return StatusNotification(
            type="anomaly_check",
            title="Is everything ok?",
            message="Sova noticed a change in your health signals.",
            requiresResponse=True,
            actions=["Yes", "No"],
        )
    return None


@app.post("/v1/patients/{patient_id}/simulation", response_model=SimulationModeResponse)
async def set_patient_simulation(patient_id: str, request: SimulationModeRequest):
    mode = normalized_simulation_mode(request.mode)
    simulation_key = DEFAULT_SIMULATION_PATIENT_ID if patient_id in {"*", "all", "default"} else patient_id
    simulation_modes[simulation_key] = mode
    with high_risk_lock:
        if simulation_key == DEFAULT_SIMULATION_PATIENT_ID:
            high_risk_episodes.clear()
            clear_all_deliberation_state()
        else:
            high_risk_episodes.pop(patient_id, None)
            clear_deliberation_state(patient_id)
    return SimulationModeResponse(
        patientId=simulation_key,
        mode=mode,
        statusUrl=f"/v1/patients/{simulation_key}/status",
    )


@app.get("/v1/patients/{patient_id}/status", response_model=PatientStatusResponse)
async def patient_status(patient_id: str):
    raw_vitals = {}
    profile = {}

    if USE_BIGQUERY_VITALS:
        try:
            raw_vitals = get_latest_vitals(patient_id)
        except Exception as exc:
            print(f"Unable to read latest vitals from BigQuery for patientId={patient_id}; using simulated vitals. {exc}")

        try:
            profile = get_patient_profile(patient_id)
        except Exception as exc:
            print(f"Unable to read patient profile from BigQuery for patientId={patient_id}: {exc}")

    if not raw_vitals:
        mode = simulation_modes.get(patient_id, simulation_modes.get(DEFAULT_SIMULATION_PATIENT_ID, "low"))
        raw_vitals = simulated_vitals_row(patient_id, mode)
    else:
        mode = risk_level_for_anomaly(simple_vitals_anomaly(status_vitals_from_row(patient_id, raw_vitals)))

    vitals = status_vitals_from_row(patient_id, raw_vitals)
    anomaly_level = infer_patient_anomaly(patient_id, raw_vitals, vitals)
    risk_level = risk_level_for_anomaly(anomaly_level)
    action = recommended_action_for_risk(risk_level)
    escalation = StatusEscalation()

    if risk_level == "high":
        request = analyze_request_for_status(patient_id, profile, vitals, anomaly_level, risk_level)
        escalation = trigger_high_risk_once(patient_id, request, raw_vitals, asyncio.get_event_loop())
    else:
        with high_risk_lock:
            high_risk_episodes.pop(patient_id, None)

    return PatientStatusResponse(
        patientId=patient_id,
        vitals=vitals,
        anomalyLevel=anomaly_level,
        riskLevel=risk_level,
        recommendedAction=action,
        escalation=escalation,
        deliberation=current_deliberation(patient_id),
        trajectory=simulated_trajectory(mode, patient_id),
        notification=notification_for_status(risk_level, anomaly_level, escalation),
    )


def normalized_profile_context(raw: Optional[dict]) -> dict:
    if not raw:
        return {}

    def first_value(*keys: str):
        for key in keys:
            value = raw.get(key)
            if value not in (None, "", [], {}):
                return value
        return None

    normalized = {
        "Age": first_value("age", "Age"),
        "Gender": first_value("gender", "Gender", "sex", "Sex"),
        "DateOfBirth": first_value("dateOfBirth", "DateOfBirth", "dob"),
        "Address": first_value("address", "Address"),
        "Surgery": first_value("surgery", "Surgery"),
        "DischargeDate": first_value("dischargeDate", "DischargeDate"),
        "Allergies": first_value("allergies", "Allergies"),
        "CurrentMedications": first_value("currentMedications", "CurrentMedications", "medications"),
        "EmergencyContactName": first_value("emergencyContactName", "EmergencyContactName"),
        "EmergencyContactPhone": first_value("emergencyContactPhone", "EmergencyContactPhone"),
        "DoctorPhoneNumber": first_value("doctorPhoneNumber", "DoctorPhoneNumber"),
    }
    return {key: value for key, value in normalized.items() if value not in (None, "", [], {})}


def compact_profile_parts(profile: dict) -> list[str]:
    labels = {
        "Age": "age",
        "Gender": "gender",
        "DateOfBirth": "DOB",
        "Surgery": "surgery",
        "DischargeDate": "discharged",
        "Allergies": "allergies",
        "CurrentMedications": "medications",
        "EmergencyContactName": "emergency contact",
        "DoctorPhoneNumber": "caregiver phone",
    }
    parts = []
    for key in ("Age", "Gender", "DateOfBirth", "Surgery", "DischargeDate", "Allergies", "CurrentMedications", "EmergencyContactName", "DoctorPhoneNumber"):
        value = profile.get(key)
        if value and str(value).lower() != "none":
            parts.append(f"{labels[key]}: {value}")
    return parts


def call_context_summary(status: PatientStatusResponse, specialist: dict, profile: dict) -> str:
    vitals = status.vitals
    profile_parts = compact_profile_parts(profile)
    trajectory = ", ".join(f"{point.label} {point.riskLevel} {point.riskScore}" for point in status.trajectory)
    specialist_scope = specialist.get("system_prompt") or ""
    return (
        f"You are {specialist['name']}, specialty {specialist['specialty']}. "
        f"You are a Sova AI care specialist working inside the Sova health monitoring app. "
        f"If the patient asks who you are, say: I’m {specialist['name']}, a {specialist['specialty']} specialist working with Sova. "
        f"Specialty instructions: {specialist_scope} "
        f"Patient {status.patientId} has risk {status.riskLevel} with anomaly level {status.anomalyLevel}. "
        f"Recommended action: {status.recommendedAction}. "
        f"Vitals: heart rate {vitals.heartRate}, HRV {vitals.hrv}, SpO2 {vitals.spo2}, "
        f"blood pressure {vitals.bloodPressure}, temperature {vitals.temperature}, sleep {vitals.sleepHours}. "
        f"Trajectory: {trajectory}. "
        f"Profile: {'; '.join(profile_parts) if profile_parts else 'not available'}. "
        "Conversation rule: do not say only that vitals are bad or that you do not know what is going on. "
        "Use your specialty to explain the most likely concern from the available context, name one concrete thing you are checking, "
        "and give one practical next step or one targeted question. Do not diagnose definitively. "
        "Reply in one or two short spoken sentences unless the patient asks for detail. Advise urgent escalation only for severe symptoms."
    )


def specialist_log(event: str, **metadata):
    safe_metadata = {key: value for key, value in metadata.items() if value is not None}
    print(
        "SovaVoice "
        + json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": event,
                **safe_metadata,
            },
            default=str,
        ),
        flush=True,
    )


def log_preview(text: str, limit: int = 140) -> str:
    compact = " ".join(text.split())
    return compact[:limit]


def specialist_greeting(status: PatientStatusResponse, specialist: dict) -> str:
    return f"Hi, I’m {specialist['name']}, how are you doing?"


def llm_specialist_reply(context: str, user_text: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured for specialist voice calls.")

    from openai import OpenAI

    started = time.perf_counter()
    model = os.getenv("SOVA_SPECIALIST_MODEL", "gpt-4o-mini")
    max_tokens = int(os.getenv("SOVA_SPECIALIST_MAX_TOKENS", "110"))
    temperature = float(os.getenv("SOVA_SPECIALIST_TEMPERATURE", "0.2"))
    timeout_seconds = float(os.getenv("SOVA_OPENAI_TIMEOUT_SECONDS", "8"))
    specialist_log("llm.request.started", textChars=len(user_text), model=model, maxTokens=max_tokens)
    client = OpenAI(api_key=api_key, timeout=timeout_seconds)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": context + " Keep every response under 35 words, specific to your specialty, and natural on a phone call.",
            },
            {"role": "user", "content": user_text},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    reply = response.choices[0].message.content.strip()
    specialist_log("llm.request.succeeded", replyChars=len(reply), durationMs=round((time.perf_counter() - started) * 1000))
    return reply


def openai_transcribe_audio(audio: bytes, audio_format: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured for speech recognition.")

    from openai import OpenAI

    started = time.perf_counter()
    specialist_log("stt.request.started", provider="openai", audioBytes=len(audio), audioFormat=audio_format)
    client = OpenAI(api_key=api_key, timeout=float(os.getenv("SOVA_OPENAI_TIMEOUT_SECONDS", "8")))
    audio_file = io.BytesIO(audio)
    audio_file.name = f"audio.{audio_format}"
    result = client.audio.transcriptions.create(
        model=os.getenv("SOVA_STT_MODEL", "whisper-1"),
        file=audio_file,
    )
    text = (getattr(result, "text", "") or "").strip()
    specialist_log("stt.request.succeeded", provider="openai", textChars=len(text), durationMs=round((time.perf_counter() - started) * 1000))
    return text


def transcribe_audio(audio: bytes, audio_format: str) -> str:
    provider = os.getenv("SOVA_STT_PROVIDER", "openai").strip().lower()
    if provider == "elevenlabs":
        from claw.convo_elevenlabs import speech_to_text
        specialist_log("stt.request.started", provider="elevenlabs", audioBytes=len(audio), audioFormat=audio_format)
        text = speech_to_text(audio, audio_format=audio_format).strip()
        specialist_log("stt.request.succeeded", provider="elevenlabs", textChars=len(text))
        return text
    return openai_transcribe_audio(audio, audio_format)


async def transcribe_audio_turn(audio: bytes, audio_format: str) -> str:
    timeout_seconds = float(os.getenv("SOVA_STT_TIMEOUT_SECONDS", "12"))
    started = time.perf_counter()
    specialist_log(
        "stt.turn.started",
        audioBytes=len(audio),
        audioFormat=audio_format,
        timeoutSeconds=timeout_seconds,
    )
    try:
        text = await asyncio.wait_for(
            asyncio.to_thread(transcribe_audio, audio, audio_format),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        specialist_log(
            "stt.turn.timeout",
            audioBytes=len(audio),
            audioFormat=audio_format,
            durationMs=round((time.perf_counter() - started) * 1000),
        )
        raise RuntimeError("Speech recognition timed out.")
    specialist_log(
        "stt.turn.finished",
        textChars=len(text.strip()),
        durationMs=round((time.perf_counter() - started) * 1000),
    )
    return text


def openai_tts_audio(text: str) -> Optional[dict]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    from openai import OpenAI

    started = time.perf_counter()
    specialist_log("tts.request.started", provider="openai", textChars=len(text))
    client = OpenAI(api_key=api_key, timeout=float(os.getenv("SOVA_OPENAI_TIMEOUT_SECONDS", "8")))
    response = client.audio.speech.create(
        model=os.getenv("SOVA_TTS_MODEL", "tts-1"),
        voice=os.getenv("SOVA_TTS_VOICE", "alloy"),
        input=text,
        response_format="wav",
    )
    audio = response.read()
    specialist_log("tts.request.succeeded", provider="openai", audioBytes=len(audio), format="wav", durationMs=round((time.perf_counter() - started) * 1000))
    return {"audio": base64.b64encode(audio).decode("ascii"), "format": "wav"}


def maybe_tts_audio(text: str) -> Optional[dict]:
    provider = os.getenv("SOVA_TTS_PROVIDER", "elevenlabs").strip().lower()
    if provider == "openai":
        try:
            return openai_tts_audio(text)
        except Exception as exc:
            specialist_log("tts.request.failed", provider="openai", error=str(exc))
            return None

    if os.getenv("ELEVENLABS_API_KEY"):
        try:
            from claw.convo_elevenlabs import text_to_speech
            started = time.perf_counter()
            specialist_log("tts.request.started", provider="elevenlabs", textChars=len(text))
            audio = text_to_speech(text)
            specialist_log("tts.request.succeeded", provider="elevenlabs", audioBytes=len(audio), format="mp3", durationMs=round((time.perf_counter() - started) * 1000))
            return {"audio": base64.b64encode(audio).decode("ascii"), "format": "mp3"}
        except Exception as exc:
            specialist_log("tts.request.failed", provider="elevenlabs", error=str(exc))

    try:
        return openai_tts_audio(text)
    except Exception as exc:
        specialist_log("tts.request.failed", provider="openai", fallback=True, error=str(exc))
        return None


def required_tts_audio(text: str, *, patient_id: str, specialist: dict, session_id: str, phase: str) -> dict:
    audio = maybe_tts_audio(text)
    if not audio:
        specialist_log(
            "audio.required.missing",
            patientId=patient_id,
            specialistId=specialist["id"],
            sessionId=session_id,
            phase=phase,
        )
        raise RuntimeError(f"Unable to synthesize {phase} audio.")
    return audio


def pcm16_to_wav_bytes(pcm: bytes, sample_rate: int = 16_000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output.getvalue()


def specialist_reply_payload(session: dict, user_text: str) -> dict:
    reply = llm_specialist_reply(session["context"], user_text)
    patient_id = session["patient_id"]
    specialist = session["specialist"]
    session_id = session["session_id"]
    audio = required_tts_audio(
        reply,
        patient_id=patient_id,
        specialist=specialist,
        session_id=session_id,
        phase="reply",
    )
    return {"text": reply, "audio": audio["audio"], "format": audio["format"]}


@app.post("/v1/patients/{patient_id}/specialist-calls", response_model=SpecialistCallStartResponse)
async def start_specialist_call(patient_id: str, request: SpecialistCallStartRequest):
    specialist = specialist_by_id(request.specialistId)
    status = await patient_status(patient_id)
    profile = normalized_profile_context(request.patientContext)
    if not profile and USE_BIGQUERY_VITALS:
        try:
            profile = normalized_profile_context(get_patient_profile(patient_id))
        except Exception as exc:
            print(f"Unable to read specialist call profile for patientId={patient_id}: {exc}")

    session_id = str(uuid.uuid4())
    context = call_context_summary(status, specialist, profile)
    specialist_log(
        "context.built",
        patientId=patient_id,
        specialistId=specialist["id"],
        sessionId=session_id,
        hasProfile=bool(profile),
        hasVitals=bool(status.vitals),
        riskLevel=status.riskLevel,
        contextChars=len(context),
    )
    specialist_call_sessions[session_id] = {
        "session_id": session_id,
        "client_session_id": request.clientSessionId,
        "patient_id": patient_id,
        "specialist": specialist,
        "status": status.model_dump(),
        "context": context,
        "audio_chunks": [],
        "audio_chunk_count": 0,
        "audio_byte_count": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    specialist_log(
        "session.created",
        patientId=patient_id,
        specialistId=specialist["id"],
        sessionId=session_id,
        clientSessionId=request.clientSessionId,
    )
    return SpecialistCallStartResponse(
        sessionId=session_id,
        websocketUrl=f"/v1/specialist-calls/{session_id}/stream",
    )


@app.websocket("/v1/specialist-calls/{session_id}/stream")
async def specialist_call_stream(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = specialist_call_sessions.get(session_id)
    if not session:
        await websocket.send_json({"type": "session.error", "message": "Unknown specialist call session."})
        await websocket.close()
        return

    patient_id = session["patient_id"]
    specialist = session["specialist"]
    status = PatientStatusResponse(**session["status"])
    greeting = specialist_greeting(status, specialist)
    send_lock = asyncio.Lock()
    session["reply_generation"] = 0
    reply_task: Optional[asyncio.Task] = None
    specialist_log("websocket.accepted", patientId=patient_id, specialistId=specialist["id"], sessionId=session_id)

    async def send_event(payload: dict):
        async with send_lock:
            try:
                await websocket.send_json(payload)
            except RuntimeError as exc:
                specialist_log(
                    "websocket.send.skipped",
                    patientId=patient_id,
                    specialistId=specialist["id"],
                    sessionId=session_id,
                    eventType=payload.get("type"),
                    error=str(exc),
                )

    async def generate_and_send_reply(generation: int, user_text: str):
        try:
            reply_payload = await asyncio.to_thread(specialist_reply_payload, session, user_text)
        except asyncio.CancelledError:
            specialist_log(
                "reply.pipeline.cancelled",
                patientId=patient_id,
                specialistId=specialist["id"],
                sessionId=session_id,
                generation=generation,
            )
            raise
        except Exception as exc:
            specialist_log("reply.pipeline.failed", patientId=patient_id, specialistId=specialist["id"], sessionId=session_id, generation=generation, error=str(exc))
            await send_event({
                "type": "session.error",
                "sessionId": session_id,
                "message": "One moment.",
            })
            return

        if generation != session.get("reply_generation"):
            specialist_log(
                "reply.pipeline.discarded_stale",
                patientId=patient_id,
                specialistId=specialist["id"],
                sessionId=session_id,
                generation=generation,
                currentGeneration=session.get("reply_generation"),
            )
            return

        specialist_log(
            "agent.reply.text.ready",
            patientId=patient_id,
            specialistId=specialist["id"],
            sessionId=session_id,
            generation=generation,
            textChars=len(reply_payload["text"]),
            textPreview=log_preview(reply_payload["text"]),
        )
        audio = reply_payload["audio"]
        await send_event({
            "type": "agent.audio",
            "sessionId": session_id,
            "turnId": f"reply-{generation}",
            "format": reply_payload.get("format") or "mp3",
            "audioBase64": audio,
        })
        specialist_log(
            "agent.audio.sent",
            patientId=patient_id,
            specialistId=specialist["id"],
            sessionId=session_id,
            generation=generation,
            format=reply_payload.get("format") or "mp3",
            base64Chars=len(audio),
        )

    async def send_spoken_status(text: str, reason: str):
        specialist_log(
            "spoken.status.started",
            patientId=patient_id,
            specialistId=specialist["id"],
            sessionId=session_id,
            reason=reason,
            textChars=len(text),
        )
        try:
            audio = required_tts_audio(
                text,
                patient_id=patient_id,
                specialist=specialist,
                session_id=session_id,
                phase=reason,
            )
            await send_event({
                "type": "agent.audio",
                "sessionId": session_id,
                "turnId": f"status-{reason}",
                "format": audio["format"],
                "audioBase64": audio["audio"],
            })
            specialist_log(
                "spoken.status.sent",
                patientId=patient_id,
                specialistId=specialist["id"],
                sessionId=session_id,
                reason=reason,
                format=audio["format"],
                base64Chars=len(audio["audio"]),
            )
        except Exception as exc:
            specialist_log(
                "spoken.status.failed",
                patientId=patient_id,
                specialistId=specialist["id"],
                sessionId=session_id,
                reason=reason,
                error=str(exc),
            )

    await websocket.send_json({
        "type": "session.started",
        "sessionId": session_id,
        "patientId": patient_id,
        "specialistId": specialist["id"],
        "specialistName": specialist["name"],
    })
    specialist_log("session.started.sent", patientId=patient_id, specialistId=specialist["id"], sessionId=session_id)
    specialist_log("greeting.text.ready", patientId=patient_id, specialistId=specialist["id"], sessionId=session_id, textChars=len(greeting))
    try:
        audio = required_tts_audio(
            greeting,
            patient_id=patient_id,
            specialist=specialist,
            session_id=session_id,
            phase="greeting",
        )
        await websocket.send_json({
            "type": "agent.audio",
            "sessionId": session_id,
            "format": audio["format"],
            "audioBase64": audio["audio"],
        })
        specialist_log(
            "agent.audio.sent",
            patientId=patient_id,
            specialistId=specialist["id"],
            sessionId=session_id,
            format=audio["format"],
            base64Chars=len(audio["audio"]),
        )
    except Exception as exc:
        specialist_log("greeting.audio.failed", patientId=patient_id, specialistId=specialist["id"], sessionId=session_id, error=str(exc))
        await websocket.send_json({
            "type": "session.error",
            "sessionId": session_id,
            "message": "Audio is unavailable for this specialist call.",
        })
        await websocket.close()
        return

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                specialist_log("websocket.disconnected", patientId=patient_id, specialistId=specialist["id"], sessionId=session_id)
                return
            if "bytes" in message and message["bytes"]:
                continue
            raw = message.get("text")
            if raw is None:
                continue
            payload = json.loads(raw)
            event_type = payload.get("type")
            if event_type == "session.end":
                if reply_task and not reply_task.done():
                    reply_task.cancel()
                await websocket.send_json({"type": "session.ended", "sessionId": session_id})
                await websocket.close()
                return
            if event_type == "audio.speech_start":
                session["reply_generation"] = session.get("reply_generation", 0) + 1
                session["audio_chunks"] = []
                session["audio_chunk_count"] = 0
                session["audio_byte_count"] = 0
                if reply_task and not reply_task.done():
                    reply_task.cancel()
                specialist_log(
                    "audio.speech_start.received",
                    patientId=patient_id,
                    specialistId=specialist["id"],
                    sessionId=session_id,
                    generation=session["reply_generation"],
                    turnId=payload.get("turnId"),
                )
                continue
            if event_type == "audio.chunk":
                chunk = payload.get("audioBase64")
                if chunk:
                    audio_chunk = base64.b64decode(chunk)
                    session.setdefault("audio_chunks", []).append(audio_chunk)
                    session["audio_chunk_count"] = session.get("audio_chunk_count", 0) + 1
                    session["audio_byte_count"] = session.get("audio_byte_count", 0) + len(audio_chunk)
                    if session["audio_chunk_count"] == 1 or session["audio_chunk_count"] % 25 == 0:
                        specialist_log(
                            "audio.chunk.received",
                            patientId=patient_id,
                            specialistId=specialist["id"],
                            sessionId=session_id,
                            chunks=session["audio_chunk_count"],
                            audioBytes=session["audio_byte_count"],
                        )
                continue
            if event_type == "audio.end":
                if not session.get("audio_chunks"):
                    specialist_log("audio.end.empty", patientId=patient_id, specialistId=specialist["id"], sessionId=session_id)
                    continue
                try:
                    audio_bytes = b"".join(session.get("audio_chunks", []))
                    session["audio_chunks"] = []
                    specialist_log(
                        "audio.end.received",
                        patientId=patient_id,
                        specialistId=specialist["id"],
                        sessionId=session_id,
                        audioBytes=len(audio_bytes),
                        chunks=session.get("audio_chunk_count", 0),
                    )
                    session["audio_chunk_count"] = 0
                    session["audio_byte_count"] = 0
                    if payload.get("format") == "pcm16":
                        audio_seconds = len(audio_bytes) / 32_000
                        if audio_seconds < 0.25:
                            specialist_log(
                                "audio.turn.too_short",
                                patientId=patient_id,
                                specialistId=specialist["id"],
                                sessionId=session_id,
                                audioSeconds=round(audio_seconds, 2),
                            )
                            continue
                        if audio_seconds < 0.45:
                            specialist_log(
                                "audio.turn.borderline_short",
                                patientId=patient_id,
                                specialistId=specialist["id"],
                                sessionId=session_id,
                                audioSeconds=round(audio_seconds, 2),
                            )
                        audio_bytes = pcm16_to_wav_bytes(audio_bytes)
                        audio_format = "wav"
                    else:
                        audio_format = payload.get("format") or "ogg"
                    user_text = (await transcribe_audio_turn(audio_bytes, audio_format=audio_format)).strip()
                except Exception as exc:
                    specialist_log("stt.request.failed", patientId=patient_id, specialistId=specialist["id"], sessionId=session_id, error=str(exc))
                    await send_spoken_status("I did not catch that clearly. Could you say that once more?", "stt_failed")
                    continue
                if not user_text:
                    specialist_log("stt.empty", patientId=patient_id, specialistId=specialist["id"], sessionId=session_id)
                    await send_spoken_status("I did not catch that clearly. Could you say that once more?", "stt_empty")
                    continue
                event_type = "user.transcript.final"
                payload = {"text": user_text}
            if event_type == "user.transcript.final":
                user_text = (payload.get("text") or "").strip()
                if not user_text:
                    continue
                specialist_log(
                    "user.text.ready",
                    patientId=patient_id,
                    specialistId=specialist["id"],
                    sessionId=session_id,
                    textChars=len(user_text),
                    textPreview=log_preview(user_text),
                )
                specialist_log(
                    "reply.pipeline.started",
                    patientId=patient_id,
                    specialistId=specialist["id"],
                    sessionId=session_id,
                    generation=session.get("reply_generation", 0),
                )
                reply_task = asyncio.create_task(
                    generate_and_send_reply(session.get("reply_generation", 0), user_text)
                )
    except WebSocketDisconnect:
        if reply_task and not reply_task.done():
            reply_task.cancel()
        specialist_log("websocket.disconnected", patientId=patient_id, specialistId=specialist["id"], sessionId=session_id)
        return
    except Exception as exc:
        if reply_task and not reply_task.done():
            reply_task.cancel()
        specialist_log("websocket.failed", patientId=patient_id, specialistId=specialist["id"], sessionId=session_id, error=str(exc))
        try:
            await websocket.send_json({
                "type": "session.error",
                "sessionId": session_id,
                "message": "The specialist is reconnecting.",
            })
        except Exception:
            return


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """
    Single-call endpoint for external systems (e.g. OpenClaw).
    Send full patient data, get back a council decision + doctor report.
    """
    patient_dict = build_patient_dict(request)
    council = LangGraphMedicalCouncil(max_utterances=8)
    result = council.orchestrate_debate(patient_dict, webhook_url=request.webhook_url)
    final_decision = result["final_decision"]

    return {
        "patient_id": request.patient_id,
        "immediate_action": final_decision.get("immediate_action", "Sleep"),
        "decision": final_decision["consensus_recommendation"],
        "doctor_report": final_decision.get("doctor_report", ""),
        "urgency_level": final_decision["urgency_level"],
        "confidence": final_decision["confidence_score"],
        "actions": final_decision["action_items"],
        "escalation_level": {"low": 1, "medium": 2, "high": 3, "critical": 4}.get(
            final_decision["urgency_level"], 2
        ),
        "debate_rounds": result["convergence_state"]["rounds_completed"],
    }

@app.post("/ingest_vitals/{patient_id}")
async def ingest_vitals(patient_id: str, vitals: PatientVitals):
    """Ingest vitals from wearables"""
    if patient_id not in patients:
        patients[patient_id] = PatientState(
            patient_id=patient_id,
            vitals_history=[],
            symptoms=[],
            medications=[]
        )

    patients[patient_id].vitals_history.append(vitals)
    return {"status": "Vitals ingested successfully"}

@app.post("/voice_checkin/{patient_id}")
async def voice_checkin(patient_id: str, transcript: str):
    """Process voice check-in transcript"""
    # Here we would integrate with ElevenLabs for voice processing
    # For now, simulate processing

    if patient_id not in patients:
        patients[patient_id] = PatientState(
            patient_id=patient_id,
            vitals_history=[],
            symptoms=[],
            medications=[]
        )

    # Extract symptoms from transcript (simplified)
    symptoms = extract_symptoms_from_transcript(transcript)
    patients[patient_id].symptoms.extend(symptoms)
    patients[patient_id].last_checkin = datetime.now()

    # Check if council is needed
    needs_council = assess_risk(patients[patient_id])

    return CheckinResponse(
        transcript=transcript,
        patient_state=patients[patient_id],
        needs_council=needs_council
    )

@app.post("/simulate_risk/{patient_id}")
async def simulate_risk(patient_id: str):
    """Simulate deterioration risk trajectory"""
    if patient_id not in patients:
        raise HTTPException(status_code=404, detail="Patient not found")

    patient = patients[patient_id]

    # Here integrate with IFM K2 Think V2 for risk simulation
    # For now, return mock risk assessment
    risk_score = calculate_risk_score(patient)

    return {
        "patient_id": patient_id,
        "risk_score": risk_score,
        "trajectory": "stable" if risk_score < 0.3 else "concerning" if risk_score < 0.7 else "critical"
    }

@app.post("/council_debate/{patient_id}")
async def council_debate(patient_id: str):
    """Convene AI specialist council"""
    if patient_id not in patients:
        raise HTTPException(status_code=404, detail="Patient not found")

    patient = patients[patient_id]

    # Here integrate with Claude API for multi-agent debate
    # For now, return mock decision
    decision = await run_agent_council(patient)

    return decision

@app.post("/escalate/{patient_id}")
async def escalate(patient_id: str, decision: CouncilDecision):
    """Execute escalation actions"""
    # Here integrate with Twilio for SMS/calls
    # Send to patient, family, or doctor based on escalation level

    return {"status": "Escalation executed", "decision": decision.dict()}

# Helper functions
def extract_symptoms_from_transcript(transcript: str) -> List[str]:
    # Simple keyword extraction (in production, use NLP)
    symptoms = []
    symptom_keywords = ["breathless", "pain", "dizzy", "fatigue", "anxious"]
    for keyword in symptom_keywords:
        if keyword.lower() in transcript.lower():
            symptoms.append(keyword)
    return symptoms

def assess_risk(patient: PatientState) -> bool:
    # Simple risk assessment (in production, use ML model)
    if len(patient.vitals_history) > 0:
        latest_vitals = patient.vitals_history[-1]
        if latest_vitals.heart_rate > 100 or latest_vitals.spo2 < 95:
            return True
    return len(patient.symptoms) > 2

def calculate_risk_score(patient: PatientState) -> float:
    # Mock risk calculation
    score = 0.0
    if len(patient.vitals_history) > 0:
        latest = patient.vitals_history[-1]
        if latest.heart_rate > 100:
            score += 0.3
        if latest.spo2 < 95:
            score += 0.4
    score += len(patient.symptoms) * 0.1
    return min(score, 1.0)

async def run_agent_council(patient: PatientState, use_langgraph: bool = True) -> CouncilDecision:
    """Convene AI specialist council using LangGraph or original orchestrator"""

    # Convert PatientState to dict format expected by orchestrator
    patient_dict = {
        "patient_id": patient.patient_id,
        "age": 67,  # Mock - in production get from patient record
        "gender": "female",  # Mock - in production get from patient record
        "admission_date": "2024-01-15",  # Mock
        "discharge_date": "2024-01-20",  # Mock
        "diagnosis": "Acute myocardial infarction",  # Mock
        "heart_rate": patient.vitals_history[-1].heart_rate if patient.vitals_history else 98,
        "spo2": patient.vitals_history[-1].spo2 if patient.vitals_history else 93,
        "blood_pressure": patient.vitals_history[-1].blood_pressure if patient.vitals_history else "142/88",
        "temperature": 98.2,  # Mock
        "symptoms": patient.symptoms,
        "medications": [
            {
                "name": med.get("name", "Unknown"),
                "dose": med.get("dose", "Unknown"),
                "compliance": med.get("compliance", "unknown")
            } for med in patient.medications
        ] if patient.medications else [],
        "lab_results": {
            "troponin": 0.15,  # Mock
            "creatinine": 1.2,  # Mock
            "ef": 45  # Mock
        },
        "lifestyle": {
            "smoking": "former",  # Mock
            "exercise": "sedentary",  # Mock
            "diet": "high sodium"  # Mock
        },
        "trigger": "post_mi_monitoring"
    }

    if use_langgraph:
        # Use LangGraph orchestrator
        council = LangGraphMedicalCouncil(max_utterances=8)
        result = council.orchestrate_debate(patient_dict)
    else:
        # Use original orchestrator
        council = MedicalCouncilOrchestrator()
        result = council.orchestrate_debate(patient_dict, "post_mi_monitoring")

    # Convert orchestrator result to CouncilDecision format
    final_decision = result["final_decision"]

    return CouncilDecision(
        decision=final_decision["consensus_recommendation"],
        doctor_report=final_decision.get("doctor_report", ""),
        confidence=final_decision["confidence_score"],
        actions=final_decision["action_items"],
        escalation_level={
            "low": 1,
            "medium": 2,
            "high": 3,
            "critical": 4
        }.get(final_decision["urgency_level"], 2)
    )

@app.post("/start-debate/{patient_id}")
async def start_debate(patient_id: str, request: AnalyzeRequest):
    """
    OpenClaw calls this. Starts the debate in the background and returns immediately.
    - Streams doctor events to frontend via GET /stream/{patient_id}
    - Pushes final decision + immediate_action to webhook_url when done
    """
    return start_debate_session(patient_id, request, asyncio.get_event_loop())


@app.get("/stream/{patient_id}")
async def stream_debate(patient_id: str):
    """
    Frontend calls this. Pure SSE — just waits and receives events as doctors speak.
    Never needs to send any data.
    """
    async def generate():
        # Wait up to 5 minutes for a debate to start
        for _ in range(300):
            if patient_id in debate_histories:
                break
            await asyncio.sleep(1)
        else:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Timed out waiting for debate to start'})}\n\n"
            return

        for event in debate_histories.get(patient_id, []):
            yield f"data: {json.dumps(event, default=str)}\n\n"
            if event.get("type") in ("done", "error"):
                return

        queue: asyncio.Queue = asyncio.Queue()
        debate_subscribers.setdefault(patient_id, []).append(queue)
        try:
            while True:
                event = await queue.get()
                yield f"data: {json.dumps(event, default=str)}\n\n"
                if event.get("type") in ("done", "error"):
                    break
        finally:
            subscribers = debate_subscribers.get(patient_id, [])
            if queue in subscribers:
                subscribers.remove(queue)


    return StreamingResponse(generate(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no"
    })


@app.get("/ui", response_class=HTMLResponse)
async def ui():
    """Serve the live debate frontend"""
    with open(os.path.join(static_dir, "index.html")) as f:
        return f.read()

@app.post("/stream-analyze")
async def stream_analyze(request: AnalyzeRequest):
    """
    Streaming endpoint — yields each agent's response as SSE the moment it's generated.
    Used by the live debate UI.
    """
    async def generate():
        patient_dict = request.model_dump()
        council = LangGraphMedicalCouncil(max_utterances=8)

        initial_state = {
            "patient_data": patient_dict,
            "debate_history": [],
            "relevant_agents": [],
            "next_agent": "",
            "convergence_score": 0.0,
            "final_decision": None,
            "max_utterances": 8,
            "total_utterances": 0,
        }

        seen_entries = 0
        for chunk in council.graph.stream(initial_state):
            for node_name, state in chunk.items():
                if node_name == "agent_speak":
                    history = state.get("debate_history", [])
                    if len(history) > seen_entries:
                        entry = history[-1]
                        seen_entries = len(history)
                        payload = {
                            "type": "agent",
                            "round": entry["round"],
                            "agent": entry["agent"],
                            "specialty": entry["specialty"],
                            "statement": entry["response"]["statement"],
                            "convergence": state.get("convergence_score", 0.0)
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                        await asyncio.sleep(0)

                elif node_name == "finalize_decision":
                    decision = state.get("final_decision")
                    if decision:
                        payload = {
                            "type": "decision",
                            "immediate_action": decision.get("immediate_action", "Sleep"),
                            "decision": decision["consensus_recommendation"],
                            "doctor_report": decision.get("doctor_report", ""),
                            "urgency_level": decision["urgency_level"],
                            "confidence": decision["confidence_score"],
                            "actions": decision.get("action_items", [])
                        }
                        yield f"data: {json.dumps(payload)}\n\n"

        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
