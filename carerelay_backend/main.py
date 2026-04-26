from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import os
import sys
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date, timedelta, timezone
from carerelay_backend.agents import AGENT_PROMPTS
import anthropic  # For Claude API integration

# Add parent directory to path to import agentic_convo and langgraph_council
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agentic_convo import MedicalCouncilOrchestrator
from langgraph_council import LangGraphMedicalCouncil
from claw.anomaly import infer_anomaly_level
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

class PatientStatusResponse(BaseModel):
    patientId: str
    vitals: StatusVitals
    anomalyLevel: int
    riskLevel: str
    recommendedAction: str
    escalation: StatusEscalation
    deliberation: StatusDeliberation

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
high_risk_lock = threading.Lock()
HIGH_RISK_COOLDOWN = timedelta(minutes=15)
executor = ThreadPoolExecutor()

@app.get("/")
async def root():
    return {"message": "CareRelay API is running"}

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


def status_vitals_from_row(patient_id: str, raw: dict) -> StatusVitals:
    return StatusVitals(
        heartRate=_int_or_none(_value(raw, "heartRate", "HeartRate", "resting_heart_rate", "rhr")),
        hrv=_int_or_none(_value(raw, "hrv", "HRV")),
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
    return level


def infer_patient_anomaly(patient_id: str, raw: dict, vitals: StatusVitals) -> int:
    model_level = infer_anomaly_level(anomaly_snapshot(patient_id, raw, vitals))
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

    def run():
        try:
            council = LangGraphMedicalCouncil(max_utterances=8, event_callback=on_event)
            council.orchestrate_debate(build_patient_dict(request), webhook_url=request.webhook_url)
        except Exception as exc:
            on_event({
                "type": "error",
                "event": "error",
                "patient_id": patient_id,
                "message": str(exc),
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


@app.get("/v1/patients/{patient_id}/status", response_model=PatientStatusResponse)
async def patient_status(patient_id: str):
    try:
        raw_vitals = get_latest_vitals(patient_id)
    except Exception as exc:
        print(f"Unable to read latest vitals from BigQuery for patientId={patient_id}: {exc}")
        raise HTTPException(status_code=503, detail="Unable to read latest vitals")

    if not raw_vitals:
        print(f"No vitals row found in BigQuery for patientId={patient_id}")
        raise HTTPException(status_code=404, detail="No vitals found for patient")

    try:
        profile = get_patient_profile(patient_id)
    except Exception as exc:
        print(f"Unable to read patient profile from BigQuery for patientId={patient_id}: {exc}")
        profile = {}

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
    )


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
