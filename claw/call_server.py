"""
call_server.py — FastAPI webhook server for Twilio outbound calls.

Twilio POSTs to /call/911/start or /call/caregiver/start, gets back TwiML <Say>,
and reads the pre-built script aloud. No ElevenLabs, no audio files.

Run:
    uvicorn call_server:app --port 8080
Expose:
    ngrok http 8080
"""
from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse

app = FastAPI()

_call_data: dict[str, dict] = {}


def _load_config() -> dict:
    return {
        "patient": {
            "name": os.getenv("SOVA_PATIENT_NAME", "Ved"),
            "address": os.getenv("SOVA_PATIENT_ADDRESS", "unavailable"),
            "conditions": [
                item.strip()
                for item in os.getenv("SOVA_PATIENT_CONDITIONS", "post-discharge recovery").split(",")
                if item.strip()
            ],
            "caregiver_name": os.getenv("SOVA_CAREGIVER_NAME", "caregiver"),
        }
    }


def _urgency(data: dict) -> str:
    try:
        score = int(data.get("recovery_score", 0))
        return "high" if score < 40 else "medium" if score < 65 else "low"
    except (TypeError, ValueError):
        return "high"


@app.post("/call/911/start")
async def emergency_start(_request: Request):
    cfg = _load_config()
    patient = cfg["patient"]

    script = (
        f"Emergency medical alert. "
        f"Patient name: {patient['name']}. "
        f"Address: {patient['address']}. "
        f"Known conditions: {', '.join(patient['conditions'])}. "
        f"This is an automated CareRelay emergency alert. Please dispatch immediately."
    )

    resp = VoiceResponse()
    resp.say(script)
    return Response(content=str(resp), media_type="text/xml")


@app.post("/call/caregiver/start")
async def caregiver_start(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid", "")
    cfg = _load_config()
    patient = cfg["patient"]
    data = _call_data.get(call_sid, {})

    urgency = _urgency(data)
    recovery = data.get("recovery_score", "unavailable")
    hrv      = data.get("hrv", "unavailable")
    rhr      = data.get("resting_heart_rate", "unavailable")

    script = (
        f"Hello {patient['caregiver_name']}, this is CareRelay calling about {patient['name']}. "
        f"An anomaly has been detected. Urgency level: {urgency}. "
        f"Current vitals — recovery score {recovery} out of 100, "
        f"HRV {hrv} milliseconds, resting heart rate {rhr} beats per minute. "
        f"Known conditions: {', '.join(patient['conditions'])}. "
        f"Please follow up with the patient immediately."
    )

    resp = VoiceResponse()
    resp.say(script)
    return Response(content=str(resp), media_type="text/xml")


def register_call_data(call_sid: str, data: dict) -> None:
    _call_data[call_sid] = data


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
