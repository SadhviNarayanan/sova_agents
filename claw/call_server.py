"""
call_server.py — FastAPI webhook server for live AI-powered Twilio calls.

Twilio POSTs to this server each turn of the conversation.
GPT generates the response; Twilio speaks it and listens for the next reply.

Run:
    uvicorn call_server:app --port 8080

Then expose with ngrok:
    ngrok http 8080

Put the ngrok URL in config.json → backend.webhook_url
"""
from __future__ import annotations

import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response
from openai import OpenAI
from twilio.twiml.voice_response import Gather, VoiceResponse

app = FastAPI()

# In-memory conversation store keyed by Twilio CallSid
_conversations: dict[str, list] = {}
_call_data: dict[str, dict] = {}


def _load_config() -> dict:
    path = Path(__file__).parent / "config.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _gpt(history: list, cfg: dict) -> str:
    client = OpenAI(api_key=cfg["backend"]["chat_api"])
    system = (
        "You are CareRelay, an AI care assistant on a live phone call with a caregiver or emergency responder. "
        "You already introduced the situation. Now answer questions, provide more detail, "
        "and guide next steps based on the patient's data. "
        "Keep every response to 2-3 short sentences — this is a phone call."
    )
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system}] + history,
        max_tokens=120,
    )
    return resp.choices[0].message.content.strip()


def _gather_twiml(say_text: str, action: str) -> str:
    resp = VoiceResponse()
    gather = Gather(
        input="speech",
        action=action,
        method="POST",
        speech_timeout="auto",
        language="en-US",
    )
    gather.say(say_text)
    resp.append(gather)
    resp.say("I didn't catch that. Goodbye.")
    return str(resp)


# ---------------------------------------------------------------------------
# Caregiver call endpoints
# ---------------------------------------------------------------------------

@app.post("/call/caregiver/start")
async def caregiver_start(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid")
    cfg = _load_config()
    patient = cfg["patient"]
    data = _call_data.get(call_sid, {})

    intro = (
        f"Hello, this is CareRelay calling about your patient, {patient['name']}. "
        f"An anomaly has been detected in their health data. "
        f"Their current recovery score is {data.get('recovery_score', 'unavailable')} out of 100, "
        f"with an HRV of {data.get('hrv', 'unavailable')} milliseconds "
        f"and a resting heart rate of {data.get('resting_heart_rate', 'unavailable')} beats per minute. "
        f"Known conditions include {', '.join(patient['conditions'])}. "
        f"Do you have any questions, or would you like more detail?"
    )

    _conversations[call_sid] = [{"role": "assistant", "content": intro}]
    return Response(
        content=_gather_twiml(intro, "/call/caregiver/respond"),
        media_type="text/xml",
    )


@app.post("/call/caregiver/respond")
async def caregiver_respond(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid")
    speech = (form.get("SpeechResult") or "").strip()

    cfg = _load_config()
    history = _conversations.get(call_sid, [])

    if speech:
        history.append({"role": "user", "content": speech})
        reply = _gpt(history, cfg)
        history.append({"role": "assistant", "content": reply})
        _conversations[call_sid] = history
    else:
        reply = "I didn't catch that. Is there anything else I can help with?"

    return Response(
        content=_gather_twiml(reply, "/call/caregiver/respond"),
        media_type="text/xml",
    )


# ---------------------------------------------------------------------------
# 911 call endpoints
# ---------------------------------------------------------------------------

@app.post("/call/911/start")
async def emergency_start(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid")
    cfg = _load_config()
    patient = cfg["patient"]
    data = _call_data.get(call_sid, {})

    intro = (
        f"Emergency medical alert. Patient name: {patient['name']}. "
        f"Date of birth: {patient['dob']}. "
        f"Address: {patient['address']}. "
        f"Known conditions: {', '.join(patient['conditions'])}. "
        f"Current vitals — "
        f"resting heart rate: {data.get('resting_heart_rate', 'unknown')} beats per minute, "
        f"HRV: {data.get('hrv', 'unknown')} milliseconds, "
        f"recovery score: {data.get('recovery_score', 'unknown')} out of 100. "
        f"This is an automated CareRelay emergency alert. Please dispatch immediately. "
        f"Do you need me to repeat any information?"
    )

    _conversations[call_sid] = [{"role": "assistant", "content": intro}]
    return Response(
        content=_gather_twiml(intro, "/call/911/respond"),
        media_type="text/xml",
    )


@app.post("/call/911/respond")
async def emergency_respond(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid")
    speech = (form.get("SpeechResult") or "").strip()

    cfg = _load_config()
    history = _conversations.get(call_sid, [])

    if speech:
        history.append({"role": "user", "content": speech})
        reply = _gpt(history, cfg)
        history.append({"role": "assistant", "content": reply})
        _conversations[call_sid] = history
    else:
        reply = "Standing by. Do you need any information repeated?"

    return Response(
        content=_gather_twiml(reply, "/call/911/respond"),
        media_type="text/xml",
    )


# ---------------------------------------------------------------------------
# Helper — register call data before dialing so the webhook can access it
# ---------------------------------------------------------------------------

def register_call_data(call_sid: str, data: dict) -> None:
    _call_data[call_sid] = data


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
