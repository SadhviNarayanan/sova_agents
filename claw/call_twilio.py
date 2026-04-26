"""
call_twilio.py — outbound call and alert dispatch for SOVA anomalies.

  call_911       — outbound Twilio call to 911 with patient name, address, conditions
  call_caregiver — outbound Twilio call to caregiver with vitals + conditions
  text_caregiver — Telegram message to caregiver chat

No webhook server or ngrok needed — TwiML is passed directly to Twilio.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse


def _load_config() -> dict:
    path = Path(__file__).parent / "config.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _twilio_client(cfg: dict) -> Client:
    t = cfg["twilio"]
    if t.get("api_key_sid") and t.get("api_key_secret"):
        return Client(t["api_key_sid"], t["api_key_secret"], t["account_sid"])
    return Client(t["account_sid"], t["auth_token"])


def _from_number(cfg: dict) -> str:
    return cfg["twilio"]["phone_number"]


def _urgency_word(data: dict) -> str:
    try:
        score = int(data.get("recovery_score", 0))
        return "high" if score < 40 else "medium" if score < 65 else "low"
    except (TypeError, ValueError):
        return "high"


def _urgency_emoji(data: dict) -> str:
    try:
        score = int(data.get("recovery_score", 0))
        return "🔴 HIGH" if score < 40 else "🟡 MEDIUM" if score < 65 else "🟢 LOW"
    except (TypeError, ValueError):
        return "🔴 HIGH"


def _telegram_send(bot_token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    data = urllib.parse.urlencode(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        resp.read()


def call_911(data: dict | None = None) -> None:
    data = data or {}
    cfg = _load_config()
    patient = cfg["patient"]
    client = _twilio_client(cfg)

    script = (
        f"Emergency medical alert. "
        f"Patient name: {patient['name']}. "
        f"Address: {patient['address']}. "
        f"Known conditions: {', '.join(patient['conditions'])}. "
        f"This is an automated alert from SOVA, your AI health assistant. Please dispatch immediately."
    )
    resp = VoiceResponse()
    resp.say(script)

    print(f"⚠  EMERGENCY — calling 911 for {patient['name']}...")
    call = client.calls.create(
        from_=_from_number(cfg),
        to="+19112",
        twiml=str(resp),
    )
    print(f"   911 call SID: {call.sid}")


def call_caregiver(data: dict | None = None) -> None:
    data = data or {}
    cfg = _load_config()
    patient = cfg["patient"]
    client = _twilio_client(cfg)

    script = (
        f"Hello {patient['caregiver_name']}, this is SOVA, an AI health assistant, calling about {patient['name']}. "
        f"An anomaly has been detected. Urgency level: {_urgency_word(data)}. "
        f"Current vitals: recovery score {data.get('recovery_score', 'unavailable')} out of 100, "
        f"HRV {data.get('hrv', 'unavailable')} milliseconds, "
        f"resting heart rate {data.get('resting_heart_rate', 'unavailable')} beats per minute. "
        f"Known conditions: {', '.join(patient['conditions'])}. "
        f"Please follow up with the patient immediately."
    )
    resp = VoiceResponse()
    resp.say(script)

    print(f"⚠  Calling caregiver {patient['caregiver_name']} for {patient['name']}...")
    call = client.calls.create(
        from_=_from_number(cfg),
        to=patient["caregiver_phone"],
        twiml=str(resp),
    )
    print(f"   Caregiver call SID: {call.sid}")


def text_caregiver(data: dict | None = None) -> None:
    data = data or {}
    cfg = _load_config()
    patient = cfg["patient"]
    tg = cfg["telegram"]

    text = (
        f"<b>SOVA Alert — {patient['name']}</b>\n\n"
        f"<b>Urgency:</b> {_urgency_emoji(data)}\n\n"
        f"<b>Issue:</b> Anomaly detected in live vitals — immediate review needed.\n\n"
        f"<b>Conditions:</b> {', '.join(patient['conditions'])}\n\n"
        f"<b>Live Vitals:</b>\n"
        f"  • Recovery score: {data.get('recovery_score', 'N/A')} / 100\n"
        f"  • HRV: {data.get('hrv', 'N/A')} ms\n"
        f"  • Resting heart rate: {data.get('resting_heart_rate', 'N/A')} bpm\n\n"
        f"Please follow up with the patient immediately."
    )

    print(f"⚠  Sending Telegram alert to caregiver for {patient['name']}...")
    _telegram_send(tg["bot_token"], tg["chat_id"], text)
    print("   Telegram alert sent.")


if __name__ == "__main__":
    import sys
    from synthetic_data import get_data

    commands = {"911": call_911, "caregiver": call_caregiver, "text": text_caregiver}
    data = get_data()

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("Usage: python call_twilio.py [911 | caregiver | text]")
        print("  911       — outbound call to 911 with name, address, conditions")
        print("  caregiver — outbound call to caregiver with vitals + conditions")
        print("  text      — Telegram message to caregiver")
        sys.exit(1)

    fn = commands[sys.argv[1]]
    print(f"\nTesting: {sys.argv[1]}")
    print(f"Patient data: recovery={data['recovery_score']}  hrv={data['hrv']}  rhr={data['resting_heart_rate']}\n")
    fn(data)
    print("\nDone.")
