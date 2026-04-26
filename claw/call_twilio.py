"""
call_twilio.py — outbound call and alert dispatch for SOVA anomalies.

  call_911       — outbound Twilio call to 911 with patient name, address, conditions
  call_caregiver — outbound Twilio call to caregiver with vitals + conditions
  text_caregiver — Telegram message to caregiver chat

No webhook server or ngrok needed — TwiML is passed directly to Twilio.
"""

from __future__ import annotations

import os
import urllib.parse
import urllib.request

from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse


def _load_config() -> dict:
    return {
        "twilio": {
            "account_sid": os.getenv("TWILIO_ACCOUNT_SID", ""),
            "auth_token": os.getenv("TWILIO_AUTH_TOKEN", ""),
            "api_key_sid": os.getenv("TWILIO_API_KEY_SID", ""),
            "api_key_secret": os.getenv("TWILIO_API_KEY_SECRET", ""),
            "phone_number": os.getenv("TWILIO_PHONE_NUMBER", ""),
        },
        "patient": {
            "name": os.getenv("SOVA_PATIENT_NAME", "Ved"),
            "address": os.getenv("SOVA_PATIENT_ADDRESS", "unavailable"),
            "conditions": [
                item.strip()
                for item in os.getenv(
                    "SOVA_PATIENT_CONDITIONS", "post-discharge recovery"
                ).split(",")
                if item.strip()
            ],
            "caregiver_name": os.getenv("SOVA_CAREGIVER_NAME", "caregiver"),
            "caregiver_phone": os.getenv("SOVA_CAREGIVER_PHONE", ""),
        },
        "telegram": {
            "bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
            "chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
        },
    }


def _patient_value(
    data: dict, patient: dict, *keys: str, fallback: str = "unavailable"
) -> str:
    for key in keys:
        value = data.get(key)
        if value:
            return str(value)
    for key in keys:
        value = patient.get(key)
        if value:
            return str(value)
    return fallback


def _first_present(data: dict, *keys: str):
    for key in keys:
        value = data.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _twilio_client(cfg: dict) -> Client:
    t = cfg["twilio"]
    if not t.get("account_sid"):
        raise RuntimeError("TWILIO_ACCOUNT_SID is not configured.")
    if t.get("api_key_sid") and t.get("api_key_secret"):
        return Client(t["api_key_sid"], t["api_key_secret"], t["account_sid"])
    if not t.get("auth_token"):
        raise RuntimeError("TWILIO_AUTH_TOKEN is not configured.")
    return Client(t["account_sid"], t["auth_token"])


def _from_number(cfg: dict) -> str:
    number = cfg["twilio"]["phone_number"]
    if not number:
        raise RuntimeError("TWILIO_PHONE_NUMBER is not configured.")
    return number


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
        f"Address: {patient['address'][:50]}. "
        f"Known conditions: {', '.join(patient['conditions'])}. "
        f"This is an automated alert from SOVA, your AI health assistant. Please dispatch immediately."
    )
    resp = VoiceResponse()
    resp.say(script)

    print(f"⚠  EMERGENCY — calling 911 for {patient['name']}...")
    call = client.calls.create(
        from_=_from_number(cfg),
        to="+16509449481",
        twiml=str(resp),
    )
    print(f"   911 call SID: {call.sid}")


def call_caregiver(data: dict | None = None) -> None:
    data = data or {}
    cfg = _load_config()
    patient = cfg["patient"]
    client = _twilio_client(cfg)
    caregiver_name = _patient_value(
        data, patient, "caregiver_name", "EmergencyContactName", fallback="caregiver"
    )
    caregiver_phone = _patient_value(
        data, patient, "caregiver_phone", "EmergencyContactPhone", "DoctorPhoneNumber"
    )
    if not caregiver_phone or caregiver_phone == "unavailable":
        raise RuntimeError(
            "Caregiver phone is not configured. Set SOVA_CAREGIVER_PHONE or pass EmergencyContactPhone."
        )
    patient_name = _patient_value(
        data,
        patient,
        "name",
        "patientName",
        "patient_id",
        "patientId",
        fallback="the patient",
    )
    if patient_name == "the patient" or patient_name.startswith(("CR-", "PAT-")) or "-" in patient_name:
        patient_name = patient.get("name") or "Ved"
    conditions = patient.get("conditions") or [
        data.get("Surgery") or data.get("surgery") or "post-discharge recovery"
    ]
    vitals_parts = []
    recovery_score = _first_present(data, "recovery_score", "recoveryScore")
    hrv = _first_present(data, "hrv", "HRV")
    heart_rate = _first_present(data, "resting_heart_rate", "HeartRate", "heartRate")
    blood_pressure = _first_present(
        data, "blood_pressure", "BloodPressure", "bloodPressure"
    )
    temperature = _first_present(data, "temperature", "Temperature")

    if recovery_score is not None:
        vitals_parts.append(f"recovery score {recovery_score} out of 100")
    if hrv is not None:
        vitals_parts.append(f"HRV {hrv} milliseconds")
    if heart_rate is not None:
        vitals_parts.append(f"heart rate {heart_rate} beats per minute")
    if blood_pressure is not None:
        vitals_parts.append(f"blood pressure {blood_pressure}")
    if temperature is not None:
        vitals_parts.append(f"temperature {temperature} degrees Fahrenheit")

    script_parts = [
        f"Hello {caregiver_name}, this is SOVA, an AI health assistant, calling about {patient_name}.",
        f"An anomaly has been detected. Urgency level: {_urgency_word(data)}.",
    ]
    if vitals_parts:
        script_parts.append(f"Current vitals: {', '.join(vitals_parts)}.")
    if conditions:
        script_parts.append(f"Known conditions: {', '.join(conditions)}.")
    script_parts.append("Please follow up with the patient immediately.")

    script = " ".join(script_parts)
    resp = VoiceResponse()
    resp.say(script)

    print(f"⚠  Calling caregiver {caregiver_name} for {patient_name}...")
    call = client.calls.create(
        from_=_from_number(cfg),
        to=caregiver_phone,
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
    print(
        f"Patient data: recovery={data['recovery_score']}  hrv={data['hrv']}  rhr={data['resting_heart_rate']}\n"
    )
    fn(data)
    print("\nDone.")
