"""
call_twilio.py — outbound call and alert dispatch for CareRelay anomalies.

  call_911          — places an outbound Twilio call to 911 with patient vitals + address
  call_caregiver    — places an outbound Twilio call to the caregiver with patient conditions
  text_caregiver    — sends a Telegram message to the caregiver chat with patient conditions

All patient and credential config is read from config.json in this directory.
Each function accepts the WHOOP data dict from synthetic_data / db so it can
include live vitals in the alert.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

from twilio.rest import Client


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Telegram helper (replicates send_message from tg.py without importing it)
# ---------------------------------------------------------------------------

def _telegram_send(bot_token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    data = urllib.parse.urlencode(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        resp.read()


def _caregiver_telegram_text(patient: dict, data: dict) -> str:
    return (
        f"<b>CareRelay Alert — {patient['name']}</b>\n\n"
        f"An anomaly has been detected. Caregiver review needed.\n\n"
        f"<b>Conditions:</b> {', '.join(patient['conditions'])}\n"
        f"<b>Live Vitals:</b>\n"
        f"  • Recovery score: {data.get('recovery_score', 'N/A')} / 100\n"
        f"  • HRV: {data.get('hrv', 'N/A')} ms\n"
        f"  • Resting heart rate: {data.get('resting_heart_rate', 'N/A')} bpm\n"
        f"Please follow up with the patient immediately."
    )


# ---------------------------------------------------------------------------
# Public functions — called by heartbeat.py
# ---------------------------------------------------------------------------

def _webhook_url(cfg: dict) -> str:
    url = cfg.get("backend", {}).get("webhook_url", "").rstrip("/")
    if not url:
        raise EnvironmentError("backend.webhook_url not set in config.json — run ngrok and add the URL.")
    return url


def call_911(data: dict | None = None) -> None:
    from call_server import register_call_data
    data = data or {}
    cfg = _load_config()
    patient = cfg["patient"]
    client = _twilio_client(cfg)
    base = _webhook_url(cfg)

    print(f"⚠  EMERGENCY — calling 911 for {patient['name']}...")
    call = client.calls.create(
        from_=_from_number(cfg),
        to="+19112",  # replace with real PSAP number for prod
        url=f"{base}/call/911/start",
        method="POST",
    )
    register_call_data(call.sid, data)
    print(f"   911 call SID: {call.sid}")


def call_caregiver(data: dict | None = None) -> None:
    from call_server import register_call_data
    data = data or {}
    cfg = _load_config()
    patient = cfg["patient"]
    client = _twilio_client(cfg)
    base = _webhook_url(cfg)

    print(f"⚠  Calling caregiver {patient['caregiver_name']} for {patient['name']}...")
    call = client.calls.create(
        from_=_from_number(cfg),
        to=patient["caregiver_phone"],
        url=f"{base}/call/caregiver/start",
        method="POST",
    )
    register_call_data(call.sid, data)
    print(f"   Caregiver call SID: {call.sid}")


def text_caregiver(data: dict | None = None) -> None:
    data = data or {}
    cfg = _load_config()
    patient = cfg["patient"]
    tg = cfg["telegram"]

    print(f"⚠  Sending Telegram alert to caregiver for {patient['name']}...")
    _telegram_send(tg["bot_token"], tg["chat_id"], _caregiver_telegram_text(patient, data))
    print("   Telegram alert sent.")


# ---------------------------------------------------------------------------
# Manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from synthetic_data import get_data

    commands = {"911": call_911, "caregiver": call_caregiver, "text": text_caregiver}
    data = get_data()

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("Usage: python call_twilio.py [911 | caregiver | text]")
        print("  911        — outbound Twilio call to 911 with vitals + address")
        print("  caregiver  — outbound Twilio call to caregiver with conditions")
        print("  text       — Telegram message to caregiver chat")
        sys.exit(1)

    fn = commands[sys.argv[1]]
    print(f"\nTesting: {sys.argv[1]}")
    print(f"Patient data: recovery={data['recovery_score']}  hrv={data['hrv']}  rhr={data['resting_heart_rate']}\n")
    fn(data)
    print("\nDone.")
