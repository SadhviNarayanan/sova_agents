# Sova

> AI doctor council that monitors patients after hospital discharge and decides in real time whether they need emergency care.

## Team

- Sadhvi Narayanan
- Ishita Jain
- Ved Panse
- Varshini Vijay

---

## What We're Building

Patients discharged from hospital enter a dangerous gap — no continuous monitoring, no specialist oversight, and a follow-up appointment days away. Sova fills that gap.

When a patient's wearable detects an anomaly, Sova automatically convenes a council of AI specialist doctors. They debate the patient's vitals, medications, surgery history, and risk level in real time, then issue a single decision: **Call 911, call caregiver, text caregiver, initiate conversation, or do nothing.**

Every decision is calibrated — not every elevated heart rate is an emergency. The council argues about it first.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph (StateGraph) |
| LLM | OpenAI GPT-4o-mini |
| Backend | FastAPI (Python) |
| Real-time streaming | Server-Sent Events (SSE) |
| Voice calls | Twilio + ElevenLabs |
| Wearable data | WHOOP via BigQuery |
| Database | Google BigQuery |
| Deployment | Render |

---

## How It Works

1. **Wearable detects anomaly** — heartbeat monitor computes `anomaly_level` (0–4) from live vitals
2. **Council is convened** — LangGraph dynamically selects the relevant specialists for this patient's case
3. **Specialists debate** — each agent speaks in turn, responds to each other, challenges assumptions
4. **Convergence** — when the council agrees, a facilitator synthesizes the debate into a structured decision
5. **Dispatch** — decision fires to the frontend via SSE stream and to OpenClaw via webhook

### Specialist Roster
Cardiologist, Critical Care, Pulmonologist, Hematologist, Nephrologist, Pharmacist, Physiotherapist, General Physician, Nutritionist, OB/GYN — assembled per-patient based on their specific case.

---

## API

### Start a debate (async)
```bash
curl -X POST 'https://sova-agents.onrender.com/start-debate/{patientId}' \
  -H 'Content-Type: application/json' \
  -d '{
    "patientId": "CR-002",
    "Age": 72,
    "Gender": "male",
    "Surgery": "CABG",
    "DischargeDate": "2026-04-20",
    "RiskLevel": "High",
    "CurrentMedications": "Furosemide 40mg, Spironolactone 25mg",
    "severity": 2,
    "stage": 1,
    "vitals": { "HeartRate": 112, "BloodPressure": "152/94", "Temperature": 98.8 },
    "anomaly_level": 3,
    "webhook_url": "https://your-system.com/callback"
  }'
```
Returns immediately. Final decision POSTed to `webhook_url` when debate completes.

### Watch the debate live (SSE)
```bash
curl -N 'https://sova-agents.onrender.com/stream/{patientId}'
```
Doctor messages arrive one by one. Connect before or after the debate starts — stream waits up to 5 minutes.

### Blocking call (get result directly)
```bash
curl -X POST 'https://sova-agents.onrender.com/analyze' \
  -H 'Content-Type: application/json' -d @payload.json
```
Blocks ~30–60s, returns full decision JSON.

### Webhook response payload
```json
{
  "event": "decision",
  "patient_id": "CR-002",
  "immediate_action": "Call caregiver",
  "decision": "...",
  "doctor_report": "...",
  "urgency_level": "high",
  "confidence_score": 0.87,
  "action_items": [...]
}
```

---

## Local Setup

```bash
pip install -r requirements.txt
```

Environment variables:
```
OPENAI_API_KEY
ANTHROPIC_API_KEY
ELEVENLABS_API_KEY
TWILIO_ACCOUNT_SID
TWILIO_AUTH_TOKEN
GOOGLE_APPLICATION_CREDENTIALS_JSON
```

Run locally:
```bash
cd Sova_backend && python main.py
```
