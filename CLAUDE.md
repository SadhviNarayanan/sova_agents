# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CareRelay is an AI-powered post-hospital care monitoring system. Its core innovation is a **multi-agent council** of 7 specialist medical agents that debate conversationally and reach consensus before making clinical decisions (e.g., escalating a post-MI cardiac patient).

## Setup

```bash
pip install -r carerelay_backend/requirements.txt
```

Required environment variables:
- `OPENAI_API_KEY` — primary LLM (GPT-4, GPT-4o-mini) used by all agents
- `ANTHROPIC_API_KEY` — Claude integration in the FastAPI backend
- `ELEVENLABS_API_KEY` — voice synthesis for patient check-ins
- `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` — SMS/call escalation

## Common Commands

```bash
# Run original orchestrator demo
python agentic_convo.py

# Run LangGraph-based orchestrator demo
python demo_langgraph.py

# Compare both orchestrators side-by-side
python demo_comparison.py

# Start FastAPI backend (http://localhost:8000)
cd carerelay_backend && python main.py

# Run backend demo
python carerelay_backend/demo_council.py

# Run API tests
python carerelay_backend/test_api.py
```

## Architecture

There are two parallel orchestration implementations:

### 1. Original Orchestrator (`agentic_convo.py`)
Manual Python orchestration loop. `MedicalCouncilOrchestrator` manages 7 `MedicalAgent` instances. Each agent has a specialty, short-term memory (last 3 interactions), confidence score, and speak count. The orchestrator routes agents in a dynamic speaking order (specialty-weighted by patient case), runs rounds of debate, and detects convergence via keyword heuristics + optional LLM judge.

### 2. LangGraph Orchestrator (`langgraph_council.py`)
Refactor of the above using `StateGraph` with typed state, conditional edges, and better debugging. Uses the same 7 agents and convergence logic. `demo_langgraph.py` / `demo_comparison.py` exercise this path.

### FastAPI Backend (`carerelay_backend/`)
- **`main.py`** — REST endpoints (see API section below). All imports use `carerelay_backend.*` package paths (required for Render deployment from repo root).
- **`agents.py`** — System prompts for the 6 backend-specific agent roles (Cardiologist, Pharmacist, Advocate, GP, Nurse, Facilitator).
- **`voice_checkin.py`** — ElevenLabs TTS scripts (not yet wired into the API).

### API Endpoints

| Endpoint | Caller | Behavior |
|---|---|---|
| `POST /start-debate/{patient_id}` | OpenClaw | Returns immediately; runs debate in background; POSTs final decision to `webhook_url` when done |
| `GET /stream/{patient_id}` | Frontend | SSE stream — doctor messages arrive one by one, then the final decision event |
| `POST /analyze` | OpenClaw (blocking) | Blocks until debate complete, returns full decision in one response |

### Integration Pattern

**OpenClaw** (async with webhook):
```bash
curl -X POST 'https://sova-agents.onrender.com/start-debate/CR-002' \
  -H 'Content-Type: application/json' -d @payload.json
# Returns: {"status": "debate started", "stream_url": "/stream/CR-002"}
# Later POSTs to webhook_url: {"event": "decision", "immediate_action": "...", "decision": "..."}
```

**Frontend** (SSE listener):
```js
const stream = await fetch('https://sova-agents.onrender.com/stream/CR-002');
// Events: {type:"agent", agent:"...", statement:"..."} ... {type:"decision"} {type:"done"}
```

**OpenClaw** (blocking, no webhook needed):
```bash
curl -X POST 'https://sova-agents.onrender.com/analyze' \
  -H 'Content-Type: application/json' -d @payload.json
# Blocks ~30-60s, returns full decision JSON
```

### Webhook Payload (POSTed to `webhook_url` on debate completion)
```json
{
  "type": "decision",
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

### Deployment
- Hosted on Render: `https://sova-agents.onrender.com`
- `runtime.txt` pins Python version; `render.yaml` sets build/start commands
- Free tier spins down after 15min idle — first request after idle takes ~30s cold start
- `requirements.txt` uses `>=` version pins to ensure Python 3.14-compatible wheels

### Convergence Detection
Dual strategy: fast keyword heuristics (scanning agent responses for agreement/disagreement terms) + slower LLM judge call. The orchestrator exits early when convergence score exceeds threshold or max rounds is reached.

### Data Flow
```
OpenClaw POST /start-debate  →  debate runs in background thread
                             →  frontend GET /stream gets SSE doctor messages
                             →  on completion: webhook POST to OpenClaw callback
                                              + SSE "done" event to frontend
```

## Current State / Known Gaps

- **No database** — all patient state is in-memory only.
- **IFM K2 Think V2 risk simulation** is stubbed (mock returns).
- **ElevenLabs voice** is scripted but not connected to the API.
- **Auth0** authentication is planned but not implemented.
- **No automated test suite** — only manual demo scripts exist.
- LangGraph implementation uses GPT-4 directly; no fallback handling yet.
