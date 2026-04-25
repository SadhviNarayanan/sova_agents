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
- **`main.py`** — 5 REST endpoints: `ingest_vitals`, `voice_checkin`, `simulate_risk`, `council_debate`, `escalate`. Uses Anthropic Claude for council debates.
- **`agents.py`** — System prompts for the 6 backend-specific agent roles (Cardiologist, Pharmacist, Advocate, GP, Nurse, Facilitator).
- **`voice_checkin.py`** — ElevenLabs TTS scripts (not yet wired into the API).

### Convergence Detection
Dual strategy: fast keyword heuristics (scanning agent responses for agreement/disagreement terms) + slower LLM judge call. The orchestrator exits early when convergence score exceeds threshold or max rounds is reached.

### Data Flow
```
Patient vitals → Council initialization
→ Dynamic speaking order (specialty relevance)
→ Iterative debate rounds (agents respond to each other)
→ Convergence detection
→ Final consensus + escalation decision
→ Twilio SMS/call (if escalation triggered)
```

## Current State / Known Gaps

- **No database** — all patient state is in-memory only.
- **IFM K2 Think V2 risk simulation** is stubbed (mock returns).
- **ElevenLabs voice** is scripted but not connected to the API.
- **Auth0** authentication is planned but not implemented.
- **No automated test suite** — only manual demo scripts exist.
- LangGraph implementation uses GPT-4 directly; no fallback handling yet.
