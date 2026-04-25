# CareRelay Backend

AI-powered post-hospital care monitoring system with **conversational multi-agent council**.

## Features

- **Vitals Ingestion**: Collect data from Apple HealthKit/Fitbit
- **Voice Check-ins**: Natural conversation via ElevenLabs
- **Risk Simulation**: Clinical reasoning with IFM K2 Think V2
- **🤖 Agent Council**: **7 specialist agents debate conversationally until convergence**
- **Automated Escalation**: Twilio SMS/calls based on council decisions

## The Agent Council System

Unlike simple AI responses, CareRelay features a **multi-round conversational debate** between specialist agents:

- **7 Medical Specialists**: Cardiologist, Surgeon, Pharmacist, Nutritionist, General Physician, OBGYN, Critical Care
- **Conversational Debate**: Agents respond to each other, agree/disagree, ask for clarification
- **Convergence Detection**: Debate continues until consensus is reached or max rounds hit
- **Memory & Context**: Each agent remembers previous statements and patient history
- **Dynamic Speaking Order**: Agents speak based on confidence, disagreements, and urgency

### Example Council Debate:
```
🔄 ROUND 1
👨‍⚕️ Dr. Elena Vasquez (Cardiology):
   📋 Assessment: Heart rate elevated, concerning post-MI
   💊 Recommendation: Immediate cardiology follow-up
   ❌ Disagrees with: Dr. Marcus Chen

👨‍⚕️ Dr. Marcus Chen (Pharmacy):
   📋 Assessment: Medication non-compliance likely cause
   💊 Recommendation: Reinforce medication adherence first
   ✅ Agrees with: Dr. Elena Vasquez on monitoring needs

🔄 ROUND 2
👨‍⚕️ Dr. Elena Vasquez (Cardiology):
   📋 Assessment: Acknowledge medication factor, but cardiac risk remains high
   💊 Recommendation: Medication + telehealth within 2 hours
   ✅ Agrees with: Dr. Marcus Chen on medication priority
```

## API Endpoints

- `POST /ingest_vitals/{patient_id}` - Wearable data ingestion
- `POST /voice_checkin/{patient_id}` - Voice conversation processing
- `POST /simulate_risk/{patient_id}` - Risk trajectory analysis
- `POST /council_debate/{patient_id}` - **Full agent council debate**
- `POST /escalate/{patient_id}` - Automated alerts

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-claude-key"  # For council debates
export ELEVENLABS_API_KEY="your-elevenlabs-key"  # For voice
export TWILIO_ACCOUNT_SID="your-twilio-sid"
export TWILIO_AUTH_TOKEN="your-twilio-token"
```

3. Run the demo:
```bash
python demo_council.py
```

4. Run the API server:
```bash
python main.py
```

## Architecture

- `main.py` - FastAPI server with agent council integration
- `agents.py` - Specialist agent prompts and configurations
- `voice_checkin.py` - ElevenLabs conversation scripts
- `demo_council.py` - Interactive agent council demonstration
- `agentic_convo.py` - Core conversational orchestration engine

## Integration Points

- **OpenAI GPT-4**: Agent reasoning and debate responses
- **ElevenLabs**: Voice synthesis and processing
- **IFM K2 Think V2**: Risk simulation (planned)
- **Twilio**: SMS/call escalation
- **Auth0**: Authentication (planned)
- **IFM K2 Think V2**: Risk simulation (planned)
- **Twilio**: SMS/call escalation
- **Auth0**: Authentication (planned)