#!/usr/bin/env python3
"""Quick test script for K2-Think-v2 council integration."""

import json
import os
import sys

# Load K2 key from claw/config.json if not already in env
if not os.environ.get("K2_API_KEY"):
    config_path = os.path.join(os.path.dirname(__file__), "claw", "config.json")
    try:
        with open(config_path) as f:
            config = json.load(f)
        os.environ["K2_API_KEY"] = config["backend"]["k2_api"]
    except Exception as e:
        print(f"Error: couldn't load K2_API_KEY from claw/config.json: {e}")
        sys.exit(1)

from langgraph_council import LangGraphMedicalCouncil

patient = {
    "patient_id": "test-k2-001",
    "age": 67,
    "gender": "female",
    "surgery": "post-MI",
    "risk_level": "high",
    "vitals": {"heart_rate": 98, "blood_pressure": "142/88"},
    "anomaly_level": 2,
    "current_medications": "Metoprolol 25mg (missed morning dose), Furosemide 20mg (missed)",
    "symptoms": ["mild dyspnea", "fatigue"],
}

print("Running council with K2-Think-v2 (4 utterances)...\n")
council = LangGraphMedicalCouncil(max_utterances=4)
result = council.orchestrate_debate(patient)

print("\n--- Final Decision ---")
print(json.dumps(result["final_decision"], indent=2))
print(f"\nRounds: {result['total_rounds']}  |  Convergence: {result['convergence_state']['convergence_score']:.2f}")
