#!/usr/bin/env python3
"""
CareRelay Agent Council Demo
Shows the full conversational debate between specialist agents
"""

import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agentic_convo import MedicalCouncilOrchestrator

def demo_agent_council():
    """Demonstrate the full agentic conversation system"""

    print("🏥 CareRelay Agentic Council Demo")
    print("=" * 60)
    print("Watch 7 specialist agents debate a post-MI patient case...")
    print("They will converse back and forth until convergence!")
    print()

    # Sample patient case - post myocardial infarction
    patient_case = {
        "patient_id": "CR-2024-001",
        "age": 67,
        "gender": "female",
        "admission_date": "2024-01-15",
        "discharge_date": "2024-01-20",
        "diagnosis": "Acute myocardial infarction",
        "heart_rate": 98,  # Elevated
        "spo2": 93,  # Low
        "blood_pressure": "142/88",
        "temperature": 98.2,
        "symptoms": ["mild dyspnea", "fatigue", "occasional chest discomfort"],
        "medications": [
            {"name": "Aspirin", "dose": "81mg daily", "compliance": "good"},
            {"name": "Metoprolol", "dose": "25mg BID", "compliance": "missed morning dose"},
            {"name": "Atorvastatin", "dose": "40mg daily", "compliance": "good"},
            {"name": "Furosemide", "dose": "20mg daily", "compliance": "missed"}
        ],
        "lab_results": {
            "troponin": 0.15,
            "creatinine": 1.2,
            "ef": 45
        },
        "lifestyle": {
            "smoking": "former",
            "exercise": "sedentary",
            "diet": "high sodium"
        }
    }

    print("📋 Patient Case:")
    print(f"   • 67-year-old female, 5 days post-MI")
    print(f"   • HR: {patient_case['heart_rate']} bpm, SpO2: {patient_case['spo2']}%")
    print(f"   • Symptoms: {', '.join(patient_case['symptoms'])}")
    print(f"   • Missed medications: Metoprolol, Furosemide")
    print()

    # Initialize and run the council
    council = MedicalCouncilOrchestrator()
    result = council.orchestrate_debate(patient_case, "post_mi_monitoring")

    print("\n🎯 Council Results:")
    print(f"   • Rounds completed: {result['total_rounds']}")
    print(f"   • Final decision: {result['final_decision']['consensus_recommendation']}")
    print(f"   • Confidence: {result['final_decision']['confidence_score']:.1%}")
    print(f"   • Urgency: {result['final_decision']['urgency_level']}")
    print(f"   • Action items: {len(result['final_decision']['action_items'])}")

    return result

if __name__ == "__main__":
    # Set dummy API key for demo
    os.environ["OPENAI_API_KEY"] = "demo-key"

    try:
        result = demo_agent_council()
        print("\n✅ Demo completed successfully!")
        print("The agents debated, converged, and reached a consensus decision.")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Make sure OPENAI_API_KEY is set to a valid key for actual API calls.")