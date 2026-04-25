"""
LangGraph-based Medical Council Demo
Demonstrates the agentic conversation system using LangGraph for orchestration
"""

import os
import json
from datetime import datetime
from langgraph_council import LangGraphMedicalCouncil

def demo_langgraph_council():
    """Demonstrate the LangGraph-based medical council"""

    print("🏥 CareRelay LangGraph Medical Council Demo")
    print("============================================")
    print("Watch 7 specialist agents debate using LangGraph orchestration...")
    print("Agents will converse naturally until convergence!\n")

    # Sample patient case
    patient_case = {
        "patient_id": "CR-2024-001",
        "age": 67,
        "gender": "female",
        "admission_date": "2024-01-15",
        "discharge_date": "2024-01-20",
        "diagnosis": "Acute myocardial infarction",
        "heart_rate": 98,
        "spo2": 93,
        "blood_pressure": "142/88",
        "temperature": 98.2,
        "symptoms": ["mild dyspnea", "fatigue", "occasional chest discomfort"],
        "medications": [
            {"name": "Metoprolol", "dose": "25mg BID", "compliance": "missed doses"},
            {"name": "Furosemide", "dose": "20mg daily", "compliance": "missed doses"}
        ],
        "trigger": "post_mi_monitoring"
    }

    print("📋 Patient Case:")
    print(f"   • {patient_case['age']}-year-old {patient_case['gender']}, 5 days post-MI")
    print(f"   • HR: {patient_case['heart_rate']} bpm, SpO2: {patient_case['spo2']}%")
    print(f"   • Symptoms: {', '.join(patient_case['symptoms'])}")
    print(f"   • Missed medications: {', '.join([med['name'] for med in patient_case['medications']])}")
    print()

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found!")
        print("Set your OpenAI API key to run with real agents:")
        print("export OPENAI_API_KEY='your-key-here'")
        print("\nRunning with mock responses for demo...\n")
        use_mock = True
    else:
        print("✅ OpenAI API key found - running with real agents!\n")
        use_mock = False

    print("🏥 MEDICAL COUNCIL CONVENED")
    print("============================================================\n")

    # Initialize LangGraph orchestrator
    orchestrator = LangGraphMedicalCouncil(max_rounds=2)

    # Run the debate
    try:
        result = orchestrator.orchestrate_debate(patient_case)

        # Display final decision
        if result["final_decision"]:
            decision = result["final_decision"]
            print("✅ CONSENSUS REACHED")
            print("\n🏆 FINAL CONSENSUS DECISION")
            print("=" * 50)
            print(f"📋 Recommendation: {decision['consensus_recommendation']}")
            print(f"🎯 Confidence: {decision['confidence_score']:.1%}")
            print(f"🔥 Urgency: {decision['urgency_level'].upper()}")
            print(f"👥 Support: {decision['supporting_agents']}/7 agents")

            if decision.get('doctor_report'):
                print(f"\n📝 Doctor Report:\n   {decision['doctor_report']}")

            if decision.get('action_items'):
                print("✅ Action Items:")
                for item in decision['action_items']:
                    print(f"   • {item}")

            print(f"\n📊 Debate Summary:")
            print(f"   • Total Rounds: {result['convergence_state']['rounds_completed']}")
            print(f"   • Convergence Score: {result['convergence_state']['convergence_score']:.1%}")
            print(f"   • Information Gaps: {len(result['convergence_state']['needs_more_info'])}")

        print("\n✅ Demo completed successfully!")
        print("The LangGraph-based agents debated and reached consensus!")

    except Exception as e:
        print(f"❌ Error during debate: {e}")
        print("Make sure your OpenAI API key is valid and you have internet connection.")

if __name__ == "__main__":
    demo_langgraph_council()