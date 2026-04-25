"""
LangGraph vs Original Orchestrator Comparison Demo
Shows the differences between the two agent orchestration approaches
"""

import os
import json
from datetime import datetime
from agentic_convo import MedicalCouncilOrchestrator

def demo_comparison():
    """Compare original vs LangGraph approaches"""

    print("🏥 CareRelay Agent Orchestration Comparison")
    print("============================================")
    print("Comparing Original Orchestrator vs LangGraph Implementation\n")

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

    print("🔄 TESTING ORIGINAL ORCHESTRATOR")
    print("============================================================\n")

    # Test original orchestrator
    try:
        original_orchestrator = MedicalCouncilOrchestrator()
        result = original_orchestrator.orchestrate_debate(patient_case, "post_mi_monitoring")

        print("✅ Original Orchestrator Results:")
        print(f"   • Rounds completed: {len(set(entry['round'] for entry in result['debate_history']))}")
        print(f"   • Total statements: {len(result['debate_history'])}")
        print(f"   • Consensus reached: {'Yes' if result['final_decision'] else 'No'}")

        if result['final_decision']:
            decision = result['final_decision']
            print(f"   • Recommendation: {decision['consensus_recommendation']}")
            print(f"   • Confidence: {decision['confidence_score']:.1%}")

    except Exception as e:
        print(f"❌ Original orchestrator error: {e}")

    print("\n🔄 LANGGRAPH ORCHESTRATOR (Conceptual Overview)")
    print("============================================================\n")

    print("📊 LangGraph Advantages:")
    print("   • State Management: TypedDict-based state with clear schema")
    print("   • Conditional Logic: Explicit conditional edges for convergence")
    print("   • Visualization: Built-in graph visualization capabilities")
    print("   • Modularity: Separate nodes for each orchestration step")
    print("   • Debugging: Better error handling and state inspection")
    print()

    print("🏗️  LangGraph Workflow Structure:")
    print("   1. initialize_council → Set up debate state and speaking order")
    print("   2. agent_speak → Have next agent contribute to discussion")
    print("   3. update_convergence → Analyze discussion for consensus signals")
    print("   4. check_convergence → Decide if debate should continue")
    print("   5. generate_consensus → Create final recommendation")
    print("   6. finalize_decision → Return structured decision")
    print()

    print("🔧 To run LangGraph version:")
    print("   1. Create virtual environment: python3 -m venv venv")
    print("   2. Activate: source venv/bin/activate")
    print("   3. Install deps: pip install langgraph langchain-openai langchain-core openai")
    print("   4. Run demo: python3 demo_langgraph.py")
    print()

    print("✅ Both implementations provide:")
    print("   • Natural language agent conversations")
    print("   • Multi-round debate until convergence")
    print("   • Structured consensus decisions")
    print("   • Integration with FastAPI backend")
    print()

    print("🎯 Recommendation: Use LangGraph for production due to:")
    print("   • Better state management and debugging")
    print("   • More maintainable and extensible code")
    print("   • Built-in workflow visualization")
    print("   • Industry-standard orchestration patterns")

if __name__ == "__main__":
    demo_comparison()