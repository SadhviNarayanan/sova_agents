"""
LangGraph-based Medical Council Orchestrator
Refactored from the original agentic_convo.py to use LangGraph for better state management
and agent orchestration.
"""

from typing import Dict, List, Any, TypedDict, Optional
from datetime import datetime
import os
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json

# State definition for LangGraph
class CouncilState(TypedDict):
    """State for the medical council debate"""
    patient_data: Dict[str, Any]
    debate_history: List[Dict[str, Any]]
    current_round: int
    current_speaker_index: int
    speaking_order: List[str]
    convergence_score: float
    needs_more_info: List[str]
    active_disagreements: List[str]
    final_decision: Optional[Dict[str, Any]]
    max_rounds: int

# Agent definitions with LangChain prompts
class MedicalAgent:
    def __init__(self, name: str, specialty: str, system_prompt: str):
        self.name = name
        self.specialty = specialty
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=120)
        self.system_prompt = system_prompt
        self.speak_count = 0

    def speak(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate agent response using LangChain"""
        self.speak_count += 1

        debate_context = "\n".join([
            f"{entry['agent']} ({entry['specialty']}): {entry['response']['statement']}"
            for entry in context.get('debate_history', [])[-5:]
        ])

        human_content = (
            f"Patient: {json.dumps(context['patient_data'], indent=2)}\n\n"
            f"Round {context.get('current_round', 1)} debate so far:\n{debate_context}\n\n"
            f"Give your view as {self.specialty}. 2-3 sentences max. Be direct and conversational."
        )

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=human_content)
        ]

        response = self.llm.invoke(messages)
        statement = response.content.strip()

        return {
            "speaker": self.name,
            "statement": statement,
            "timestamp": datetime.now().isoformat(),
            "agent": self.name,
            "specialty": self.specialty
        }

# Agent configurations
AGENT_CONFIGS = {
    "cardiologist": {
        "name": "Dr. Elena Vasquez",
        "specialty": "Cardiology",
        "system_prompt": """You are Dr. Elena Vasquez, cardiologist. Speak in 2-3 sentences only.
Focus exclusively on: LVEF implications, arrhythmia risk from the specific HR shown, whether the SpO2 suggests pulmonary edema or low-output state, and whether beta-blocker/diuretic doses need urgent adjustment.
Reference the actual numbers from this patient's chart. Do not repeat what others have said — add new cardiac-specific insight only.
Never discuss nutrition, lifestyle, or social factors — those belong to other specialists."""
    },

    "critical_care": {
        "name": "Dr. Jennifer Liu",
        "specialty": "Critical Care Medicine",
        "system_prompt": """You are Dr. Jennifer Liu, critical care specialist. Speak in 2-3 sentences only.
Focus exclusively on: hemodynamic instability thresholds, whether this patient's vitals meet ICU transfer criteria, early warning score interpretation, and the specific window before decompensation if current trend continues.
Reference the actual vitals from this patient. Do not repeat cardiac or pharmacy points already raised — add ICU-specific triage perspective only."""
    },

    "pharmacist": {
        "name": "Dr. David Park",
        "specialty": "Clinical Pharmacy",
        "system_prompt": """You are Dr. David Park, clinical pharmacist. Speak in 2-3 sentences only.
Focus exclusively on: the pharmacokinetic consequence of this patient's specific missed doses (Metoprolol half-life rebound, Furosemide volume retention timeline), interaction risks, and whether dose timing or formulation change would improve adherence.
Name the actual drugs and doses. Do not discuss vitals or surgical options — stay in pharmacy lane."""
    },

    "surgeon": {
        "name": "Dr. Marcus Chen",
        "specialty": "Cardiothoracic Surgery",
        "system_prompt": """You are Dr. Marcus Chen, cardiothoracic surgeon. Speak in 2-3 sentences only.
Focus exclusively on: whether current presentation has any surgical indication (e.g. mechanical complication post-MI, refractory angina), what EF threshold would make revascularization vs. medical management the call, and operative risk given this patient's current status.
Be specific about surgical thresholds. Only speak if there is a genuine surgical angle — do not restate medical management points."""
    },

    "general_physician": {
        "name": "Dr. Robert Kim",
        "specialty": "Family Medicine",
        "system_prompt": """You are Dr. Robert Kim, family physician. Speak in 2-3 sentences only.
Focus exclusively on: care coordination gaps (who is following up and when), whether this patient has a support system to manage the missed medications at home, and what the realistic discharge plan looked like vs. what is happening now.
Do not repeat clinical findings — add the care-coordination and system-level view that specialists miss."""
    },

    "nutritionist": {
        "name": "Dr. Sarah Johnson",
        "specialty": "Clinical Nutrition",
        "system_prompt": """You are Dr. Sarah Johnson, cardiac dietitian. Speak in 2-3 sentences only.
Focus exclusively on: sodium and fluid load given Furosemide non-compliance, whether dietary sodium is counteracting the diuretic, and a specific dietary target (e.g. <2g Na/day) relevant to this patient's fluid status and symptoms.
Give a concrete dietary recommendation. Do not repeat medication or cardiac points."""
    },

    "obgyn": {
        "name": "Dr. Maria Rodriguez",
        "specialty": "Obstetrics & Gynecology",
        "system_prompt": """You are Dr. Maria Rodriguez, OB/GYN. Speak in 2-3 sentences only.
Focus exclusively on: estrogen/progesterone status and its effect on this patient's cardiovascular risk, whether hormone therapy is contraindicated post-MI, and any gender-specific symptom presentation differences that may have led to atypical MI symptoms being underrecognized.
Only speak to the gender-specific angle — do not restate anything already covered."""
    }
}

class LangGraphMedicalCouncil:
    """LangGraph-based medical council orchestrator"""

    def __init__(self, max_rounds: int = 5):
        self.max_rounds = max_rounds
        self.agents = {key: MedicalAgent(**config) for key, config in AGENT_CONFIGS.items()}
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(CouncilState)

        # Add nodes
        workflow.add_node("initialize_council", self._initialize_council)
        workflow.add_node("route_speaking_order", self._route_speaking_order)
        workflow.add_node("agent_speak", self._agent_speak)
        workflow.add_node("update_convergence", self._update_convergence)
        workflow.add_node("check_convergence", self._check_convergence)
        workflow.add_node("generate_consensus", self._generate_consensus)
        workflow.add_node("finalize_decision", self._finalize_decision)

        # Add edges
        workflow.set_entry_point("initialize_council")
        workflow.add_edge("initialize_council", "route_speaking_order")
        workflow.add_edge("route_speaking_order", "agent_speak")
        workflow.add_edge("agent_speak", "update_convergence")
        workflow.add_edge("update_convergence", "check_convergence")

        # Conditional edges from check_convergence
        workflow.add_conditional_edges(
            "check_convergence",
            self._should_continue,
            {
                "continue": "agent_speak",
                "consensus": "generate_consensus"
            }
        )

        workflow.add_edge("generate_consensus", "finalize_decision")
        workflow.add_edge("finalize_decision", END)

        return workflow.compile()

    def _initialize_council(self, state: CouncilState) -> CouncilState:
        """Initialize the council debate"""
        return {
            **state,
            "current_round": 1,
            "current_speaker_index": 0,
            "speaking_order": [],
            "debate_history": [],
            "convergence_score": 0.0,
            "needs_more_info": [],
            "active_disagreements": [],
            "final_decision": None
        }

    def _route_speaking_order(self, state: CouncilState) -> CouncilState:
        """Use an LLM chain to dynamically determine specialist speaking order"""
        valid_keys = list(AGENT_CONFIGS.keys())
        patient = state["patient_data"]

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        messages = [
            SystemMessage(content=(
                "You are a medical council facilitator. Given the patient case details, "
                "return a JSON object with a single key 'order' whose value is a list of specialist "
                f"keys ordered by clinical priority (most relevant first). "
                f"Valid keys are: {valid_keys}. Include all keys exactly once. "
                "Return only valid JSON, no explanation."
            )),
            HumanMessage(content=(
                f"Patient trigger: {patient.get('trigger', 'post_mi_monitoring')}\n"
                f"Patient data: {json.dumps(patient, indent=2)}\n\n"
                "Return the speaking order as JSON."
            ))
        ]

        try:
            result = (llm | JsonOutputParser()).invoke(messages)
            order = result.get("order", valid_keys)
            # Ensure all keys present and valid
            order = [k for k in order if k in valid_keys]
            missing = [k for k in valid_keys if k not in order]
            order = order + missing
        except Exception:
            order = valid_keys

        return {**state, "speaking_order": order}

    def _agent_speak(self, state: CouncilState) -> CouncilState:
        """Have the next agent in the speaking order speak, then advance the index"""
        speaking_order = state["speaking_order"]
        debate_history = state["debate_history"]
        speaker_index = state["current_speaker_index"]
        current_round = state["current_round"]

        agent_key = speaking_order[speaker_index]
        agent = self.agents[agent_key]

        context = {
            "patient_data": state["patient_data"],
            "debate_history": debate_history,
            "current_round": current_round
        }

        response = agent.speak(context)

        print(f"\n👨‍⚕️ {agent.name} ({agent.specialty}):")
        print(f"   {response['statement']}\n", flush=True)

        debate_entry = {
            "round": current_round,
            "agent": agent.name,
            "specialty": agent.specialty,
            "response": response,
            "timestamp": datetime.now()
        }

        next_index = speaker_index + 1
        next_round = current_round
        if next_index >= len(speaking_order):
            next_index = 0
            next_round = current_round + 1

        return {
            **state,
            "debate_history": debate_history + [debate_entry],
            "current_speaker_index": next_index,
            "current_round": next_round
        }

    def _update_convergence(self, state: CouncilState) -> CouncilState:
        """Update convergence metrics"""
        debate_history = state["debate_history"]

        if len(debate_history) < 2:
            return state

        # Analyze recent statements for convergence
        recent_statements = [
            entry['response']['statement']
            for entry in debate_history[-len(self.agents):]  # Last round
        ]

        # Simple convergence heuristics
        agreement_signals = ['agree', 'concurs', 'good point', 'yes', 'right', 'makes sense']
        disagreement_signals = ['disagree', 'however', 'but', 'concern', 'worried']

        agreements = sum(1 for stmt in recent_statements
                        for signal in agreement_signals if signal in stmt.lower())
        disagreements = sum(1 for stmt in recent_statements
                           for signal in disagreement_signals if signal in stmt.lower())

        total_signals = agreements + disagreements
        convergence_score = agreements / total_signals if total_signals > 0 else 0.5

        # Track information needs
        info_signals = ['need to know', 'what about', 'clarify', 'more information']
        needs = set()
        for stmt in recent_statements:
            for signal in info_signals:
                if signal in stmt.lower():
                    needs.add(signal)

        return {
            **state,
            "convergence_score": convergence_score,
            "needs_more_info": list(needs)
        }

    def _check_convergence(self, state: CouncilState) -> CouncilState:
        """Check if debate has converged"""
        return state  # Logic handled in conditional edge

    def _should_continue(self, state: CouncilState) -> str:
        """Determine if debate should continue or reach consensus"""
        current_round = state["current_round"]
        convergence_score = state["convergence_score"]

        if current_round <= self.max_rounds and convergence_score < 0.7:
            return "continue"

        return "consensus"

    def _generate_consensus(self, state: CouncilState) -> CouncilState:
        """Generate final consensus using LLM synthesis of the debate"""
        debate_history = state["debate_history"]

        debate_summary = "\n".join([
            f"{e['agent']} ({e['specialty']}): {e['response']['statement']}"
            for e in debate_history
        ])

        messages = [
            SystemMessage(content=(
                "You are a medical council facilitator. Synthesize the specialist debate into a final decision. "
                "Return valid JSON only with these keys:\n"
                "- decision: clear actionable clinical decision, 1-2 sentences\n"
                "- doctor_report: 3-4 sentence clinical handoff summary for another care agent or physician\n"
                "- urgency_level: one of low, medium, high, critical\n"
                "- confidence_score: float 0.0-1.0\n"
                "- action_items: list of 2-4 specific actions\n"
                "Make the decision and report specific to this patient, not generic."
            )),
            HumanMessage(content=(
                f"Patient: {json.dumps(state['patient_data'], indent=2)}\n\n"
                f"Council debate:\n{debate_summary}\n\nSynthesize the consensus."
            ))
        ]

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, max_tokens=400)
        result = (llm | JsonOutputParser()).invoke(messages)

        final_decision = {
            "consensus_recommendation": result.get("decision", "Continue current monitoring protocol."),
            "doctor_report": result.get("doctor_report", ""),
            "urgency_level": result.get("urgency_level", "medium"),
            "confidence_score": result.get("confidence_score", 0.75),
            "supporting_agents": len(set(e['agent'] for e in debate_history)),
            "action_items": result.get("action_items", []),
        }

        return {**state, "final_decision": final_decision}

    def _finalize_decision(self, state: CouncilState) -> CouncilState:
        """Finalize and return the decision"""
        return state

    def orchestrate_debate(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete council debate using LangGraph"""
        initial_state: CouncilState = {
            "patient_data": patient_data,
            "debate_history": [],
            "current_round": 1,
            "current_speaker_index": 0,
            "speaking_order": [],
            "convergence_score": 0.0,
            "needs_more_info": [],
            "active_disagreements": [],
            "final_decision": None,
            "max_rounds": self.max_rounds
        }

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        return {
            "debate_history": final_state["debate_history"],
            "final_decision": final_state["final_decision"],
            "convergence_state": {
                "rounds_completed": final_state["current_round"],
                "convergence_score": final_state["convergence_score"],
                "needs_more_info": final_state["needs_more_info"]
            },
            "total_rounds": len(set(entry["round"] for entry in final_state["debate_history"]))
        }

    def get_graph_visualization(self):
        """Get graph visualization (for debugging/development)"""
        try:
            from langchain_core.runnables.graph import MermaidDrawMethod
            return self.graph.get_graph().draw_mermaid()
        except ImportError:
            return "Graph visualization requires additional dependencies"