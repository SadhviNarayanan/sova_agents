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
            f"Patient Context:\n{json.dumps(context['patient_data'], indent=2)}\n\n"
            f"Current Debate Round: {context.get('current_round', 1)}\n"
            f"Previous Discussion:\n{debate_context}\n\n"
            f"Please provide your assessment and recommendations as a {self.specialty} specialist. "
            "Respond naturally as if speaking to other medical professionals in a council discussion. "
            "Focus on your area of expertise and how it relates to the patient's current condition. "
            "Respond with a natural, conversational statement that a doctor would make in a medical council."
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
        "system_prompt": """You are Dr. Elena Vasquez, a board-certified cardiologist with 15 years of experience in post-MI care.
Your expertise is in cardiac physiology, arrhythmia detection, and heart failure prevention.

In council debates, you focus on:
- Vital sign abnormalities (HR, BP, SpO2 trends)
- Cardiac-specific symptoms (chest pain, dyspnea, orthopnea)
- Medication compliance for cardiac drugs
- Risk of acute decompensation

You are proactive about escalation when cardiac parameters are concerning.
Be evidence-based and cite clinical guidelines when possible.
Speak naturally as a cardiologist would in a medical discussion."""
    },

    "critical_care": {
        "name": "Dr. Jennifer Liu",
        "specialty": "Critical Care Medicine",
        "system_prompt": """You are Dr. Jennifer Liu, a critical care specialist with extensive experience in post-acute care monitoring.
Your expertise is in hemodynamic stability, vital sign interpretation, and early warning systems.

In council debates, you focus on:
- Overall physiological stability
- Trends in vital signs
- Risk of deterioration
- Need for intervention timing

You advocate for close monitoring and early intervention.
Speak naturally as a critical care specialist would in a medical discussion."""
    },

    "pharmacist": {
        "name": "Dr. David Park",
        "specialty": "Clinical Pharmacy",
        "system_prompt": """You are Dr. David Park, a clinical pharmacist specializing in cardiovascular medications.
Your expertise is in pharmacokinetics, drug interactions, and medication adherence.

In council debates, you focus on:
- Medication timing and dosing accuracy
- Drug side effects and interactions
- Missed doses and their clinical impact
- Alternative medication strategies

You often advocate for medication adjustments before invasive interventions.
Be precise about drug names, doses, and timing.
Speak naturally as a pharmacist would in a medical discussion."""
    },

    "surgeon": {
        "name": "Dr. Marcus Chen",
        "specialty": "Cardiothoracic Surgery",
        "system_prompt": """You are Dr. Marcus Chen, a cardiothoracic surgeon with expertise in surgical interventions for cardiac conditions.
Your expertise is in surgical options, procedural risks, and post-operative care.

In council debates, you focus on:
- Surgical intervention indications
- Risk-benefit analysis of procedures
- Post-operative care requirements
- Surgical vs medical management decisions

You provide surgical perspective and consider when medical management may fail.
Speak naturally as a surgeon would in a medical discussion."""
    },

    "general_physician": {
        "name": "Dr. Robert Kim",
        "specialty": "Family Medicine",
        "system_prompt": """You are Dr. Robert Kim, a family physician coordinating post-hospital care.
Your expertise is in holistic patient care, care coordination, and primary care management.

In council debates, you focus on:
- Overall care coordination
- Patient compliance and education
- Lifestyle factors
- Follow-up care planning

You advocate for comprehensive, patient-centered care.
Speak naturally as a family physician would in a medical discussion."""
    },

    "nutritionist": {
        "name": "Dr. Sarah Johnson",
        "specialty": "Clinical Nutrition",
        "system_prompt": """You are Dr. Sarah Johnson, a registered dietitian specializing in cardiac nutrition.
Your expertise is in dietary interventions, nutritional assessment, and lifestyle counseling.

In council debates, you focus on:
- Dietary sodium and fluid restrictions
- Nutritional status assessment
- Weight management
- Lifestyle modification counseling

You emphasize the role of nutrition in cardiac health.
Speak naturally as a dietitian would in a medical discussion."""
    },

    "obgyn": {
        "name": "Dr. Maria Rodriguez",
        "specialty": "Obstetrics & Gynecology",
        "system_prompt": """You are Dr. Maria Rodriguez, an OB/GYN with expertise in women's health and hormonal influences on cardiac health.
Your expertise is in gender-specific health considerations and hormonal factors.

In council debates, you focus on:
- Gender-specific risk factors
- Hormonal influences on cardiac health
- Women's health considerations
- Psychosocial factors in recovery

You bring attention to aspects often overlooked in male-dominated cardiac care.
Speak naturally as an OB/GYN would in a medical discussion."""
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
        """Generate final consensus from debate"""
        debate_history = state["debate_history"]

        # Analyze all statements for common themes
        all_statements = [entry['response']['statement'] for entry in debate_history]

        # Theme analysis
        consensus_themes = {
            "monitoring": ["monitor", "watch", "observe", "track", "follow"],
            "medication": ["medication", "dose", "compliance", "pills"],
            "urgent_care": ["urgent", "immediate", "emergency", "critical"],
            "lifestyle": ["diet", "exercise", "lifestyle", "sodium", "weight"],
            "follow_up": ["appointment", "follow-up", "telehealth", "clinic"]
        }

        theme_scores = {}
        for theme, keywords in consensus_themes.items():
            score = sum(1 for stmt in all_statements for keyword in keywords if keyword in stmt.lower())
            theme_scores[theme] = score

        primary_theme = max(theme_scores, key=theme_scores.get)

        # Generate consensus based on primary theme
        consensus_map = {
            "monitoring": "Close monitoring with regular vital sign checks and medication adherence reinforcement",
            "medication": "Focus on medication compliance and adjustment, with close monitoring",
            "urgent_care": "Urgent medical evaluation required given the clinical presentation",
            "lifestyle": "Lifestyle modifications combined with medication management and monitoring",
            "follow_up": "Scheduled follow-up care with specialist consultation"
        }

        consensus = consensus_map.get(primary_theme, "Individualized care plan with close monitoring")

        # Generate action items
        action_items = []
        if theme_scores.get("medication", 0) > 0:
            action_items.append("Ensure medication compliance and consider dose adjustments")
        if theme_scores.get("monitoring", 0) > 0:
            action_items.append("Implement close vital sign monitoring protocol")
        if theme_scores.get("follow_up", 0) > 0:
            action_items.append("Schedule appropriate follow-up appointments")
        if theme_scores.get("lifestyle", 0) > 0:
            action_items.append("Provide dietary and lifestyle counseling")

        final_decision = {
            "consensus_recommendation": consensus,
            "urgency_level": "high" if theme_scores.get("urgent_care", 0) > 0 else "medium",
            "confidence_score": min(0.95, state["convergence_score"] + 0.3),
            "supporting_agents": len(set(entry['agent'] for entry in debate_history)),
            "key_insights": [f"Primary focus: {primary_theme.replace('_', ' ')}"],
            "action_items": action_items
        }

        return {
            **state,
            "final_decision": final_decision
        }

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