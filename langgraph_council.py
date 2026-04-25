"""
LangGraph-based Medical Council Orchestrator
Dynamic speaking order — agents are filtered for relevance and a moderator picks
who speaks next. Convergence pressure increases as the debate progresses.
"""

from typing import Dict, List, Any, TypedDict, Optional
from datetime import datetime
import os
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
import json
import requests as http_requests


class CouncilState(TypedDict):
    patient_data: Dict[str, Any]
    debate_history: List[Dict[str, Any]]
    relevant_agents: List[str]
    next_agent: str
    convergence_score: float
    final_decision: Optional[Dict[str, Any]]
    max_utterances: int
    total_utterances: int
    webhook_url: Optional[str]


class MedicalAgent:
    def __init__(self, name: str, specialty: str, system_prompt: str):
        self.name = name
        self.specialty = specialty
        self.system_prompt = system_prompt
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=100)

    def speak(self, context: Dict[str, Any], convergence_pressure: float) -> Dict[str, Any]:
        self.llm.max_tokens = 100

        debate_context = "\n".join([
            f"{e['agent']} ({e['specialty']}): {e['response']['statement']}"
            for e in context.get('debate_history', [])[-6:]
        ])

        if convergence_pressure < 0.4:
            tone = "Raise your single most important concern about this specific patient. Be direct. 1-2 sentences."
        elif convergence_pressure < 0.7:
            tone = "Respond to what's been said. Build on or challenge a specific point. 1-2 sentences."
        else:
            tone = "We're converging. Either endorse the emerging plan briefly, or raise only your single most critical remaining concern. 1 sentence max."

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=(
                f"Patient: {json.dumps(context['patient_data'], indent=2)}\n\n"
                f"Debate so far:\n{debate_context if debate_context else 'You are first to speak.'}\n\n"
                f"{tone}"
            ))
        ]

        response = self.llm.invoke(messages)
        statement = response.content.strip()

        print(f"\n{self.name} ({self.specialty}): {statement}\n", flush=True)

        return {
            "speaker": self.name,
            "statement": statement,
            "timestamp": datetime.now().isoformat(),
            "agent": self.name,
            "specialty": self.specialty
        }


AGENT_CONFIGS = {
    "cardiologist": {
        "name": "Cardiologist",
        "specialty": "Cardiology",
        "system_prompt": (
            "You are a board-certified cardiologist. Speak in 1-2 sentences only. "
            "Focus only on: arrhythmia risk from this HR, LVEF implications, whether SpO2 suggests pulmonary edema or low-output, "
            "and whether beta-blocker/diuretic doses need urgent adjustment. "
            "Reference actual numbers. Never discuss nutrition or social factors."
        )
    },
    "critical_care": {
        "name": "Critical Care Specialist",
        "specialty": "Critical Care",
        "system_prompt": (
            "You are a critical care specialist. Speak in 1-2 sentences only. "
            "Focus only on: whether vitals meet ICU transfer criteria, the decompensation window if trend continues, "
            "and specific early warning thresholds for this patient. "
            "Reference actual vitals. Do not repeat cardiac or pharmacy points."
        )
    },
    "pharmacist": {
        "name": "Clinical Pharmacist",
        "specialty": "Clinical Pharmacy",
        "system_prompt": (
            "You are a clinical pharmacist. Speak in 1-2 sentences only. "
            "Focus only on: the pharmacokinetic consequence of this patient's specific missed doses, "
            "interaction risks, and whether a formulation or timing change would improve adherence. "
            "Name actual drugs and doses. Do not discuss vitals or surgical options."
        )
    },
    "surgeon": {
        "name": "Cardiothoracic Surgeon",
        "specialty": "Cardiothoracic Surgery",
        "system_prompt": (
            "You are a cardiothoracic surgeon. Speak in 1-2 sentences only. "
            "Only speak if there is a genuine surgical angle — revascularization threshold, mechanical complication, "
            "or operative risk given this patient's current EF and status. "
            "If medical management is clearly appropriate, say so in one sentence and yield to others."
        )
    },
    "general_physician": {
        "name": "General Physician",
        "specialty": "Family Medicine",
        "system_prompt": (
            "You are a family medicine physician. Speak in 1-2 sentences only. "
            "Focus only on: care coordination gaps, whether the patient has support to manage medications at home, "
            "and what the realistic follow-up plan should be. "
            "Do not repeat clinical findings — add the care-coordination view specialists miss."
        )
    },
    "nutritionist": {
        "name": "Clinical Nutritionist",
        "specialty": "Clinical Nutrition",
        "system_prompt": (
            "You are a clinical nutritionist specializing in cardiac care. Speak in 1-2 sentences only. "
            "Focus only on: sodium/fluid load relative to this patient's diuretic status, "
            "and one specific dietary target relevant to their fluid retention and symptoms. "
            "Do not repeat medication or cardiac points."
        )
    },
    "obgyn": {
        "name": "OB/GYN Specialist",
        "specialty": "OB/GYN",
        "system_prompt": (
            "You are an OB/GYN specialist. Speak in 1-2 sentences only. "
            "Only relevant for female patients — address estrogen/progesterone status and cardiovascular risk, "
            "hormone therapy contraindications post-MI, or gender-specific symptom presentation. "
            "If the patient is male, do not speak."
        )
    }
}


def _fire_webhook(url: Optional[str], payload: Dict[str, Any]) -> None:
    if not url:
        return
    try:
        http_requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"[webhook] failed to POST to {url}: {e}", flush=True)


class LangGraphMedicalCouncil:

    def __init__(self, max_utterances: int = 8):
        self.max_utterances = max_utterances
        self.agents = {key: MedicalAgent(**config) for key, config in AGENT_CONFIGS.items()}
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(CouncilState)

        workflow.add_node("initialize_council", self._initialize_council)
        workflow.add_node("filter_relevant_agents", self._filter_relevant_agents)
        workflow.add_node("moderator_pick_next", self._moderator_pick_next)
        workflow.add_node("agent_speak", self._agent_speak)
        workflow.add_node("update_convergence", self._update_convergence)
        workflow.add_node("check_convergence", self._check_convergence)
        workflow.add_node("generate_consensus", self._generate_consensus)
        workflow.add_node("finalize_decision", self._finalize_decision)

        workflow.set_entry_point("initialize_council")
        workflow.add_edge("initialize_council", "filter_relevant_agents")
        workflow.add_edge("filter_relevant_agents", "moderator_pick_next")
        workflow.add_edge("moderator_pick_next", "agent_speak")
        workflow.add_edge("agent_speak", "update_convergence")
        workflow.add_edge("update_convergence", "check_convergence")
        workflow.add_conditional_edges(
            "check_convergence",
            self._should_continue,
            {"continue": "moderator_pick_next", "consensus": "generate_consensus"}
        )
        workflow.add_edge("generate_consensus", "finalize_decision")
        workflow.add_edge("finalize_decision", END)

        return workflow.compile()

    def _initialize_council(self, state: CouncilState) -> CouncilState:
        return {
            **state,
            "debate_history": [],
            "relevant_agents": [],
            "next_agent": "",
            "convergence_score": 0.0,
            "final_decision": None,
            "total_utterances": 0,
        }

    def _filter_relevant_agents(self, state: CouncilState) -> CouncilState:
        """LLM decides which specialists are actually relevant for this patient."""
        patient = state["patient_data"]
        valid_keys = list(AGENT_CONFIGS.keys())

        messages = [
            SystemMessage(content=(
                "You are a medical council coordinator. Given the patient case, return a JSON object "
                "with key 'relevant' containing only the specialist keys that are genuinely relevant. "
                "Exclude specialists with no meaningful contribution — e.g. exclude 'obgyn' for male patients, "
                "exclude 'surgeon' unless there's a clear surgical indication, exclude 'nutritionist' unless "
                "diet/fluid is a primary factor. "
                f"Valid keys: {valid_keys}. Return only valid JSON."
            )),
            HumanMessage(content=(
                f"Patient: {json.dumps(patient, indent=2)}\n"
                "Which specialists are relevant? Return JSON."
            ))
        ]

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        try:
            result = (llm | JsonOutputParser()).invoke(messages)
            relevant = [k for k in result.get("relevant", valid_keys) if k in valid_keys]
            if len(relevant) < 2:
                relevant = valid_keys
        except Exception:
            relevant = [k for k in valid_keys if k != "obgyn" or patient.get("gender") == "female"]

        return {**state, "relevant_agents": relevant}

    def _moderator_pick_next(self, state: CouncilState) -> CouncilState:
        """LLM moderator picks who should speak next based on debate gaps."""
        debate_history = state["debate_history"]
        relevant = state["relevant_agents"]
        total = state["total_utterances"]
        max_u = state["max_utterances"]
        pressure = total / max_u

        spoken_counts = {}
        for e in debate_history:
            key = next((k for k, v in AGENT_CONFIGS.items() if v["name"] == e["agent"]), None)
            if key:
                spoken_counts[key] = spoken_counts.get(key, 0) + 1

        recent_context = "\n".join([
            f"{e['agent']}: {e['response']['statement']}"
            for e in debate_history[-4:]
        ]) if debate_history else "No one has spoken yet."

        if pressure >= 0.7:
            instruction = "We are converging. Pick whoever can best synthesize or raise the single remaining critical gap."
        elif pressure >= 0.4:
            instruction = "Pick whoever has the most important unreplied point or gap to fill based on the debate so far."
        else:
            instruction = "Pick whoever has the most important first contribution to make given the patient's primary issues."

        messages = [
            SystemMessage(content=(
                "You are a medical council moderator. Pick the next specialist to speak. "
                "Return JSON with key 'next' containing one specialist key. "
                f"Relevant specialists: {relevant}. "
                f"Speak counts so far: {spoken_counts}. "
                f"{instruction} "
                "Prefer specialists who haven't spoken yet, but allow repeats if they have the most relevant next point. "
                "Return only valid JSON."
            )),
            HumanMessage(content=(
                f"Patient: {json.dumps(state['patient_data'], indent=2)}\n\n"
                f"Recent debate:\n{recent_context}\n\n"
                "Who speaks next?"
            ))
        ]

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        try:
            result = (llm | JsonOutputParser()).invoke(messages)
            next_agent = result.get("next", "")
            if next_agent not in relevant:
                unspoken = [k for k in relevant if k not in spoken_counts]
                next_agent = unspoken[0] if unspoken else relevant[0]
        except Exception:
            unspoken = [k for k in relevant if k not in spoken_counts]
            next_agent = unspoken[0] if unspoken else relevant[0]

        return {**state, "next_agent": next_agent}

    def _agent_speak(self, state: CouncilState) -> CouncilState:
        agent_key = state["next_agent"]
        agent = self.agents.get(agent_key)
        if not agent:
            return state

        total = state["total_utterances"]
        pressure = total / state["max_utterances"]

        context = {
            "patient_data": state["patient_data"],
            "debate_history": state["debate_history"],
        }

        response = agent.speak(context, pressure)

        entry = {
            "round": (total // max(len(state["relevant_agents"]), 1)) + 1,
            "agent": agent.name,
            "specialty": agent.specialty,
            "response": response,
            "timestamp": datetime.now()
        }

        _fire_webhook(state.get("webhook_url"), {
            "event": "agent_response",
            "patient_id": state["patient_data"].get("patient_id"),
            "agent": agent.name,
            "specialty": agent.specialty,
            "statement": response["statement"],
            "utterance_number": total + 1,
            "timestamp": response["timestamp"],
        })

        return {
            **state,
            "debate_history": state["debate_history"] + [entry],
            "total_utterances": total + 1,
        }

    def _update_convergence(self, state: CouncilState) -> CouncilState:
        history = state["debate_history"]
        if len(history) < 2:
            return state

        recent = [e['response']['statement'] for e in history[-4:]]
        agreement_signals = ['agree', 'concur', 'correct', 'yes', 'right', 'endorse', 'support', 'align']
        disagreement_signals = ['disagree', 'however', 'but', 'concern', 'worried', 'not yet', 'caution']

        agreements = sum(1 for s in recent for w in agreement_signals if w in s.lower())
        disagreements = sum(1 for s in recent for w in disagreement_signals if w in s.lower())
        total = agreements + disagreements

        # Also factor in utterance progress
        utterance_progress = state["total_utterances"] / state["max_utterances"]
        keyword_score = agreements / total if total > 0 else 0.5
        convergence_score = (keyword_score * 0.5) + (utterance_progress * 0.5)

        return {**state, "convergence_score": min(convergence_score, 1.0)}

    def _check_convergence(self, state: CouncilState) -> CouncilState:
        return state

    def _should_continue(self, state: CouncilState) -> str:
        if (state["total_utterances"] < state["max_utterances"] and
                state["convergence_score"] < 0.85):
            return "continue"
        return "consensus"

    def _generate_consensus(self, state: CouncilState) -> CouncilState:
        debate_summary = "\n".join([
            f"{e['agent']} ({e['specialty']}): {e['response']['statement']}"
            for e in state["debate_history"]
        ])

        messages = [
            SystemMessage(content=(
                "You are a medical council facilitator. Synthesize the specialist debate into a final decision. "
                "Return valid JSON only with these keys:\n"
                "- decision: clear actionable clinical decision, 1-2 sentences, specific to this patient\n"
                "- doctor_report: 3-4 sentence clinical handoff summary for another care agent or physician\n"
                "- urgency_level: one of low, medium, high, critical\n"
                "- confidence_score: float 0.0-1.0\n"
                "- action_items: list of 2-4 specific actions\n"
                "- immediate_action: the single most urgent next action. Must be EXACTLY one of these strings with no modifications:\n"
                "  'Call 911'\n"
                "  'Call caregiver'\n"
                "  'Text caregiver'\n"
                "  'Initiate conversation with patient'\n"
                "  'Sleep'\n"
                "Use 'Call 911' only for life-threatening emergencies. "
                "'Sleep' means no action needed. "
                "Be specific to this patient, not generic."
            )),
            HumanMessage(content=(
                f"Patient: {json.dumps(state['patient_data'], indent=2)}\n\n"
                f"Council debate:\n{debate_summary}\n\nSynthesize the consensus."
            ))
        ]

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, max_tokens=400)
        result = (llm | JsonOutputParser()).invoke(messages)

        valid_actions = {"Call 911", "Call caregiver", "Text caregiver", "Initiate conversation with patient", "Sleep"}
        immediate_action = result.get("immediate_action", "Sleep")
        if immediate_action not in valid_actions:
            immediate_action = "Sleep"

        final_decision = {
            "consensus_recommendation": result.get("decision", "Continue current monitoring protocol."),
            "doctor_report": result.get("doctor_report", ""),
            "urgency_level": result.get("urgency_level", "medium"),
            "confidence_score": result.get("confidence_score", 0.75),
            "supporting_agents": len(set(e['agent'] for e in state["debate_history"])),
            "action_items": result.get("action_items", []),
            "immediate_action": immediate_action,
        }

        return {**state, "final_decision": final_decision}

    def _finalize_decision(self, state: CouncilState) -> CouncilState:
        decision = state.get("final_decision")
        if decision:
            _fire_webhook(state.get("webhook_url"), {
                "event": "decision",
                "patient_id": state["patient_data"].get("patient_id"),
                "immediate_action": decision.get("immediate_action", "Sleep"),
                "decision": decision["consensus_recommendation"],
                "doctor_report": decision.get("doctor_report", ""),
                "urgency_level": decision["urgency_level"],
                "confidence_score": decision["confidence_score"],
                "action_items": decision.get("action_items", []),
            })
        return state

    def orchestrate_debate(self, patient_data: Dict[str, Any], webhook_url: Optional[str] = None) -> Dict[str, Any]:
        initial_state: CouncilState = {
            "patient_data": patient_data,
            "debate_history": [],
            "relevant_agents": [],
            "next_agent": "",
            "convergence_score": 0.0,
            "final_decision": None,
            "max_utterances": self.max_utterances,
            "total_utterances": 0,
            "webhook_url": webhook_url,
        }

        final_state = self.graph.invoke(initial_state)

        return {
            "debate_history": final_state["debate_history"],
            "final_decision": final_state["final_decision"],
            "convergence_state": {
                "rounds_completed": final_state["total_utterances"],
                "convergence_score": final_state["convergence_score"],
                "needs_more_info": [],
            },
            "total_rounds": final_state["total_utterances"],
        }
