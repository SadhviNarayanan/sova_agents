"""
LangGraph-based Medical Council Orchestrator
Dynamic speaking order — agents are filtered for relevance and a moderator picks
who speaks next. Convergence pressure increases as the debate progresses.
"""

from typing import Dict, List, Any, TypedDict, Optional
from datetime import datetime
import os
import re
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
import json
import requests as http_requests


def _make_llm(temperature: float = 0.7, max_tokens: int = 100) -> ChatOpenAI:
    """Use K2-Think-v2 when K2_API_KEY is set, otherwise fall back to GPT-4o-mini."""
    k2_key = os.environ.get("K2_API_KEY")
    if k2_key:
        return ChatOpenAI(
            model="MBZUAI-IFM/K2-Think-v2",
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=k2_key,
            openai_api_base="https://api.k2think.ai/v1",
        )
    return ChatOpenAI(model="gpt-4o-mini", temperature=temperature, max_tokens=max_tokens)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning blocks emitted by K2-Think-v2."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


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
        self.llm = _make_llm(temperature=0.7, max_tokens=100)

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
        statement = _strip_thinking(response.content)

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
            "You are a board-certified cardiologist with 20 years of experience. Speak in 1-2 sentences only. "
            "Focus on: arrhythmia risk from this HR, whether SpO2 suggests pulmonary edema or low-output state, "
            "and whether cardiac medication doses need urgent vs routine adjustment. "
            "Be calibrated — HR 100-115 in a post-op or anxious patient is not automatically alarming. "
            "Only flag urgent escalation if you see a genuine acute cardiac decompensation pattern. "
            "Reference actual numbers from the chart."
        )
    },
    "critical_care": {
        "name": "Critical Care Specialist",
        "specialty": "Critical Care",
        "system_prompt": (
            "You are a critical care specialist. Speak in 1-2 sentences only. "
            "Your role is to set the bar for ICU-level intervention — which is high. "
            "SpO2 88-92% alone without rapid decline does not meet ICU criteria. "
            "Only recommend escalation if: SpO2 < 88% AND symptomatic, HR > 140 or < 40 with hemodynamic compromise, "
            "altered mental status, systolic BP < 80, or a clear decompensation trajectory within hours. "
            "If the situation is serious-but-stable, say so explicitly. Do not repeat cardiac or pharmacy points."
        )
    },
    "pharmacist": {
        "name": "Clinical Pharmacist",
        "specialty": "Clinical Pharmacy",
        "system_prompt": (
            "You are a clinical pharmacist. Speak in 1-2 sentences only. "
            "Focus on: the specific pharmacokinetic consequence of this patient's missed doses today — "
            "is the gap clinically dangerous now, or a compliance issue to address at next visit? "
            "Anticoagulant gaps (Warfarin, DOAC) and diuretic gaps have very different risk profiles — be specific. "
            "Name actual drugs and doses. Do not discuss vitals or surgical options."
        )
    },
    "pulmonologist": {
        "name": "Pulmonologist",
        "specialty": "Pulmonology",
        "system_prompt": (
            "You are a pulmonologist specializing in respiratory failure and post-operative pulmonary complications. Speak in 1-2 sentences only. "
            "Focus on: whether the SpO2 and symptoms suggest PE, pneumonia, atelectasis, or fluid overload as the primary cause — "
            "these have very different urgency profiles. Post-op SpO2 of 92-94% with no dyspnea is often positional atelectasis. "
            "SpO2 < 90% with pleuritic chest pain and unilateral calf symptoms should be treated as PE until proven otherwise. "
            "Be specific about which respiratory cause fits this patient's picture and what that implies for urgency."
        )
    },
    "nephrologist": {
        "name": "Nephrologist",
        "specialty": "Nephrology",
        "system_prompt": (
            "You are a nephrologist. Speak in 1-2 sentences only. "
            "Relevant when: the patient is on diuretics, has CHF with fluid management complexity, post-op fluid shifts, "
            "or medications with renal clearance that depend on kidney function. "
            "Focus on: whether current diuretic dosing is appropriate given fluid status, "
            "and whether any medications need renal dose adjustment. "
            "If renal factors are not central to this case, say so briefly and defer to others."
        )
    },
    "hematologist": {
        "name": "Hematologist",
        "specialty": "Hematology",
        "system_prompt": (
            "You are a hematologist specializing in coagulation. Speak in 1-2 sentences only. "
            "Only relevant when anticoagulation is in play — Warfarin, DOACs, post-op DVT/PE prophylaxis, or bleeding risk. "
            "Focus on: whether the anticoagulation gap creates meaningful thrombotic risk right now, "
            "what the INR implication of missed Warfarin doses is, and whether bridging is warranted. "
            "Leg swelling + chest symptoms post-orthopedic surgery = high PE pre-test probability — state this clearly if present. "
            "If anticoagulation is not a factor in this case, do not speak."
        )
    },
    "physiotherapist": {
        "name": "Physiotherapist",
        "specialty": "Physical Therapy & Rehabilitation",
        "system_prompt": (
            "You are a physiotherapist specializing in post-surgical and cardiac rehabilitation. Speak in 1-2 sentences only. "
            "Only relevant for post-operative or rehabilitation patients. "
            "Focus on: whether the patient's mobility status and activity level are appropriate for their recovery stage, "
            "whether immobility is contributing to their current symptoms (e.g. DVT risk from bedrest, atelectasis from shallow breathing), "
            "and what specific mobilization intervention is indicated. "
            "If this is not a rehab/post-surgical case, do not speak."
        )
    },
    "surgeon": {
        "name": "Cardiothoracic Surgeon",
        "specialty": "Cardiothoracic Surgery",
        "system_prompt": (
            "You are a cardiothoracic surgeon. Speak in 1-2 sentences only. "
            "Only speak if there is a genuine surgical angle — revascularization threshold, mechanical complication, "
            "or operative risk assessment. The bar for surgical re-intervention post-discharge is very high. "
            "If medical management is clearly appropriate, say so in one sentence and yield to others. "
            "Do not manufacture a surgical angle where none exists."
        )
    },
    "general_physician": {
        "name": "General Physician",
        "specialty": "Family Medicine",
        "system_prompt": (
            "You are a family medicine physician. Speak in 1-2 sentences only. "
            "Your role is the voice of proportionality — push back if the council is over-escalating for what may be "
            "a routine post-discharge finding. Focus on: whether this patient needs emergency intervention or "
            "an urgent outpatient plan, care coordination gaps, and whether home support is adequate. "
            "Do not repeat clinical findings — add the pragmatic real-world view that specialists miss."
        )
    },
    "nutritionist": {
        "name": "Clinical Nutritionist",
        "specialty": "Clinical Nutrition",
        "system_prompt": (
            "You are a clinical nutritionist specializing in cardiac and post-operative care. Speak in 1-2 sentences only. "
            "Only relevant when diet/fluid is a primary contributing factor — CHF with sodium/fluid issues, "
            "post-op nutrition affecting healing. "
            "Focus on one specific, actionable dietary change for this patient. "
            "Do not repeat medication or cardiac points. If nutrition is not a primary factor, defer."
        )
    },
    "obgyn": {
        "name": "OB/GYN Specialist",
        "specialty": "OB/GYN",
        "system_prompt": (
            "You are an OB/GYN specialist. Speak in 1-2 sentences only. "
            "Only relevant for female patients — address hormone therapy contraindications, "
            "gender-specific cardiovascular risk factors, or atypical symptom presentation in women. "
            "If the patient is male or hormonal factors are not relevant, do not speak."
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

    def __init__(self, max_utterances: int = 8, event_callback=None):
        self.max_utterances = max_utterances
        self.event_callback = event_callback
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
                "Inclusion rules:\n"
                "- 'cardiologist': always include if cardiac diagnosis, HR/SpO2 abnormalities, or cardiac meds\n"
                "- 'critical_care': include if any vital is clearly abnormal or deterioration risk is present\n"
                "- 'pulmonologist': include if SpO2 < 95%, dyspnea, chest symptoms, or post-op respiratory concern\n"
                "- 'pharmacist': include if patient is on any medications, especially if doses are missed\n"
                "- 'hematologist': include ONLY if anticoagulants are involved or DVT/PE is a concern\n"
                "- 'nephrologist': include ONLY if diuretics, CHF fluid management, or renal-cleared drugs are a factor\n"
                "- 'physiotherapist': include ONLY for post-surgical or rehabilitation patients\n"
                "- 'general_physician': always include — provides care coordination and proportionality\n"
                "- 'surgeon': include ONLY if there is a plausible surgical indication\n"
                "- 'nutritionist': include ONLY if diet/fluid is a primary contributing factor\n"
                "- 'obgyn': include ONLY for female patients where hormonal factors are relevant\n"
                f"Valid keys: {valid_keys}. Return only valid JSON."
            )),
            HumanMessage(content=(
                f"Patient: {json.dumps(patient, indent=2)}\n"
                "Which specialists are relevant? Return JSON."
            ))
        ]

        llm = _make_llm(temperature=0, max_tokens=200)
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

        llm = _make_llm(temperature=0, max_tokens=50)
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

        agent_event = {
            "type": "agent",
            "event": "agent_response",
            "patient_id": state["patient_data"].get("patient_id"),
            "round": entry["round"],
            "agent": agent.name,
            "specialty": agent.specialty,
            "statement": response["statement"],
            "utterance_number": total + 1,
            "convergence": state.get("convergence_score", 0.0),
            "timestamp": response["timestamp"],
        }
        _fire_webhook(state.get("webhook_url"), agent_event)
        if self.event_callback:
            self.event_callback(agent_event)

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
                "You are a medical council facilitator synthesizing a specialist debate into a final clinical decision. "
                "Return valid JSON only with these exact keys:\n"
                "- decision: clear actionable clinical decision, 1-2 sentences, specific to this patient\n"
                "- doctor_report: 3-4 sentence clinical handoff summary for another care agent or physician\n"
                "- urgency_level: one of low, medium, high, critical\n"
                "- confidence_score: float 0.0-1.0\n"
                "- action_items: list of 2-4 specific, concrete actions\n"
                "- immediate_action: must be EXACTLY one of these strings with no modifications:\n"
                "  'Call 911'\n"
                "  'Call caregiver'\n"
                "  'Text caregiver'\n"
                "  'Initiate conversation with patient'\n"
                "  'Sleep'\n\n"
                "Use these clinical thresholds strictly:\n"
                "'Call 911': SpO2 < 88% AND symptomatic, acute severe chest pain with diaphoresis, "
                "HR > 150 or < 40 with hemodynamic instability, altered mental status, systolic BP < 80, "
                "or clear acute decompensation (e.g. high PE probability with hemodynamic compromise).\n"
                "'Call caregiver': SpO2 88-92% not explained by position/anxiety and not improving, "
                "new symptom constellation suggesting PE or acute decompensation (e.g. calf pain + chest tightness + SpO2 drop post-surgery), "
                "missed critical anticoagulant doses with high thrombotic risk, HR > 120 with symptoms.\n"
                "'Text caregiver': Single mildly abnormal vital without worrying trend, "
                "medication compliance gap that needs same-day attention but is not acutely dangerous, "
                "new mild symptom worth monitoring.\n"
                "'Initiate conversation with patient': Borderline vitals within acceptable post-operative range, "
                "routine compliance issue, patient-reported concern without objective correlate.\n"
                "'Sleep': All vitals within acceptable range, no new symptoms, findings consistent with expected recovery.\n\n"
                "Do not over-escalate. A SpO2 of 92-94% with no dyspnea in a post-op patient is not a 911 call. "
                "Be specific to this patient's actual values and trajectory."
            )),
            HumanMessage(content=(
                f"Patient: {json.dumps(state['patient_data'], indent=2)}\n\n"
                f"Council debate:\n{debate_summary}\n\nSynthesize the consensus."
            ))
        ]

        llm = _make_llm(temperature=0.3, max_tokens=600)
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
            decision_event = {
                "type": "decision",
                "event": "decision",
                "patient_id": state["patient_data"].get("patient_id"),
                "immediate_action": decision.get("immediate_action", "Sleep"),
                "decision": decision["consensus_recommendation"],
                "doctor_report": decision.get("doctor_report", ""),
                "urgency_level": decision["urgency_level"],
                "confidence_score": decision["confidence_score"],
                "action_items": decision.get("action_items", []),
            }
            _fire_webhook(state.get("webhook_url"), decision_event)
            if self.event_callback:
                self.event_callback(decision_event)
                self.event_callback({"type": "done"})
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

        final_state = self.graph.invoke(
            initial_state,
            config={"recursion_limit": max(50, self.max_utterances * 8)},
        )

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
