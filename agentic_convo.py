from openai import OpenAI
import json
import random
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# 🧠 SPECIALIST AGENTS
# -----------------------------
class MedicalAgent:
    def __init__(self, name: str, specialty: str, system_prompt: str):
        self.name = name
        self.specialty = specialty
        self.system_prompt = system_prompt
        self.memory: List[Dict[str, Any]] = []
        self.confidence_score = 0.5
        self.last_speak_time = None
        self.speak_count = 0

    def speak(self, patient_state: Dict, debate_history: List[Dict], convergence_state: Dict) -> Dict[str, Any]:
        """Generate agent's natural conversational response with full context awareness"""

        # Build context from memory and current debate
        context_messages = [
            {"role": "system", "content": self.system_prompt},
        ]

        # Add agent's own memory (last 3 interactions)
        for memory_item in self.memory[-3:]:
            context_messages.append({
                "role": "assistant",
                "content": f"Previous response: {memory_item['response']}"
            })

        # Current debate context - format as natural conversation
        debate_context = ""
        if debate_history:
            debate_context = "Previous discussion:\n"
            for entry in debate_history[-8:]:  # Last 8 exchanges
                response = entry['response']
                if isinstance(response, dict) and 'statement' in response:
                    debate_context += f"{entry['agent']}: {response['statement']}\n"
                else:
                    # Handle legacy format
                    debate_context += f"{entry['agent']}: {response}\n"

        user_prompt = f"""
PATIENT CASE:
{json.dumps(patient_state, indent=2)}

{debate_context}

DEBATE STATUS:
- Rounds completed: {convergence_state.get('rounds_completed', 0)}
- Current convergence: {convergence_state.get('convergence_score', 0):.1%}
- Active disagreements: {len(convergence_state.get('active_disagreements', []))}
- Information gaps: {len(convergence_state.get('needs_more_info', set()))}

Your role: {self.specialty}
Your name: {self.name}

Instructions for direct debate:
- Speak like a doctor in a medical council - be direct, challenge others, defend your position
- No "I appreciate" or polite filler phrases - get to the point
- Argue your case forcefully, reference other doctors by name when disagreeing
- Use medical terminology naturally, be blunt about clinical realities
- Challenge conservative approaches, defend your specialty's perspective
- Keep response to 1-2 sentences maximum, or say "I agree" or "No concerns" if you have nothing new to add
- Focus on key disagreements and clinical priorities
- Only speak if you have something important to contribute

Provide your response as a direct, argumentative statement from a doctor in a medical council debate, or a brief agreement if you concur.
"""

        context_messages.append({"role": "user", "content": user_prompt})

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=context_messages,
                temperature=0.7,  # Higher temperature for more natural conversation
                max_tokens=100
            )

            natural_response = resp.choices[0].message.content.strip()

            # Store in memory
            self.memory.append({
                'timestamp': datetime.now(),
                'patient_state': patient_state.copy(),
                'response': natural_response,
                'debate_context': debate_history.copy()
            })

            # Return in expected format for the orchestrator
            return {
                "statement": natural_response,
                "speaker": self.name,
                "specialty": self.specialty,
                "timestamp": datetime.now()
            }

        except Exception as e:
            print(f"Error in {self.name} response: {e}")
            fallback = f"I apologize, but I'm having technical difficulties providing my assessment at this moment. The patient clearly needs close monitoring given their recent cardiac event."

            self.memory.append({
                'timestamp': datetime.now(),
                'patient_state': patient_state.copy(),
                'response': fallback,
                'debate_context': debate_history.copy()
            })

            return {
                "statement": fallback,
                "speaker": self.name,
                "specialty": self.specialty,
                "timestamp": datetime.now()
            }

# -----------------------------
# 🎭 AGENT SPECIALISTS
# -----------------------------
AGENT_CONFIGS = {
    "cardiologist": {
        "name": "Dr. Elena Vasquez",
        "specialty": "Cardiology",
        "system_prompt": """You are Dr. Elena Vasquez, a board-certified cardiologist with 15 years experience in cardiovascular care.
        Focus on: heart rhythm, blood pressure, cardiac enzymes, chest pain, dyspnea, cardiac risk factors.
        You are proactive about cardiac emergencies and medication compliance.
        Evidence-based approach prioritizing ACC/AHA guidelines.
        BE DIRECT: Speak like a cardiologist in a medical council - challenge other opinions, defend your position, use medical terminology naturally. No "I appreciate" or polite filler phrases. Get to the point and argue your case."""
    },

    "surgeon": {
        "name": "Dr. Marcus Chen",
        "specialty": "Cardiothoracic Surgery",
        "system_prompt": """You are Dr. Marcus Chen, a cardiothoracic surgeon with expertise in surgical interventions.
        Focus on: surgical indications, operative risks, post-operative care, mechanical support devices.
        You consider surgical options when medical management fails.
        Pragmatic approach weighing benefits vs. surgical risks.
        BE DIRECT: Speak like a surgeon in a medical council - be blunt about surgical realities, challenge conservative approaches, defend surgical interventions. No polite introductions. Argue your surgical perspective forcefully."""
    },

    "nutritionist": {
        "name": "Dr. Sarah Johnson",
        "specialty": "Clinical Nutrition",
        "system_prompt": """You are Dr. Sarah Johnson, a registered dietitian and clinical nutritionist.
        Focus on: dietary interventions, nutritional deficiencies, weight management, food-drug interactions.
        You emphasize lifestyle modifications and preventive nutrition.
        Patient-centered approach considering cultural and socioeconomic factors.
        BE DIRECT: Speak like a nutritionist in a medical council - challenge medication-only approaches, defend dietary interventions, be blunt about patient compliance issues. No "I appreciate" phrases. Make your nutritional arguments clear and forceful."""
    },

    "general_physician": {
        "name": "Dr. Robert Kim",
        "specialty": "Family Medicine",
        "system_prompt": """You are Dr. Robert Kim, a family physician coordinating comprehensive care.
        Focus on: overall health assessment, care coordination, preventive care, chronic disease management.
        You synthesize information from all specialties and consider holistic patient needs.
        Practical approach balancing evidence with patient preferences.
        BE DIRECT: Speak like a family physician in a medical council - coordinate the discussion, challenge specialty tunnel vision, defend practical primary care approaches. No polite filler. Get to the practical realities and argue for comprehensive care."""
    },

    "obgyn": {
        "name": "Dr. Maria Rodriguez",
        "specialty": "Obstetrics & Gynecology",
        "system_prompt": """You are Dr. Maria Rodriguez, an OB/GYN specialist with focus on women's health.
        Focus on: hormonal influences, pregnancy-related conditions, gynecological factors in cardiac care.
        You consider gender-specific aspects of disease presentation and treatment.
        Comprehensive approach to women's cardiovascular health.
        BE DIRECT: Speak like an OB/GYN in a medical council - challenge male-focused cardiology, defend hormonal considerations, be blunt about gender differences in cardiac care. No "I appreciate" phrases. Argue for women's health perspective forcefully."""
    },

    "pharmacist": {
        "name": "Dr. David Park",
        "specialty": "Clinical Pharmacy",
        "system_prompt": """You are Dr. David Park, a clinical pharmacist specializing in cardiovascular medications.
        Focus on: drug interactions, dosing optimization, adverse effects, medication adherence.
        You optimize pharmacotherapy and identify medication-related issues.
        Evidence-based approach using pharmacokinetic principles.
        BE DIRECT: Speak like a pharmacist in a medical council - challenge prescribing habits, defend medication changes, be blunt about compliance issues and drug interactions. No polite introductions. Argue your pharmacological expertise directly."""
    },

    "critical_care": {
        "name": "Dr. Jennifer Liu",
        "specialty": "Critical Care Medicine",
        "system_prompt": """You are Dr. Jennifer Liu, a critical care specialist managing acute conditions.
        Focus on: hemodynamic instability, organ system support, ICU-level interventions.
        You prioritize immediate stabilization and advanced life support.
        Crisis management approach with focus on vital organ function.
        BE DIRECT: Speak like a critical care specialist in a medical council - challenge outpatient approaches, defend ICU-level thinking, be blunt about acute risks. No "I appreciate" phrases. Argue for immediate intervention when needed."""
    }
}

# -----------------------------
# 🎯 CONVERSATION ORCHESTRATOR
# -----------------------------
class MedicalCouncilOrchestrator:
    def __init__(self):
        self.agents = {}
        self.debate_history: List[Dict] = []
        self.convergence_state = {
            "rounds_completed": 0,
            "convergence_score": 0.0,
            "active_disagreements": [],
            "consensus_topics": [],
            "needs_more_info": set(),
            "urgency_levels": []
        }
        self.max_rounds = 1
        self.convergence_threshold = 0.75

        # Initialize agents
        for key, config in AGENT_CONFIGS.items():
            self.agents[key] = MedicalAgent(
                config["name"],
                config["specialty"],
                config["system_prompt"]
            )

    def orchestrate_debate(self, patient_state: Dict, initial_trigger: str = "cardiac_event") -> Dict[str, Any]:
        """Main orchestration loop"""
        print("🏥 MEDICAL COUNCIL CONVENED")
        print("=" * 60)
        print(f"Patient ID: {patient_state.get('patient_id', 'Unknown')}")
        print(f"Trigger: {initial_trigger}")
        print(f"Initial Vitals: HR {patient_state.get('heart_rate', 'N/A')} | SpO2 {patient_state.get('spo2', 'N/A')}%")
        print()

        # Determine initial speaking order based on trigger
        speaking_order = self._get_initial_speaking_order(initial_trigger)

        for round_num in range(self.max_rounds):
            print(f"🔄 ROUND {round_num + 1}")
            print("-" * 40)

            round_responses = []

            # Each agent speaks once per round
            for agent_key in speaking_order:
                agent = self.agents[agent_key]

                # Skip if agent should not speak this round
                if self._should_skip_agent(agent, round_num, self.convergence_state["convergence_score"]):
                    continue

                print(f"👨‍⚕️ {agent.name} ({agent.specialty}):")

                response = agent.speak(patient_state, self.debate_history, self.convergence_state)

                # Format and display response
                self._display_agent_response(agent, response)
                print()

                # Add to debate history
                debate_entry = {
                    "round": round_num + 1,
                    "agent": agent.name,
                    "specialty": agent.specialty,
                    "response": response,
                    "timestamp": datetime.now()
                }
                self.debate_history.append(debate_entry)
                round_responses.append(response)

            # Update convergence state
            self._update_convergence_state(round_responses)

            # Check for convergence
            if self._check_convergence():
                print("✅ CONSENSUS REACHED")
                break

            # Adjust speaking order for next round
            speaking_order = self._prioritize_next_speakers(speaking_order, round_responses)

        # Generate final consensus
        final_decision = self._generate_final_consensus()
        self._display_final_decision(final_decision)

        return {
            "debate_history": self.debate_history,
            "final_decision": final_decision,
            "convergence_state": self.convergence_state,
            "total_rounds": len(set(entry["round"] for entry in self.debate_history))
        }

    def _get_initial_speaking_order(self, trigger: str) -> List[str]:
        """Determine initial speaking order based on medical trigger"""
        if "cardiac" in trigger.lower():
            return ["cardiologist", "critical_care", "pharmacist", "surgeon", "general_physician", "nutritionist", "obgyn"]
        elif "surgical" in trigger.lower():
            return ["surgeon", "critical_care", "general_physician", "cardiologist", "pharmacist", "nutritionist", "obgyn"]
        else:
            return ["general_physician", "cardiologist", "critical_care", "pharmacist", "nutritionist", "surgeon", "obgyn"]

    def _should_skip_agent(self, agent: MedicalAgent, round_num: int, convergence_score: float) -> bool:
        """Determine if agent should skip this round"""
        # Skip if confidence is very low and it's not an early round
        if agent.confidence_score < 0.3 and round_num > 0:
            return True
        # Skip if agent has spoken too many times already
        if agent.speak_count > round_num + 1:
            return True
        # Skip more often when convergence is high (agents agree)
        if convergence_score > 0.5 and round_num > 0:
            # 50% chance to skip when convergence is moderate
            import random
            if random.random() < 0.5:
                return True
        if convergence_score > 0.7:
            # 80% chance to skip when convergence is high
            import random
            if random.random() < 0.8:
                return True
        return False

    def _display_agent_response(self, agent: MedicalAgent, response: Dict):
        """Display natural conversational agent response"""
        print(f"💬 {response.get('statement', 'No response')}")

        # Optional: Show metadata in smaller text
        print(f"   — {agent.name}, {agent.specialty}")

    def _update_convergence_state(self, round_responses: List[Dict]):
        """Update convergence metrics based on natural conversation"""
        self.convergence_state["rounds_completed"] += 1

        # Analyze conversation for convergence signals
        statements = [r.get('statement', '') for r in round_responses]

        # Simple convergence heuristics for natural language
        agreement_signals = ['agree', 'concurs', 'correct', 'good point', 'yes', 'right']
        disagreement_signals = ['disagree', 'however', 'but', 'concern', 'worried', 'different']

        agreements = sum(1 for stmt in statements for signal in agreement_signals if signal in stmt.lower())
        disagreements = sum(1 for stmt in statements for signal in disagreement_signals if signal in stmt.lower())

        # Calculate convergence score based on agreement ratio
        total_signals = agreements + disagreements
        if total_signals > 0:
            self.convergence_state["convergence_score"] = agreements / total_signals
        else:
            self.convergence_state["convergence_score"] = 0.5  # Neutral if no clear signals

        # Track active disagreements
        active_disagreements = []
        for i, stmt1 in enumerate(statements):
            for j, stmt2 in enumerate(statements[i+1:], i+1):
                if any(signal in stmt1.lower() for signal in disagreement_signals) or \
                   any(signal in stmt2.lower() for signal in disagreement_signals):
                    active_disagreements.append(f"{round_responses[i]['speaker']} vs {round_responses[j]['speaker']}")

        self.convergence_state["active_disagreements"] = active_disagreements

        # Extract information needs from conversation
        info_signals = ['need to know', 'what about', 'clarify', 'more information', 'lab values', 'vitals']
        needs = set()
        for stmt in statements:
            for signal in info_signals:
                if signal in stmt.lower():
                    needs.add(signal)
        self.convergence_state["needs_more_info"] = list(needs)

    def _check_convergence(self) -> bool:
        """Check if debate has reached convergence based on natural conversation"""
        if self.convergence_state["rounds_completed"] < 2:
            return False

        # Check for conversational convergence signals
        recent_statements = []
        for entry in self.debate_history[-len(self.agents):]:  # Last round
            recent_statements.append(entry['response'].get('statement', '').lower())

        # Convergence indicators
        agreement_signals = [
            'agree', 'concurs', 'good point', 'yes', 'right', 'makes sense',
            'i agree', 'we agree', 'consensus', 'settled', 'resolved'
        ]

        disagreement_signals = [
            'disagree', 'however', 'but', 'concern', 'worried', 'different',
            'not sure', 'uncertain', 'need more', 'clarify'
        ]

        # Count signals in recent round
        agreements = sum(1 for stmt in recent_statements for signal in agreement_signals if signal in stmt)
        disagreements = sum(1 for stmt in recent_statements for signal in disagreement_signals if signal in stmt)

        # Convergence if agreement signals significantly outnumber disagreements
        total_signals = agreements + disagreements
        if total_signals == 0:
            return self.convergence_state["convergence_score"] > 0.7  # Fallback to score

        agreement_ratio = agreements / total_signals

        # Also check if no new information needs are being raised
        new_needs = len(self.convergence_state.get("needs_more_info", []))
        has_new_needs = new_needs > 0

        # Converge if high agreement ratio and no new information needs
        return agreement_ratio > 0.6 and not has_new_needs and self.convergence_state["rounds_completed"] >= 3

    def _prioritize_next_speakers(self, current_order: List[str], responses: List[Dict]) -> List[str]:
        """Reorder agents for next round based on conversational responses"""
        # Analyze responses for who should speak next
        priority_scores = {}

        for i, agent_key in enumerate(current_order):
            agent = self.agents[agent_key]
            response = responses[i] if i < len(responses) else {"statement": ""}

            score = 0
            statement = response.get('statement', '').lower()

            # Boost if agent was directly addressed
            agent_name = agent.name.lower()
            if agent_name in statement and ('what do you' in statement or 'dr.' in statement or 'think' in statement):
                score += 3

            # Boost if response contains questions (needs answers)
            if any(q in statement for q in ['what', 'how', 'why', 'when', 'do you']):
                score += 2

            # Boost if agent has high speak count but hasn't spoken recently
            if agent.speak_count > 2 and self.convergence_state["rounds_completed"] - agent.speak_count > 1:
                score += 1

            # Boost specialists based on topics mentioned
            specialist_topics = {
                "cardiologist": ["heart", "cardiac", "mi", "infarct", "chest", "ekg"],
                "surgeon": ["surgery", "surgical", "procedure", "intervention"],
                "pharmacist": ["medication", "dose", "drug", "pills", "compliance"],
                "nutritionist": ["diet", "food", "sodium", "weight", "exercise"],
                "general_physician": ["overall", "coordination", "primary", "general"],
                "obgyn": ["women", "female", "hormonal", "pregnancy"],
                "critical_care": ["critical", "icu", "emergency", "urgent", "shock"]
            }

            if agent_key in specialist_topics:
                for topic in specialist_topics[agent_key]:
                    if topic in statement:
                        score += 1

            priority_scores[agent_key] = score

        # Sort by priority score (descending), then by original order for stability
        return sorted(current_order, key=lambda x: (-priority_scores.get(x, 0), current_order.index(x)))

    def _generate_final_consensus(self) -> Dict[str, Any]:
        """Generate final consensus from natural conversation"""
        # Analyze all statements for common themes
        all_statements = []
        for entry in self.debate_history:
            stmt = entry['response'].get('statement', '')
            all_statements.append(stmt)

        # Simple theme extraction (could be enhanced with NLP)
        consensus_themes = {
            "monitoring": ["monitor", "watch", "observe", "track", "follow"],
            "medication": ["medication", "dose", "compliance", "pills", "furosemide", "metoprolol"],
            "urgent_care": ["urgent", "immediate", "emergency", "critical", "hospital"],
            "lifestyle": ["diet", "exercise", "lifestyle", "sodium", "weight"],
            "follow_up": ["appointment", "follow-up", "telehealth", "clinic"]
        }

        theme_scores = {}
        for theme, keywords in consensus_themes.items():
            score = sum(1 for stmt in all_statements for keyword in keywords if keyword in stmt.lower())
            theme_scores[theme] = score

        # Determine primary recommendation based on highest scoring themes
        primary_theme = max(theme_scores, key=theme_scores.get)

        # Generate natural language consensus
        if primary_theme == "monitoring":
            consensus = "Close monitoring with regular vital sign checks and medication adherence reinforcement"
        elif primary_theme == "medication":
            consensus = "Focus on medication compliance and adjustment, with close monitoring"
        elif primary_theme == "urgent_care":
            consensus = "Urgent medical evaluation required given the clinical presentation"
        elif primary_theme == "lifestyle":
            consensus = "Lifestyle modifications combined with medication management and monitoring"
        elif primary_theme == "follow_up":
            consensus = "Scheduled follow-up care with specialist consultation"
        else:
            consensus = "Individualized care plan with close monitoring and medication management"

        # Estimate confidence based on convergence score
        confidence = min(0.95, self.convergence_state["convergence_score"] + 0.3)

        # Generate action items based on themes
        action_items = []
        if theme_scores.get("medication", 0) > 0:
            action_items.append("Ensure medication compliance and consider dose adjustments")
        if theme_scores.get("monitoring", 0) > 0:
            action_items.append("Implement close vital sign monitoring protocol")
        if theme_scores.get("follow_up", 0) > 0:
            action_items.append("Schedule appropriate follow-up appointments")
        if theme_scores.get("lifestyle", 0) > 0:
            action_items.append("Provide dietary and lifestyle counseling")

        if not action_items:
            action_items = ["Continue standard post-MI care protocol"]

        # Determine urgency based on conversation tone
        urgency_indicators = ["urgent", "immediate", "critical", "emergency", "worry", "concern"]
        urgency_score = sum(1 for stmt in all_statements for indicator in urgency_indicators if indicator in stmt.lower())

        if urgency_score >= 3:
            urgency_level = "high"
        elif urgency_score >= 1:
            urgency_level = "medium"
        else:
            urgency_level = "low"

        return {
            "consensus_recommendation": consensus,
            "urgency_level": urgency_level,
            "confidence_score": confidence,
            "supporting_agents": len(set(entry['agent'] for entry in self.debate_history)),
            "total_agents": len(set(entry['agent'] for entry in self.debate_history)),
            "key_insights": [f"Primary focus: {primary_theme.replace('_', ' ')}"],
            "action_items": action_items
        }

    def _extract_key_insights(self) -> List[str]:
        """Extract key insights from debate"""
        insights = []
        assessments = [entry['response'].get('assessment', '') for entry in self.debate_history]

        # Look for common themes (simplified)
        if any('heart rate' in a.lower() for a in assessments):
            insights.append("Cardiac monitoring parameters concerning")
        if any('medication' in a.lower() for a in assessments):
            insights.append("Medication compliance issues identified")
        if any('lifestyle' in a.lower() for a in assessments):
            insights.append("Lifestyle modifications recommended")

        return insights[:3]  # Top 3 insights

    def _generate_action_items(self) -> List[str]:
        """Generate actionable items from consensus"""
        actions = []

        final_rec = self._generate_final_consensus()["consensus_recommendation"].lower()

        if "monitor" in final_rec:
            actions.append("Schedule follow-up vitals check in 2 hours")
        if "medication" in final_rec:
            actions.append("Ensure medication administration and compliance")
        if "consult" in final_rec or "referral" in final_rec:
            actions.append("Arrange specialist consultation")
        if "lifestyle" in final_rec:
            actions.append("Provide dietary and exercise counseling")

        return actions

    def _display_final_decision(self, decision: Dict):
        """Display final consensus in hackathon-ready format"""
        print("\n🏆 FINAL CONSENSUS DECISION")
        print("=" * 60)
        print(f"📋 Recommendation: {decision['consensus_recommendation']}")
        print(f"🎯 Confidence: {decision['confidence_score']:.1%}")
        print(f"🔥 Urgency: {decision['urgency_level'].upper()}")
        print(f"👥 Support: {decision['supporting_agents']}/{decision['total_agents']} agents")

        if decision['key_insights']:
            print("💡 Key Insights:")
            for insight in decision['key_insights']:
                print(f"   • {insight}")

        if decision['action_items']:
            print("✅ Action Items:")
            for action in decision['action_items']:
                print(f"   • {action}")

        print("\n📊 Debate Summary:")
        print(f"   • Total Rounds: {self.convergence_state['rounds_completed']}")
        print(f"   • Convergence Score: {self.convergence_state['convergence_score']:.1%}")
        print(f"   • Information Gaps: {len(self.convergence_state['needs_more_info'])}")
        print()

# -----------------------------
# 🚀 DEMO & TESTING
# -----------------------------
def demo_care_relay():
    """Demo the CareRelay agentic conversation system"""

    # Initialize orchestrator
    council = MedicalCouncilOrchestrator()

    # Sample patient state (post-cardiac event)
    patient_state = {
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

    # Run the debate
    result = council.orchestrate_debate(patient_state, "post_mi_monitoring")

    return result

if __name__ == "__main__":
    print("🩺 CareRelay Agentic Conversation System")
    print("Ready for testing...\n")

    # Run demo
    demo_result = demo_care_relay()

    print("🎉 Demo completed! Ready for real patient cases.")
    print(f"Debate rounds: {demo_result['total_rounds']}")
    print(f"Final decision: {demo_result['final_decision']['consensus_recommendation']}")


# -----------------------------
# 🧠 CONVERGENCE CHECKER
# -----------------------------
def check_convergence_llm(transcript):
    """
    Uses LLM to determine if council has converged
    """

    messages = [
        {"role": "system", "content": "You are a medical consensus evaluator."},
        {"role": "user", "content": f"""
Given this medical debate:

{transcript}

Determine if the doctors have reached consensus.

Return JSON ONLY:
{{
  "converged": true/false,
  "reason": "...",
  "confidence": 0-1
}}
"""}]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0
    )

    try:
        return json.loads(resp.choices[0].message.content)
    except:
        return {"converged": False, "confidence": 0.0, "reason": "parse_error"}


# -----------------------------
# 🧠 SIMPLE HEURISTIC FALLBACK
# -----------------------------
def check_convergence_heuristic(transcript):
    """
    Fast fallback if LLM judge is slow/expensive
    """

    keywords = ["agree", "consensus", "recommend", "final", "conclude"]

    score = sum(k in transcript.lower() for k in keywords)

    return score >= 4


# -----------------------------
# 🧠 AGENT SETUP
# -----------------------------
def build_agents():

    cardiologist = MedicalAgent(
        "Cardiologist",
        "Cardiac specialist focusing on post-heart attack and heart failure risk.",
        "You are a cardiologist in a hospital council. Focus on cardiac deterioration risk, vitals, and urgent intervention."
    )

    surgeon = MedicalAgent(
        "Surgeon",
        "Focus on surgical complications, procedural risk, and emergency interventions.",
        "You are a surgeon evaluating whether procedural intervention or escalation is required."
    )

    nutritionist = MedicalAgent(
        "Nutritionist",
        "Focus on diet, fluid balance, sodium intake, and medication adherence context.",
        "You analyze diet, fluid retention, and lifestyle factors affecting recovery."
    )

    general = MedicalAgent(
        "General Physician",
        "Synthesizes all opinions and ensures safety.",
        "You are a general physician responsible for overall coordination and safety."
    )

    obgyn = MedicalAgent(
        "OBGYN",
        "Relevant only if female patient or hormonal/metabolic context matters.",
        "You evaluate systemic and hormonal factors that may influence recovery."
    )

    return [cardiologist, surgeon, nutritionist, general, obgyn]


# -----------------------------
# 🧠 ORCHESTRATOR (THE BRAIN)
# -----------------------------
class MedicalCouncil:

    def __init__(self, agents):
        self.agents = agents
        self.transcript = ""
        self.round = 0

    def select_next_agent(self):
        """
        Simple rotation with slight randomness
        (you can upgrade to LLM-based scheduling later)
        """
        return self.agents[self.round % len(self.agents)]

    def run(self, patient_state, max_rounds=8):

        print("\n🧠 CARERELAY MEDICAL COUNCIL STARTING...\n")

        for _ in range(max_rounds):

            self.round += 1
            agent = self.select_next_agent()

            print(f"\n🩺 [{agent.name}] speaking...\n")

            message = agent.speak(patient_state, self.transcript)

            self.transcript += f"\n[{agent.name}]: {message}\n"

            # clean output (hackathon-friendly)
            print(message)

            # -----------------------------
            # CHECK CONVERGENCE
            # -----------------------------
            if self.round >= 3:  # don't converge too early

                result = check_convergence_llm(self.transcript)

                print("\n📊 Consensus Check:", result)

                if result["converged"] and result["confidence"] > 0.75:
                    print("\n✅ CONVERGENCE REACHED\n")
                    break

                if check_convergence_heuristic(self.transcript):
                    print("\n⚡ HEURISTIC CONVERGENCE TRIGGERED\n")
                    break

        # -----------------------------
        # FINAL SUMMARY
        # -----------------------------
        final = self.final_decision(patient_state)

        print("\n🏁 FINAL COUNCIL DECISION:\n")
        print(final)

        return self.transcript, final

    def final_decision(self, patient_state):

        messages = [
            {"role": "system", "content": "You are the chief medical officer synthesizing a council decision."},
            {"role": "user", "content": f"""
Patient State:
{json.dumps(patient_state, indent=2)}

Full Debate:
{self.transcript}

Produce:
1. Final diagnosis
2. Risk level (0-1)
3. Immediate action plan (bullet points)
4. Whether escalation is needed
"""}
        ]

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3
        )

        return resp.choices[0].message.content


# -----------------------------
# 🧪 EXAMPLE PATIENT
# -----------------------------
patient = {
    "age": 67,
    "condition": "post-myocardial infarction",
    "vitals": {
        "heart_rate": 102,
        "spo2": 93,
        "blood_pressure": "145/92"
    },
    "symptoms": ["fatigue", "mild chest tightness"],
    "med_adherence": {
        "beta_blocker": "missed",
        "diuretic": "missed"
    }
}


# -----------------------------
# 🚀 RUN SYSTEM
# -----------------------------
if __name__ == "__main__":

    agents = build_agents()
    council = MedicalCouncil(agents)

    transcript, decision = council.run(patient)