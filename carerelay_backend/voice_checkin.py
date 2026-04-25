# CareRelay Voice Check-in Conversation Script
# Designed for ElevenLabs TTS integration

CHECKIN_SCRIPT = {
    "greeting": "Hello! This is CareRelay calling. How are you feeling today?",

    "follow_up_questions": [
        "On a scale of 1 to 10, with 10 being the best you've ever felt, how would you rate your energy level today?",
        "Have you noticed any shortness of breath, chest pain, or unusual fatigue in the last 24 hours?",
        "How has your sleep been? Any trouble sleeping or waking up feeling rested?",
        "Have you taken all your medications as prescribed today?",
        "Is there anything else you'd like to tell me about how you're feeling or any concerns you have?"
    ],

    "medication_reminder": "I notice you might have missed your {medication_name}. Would you like me to help remind your caregiver to administer it?",

    "concern_acknowledgment": "Thank you for sharing that. It sounds like {symptom}. I'll make sure your care team reviews this right away.",

    "closing_positive": "Thank you for checking in today. Your care team will review everything and follow up if needed. Take care!",

    "closing_concern": "Thank you for your honesty about how you're feeling. Your care team is reviewing your information now and may reach out soon. Please don't hesitate to call if anything changes.",

    "emergency_guidance": "If you're experiencing severe chest pain, difficulty breathing, or feel like you need immediate help, please call emergency services at 911 right away."
}

# Conversation flow logic
def generate_checkin_script(patient_context):
    """
    Generate personalized check-in script based on patient history
    """
    script = [CHECKIN_SCRIPT["greeting"]]

    # Add personalized questions based on recent vitals/symptoms
    if patient_context.get("recent_symptoms"):
        script.append("I wanted to follow up on the symptoms you mentioned last time...")

    # Add medication check
    if patient_context.get("missed_meds"):
        for med in patient_context["missed_meds"]:
            script.append(CHECKIN_SCRIPT["medication_reminder"].format(medication_name=med))

    # Add standard questions
    script.extend(CHECKIN_SCRIPT["follow_up_questions"])

    # Add closing based on risk assessment
    if patient_context.get("high_risk", False):
        script.append(CHECKIN_SCRIPT["closing_concern"])
    else:
        script.append(CHECKIN_SCRIPT["closing_positive"])

    return script

# ElevenLabs integration helper
def prepare_tts_request(script_lines, voice_settings):
    """
    Prepare request for ElevenLabs API
    """
    full_script = " ".join(script_lines)

    return {
        "text": full_script,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": voice_settings,
        "output_format": "mp3_22050_32"
    }

# Voice settings for compassionate, professional tone
VOICE_SETTINGS = {
    "stability": 0.5,
    "similarity_boost": 0.8,
    "style": 0.5,
    "use_speaker_boost": True
}