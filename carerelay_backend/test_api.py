#!/usr/bin/env python3
"""
CareRelay API Test Script
Demonstrates the core functionality
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_api():
    print("🩺 CareRelay API Test")
    print("=" * 50)

    # Test root endpoint
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ API Status: {response.json()}")
    except:
        print("❌ API not running. Start with: python main.py")

    # Test vitals ingestion
    patient_id = "test_patient_001"
    vitals_data = {
        "heart_rate": 98.0,
        "spo2": 93.0,
        "blood_pressure": "140/90",
        "timestamp": datetime.now().isoformat()
    }

    try:
        response = requests.post(
            f"{BASE_URL}/ingest_vitals/{patient_id}",
            json=vitals_data
        )
        print(f"✅ Vitals ingested: {response.json()}")
    except Exception as e:
        print(f"❌ Vitals ingestion failed: {e}")

    # Test voice check-in
    checkin_transcript = "Hello, I've been feeling a bit breathless today and missed my morning medication."

    try:
        response = requests.post(
            f"{BASE_URL}/voice_checkin/{patient_id}",
            json={"transcript": checkin_transcript}
        )
        result = response.json()
        print(f"✅ Voice check-in processed: {result['needs_council']}")
    except Exception as e:
        print(f"❌ Voice check-in failed: {e}")

    # Test risk simulation
    try:
        response = requests.post(f"{BASE_URL}/simulate_risk/{patient_id}")
        risk = response.json()
        print(f"✅ Risk assessment: {risk['trajectory']} (score: {risk['risk_score']:.2f})")
    except Exception as e:
        print(f"❌ Risk simulation failed: {e}")

    print("\n🎯 CareRelay backend is ready for integration!")
    print("Next steps:")
    print("- Add API keys for Anthropic, ElevenLabs, Twilio")
    print("- Implement real integrations")
    print("- Add authentication with Auth0")
    print("- Build the React frontend")

if __name__ == "__main__":
    test_api()