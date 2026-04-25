import requests

response = requests.post("http://localhost:8000/analyze", json={
    "patient_id": "CR-002",
    "age": 72,
    "gender": "male",
    "diagnosis": "Congestive heart failure with reduced ejection fraction (HFrEF), Type 2 diabetes",
    "heart_rate": 112,
    "spo2": 91,
    "blood_pressure": "158/96",
    "temperature": 99.1,
    "symptoms": ["severe leg swelling", "waking up at night to breathe", "gained 4 lbs in 2 days"],
    "medications": [
        {"name": "Lisinopril", "dose": "10mg daily", "compliance": "compliant"},
        {"name": "Carvedilol", "dose": "12.5mg BID", "compliance": "compliant"},
        {"name": "Spironolactone", "dose": "25mg daily", "compliance": "missed doses"},
        {"name": "Metformin", "dose": "500mg BID", "compliance": "compliant"}
    ],
    "lab_results": {
        "BNP": 890,
        "creatinine": 1.6,
        "potassium": 3.2,
        "HbA1c": 7.8,
        "ejection_fraction": 32
    },
    "lifestyle": {
        "diet": "high sodium, fast food daily",
        "fluid_intake": "unrestricted",
        "exercise": "none, too short of breath"
    },
    "trigger": "heart_failure_decompensation"
})

result = response.json()
print("DECISION:", result["decision"])
print("URGENCY:", result["urgency_level"])
print("\nDOCTOR REPORT:", result["doctor_report"])
print("\nACTIONS:")
for action in result["actions"]:
    print(f"  - {action}")