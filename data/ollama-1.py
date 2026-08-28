import time
from ollama import chat
import json


# ============================================================
# DATABASE VOCABULARY
# ============================================================

SPECIALTIES = [
    "Anaesthesiology",
    "Cardiology",
    "Cardiothoracic Surgery",
    "Critical Care",
    "Dentistry",
    "Dermatology",
    "Diabetology",
    "Emergency Medicine",
    "Endocrinology",
    "ENT",
    "General Medicine",
    "Geriatrics",
    "Maxillofacial Surgery",
    "Medical Oncology",
    "Nephrology",
    "Neurosurgery",
    "Neurology",
    "Nuclear Medicine",
    "Obstetrics and Gynaecology",
    "Ophthalmology",
    "Orthopaedics",
    "Paediatric Surgery",
    "Physiotherapy",
    "Plastic Surgery",
    "Psychiatry",
    "Psychology",
    "Pulmonology",
    "Radiation Oncology",
    "Gastroenterology",
    "Surgical Oncology",
    "Urology"
]


PROCEDURES = [
    "Spine Surgery",
    "Bariatric Surgery",
    "Endoscopy",
    "Laparoscopic Surgery",
    "Joint Replacement",
    "IVF",
    "Arthroscopy",
    "Cataract Surgery",
    "Dialysis",
    "Liver Transplant",
    "Burns Treatment",
    "Limb Reconstruction",
    "Skull Base Surgery",
    "Coblation Tonsillectomy and Adenoidectomy",
    "Ear Surgery",
    "Endolaryngeal Surgery",
    "Endoscopic Dacryocystorhinostomy",
    "Sinus Surgery",
    "Stapedectomy"
]


# Services will be added after we finalize the Services column.
SERVICES = []


# ============================================================
# QWEN PROMPT
# ============================================================

def extract_medical_intent(user_input):

    prompt = f"""
You are a medical-intent extraction system for a hospital
recommendation application.

Your ONLY task is to extract the medical requirements from
the user's description.

Do NOT diagnose the patient.
Do NOT recommend a hospital.
Do NOT give medical advice.
Do NOT invent medical specialties or procedures.

You MUST use the terminology from the supplied database lists.

AVAILABLE SPECIALTIES:
{SPECIALTIES}

AVAILABLE PROCEDURES:
{PROCEDURES}

AVAILABLE SERVICES:
{SERVICES}

Return ONLY valid JSON in exactly this format:

{{
    "specialty": null,
    "procedure": null,
    "service": null,
    "emergency": false
}}

Rules:

1. If a relevant specialty is known, return the closest
   specialty from AVAILABLE SPECIALTIES.

2. If the user mentions a specific procedure, return it
   only if it exists in AVAILABLE PROCEDURES.

3. If the exact procedure is not in AVAILABLE PROCEDURES,
   return null for procedure. Still identify the relevant
   specialty if possible.

4. Do not invent database entries.

5. If multiple specialties are relevant, return the most
   relevant one.

6. Set emergency to true only when the user explicitly
   describes an emergency or urgent life-threatening
   situation.

7. Return ONLY JSON. No explanation.

USER INPUT:
{user_input}
"""

    start_time = time.perf_counter()

    response = chat(
        model="qwen3:8b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # Stop timer
    end_time = time.perf_counter()

    runtime = end_time - start_time

    print(f"\nQwen runtime: {runtime:.2f} seconds")

    content = response["message"]["content"].strip()

    # Remove markdown code fences if Qwen adds them
    if content.startswith("```"):
        content = content.replace("```json", "")
        content = content.replace("```", "")
        content = content.strip()

    try:
        return json.loads(content)

    except json.JSONDecodeError:
        print("Qwen returned invalid JSON:")
        print(content)

        return None


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    user_input = input("Describe your medical requirement: ")

    result = extract_medical_intent(user_input)

    print("\nExtracted information:")

    if result:
        print(json.dumps(result, indent=4))
    else:
        print("Could not extract information.")
