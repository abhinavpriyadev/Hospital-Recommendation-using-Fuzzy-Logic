import os
import pandas as pd
import re
import json
import time
from ollama import chat


# ============================================================
# SETTINGS
# ============================================================

# Always work from the folder containing this Python file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

CSV_FILE = "hospitals-final-cleaned.csv"
MODEL_NAME = "qwen3:8b"


# ============================================================
# LOAD HOSPITAL DATABASE
# ============================================================

print("Loading hospital database...")

df = pd.read_csv(
    CSV_FILE,
    dtype=str
).fillna("")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Specialties",
    "Procedures",
    "Services"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Column '{column}' was not found in the CSV.\n"
            f"Available columns:\n{list(df.columns)}"
        )


# ============================================================
# EXTRACT UNIQUE VALUES FROM A COLUMN
# ============================================================

def get_unique_values(column):

    values = set()

    for cell in df[column]:

        # Split entries separated by ;
        items = str(cell).split(";")

        for item in items:

            # Remove newlines and repeated spaces
            item = re.sub(r"\s+", " ", item).strip()

            if item:
                values.add(item)

    return sorted(values)


# ============================================================
# CREATE DATABASE VOCABULARY
# ============================================================

SPECIALTIES = get_unique_values("Specialties")
PROCEDURES = get_unique_values("Procedures")
SERVICES = get_unique_values("Services")


# Convert lists into text for Qwen

SPECIALTIES_TEXT = "\n".join(
    f"- {item}"
    for item in SPECIALTIES
)

PROCEDURES_TEXT = "\n".join(
    f"- {item}"
    for item in PROCEDURES
)

SERVICES_TEXT = "\n".join(
    f"- {item}"
    for item in SERVICES
)


# ============================================================
# DISPLAY DATABASE INFORMATION
# ============================================================

print(f"Database loaded successfully.")
print(f"Hospitals: {len(df)}")
print(f"Specialties: {len(SPECIALTIES)}")
print(f"Procedures: {len(PROCEDURES)}")
print(f"Services: {len(SERVICES)}")

print("\nQwen Hospital Input Parser")
print("Type 'exit' to quit.")
print("=" * 60)


# ============================================================
# QWEN INPUT PROCESSOR
# ============================================================

def extract_medical_intent(user_input):

    prompt = f"""
You are the medical intent classification component of a
hospital recommendation system.

Your job is to convert a patient's natural-language request
into structured information that can be used to search a
hospital database.

You are NOT diagnosing the patient.

============================================================
AVAILABLE SPECIALTIES
============================================================

{SPECIALTIES_TEXT}

============================================================
AVAILABLE PROCEDURES
============================================================

{PROCEDURES_TEXT}

============================================================
AVAILABLE SERVICES
============================================================

{SERVICES_TEXT}

============================================================
RULES
============================================================

1. Use terminology from the supplied lists whenever possible.

2. If the patient explicitly mentions a specialty, identify
   the matching specialty from AVAILABLE SPECIALTIES.

3. If the patient explicitly mentions a procedure that exists
   in AVAILABLE PROCEDURES, return that exact procedure.

4. If the patient mentions a procedure that does NOT exist in
   AVAILABLE PROCEDURES, return null for procedure.

5. If the patient does not explicitly mention a specialty,
   infer the most appropriate specialty from the patient's
   symptoms, condition, or request.

6. When inferring a specialty, you MUST select one from
   AVAILABLE SPECIALTIES.

7. NEVER invent a specialty.

8. NEVER invent a procedure.

9. NEVER invent a service.

10. If no reasonable specialty from AVAILABLE SPECIALTIES
    can be identified or inferred, return null.

11. The procedure field MUST contain either:
    - a procedure from AVAILABLE PROCEDURES
    - or null

12. The service field MUST contain either:
    - a service from AVAILABLE SERVICES
    - or null

13. Set emergency to true only when the patient's input
    indicates a potentially urgent or emergency situation.

14. Do not diagnose the patient.

15. Do not provide medical advice.

16. Do not explain your reasoning.

17. Return ONLY valid JSON.

============================================================
OUTPUT FORMAT
============================================================

Return exactly this structure:

{{
    "specialty": null,
    "procedure": null,
    "service": null,
    "emergency": false
}}

============================================================
EXAMPLES
============================================================

Patient:
I need a cardiologist

Output:
{{
    "specialty": "Cardiology",
    "procedure": null,
    "service": null,
    "emergency": false
}}

Patient:
I need a knee replacement

Output:
{{
    "specialty": "Orthopaedics",
    "procedure": "Joint Replacement",
    "service": null,
    "emergency": false
}}

Patient:
I need ACL reconstruction

Output:
{{
    "specialty": "Orthopaedics",
    "procedure": null,
    "service": null,
    "emergency": false
}}

Patient:
I have chest pain

Output:
{{
    "specialty": "Cardiology",
    "procedure": null,
    "service": null,
    "emergency": true
}}

Patient:
I need cataract surgery

Output:
{{
    "specialty": "Ophthalmology",
    "procedure": "Cataract Surgery",
    "service": null,
    "emergency": false
}}

Patient:
I need dialysis

Output:
{{
    "specialty": "Nephrology",
    "procedure": "Dialysis",
    "service": null,
    "emergency": false
}}

============================================================
PATIENT INPUT
============================================================

{user_input}
"""

    start_time = time.perf_counter()

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        think=False,
        keep_alive=-1,
        options={
            "temperature": 0,
            "num_predict": 100
        }
    )

    runtime = time.perf_counter() - start_time

    output = response["message"]["content"].strip()

    return output, runtime


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    user_input = input("\nPatient: ").strip()

    if user_input.lower() == "exit":

        print("\nExiting...")
        break

    if not user_input:
        continue

    try:

        result, runtime = extract_medical_intent(user_input)

        print("\nQwen output:")

        # Try to interpret response as JSON
        try:

            parsed = json.loads(result)

            print(
                json.dumps(
                    parsed,
                    indent=4,
                    ensure_ascii=False
                )
            )

        except json.JSONDecodeError:

            print(result)

        print(f"\nRuntime: {runtime:.2f} seconds")
        print("-" * 60)

    except Exception as e:

        print("\nERROR:")
        print(e)
        print("-" * 60)
