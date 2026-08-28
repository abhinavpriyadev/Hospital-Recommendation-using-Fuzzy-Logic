import os
import sys
import subprocess
import time
import json
import re
import pandas as pd
from ollama import chat

# SETTINGS
MODEL_NAME = "qwen3:8b"


def ensure_ollama_running():

    # Check whether Ollama server is already running
    try:
        import urllib.request

        urllib.request.urlopen(
            "http://127.0.0.1:11434/api/tags",
            timeout=2
        )

        print("Ollama server is already running.")
        return

    except Exception:
        pass

    print("Ollama server is not running.")
    print("Starting Ollama...")

    ollama_path = os.path.expandvars(
        r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
    )

    subprocess.Popen(
        [ollama_path, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW
    )

    # Give Ollama time to start
    for _ in range(20):

        try:
            import urllib.request

            urllib.request.urlopen(
                "http://127.0.0.1:11434/api/tags",
                timeout=2
            )

            print("Ollama server started.")
            return

        except Exception:
            time.sleep(0.5)

    raise RuntimeError(
        "Ollama server could not be started."
    )
ensure_ollama_running()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_DIR = os.path.dirname(BASE_DIR)

CSV_FILE = os.path.join(
    PROJECT_DIR,
    "data",
    "hospitals-final-cleaned.csv"
)
# LOAD HOSPITAL DATABASE

df = pd.read_csv(
    CSV_FILE,
    dtype=str
).fillna("")

# EXTRACT UNIQUE VALUES FROM A COLUMN

def get_unique_values(column):

    values = set()

    for cell in df[column]:

        items = str(cell).split(";")

        for item in items:

            item = re.sub(
                r"\s+",
                " ",
                item
            ).strip()

            if item:
                values.add(item)

    return sorted(values)



SPECIALTIES = get_unique_values("Specialties")
PROCEDURES = get_unique_values("Procedures")
SERVICES = get_unique_values("Services")


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

# DATABASE INFORMATION


print("Database loaded successfully.")

print(f"Hospitals:   {len(df)}")
print(f"Specialties: {len(SPECIALTIES)}")
print(f"Procedures:  {len(PROCEDURES)}")
print(f"Services:    {len(SERVICES)}")

print("\nQwen Hospital Input Parser")
print("Type 'exit' to quit.")

print("=" * 60)


# MEDICAL INTENT EXTRACTION


def extract_medical_intent(user_input):

    prompt = f"""
You are the medical intent classification component of a
hospital recommendation system.

Your job is to understand the patient's request and convert it
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
IMPORTANT RULES
============================================================

1. The lists above come directly from the hospital database.

2. If the patient explicitly mentions a specialty, choose the
   matching specialty from AVAILABLE SPECIALTIES.

3. If the patient explicitly mentions a procedure and that
   procedure exists in AVAILABLE PROCEDURES, select it.

4. If the patient mentions a procedure that does not exist in
   AVAILABLE PROCEDURES, return null for procedure.

5. If the patient does not explicitly mention a specialty,
   infer the most appropriate specialty from the patient's
   symptoms or request.

6. When inferring a specialty, you MUST choose it from
   AVAILABLE SPECIALTIES.

7. NEVER invent a specialty.

8. NEVER invent a procedure.

9. NEVER invent a service.

10. If no appropriate specialty exists in AVAILABLE
    SPECIALTIES, return null.

11. Set emergency to true when the patient's input indicates
    a potentially urgent or emergency situation.

12. Do not diagnose the patient.

13. Do not provide medical advice.

14. Do not explain your reasoning.

15. Return ONLY valid JSON.

============================================================
OUTPUT FORMAT
============================================================

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

        # Disable Qwen's visible thinking output
        think=False,

        # Keep model loaded between requests
        keep_alive=-1,

        options={
            "temperature": 0,
            "num_predict": 100
        }
    )


    runtime = time.perf_counter() - start_time


    output = response[
        "message"
    ][
        "content"
    ].strip()


    return output, runtime


while True:

    user_input = input(
        "\nPatient: "
    ).strip()



    if user_input.lower() == "exit":

        print("\nExiting...")
        break


    if not user_input:
        continue


    try:

        result, runtime = extract_medical_intent(
            user_input
        )


        print("\nQwen output:")


        
        # PARSE JSON

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


        # RUNTIME


        print(
            f"\nRuntime: {runtime:.2f} seconds"
        )

        print("-" * 60)


    except Exception as e:

        print("\nERROR:")
        print(e)

        print("-" * 60)

