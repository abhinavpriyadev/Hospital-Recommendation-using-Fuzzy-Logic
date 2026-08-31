import os
import sys
import subprocess
import time
import json
import re
import pandas as pd
from ollama import chat


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "qwen3:8b"


# ============================================================
# START OLLAMA AUTOMATICALLY
# ============================================================

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


# ============================================================
# FILE PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    BASE_DIR
)

CSV_FILE = os.path.join(
    PROJECT_DIR,
    "data",
    "hospitals-final-cleaned.csv"
)


# ============================================================
# LOAD HOSPITAL DATABASE
# ============================================================

df = pd.read_csv(
    CSV_FILE,
    dtype=str
).fillna("")


# ============================================================
# EXTRACT UNIQUE VALUES FROM A COLUMN
# ============================================================

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


SPECIALTIES = get_unique_values(
    "Specialties"
)

PROCEDURES = get_unique_values(
    "Procedures"
)

SERVICES = get_unique_values(
    "Services"
)


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
# DATABASE INFORMATION
# ============================================================

print("Database loaded successfully.")

print(
    f"Hospitals:   {len(df)}"
)

print(
    f"Specialties: {len(SPECIALTIES)}"
)

print(
    f"Procedures:  {len(PROCEDURES)}"
)

print(
    f"Services:    {len(SERVICES)}"
)


print(
    "\nQwen Hospital Input Parser"
)

print(
    "Type 'exit' to quit."
)

print(
    "=" * 60
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(text):

    if pd.isna(text):
        return ""

    text = str(text)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip().lower()


# ============================================================
# CHECK WHETHER A HOSPITAL HAS A VALUE
# ============================================================

def contains_value(
    cell,
    requested_value
):

    if not requested_value:
        return False

    requested_value = normalize(
        requested_value
    )

    items = str(cell).split(";")

    for item in items:

        if normalize(item) == requested_value:
            return True

    return False


# ============================================================
# FIND ELIGIBLE HOSPITALS
# ============================================================

def find_eligible_hospitals(
    specialty=None,
    procedure=None,
    service=None
):

    eligible_hospitals = []


    # --------------------------------------------------------
    # CHECK EVERY HOSPITAL
    # --------------------------------------------------------

    for _, hospital in df.iterrows():


        # ====================================================
        # SPECIALTY MATCH
        # ====================================================

        specialty_match = False

        if specialty:

            specialty_match = contains_value(
                hospital["Specialties"],
                specialty
            )

            # ------------------------------------------------
            # SPECIALTY IS THE ELIGIBILITY CONDITION
            # ------------------------------------------------

            if not specialty_match:
                continue


        # ====================================================
        # PROCEDURE MATCH
        # ====================================================
        # This does NOT determine eligibility.
        # It is only recorded.
        # ====================================================

        procedure_match = False

        if procedure:

            procedure_match = contains_value(
                hospital["Procedures"],
                procedure
            )


        # ====================================================
        # SERVICE MATCH
        # ====================================================
        # This does NOT determine eligibility either.
        # It is only recorded.
        # ====================================================

        service_match = False

        if service:

            service_match = contains_value(
                hospital["Services"],
                service
            )


        # ====================================================
        # ADD HOSPITAL
        # ====================================================

        eligible_hospitals.append({

            "Hospital_Name":
                hospital["Hospital_Name"],

            "Specialty_Match":
                specialty_match,

            "Procedure_Match":
                procedure_match,

            "Service_Match":
                service_match
        })


    return eligible_hospitals

# ============================================================
# CREATE HOSPITAL MAP
# ============================================================

def create_hospital_map(
    specialty,
    procedure,
    service
):

    import folium

    # Find eligible hospitals directly from the database
    eligible_rows = []

    for _, hospital in df.iterrows():

        # Specialty is the eligibility condition
        if specialty:

            if not contains_value(
                hospital["Specialties"],
                specialty
            ):
                continue

        # Check procedure only for information
        procedure_match = False

        if procedure:

            procedure_match = contains_value(
                hospital["Procedures"],
                procedure
            )

        # Check service only for information
        service_match = False

        if service:

            service_match = contains_value(
                hospital["Services"],
                service
            )

        eligible_rows.append({
            "hospital": hospital,
            "procedure_match": procedure_match,
            "service_match": service_match
        })


    # --------------------------------------------------------
    # CHECK WHETHER ANY HOSPITALS WERE FOUND
    # --------------------------------------------------------

    if not eligible_rows:

        print(
            "\nNo eligible hospitals to plot."
        )

        return


    # --------------------------------------------------------
    # GET VALID COORDINATES
    # --------------------------------------------------------

    hospitals_with_coordinates = []

    for item in eligible_rows:

        hospital = item["hospital"]

        try:

            latitude = float(
                hospital["Latitude"]
            )

            longitude = float(
                hospital["Longitude"]
            )

        except (ValueError, TypeError):

            continue


        hospitals_with_coordinates.append({
            "hospital": hospital,
            "latitude": latitude,
            "longitude": longitude,
            "procedure_match":
                item["procedure_match"],
            "service_match":
                item["service_match"]
        })


    if not hospitals_with_coordinates:

        print(
            "\nEligible hospitals were found, "
            "but none have valid coordinates."
        )

        return


    # --------------------------------------------------------
    # CALCULATE MAP CENTER
    # --------------------------------------------------------

    avg_latitude = sum(
        h["latitude"]
        for h in hospitals_with_coordinates
    ) / len(hospitals_with_coordinates)

    avg_longitude = sum(
        h["longitude"]
        for h in hospitals_with_coordinates
    ) / len(hospitals_with_coordinates)


    # --------------------------------------------------------
    # CREATE MAP
    # --------------------------------------------------------

    hospital_map = folium.Map(

        location=[
            avg_latitude,
            avg_longitude
        ],

        zoom_start=12
    )


    # --------------------------------------------------------
    # ADD HOSPITAL MARKERS
    # --------------------------------------------------------

    for number, item in enumerate(
        hospitals_with_coordinates,
        start=1
    ):

        hospital = item["hospital"]

        hospital_name = (
            hospital["Hospital_Name"]
        )

        latitude = item["latitude"]

        longitude = item["longitude"]

        procedure_match = (
            item["procedure_match"]
        )

        service_match = (
            item["service_match"]
        )


        # ----------------------------------------------------
        # POPUP INFORMATION
        # ----------------------------------------------------

        popup_html = f"""
        <b>{hospital_name}</b>
        <br><br>
        <b>Specialty:</b> {specialty}
        """


        if procedure:

            popup_html += f"""
            <br>
            <b>Procedure:</b> {procedure}
            <br>
            <b>Procedure Match:</b>
            {procedure_match}
            """


        if service:

            popup_html += f"""
            <br>
            <b>Service:</b> {service}
            <br>
            <b>Service Match:</b>
            {service_match}
            """


        popup_html += f"""
        <br><br>
        <b>Latitude:</b> {latitude}
        <br>
        <b>Longitude:</b> {longitude}
        """


        popup = folium.Popup(
            popup_html,
            max_width=350
        )


        # ----------------------------------------------------
        # ADD MARKER
        # ----------------------------------------------------

        folium.Marker(

            location=[
                latitude,
                longitude
            ],

            popup=popup,

            tooltip=(
                f"{number}. {hospital_name}"
            )

        ).add_to(hospital_map)


    # --------------------------------------------------------
    # SAVE MAP
    # --------------------------------------------------------

 map_file = os.path.join(
    BASE_DIR,
    "eligible_hospitals_map.html"
)

hospital_map.save(map_file)

print(
    "\nMap created successfully."
)

print(
    f"Map file: {map_file}"
)

print(
    f"Hospitals plotted: "
    f"{len(hospitals_with_coordinates)}"
)


# ============================================================
# OPEN MAP AUTOMATICALLY IN BROWSER
# ============================================================

import webbrowser

webbrowser.open(
    "file:///" + os.path.abspath(map_file)
)
# ============================================================
# DISPLAY ELIGIBLE HOSPITALS
# ============================================================

def display_eligible_hospitals(
    hospitals,
    specialty,
    procedure,
    service
):

    print(
        "\n" + "=" * 60
    )

    print(
        "ELIGIBLE HOSPITALS"
    )

    print(
        "=" * 60
    )


    # --------------------------------------------------------
    # NO SPECIALTY
    # --------------------------------------------------------

    if not specialty:

        print(
            "\nNo valid specialty was identified."
        )

        print(
            "Hospital eligibility cannot be determined."
        )

        return


    # --------------------------------------------------------
    # NO HOSPITALS
    # --------------------------------------------------------

    if not hospitals:

        print(
            f"\nNo hospitals in the database "
            f"provide: {specialty}"
        )

        return


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print(
        f"\nRequired specialty: {specialty}"
    )

    if procedure:

        print(
            f"Requested procedure: {procedure}"
        )

    if service:

        print(
            f"Requested service: {service}"
        )


    print(
        f"\nNumber of eligible hospitals: "
        f"{len(hospitals)}"
    )


    print(
        "\n" + "-" * 60
    )


    for number, hospital in enumerate(
        hospitals,
        start=1
    ):

        print(
            f"\n{number}. "
            f"{hospital['Hospital_Name']}"
        )

        print(
            "   Specialty match: YES"
        )


        if procedure:

            if hospital["Procedure_Match"]:

                print(
                    "   Procedure match: YES"
                )

            else:

                print(
                    "   Procedure match: NO"
                )


        if service:

            if hospital["Service_Match"]:

                print(
                    "   Service match: YES"
                )

            else:

                print(
                    "   Service match: NO"
                )


    print(
        "\n" + "=" * 60
    )


# ============================================================
# MEDICAL INTENT EXTRACTION
# ============================================================

def extract_medical_intent(
    user_input
):

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

5. If the patient explicitly mentions a service and that service
   exists in AVAILABLE SERVICES, select it.

6. If the patient mentions a service that does not exist in
   AVAILABLE SERVICES, return null for service.

7. If the patient does not explicitly mention a specialty,
   infer the most appropriate specialty from the patient's
   symptoms or request.

8. When inferring a specialty, you MUST choose it from
   AVAILABLE SPECIALTIES.

9. NEVER invent a specialty.

10. NEVER invent a procedure.

11. NEVER invent a service.

12. If no appropriate specialty exists in AVAILABLE
    SPECIALTIES, return null for specialty.

13. Set emergency to true when the patient's input indicates
    a potentially urgent or emergency situation.

14. Do not diagnose the patient.

15. Do not provide medical advice.

16. Do not explain your reasoning.

17. Return ONLY valid JSON.

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


    # ========================================================
    # START TIMER
    # ========================================================

    start_time = time.perf_counter()


    # ========================================================
    # QWEN REQUEST
    # ========================================================

    response = chat(

        model=MODEL_NAME,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        # Disable visible thinking
        think=False,

        # Keep model loaded
        keep_alive=-1,

        options={
            "temperature": 0,
            "num_predict": 100
        }
    )


    # ========================================================
    # RUNTIME
    # ========================================================

    runtime = (
        time.perf_counter()
        - start_time
    )


    # ========================================================
    # MODEL OUTPUT
    # ========================================================

    output = response[
        "message"
    ][
        "content"
    ].strip()


    return output, runtime


# ============================================================
# MAIN LOOP
# ============================================================

while True:


    user_input = input(
        "\nPatient: "
    ).strip()


    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    if user_input.lower() == "exit":

        print(
            "\nExiting..."
        )

        break


    # --------------------------------------------------------
    # EMPTY INPUT
    # --------------------------------------------------------

    if not user_input:

        continue


    try:


        # ====================================================
        # QWEN
        # ====================================================

        result, runtime = (
            extract_medical_intent(
                user_input
            )
        )


        print(
            "\nQwen output:"
        )


        # ====================================================
        # PARSE JSON
        # ====================================================

        try:

            parsed = json.loads(
                result
            )


            print(
                json.dumps(
                    parsed,
                    indent=4,
                    ensure_ascii=False
                )
            )


        except json.JSONDecodeError:

            print(
                "Qwen did not return valid JSON:"
            )

            print(
                result
            )

            print(
                f"\nRuntime: "
                f"{runtime:.2f} seconds"
            )

            print(
                "-" * 60
            )

            continue


        # ====================================================
        # EXTRACT QWEN RESULTS
        # ====================================================

        specialty = parsed.get(
            "specialty"
        )

        procedure = parsed.get(
            "procedure"
        )

        service = parsed.get(
            "service"
        )

        emergency = parsed.get(
            "emergency",
            False
        )


        # ====================================================
        # FIND ELIGIBLE HOSPITALS
        # ====================================================

        eligible_hospitals = (
            find_eligible_hospitals(
                specialty=specialty,
                procedure=procedure,
                service=service
            )
        )


        # ====================================================
        # DISPLAY ELIGIBLE HOSPITALS
        # ====================================================

        display_eligible_hospitals(

            eligible_hospitals,

            specialty,

            procedure,

            service
        )


        # ====================================================
        # EMERGENCY STATUS
        # ====================================================

        print(
            f"\nEmergency detected: "
            f"{emergency}"
        )


        # ====================================================
        # QWEN RUNTIME
        # ====================================================

        print(
            f"Qwen Runtime: "
            f"{runtime:.2f} seconds"
        )


        print(
            "-" * 60
        )


    except Exception as e:

        print(
            "\nERROR:"
        )

        print(
            e
        )

        print(
            "-" * 60
        )

