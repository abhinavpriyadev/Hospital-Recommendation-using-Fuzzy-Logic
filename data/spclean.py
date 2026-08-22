import pandas as pd
import re
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "hospitals-final.csv"
OUTPUT_FILE = "hospitals-final-cleaned.csv"
UNKNOWN_FILE = "unrecognized_terms.csv"

SPECIALTY_COLUMN = "Specialties"


# ============================================================
# CANONICAL SPECIALTIES
# ============================================================

SPECIALTY_ALIASES = {

    # Core medical specialties
    "Anaesthesiology": [
        "anaesthesiology",
        "anesthesiology",
        "anaesthesia",
        "anesthesia",
        "anaesthesiology and pain clinic",
        "cardiac anaesthesia",
        "cardiac anesthesiology",
        "anasthesiology",
    ],

    "Cardiology": [
        "cardiology",
        "cardiologist",
        "cardio",
        "cardiac",
        "cardiology and interventional",
        "interventional cardiology",
        "cardiovascular disease",
    ],

    "Cardiothoracic Surgery": [
        "cardiothoracic surgery",
        "cardio thoracic surgery",
        "cardiothoracic surgery paediatric",
        "cardiothoracic surgery (paediatric)",
        "cardiac surgery",
        "cardiovascular surgery",
        "cardio vascular surgery",
    ],

    "Critical Care": [
        "critical care",
        "critical care medicine",
        "intensive care",
        "intensive care medicine",
    ],

    "Dentistry": [
        "dentistry",
        "dental sciences",
        "dental department",
        "dental clinic",
    ],

    "Dermatology": [
        "dermatology",
        "dermatologist",
        "skin",
        "skin department",
        "dermatology skin department",
    ],

    "Diabetology": [
        "diabetology",
        "diabetic",
        "diabetes care and education",
    ],

    "Emergency Medicine": [
        "emergency medicine",
        "accident and emergency",
        "accident and emergency services 24 hours",
        "casualty accident and emergency medicine",
        "emergency care",
        "emergency and trauma care",
        "emergency and critical care",
        "trauma care",
    ],

    "Endocrinology": [
        "endocrinology",
        "endocrinology and metabolism",
        "endocrinolagy",
        "endocrinologist",
        "endocrinology department",
    ],

    "ENT": [
        "ent",
        "e.n.t.",
        "ent and head and neck surgery",
        "ent and head neck surgery",
        "ear nose and throat",
    ],

    "Gastroenterology": [
        "gastroenterology",
        "gastro",
        "medical gastroenterology",
        "gastroenterology medical",
        "gastroenterology surgical",
        "gastroenterology medical and surgical",
        "medical and surgical gastroenterology",
        "surgical gastroenterology",
        "surgical gastroenterology and endoscopy",
    ],

    "General Medicine": [
        "general medicine",
        "general medicine and geriatrics",
        "internal medicine",
        "medicine",
        "internist",
        "medical department",
    ],

    "General Surgery": [
        "general surgery",
        "general surgeon",
        "general surgery and oncology",
        "general medicine and general surgery",
        "general and laparoscopic surgery",
        "general surgical and laparoscopic",
        "general surgery gynaecology",
        "general and pediatric surgery",
    ],

    "Geriatrics": [
        "geriatrics",
        "geriatric",
        "geriatrice",
        "general medicine and geriatrics",
        "geriatric psychiatry",
    ],

    "Haematology": [
        "haematology",
        "hematology",
        "haematology imaging sciences",
    ],

    "Neonatology": [
        "neonatology",
        "paediatrics and neonatology",
        "paediatric and neonatology",
        "pediatrician and neonatologist",
        "pediatric and neonatology",
    ],

    "Nephrology": [
        "nephrology",
        "nephrologist",
        "kidney",
        "nephro",
        "urology and nephrology",
        "nephrology and urology",
        "nephro and uro care",
    ],

    "Neurology": [
        "neurology",
        "neurologist",
        "neuromedicine",
        "neuro",
        "neurology and neurosurgery",
        "neurology and neuro surgery",
        "ent and neurology",
    ],

    "Neurosurgery": [
        "neurosurgery",
        "neuro surgery",
        "neuro-surgery",
        "neurological surgery",
        "neurosurgeon",
        "neurology/neurosurgery",
    ],

    "Nuclear Medicine": [
        "nuclear medicine",
    ],

    "Obstetrics and Gynaecology": [
        "obstetrics and gynaecology",
        "obstetrics and gynecology",
        "obstetrics andgynaecology",
        "obstetrics and gynacology",
        "gynaecology and obstetric",
        "gynaecology",
        "gynecology",
        "obstetrician and gynaecologist",
        "woman",
        "women care",
    ],

    "Ophthalmology": [
        "ophthalmology",
        "ophthalmologist",
        "opthamology",
        "eye",
        "ophthalmology department",
    ],

    "Orthopaedics": [
        "orthopaedics",
        "orthopedics",
        "orthopaedic",
        "orthopaedician",
        "orthopaedic department",
        "orthopaedic surgery",
        "orthopaedics and traumatology",
        "ortho",
        "bone and joint care",
        "bone and joint center",
    ],

    "Paediatrics": [
        "paediatrics",
        "pediatrics",
        "paediatric",
        "pediatric",
        "paediatric department",
        "paediatrician",
        "pediatrician",
        "general paediatrics",
        "general pediatric surgery",
        "child care",
        "child",
    ],

    "Paediatric Surgery": [
        "paediatric surgery",
        "pediatric surgery",
        "paediatric surgeon",
        "paediatric department surgery",
        "cardiothoracic surgery paediatric",
    ],

    "Physiotherapy": [
        "physiotherapy",
        "physio",
        "physio back school",
        "physical medicine and rehabilitation",
        "rehabilitation medicine",
    ],

    "Plastic Surgery": [
        "plastic surgery",
        "plastic surgeon",
        "plastic and reconstructive surgery",
        "plastic and reconstructive",
        "plastic surgery and cosmetic surgery",
        "plastic and cosmetic surgery",
        "cosmetic and plastic surgery",
        "cosmetologist plastic surgeon",
    ],

    "Psychiatry": [
        "psychiatry",
        "psychiatrist",
        "adult psychiatry",
        "child psychiatry",
        "child psychiatry service offered",
        "geriatric psychiatric services",
    ],

    "Psychology": [
        "psychology",
        "psychological services",
        "psychologist",
    ],

    "Pulmonology": [
        "pulmonology",
        "pulmonologist",
        "respiratory medicine",
        "chest physician",
        "chest",
        "chest medicine",
        "chest department",
        "chest diseases department",
        "chest and tb",
    ],

    "Radiology": [
        "radiology",
        "radiologist",
        "radiology and imageology",
        "radiology and imaging sciences",
        "imaging sciences",
        "radiology and sonology",
    ],

    "Rheumatology": [
        "rheumatology",
        "rheumatologist",
        "joint pain",
        "anaemia and arthritis clinic",
    ],

    "Urology": [
        "urology",
        "urologist",
        "urologoy",
        "uro surgery",
        "urology and nephrology",
        "nephrology and urology",
        "nephro and uro care",
    ],

    "Vascular Surgery": [
        "vascular surgery",
        "vascular",
        "cardiovascular surgery",
    ],

    "Oncology": [
        "oncology",
        "cancer",
        "cancer medical and surgical",
        "oncology cancer medical and surgical",
        "oncology department",
        "general surgery and oncology",
    ],

    "Medical Oncology": [
        "medical oncology",
        "hemato oncology",
    ],

    "Surgical Oncology": [
        "surgical oncology",
    ],

    "Radiation Oncology": [
        "radiation oncology",
    ],

    "Pathology": [
        "pathology",
        "laboratory medicine",
        "lab",
        "diagnostic laboratory",
        "state of art diagnostic laboratory",
    ],

    "Microbiology": [
        "microbiology",
        "micro-biology",
    ],

    "Genetics": [
        "genetics",
    ],

    "Infectious Diseases": [
        "infectious diseases",
    ],

    "Family Medicine": [
        "family medicine",
    ],

    "Preventive Medicine": [
        "preventive medicine",
        "preventive health checks",
    ],

    "Adolescent Medicine": [
        "adolescent medicine",
    ],

    "Occupational Therapy": [
        "occupational therapy",
    ],

    "Speech and Hearing Therapy": [
        "speech and hearing therapy",
        "speech therapy",
    ],

    "Dietetics": [
        "dietetics",
        "diet counselling",
        "diet counselling/dietetics",
        "nutrition and diet counselling",
        "nutritionist",
    ],

    "Reproductive Medicine": [
        "reproductive medicine",
        "reproductive technology",
        "in vitro fertilization",
        "in vitro fertilization / reproductive medicine",
        "ivf and infertility",
        "infertility",
        "obstetrics and infertility",
    ],

    "Andrology": [
        "andrology",
    ],

    "Ayurveda": [
        "ayurveda",
    ],

    "Homeopathy": [
        "homeopathy",
    ],

    "Unani": [
        "unani",
    ],
    "Allergy and Immunology": [
    "allergy",
    "allergy and asthma clinic",
    "allergy and arthma clinic",
],

"Paediatric Cardiology": [
    "paediatric cardiology",
    "pediatric cardiology",
    "cyanotic and acyanotic heart diseases (simple and complex)",
],

"Behavioural Medicine": [
    "behavioral medicine",
    "behavioural medicine",
],

"Biochemistry": [
    "biochemistry",
    "bio-chemistry",
],

"Community Medicine": [
    "community medicine",
],

"Forensic Medicine and Toxicology": [
    "forensic medicine and toxicology",
],

"Cornea and External Eye Disease": [
    "cornea",
],

"Cornea and Refractive Surgery": [
    "cornea and refractive",
],

"Glaucoma": [
    "glaucoma",
],

"Maxillofacial Surgery": [
    "dentistry and maxillofacial surgery",
    "maxillofacial surgery",
],

"Cosmetic Surgery": [
    "cosmetic surgery",
],

"Digestive Medicine": [
    "digestive care",
],

"Addiction Medicine": [
    "alcohol abuse clinic",
],
}


# ============================================================
# PROCEDURES
# ============================================================

PROCEDURE_ALIASES = {

    "Laparoscopic Surgery": [
        "laparoscopic surgery",
        "lapraroscopic surgery",
        "general and laparoscopic surgery",
        "general surgical and laparoscopic",
        "advanced key hole surgery",
        "key hole surgery",
    ],

    "Joint Replacement": [
        "joint replacement surgery",
        "joint replacement",
        "knee replacement",
        "centre for knee replacement",
        "total hip replacement",
        "knee reconstruction and replacement",
        "ortho and joint replacement",
    ],

    "Spine Surgery": [
        "spine surgery",
        "spinal surgery",
        "spine related disorders",
        "brain and spine care",
    ],

    "Bariatric Surgery": [
        "bariatric surgery",
        "obesity and bariatric surgery",
        "obesity and bariatric",
    ],

    "Liver Transplant": [
        "liver transplant",
        "liver transplantation",
    ],

    "Kidney Transplant": [
        "kidney transplant",
        "renal transplant",
    ],

    "IVF": [
        "in vitro fertilization",
        "in vitro fertilisation",
        "ivf",
        "ivf and infertility",
        "reproductive technology",
    ],

    "Endoscopy": [
        "endoscopy",
        "gastrointestinal endoscopy",
    ],

    "Stapedectomy": [
        "stapedectomy",
    ],

    "Sinus Surgery": [
        "sinus surgery",
    ],

    "Endolaryngeal Surgery": [
        "endolaryngeal surgery",
    ],

    "Coblation Tonsillectomy and Adenoidectomy": [
        "coblation tonsillectomy and adenoidectomy",
    ],

    "Endoscopic Dacryocystorhinostomy": [
        "endoscopic dacryocystorhinostomy",
    ],

    "Skull Base Surgery": [
        "advanced skull base surgeries",
        "skull base surgery",
    ],

    "Arthroscopy": [
        "arthroscopy",
    ],

    "Limb Reconstruction": [
        "limb reconstruction",
    ],

    "Burns Treatment": [
        "burns unit",
        "burn treatment",
    ],

    "Dialysis": [
        "dialysis",
    ],
    "Cataract Surgery": [
    "cataract",
],

"Ear Surgery": [
    "ear surgery for discharging ear",
],

"Laparoscopic Surgery": [
    "general surgery and laparoscopic surgery",
    "general surgical and laproscopic",
],
}


# ============================================================
# SERVICES
# ============================================================

SERVICE_ALIASES = {

    "Emergency Services": [
        "emergency care",
        "emergency medicine",
        "emergency and trauma care",
        "emergency and critical care",
        "accident and emergency",
        "accident and emergency services 24 hours",
        "cardiac emergency care",
        "24 hour emergency",
        "24hrs emergency",
    ],

    "Intensive Care": [
        "intensive care unit",
        "intensive care",
        "icu",
        "critical care",
        "critical care medicine",
        "coronary care unit",
    ],

    "Diagnostic Laboratory": [
        "diagnostic laboratory",
        "state of art diagnostic laboratory",
        "laboratory medicine",
        "lab",
    ],

    "Pharmacy": [
        "24hrs pharmacy",
        "24 hours pharmacy",
        "pharmacy",
    ],

    "Radiology and Imaging": [
        "x ray",
        "x-ray",
        "sonology",
        "imaging sciences",
    ],

    "Cardiac Care": [
        "cardiac care center",
        "heart care",
        "healthy heart clinic",
        "centre for heart failure management",
        "cardiac emergency care",
    ],

    "Diabetes Care": [
        "diabetes care and education",
        "diabetes heart evaluation",
        "diabetes neuropathy clinic",
        "diabetic eye care",
        "preventive diabetes foot care",
        "special diabetes counselling",
        "special clinic for children with diabetes",
    ],

    "Pain Management": [
        "pain clinic",
        "pain and palliative care",
        "pain clinic for special problems",
    ],

    "Cancer Screening": [
        "cancer screening",
    ],

    "Weight Management": [
        "weight management clinic",
        "obesity and lifestyle counselling",
    ],

    "Women Wellness": [
        "women wellness clinic",
        "women care",
    ],

    "Fertility Services": [
        "cradle fertility center",
        "fertility center",
        "infertility services",
        "pre conceptional clinic",
    ],

    "Rehabilitation": [
        "rehabilitation medicine",
        "physical medicine and rehabilitation",
    ],

    "Sports Medicine": [
        "sports medicine",
        "sports injury",
    ],
    "Surgical Services": [
    "surgical center",
],

"Maternity Services": [
    "birthing center",
],

"Emergency Services": [
    "casualty (accident and emergency medicine)",
],

"Epilepsy Care": [
    "centre for comprehensive epilepsy care",
],

"Contact Lens Services": [
    "contact lens",
],

"Trauma Care": [
    "head injury",
],

"Nutrition and Psychological Services": [
    "holistic approach clinic nutritionist and psycologist",
],
    "Digestive Care": [
    "digestive care",
],
}


# ============================================================
# THINGS TO REMOVE
# ============================================================

REMOVE_TERMS = {
    "0",
    "",
    "community",
    "community health",
    "holistic approach",
    "reiki",
    "non surgical treatment",
    "associate specialties",
    "para medical facilities",
    "dia-shoppe with products and medicines for diabetes treatment",
    "trained orthotist and complete in-house footwear manufacturing unit for special needs of diabetic patients with all types of foot problems",
    "appropriate dental care for diabetic patients",
    "special yoga classes for diabetes control",
    "sexual problems",
    "pre marital counseling and family planning",
    "menopausal clinic",
    "surgery",
"cosmetology",
"anatomy",
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):
    """
    Normalize a piece of text so matching becomes easier.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Decode literal \n sequences
    text = text.replace("\\n", " ")

    # Actual newlines/tabs
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")

    # Normalize brackets
    text = text.replace("[", "")
    text = text.replace("]", "")

    # Normalize ampersand
    text = text.replace("&", " and ")

    # Common spelling corrections
    replacements = {
        "opthamology": "ophthalmology",
        "opthamologist": "ophthalmologist",
        "urologoy": "urology",
        "lapraroscopic": "laparoscopic",
        "anasthesiology": "anaesthesiology",
        "anesthesiology": "anaesthesiology",
        "anesthesia": "anaesthesia",
        "endocrinolagy": "endocrinology",
        "gynacology": "gynaecology",
        "gynecology": "gynaecology",
        "pediatric": "paediatric",
        "pediatrics": "paediatrics",
        "orthopedics": "orthopaedics",
        "orthopaedician": "orthopaedics",
        "geriatrice": "geriatrics",
        "lnterventional": "interventional",
        "lnjury": "injury",
        "medicne": "medicine",
        "physcian": "physician",
        "alcoho abuse": "alcohol abuse",
        "onocology": "oncology",
        "haematology": "haematology",
    }

    text = text.lower()

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Normalize punctuation around separators
    text = re.sub(r"\s+", " ", text)

    # Remove trailing punctuation
    text = text.strip(" ,;:-")

    return text.strip()


# ============================================================
# BUILD LOOKUP TABLES
# ============================================================

def build_lookup(alias_dict):
    lookup = {}

    for canonical, aliases in alias_dict.items():

        # Canonical name itself
        lookup[normalize(canonical)] = canonical

        for alias in aliases:
            lookup[normalize(alias)] = canonical

    return lookup


SPECIALTY_LOOKUP = build_lookup(SPECIALTY_ALIASES)
PROCEDURE_LOOKUP = build_lookup(PROCEDURE_ALIASES)
SERVICE_LOOKUP = build_lookup(SERVICE_ALIASES)


# ============================================================
# SPECIAL COMBINATIONS
# ============================================================

COMBINATION_REPLACEMENTS = {

    # These occur because the source sometimes loses separators.
    "medical oncology nephrology":
        ["Medical Oncology", "Nephrology"],

    "nuclear medicine obstetrics and gynaecology":
        ["Nuclear Medicine", "Obstetrics and Gynaecology"],

    "plastic surgery psychiatry":
        ["Plastic Surgery", "Psychiatry"],

    "surgical oncology urology":
        ["Surgical Oncology", "Urology"],

    "joint replacement surgery nephrology":
        ["Joint Replacement", "Nephrology"],

    "haematology imaging sciences":
        ["Haematology", "Radiology"],

    "general surgery gynaecology":
        ["General Surgery", "Obstetrics and Gynaecology"],

    "endocrinology ent":
        ["Endocrinology", "ENT"],

    "pulmonology radiology":
        ["Pulmonology", "Radiology"],

    "obstetrics and gynaecology ophthalmology":
        ["Obstetrics and Gynaecology", "Ophthalmology"],

    "liver transplant microbiology":
        ["Liver Transplant", "Microbiology"],

    "neurology orthopaedics":
        ["Neurology", "Orthopaedics"],

    "neurology and neurosurgery":
        ["Neurology", "Neurosurgery"],

    "nephrology and urology":
        ["Nephrology", "Urology"],

    "urology vascular surgery":
        ["Urology", "Vascular Surgery"],

    "surgical gastroenterology urology":
        ["Gastroenterology", "Urology"],
    "cardio vascular disease and robotic surgery": [
    "Cardiovascular Surgery",
    "Robotic Surgery",
],
}


# ============================================================
# SPLITTING
# ============================================================

def split_raw_cell(text):
    """
    Split raw hospital data into probable individual entities.

    We deliberately do NOT blindly split on every space.
    """

    if pd.isna(text):
        return []

    text = str(text)

    # Literal newlines
    text = text.replace("\\n", "\n")

    # First split obvious separators
    parts = re.split(r"[,;\n\r]+", text)

    result = []

    for part in parts:

        part = part.strip()

        if not part:
            continue

        # Slash often separates specialties:
        # Cardiology / Cardiothoracic Surgery
        slash_parts = re.split(r"\s*/\s*", part)

        for p in slash_parts:
            p = p.strip()

            if p:
                result.append(p)

    return result


# ============================================================
# MAP A SINGLE ENTITY
# ============================================================

def map_entity(entity):
    """
    Map one raw entity into:
        specialty
        procedure
        service
        remove
        unknown
    """

    original = entity
    n = normalize(entity)

    if not n:
        return "remove", None

    # Direct removal
    if n in REMOVE_TERMS:
        return "remove", None

    # Combination handling
    if n in COMBINATION_REPLACEMENTS:
        return "combination", COMBINATION_REPLACEMENTS[n]

    # Exact specialty
    if n in SPECIALTY_LOOKUP:
        return "specialty", SPECIALTY_LOOKUP[n]

    # Exact procedure
    if n in PROCEDURE_LOOKUP:
        return "procedure", PROCEDURE_LOOKUP[n]

    # Exact service
    if n in SERVICE_LOOKUP:
        return "service", SERVICE_LOOKUP[n]

    # --------------------------------------------------------
    # Handle some descriptive variants
    # --------------------------------------------------------

    # Anything containing a known specialty as a clear phrase
    # can sometimes be mapped safely.
    phrase_checks = [
        ("plastic and reconstructive surgery", "Plastic Surgery"),
        ("plastic surgery and cosmetic surgery", "Plastic Surgery"),
        ("cosmetic and plastic surgery", "Plastic Surgery"),
        ("plastic and cosmetic surgery", "Plastic Surgery"),
        ("cardio thoracic surgery", "Cardiothoracic Surgery"),
        ("cardiothoracic surgery", "Cardiothoracic Surgery"),
        ("medical oncology", "Medical Oncology"),
        ("surgical oncology", "Surgical Oncology"),
        ("radiation oncology", "Radiation Oncology"),
        ("surgical gastroenterology", "Gastroenterology"),
        ("medical gastroenterology", "Gastroenterology"),
        ("general surgery", "General Surgery"),
        ("orthopaedic surgery", "Orthopaedics"),
        ("orthopaedic department", "Orthopaedics"),
        ("paediatric department", "Paediatrics"),
        ("pediatric department", "Paediatrics"),
        ("chest physician", "Pulmonology"),
        ("chest department", "Pulmonology"),
        ("chest diseases", "Pulmonology"),
        ("dermatology department", "Dermatology"),
        ("oncology department", "Oncology"),
        ("endocrinology department", "Endocrinology"),
        ("neurology department", "Neurology"),
        ("neuro surgery", "Neurosurgery"),
        ("urology", "Urology"),
        ("nephrology", "Nephrology"),
        ("ophthalmology", "Ophthalmology"),
        ("radiology", "Radiology"),
    ]

    for phrase, canonical in phrase_checks:
        if phrase in n and len(n) <= len(phrase) + 20:
            return "specialty", canonical

    return "unknown", original


# ============================================================
# PROCESS ONE CELL
# ============================================================

def process_cell(text):

    specialties = []
    procedures = []
    services = []
    unknown = []

    raw_entities = split_raw_cell(text)

    for raw in raw_entities:

        category, value = map_entity(raw)

        if category == "specialty":
            specialties.append(value)

        elif category == "procedure":
            procedures.append(value)

        elif category == "service":
            services.append(value)

        elif category == "combination":
            for item in value:
                # Re-process each generated item
                c, v = map_entity(item)

                if c == "specialty":
                    specialties.append(v)

                elif c == "procedure":
                    procedures.append(v)

                elif c == "service":
                    services.append(v)

                elif c == "unknown":
                    unknown.append(v)

        elif category == "unknown":
            unknown.append(value)

    # Deduplicate while preserving order
    specialties = list(dict.fromkeys(specialties))
    procedures = list(dict.fromkeys(procedures))
    services = list(dict.fromkeys(services))

    return specialties, procedures, services, unknown


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("HOSPITAL SPECIALTY / PROCEDURE / SERVICE CLEANER")
print("=" * 60)

print()
print(f"Loading: {INPUT_FILE}")

try:
    df = pd.read_csv(
        INPUT_FILE,
        dtype=str,
        keep_default_na=False
    )
except FileNotFoundError:
    print()
    print("ERROR: Input file not found.")
    print()
    print("Make sure hospitals-final.csv is in the SAME folder")
    print("as this Python script.")
    print()
    raise SystemExit(1)

print(f"Rows loaded: {len(df)}")

if SPECIALTY_COLUMN not in df.columns:

    print()
    print("ERROR:")
    print(f"Column '{SPECIALTY_COLUMN}' was not found.")
    print()
    print("Available columns:")
    for col in df.columns:
        print(f"  - {col}")

    raise SystemExit(1)

print(f"Specialty column: {SPECIALTY_COLUMN}")
print()


# ============================================================
# PROCESS DATA
# ============================================================

all_unknown = []

cleaned_specialties = []
cleaned_procedures = []
cleaned_services = []

for _, row in df.iterrows():

    raw_text = row[SPECIALTY_COLUMN]

    specialties, procedures, services, unknown = process_cell(raw_text)

    cleaned_specialties.append("; ".join(specialties))
    cleaned_procedures.append("; ".join(procedures))
    cleaned_services.append("; ".join(services))

    all_unknown.extend(unknown)


# ============================================================
# ADD OUTPUT COLUMNS
# ============================================================

df["Specialties_Cleaned"] = cleaned_specialties
df["Procedures_Cleaned"] = cleaned_procedures
df["Services_Cleaned"] = cleaned_services


# ============================================================
# UNKNOWN TERMS
# ============================================================

unknown_counter = defaultdict(int)

for term in all_unknown:

    normalized = normalize(term)

    if normalized:
        unknown_counter[normalized] += 1


unknown_sorted = sorted(
    unknown_counter.items(),
    key=lambda x: (-x[1], x[0])
)


# ============================================================
# SAVE UNKNOWN TERMS
# ============================================================

unknown_df = pd.DataFrame(
    [
        {
            "Unrecognized_Term": term,
            "Occurrences": count
        }
        for term, count in unknown_sorted
    ]
)

unknown_df.to_csv(
    UNKNOWN_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SAVE CLEANED FILE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# REPORT
# ============================================================

print("=" * 60)
print("CLEANING COMPLETE")
print("=" * 60)

print()
print("Input file:")
print(f"  {INPUT_FILE}")

print()
print("Output file:")
print(f"  {OUTPUT_FILE}")

print()
print("Unrecognized terms:")
print(f"  {UNKNOWN_FILE}")

print()
print(f"Hospitals processed: {len(df)}")
print(f"Unique unrecognized terms: {len(unknown_sorted)}")

print()
print("New columns:")
print("  - Specialties_Cleaned")
print("  - Procedures_Cleaned")
print("  - Services_Cleaned")

print()

if unknown_sorted:

    print("Remaining unrecognized terms:")
    print("-" * 40)

    for term, count in unknown_sorted:
        print(f"  {term} ({count})")

else:

    print("No unrecognized terms remain.")

print()
print("Done.")
