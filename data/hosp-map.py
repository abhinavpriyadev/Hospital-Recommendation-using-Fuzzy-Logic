import pandas as pd
from geopy.geocoders import Nominatim
import folium
import time

# =========================================================
# SETTINGS
# =========================================================

INPUT_FILE = "hospitals_final.csv"

HOSPITAL_COLUMN = "Hospital_Name"
ADDRESS_COLUMN = "Location"

GEOCODED_FILE = "hospitals_geocoded.csv"
FAILED_FILE = "hospitals_not_found.csv"
MAP_FILE = "hospital_map.html"


# =========================================================
# READ CSV
# =========================================================

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

df.columns = df.columns.str.strip()

print("Total records:", len(df))


# =========================================================
# GEOCODER
# =========================================================

geolocator = Nominatim(
    user_agent="hospital_recommendation_project",
    timeout=10
)


# =========================================================
# STORAGE
# =========================================================

latitudes = []
longitudes = []
queries_used = []
geocoding_status = []


# =========================================================
# GEOCODE
# =========================================================

for index, row in df.iterrows():

    hospital = str(row[HOSPITAL_COLUMN]).strip()
    address = str(row[ADDRESS_COLUMN]).strip()

    if hospital.lower() == "nan":
        hospital = ""

    if address.lower() == "nan":
        address = ""

    print("\n----------------------------------------")
    print(f"Processing {index + 1}/{len(df)}")
    print("Hospital:", hospital)
    print("Address:", address)

    # Try several searches
    queries = [
        f"{hospital}, {address}, Chennai, Tamil Nadu, India",
        f"{hospital}, Chennai, Tamil Nadu, India",
        f"{address}, Chennai, Tamil Nadu, India"
    ]

    location = None
    successful_query = ""

    for query in queries:

        print("Trying:", query)

        try:

            location = geolocator.geocode(query)

            if location:

                print("FOUND!")
                print(
                    "Latitude:",
                    location.latitude
                )
                print(
                    "Longitude:",
                    location.longitude
                )

                successful_query = query

                break

            else:

                print("Not found.")

        except Exception as e:

            print("Error:", e)

        time.sleep(1)

    # Store result

    if location:

        latitudes.append(location.latitude)
        longitudes.append(location.longitude)
        queries_used.append(successful_query)
        geocoding_status.append("Found")

    else:

        latitudes.append(None)
        longitudes.append(None)
        queries_used.append("")
        geocoding_status.append("Not Found")

        print(">>> NOT FOUND <<<")

    # Respect Nominatim rate limit
    time.sleep(1)


# =========================================================
# ADD COORDINATES
# =========================================================

df["Latitude"] = latitudes
df["Longitude"] = longitudes
df["Geocoding_Query"] = queries_used
df["Geocoding_Status"] = geocoding_status


# =========================================================
# SAVE GEOCODED DATA
# =========================================================

df.to_csv(
    GEOCODED_FILE,
    index=False
)


# =========================================================
# SAVE FAILED RECORDS
# =========================================================

failed = df[
    df["Geocoding_Status"] == "Not Found"
]

failed.to_csv(
    FAILED_FILE,
    index=False
)


# =========================================================
# CREATE MAP
# =========================================================

m = folium.Map(
    location=[13.0827, 80.2707],
    zoom_start=11
)


# =========================================================
# ADD MARKERS
# =========================================================

valid_hospitals = df.dropna(
    subset=["Latitude", "Longitude"]
)

for _, row in valid_hospitals.iterrows():

    hospital_name = str(
        row[HOSPITAL_COLUMN]
    )

    address = str(
        row[ADDRESS_COLUMN]
    )

    popup_text = f"""
    <b>{hospital_name}</b><br>
    {address}
    """

    folium.Marker(
        location=[
            row["Latitude"],
            row["Longitude"]
        ],

        popup=folium.Popup(
            popup_text,
            max_width=300
        ),

        tooltip=hospital_name

    ).add_to(m)


# =========================================================
# SAVE MAP
# =========================================================

m.save(MAP_FILE)


# =========================================================
# SUMMARY
# =========================================================

print("\n========================================")
print("GEOCODING COMPLETE")
print("========================================")

print("Total hospitals:", len(df))
print("Successfully found:", len(valid_hospitals))
print("Not found:", len(failed))

print(
    "Success rate:",
    round(
        len(valid_hospitals) / len(df) * 100,
        2
    ),
    "%"
)

print("\nFiles created:")
print(GEOCODED_FILE)
print(FAILED_FILE)
print(MAP_FILE)

print("\nOpen hospital_map.html in your browser.")
