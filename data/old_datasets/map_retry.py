import pandas as pd
from geopy.geocoders import Nominatim
import time

INPUT_FILE = "hospitals_geocoded.csv"
OUTPUT_FILE = "hospitals_geocoded_updated.csv"

HOSPITAL_COLUMN = "Hospital_Name"
ADDRESS_COLUMN = "Location"

df = pd.read_csv(INPUT_FILE, low_memory=False)

geolocator = Nominatim(
    user_agent="hospital_recommendation_project",
    timeout=10
)

failed_indices = df[
    df["Geocoding_Status"] == "Not Found"
].index

print("Hospitals needing retry:", len(failed_indices))

for index in failed_indices:

    hospital = str(
        df.loc[index, HOSPITAL_COLUMN]
    ).strip()

    address = str(
        df.loc[index, ADDRESS_COLUMN]
    ).strip()

    queries = [
        f"{hospital}, Chennai, India",
        f"{hospital}, Tamil Nadu, India",
        f"{address}, Chennai, India"
    ]

    print("\n--------------------------------")
    print("Hospital:", hospital)
    print("Address:", address)

    location = None

    for query in queries:

        print("Trying:", query)

        try:
            location = geolocator.geocode(query)

            if location:
                print("FOUND!")
                print(
                    location.latitude,
                    location.longitude
                )
                break

        except Exception as e:
            print("Error:", e)

        time.sleep(1)

    if location:

        df.loc[index, "Latitude"] = location.latitude
        df.loc[index, "Longitude"] = location.longitude
        df.loc[index, "Geocoding_Query"] = query
        df.loc[index, "Geocoding_Status"] = "Found - Retry"

    else:

        print("Still not found.")

    time.sleep(1)


df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nDone!")
print("Saved:", OUTPUT_FILE)
