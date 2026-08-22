import pandas as pd
import folium

# =========================================================
# SETTINGS
# =========================================================

INPUT_FILE = "hospitals_final.csv"
MAP_FILE = "hospital_map.html"

HOSPITAL_COLUMN = "Hospital_Name"
ADDRESS_COLUMN = "Location"
LATITUDE_COLUMN = "Latitude"
LONGITUDE_COLUMN = "Longitude"


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
# CHECK COORDINATES
# =========================================================

print("\nMissing coordinates:")

print(
    df[
        [LATITUDE_COLUMN, LONGITUDE_COLUMN]
    ].isnull().sum()
)


# =========================================================
# CREATE MAP
# =========================================================

# Center of Chennai
m = folium.Map(
    location=[13.0827, 80.2707],
    zoom_start=11
)


# =========================================================
# ADD HOSPITAL MARKERS
# =========================================================

for _, row in df.iterrows():

    hospital_name = str(
        row[HOSPITAL_COLUMN]
    )

    address = str(
        row[ADDRESS_COLUMN]
    )

    latitude = row[LATITUDE_COLUMN]
    longitude = row[LONGITUDE_COLUMN]

    # Skip hospitals without coordinates
    if pd.isna(latitude) or pd.isna(longitude):
        continue

    popup_text = f"""
    <b>{hospital_name}</b><br>
    {address}<br>
    Latitude: {latitude}<br>
    Longitude: {longitude}
    """

    folium.Marker(
        location=[
            latitude,
            longitude
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

print("\nMap created successfully!")
print("File:", MAP_FILE)
