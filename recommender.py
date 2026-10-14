"""
recommender.py - ranks hospitals with the Mamdani FIS (fis_engine.py).

Project layout expected:
    project/
      data/hospitals-model-ready-insurance.csv
      data/specialty-costs-clean.csv
      fis_engine.py   recommender.py   test_fis.py

Use from the Qwen parser output:
    from recommender import recommend, build_map
    res = recommend(specialty=parsed["specialty"], user_lat=lat, user_lon=lon,
                    emergency=parsed["emergency"], procedure=parsed.get("procedure"),
                    service=parsed.get("service"))
    build_map(res, lat, lon, "results_map.html")

CLI demo (no LLM needed):
    python recommender.py --specialty Cardiology --lat 13.0827 --lon 80.2707 --map map.html
    python recommender.py --specialty Cardiology --emergency
    python recommender.py --specialty Cardiology --insurance "Star Health"
"""
import argparse
import math
import os
import re

import numpy as np
import pandas as pd

from fis_engine import evaluate, label_for

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
HOSPITAL_CSV = "hospitals-model-ready-insurance.csv"
COST_CSV = "specialty-costs-clean.csv"

DEFAULT_LAT, DEFAULT_LON = 13.0827, 80.2707   # approximate Chennai city centre (demo default only)

INSURANCE_DISCLAIMER = ("Insurance information is SYNTHETIC DEMONSTRATION DATA. It does not represent actual hospital "
                        "empanelment, policy terms or insurer coverage. Confirm with the hospital and your insurer.")
COST_DISCLAIMER = ("Costs are specialty-level reference estimates (INR), not procedure-specific prices or hospital quotations. "
                   "Cost score is a RELATIVE position among the compared hospitals, not a rupee amount.")
SCORE_DISCLAIMER = ("The suitability score is a ranking aid from a fuzzy-logic prototype. It is not a probability, a clinical "
                    "success rate or a measure of medical quality, and this tool does not diagnose or replace medical advice.")


# ------------------------------------------------------------------------------ helpers
def split_list(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [x.strip() for x in str(value).split(";") if x.strip()]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def load_data(data_dir=DATA_DIR):
    h = pd.read_csv(os.path.join(data_dir, HOSPITAL_CSV), dtype=str, keep_default_na=False)
    h["Latitude"] = h["Latitude"].astype(float)
    h["Longitude"] = h["Longitude"].astype(float)
    c = pd.read_csv(os.path.join(data_dir, COST_CSV))
    return h, c


def normalise_cost(costs):
    """Relative cost 0..100 within ONE comparable specialty group.
    Min-max over the known costs. Returns None for every hospital when the group cannot be compared
    (fewer than 2 known costs, or spread under 5% of the highest cost): hospitals are then treated as
    indistinguishable on cost (cost=50, neutral) rather than stretched to artificial extremes."""
    known = [v for v in costs if v is not None and not pd.isna(v)]
    out = []
    if len(known) < 2 or (max(known) - min(known)) < 0.05 * max(known):
        return [None if (v is None or pd.isna(v)) else 50.0 for v in costs], False
    lo, hi = min(known), max(known)
    for v in costs:
        out.append(None if (v is None or pd.isna(v)) else (v - lo) / (hi - lo) * 100.0)
    return out, True


# ------------------------------------------------------------------------------ explanation
def explain(row, em_request, comparable, n_group):
    bits = [f"{row['Distance_km']:.1f} km away"]
    if row["Estimated_Cost_INR"] is None or pd.isna(row["Estimated_Cost_INR"]):
        bits.append("no cost estimate available (treated as unknown - not assumed cheap or expensive)")
    elif comparable:
        pos = row["Cost_Score"]
        word = "among the lower-cost" if pos <= 33 else ("mid-range" if pos <= 66 else "among the higher-cost")
        bits.append(f"specialty-level estimate about Rs {int(row['Estimated_Cost_INR']):,}, {word} of the {n_group} compared hospitals")
    else:
        bits.append(f"specialty-level estimate about Rs {int(row['Estimated_Cost_INR']):,} (hospitals not distinguishable on cost)")
    lvl = row["Emergency_Level"]
    if lvl == "Confirmed":
        bits.append("emergency/casualty/trauma care is listed in the dataset")
    elif lvl == "Partial":
        bits.append("only ICU/CCU/ambulance is listed - not confirmed full emergency care")
    else:
        bits.append("emergency capability is not established in the dataset (unknown, not 'none')")
    txt = "; ".join(bits)
    mode = "Emergency-request rules: distance and emergency evidence weighted most." if em_request else "Routine rules: distance and cost dominate; emergency evidence has a small effect."
    top = row["_fired"][0] if row["_fired"] else None
    key = f" Strongest rule: IF {top['rule']} THEN {top['then']} (strength {top['strength']})." if top else ""
    return f"{txt}. {mode}{key}"


# ------------------------------------------------------------------------------ main API
def recommend(specialty, user_lat=DEFAULT_LAT, user_lon=DEFAULT_LON, emergency=False,
              procedure=None, service=None, data_dir=DATA_DIR, top_n=None):
    h, c = load_data(data_dir)
    spec_key = str(specialty).strip().lower()
    mask = h["Specialties"].apply(lambda s: spec_key in [x.lower() for x in split_list(s)])
    elig = h[mask].copy()
    if elig.empty:
        return pd.DataFrame()

    canon = next(x for s in elig["Specialties"] for x in split_list(s) if x.lower() == spec_key)  # exact dataset spelling
    cost_lookup = (c[c["Specialty"].str.lower() == spec_key]
                   .set_index("Hospital_ID")[["average_treatment_cost", "minimum_treatment_cost",
                                              "maximum_treatment_cost", "Cost_Basis"]])
    elig = elig.join(cost_lookup, on="Hospital_ID")          # join on Hospital_ID + Specialty, never on name
    elig["Distance_km"] = [haversine_km(user_lat, user_lon, a, b) for a, b in zip(elig["Latitude"], elig["Longitude"])]
    elig["Estimated_Cost_INR"] = elig["average_treatment_cost"]
    elig["Cost_Score"], comparable = normalise_cost(list(elig["Estimated_Cost_INR"]))

    scores, fired = [], []
    for _, r in elig.iterrows():
        res = evaluate(r["Distance_km"], None if pd.isna(r["Cost_Score"]) else r["Cost_Score"],
                       r["Emergency_Level"], emergency_request=bool(emergency))
        scores.append(res["score"]); fired.append(res["fired_rules"])
    elig["Suitability_Score"] = scores
    elig["_fired"] = fired
    elig["Suitability_Label"] = elig["Suitability_Score"].apply(label_for)

    # informational matches only - never used to exclude a hospital
    def match(cell, wanted):
        if not wanted:
            return ""
        return "Yes" if wanted.strip().lower() in [x.lower() for x in split_list(cell)] else "No"
    elig["Procedure_Match"] = elig["Procedures"].apply(lambda s: match(s, procedure))
    elig["Service_Match"] = elig["Services"].apply(lambda s: match(s, service))

    n = len(elig)
    elig["Explanation"] = elig.apply(lambda r: explain(r, bool(emergency), comparable, n), axis=1)
    elig["Specialty"] = canon
    elig["Rule_Set"] = "emergency-request" if emergency else "routine"

    elig = elig.sort_values(["Suitability_Score", "Distance_km", "Cost_Score"],
                            ascending=[False, True, True], na_position="last").reset_index(drop=True)
    elig.insert(0, "Rank", range(1, len(elig) + 1))
    cols = ["Rank", "Hospital_ID", "Hospital_Name", "Address", "Specialty", "Distance_km", "Estimated_Cost_INR",
            "minimum_treatment_cost", "maximum_treatment_cost", "Cost_Basis", "Cost_Score", "Emergency_Level",
            "Emergency_Evidence", "Suitability_Score", "Suitability_Label", "Rule_Set", "Explanation",
            "Procedure_Match", "Service_Match", "Insurance_Providers", "Emergency_Num", "Website",
            "Latitude", "Longitude"]
    res = elig[cols + ["_fired"]].copy()
    res["Distance_km"] = res["Distance_km"].round(2)
    res["Cost_Score"] = res["Cost_Score"].round(1)
    res = res.drop(columns="_fired")
    return res.head(top_n) if top_n else res


def insurance_view(results, provider=None):
    """Separate insurance interface: which of the ranked hospitals list a provider (synthetic data)."""
    rows = []
    for _, r in results.iterrows():
        provs = split_list(r["Insurance_Providers"])
        if provider is None or provider.strip().lower() in [p.lower() for p in provs]:
            rows.append({"Rank": r["Rank"], "Hospital_Name": r["Hospital_Name"], "Providers": "; ".join(provs)})
    return pd.DataFrame(rows), INSURANCE_DISCLAIMER


def build_map(results, user_lat, user_lon, out_path="results_map.html"):
    try:
        import folium
    except ImportError:
        raise SystemExit("folium is not installed: pip install folium")
    m = folium.Map(location=[user_lat, user_lon], zoom_start=11, tiles=None)
    folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles &copy; Esri", name="Esri Streets").add_to(m)
    folium.Marker([user_lat, user_lon], tooltip="Your location", icon=folium.Icon(color="blue", icon="user", prefix="fa")).add_to(m)
    colour = {"High": "green", "Moderate": "orange", "Low": "red"}
    for _, r in results.iterrows():
        cost = ("n/a" if pd.isna(r["Estimated_Cost_INR"]) else f"Rs {int(r['Estimated_Cost_INR']):,} (specialty-level estimate)")
        html = (f"<b>#{r['Rank']} {r['Hospital_Name']}</b><br>Suitability: {r['Suitability_Score']:.1f} ({r['Suitability_Label']})<br>"
                f"Distance: {r['Distance_km']} km<br>Cost: {cost}<br>Emergency evidence: {r['Emergency_Level']}<br>"
                f"<i>Insurance (synthetic demo data): {r['Insurance_Providers']}</i>")
        folium.Marker([r["Latitude"], r["Longitude"]], popup=folium.Popup(html, max_width=320),
                      tooltip=f"#{r['Rank']} {r['Hospital_Name']}",
                      icon=folium.Icon(color=colour.get(r["Suitability_Label"], "gray"))).add_to(m)
    m.save(out_path)
    return out_path


# ------------------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specialty", required=True)
    ap.add_argument("--lat", type=float, default=DEFAULT_LAT)
    ap.add_argument("--lon", type=float, default=DEFAULT_LON)
    ap.add_argument("--emergency", action="store_true")
    ap.add_argument("--procedure"); ap.add_argument("--service")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--map"); ap.add_argument("--insurance")
    a = ap.parse_args()
    res = recommend(a.specialty, a.lat, a.lon, a.emergency, a.procedure, a.service)
    if res.empty:
        raise SystemExit(f"No hospital in the dataset lists the specialty '{a.specialty}'.")
    show = res.head(a.top)
    for _, r in show.iterrows():
        print(f"#{r['Rank']:>2} {r['Suitability_Score']:>5.1f} {r['Suitability_Label']:<8} {r['Hospital_Name']}\n      {r['Explanation']}")
    print("\n" + COST_DISCLAIMER + "\n" + SCORE_DISCLAIMER)
    if a.insurance:
        df, note = insurance_view(res, a.insurance)
        print(f"\nHospitals listing '{a.insurance}':\n{df.to_string(index=False)}\n{note}")
    if a.map:
        print("Map saved:", build_map(show, a.lat, a.lon, a.map))


if __name__ == "__main__":
    main()
