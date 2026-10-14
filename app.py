"""
app.py - connects your Qwen/Ollama parser (input_model/ollama-1-3.py) to the FIS recommender.

Flow:  user text -> extract_medical_intent() (Qwen) -> validate against dataset lists
       -> recommend() (distance + cost + Mamdani FIS) -> ranked list + explanations + Folium map

Run from the project root:
    py app.py
    py app.py --lat 13.05 --lon 80.25 --top 5 --map results_map.html
"""
import argparse
import importlib.util
import json
import os
import re

from recommender import (recommend, build_map, insurance_view,
                         DEFAULT_LAT, DEFAULT_LON, COST_DISCLAIMER, SCORE_DISCLAIMER, INSURANCE_DISCLAIMER)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARSER_FILE = os.path.join(BASE_DIR, "input_model", "ollama-1-3.py")   # file name has hyphens -> load with importlib


def load_parser():
    """Loads your parser as a module. This starts Ollama if needed and loads the hospital CSV
    (both happen at import time in the parser), but does NOT start its own input loop."""
    spec = importlib.util.spec_from_file_location("hospital_parser", PARSER_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _json_from(text):
    """Qwen is asked for JSON only, but tolerate ```json fences or stray text around it."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def _match(value, allowed, norm):
    """Return the dataset's exact spelling if value matches an allowed item, else None."""
    if not value:
        return None
    lookup = {norm(a): a for a in allowed}
    return lookup.get(norm(value))


def parse_request(parser, text):
    raw, runtime = parser.extract_medical_intent(text)
    data = _json_from(raw)
    norm = parser.normalize
    spec = _match(data.get("specialty"), parser.SPECIALTIES, norm)
    proc = _match(data.get("procedure"), parser.PROCEDURES, norm)
    serv = _match(data.get("service"), parser.SERVICES, norm)
    for key, got, kept in (("specialty", data.get("specialty"), spec), ("procedure", data.get("procedure"), proc),
                           ("service", data.get("service"), serv)):
        if got and not kept:
            print(f"(Ignored {key} '{got}': not in the hospital database lists.)")
    emergency = data.get("emergency") in (True, 1, "true", "True", "TRUE")
    return {"specialty": spec, "procedure": proc, "service": serv, "emergency": emergency}, runtime


def run_request(parsed, lat, lon, top=5, map_path=None):
    if not parsed.get("specialty"):
        print("Could not identify a supported specialty in that request. Try rephrasing (e.g. 'heart specialist', 'eye doctor').")
        return None
    res = recommend(parsed["specialty"], lat, lon, parsed["emergency"], parsed.get("procedure"), parsed.get("service"))
    if res.empty:
        print(f"No hospital in the dataset lists '{parsed['specialty']}'.")
        return None
    mode = "EMERGENCY-related request" if parsed["emergency"] else "routine request"
    print(f"\nSpecialty: {parsed['specialty']} | {mode}"
          + (f" | procedure: {parsed['procedure']}" if parsed.get("procedure") else "")
          + (f" | service: {parsed['service']}" if parsed.get("service") else ""))
    if parsed["emergency"]:
        print("If this is a real emergency, call your local emergency number (112 in India) - do not rely on this tool.")
    print(f"{len(res)} eligible hospitals; showing top {min(top, len(res))}\n")
    for _, r in res.head(top).iterrows():
        extra = ""
        if r["Procedure_Match"]:
            extra += f" | procedure listed: {r['Procedure_Match']}"
        if r["Service_Match"]:
            extra += f" | service listed: {r['Service_Match']}"
        print(f"#{r['Rank']} {r['Hospital_Name']}  - score {r['Suitability_Score']:.1f} ({r['Suitability_Label']}){extra}")
        print(f"    {r['Explanation']}")
        print(f"    Insurance (synthetic demo data): {r['Insurance_Providers']}")
    print("\n" + COST_DISCLAIMER + "\n" + SCORE_DISCLAIMER + "\n" + INSURANCE_DISCLAIMER)
    if map_path:
        print("Map saved:", build_map(res.head(top), lat, lon, map_path))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=DEFAULT_LAT)
    ap.add_argument("--lon", type=float, default=DEFAULT_LON)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--map", default="results_map.html")
    a = ap.parse_args()

    parser = load_parser()
    print("\nHospital recommendation prototype (decision support only - not a diagnosis).")
    print("Type 'exit' to quit. After a search, type 'insurance <provider>' to filter, e.g. 'insurance Star Health'.")
    last = None
    while True:
        text = input("\nPatient: ").strip()
        if text.lower() in ("exit", "quit", "q"):
            break
        if not text:
            continue
        if text.lower().startswith("insurance "):
            if last is None:
                print("Run a search first.")
                continue
            df, note = insurance_view(last, text[10:].strip())
            print(df.to_string(index=False) if not df.empty else "No eligible hospital lists that provider.")
            print(note)
            continue
        try:
            parsed, runtime = parse_request(parser, text)
        except Exception as e:      # Ollama problem, invalid JSON, etc.
            print(f"Parser error: {e}")
            continue
        print(f"Parsed: {parsed}   (Qwen {runtime:.1f}s)")
        result = run_request(parsed, a.lat, a.lon, a.top, a.map)
        if result is not None:
            last = result


if __name__ == "__main__":
    main()
