"""
server.py - web front end for the Fuzzy Logic-Based Hospital Recommendation System.

Run from the project root:
    py -m pip install flask
    py server.py
then open  http://127.0.0.1:5000

Endpoints
    GET  /             the single-page app (static/index.html)
    GET  /api/meta     specialties, insurance providers, whether the Qwen parser is available
    POST /api/search   {text | specialty, emergency, lat, lon, provider, only_provider, show_all}

Insurance is NOT a FIS input: choosing a provider only filters/highlights results after ranking.
Insurance data is synthetic demonstration data.
"""
import contextlib
import io
import json
import os

from flask import Flask, jsonify, request, send_from_directory

from recommender import (recommend, load_data, split_list, DEFAULT_LAT, DEFAULT_LON,
                         COST_DISCLAIMER, SCORE_DISCLAIMER, INSURANCE_DISCLAIMER)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
web = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")

# ---- Qwen parser (optional: manual specialty search still works if Ollama/Qwen is unavailable)
PARSER = None
PARSER_ERROR = None
parse_request = None
if os.environ.get("HOSPITAL_NO_PARSER") != "1":
    try:
        from app import load_parser, parse_request as _parse_request
        PARSER = load_parser()
        parse_request = _parse_request
    except Exception as e:                      # Ollama not installed / cannot start / CUDA problems ...
        PARSER_ERROR = str(e)
        print("WARNING: Qwen parser unavailable - manual search only.\n", e)

HOSP, _COSTS = load_data()
SPECIALTIES = sorted({s for cell in HOSP["Specialties"] for s in split_list(cell)})
PROVIDERS = sorted({p for cell in HOSP["Insurance_Providers"] for p in split_list(cell)})


def _num(v, default, lo, hi):
    try:
        v = float(v)
        return v if lo <= v <= hi else default
    except (TypeError, ValueError):
        return default


@web.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@web.get("/api/meta")
def meta():
    return jsonify({"specialties": SPECIALTIES, "providers": PROVIDERS,
                    "parser_available": PARSER is not None, "parser_error": PARSER_ERROR,
                    "default_location": [DEFAULT_LAT, DEFAULT_LON],
                    "disclaimers": {"cost": COST_DISCLAIMER, "score": SCORE_DISCLAIMER, "insurance": INSURANCE_DISCLAIMER}})


@web.post("/api/search")
def search():
    body = request.get_json(silent=True) or {}
    lat = _num(body.get("lat"), DEFAULT_LAT, -90, 90)
    lon = _num(body.get("lon"), DEFAULT_LON, -180, 180)
    provider = (body.get("provider") or "").strip()
    only = bool(body.get("only_provider"))
    notes, runtime = [], None

    text = (body.get("text") or "").strip()
    if text:
        if PARSER is None:
            return jsonify({"error": "The Qwen parser is not available right now. Use the manual specialty selector. "
                                     + (f"({PARSER_ERROR})" if PARSER_ERROR else "")}), 503
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                parsed, runtime = parse_request(PARSER, text)
        except Exception as e:
            return jsonify({"error": f"Could not understand the request: {e}"}), 502
        notes = [ln.strip("() ") for ln in buf.getvalue().splitlines() if ln.strip()]
    else:
        spec = (body.get("specialty") or "").strip()
        match = next((s for s in SPECIALTIES if s.lower() == spec.lower()), None)
        if not match:
            return jsonify({"error": "Choose a supported specialty."}), 400
        parsed = {"specialty": match, "procedure": body.get("procedure") or None,
                  "service": body.get("service") or None, "emergency": bool(body.get("emergency"))}

    if not parsed.get("specialty"):
        return jsonify({"parsed": parsed, "results": [], "notes": notes, "runtime": runtime,
                        "error": "No supported specialty was identified in that request. Try rephrasing, or pick a specialty manually."}), 200

    res = recommend(parsed["specialty"], lat, lon, parsed["emergency"], parsed.get("procedure"), parsed.get("service"))
    total = len(res)
    if provider:
        res["_has_provider"] = res["Insurance_Providers"].apply(lambda s: provider.lower() in [p.lower() for p in split_list(s)])
        if only:
            res = res[res["_has_provider"]]
        res = res.drop(columns="_has_provider")
    matching = len(res)
    if not body.get("show_all"):
        res = res.head(10)

    records = json.loads(res.to_json(orient="records"))      # NaN -> null, numpy -> plain numbers
    for r in records:
        r["Insurance_Providers"] = split_list(r["Insurance_Providers"])
    return jsonify({"parsed": parsed, "runtime": runtime, "notes": notes, "results": records,
                    "total_eligible": total, "matching": matching, "provider": provider,
                    "only_provider": only, "user": [lat, lon]})


if __name__ == "__main__":
    print("\nOpen http://127.0.0.1:5000 in your browser.  Press Ctrl+C to stop.")
    web.run(host="127.0.0.1", port=5000, debug=False)
