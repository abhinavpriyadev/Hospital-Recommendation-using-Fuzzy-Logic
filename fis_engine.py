"""
fis_engine.py - Mamdani Fuzzy Inference System for hospital suitability (numpy only).

Inputs (exactly three):
  1. Distance_km        crisp 0..50+ (values above 50 are clipped to 50 - never an error)
  2. Cost_norm          crisp 0..100 = RELATIVE cost position inside the comparable specialty group
                        (0 = cheapest known, 100 = most expensive known). None = cost not available.
  3. Emergency_Level    categorical: Confirmed / Partial / Unknown (crisp one-hot fuzzification,
                        so no false ordering is imposed on the three categories)
Context (NOT a fourth input): emergency_request (bool) selects which rule base is used.

Output: Suitability 0..100 (Low / Moderate / High). A ranking score only - not a probability,
clinical success rate or validated quality measure.

Inference: min for AND and implication, max for aggregation, centroid defuzzification.
"""
import numpy as np

# --------------------------------------------------------------------------------------
# Universes
# --------------------------------------------------------------------------------------
DIST_MAX = 50.0
OUT_X = np.linspace(0, 100, 1001)

# --------------------------------------------------------------------------------------
# Membership functions: trapezoid (a, b, c, d); triangle = trapezoid with b == c
# --------------------------------------------------------------------------------------
DIST_SETS = {                       # km  (adjusted so adjacent sets sum to 1 everywhere)
    "Near":     (0, 0, 3, 8),
    "Moderate": (3, 8, 8, 15),
    "Far":      (8, 15, 50, 50),
}
COST_SETS = {                       # relative cost 0..100 (adjusted so adjacent sets sum to 1:
    "Low":    (0, 0, 25, 50),           #  the suggested parameters left a gap with memberships summing
    "Medium": (25, 50, 50, 75),         #  to only 0.8, which made the centroid dip between sets)
    "High":   (50, 75, 100, 100),
}
OUT_SETS = {                        # suitability 0..100
    "Low":      (0, 0, 25, 45),
    "Moderate": (30, 50, 50, 70),
    "High":     (55, 75, 100, 100),
}
EMERGENCY_LEVELS = ("Confirmed", "Partial", "Unknown")


def trap(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    left = np.where(x >= b, 1.0, np.where(b > a, (x - a) / (b - a + 1e-12), 0.0))
    right = np.where(x <= c, 1.0, np.where(d > c, (d - x) / (d - c + 1e-12), 0.0))
    return np.clip(np.minimum(left, right), 0.0, 1.0)


def mu(sets, name, x):
    return float(trap(x, *sets[name]))


# --------------------------------------------------------------------------------------
# Rule bases (consequent = output set).  Key = (distance set, cost set)
# --------------------------------------------------------------------------------------
# Distance x Cost preference table (used directly for ROUTINE requests)
BASE = {
    ("Near", "Low"): "High",       ("Near", "Medium"): "High",       ("Near", "High"): "Moderate",
    ("Moderate", "Low"): "High",   ("Moderate", "Medium"): "Moderate", ("Moderate", "High"): "Low",
    ("Far", "Low"): "Moderate",    ("Far", "Medium"): "Low",         ("Far", "High"): "Low",
}

# ROUTINE configuration: 9 distance x cost rules (weight 1) + 1 low-weight emergency rule.
# Emergency evidence only nudges the result; Partial and Unknown have no influence at all.
ROUTINE_EMERGENCY_WEIGHT = 0.25

# EMERGENCY-REQUEST configuration: full 3 x 3 x 3 = 27 rules, cost de-emphasised, distance emphasised.
# Partial and Unknown are capped at Moderate: without Confirmed evidence the system will not
# call a hospital a High-suitability emergency option. This is evidence-based ranking, NOT a claim
# that Unknown hospitals lack emergency services.
EMERGENCY_TABLE = {
    "Confirmed": {
        ("Near", "Low"): "High",      ("Near", "Medium"): "High",      ("Near", "High"): "High",
        ("Moderate", "Low"): "High",  ("Moderate", "Medium"): "High",  ("Moderate", "High"): "Moderate",
        ("Far", "Low"): "Moderate",   ("Far", "Medium"): "Moderate",   ("Far", "High"): "Low",
    },
    "Partial": {
        ("Near", "Low"): "Moderate",  ("Near", "Medium"): "Moderate",  ("Near", "High"): "Moderate",
        ("Moderate", "Low"): "Moderate", ("Moderate", "Medium"): "Moderate", ("Moderate", "High"): "Low",
        ("Far", "Low"): "Low",        ("Far", "Medium"): "Low",        ("Far", "High"): "Low",
    },
    "Unknown": {
        ("Near", "Low"): "Moderate",  ("Near", "Medium"): "Moderate",  ("Near", "High"): "Low",
        ("Moderate", "Low"): "Moderate", ("Moderate", "Medium"): "Low", ("Moderate", "High"): "Low",
        ("Far", "Low"): "Low",        ("Far", "Medium"): "Low",        ("Far", "High"): "Low",
    },
}


def build_rules(emergency_request):
    """Return list of (dist_set, cost_set, emergency_label_or_None, out_set, weight)."""
    rules = []
    if emergency_request:
        for em, table in EMERGENCY_TABLE.items():
            for (d, c), o in table.items():
                rules.append((d, c, em, o, 1.0))
    else:
        for (d, c), o in BASE.items():
            rules.append((d, c, None, o, 1.0))
        # distance/cost-free nudge: Confirmed evidence slightly raises suitability
        rules.append((None, None, "Confirmed", "High", ROUTINE_EMERGENCY_WEIGHT))
    return rules


def describe_rules(emergency_request):
    out = []
    for i, (d, c, e, o, w) in enumerate(build_rules(emergency_request), 1):
        parts = []
        if d: parts.append(f"Distance is {d}")
        if c: parts.append(f"Cost is {c}")
        if e: parts.append(f"Emergency is {e}")
        out.append(f"R{i:02d}: IF " + " AND ".join(parts) + f" THEN Suitability is {o}" + (f"  [weight {w}]" if w != 1 else ""))
    return out


# --------------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------------
def evaluate(distance_km, cost_norm, emergency_level, emergency_request=False):
    """
    distance_km      float (>=0). Clipped to the 0..50 universe.
    cost_norm        float 0..100, or None when the hospital's cost is not available.
                     Missing cost is NOT replaced by an invented value: all three cost sets are
                     treated as fully possible (membership 1), so the output is the average over
                     every cost scenario instead of a favourable or unfavourable guess.
    emergency_level  'Confirmed' | 'Partial' | 'Unknown' (anything else is treated as Unknown).
    Returns dict(score, fired_rules, memberships, config).
    """
    d = float(np.clip(distance_km, 0.0, DIST_MAX))
    em = emergency_level if emergency_level in EMERGENCY_LEVELS else "Unknown"

    mu_d = {k: mu(DIST_SETS, k, d) for k in DIST_SETS}
    if cost_norm is None:
        mu_c = {k: 1.0 for k in COST_SETS}
    else:
        x = float(np.clip(cost_norm, 0.0, 100.0))
        mu_c = {k: mu(COST_SETS, k, x) for k in COST_SETS}
    mu_e = {k: 1.0 if k == em else 0.0 for k in EMERGENCY_LEVELS}

    agg = np.zeros_like(OUT_X)
    fired = []
    for (ds, cs, es, os_, w) in build_rules(emergency_request):
        terms = []
        if ds: terms.append(mu_d[ds])
        if cs: terms.append(mu_c[cs])
        if es: terms.append(mu_e[es])
        strength = min(terms) * w                                  # AND = min, then rule weight
        if strength <= 0:
            continue
        clipped = np.minimum(strength, trap(OUT_X, *OUT_SETS[os_]))  # implication = min
        agg = np.maximum(agg, clipped)                             # aggregation = max
        fired.append({"rule": " AND ".join(filter(None, [ds and f"Distance={ds}", cs and f"Cost={cs}", es and f"Emergency={es}"])),
                      "then": os_, "strength": round(strength, 3)})
    area = agg.sum()
    if area <= 0:       # cannot happen with the complete rule bases; guard anyway
        raise ValueError("No rule fired - check inputs")
    score = float((OUT_X * agg).sum() / area)                       # centroid
    fired.sort(key=lambda r: -r["strength"])
    return {"score": round(score, 2), "fired_rules": fired,
            "memberships": {"distance": mu_d, "cost": mu_c, "emergency": mu_e},
            "config": "emergency-request" if emergency_request else "routine"}


def label_for(score):
    """Linguistic label = output set with the highest membership at the score."""
    m = {k: mu(OUT_SETS, k, score) for k in OUT_SETS}
    return max(m, key=m.get)
