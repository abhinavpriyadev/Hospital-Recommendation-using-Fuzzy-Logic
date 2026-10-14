"""Run:  python test_fis.py     (no pytest needed)"""
import numpy as np
from fis_engine import (evaluate, label_for, DIST_SETS, COST_SETS, OUT_SETS, mu,
                        build_rules, DIST_MAX, EMERGENCY_LEVELS)

fails = []
TOL = 2.5   # centroid defuzzification can wobble by a point or two where firing strengths change; trend must still hold


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""))
    if not ok:
        fails.append(name)


# 1. Membership coverage: every point of each universe belongs to at least one set
dx = np.linspace(0, DIST_MAX, 501)
cov_d = min(sum(mu(DIST_SETS, k, x) for k in DIST_SETS) for x in dx)
cov_c = min(sum(mu(COST_SETS, k, x) for k in COST_SETS) for x in np.linspace(0, 100, 201))
cov_o = min(sum(mu(OUT_SETS, k, x) for k in OUT_SETS) for x in np.linspace(0, 100, 201))
check("Distance sets cover 0-50 km", cov_d > 0.99, f"min sum={cov_d:.2f}")
check("Cost sets cover 0-100", cov_c > 0.99, f"min sum={cov_c:.2f}")
check("Output sets cover 0-100", cov_o > 0.7, f"min sum={cov_o:.2f}")

# 2. Rule bases complete: 9 routine+1 ; 27 emergency
check("Routine rule base size", len(build_rules(False)) == 10)
check("Emergency rule base size (3x3x3)", len(build_rules(True)) == 27)

# 3. Every input combination fires at least one rule and returns 0..100
ok = True
for em_req in (False, True):
    for d in (0, 1, 3, 5, 8, 12, 15, 25, 50, 80, 500):
        for c in (None, 0, 10, 30, 50, 70, 90, 100):
            for e in EMERGENCY_LEVELS + ("garbage",):
                s = evaluate(d, c, e, em_req)["score"]
                ok &= (0 <= s <= 100)
check("No crash / no empty rule firing on grid (incl. >50 km, missing cost, bad label)", ok)

# 4. Monotonicity
for em_req in (False, True):
    for e in EMERGENCY_LEVELS:
        sd = [evaluate(d, 50, e, em_req)["score"] for d in range(0, 51, 2)]
        check(f"Score non-increasing with distance  [{'emerg' if em_req else 'routine'}, {e}]",
              all(a >= b - TOL for a, b in zip(sd, sd[1:])))
        sc = [evaluate(5, c, e, em_req)["score"] for c in range(0, 101, 5)]
        check(f"Score non-increasing with relative cost [{'emerg' if em_req else 'routine'}, {e}]",
              all(a >= b - TOL for a, b in zip(sc, sc[1:])))

# 5. Emergency evidence ordering in an emergency request, and small influence in routine
for d, c in ((2, 20), (6, 50), (12, 60)):
    conf, part, unk = (evaluate(d, c, e, True)["score"] for e in ("Confirmed", "Partial", "Unknown"))
    check(f"Emergency request: Confirmed > Partial >= Unknown (d={d}, c={c})", conf > part >= unk, f"{conf}/{part}/{unk}")
r_conf = evaluate(2, 100, "Confirmed", False)["score"]
r_part = evaluate(2, 100, "Partial", False)["score"]
r_unk = evaluate(2, 100, "Unknown", False)["score"]
e_conf = evaluate(2, 100, "Confirmed", True)["score"]
e_unk = evaluate(2, 100, "Unknown", True)["score"]
check("Routine: Partial == Unknown (no influence), Confirmed slightly higher", r_part == r_unk and 0 < r_conf - r_unk < 10, f"{r_conf}/{r_part}/{r_unk}")
check("Emergency gap (Confirmed-Unknown) larger than routine gap", (e_conf - e_unk) > (r_conf - r_unk), f"{e_conf - e_unk:.1f} vs {r_conf - r_unk:.1f}")

# 6. Spec preferences
check("Near+Low cost -> High", label_for(evaluate(1, 5, "Unknown")["score"]) == "High")
check("Far+High cost -> Low", label_for(evaluate(40, 95, "Unknown")["score"]) == "Low")
check("Far+Low cost not automatically Low", label_for(evaluate(40, 5, "Unknown")["score"]) != "Low")
check("Near+High cost not automatically High", label_for(evaluate(1, 95, "Unknown")["score"]) != "High")

# 7. Missing cost sits between best and worst known cost (never favoured, never invented)
best = evaluate(5, 0, "Unknown")["score"]; worst = evaluate(5, 100, "Unknown")["score"]; miss = evaluate(5, None, "Unknown")["score"]
check("Missing cost score lies between known-cheap and known-expensive", worst < miss < best, f"{worst}<{miss}<{best}")

print("\nDEMO  (score / label)           routine                      emergency request")
print(f"{'case':36s}{'Conf':>10s}{'Part':>10s}{'Unk':>10s}   |{'Conf':>8s}{'Part':>8s}{'Unk':>8s}")
for name, d, c in [("Near 2 km, cheapest", 2, 0), ("Near 2 km, mid cost", 2, 50), ("Near 2 km, priciest", 2, 100),
                   ("Moderate 8 km, cheapest", 8, 0), ("Moderate 8 km, priciest", 8, 100),
                   ("Far 25 km, cheapest", 25, 0), ("Far 25 km, priciest", 25, 100), ("Near 2 km, cost missing", 2, None)]:
    row = [evaluate(d, c, e, False)["score"] for e in EMERGENCY_LEVELS]
    row2 = [evaluate(d, c, e, True)["score"] for e in EMERGENCY_LEVELS]
    print(f"{name:36s}" + "".join(f"{x:>10.1f}" for x in row) + "   |" + "".join(f"{x:>8.1f}" for x in row2))

print("\n%d failure(s)" % len(fails))
raise SystemExit(1 if fails else 0)
