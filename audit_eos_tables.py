#!/usr/bin/env python3
# =====================================================================
#  audit_eos_tables.py
# ---------------------------------------------------------------------
#  Numerical audit of every finished crust+core EoS table that the
#  cooling analysis actually used.
#
#  These are the checks a referee would run, applied to all 864 tables
#  rather than the two or three anyone looks at by eye:
#
#    1. MONOTONIC PRESSURE.  dP/d(rho) > 0 everywhere.  A table that
#       decreases in pressure as density rises is thermodynamically
#       unstable and the TOV integration through it is meaningless.
#    2. MONOTONIC BARYON DENSITY, same reason.
#    3. CAUSALITY.  vs^2 = dP/d(eps) <= 1 through the whole table, not
#       just the core: a superluminal crust row would be just as fatal.
#    4. CRUST-CORE JOIN.  Size of the pressure jump where the HZD-NV
#       crust meets the RMF core.  A large jump means the two halves
#       were not matched, and the star gets an unphysical shell.
#    5. FINITE VALUES.  No NaN, no infinity, no negative pressure.
#
#  Rows are (rho g/cm3, P dyne/cm2, nbar 1/fm3, ...) with the table
#  written from HIGH density down to LOW, so it is reversed here.
# =====================================================================

import os
import glob
import sys

import numpy as np

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
EOSDIR = os.path.join(BASE, "NSCool", "EOS")

C2 = 8.9875e20            # c^2 in cgs, converts g/cm3 to erg/cm3


def read_table(path):
    """
    Read exactly the table NSCool reads.

    The format is strict and must be honoured or the audit invents
    problems that are not there:

        line 1            Itext  Imax  Icore
        next Itext lines  free text
        next Imax lines   the data rows, HIGH density first
        anything after    further sections NSCool reads separately

    An earlier version of this file simply scanned for any line with
    three parseable floats. That swallowed the "6 338 277" header as a
    data row and ran on past the end of the table into the next section,
    producing 350 rows where Imax says 338. Splicing two descending
    blocks together manufactured fake pressure inversions, a fake 10x
    pressure jump and speeds of sound of several thousand c -- in every
    single file, which is what gave the artefact away.

    Returns rho (g/cm3), P (dyne/cm2), nbar (1/fm3) ascending in
    density, plus the index of the crust-core boundary.
    """
    with open(path) as f:
        lines = f.readlines()
    head = lines[0].split()
    n_text, i_max, i_core = int(head[0]), int(head[1]), int(head[2])

    rows = []
    for ln in lines[1 + n_text:]:
        p = ln.split()
        if len(p) < 3:
            continue
        try:
            rows.append([float(x) for x in p[:3]])
        except ValueError:
            continue
        if len(rows) == i_max:
            break
    if len(rows) != i_max:
        return None
    a = np.array(rows)
    flipped = a[0, 0] > a[-1, 0]
    if flipped:
        a = a[::-1]
        i_core = i_max - i_core
    return a[:, 0], a[:, 1], a[:, 2], i_core


def audit(path):
    out = {"file": os.path.basename(path)}
    t = read_table(path)
    if t is None:
        out["fatal"] = "row count does not match Imax in header"
        return out
    rho, P, nb, i_core = t
    out["n_rows"] = len(rho)

    if not (np.all(np.isfinite(rho)) and np.all(np.isfinite(P))
            and np.all(np.isfinite(nb))):
        out["fatal"] = "non-finite entries"
        return out
    if np.any(P <= 0) or np.any(rho <= 0):
        out["fatal"] = "non-positive rho or P"
        return out

    dP = np.diff(P)
    out["n_P_decrease"] = int(np.sum(dP <= 0))
    dn = np.diff(nb)
    out["n_n_decrease"] = int(np.sum(dn <= 0))

    # causality across the whole table: vs^2 = dP/d(eps), eps = rho c^2
    eps = rho * C2
    deps = np.diff(eps)
    good = deps > 0
    vs2 = np.zeros_like(dP)
    vs2[good] = dP[good] / deps[good]
    out["vs2_max"] = float(np.max(vs2)) if len(vs2) else float("nan")
    out["n_acausal"] = int(np.sum(vs2 > 1.0))

    # crust-core join: biggest single-step fractional pressure jump in
    # the density window where the HZD-NV crust is glued on
    #  Compare the jump ACROSS the documented crust-core boundary with
    #  the typical jump between neighbouring rows on either side. The
    #  crust table is coarse, so a large absolute step is normal; what
    #  would signal a bad join is a step much larger than its neighbours.
    j = i_core
    if 2 < j < len(P) - 3:
        step = abs(P[j] - P[j - 1]) / max(P[j - 1], 1e-300)
        near = [abs(P[k] - P[k - 1]) / max(P[k - 1], 1e-300)
                for k in (j - 2, j - 1, j + 1, j + 2)]
        out["join_step"] = float(step)
        out["join_ratio"] = float(step / max(np.median(near), 1e-300))
        out["join_max_dP_frac"] = out["join_ratio"]
    else:
        out["join_max_dP_frac"] = float("nan")
    return out


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "*_CAT.dat"
    files = sorted(f for f in glob.glob(os.path.join(EOSDIR, pat))
                   if "MRcurve" not in os.path.basename(f))
    print("auditing %d tables in %s" % (len(files), EOSDIR))

    fatal, bad_mono, bad_causal, joins = [], [], [], []
    for f in files:
        r = audit(f)
        if r.get("fatal"):
            fatal.append((r["file"], r["fatal"]))
            continue
        if r["n_P_decrease"] or r["n_n_decrease"]:
            bad_mono.append((r["file"], r["n_P_decrease"], r["n_n_decrease"]))
        if r["n_acausal"]:
            bad_causal.append((r["file"], r["vs2_max"], r["n_acausal"]))
        if np.isfinite(r["join_max_dP_frac"]):
            joins.append((r["join_max_dP_frac"], r["file"]))

    print("\n--- 1. unreadable / non-finite / non-positive ---")
    print("  %d files" % len(fatal))
    for n, why in fatal[:10]:
        print("    %s : %s" % (n, why))

    print("\n--- 2. non-monotonic P or n (thermodynamically unstable) ---")
    print("  %d files" % len(bad_mono))
    for n, a, b in bad_mono[:10]:
        print("    %s : %d P-decreases, %d n-decreases" % (n, a, b))

    print("\n--- 3. acausal rows (vs^2 > 1) anywhere in the table ---")
    print("  %d files" % len(bad_causal))
    for n, v, c in sorted(bad_causal, key=lambda x: -x[1])[:10]:
        print("    %s : vs2_max = %.4f in %d rows" % (n, v, c))

    print("\n--- 4. crust-core join: step across boundary / neighbouring steps ---")
    print("    (1.0 = the join is no more abrupt than ordinary table spacing)")
    joins.sort(reverse=True)
    if joins:
        vals = np.array([j[0] for j in joins])
        print("  median %.4f   90th pct %.4f   worst %.4f"
              % (np.median(vals), np.percentile(vals, 90), vals[0]))
        print("  worst offenders:")
        for v, n in joins[:8]:
            print("    %-46s %.4f" % (n, v))
        print("  files where the join step exceeds 3x its neighbours: %d"
              % int(np.sum(vals > 3.0)))


if __name__ == "__main__":
    main()
