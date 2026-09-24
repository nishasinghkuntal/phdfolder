#!/usr/bin/env python3
# =====================================================================
#  fix_seam_eos.py
# ---------------------------------------------------------------------
#  Repair the crust-core seam in the EoS tables that have one.
#
#  THE BUG
#  -------
#  automationandformatting-2.py line 687 prepends a fixed crust row
#
#      rho = 1.334e14 g/cm3,  P = 6.349e32 dyne/cm2,  n = 0.0789 fm^-3
#
#  to every table, so that the HZD-NV crust always reaches up to the
#  density where the core table is supposed to begin.  It never checks
#  whether the core ALREADY extends below that density.
#
#  Usually it does not, and the result is clean:
#        ... crust n = 0.0789 | core n = 0.07932 ...      monotonic
#
#  But for 11 of the 864 surviving tables the core's first point sits at
#  n = 0.07834, BELOW the appended row:
#        ... crust n = 0.0789 | core n = 0.07834 ...      n goes DOWN
#
#  so baryon density decreases with depth across the join and the
#  pressure jumps about 30% at essentially constant rho.  TOV then
#  integrates through a thermodynamically unstable point.  All 11 sit at
#  high J (40-43), which is exactly where the E_sym constraint lives.
#
#  THE FIX
#  -------
#  Drop the appended row in precisely those files.  The crust then ends
#  at its own last genuine point, n = 0.0475, and the core takes over at
#  n = 0.07834 -- monotonic in rho, P and n, and structurally identical
#  to the healthy tables where the extra row was never needed.
#
#  We do NOT invent a replacement point.  Interpolating a new row across
#  a join between two different equations of state would be inventing
#  physics; removing a row that should never have been written is not.
#
#  Header book-keeping: line 1 is "Itext Imax Icore" with the table
#  written high density first, so Icore counts from the dense end.
#  Removing one CRUST row lowers Imax by one and leaves Icore unchanged.
# =====================================================================

import os
import shutil
import sys

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
EOSDIR = os.path.join(BASE, "NSCool", "EOS")
BACKUP = os.path.join(EOSDIR, "pre_seam_fix_backup")

#  the row automationandformatting-2.py injects, identified by its
#  baryon density rather than by exact string match so that formatting
#  differences cannot make us miss it
JOIN_N = 0.0789
TOL = 1.0e-6


def parse(path):
    lines = open(path).readlines()
    h = lines[0].split()
    n_text, i_max, i_core = int(h[0]), int(h[1]), int(h[2])
    data_idx = []
    n = 0
    for i, ln in enumerate(lines[1 + n_text:], start=1 + n_text):
        p = ln.split()
        if len(p) < 3:
            continue
        try:
            [float(x) for x in p[:3]]
        except ValueError:
            continue
        data_idx.append(i)
        n += 1
        if n == i_max:
            break
    return lines, n_text, i_max, i_core, data_idx


def needs_fix(path):
    """True when the appended row is present AND the core undercuts it."""
    lines, n_text, i_max, i_core, idx = parse(path)
    nb = [float(lines[i].split()[2]) for i in idx]
    # tables run dense-first, so the appended crust row is near the end
    for k in range(len(nb) - 1):
        # a DECREASE in baryon density going outward is the signature
        if abs(nb[k] - JOIN_N) < TOL and nb[k + 1] > nb[k]:
            return k, idx[k]
        if nb[k] < nb[k + 1] and abs(nb[k + 1] - JOIN_N) < TOL:
            return k + 1, idx[k + 1]
    return None, None


def fix(path, dry=False):
    k, line_no = needs_fix(path)
    if k is None:
        return False
    lines, n_text, i_max, i_core, idx = parse(path)
    if dry:
        return True
    if not os.path.isdir(BACKUP):
        os.makedirs(BACKUP)
    shutil.copy2(path, os.path.join(BACKUP, os.path.basename(path)))

    del lines[line_no]
    h = lines[0].rstrip("\n")
    lines[0] = h.replace(str(i_max), str(i_max - 1), 1) + "\n"
    open(path, "w").writelines(lines)
    return True


def main():
    dry = "--apply" not in sys.argv
    import csv
    tags = set()
    with open(os.path.join(BASE, "Data", "cooling_chi2_by_eos.csv")) as f:
        for r in csv.DictReader(f):
            j, l = float(r["J"]), float(r["L"])
            tags.add("%s_J%dp%d_L%dp%d"
                     % (r["model"], int(j), round(j % 1 * 10),
                        int(l), round(l % 1 * 10)))
    hit = []
    for t in sorted(tags):
        p = os.path.join(EOSDIR, "%s_CAT.dat" % t)
        if os.path.isfile(p) and fix(p, dry=dry):
            hit.append(t)
    print(("WOULD FIX" if dry else "FIXED"), len(hit), "tables")
    for t in hit:
        print("   ", t)
    if dry:
        print("\nre-run with --apply to write changes (originals backed up)")


if __name__ == "__main__":
    main()
