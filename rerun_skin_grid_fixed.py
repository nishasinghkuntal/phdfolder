#!/usr/bin/env python3
# =====================================================================
#  rerun_skin_grid_fixed.py
# ---------------------------------------------------------------------
#  Re-run every point of skin_grid_Pb208.csv / skin_grid_Ca48.csv, and
#  the six validation points of Das et al. (2021), with the corrected
#  convergence test in finite_nucleus_rmf.hartree (proton radius must be
#  stable too, and at least MIN_SWEEPS sweeps).  Writes
#  skin_grid_fixed_<nucleus>.csv and skin_validation_fixed.csv beside
#  the originals, which are left untouched for comparison.
#  Uses 4 worker processes.
# =====================================================================
import csv
import os
import sys
from multiprocessing import Pool

import rmf_lambda_and_eos as eos
import finite_nucleus_rmf as fn

HERE = os.path.dirname(os.path.abspath(__file__))
PUBLISHED_JL = {"BigApple": (31.32, 39.80), "FSUGarnet": (30.95, 51.04),
                "IOPB-I": (33.30, 63.58)}


def one(task):
    tag, model, j, l, nucleus = task
    par = eos.make_parameter_set(model, j, l)
    if not par["ok"]:
        return task, None
    res = fn.hartree(fn.apply_meson_masses(par), nucleus, n_points=451)
    return task, res


def main():
    tasks = []
    for nucleus in ("Pb208", "Ca48"):
        path = os.path.join(HERE, "skin_grid_%s.csv" % nucleus)
        for line in open(path):
            if line.startswith("#") or not line.strip():
                continue
            b = line.split(",")
            tasks.append(("grid", b[0], float(b[1]), float(b[2]), nucleus))
    for m, (j, l) in PUBLISHED_JL.items():
        for nucleus in ("Pb208", "Ca48"):
            tasks.append(("valid", m, j, l, nucleus))
    # longest (Pb) first so the pool stays busy
    tasks.sort(key=lambda t: t[4] != "Pb208")
    outs = {}
    for name in ("Pb208", "Ca48"):
        f = open(os.path.join(HERE, "skin_grid_fixed_%s.csv" % name), "w")
        f.write("# model,J,L,skin_fm,r_p,r_n,r_ch,BE_per_A,converged,iterations\n")
        outs[name] = f
    fv = open(os.path.join(HERE, "skin_validation_fixed.csv"), "w")
    fv.write("# model,nucleus,J,L,skin_fm,r_ch,BE_per_A,converged,iterations\n")
    done = 0
    with Pool(4) as pool:
        for task, res in pool.imap_unordered(one, tasks):
            done += 1
            tag, m, j, l, nucleus = task
            if res is None or "skin" not in res:
                print("FAIL", task, res and res.get("fail"), flush=True)
                continue
            if tag == "grid":
                outs[nucleus].write("%s,%.2f,%.2f,%.5f,%.5f,%.5f,%.5f,%.4f,%d,%d\n"
                                    % (m, j, l, res["skin"], res["r_p"], res["r_n"],
                                       res["r_ch"], res["BE_per_A"],
                                       1 if res.get("converged") else 0,
                                       res.get("iterations", -1)))
                outs[nucleus].flush()
            else:
                fv.write("%s,%s,%.2f,%.2f,%.5f,%.5f,%.4f,%d,%d\n"
                         % (m, nucleus, j, l, res["skin"], res["r_ch"],
                            res["BE_per_A"], 1 if res.get("converged") else 0,
                            res.get("iterations", -1)))
                fv.flush()
            print("%3d/%d %s %-9s %s J=%.2f L=%.2f skin=%.4f it=%s"
                  % (done, len(tasks), tag, m, nucleus, j, l, res["skin"],
                     res.get("iterations")), flush=True)
    for f in outs.values():
        f.close()
    fv.close()


if __name__ == "__main__":
    main()
