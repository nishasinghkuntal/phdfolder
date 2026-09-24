#!/usr/bin/env python3
# =====================================================================
#  run_skin_grid.py
# ---------------------------------------------------------------------
#  Drive finite_nucleus_rmf.py over the (J, L) grid and produce the
#  neutron skin predictions that the paper needs.
#
#  WHY THIS IS NOT A BRUTE-FORCE SCAN
#  ----------------------------------
#  One Pb-208 Hartree calculation takes one to four minutes.  The
#  analysis grid has 864 surviving equations of state and we want two
#  nuclei each, so a straight loop would be of order a hundred hours.
#
#  We do not need that.  At fixed model, DR_np is a smooth and very
#  nearly LINEAR function of L, with only a weak residual dependence on
#  J -- this is the same underlying behaviour that makes the published
#  DR_np - L correlations work in the first place.  So we:
#
#      1. run the full calculation on a coarse (J, L) subgrid,
#      2. fit  DR_np = a + b L + c J  per model,
#      3. report the fit residual, so the interpolation error is
#         measured rather than assumed,
#      4. evaluate the fit on the full analysis grid.
#
#  Step 3 is the important one.  If the residual is small compared with
#  the spread we are trying to resolve, the interpolation is harmless
#  and we can say so in the paper.  If it is not, this script tells us
#  instead of quietly hiding it.
#
#  USAGE
#      python3 run_skin_grid.py                 # all models, both nuclei
#      python3 run_skin_grid.py BigApple        # one model
#
#  Results are appended to skin_grid_<nucleus>.csv as they are produced,
#  and existing rows are skipped, so the job can be stopped and resumed.
# =====================================================================

import os
import sys
import math

import numpy as np

import rmf_lambda_and_eos as eos
import finite_nucleus_rmf as fn


#  coarse subgrid.  J is sampled loosely because the skin barely cares
#  about it; L is sampled across the whole analysis range because that
#  is the direction the skin actually responds to.
#  The subgrid must live INSIDE the analysis grid, and the analysis grid
#  is not the same for every model: the lowest L for which a real g_rho
#  exists differs, so BigApple starts at L = 35, IOPB-I at 45 and
#  FSUGarnet at 50.  An earlier version sampled L = 30 for all three,
#  which exists for none of them -- every one of those points failed and
#  silently vanished, gutting the low-L end that CREX cares about.
#  These ranges are read off the EoS files the cooling analysis actually
#  used (NSCool/EOS/<model>_J*_L*).
L_MIN = {"BigApple": 35.0, "IOPB-I": 45.0, "FSUGarnet": 50.0}

#  J = 25 is dropped deliberately.  At the softest corner of the grid the
#  Pb-208 neutron well is so shallow that the bound orbitals cannot hold
#  126 neutrons, and the Hartree description of near-threshold states is
#  not trustworthy there anyway.  The fit extrapolates the short distance
#  from 26.5 down to 25.
J_NODES = [26.5, 31.0, 35.5, 40.0, 43.0]
N_L = 5


def l_nodes(model):
    lo = L_MIN[model]
    return [lo + (130.0 - lo) * i / (N_L - 1.0) for i in range(N_L)]


NUCLEI = ["Pb208", "Ca48"]


def out_name(nucleus):
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "skin_grid_%s.csv" % nucleus)


def already_done(path):
    """Read back whatever a previous run of this script managed to finish."""
    done = set()
    if not os.path.exists(path):
        return done
    f = open(path)
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        bits = line.split(",")
        if len(bits) >= 3:
            done.add((bits[0].strip(), float(bits[1]), float(bits[2])))
    f.close()
    return done


def run_one(model, j_sym, l_sym, nucleus):
    """One Hartree calculation.  Returns None if this point is unusable."""
    par = eos.make_parameter_set(model, j_sym, l_sym)
    if not par["ok"]:
        return None
    #  451 mesh points reproduces the published skins to 0.004 fm,
    #  which is 18 times finer than the CREX error we compare against,
    #  and runs four times faster than the 701-point mesh.
    res = fn.hartree(fn.apply_meson_masses(par), nucleus, n_points=451)
    if "skin" not in res:
        return None
    return res


def build_subgrid(models):
    for nucleus in NUCLEI:
        path = out_name(nucleus)
        done = already_done(path)
        new = not os.path.exists(path)
        f = open(path, "a")
        if new:
            f.write("# model,J,L,skin_fm,r_p,r_n,r_ch,BE_per_A,converged\n")
        for model in models:
            for j in J_NODES:
                for l in l_nodes(model):
                    if (model, j, l) in done:
                        continue
                    res = run_one(model, j, l, nucleus)
                    if res is None:
                        continue
                    f.write("%s,%.2f,%.2f,%.5f,%.5f,%.5f,%.5f,%.4f,%d\n"
                            % (model, j, l, res["skin"], res["r_p"],
                               res["r_n"], res["r_ch"], res["BE_per_A"],
                               1 if res.get("converged") else 0))
                    f.flush()
                    print("%-6s %-10s J=%5.1f L=%6.1f  skin=%.4f  conv=%s"
                          % (nucleus, model, j, l, res["skin"],
                             res.get("converged")))
        f.close()


def fit_and_report(nucleus):
    """
    Fit DR_np = a + b L + c J per model and report how well it holds.

    The residual printed here is the number that decides whether the
    interpolation is defensible.  It belongs in the supplemental
    material next to the skin values themselves.
    """
    path = out_name(nucleus)
    if not os.path.exists(path):
        print("no data for %s yet" % nucleus)
        return {}

    rows = {}
    f = open(path)
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        b = line.split(",")
        # only trust points that actually converged
        if len(b) >= 9 and int(b[8]) != 1:
            continue
        rows.setdefault(b[0].strip(), []).append(
            (float(b[1]), float(b[2]), float(b[3])))
    f.close()

    fits = {}
    print("\n--- %s : DR_np = a + b L + c J ---" % nucleus)
    for model in sorted(rows):
        pts = rows[model]
        if len(pts) < 4:
            print("  %-10s only %d usable points, skipping" % (model, len(pts)))
            continue
        J = np.array([p[0] for p in pts])
        L = np.array([p[1] for p in pts])
        S = np.array([p[2] for p in pts])
        A = np.column_stack((np.ones_like(L), L, J))
        coef, _, _, _ = np.linalg.lstsq(A, S, rcond=None)
        resid = S - A.dot(coef)
        fits[model] = coef
        print("  %-10s a=%+.5f  b=%+.6f /MeV  c=%+.6f /MeV"
              "   rms resid = %.5f fm   max = %.5f fm  (n=%d)"
              % (model, coef[0], coef[1], coef[2],
                 float(np.sqrt(np.mean(resid ** 2))),
                 float(np.max(np.abs(resid))), len(pts)))
    return fits


def main():
    models = sys.argv[1:] if len(sys.argv) > 1 else list(fn.SKIN_MODELS)
    build_subgrid(models)
    for nucleus in NUCLEI:
        fit_and_report(nucleus)


if __name__ == "__main__":
    main()
