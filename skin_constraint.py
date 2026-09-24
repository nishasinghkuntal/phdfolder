#!/usr/bin/env python3
# =====================================================================
#  skin_constraint.py
# ---------------------------------------------------------------------
#  CONSTRAIN J = E_sym AND L = L_sym BY COMPARING THE PREDICTED NEUTRON
#  SKINS WITH THE PREX-II AND CREX MEASUREMENTS.
#
#  Input is the output of run_skin_grid.py: DR_np on a coarse (J, L)
#  subgrid for Pb-208 and Ca-48, for every model whose meson masses are
#  actually published.  This script fits DR_np(J, L) per model, walks
#  the full analysis grid, and forms
#
#     chi2(J,L) = [ (DR_Pb - 0.283)/0.071 ]^2
#               + [ (DR_Ca - 0.121)/0.035 ]^2
#
#  with the two measurements treated as independent.
#
#  A WARNING THAT BELONGS IN THE PAPER, NOT JUST IN THIS COMMENT
#  ------------------------------------------------------------
#  PREX-II wants a THICK Pb-208 skin and CREX wants a THIN Ca-48 skin,
#  and it is well established that no single mean-field model comfortably
#  does both.  So the combined chi2 here is expected to have a poor
#  minimum.  That is a real feature of the data, not a failure of the
#  fit, and this script reports the two experiments separately as well
#  as combined precisely so the tension stays visible instead of being
#  averaged away into a spuriously tight interval.
#
#  It is also why the cooling constraint is worth having: it is an
#  entirely independent handle on the same L, and it can sit with one
#  experiment or the other rather than splitting the difference.
# =====================================================================

import os
import sys
import math

import numpy as np

import rmf_lambda_and_eos as eos
import finite_nucleus_rmf as fn
import run_skin_grid as grid


#  the cooling result this paper already has, for overlay only
COOLING_L_RANGE = (50.0, 100.0)


def load_fits():
    """Fit DR_np = a + b L + c J for each model and nucleus."""
    fits = {}
    for nucleus in ("Pb208", "Ca48"):
        fits[nucleus] = grid.fit_and_report(nucleus)
    return fits


def predict(coef, J, L):
    return coef[0] + coef[1] * L + coef[2] * J


def analyse(fits, models):
    """
    Build chi2 over the analysis (J, L) grid for each model and report
    the implied constraints.
    """
    J_axis = np.arange(25.0, 43.0 + 0.01, 0.5)
    L_axis = np.arange(30.0, 130.0 + 0.01, 1.0)
    JJ, LL = np.meshgrid(J_axis, L_axis, indexing="ij")

    pb = fn.MEASURED_SKIN["Pb208"]
    ca = fn.MEASURED_SKIN["Ca48"]

    for model in models:
        cpb = fits["Pb208"].get(model)
        cca = fits["Ca48"].get(model)
        if cpb is None or cca is None:
            print("\n%s: not enough grid data yet" % model)
            continue

        dpb = predict(cpb, JJ, LL)
        dca = predict(cca, JJ, LL)

        chi_pb = ((dpb - pb["value"]) / pb["sigma"]) ** 2
        chi_ca = ((dca - ca["value"]) / ca["sigma"]) ** 2
        chi_both = chi_pb + chi_ca

        print("\n================  %s  ================" % model)
        for tag, chi in (("PREX-II only", chi_pb),
                         ("CREX only", chi_ca),
                         ("PREX-II + CREX", chi_both)):
            # profile over J to get the constraint on L alone; a
            # one-parameter interval is Delta chi2 <= 1
            prof = np.min(chi, axis=0)
            best = float(np.min(prof))
            ok = L_axis[prof - best <= 1.0]
            okJ = J_axis[np.min(chi, axis=1) - best <= 1.0]
            if len(ok) == 0:
                print("  %-16s no allowed region" % tag)
                continue
            print("  %-16s chi2_min = %6.2f   L = [%5.1f, %5.1f]   "
                  "J = [%4.1f, %4.1f]"
                  % (tag, best, ok.min(), ok.max(), okJ.min(), okJ.max()))

        # how badly do the two experiments disagree for THIS model?
        b_pb = float(np.min(chi_pb))
        b_ca = float(np.min(chi_ca))
        b_bo = float(np.min(chi_both))
        tension = b_bo - b_pb - b_ca
        print("  tension: chi2_both_min - chi2_Pb_min - chi2_Ca_min = %.2f"
              % tension)
        if tension > 4.0:
            print("           -> the two experiments cannot be satisfied "
                  "together by this model (> 2 sigma)")

        # overlap with the cooling result
        prof = np.min(chi_both, axis=0)
        best = float(np.min(prof))
        ok = L_axis[prof - best <= 1.0]
        if len(ok):
            lo = max(ok.min(), COOLING_L_RANGE[0])
            hi = min(ok.max(), COOLING_L_RANGE[1])
            if lo <= hi:
                print("  overlap with cooling L = [%.0f, %.0f]:  L = [%.1f, %.1f]"
                      % (COOLING_L_RANGE[0], COOLING_L_RANGE[1], lo, hi))
            else:
                print("  NO overlap with the cooling range L = [%.0f, %.0f]"
                      % COOLING_L_RANGE)


def main():
    print("Measured skins used:")
    for k, v in fn.MEASURED_SKIN.items():
        print("  %-6s %.3f +- %.3f fm   (%s)"
              % (k, v["value"], v["sigma"], v["source"]))

    fits = load_fits()
    models = sys.argv[1:] if len(sys.argv) > 1 else [
        m for m in eos.MODEL_NAMES if fn.has_finite_nucleus_masses(m)]
    analyse(fits, models)


if __name__ == "__main__":
    main()
