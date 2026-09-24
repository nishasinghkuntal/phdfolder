#!/usr/bin/env python3
# =====================================================================
#  skin_constraints.py
# ---------------------------------------------------------------------
#  CONSTRAIN L_sym AND E_sym BY COMPARING OUR OWN PREDICTED NEUTRON
#  SKINS WITH THE PREX-2 AND CREX MEASUREMENTS
# ---------------------------------------------------------------------
#
#  This is the step that turns the finite-nucleus machinery into a
#  physics result.  finite_nucleus_rmf.py predicts, for each (model,
#  J, L) point, the neutron skin of Pb-208 and Ca-48 from the SAME
#  Lagrangian that produced the equation of state and the cooling
#  curves.  Two parity-violating electron scattering experiments have
#  measured exactly those two numbers.  Demanding that the prediction
#  agree with the measurement therefore constrains (J, L) directly --
#  with no reference to any published DR_np - L correlation fitted
#  across other people's models.
#
#  THE TWO MEASUREMENTS
#  --------------------
#    PREX-2 (Pb-208):  DR_np = 0.283 +- 0.071 fm
#    CREX   (Ca-48) :  DR_np = 0.121 +- 0.026 (exp) +- 0.024 (model) fm
#
#  They pull in OPPOSITE directions.  A thick Pb skin wants a stiff
#  symmetry energy (large L); a thin Ca skin wants a soft one (small
#  L).  No mean-field model reproduces both, and that disagreement is
#  the well-known PREX-CREX puzzle.  This script does not try to hide
#  it: it reports the constraint from each experiment separately AND
#  jointly, and prints the joint chi-square so the size of the tension
#  is visible rather than buried.
#
#  WHAT TO DO WITH THE ANSWER
#  --------------------------
#  The output is three bands in the (J, L) plane -- PREX-2, CREX, and
#  the cooling constraint already in the paper.  Where the cooling band
#  sits relative to the other two IS the result.  It is an independent,
#  astrophysical vote in a dispute that has so far been purely
#  laboratory-based.
#
#  Run run_skin_grid.py first to generate skin_grid_<nucleus>.csv.
# =====================================================================

import os
import sys

import numpy as np

import finite_nucleus_rmf as fn


#  measurements.  The CREX error is the experimental and model
#  uncertainties added in quadrature, which is how CREX itself quotes
#  the combined number.
PREX2 = {"nucleus": "Pb208", "skin": 0.283, "sigma": 0.071,
         "ref": "Adhikari et al. (PREX Collaboration), PRL 126, 172502 (2021)"}
CREX = {"nucleus": "Ca48", "skin": 0.121,
        "sigma": float(np.hypot(0.026, 0.024)),
        "ref": "Adhikari et al. (CREX Collaboration), PRL 129, 042501 (2022)"}

#  the cooling constraint this paper already has, for comparison
COOLING_L = (50.0, 100.0)


def load_grid(nucleus):
    """Read the skin subgrid produced by run_skin_grid.py."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "skin_grid_%s.csv" % nucleus)
    rows = {}
    if not os.path.exists(path):
        return rows
    f = open(path)
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        b = line.split(",")
        if len(b) < 9 or int(b[8]) != 1:
            continue                      # only converged points
        rows.setdefault(b[0].strip(), []).append(
            (float(b[1]), float(b[2]), float(b[3])))
    f.close()
    return rows


def _design(J, L):
    """
    Basis for the skin surface: 1, L, L^2, J, J L.

    The quadratic in L is not cosmetic.  DR_np rises steeply with L and
    then SATURATES -- for BigApple at J = 32.5 it climbs from 0.195 fm at
    L = 50 to 0.243 fm at L = 90 and is then flat out to L = 130.  A
    plain linear fit averages the steep part against the flat part, which
    both understates the true low-L slope by a factor of two and leaves
    an rms residual of 0.010-0.014 fm.  That is a third of the CREX error
    we are comparing against, i.e. far too coarse to constrain anything.
    """
    return np.column_stack((np.ones_like(L), L, L * L, J, J * L))


def fit_surface(points):
    """
    Fit the skin surface and return the coefficients and rms residual.

    The residual is the honest measure of what the interpolation costs
    us.  It must stay well under the experimental errors (0.071 fm for
    PREX-2, 0.035 fm for CREX) or none of this constrains anything.
    """
    J = np.array([p[0] for p in points])
    L = np.array([p[1] for p in points])
    S = np.array([p[2] for p in points])
    A = _design(J, L)
    coef, _, _, _ = np.linalg.lstsq(A, S, rcond=None)
    resid = S - A.dot(coef)
    return coef, float(np.sqrt(np.mean(resid ** 2)))


def skin_of(coef, J, L):
    Ja = np.atleast_1d(np.asarray(J, dtype=float))
    La = np.atleast_1d(np.asarray(L, dtype=float))
    Ja, La = np.broadcast_arrays(Ja, La)
    v = _design(Ja.ravel(), La.ravel()).dot(coef)
    return float(v[0]) if v.size == 1 else v.reshape(Ja.shape)


def band_from_one(coef, meas, J_grid, L_grid, n_sigma=1.0):
    """
    L range allowed by a single measurement, at each J.

    Solving  |a + b L + c J - skin| <= n_sigma * error  for L.
    """
    out = {}
    for J in J_grid:
        lo, hi = None, None
        for L in L_grid:
            d = abs(skin_of(coef, J, L) - meas["skin"])
            if d <= n_sigma * meas["sigma"]:
                lo = L if lo is None else lo
                hi = L
        out[J] = (lo, hi)
    return out


def main():
    J_grid = np.arange(25.0, 43.5, 0.5)
    L_grid = np.arange(20.0, 145.5, 0.5)

    pb = load_grid("Pb208")
    ca = load_grid("Ca48")
    if not pb:
        print("No Pb208 grid yet -- run run_skin_grid.py first.")
        return

    print("=" * 74)
    print("NEUTRON SKIN CONSTRAINTS ON (E_sym, L_sym)")
    print("=" * 74)
    print("PREX-2 : DR_np(208Pb) = %.3f +- %.3f fm" % (PREX2["skin"], PREX2["sigma"]))
    print("CREX   : DR_np( 48Ca) = %.3f +- %.3f fm" % (CREX["skin"], CREX["sigma"]))
    print()

    fits = {}
    for name, data in (("Pb208", pb), ("Ca48", ca)):
        print("--- %s : DR_np = a + b L + c J ---" % name)
        for model in sorted(data):
            if len(data[model]) < 4:
                print("  %-10s only %d points, skipping" % (model, len(data[model])))
                continue
            coef, rms = fit_surface(data[model])
            fits[(model, name)] = coef
            # report local slopes at a representative point instead of
            # raw coefficients, which are not individually meaningful
            dL = (skin_of(coef, 32.5, 71.0) - skin_of(coef, 32.5, 69.0)) / 2.0
            dJ = (skin_of(coef, 33.5, 70.0) - skin_of(coef, 31.5, 70.0)) / 2.0
            print("  %-10s at (J=32.5,L=70): d(skin)/dL = %+.5f fm/MeV   "
                  "d(skin)/dJ = %+.5f fm/MeV   rms resid = %.4f fm  (n=%d)"
                  % (model, dL, dJ, rms, len(data[model])))
        print()

    # -----------------------------------------------------------------
    #  constraint from each experiment separately
    # -----------------------------------------------------------------
    for meas, tag in ((PREX2, "PREX-2"), (CREX, "CREX")):
        nuc = meas["nucleus"]
        print("--- %s alone : allowed L (1 sigma), at three J values ---" % tag)
        for model in sorted(set(m for (m, n) in fits if n == nuc)):
            coef = fits[(model, nuc)]
            band = band_from_one(coef, meas, [26.5, 32.5, 38.5], L_grid)
            bits = []
            for J in (26.5, 32.5, 38.5):
                lo, hi = band[J]
                bits.append("J=%.1f: %s" % (
                    J, "none" if lo is None else "[%.0f, %.0f]" % (lo, hi)))
            print("  %-10s %s" % (model, "   ".join(bits)))
        print()

    # -----------------------------------------------------------------
    #  joint fit
    # -----------------------------------------------------------------
    if not ca:
        print("No Ca48 grid yet, so no joint PREX-2 + CREX constraint.")
        return

    print("--- JOINT PREX-2 + CREX ---")
    print("  chi2(J,L) = [(Pb-0.283)/0.071]^2 + [(Ca-0.121)/%.3f]^2"
          % CREX["sigma"])
    print()
    for model in sorted(set(m for (m, n) in fits if n == "Pb208")):
        if (model, "Ca48") not in fits:
            continue
        cpb, cca = fits[(model, "Pb208")], fits[(model, "Ca48")]
        best = None
        for J in J_grid:
            for L in L_grid:
                chi2 = (((skin_of(cpb, J, L) - PREX2["skin"]) / PREX2["sigma"]) ** 2
                        + ((skin_of(cca, J, L) - CREX["skin"]) / CREX["sigma"]) ** 2)
                if best is None or chi2 < best[0]:
                    best = (chi2, J, L)
        chi2, Jb, Lb = best
        # 68% region for two parameters is Delta chi2 = 2.30
        Ls = [L for L in L_grid
              for J in [Jb]
              if (((skin_of(cpb, J, L) - PREX2["skin"]) / PREX2["sigma"]) ** 2
                  + ((skin_of(cca, J, L) - CREX["skin"]) / CREX["sigma"]) ** 2)
              <= chi2 + 2.30]
        rng = "[%.0f, %.0f]" % (min(Ls), max(Ls)) if Ls else "none"
        print("  %-10s chi2_min = %5.2f at J = %.1f, L = %.0f MeV; "
              "68%% L range %s" % (model, chi2, Jb, Lb, rng))
        print("             predicted  Pb %.3f (meas %.3f)   Ca %.3f (meas %.3f)"
              % (skin_of(cpb, Jb, Lb), PREX2["skin"],
                 skin_of(cca, Jb, Lb), CREX["skin"]))
    print()
    print("  Cooling constraint already in the paper: L = [%.0f, %.0f] MeV"
          % COOLING_L)
    print()

    # -----------------------------------------------------------------
    #  THE RESULT: does the cooling band overlap each experiment?
    # -----------------------------------------------------------------
    print("--- OVERLAP OF THE COOLING BAND WITH EACH EXPERIMENT ---")
    print("  cooling L = [%.0f, %.0f] MeV (combined 5-mode, Delta chi2/N <= 1)"
          % COOLING_L)
    print()
    print("  %-10s %-6s %-22s %-22s" % ("model", "J", "cooling & PREX-2",
                                        "cooling & CREX"))
    for model in sorted(set(m for (m, n) in fits if n == "Pb208")):
        if (model, "Ca48") not in fits:
            continue
        for J in (26.5, 32.5, 38.5):
            row = []
            for meas, nuc in ((PREX2, "Pb208"), (CREX, "Ca48")):
                coef = fits[(model, nuc)]
                ok = [L for L in L_grid
                      if COOLING_L[0] <= L <= COOLING_L[1]
                      and abs(skin_of(coef, J, L) - meas["skin"])
                      <= meas["sigma"]]
                row.append("none" if not ok
                           else "L = [%.0f, %.0f]" % (min(ok), max(ok)))
            print("  %-10s %-6.1f %-22s %-22s" % (model, J, row[0], row[1]))
    print()
    print("  Read this as the headline. Where the cooling band overlaps one")
    print("  experiment and not the other, neutron star thermal evolution is")
    print("  casting an independent, astrophysical vote in a disagreement")
    print("  that has so far been settled only in the laboratory.")
    print()
    print("  NOTE: with two measurements and two free parameters the joint fit")
    print("  has no degrees of freedom, so chi2_min is NOT a goodness-of-fit")
    print("  test -- it is a direct measure of how far apart PREX-2 and CREX")
    print("  pull inside this model family. A large value is the PREX-CREX")
    print("  puzzle showing up in our own calculation, not a failure of ours.")


if __name__ == "__main__":
    main()
