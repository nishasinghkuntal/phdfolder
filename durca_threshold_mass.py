#!/usr/bin/env python3
# =====================================================================
#  durca_threshold_mass.py
# ---------------------------------------------------------------------
#  Direct Urca threshold MASS M_DU for the (Esym, Lsym) grid.
#
#  rmf_lambda_and_eos.py already gives the threshold DENSITY n_DU (the
#  first density where k_Fn <= k_Fp + k_Fe).  M_DU is the mass of the
#  star whose CENTRAL density is exactly n_DU: every lighter star has no
#  direct Urca anywhere, every heavier one has it in the core.
#
#  This is needed to compare with Montefusco et al., arXiv:2609.24940,
#  who find P(M_DU <= 1.4 Msun) < 1% from nuclear data.
#
#  WHY A SMALL TOV HERE AND NOT TOVprofile2_automated.f90
#  The Fortran code needs the crust-joined NSCool tables, which live on
#  the work machine.  M(n_c) at fixed central density hardly depends on
#  the crust (the crust holds ~0.01-0.03 Msun), so here the RMF core
#  table is used down to 0.03 fm^-3 and continued outward with a
#  Gamma = 4/3 polytrope (relativistic electrons) matched in pressure
#  and energy density.  Check: GM1 with Lambda_wr = 0 gives Mmax within
#  ~0.01 Msun of the 2.360 from the full pipeline (printed below).
#  For the paper, re-read M_DU off the real MRcurve_*_CAT.dat files.
#
#  Units: G = c = 1, lengths in km, energy density and pressure in km^-2.
# =====================================================================

import math
import sys

import numpy as np
from scipy.integrate import solve_ivp

import rmf_lambda_and_eos as R

MEVFM3_TO_KM2 = 1.3234e-6        # 1 MeV/fm^3 -> km^-2  (G/c^4 * 1.602e33 erg/cm^3)
MSUN_KM = 1.476625


def eos_arrays(par):
    """(n, eps, P) in fm^-3, km^-2, km^-2, increasing density, plus n_DU."""
    rows, info = R.build_eos(par)
    ok, why = R.table_is_usable(rows, info)
    if not ok:
        return None
    n = np.array([r["n_fm3"] for r in rows])
    eps = np.array([r["eps"] / R.HBARC ** 3 for r in rows]) * MEVFM3_TO_KM2
    p = np.array([r["press"] / R.HBARC ** 3 for r in rows]) * MEVFM3_TO_KM2
    n_du = R.durca_threshold_density(rows)
    return n, eps, p, n_du


def make_eps_of_p(n, eps, p):
    """eps(P): table in the core, Gamma=4/3 polytrope below the table."""
    lp = np.log(p)
    le = np.log(eps)
    p0, e0 = p[0], eps[0]

    def f(pp):
        if pp >= p0:
            return math.exp(np.interp(math.log(pp), lp, le))
        # below the table: eps = e0 (P/P0)^(3/4)  (rest-mass dominated,
        # P ~ rho^(4/3)); only affects the outer ~1 km
        return e0 * (pp / p0) ** 0.75
    return f


def tov_mass(pc, eps_of_p, p_surf):
    def rhs(r, y):
        m, pr = y
        if pr <= p_surf:
            return [0.0, 0.0]
        e = eps_of_p(pr)
        dm = 4.0 * math.pi * r * r * e
        dp = -(e + pr) * (m + 4.0 * math.pi * r ** 3 * pr) / (r * (r - 2.0 * m))
        return [dm, dp]

    def surface(r, y):
        return y[1] - p_surf
    surface.terminal = True
    surface.direction = -1

    r0 = 1.0e-4
    e0 = eps_of_p(pc)
    y0 = [4.0 / 3.0 * math.pi * r0 ** 3 * e0, pc]
    sol = solve_ivp(rhs, (r0, 40.0), y0, events=surface, rtol=1e-8, atol=1e-14,
                    max_step=0.05)
    return sol.y[0, -1] / MSUN_KM, sol.t[-1]


def mass_radius(n, eps, p, n_c_list):
    f = make_eps_of_p(n, eps, p)
    p_surf = p[0] * 1.0e-7
    out = []
    for nc in n_c_list:
        pc = math.exp(np.interp(nc, n, np.log(p)))
        m, rad = tov_mass(pc, f, p_surf)
        out.append((nc, m, rad))
    return out


def analyse(model, j, l):
    par = R.make_parameter_set(model, j, l)
    if not par["ok"]:
        return None
    healthy, _ = R.rho_meson_is_healthy(par)
    if not healthy:
        return None
    arr = eos_arrays(par)
    if arr is None:
        return None
    n, eps, p, n_du = arr
    grid = np.geomspace(0.2, min(1.5, n[-1]), 40)
    mr = mass_radius(n, eps, p, grid)
    ms = np.array([x[1] for x in mr])
    imax = int(np.argmax(ms))
    mmax, n_max = ms[imax], grid[imax]
    # causality in the core
    vs2 = np.diff(p) / np.diff(eps)
    causal = bool(np.all(vs2[n[1:] <= n_max] < 1.0))
    r14 = float("nan")
    if mmax >= 1.4:
        r14 = float(np.interp(1.4, ms[:imax + 1], [x[2] for x in mr[:imax + 1]]))
    if n_du is None or n_du > n_max:
        m_du = float("inf")
    else:
        m_du = mass_radius(n, eps, p, [n_du])[0][1]
    return {"model": model, "J": j, "L": l, "n_DU": n_du, "M_DU": m_du,
            "Mmax": mmax, "R14": r14, "causal": causal}


def validation():
    iso = R.isoscalar_symmetry_pieces(R.isoscalar_block("GM1"))
    l0 = iso["L0"] + 3.0 * (32.5 - iso["J0"])
    res = analyse("GM1", 32.5, l0)
    print("# check: GM1, Lambda_wr = 0: Mmax = %.3f Msun "
          "(full pipeline 2.360, GM 1991: 2.35)" % res["Mmax"])


def main():
    validation()
    j_list = [28.0, 29.5, 31.0, 32.5, 34.0, 35.5]
    l_list = [50.0, 55.0, 60.0, 65.0, 70.0, 75.0]
    models = R.MODEL_NAMES
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        j_list = R.ESYM_LIST
        l_list = R.LSYM_LIST
    print("# model J L n_DU[fm-3] M_DU[Msun] Mmax[Msun] R1.4[km] passes_cuts")
    for m in models:
        for j in j_list:
            for l in l_list:
                r = analyse(m, j, l)
                if r is None:
                    continue
                keep = r["Mmax"] >= 2.0 and r["causal"]
                nd = "none" if r["n_DU"] is None else "%.4f" % r["n_DU"]
                print("%-9s %5.1f %6.1f %8s %7.3f %6.3f %7.2f %d"
                      % (m, j, l, nd, r["M_DU"], r["Mmax"], r["R14"], keep))
                sys.stdout.flush()


if __name__ == "__main__":
    main()
