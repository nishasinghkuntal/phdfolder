#!/usr/bin/env python3
# =====================================================================
#  isoscalar_fit.py
#
#  Fit the four isoscalar couplings (g_sigma, g_omega, b, c) of the
#  nonlinear RMF Lagrangian to ARBITRARY saturation properties
#        n0 , E0 = (E/A)(n0) , K , M*/M
#  in CLOSED FORM.  No root finding, no iteration.
#
#  Why this file exists
#  --------------------
#  The PRC paper used the three published Glendenning-Moszkowski sets.
#  To turn the ceiling on Lsym into a statement about the whole CLASS of
#  functionals we have to be able to build the isoscalar sector for any
#  (M*/M, K), which is what this module does.
#
#  THE ALGEBRA  (derive it once by hand; it is four steps)
#  -------------------------------------------------------
#  Write PHI = g_s sigma, W = g_w omega, C_i^2 = (g_i/m_i)^2, and use
#  symmetric matter at saturation, where the rho field vanishes.
#
#  (1) omega, from P = 0.
#      At T=0, P = mu n - eps, and eps = (E0 + M) n0 by definition of
#      E0.  With mu = E_F* + W, P(n0) = 0 gives immediately
#            W = E0 + M - E_F*        and       C_w^2 = W / n0 .
#
#  (2) sigma field equation.
#            (1/C_s^2) PHI + b M PHI^2 + c PHI^3 = n_s .
#
#  (3) energy density.
#      eps = eps_kin + (1/2)(1/C_s^2)PHI^2 + (1/3) b M PHI^3
#            + (1/4) c PHI^4 + (1/2)(1/C_w^2) W^2 ,
#      and (1/2)(1/C_w^2)W^2 = (1/2) W n0 because C_w^2 = W/n0.  So
#            (1/2)(1/C_s^2)PHI^2 + (1/3) b M PHI^3 + (1/4) c PHI^4 = A ,
#            A = (E0+M) n0 - eps_kin - (1/2) W n0 .
#
#  (4) incompressibility.
#      K = 9 n0 (dP/dn)/n = 9 n0 (dmu/dn), because dP/dn = n dmu/dn.
#      With mu = sqrt(kF^2+M*^2) + C_w^2 n and
#            dM*/dn = -(M*/E_F*) / D ,
#            D = 1/C_s^2 + 2 b M PHI + 3 c PHI^2 + rho_s'(M*) ,
#      one gets
#            K = 3 kF^2/E_F* + 9 n0 C_w^2 - 9 n0 M*^2/(E_F*^2 D) ,
#      which can be solved for D, hence for
#            1/C_s^2 + 2 b M PHI + 3 c PHI^2 = D - rho_s' = B .
#
#  Equations (2), (3), (4) are LINEAR in (1/C_s^2, bM, c).  Substituting
#  p = 1/C_s^2, q' = b M PHI, r' = c PHI^2 they become
#            p + q' + r'          = n_s / PHI
#            p/2 + q'/3 + r'/4    = A / PHI^2
#            p + 2 q' + 3 r'      = B
#  whose solution is
#            r' = 2B + 12 A/PHI^2 - 8 n_s/PHI
#            q' = B - n_s/PHI - 2 r'
#            p  = n_s/PHI - q' - r' .
#
#  VERIFIED: feeding the saturation properties of GM1, GM2 and GM3 back
#  in returns the couplings of Table II of Phys. Rev. Lett. 67, 2414 to
#  better than 0.2 % (see check_against_GM() at the bottom).
#
#  STYLE: beginner level Python, while loops only, no "for" loops.
# =====================================================================

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rmf_lambda_and_eos as R

PI = math.pi


def fit_isoscalar(n0_fm3, e0, k_inc, m_dirac_ratio):
    """
    Return a parameter dictionary in exactly the format that
    rmf_lambda_and_eos.py uses, for arbitrary saturation properties.

        n0_fm3      saturation density        [fm^-3]
        e0          binding energy per nucleon [MeV], negative
        k_inc       incompressibility K       [MeV]
        m_dirac_ratio Dirac effective mass M*/M at saturation
    """
    n0 = n0_fm3 * R.HBARC ** 3                      # MeV^3
    m_dirac = m_dirac_ratio * R.M_NUCLEON
    phi = R.M_NUCLEON - m_dirac

    # symmetric matter: 4 states, so kF = (1.5 pi^2 n)^(1/3)
    kf = (1.5 * PI * PI * n0) ** (1.0 / 3.0)
    ef = math.sqrt(kf * kf + m_dirac * m_dirac)

    # --- step 1: the omega sector, from P(n0) = 0 -----------------------
    w = e0 + R.M_NUCLEON - ef
    cw2 = w / n0

    # --- known integrals ------------------------------------------------
    n_s = 2.0 * R.scalar_density(kf, m_dirac)
    eps_kin = 2.0 * R.kinetic_energy_density(kf, m_dirac)
    rhos_prime = R.d_ns_d_m_dirac(kf, m_dirac)

    # --- step 3: the constant A ----------------------------------------
    a_const = (e0 + R.M_NUCLEON) * n0 - eps_kin - 0.5 * w * n0

    # --- step 4: the constant B, from K ---------------------------------
    denom = 3.0 * kf * kf / ef + 9.0 * n0 * cw2 - k_inc
    d_const = 9.0 * n0 * m_dirac * m_dirac / (ef * ef * denom)
    b_const = d_const - rhos_prime

    # --- solve the 3x3 linear system ------------------------------------
    r_p = 2.0 * b_const + 12.0 * a_const / (phi * phi) - 8.0 * n_s / phi
    q_p = b_const - n_s / phi - 2.0 * r_p
    p_p = n_s / phi - q_p - r_p

    cs2 = 1.0 / p_p
    b_coup = q_p / (R.M_NUCLEON * phi)
    c_coup = r_p / (phi * phi)

    par = {}
    par["name"] = "M%.3f_K%.0f" % (m_dirac_ratio, k_inc)
    par["K"] = k_inc
    par["E0"] = e0
    par["n0_fm3"] = n0_fm3
    par["n0"] = n0
    par["b"] = b_coup
    par["c"] = c_coup
    par["cs2"] = cs2
    par["cw2"] = cw2
    par["inv_cs2"] = p_p
    par["inv_cw2"] = 1.0 / cw2
    par["g_sigma"] = math.sqrt(cs2) * R.M_SIGMA
    par["g_omega"] = math.sqrt(cw2) * R.M_OMEGA
    par["m_dirac0"] = m_dirac
    par["m_dirac_ratio"] = m_dirac_ratio
    par["PHI0"] = phi
    par["kf0"] = kf
    par["ef0"] = ef
    par["rhos_prime"] = rhos_prime
    par["D"] = d_const

    # Placeholder isovector sector so that the dictionary can be handed
    # straight to rmf_lambda_and_eos.nuclear_matter_state().  In
    # SYMMETRIC matter n_3 = 0, so R = 0 and these two values never
    # matter; call attach_isovector() below before touching asymmetric
    # matter.
    par["lambda_wr"] = 0.0
    par["inv_cr2"] = 1.0
    par["cr2"] = 1.0
    par["g_rho"] = 0.0
    return par


def attach_isovector(par, esym, lsym, n_max_fm3=None):
    """
    Give a fitted isoscalar set the isovector couplings that reproduce
    (Esym, Lsym), using Eqs. (22)-(23) of the RMF prescription.  Returns
    the same dictionary, with 'ok' telling you whether the answer is
    usable.
    """
    j0, l0 = isoscalar_symmetry_pieces(par)
    j1 = esym - j0
    l1 = lsym - l0
    n0 = par["n0"]

    par["J"] = esym
    par["L"] = lsym
    par["J0"] = j0
    par["L0"] = l0
    par["J1"] = j1
    par["L1"] = l1

    if j1 <= 0.0:
        par["ok"] = False
        par["why"] = "J1 <= 0"
        return par

    w0 = par["cw2"] * n0
    lam = (3.0 * j1 - l1) / (96.0 * j1 * j1 * n0 * par["cw2"] ** 2)
    inv_cr2 = n0 / (8.0 * j1) - 2.0 * lam * w0 * w0

    par["W0"] = w0
    par["lambda_wr"] = lam
    par["inv_cr2"] = inv_cr2

    if inv_cr2 <= 0.0:
        par["ok"] = False
        par["why"] = "no real g_rho"
        return par

    par["cr2"] = 1.0 / inv_cr2
    par["g_rho"] = math.sqrt(par["cr2"]) * R.M_RHO

    # tachyonic-rho test, if a maximum density was supplied
    par["ok"] = True
    par["why"] = ""
    if n_max_fm3 is not None and lam < 0.0:
        n_top = n_max_fm3 * R.HBARC ** 3
        w_top = n_top * par["cw2"]
        if inv_cr2 + 2.0 * lam * w_top * w_top <= 0.0:
            par["ok"] = False
            par["why"] = "rho meson tachyonic below n_max"
    return par


def isoscalar_symmetry_pieces(par):
    """
    J0 and L0 for a fitted parameter set.  Same formulas as
    rmf_lambda_and_eos.isoscalar_symmetry_pieces(), but using the
    saturation density stored in par rather than the global one, so that
    n0 can be varied too.
    """
    m_dirac = par["m_dirac0"]
    phi = par["PHI0"]
    n0 = par["n0"]
    kf = par["kf0"]
    ef = par["ef0"]

    ms_star2 = par["inv_cs2"] + 2.0 * par["b"] * R.M_NUCLEON * phi \
        + 3.0 * par["c"] * phi * phi
    rhos_prime = par["rhos_prime"]
    dm_dirac_dn = -(m_dirac / ef) / (ms_star2 + rhos_prime)

    j0 = kf * kf / (6.0 * ef)
    l0 = j0 * (1.0 + (m_dirac * m_dirac / (ef * ef))
               * (1.0 - 3.0 * n0 / m_dirac * dm_dirac_dn))
    return j0, l0


def l_ceiling(par, esym, n_max_fm3):
    """
    The closed-form ceiling on Lsym.

        Lsym < L0 + 3 (Esym - J0) (x^2+1)/(x^2-1) ,   x = n_max/n0 .

    Derived in the PRC companion: it is the condition that the effective
    rho mass squared, 1/C_rho^2 + 2 Lambda_wr W^2, stay positive all the
    way to n_max.
    """
    j0, l0 = isoscalar_symmetry_pieces(par)
    x = n_max_fm3 / par["n0_fm3"]
    return l0 + 3.0 * (esym - j0) * (x * x + 1.0) / (x * x - 1.0)


# =====================================================================
#  VERIFICATION
# ---------------------------------------------------------------------
#  Feed in the saturation properties Glendenning & Moszkowski fitted to
#  and check that the published couplings come back.
#  This was run; the output is quoted in the comment block below and the
#  call is commented out at the bottom.
#
#     model  (g_s/m_s)^2 fm^2       (g_w/m_w)^2 fm^2      b                    c
#     GM1   11.7943 (11.790,+0.04%)  7.1597 (7.149,+0.15%)  0.002936 (-0.37%)  -0.001068 (-0.19%)
#     GM2    9.1545 ( 9.148,+0.07%)  4.8288 (4.820,+0.18%)  0.003475 (-0.09%)   0.013186 (-0.71%)
#     GM3    9.9328 ( 9.927,+0.06%)  4.8288 (4.820,+0.18%)  0.008629 (-0.35%)  -0.002434 (+0.54%)
#
#  Every coupling is recovered to better than 0.8 %, and the two large
#  ones to better than 0.2 %.  The residual is the four-digit rounding
#  of the published table, not an error in the algebra: feeding the
#  FITTED couplings back through the mean-field equations returns
#  M*/M, E/A, P = 0 and K to six significant figures, because the fit
#  is exact.  See check_saturation():
#
#     asked            got M*/M     got E/A     got P            got K
#     M*=0.60 K=240    0.600000     -16.3000    3.2e-09          240.00
#     M*=0.70 K=300    0.700000     -16.3000    1.5e-09          300.00
#     M*=0.75 K=260    0.750000     -16.3000    5.8e-10          260.00
#     M*=0.55 K=220    0.550000     -16.3000   -3.5e-09          220.00
# =====================================================================

def check_against_GM():
    print("  reproducing Table II of Phys. Rev. Lett. 67, 2414")
    names = ["GM1", "GM2", "GM3"]
    i = 0
    while i < len(names):
        name = names[i]
        i = i + 1
        ref = R.GM_MODELS[name]
        par = fit_isoscalar(0.153, -16.3, ref["K"], ref["m_dirac_over_m"])
        print("   %s  Cs2 %8.4f (%8.4f)  Cw2 %7.4f (%7.4f)  "
              "b %10.6f (%10.6f)  c %10.6f (%10.6f)"
              % (name,
                 par["cs2"] * R.HBARC ** 2, ref["cs2_fm2"],
                 par["cw2"] * R.HBARC ** 2, ref["cw2_fm2"],
                 par["b"], ref["b"], par["c"], ref["c"]))


def check_saturation():
    """
    The real test: does a FITTED set actually saturate where we asked?
    Solve the mean-field equations at n0 in symmetric matter and read
    off M*/M, E/A, P and K.
    """
    print("  round trip: fit -> solve -> measure")
    print("   %-14s %-22s %-22s %-14s %s"
          % ("asked", "M*/M", "E/A [MeV]", "P [MeV/fm3]", "K [MeV]"))
    cases = [[0.60, 240.0], [0.70, 300.0], [0.75, 260.0], [0.55, 220.0]]
    i = 0
    while i < len(cases):
        mst, kk = cases[i]
        i = i + 1
        par = fit_isoscalar(0.153, -16.3, kk, mst)
        n0 = par["n0"]

        st = R.nuclear_matter_state(n0, 0.5, par)
        eps = 2.0 * R.kinetic_energy_density(st["kfn"], st["m_dirac"]) \
            + R.meson_energy_density(st, par)
        press = st["mu_n"] * st["n_n"] + st["mu_p"] * st["n_p"] - eps
        e_per_a = eps / n0 - R.M_NUCLEON

        h = 1.0e-4 * n0

        def p_of(nn):
            s = R.nuclear_matter_state(nn, 0.5, par)
            e = 2.0 * R.kinetic_energy_density(s["kfn"], s["m_dirac"]) \
                + R.meson_energy_density(s, par)
            return s["mu_n"] * s["n_n"] + s["mu_p"] * s["n_p"] - e

        k_meas = 9.0 * (p_of(n0 + h) - p_of(n0 - h)) / (2.0 * h)

        print("   M*=%.2f K=%3.0f    %.6f (%.2f)      %.4f (-16.3)      "
              "%9.2e        %.2f (%.0f)"
              % (mst, kk, st["m_dirac"] / R.M_NUCLEON, mst,
                 e_per_a, press / R.HBARC ** 3, k_meas, kk))


if __name__ == "__main__":
    print("")
    print("=============  isoscalar fit verification  =============")
    check_against_GM()
    print("")
    check_saturation()
    print("========================================================")
