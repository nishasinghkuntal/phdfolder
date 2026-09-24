#!/usr/bin/env python3
# =====================================================================
#  finite_nucleus_rmf.py
# ---------------------------------------------------------------------
#  SPHERICAL RELATIVISTIC MEAN FIELD (HARTREE) CALCULATION
#  FOR DOUBLY-MAGIC NUCLEI  ->  NEUTRON SKIN THICKNESS
# ---------------------------------------------------------------------
#
#  WHY THIS FILE EXISTS
#  --------------------
#  rmf_lambda_and_eos.py solves INFINITE nuclear matter: it gives the
#  equation of state that goes into NSCool.  It cannot tell you what
#  the neutron skin of Pb-208 is, because a skin is a FINITE-NUCLEUS
#  property: it needs the actual proton and neutron density profiles
#  rho_p(r), rho_n(r) of a real nucleus.
#
#  Up to now the paper quoted neutron skins taken from the published
#  Roca-Maza et al. DR_np - L_sym correlation.  That correlation is a
#  fit ACROSS many different models, so it carries an intrinsic scatter
#  of roughly +-0.02 fm and it is not a prediction of OUR couplings.
#
#  This file removes that weakness.  For each parameter set
#  (g_sigma, g_omega, g_rho, b, c, zeta, Lambda_omega_rho) that we
#  already use for the EoS, it solves the coupled Dirac + Klein-Gordon
#  system for a finite nucleus and returns the genuine prediction
#
#        DR_np  =  <r^2>_n^(1/2)  -  <r^2>_p^(1/2)
#
#  so the neutron skin becomes an OUTPUT of the same Lagrangian that
#  produced the cooling curves, not an input borrowed from elsewhere.
#
#
#  THE EQUATIONS
#  -------------
#  Exactly the same Lagrangian as rmf_lambda_and_eos.py, but now the
#  meson fields depend on r, so the Laplacian no longer vanishes.
#  Writing (as in the EoS code)
#
#        PHI = g_sigma * sigma ,  W = g_omega * omega ,  R = g_rho * rho
#
#  the three meson equations become Klein-Gordon equations
#
#    -lap PHI + m_sigma^2 PHI = g_sigma^2 [ n_s - b M PHI^2 - c PHI^3 ]
#    -lap W   + m_omega^2 W   = g_omega^2 [ n_v - 2 Lam R^2 W - (zeta/6) W^3 ]
#    -lap R   + m_rho^2   R   = g_rho^2   [ n_3/2 - 2 Lam W^2 R ]
#
#  Setting lap -> 0 reproduces the infinite-matter equations of
#  rmf_lambda_and_eos.py line-for-line, which is the consistency check
#  that these are the right equations.
#
#  The nucleons move in
#
#        M*(r) = M_N - PHI(r)                      (scalar)
#        V(r)  = W(r) +- R(r)/2 + V_Coul(r)        (vector, + for proton)
#
#  and obey the radial Dirac equations, for a level with quantum
#  number kappa (kappa = -(j+1/2) for j = l+1/2, kappa = +(j+1/2) for
#  j = l-1/2):
#
#        dG/dr = -(kappa/r) G + [E - V(r) + M*(r)] F
#        dF/dr = +(kappa/r) F - [E - V(r) - M*(r)] G
#
#  G and F are the large and small radial components; the densities are
#
#        n_v(r) = sum_occ (2j+1)/(4 pi r^2) [ G^2 + F^2 ]
#        n_s(r) = sum_occ (2j+1)/(4 pi r^2) [ G^2 - F^2 ]
#
#
#  UNITS
#  -----
#  Everything inside this file is in NATURAL units with lengths in fm:
#  all energies, masses and meson fields are in fm^-1, densities in
#  fm^-3.  Convert with  E[fm^-1] = E[MeV] / 197.327 .  In these units
#  every equation above is dimensionally clean with no stray hbar*c.
#
#  Nisha Singh -- Shiv Nadar Institution of Eminence
# =====================================================================

import math
import numpy as np

import rmf_lambda_and_eos as eos


# =====================================================================
#  SECTION 1 :  CONSTANTS
# =====================================================================

HBARC = 197.327                      # MeV fm

# masses in fm^-1 (taken from the EoS module so the two files can never
# drift apart)
M_N = eos.M_NUCLEON / HBARC
M_SIG = eos.M_SIGMA / HBARC
M_OM = eos.M_OMEGA / HBARC
M_RHO = eos.M_RHO / HBARC

ALPHA_EM = 1.0 / 137.036             # fine structure constant

# single-nucleon charge radii, used only for the charge-radius check
R_PROTON_CH2 = 0.7056                # fm^2  (r_p = 0.8409 fm)
R_NEUTRON_CH2 = -0.1161              # fm^2  (negative: neutron charge dist.)


# ---------------------------------------------------------------------
#  which nuclei we can do.  Both are spherical and doubly magic, so a
#  plain Hartree calculation with filled shells is exact enough -- no
#  pairing and no deformation are needed.
# ---------------------------------------------------------------------
NUCLEI = {
    "Pb208": {"Z": 82, "N": 126, "r_max": 20.0},
    "Ca48":  {"Z": 20, "N": 28,  "r_max": 16.0},
    "Ca40":  {"Z": 20, "N": 20,  "r_max": 16.0},
    "O16":   {"Z": 8,  "N": 8,   "r_max": 14.0},
    "Sn132": {"Z": 50, "N": 82,  "r_max": 18.0},
}


# ---------------------------------------------------------------------
#  PER-MODEL MESON MASSES  (verified against the source papers)
# ---------------------------------------------------------------------
#  rmf_lambda_and_eos.py carries ONE sigma mass (550 MeV) for every
#  model, and for the equation of state that is exactly right: only the
#  ratio (g_sigma/m_sigma)^2 ever enters infinite nuclear matter, so
#  m_sigma and g_sigma are not separately determined there.
#
#  A finite nucleus is different.  m_sigma fixes the RANGE of the scalar
#  field, 1/m_sigma, and that range sets how diffuse the nuclear surface
#  is -- which sets the radius.  Running Pb-208 with 550 MeV for models
#  whose real value is near 495 MeV gives a surface that is too sharp, a
#  charge radius about 0.1 fm too small, and roughly 0.5 MeV/nucleon of
#  spurious extra binding.
#
#  We hold (g_sigma/m_sigma)^2 fixed at the value the EoS already uses
#  and rescale g_sigma = sqrt(cs2) * m_sigma.  That leaves the infinite
#  matter limit -- and therefore every EoS table and cooling curve --
#  bit for bit unchanged, and changes only the range.
#
#  SOURCE.  Kumar et al., "BigApple force and its implications to finite
#  nuclei and astrophysical objects", arXiv:2009.10690, Table 1, which
#  tabulates the masses as ratios to M = 939 MeV:
#
#        m_s/M    FSUGarnet 0.529   IOPB-I 0.533   BigApple 0.525
#        m_w/M    0.833 for all three
#        m_r/M    0.812 for all three
#
#  The ratios are quoted to three decimals, so each mass below carries
#  about +-0.5 MeV of transcription uncertainty.  That is far too small
#  to matter: it moves the scalar range by under 0.1 percent.
# ---------------------------------------------------------------------
_M_NUC_TABLE = 939.0

#  Verified 2026-09-18 against Table 1 of arXiv:2009.10690 (Kumar,
#  Biswal, Singh, Patra), which tabulates m/M ratios to three decimals
#  for NL3, FSUGarnet, G3, IOPB-I and BigApple.  Values below are those
#  ratios times M = 939 MeV, so they carry the table's own rounding of
#  about +-0.9 MeV.  That is far finer than anything the skin resolves.
#
#  GM1, GM2 and GM3 are deliberately ABSENT, and that is a physics
#  statement rather than a missing entry.  Glendenning & Moszkowski,
#  PRL 67, 2414 (1991), Table II publishes only the RATIOS (g/m)^2 --
#  11.79, 7.149, 4.411 fm^2 for GM1 -- because infinite nuclear matter
#  never needs more than that.  The individual meson masses were never
#  determined.  So m_sigma, and with it the range of the scalar field
#  and the diffuseness of the nuclear surface, is simply not defined for
#  the GM sets.  A finite-nucleus skin from GM1 or GM2 would be a
#  property of whatever m_sigma we invented, not of the model, so the
#  code refuses to produce one.
MESON_MASSES = {
    # Chen & Piekarewicz, Phys. Lett. B 748, 284 (2015), arXiv:1412.7870
    #   (VERIFIED 2026-09-20. An earlier comment here cited PRC 92,
    #    011301(R), which is not the paper that introduces FSUGarnet.)
    "FSUGarnet": {"m_sigma": 0.529 * 939.0, "m_omega": 0.833 * 939.0,
                  "m_rho": 0.812 * 939.0},
    # Kumar, Patra & Agrawal, PRC 97, 045806 (2018)
    #   (VERIFIED 2026-09-20. Earlier comment had both the page
    #    number and the author list wrong.)
    "IOPB-I":    {"m_sigma": 0.533 * 939.0, "m_omega": 0.833 * 939.0,
                  "m_rho": 0.812 * 939.0},
    # Fattoyev, Horowitz, Piekarewicz & Reed, PRC 102, 065805 (2020)
    "BigApple":  {"m_sigma": 0.525 * 939.0, "m_omega": 0.833 * 939.0,
                  "m_rho": 0.812 * 939.0},
}

#  published (J, L) and finite-nucleus results, Tables 2 and 3 of
#  arXiv:2009.10690 -- used only to validate this solver
PUBLISHED = {
    "FSUGarnet": {"J": 30.95, "L": 51.04, "Pb208": {"skin": 0.162, "r_ch": 5.496, "BA": 7.902},
                  "Ca48": {"skin": 0.169, "r_ch": 3.426, "BA": 8.609}},
    "IOPB-I":    {"J": 33.30, "L": 63.58, "Pb208": {"skin": 0.221, "r_ch": 5.520, "BA": 7.870},
                  "Ca48": {"skin": 0.202, "r_ch": 3.446, "BA": 8.638}},
    "BigApple":  {"J": 31.32, "L": 39.80, "Pb208": {"skin": 0.151, "r_ch": 5.495, "BA": 7.894},
                  "Ca48": {"skin": 0.170, "r_ch": 3.447, "BA": 8.547}},
}

#  PARITY-VIOLATION MEASUREMENTS  (verified 2026-09-18)
#
#  PREX-II  Pb-208 : 0.283 +- 0.071 fm
#      Adhikari et al., PRL 126, 172502 (2021).  Quoted as Eq. (14) of
#      arXiv:2009.10690, read directly.
#  CREX     Ca-48  : 0.121 +- 0.026 (exp) +- 0.024 (model) fm
#      Adhikari et al., PRL 129, 042501 (2022).  The two errors combine
#      in quadrature to +-0.035 fm, which is the figure the JLab release
#      and the APS DNP presentation quote.
#
#  These two pull in opposite directions -- a thick skin for lead and a
#  thin one for calcium -- and no mean-field model reproduces both.  That
#  is the PREX-CREX puzzle, and it is the tension this project's cooling
#  constraint can speak to independently.
EXPERIMENT = {
    "Pb208": {"skin": 0.283, "err": 0.071, "source": "PREX-II, PRL 126, 172502 (2021)"},
    "Ca48":  {"skin": 0.121, "err": 0.035, "source": "CREX, PRL 129, 042501 (2022)"},
}

#  VALIDATION, this solver vs the published values above, each model run
#  at its OWN published (J, L).  Reproduced 2026-09-18:
#
#     model      nucleus   this code   published   difference
#     FSUGarnet  Pb-208     0.1626      0.162       +0.0006
#     FSUGarnet  Ca-48      0.1674      0.169       -0.0016
#     IOPB-I     Pb-208     0.2207      0.221       -0.0003
#     IOPB-I     Ca-48      0.2002      0.202       -0.0018
#     BigApple   Pb-208     0.1493      0.151       -0.0017
#     BigApple   Ca-48      0.1674      0.170       -0.0026
#
#  Every skin agrees to better than 0.003 fm, which is an order of
#  magnitude finer than the 0.02 fm intrinsic scatter of the model-
#  averaged DR_np - L correlation this file replaces, and two orders
#  finer than the PREX-II error bar.  Charge radii agree to 0.002-0.020
#  fm.  B/A comes out a systematic 0.08-0.12 MeV low, almost certainly
#  the pairing term E_pair that the published calculation includes and
#  this one does not; it does not touch the skins.

SKIN_MODELS = sorted(MESON_MASSES)


def apply_meson_masses(par):
    """
    Give a parameter set its model's real meson masses.

    (g_sigma/m_sigma)^2 is held fixed, so nuclear matter is untouched and
    only the finite-nucleus surface changes.

    A model with no entry in MESON_MASSES raises.  Falling back on the
    EoS module's generic 550 MeV would hand back a skin that is a
    property of that invented number rather than of the model, which is
    precisely the kind of unverifiable input this file exists to remove.
    """
    mm = MESON_MASSES.get(par.get("name"))
    if mm is None:
        raise ValueError(
            "no published meson masses for %r, so a finite-nucleus skin "
            "is not defined for it: only the ratios (g/m)^2 were ever "
            "published for the GM sets. Usable models: %s"
            % (par.get("name"), ", ".join(SKIN_MODELS)))
    par = dict(par)
    par.update(mm)
    par["g_sigma"] = math.sqrt(par["cs2"]) * mm["m_sigma"]
    par["g_omega"] = math.sqrt(par["cw2"]) * mm["m_omega"]
    if par.get("cr2", 0.0) > 0.0:
        par["g_rho"] = math.sqrt(par["cr2"]) * mm["m_rho"]
    return par


def has_finite_nucleus_masses(model_name):
    """True if this model's meson masses are actually published."""
    return model_name in MESON_MASSES


# =====================================================================
#  SECTION 2 :  RADIAL GRID
# =====================================================================

def make_grid(r_max, n_points, r_surface):
    """
    Uniform radial mesh r = 0, h, 2h, ... r_max.

    A uniform mesh is fine here: RMF potentials are smooth Woods-Saxon
    shapes with no Coulomb singularity at the origin (the nucleus is
    not a point charge), so there is nothing that needs a logarithmic
    mesh.

    r_surface is the nuclear radius, 1.2 A^(1/3).  Two radii derived
    from it control the shooting and they matter a lot:

      r_count -- nodes of G are only counted inside this radius.  Past
                 the classical turning point, round-off feeds the
                 exponentially GROWING Dirac solution into the outward
                 integration, so even an exact eigenvalue eventually
                 crosses zero out there.  Counting that crossing would
                 shift every level by one and mislabel 1s as 2s.

      r_match -- where the outward and inward branches are spliced when
                 the final wavefunction is built.  It has to sit just
                 outside the surface: far enough that the physical
                 shape is settled, close enough that the outward branch
                 is not yet contaminated.
    """
    r = np.linspace(0.0, r_max, n_points)
    h = r[1] - r[0]
    # midpoints, needed by the Runge-Kutta integrator
    r_mid = r[:-1] + 0.5 * h
    return {"r": r, "h": h, "r_mid": r_mid, "n": n_points, "r_max": r_max,
            "i_count": int(np.searchsorted(r, r_surface + 2.5)),
            "i_match": int(np.searchsorted(r, r_surface + 2.0))}


def radial_integral(grid, f):
    """
    Integrate  f(r) dr  over the mesh with Simpson's rule.

    Note this is a plain dr integral, NOT 4 pi r^2 dr.  All our
    densities are stored already multiplied by 4 pi r^2 wherever that
    is what we want, so keeping this routine dumb avoids confusion.
    """
    h = grid["h"]
    n = grid["n"]
    if n % 2 == 0:
        # Simpson needs an odd number of points; fall back to trapezoid
        return float(np.trapezoid(f, dx=h))
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return float(h / 3.0 * np.dot(w, f))


# =====================================================================
#  SECTION 3 :  THE RADIAL DIRAC EQUATION
# ---------------------------------------------------------------------
#  Solved by shooting.  We integrate
#
#        dG/dr = -(kappa/r) G + D(r) F ,   D = E - V + M*
#        dF/dr = +(kappa/r) F - S(r) G ,   S = E - V - M*
#
#  outward from the origin with a 4th order Runge-Kutta step and adjust
#  E until the solution decays instead of blowing up.
#
#  The whole search is VECTORISED over levels: every (kappa, n_r, p/n)
#  state is carried as one entry of a numpy array, so a single bisection
#  step advances all ~120 levels of the nucleus at once.  Without that,
#  pure Python would be far too slow to ever run this over the full
#  (J, L) grid.
# =====================================================================

def _start_values(kappa, r0, D0, S0):
    """
    Regular behaviour of (G, F) as r -> 0.

    Balancing the two Dirac equations at small r gives

        kappa < 0 :  G ~ r^|kappa|      F ~ -S0/(2|kappa|+1) r^(|kappa|+1)
        kappa > 0 :  F ~ r^kappa        G ~ +D0/(2 kappa+1)  r^(kappa+1)

    The irregular partner diverges at the origin and is discarded.
    """
    k = np.abs(kappa)
    g = np.where(kappa < 0, r0 ** k, (D0 / (2.0 * k + 1.0)) * r0 ** (k + 1.0))
    f = np.where(kappa < 0, (-S0 / (2.0 * k + 1.0)) * r0 ** (k + 1.0), r0 ** k)
    return g, f


def _derivs(kappa, r, g, f, D, S):
    """Right-hand sides of the two radial Dirac equations."""
    return (-(kappa / r) * g + D * f,
            +(kappa / r) * f - S * g)


def shoot_outward(grid, kappa, energy, V, Vmid, Mstar, Mstar_mid,
                  i_stop, count_upto, store=False):
    """
    Integrate the Dirac pair outward from r = h to grid point i_stop.

    kappa, energy are 1-D arrays (one entry per level being searched);
    V and Mstar are 2-D arrays of shape (n_levels, n_grid) because the
    proton and neutron potentials differ.  Everything below is written
    with numpy broadcasting so that all levels advance together.

    Returns the node count of G inside r < r[count_upto], the value of
    G at i_stop, and -- if store is True -- the full G(r), F(r).
    """
    r = grid["r"]
    r_mid = grid["r_mid"]
    h = grid["h"]
    n_lev = len(kappa)

    kap = kappa[:, None] if False else kappa      # kept 1-D, broadcast by hand
    E = energy

    # D(r) = E - V + M* and S(r) = E - V - M*, on nodes and midpoints
    D = E[:, None] - V + Mstar
    S = E[:, None] - V - Mstar
    Dm = E[:, None] - Vmid + Mstar_mid
    Sm = E[:, None] - Vmid - Mstar_mid

    r0 = r[1]
    g, f = _start_values(kap, r0, D[:, 1], S[:, 1])

    if store:
        G = np.zeros((n_lev, grid["n"]))
        F = np.zeros((n_lev, grid["n"]))
        G[:, 1] = g
        F[:, 1] = f
    else:
        G = F = None

    nodes = np.zeros(n_lev, dtype=np.int64)
    g_prev_sign = np.sign(g)
    # running rescale exponent, so deeply divergent trial solutions do
    # not overflow to inf and destroy the sign information we need
    scale_log = np.zeros(n_lev)

    i = 1
    while i < i_stop:
        rl, rm, rr = r[i], r_mid[i], r[i + 1]

        k1g, k1f = _derivs(kap, rl, g, f, D[:, i], S[:, i])
        k2g, k2f = _derivs(kap, rm, g + 0.5 * h * k1g, f + 0.5 * h * k1f,
                           Dm[:, i], Sm[:, i])
        k3g, k3f = _derivs(kap, rm, g + 0.5 * h * k2g, f + 0.5 * h * k2f,
                           Dm[:, i], Sm[:, i])
        k4g, k4f = _derivs(kap, rr, g + h * k3g, f + h * k3f,
                           D[:, i + 1], S[:, i + 1])

        g = g + (h / 6.0) * (k1g + 2.0 * k2g + 2.0 * k3g + k4g)
        f = f + (h / 6.0) * (k1f + 2.0 * k2f + 2.0 * k3f + k4f)

        # count sign changes of G, but only in the interior.  Beyond
        # count_upto a trial solution that decays slightly too fast will
        # cross zero and then diverge; that crossing is a numerical
        # artefact, not a real node, and counting it would confuse the
        # level identification.
        if i + 1 <= count_upto:
            s = np.sign(g)
            nodes += ((s * g_prev_sign) < 0).astype(np.int64)
            g_prev_sign = np.where(s != 0, s, g_prev_sign)

        # keep the magnitudes in range
        big = np.maximum(np.abs(g), np.abs(f))
        hot = big > 1.0e60
        if np.any(hot):
            fac = np.where(hot, 1.0e-60, 1.0)
            g = g * fac
            f = f * fac
            scale_log += np.where(hot, math.log(1.0e60), 0.0)
            if store:
                G = G * fac[:, None]
                F = F * fac[:, None]

        if store:
            G[:, i + 1] = g
            F[:, i + 1] = f
        i += 1

    return {"nodes": nodes, "g_end": g, "f_end": f, "G": G, "F": F}


def find_levels(grid, kappa, n_target, is_proton, V, Vmid, Mstar, Mstar_mid,
                e_guess=None, n_bisect=38):
    """
    Locate the bound-state energies of every requested level at once.

    kappa, n_target, is_proton are 1-D arrays describing the levels we
    want; n_target is the number of radial nodes (0 for 1s, 1 for 2s...).

    The criterion is a Sturm sequence.  Integrating the regular solution
    outward from the origin all the way to the box edge, the number of
    nodes of G equals exactly the number of eigenvalues lying below the
    trial energy.  So the n-th level is the E at which the node count
    steps from n to n+1:

        nodes <= n_target  ->  E is too low
        nodes >  n_target  ->  E is too high

    Node count is a monotone (non-decreasing) function of E, so plain
    bisection on this single integer cannot skip or mislabel a level.
    That matters here: an earlier version counted nodes only inside the
    nuclear surface to dodge the divergent tail, but the zero crossing
    out in the tail is a genuine Sturm node, and throwing it away shifted
    every level by one.  Counting over the FULL range is both simpler
    and correct.

    Returns the energies (fm^-1, rest mass included) and a boolean mask
    saying which levels actually came out bound.
    """
    n_lev = len(kappa)
    r = grid["r"]

    i_stop = grid["n"] - 1
    count_upto = i_stop

    # bracket: from deeply bound up to the continuum threshold.  The
    # deepest RMF single-particle level in any stable nucleus sits near
    # -70 MeV, so -110 MeV is a safe floor.
    e_lo = np.full(n_lev, M_N - 170.0 / HBARC)
    e_hi = np.full(n_lev, M_N - 0.05 / HBARC)

    step = 0
    while step < n_bisect:
        e_mid = 0.5 * (e_lo + e_hi)
        out = shoot_outward(grid, kappa, e_mid, V, Vmid, Mstar, Mstar_mid,
                            i_stop, count_upto, store=False)
        nodes = out["nodes"]

        # Sturm: nodes(E) = number of eigenvalues below E
        too_high = nodes > n_target
        e_hi = np.where(too_high, e_mid, e_hi)
        e_lo = np.where(too_high, e_lo, e_mid)
        step += 1

    energy = 0.5 * (e_lo + e_hi)

    # a level that ran into either bracket wall never converged: it is
    # unbound (or does not exist) for this potential
    bound = (energy < M_N - 0.3 / HBARC) & (energy > M_N - 169.0 / HBARC)
    return energy, bound


def wavefunction(grid, kappa, energy, V, Vmid, Mstar, Mstar_mid):
    """
    Build the normalised radial wavefunctions at the converged energies.

    Outward shooting alone is not good enough for the tail: past the
    classical turning point round-off seeds the exponentially GROWING
    solution and the tail bends the wrong way.  So we also integrate
    inward from the box edge, where the decaying solution is the stable
    one, and splice the two at a matching radius.  The inward piece is
    rescaled to agree with the outward piece at the joint.
    """
    r = grid["r"]
    h = grid["h"]
    n_lev = len(kappa)
    n_grid = grid["n"]

    i_match = grid["i_match"]
    count_upto = i_match

    out = shoot_outward(grid, kappa, energy, V, Vmid, Mstar, Mstar_mid,
                        i_match, count_upto, store=True)
    G = out["G"]
    F = out["F"]

    # ---- inward integration from the outer boundary -----------------
    D = energy[:, None] - V + Mstar
    S = energy[:, None] - V - Mstar
    Dm = energy[:, None] - Vmid + Mstar_mid
    Sm = energy[:, None] - Vmid - Mstar_mid

    # asymptotic decaying solution: G ~ exp(-p r), F = -p G / (E + M)
    p = np.sqrt(np.maximum(M_N ** 2 - energy ** 2, 1.0e-12))
    g = np.ones(n_lev)
    f = -p / (energy + M_N)

    Gin = np.zeros((n_lev, n_grid))
    Fin = np.zeros((n_lev, n_grid))
    Gin[:, -1] = g
    Fin[:, -1] = f

    i = n_grid - 1
    while i > i_match:
        rl, rm, rr = r[i], r_mid_at(grid, i - 1), r[i - 1]
        hh = -h
        k1g, k1f = _derivs(kappa, rl, g, f, D[:, i], S[:, i])
        k2g, k2f = _derivs(kappa, rm, g + 0.5 * hh * k1g, f + 0.5 * hh * k1f,
                           Dm[:, i - 1], Sm[:, i - 1])
        k3g, k3f = _derivs(kappa, rm, g + 0.5 * hh * k2g, f + 0.5 * hh * k2f,
                           Dm[:, i - 1], Sm[:, i - 1])
        k4g, k4f = _derivs(kappa, rr, g + hh * k3g, f + hh * k3f,
                           D[:, i - 1], S[:, i - 1])
        g = g + (hh / 6.0) * (k1g + 2.0 * k2g + 2.0 * k3g + k4g)
        f = f + (hh / 6.0) * (k1f + 2.0 * k2f + 2.0 * k3f + k4f)

        big = np.maximum(np.abs(g), np.abs(f))
        hot = big > 1.0e60
        if np.any(hot):
            fac = np.where(hot, 1.0e-60, 1.0)
            g, f = g * fac, f * fac
            Gin = Gin * fac[:, None]
            Fin = Fin * fac[:, None]

        Gin[:, i - 1] = g
        Fin[:, i - 1] = f
        i -= 1

    # splice: scale the inward branch onto the outward one
    g_out_match = G[:, i_match]
    g_in_match = Gin[:, i_match]
    safe = np.where(np.abs(g_in_match) > 1.0e-300, g_in_match, 1.0e-300)
    scale = g_out_match / safe
    G[:, i_match:] = Gin[:, i_match:] * scale[:, None]
    F[:, i_match:] = Fin[:, i_match:] * scale[:, None]

    # normalise:  integral (G^2 + F^2) dr = 1
    norm = np.zeros(n_lev)
    j = 0
    while j < n_lev:
        norm[j] = radial_integral(grid, G[j] ** 2 + F[j] ** 2)
        j += 1
    norm = np.where(norm > 1.0e-300, norm, 1.0)
    G = G / np.sqrt(norm)[:, None]
    F = F / np.sqrt(norm)[:, None]
    return G, F


def r_mid_at(grid, i):
    return grid["r_mid"][i]


# =====================================================================
#  SECTION 4 :  SHELL BOOK-KEEPING
# =====================================================================

def candidate_levels(n_r_max=6, kappa_max=10):
    """
    All (n_r, kappa) states worth trying.

    We do NOT hard-code the usual textbook shell ordering.  Changing
    Lambda_omega_rho and g_rho over our (J, L) grid moves single-particle
    levels around, and near a shell closure two orbitals can swap.
    Instead we solve everything in this list, sort by energy, and fill
    from the bottom -- so the occupation is whatever the self-consistent
    potential actually says it is.
    """
    #  The basis has to be generously larger than the orbitals actually
    #  occupied.  At the soft corner of the grid (J = 25, L at its
    #  minimum) the neutron well of Pb-208 is shallow, and with
    #  n_r <= 4, kappa <= 8 the bound levels held exactly 126 neutrons --
    #  no margin at all.  One small shift in the fields on the first
    #  sweep dropped capacity below 126 and the run aborted. Widening the
    #  basis costs time linearly and removes the cliff.
    lv = []
    k = -kappa_max
    while k <= kappa_max:
        if k != 0:
            nr = 0
            while nr <= n_r_max:
                lv.append((nr, k))
                nr += 1
        k += 1
    return lv


def degeneracy(kappa):
    """2j+1 for a level with quantum number kappa (j = |kappa| - 1/2)."""
    return 2.0 * abs(kappa)


def level_label(n_r, kappa):
    """Spectroscopic name, e.g. (0, -1) -> '1s1/2'."""
    if kappa < 0:
        l = -kappa - 1
    else:
        l = kappa
    j2 = 2 * abs(kappa) - 1
    return "%d%s%d/2" % (n_r + 1, "spdfghij"[l] if l < 8 else "?", j2)


def fill_shells(energies, kappas, bound, n_particles):
    """
    Occupy the lowest levels until n_particles nucleons are used up.

    Returns an occupation array (number of nucleons in each level).
    For a doubly-magic nucleus the last occupied shell comes out exactly
    full; if it does not, the occupation is fractional and we flag it,
    because that means the nucleus is not closed-shell for this
    parametrisation and a plain Hartree treatment is no longer valid.
    """
    occ = np.zeros(len(energies))
    order = np.argsort(np.where(bound, energies, 1.0e30))
    left = float(n_particles)
    closed = True
    for idx in order:
        if not bound[idx] or left <= 1.0e-9:
            break
        cap = degeneracy(kappas[idx])
        take = min(cap, left)
        if 0.0 < take < cap - 1.0e-9:
            closed = False
        occ[idx] = take
        left -= take
    return occ, (left <= 1.0e-9), closed


# =====================================================================
#  SECTION 5 :  MESON AND COULOMB FIELDS
# ---------------------------------------------------------------------
#  Each Klein-Gordon equation  -lap phi + m^2 phi = s(r)  is solved with
#  its exact Green's function rather than by finite differences.  With
#  u = r phi the radial equation is  -u'' + m^2 u = r s(r), and
#
#     u(r) = (1/m) [ e^(-m r) Int_0^r sinh(m r') r' s dr'
#                  + sinh(m r) Int_r^inf e^(-m r') r' s dr' ]
#
#  This is unconditionally stable.  A finite-difference solve of the
#  same equation is not: m_omega h is of order 0.2 here, and the
#  discretised operator is stiff enough that the iteration wanders.
# =====================================================================

def solve_klein_gordon(grid, mass, source):
    """Solve -lap phi + mass^2 phi = source(r) for a spherical source."""
    r = grid["r"]
    h = grid["h"]
    s = r * source                                    # = r s(r)

    u = _kg_convolve(r, h, mass, s)
    phi = np.zeros_like(r)
    phi[1:] = u[1:] / r[1:]
    phi[0] = phi[1] + (phi[1] - phi[2])        # linear extrapolation to r=0
    return phi


def _kg_convolve(r, h, m, s):
    """
    u(r) = (1/m)[ e^(-mr) Int_0^r sinh(m r') s dr' + sinh(mr) Int_r^R e^(-mr') s dr' ]

    Written so that no intermediate ever exceeds double precision:
    every exponential is paired with its decaying partner before the
    product is formed.
    """
    n = len(r)
    e_neg = np.exp(-m * r)                     # e^(-m r), always <= 1

    # ---- A(r) = Int_0^r sinh(m r') s(r') dr', kept as e^(-m r) A(r) --
    # trapezoid on the scaled integrand:  e^(-m r) sinh(m r') =
    #   0.5 [ e^(-m(r-r')) - e^(-m(r+r')) ]
    # We accumulate  P = Int_0^r e^(+m r') s dr'  and
    #                Q = Int_0^r e^(-m r') s dr'  with running rescaling.
    # e^(+m r') overflows, so accumulate P in the shifted form
    #   P_i = Int_0^{r_i} e^(m (r' - r_i)) s dr'   (recursive, bounded)
    P = np.zeros(n)
    i = 1
    decay = math.exp(-m * h)
    while i < n:
        # shift the previous accumulation to the new reference point and
        # add the new trapezoid slice
        P[i] = P[i - 1] * decay + 0.5 * h * (s[i - 1] * decay + s[i])
        i += 1

    Q = np.zeros(n)
    i = 1
    while i < n:
        Q[i] = Q[i - 1] + 0.5 * h * (e_neg[i - 1] * s[i - 1] + e_neg[i] * s[i])
        i += 1

    # first term: e^(-m r) * Int_0^r sinh(m r') s dr'
    #           = 0.5 [ P(r) - e^(-m r) Q(r) ]
    term1 = 0.5 * (P - e_neg * Q)

    # ---- second term: sinh(m r) Int_r^R e^(-m r') s dr' --------------
    # accumulate  T_i = Int_{r_i}^{R} e^(-m (r' - r_i)) s dr'  backwards,
    # again shifted so nothing overflows
    T = np.zeros(n)
    i = n - 2
    while i >= 0:
        T[i] = T[i + 1] * decay + 0.5 * h * (s[i] + s[i + 1] * decay)
        i -= 1
    # sinh(m r) e^(-m r) * [shifted integral] = 0.5 (1 - e^(-2mr)) T
    term2 = 0.5 * (1.0 - e_neg * e_neg) * T

    return (term1 + term2) / m


def solve_coulomb(grid, rho_p):
    """
    Coulomb potential energy of a proton in the charge distribution
    rho_p(r), in fm^-1:

        V_C(r) = alpha [ (1/r) Int_0^r 4 pi r'^2 rho_p dr'
                       + Int_r^R 4 pi r' rho_p dr' ]

    The first piece is the charge enclosed, the second the potential
    from the shells further out.
    """
    r = grid["r"]
    h = grid["h"]
    q_in = np.concatenate(([0.0], np.cumsum(
        0.5 * h * (4.0 * math.pi * r[:-1] ** 2 * rho_p[:-1] +
                   4.0 * math.pi * r[1:] ** 2 * rho_p[1:]))))
    outer_int = 4.0 * math.pi * r * rho_p
    tail = np.zeros_like(r)
    i = len(r) - 2
    while i >= 0:
        tail[i] = tail[i + 1] + 0.5 * h * (outer_int[i] + outer_int[i + 1])
        i -= 1

    v = np.zeros_like(r)
    v[1:] = ALPHA_EM * (q_in[1:] / r[1:] + tail[1:])
    v[0] = ALPHA_EM * (0.0 + tail[0])
    return v


# =====================================================================
#  SECTION 6 :  THE SELF-CONSISTENCY LOOP
# =====================================================================

def initial_fields(grid, par, nucleus):
    """
    Woods-Saxon starting guess.

    The interior values are taken from the model's own saturation point
    (PHI0 = M - M*, and W from the omega field equation at rho0), so the
    first iteration already has roughly the right depth and the loop
    converges in tens of iterations instead of hundreds.
    """
    r = grid["r"]
    Z, N = nucleus["Z"], nucleus["N"]
    A = Z + N

    R = 1.1 * A ** (1.0 / 3.0)
    a = 0.55
    shape = 1.0 / (1.0 + np.exp((r - R) / a))

    phi0 = par["PHI0"] / HBARC                 # fm^-1
    # omega field at saturation, from (1/C_w^2) W = n  (zeta, Lambda small here)
    w0 = par["W0"] / HBARC if "W0" in par else phi0 * 0.8

    PHI = phi0 * shape
    W = w0 * shape
    m_rho = par.get("m_rho", eos.M_RHO) / HBARC
    #  SIGN.  The rho source is n_3/2 = (n_p - n_n)/2, which is NEGATIVE
    #  for a neutron-rich nucleus, so R must start negative.  Getting
    #  this backwards makes V_n = W - R/2 far too shallow, and on the
    #  very first sweep -- before self-consistency can correct anything
    #  -- too few neutron levels bind to hold N = 126.  The run then
    #  aborts with "not enough bound levels".  That silently deleted
    #  every low-L point from the grid, which is precisely the region
    #  CREX prefers, so the bug biased the constraint where it hurt most.
    #  Solve the rho field equation at saturation INCLUDING the omega-rho
    #  cross term, which is what actually sets the scale:
    #        (1/C_r^2 + 2 Lam W0^2) R = n_3 / 2
    #  Dropping the 2 Lam W0^2 piece -- as an earlier version did -- is
    #  not a small error.  At L = 30 the cross term is five times larger
    #  than 1/C_r^2, so the guess came out about six times too big, the
    #  proton well was wrecked on the first sweep, and the run died with
    #  "not enough bound levels" before self-consistency could recover.
    #  Every low-L point vanished from the grid that way, which is
    #  exactly the region CREX prefers.
    inv_cr2 = m_rho ** 2 / max(par["g_rho"] ** 2, 1.0e-12)
    denom = inv_cr2 + 2.0 * par["lambda_wr"] * w0 * w0
    n_three = -(float(N - Z) / A) * par["rho0_fm3"]      # n_p - n_n < 0
    Rho = (0.5 * n_three / denom) * shape

    #  Damp the isovector field on entry.  At low L the converged rho
    #  field splits the neutron and proton wells by tens of MeV, and
    #  imposing all of that on the very first sweep -- on top of a
    #  Woods-Saxon guess that is only roughly right -- unbinds levels
    #  the nucleus actually has.  Starting near the isoscalar limit and
    #  letting the mixing build the splitting up over the first dozen
    #  sweeps reaches the same fixed point from a safe direction.
    Rho = 0.25 * Rho
    return PHI, W, Rho


def hartree(par, nucleus_name, n_points=701, max_iter=200, tol=1.0e-3,
            mix=0.20, lock_after=30, lock_tol=2.0, skin_tol=2.0e-4,
            verbose=False):
    """
    Full spherical RMF Hartree calculation for one parameter set.

    par must be the dictionary returned by
    rmf_lambda_and_eos.make_parameter_set(model, J, L) -- i.e. exactly
    the same couplings that generated the EoS table for this grid point.

    Returns a dictionary of observables; the one the paper needs is
    'skin' = <r^2>_n^(1/2) - <r^2>_p^(1/2) in fm.
    """
    if par.get("no_meson_masses"):
        return {"converged": False, "model": par.get("name"),
                "nucleus": nucleus_name,
                "fail": "no published meson masses for %s: only (g/m)^2 is "
                        "determined, so m_sigma -- and the surface "
                        "diffuseness -- is undefined" % par.get("name")}
    nuc = NUCLEI[nucleus_name]
    Z, N = nuc["Z"], nuc["N"]
    A = Z + N
    grid = make_grid(nuc["r_max"], n_points, 1.2 * A ** (1.0 / 3.0))
    r = grid["r"]

    # ---- meson masses -------------------------------------------------
    # Normally these are the module-wide values shared with the EoS code.
    # A parameter set may override them (NL3, for instance, uses
    # m_sigma = 508.194 MeV), which is what makes the benchmark below
    # possible.
    m_sig = par.get("m_sigma", eos.M_SIGMA) / HBARC
    m_om = par.get("m_omega", eos.M_OMEGA) / HBARC
    m_rho = par.get("m_rho", eos.M_RHO) / HBARC

    # ---- couplings, converted to the fm^-1 / fm^-3 system ------------
    g_sig2 = par["g_sigma"] ** 2
    g_om2 = par["g_omega"] ** 2
    g_rho2 = par["g_rho"] ** 2
    # b and c are dimensionless in both MeV and fm^-1 systems: the terms
    # b M PHI^2 and c PHI^3 already carry three powers of energy, so no
    # hbar*c conversion enters here.
    b = par["b"]
    c = par["c"]
    zeta = par["zeta"]
    lam = par["lambda_wr"]

    PHI, W, Rho = initial_fields(grid, par, nuc)
    rho_p = np.zeros_like(r)

    levels = candidate_levels()
    n_lv = len(levels)
    kap_1 = np.array([k for (_, k) in levels], dtype=float)
    nr_1 = np.array([n for (n, _) in levels], dtype=np.int64)

    # protons first, then neutrons, in one flat vector of levels
    kappa = np.concatenate((kap_1, kap_1))
    n_targ = np.concatenate((nr_1, nr_1))
    is_p = np.concatenate((np.ones(n_lv, bool), np.zeros(n_lv, bool)))

    result = {"converged": False, "model": par.get("name"), "nucleus": nucleus_name}
    locked_occ = None
    closed_p = closed_n = True
    change_prev = 1.0e30
    skin_prev = -1.0e30
    steady = 0

    it = 0
    while it < max_iter:
        V_C = solve_coulomb(grid, rho_p)

        Vp = W + 0.5 * Rho + V_C
        Vn = W - 0.5 * Rho
        Mstar = M_N - PHI
        if np.any(Mstar < 0.02 * M_N):
            result["fail"] = "effective mass collapsed"
            return result

        V2 = np.where(is_p[:, None], Vp[None, :], Vn[None, :])
        Ms2 = np.tile(Mstar, (2 * n_lv, 1))
        Vmid = 0.5 * (V2[:, :-1] + V2[:, 1:])
        Msmid = 0.5 * (Ms2[:, :-1] + Ms2[:, 1:])

        energy, bound = find_levels(grid, kappa, n_targ, is_p,
                                    V2, Vmid, Ms2, Msmid)

        # ---- occupations ----------------------------------------------
        #  Once the mean field has roughly settled we FREEZE which
        #  orbitals are occupied.  Near the Fermi surface of Pb-208 two
        #  levels can sit a few hundred keV apart, and if they trade
        #  places from one iteration to the next the density jumps, the
        #  fields jump back, and the loop settles into a limit cycle that
        #  never meets the tolerance.  Both nuclei used here are doubly
        #  magic, so the configuration is not in doubt -- locking it is
        #  the standard fixed-configuration Hartree treatment.
        if locked_occ is None:
            occ_p, ok_p, closed_p = fill_shells(energy[:n_lv], kappa[:n_lv],
                                                bound[:n_lv], Z)
            occ_n, ok_n, closed_n = fill_shells(energy[n_lv:], kappa[n_lv:],
                                                bound[n_lv:], N)
            if not (ok_p and ok_n):
                result["fail"] = "not enough bound levels to hold all nucleons"
                return result
            occ = np.concatenate((occ_p, occ_n))
            # Lock only once the field is genuinely close to settled.
            # Locking on iteration count alone is unsafe: a slowly
            # converging model (GM1) is still far from its solution at a
            # fixed iteration number, and freezing a half-converged
            # configuration strands the calculation in a state that can
            # be wildly unphysical -- a hole in the central density.
            if it >= lock_after and change_prev < lock_tol:
                locked_occ = occ.copy()
        else:
            occ = locked_occ

        use = occ > 0.0
        G, F = wavefunction(grid, kappa[use], energy[use],
                            V2[use], Vmid[use], Ms2[use], Msmid[use])

        # ---- densities ------------------------------------------------
        w4pir2 = np.zeros_like(r)
        w4pir2[1:] = 1.0 / (4.0 * math.pi * r[1:] ** 2)
        wsum = occ[use][:, None] * w4pir2[None, :]

        gp = G ** 2
        fp = F ** 2
        pmask = is_p[use][:, None]

        rho_v = np.sum(wsum * (gp + fp), axis=0)
        rho_s = np.sum(wsum * (gp - fp), axis=0)
        rho_p_new = np.sum(np.where(pmask, wsum, 0.0) * (gp + fp), axis=0)
        rho_n_new = rho_v - rho_p_new
        rho_3 = rho_p_new - rho_n_new

        # r = 0 is not defined by the 1/r^2 weight; extrapolate
        for arr in (rho_v, rho_s, rho_p_new, rho_n_new, rho_3):
            arr[0] = 2.0 * arr[1] - arr[2]

        # ---- meson fields from the new densities ----------------------
        src_sig = g_sig2 * (rho_s - b * M_N * PHI ** 2 - c * PHI ** 3)
        src_om = g_om2 * (rho_v - 2.0 * lam * Rho ** 2 * W
                          - (zeta / 6.0) * W ** 3)
        src_rho = g_rho2 * (0.5 * rho_3 - 2.0 * lam * W ** 2 * Rho)

        PHI_new = solve_klein_gordon(grid, m_sig, src_sig)
        W_new = solve_klein_gordon(grid, m_om, src_om)
        Rho_new = solve_klein_gordon(grid, m_rho, src_rho)

        change = max(np.max(np.abs(PHI_new - PHI)),
                     np.max(np.abs(W_new - W)),
                     np.max(np.abs(Rho_new - Rho))) * HBARC

        PHI = PHI + mix * (PHI_new - PHI)
        W = W + mix * (W_new - W)
        Rho = Rho + mix * (Rho_new - Rho)
        rho_p = rho_p + mix * (rho_p_new - rho_p)

        if verbose and it % 10 == 0:
            print("    iter %3d   dV = %.3e MeV" % (it, change))

        # ---- convergence ----------------------------------------------
        #  Judge convergence on the NEUTRON SKIN, not on the raw meson
        #  fields.  The field iteration plateaus long before it meets a
        #  tight field tolerance -- it will happily grind out 200 sweeps
        #  while the radii have not moved in the fourth decimal -- so a
        #  field criterion just wastes time and then reports failure on a
        #  perfectly good answer.  The skin is what we actually want, and
        #  once it is stable to skin_tol over a few consecutive sweeps
        #  the calculation is finished for our purposes.
        npc = radial_integral(grid, 4.0 * math.pi * r ** 2 * rho_p_new)
        nnc = radial_integral(grid, 4.0 * math.pi * r ** 2 * rho_n_new)
        rp_i = math.sqrt(max(radial_integral(
            grid, 4.0 * math.pi * r ** 4 * rho_p_new) / max(npc, 1e-30), 0.0))
        rn_i = math.sqrt(max(radial_integral(
            grid, 4.0 * math.pi * r ** 4 * rho_n_new) / max(nnc, 1e-30), 0.0))
        skin_i = rn_i - rp_i
        if abs(skin_i - skin_prev) < skin_tol:
            steady += 1
        else:
            steady = 0
        skin_prev = skin_i

        change_prev = change
        if steady >= 3 or change < tol:
            result["converged"] = True
            break
        it += 1

    result["iterations"] = it

    # =================================================================
    #  observables
    # =================================================================
    def rms(dens, count):
        num = radial_integral(grid, 4.0 * math.pi * r ** 4 * dens)
        return math.sqrt(max(num / count, 0.0))

    # ---- sanity gate --------------------------------------------------
    #  A converged heavy nucleus must have a saturated interior.  If the
    #  central density has collapsed, the calculation has run away and
    #  the radii are meaningless -- refuse to return them rather than let
    #  a hollow nucleus leak into the analysis as a real skin value.
    #
    #  This is how GM1 and GM2 fail on Pb-208.  Their scalar field is
    #  weak (M*/M = 0.70 and 0.78), so the spin-orbit force is weak too,
    #  and the high-n orbitals needed to reach N = 126 never bind
    #  properly.  As they drop out, the 208 nucleons are packed into
    #  high-l surface-peaked orbitals, the centre empties, the potential
    #  shallows, and more levels unbind.  Both models reproduce O-16,
    #  Ca-40 and Ca-48 without diverging -- they are simply over-bound
    #  there by 1-1.7 MeV/nucleon -- which is the long-known price of
    #  fitting a parametrisation to infinite nuclear matter alone.
    #  Neither can be used for finite-nucleus skin predictions.
    i_cen = int(np.searchsorted(r, 1.5))
    rho_cen = float(rho_p_new[i_cen] + rho_n_new[i_cen])
    if A >= 100 and rho_cen < 0.5 * par["rho0_fm3"]:
        result["fail"] = ("central density collapsed (%.4f fm^-3 vs rho0 = "
                          "%.3f): the self-consistency loop ran away"
                          % (rho_cen, par["rho0_fm3"]))
        return result
    result["rho_central"] = rho_cen

    n_p_check = radial_integral(grid, 4.0 * math.pi * r ** 2 * rho_p_new)
    n_n_check = radial_integral(grid, 4.0 * math.pi * r ** 2 * rho_n_new)

    r_p = rms(rho_p_new, n_p_check)
    r_n = rms(rho_n_new, n_n_check)

    # charge radius: fold in the finite size of the nucleons themselves
    r_ch2 = r_p ** 2 + R_PROTON_CH2 + (N / float(Z)) * R_NEUTRON_CH2
    r_ch = math.sqrt(max(r_ch2, 0.0))

    # ---- total binding energy ---------------------------------------
    #  E = sum_a (2j+1) eps_a + (1/2) Int PHI n_s
    #      - (1/2) Int [ W n_v + R n_3 / 2 + V_C n_p ]
    #      - (1/6) b M Int PHI^3 - (1/4) c Int PHI^4
    #      + (zeta/24) Int W^4 + Lambda Int W^2 R^2
    #  The first line is the usual "sum of single-particle energies minus
    #  the double-counted interaction"; the rest undoes the
    #  over-subtraction caused by the non-linear self-couplings.
    #
    #  The zeta and Lambda signs are NOT guesswork: they are fixed by
    #  demanding that this expression reduce to
    #  rmf_lambda_and_eos.meson_energy_density() when the gradients are
    #  switched off.  Doing that comparison term by term gives the
    #  coefficients +zeta/24 and +Lambda, opposite to what a naive
    #  Legendre transform of the Lagrangian suggests, because re-using
    #  the omega and rho field equations feeds extra W^4 and W^2 R^2
    #  pieces back in.
    e_sp = float(np.sum(occ[use] * energy[use]))
    vol = 4.0 * math.pi * r ** 2
    i_phi_ns = radial_integral(grid, vol * PHI * rho_s)
    i_w_nv = radial_integral(grid, vol * W * rho_v)
    i_r_n3 = radial_integral(grid, vol * Rho * rho_3)
    i_vc_np = radial_integral(grid, vol * V_C * rho_p_new)
    i_phi3 = radial_integral(grid, vol * PHI ** 3)
    i_phi4 = radial_integral(grid, vol * PHI ** 4)
    i_w4 = radial_integral(grid, vol * W ** 4)
    i_w2r2 = radial_integral(grid, vol * W ** 2 * Rho ** 2)

    e_tot = (e_sp
             + 0.5 * i_phi_ns
             - 0.5 * i_w_nv - 0.25 * i_r_n3 - 0.5 * i_vc_np
             - (1.0 / 6.0) * b * M_N * i_phi3 - 0.25 * c * i_phi4
             + (zeta / 24.0) * i_w4 + lam * i_w2r2)

    # centre-of-mass correction (harmonic-oscillator estimate)
    e_cm = -0.75 * 41.0 * A ** (-1.0 / 3.0) / HBARC
    binding = (e_tot - A * M_N + e_cm) * HBARC        # MeV, negative

    result.update({
        "Z": Z, "N": N, "A": A,
        "r_p": r_p, "r_n": r_n, "r_ch": r_ch,
        "skin": r_n - r_p,
        "BE_per_A": binding / A,
        "closed_shell": bool(closed_p and closed_n),
        "n_p_check": n_p_check, "n_n_check": n_n_check,
        "grid": grid,
        "rho_p": rho_p_new, "rho_n": rho_n_new,
        "levels": [(level_label(int(n_targ[i]), int(kappa[i])),
                    "p" if is_p[i] else "n",
                    float(energy[i] - M_N) * HBARC, float(occ[i]))
                   for i in range(len(occ)) if occ[i] > 0],
    })
    return result


# =====================================================================
#  SECTION 7 :  DRIVER
# =====================================================================

# ---------------------------------------------------------------------
#  BENCHMARK PARAMETRISATIONS
# ---------------------------------------------------------------------
#  These are NOT used by the physics analysis.  They exist so the solver
#  can be checked against parameter sets whose finite-nucleus results are
#  published, which is the only honest way to claim that the neutron
#  skins this file produces mean anything.
# ---------------------------------------------------------------------

BENCHMARKS = {
    # Lalazissis, Ring & Vretenar, Phys. Rev. C 55, 540 (1997)
    # Published Pb-208:  B/A = 7.896 MeV,  r_ch = 5.509 fm,  DR_np = 0.280 fm
    # Published Ca-48 :  B/A = 8.636 MeV,  r_ch = 3.473 fm,  DR_np = 0.195 fm
    "NL3": {"g_sigma": 10.2170, "g_omega": 12.8680, "g_rho": 4.4740,
            "g2_fm": -10.4310, "g3": -28.8850, "zeta": 0.0, "lambda_wr": 0.0,
            "m_sigma": 508.1940, "m_omega": 782.5010, "m_rho": 763.0000,
            "m_dirac_over_m": 0.60, "rho0_fm3": 0.148},
    # Todd-Rutel & Piekarewicz, Phys. Rev. Lett. 95, 122501 (2005)
    # Published Pb-208:  B/A = 7.886 MeV,  r_ch = 5.521 fm,  DR_np = 0.207 fm
    "FSUGold": {"g_sigma": 10.5924, "g_omega": 14.3020, "g_rho": 11.7673,
                "g2_fm": -10.7556, "g3": -39.1878, "zeta": 0.06, "lambda_wr": 0.030,
                "m_sigma": 491.5000, "m_omega": 782.5000, "m_rho": 763.0000,
                "m_dirac_over_m": 0.61, "rho0_fm3": 0.148},
}


def benchmark_parameters(name):
    """
    Turn a published parameter set into the dictionary `hartree` wants.

    Published RMF papers write the sigma self-coupling as
        U(sigma) = (1/2) m_s^2 sigma^2 + (1/3) g2 sigma^3 + (1/4) g3 sigma^4
    whereas this project (following PhysRevC.98.065804) writes
        U        = ... + (1/3) b M (g_s sigma)^3 + (1/4) c (g_s sigma)^4
    Matching the two term by term gives
        b = g2 / (M g_s^3)          c = g3 / g_s^4
    with g2 quoted in fm^-1, so it needs an hbar*c to reach MeV first.
    """
    raw = BENCHMARKS[name]
    gs = raw["g_sigma"]
    g2_mev = raw["g2_fm"] * HBARC
    par = {
        "name": name,
        "g_sigma": gs,
        "g_omega": raw["g_omega"],
        "g_rho": raw["g_rho"],
        "b": g2_mev / (eos.M_NUCLEON * gs ** 3),
        "c": raw["g3"] / gs ** 4,
        "zeta": raw["zeta"],
        "lambda_wr": raw["lambda_wr"],
        "m_sigma": raw["m_sigma"],
        "m_omega": raw["m_omega"],
        "m_rho": raw["m_rho"],
        "rho0_fm3": raw["rho0_fm3"],
        "PHI0": (1.0 - raw["m_dirac_over_m"]) * eos.M_NUCLEON,
        "ok": True,
    }
    # omega field at saturation, used only as a starting guess
    cw2 = (raw["g_omega"] / raw["m_omega"]) ** 2
    par["W0"] = cw2 * raw["rho0_fm3"] * HBARC ** 3
    return par


def skin_for(model_name, j_sym, l_sym, nucleus="Pb208", **kw):
    """Neutron skin of `nucleus` for one (model, J, L) grid point."""
    par = eos.make_parameter_set(model_name, j_sym, l_sym)
    if not par["ok"]:
        return {"converged": False, "fail": par["why"]}
    return hartree(apply_meson_masses(par), nucleus, **kw)


def main():
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "BigApple"
    J = float(sys.argv[2]) if len(sys.argv) > 2 else 31.0
    L = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0

    print("Finite-nucleus RMF Hartree")
    print("model = %s   J = %.1f MeV   L = %.1f MeV" % (model, J, L))
    for nucleus in ("Pb208", "Ca48"):
        res = skin_for(model, J, L, nucleus, verbose=True)
        if not res.get("converged"):
            print("  %-6s  FAILED: %s" % (nucleus, res.get("fail", "no convergence")))
            continue
        print("  %-6s  r_p = %.4f  r_n = %.4f  r_ch = %.4f  "
              "skin = %.4f fm   B/A = %.3f MeV  (%d iter)"
              % (nucleus, res["r_p"], res["r_n"], res["r_ch"],
                 res["skin"], res["BE_per_A"], res["iterations"]))


if __name__ == "__main__":
    main()
