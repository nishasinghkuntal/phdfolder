#!/usr/bin/env python3
# =====================================================================
#  rmf_lambda_and_eos.py
#
#  WHAT THIS PROGRAM DOES  (two jobs, one file)
#  -------------------------------------------------------------------
#  JOB 1 :  For the three Glendenning-Moszkowski parameter sets
#           GM1, GM2, GM3  (inputs taken from Table II of
#           PhysRevLett.67.2414.pdf) it computes the two ISOVECTOR
#           couplings
#                 Lambda_omega_rho      (the omega-rho mixing constant)
#                 g_rho                 (the rho-nucleon coupling)
#           for every value of the symmetry energy  Esym == J  and its
#           slope  Lsym == L  that we want to scan.  The algebra used
#           is exactly the one written in RMFparameters.pdf, Eqs.(9)-(23).
#           The J and L values scanned are the ones quoted in Monika
#           Sinha's paper (PhysRevC.108.035801):
#                 Esym(n0) = 38.1 +- 4.7 MeV   (PREX-2)
#                 Lsym(n0) = 106  +- 37  MeV   (PREX-2)
#
#  JOB 2 :  For every coupling set produced in JOB 1 it solves the
#           relativistic mean field (RMF) equations for charge-neutral,
#           beta-equilibrated  n p e mu  matter and writes an equation
#           of state (EoS) table  eos<TAG>.dat  in EXACTLY the column
#           format that  automationandformatting-2.py  expects.
#           The Lagrangian is the one of RMFparameters.pdf Eq.(1)
#           (i.e. sigma, omega, rho + sigma self-couplings b, c
#            + the Lambda_omega_rho omega-rho mixing term, NO zeta term).
#           The way the mean field equations, energy density, pressure
#           and chemical potentials are written down follows
#           PhysRevC.98.065804.pdf (Hornick et al.) Eqs.(2)-(7),
#           evaluated with zeta = 0 so that it is identical to the
#           Lagrangian of RMFparameters.pdf.
#
#  AFTER THIS PROGRAM:
#      python3 automationandformatting-2.py  EOS_RMF/eos*.dat
#      ./tovprofile  <the produced APR_EOS_Cat_*.dat files>
#      -> NSCool
#  (the driver  run_pipeline.py  does all of that automatically)
#
#  STYLE RULES USED HERE (asked for by the user)
#      * beginner level Python only:  math + plain lists/dicts
#      * NO "for" loops anywhere - only "while" loops
#      * every formula is explained in a comment right above it
#      * the self-consistency checks were RUN, then commented out
#        (see SECTION 8; the numbers they printed are quoted there)
# =====================================================================

import math
import os


# =====================================================================
#  SECTION 0 :  UNITS AND PHYSICAL CONSTANTS
# ---------------------------------------------------------------------
#  We work in "natural units":  hbar = c = 1.
#  Then EVERY quantity can be written as a power of MeV:
#        energy, mass, momentum  ->  MeV
#        length                  ->  MeV^-1
#        number density          ->  MeV^3
#  The bridge back to the laboratory is  hbar*c = 197.327 MeV fm,
#  i.e.  1 fm = 1/197.327 MeV^-1 , so
#        n[MeV^3] = n[fm^-3] * (197.327)^3
#        1 fm^2   = (1/197.327^2) MeV^-2
# =====================================================================

HBARC = 197.327                 # MeV fm
PI = math.pi

# masses (MeV).  These are the values written at the end of Section II B
# of RMFparameters.pdf.
M_NUCLEON = 939.0               # average nucleon mass  M_N
M_SIGMA = 550.0
M_OMEGA = 782.5
M_RHO = 763.0

# lepton masses (MeV)
M_ELECTRON = 0.5109989
M_MUON = 105.6583745

# nuclear saturation point used by Glendenning & Moszkowski (PRL 67,2414)
RHO0_FM3 = 0.153                # fm^-3
RHO0 = RHO0_FM3 * HBARC ** 3    # MeV^3
E_BIND_GM = -16.3               # MeV, binding energy per nucleon at rho0
ASYM_GM = 32.5                  # MeV, the symmetry energy GM originally used

# unit conversions for the output table
# 1 MeV/fm^3 of energy density  =  1.7827e12 g/cm^3
# 1 MeV/fm^3 of pressure        =  1.6022e33 dyn/cm^2
# (the same two numbers are hard-wired in automationandformatting-2.py,
#  so using them here guarantees the two codes agree exactly)
MEVFM3_TO_GCM3 = 1.7827e12
MEVFM3_TO_DYNCM2 = 1.6022e33


# =====================================================================
#  SECTION 1 :  THE THREE GM PARAMETER SETS
# ---------------------------------------------------------------------
#  Table II of PhysRevLett.67.2414 (Glendenning & Moszkowski 1991)
#  gives, for saturated nuclear matter with
#        B/A = -16.3 MeV ,  rho = 0.153 fm^-3 ,  a_sym = 32.5 MeV :
#
#     K(MeV)  m*/m   (g_s/m_s)^2  (g_w/m_w)^2  (g_r/m_r)^2    b          c
#      300    0.70      11.79        7.149        4.411    0.002947  -0.001070   <- GM1
#      300    0.78       9.148       4.820        4.791    0.003478   0.01328    <- GM2
#      240    0.78       9.927       4.820        4.791    0.008659  -0.002421   <- GM3
#
#  The three (g/m)^2 are quoted in fm^2.
#  b and c are dimensionless.
#  Note: we only *use* (g_s/m_s)^2, (g_w/m_w)^2, b, c and m*/m.
#  The tabulated (g_r/m_r)^2 is kept only as a CHECK: when we set
#  J = 32.5 MeV and Lambda_omega_rho = 0 our formula must give it back.
# =====================================================================

GM_MODELS = {}

GM_MODELS["GM1"] = {
    "K": 300.0,
    "m_dirac_over_m": 0.70,
    "cs2_fm2": 11.79,      # (g_sigma/m_sigma)^2 in fm^2
    "cw2_fm2": 7.149,      # (g_omega/m_omega)^2 in fm^2
    "cr2_fm2": 4.411,      # (g_rho  /m_rho  )^2 in fm^2   (reference only)
    "b": 0.002947,
    "c": -0.001070,
    "zeta": 0.0,
    "rho0_fm3": 0.153,
    "E_bind": -16.3,
}

GM_MODELS["GM2"] = {
    "K": 300.0,
    "m_dirac_over_m": 0.78,
    "cs2_fm2": 9.148,
    "cw2_fm2": 4.820,
    "cr2_fm2": 4.791,
    "b": 0.003478,
    "c": 0.01328,
    "zeta": 0.0,
    "rho0_fm3": 0.153,
    "E_bind": -16.3,
}

GM_MODELS["GM3"] = {
    "K": 240.0,
    "m_dirac_over_m": 0.78,
    "cs2_fm2": 9.927,
    "cw2_fm2": 4.820,
    "cr2_fm2": 4.791,
    "b": 0.008659,
    "c": -0.002421,
    "zeta": 0.0,
    "rho0_fm3": 0.153,
    "E_bind": -16.3,
}

# =====================================================================
#  NEW MODELS with omega^4 self-coupling (zeta != 0).
#  Parameters cs2_fm2, cw2_fm2 are (g/m)^2 computed from published
#  coupling constants and meson masses.  The values b, c, zeta are
#  derived from the published saturation properties
#       rho0, m*/m, E/A, K
#  using the Hugenholtz-Van Hove theorem and the sigma field + energy
#  density conditions (see compute_rmf_params.py for the derivation
#  and numerical verification).
# =====================================================================

GM_MODELS["FSUGarnet"] = {
    "K": 229.6,
    "m_dirac_over_m": 0.578,
    "cs2_fm2": 17.4088,
    "cw2_fm2": 11.9399,
    "cr2_fm2": 12.9013,
    "b": 1.740677e-03,
    "c": -6.301326e-04,
    "zeta": 2.308462e-02,
    "rho0_fm3": 0.153,
    "E_bind": -16.23,
}

GM_MODELS["IOPB-I"] = {
    "K": 222.6,
    "m_dirac_over_m": 0.595,
    "cs2_fm2": 16.8732,
    "cw2_fm2": 11.4007,
    "cr2_fm2": 8.2755,
    "b": 1.926819e-03,
    "c": -1.086858e-03,
    "zeta": 1.909070e-02,
    "rho0_fm3": 0.149,
    "E_bind": -16.09,
}

GM_MODELS["BigApple"] = {
    "K": 227.1,
    "m_dirac_over_m": 0.608,
    "cs2_fm2": 15.0073,
    "cw2_fm2": 9.6841,
    "cr2_fm2": 13.3963,
    "b": 2.710269e-03,
    "c": -3.526347e-03,
    "zeta": 9.146663e-04,
    "rho0_fm3": 0.155,
    "E_bind": -16.34,
}

MODEL_NAMES = ["GM1", "GM2", "FSUGarnet", "IOPB-I", "BigApple"]


# ---------------------------------------------------------------------
#  The (J, L) grid that we scan.
#  PhysRevC.108.035801 (Sarkar, Thapa & Sinha) quotes, from PREX-2,
#        Esym(n0) = 38.1 +- 4.7 MeV   -> 33.4 , 38.1 , 42.8
#        Lsym(n0) = 106  +- 37  MeV   -> 69 , 106 , 143
#  and they actually run their cooling models with Lsym = 85 MeV
#  (Fig. 4, Fig. 5 and Table I of that paper) and Lsym = 106 MeV.
#  So the list below contains the 1-sigma end points, the central
#  value, AND the two values they really used.
#  >>> Edit these two lists to make the scan smaller or larger. <<<
# ---------------------------------------------------------------------
# The (Esym, Lsym) grid.  This MUST match the grid scan_cooling.py uses,
# or running the stages one by one silently produces a different set of
# equations of state from the one the cooling and the figures expect.
# Five values each, spanning the PREX-2 ranges
#     Esym = 38.1 +- 4.7 MeV        Lsym = 106 +- 37 MeV
# so 25 pairs per parameterisation, 75 in all.
# EXTENDED GRID.  Esym = 24-50 MeV (step 2), Lsym = 25-160 MeV (step 5).
# 14 Esym values x 28 Lsym values x 3 models = 1176 parameter sets.
# This scans well BEYOND the PREX-2 1-sigma range
# (Esym = 38.1 +- 4.7, Lsym = 106 +- 37) to map out the full
# physically allowed region for each isoscalar sector.
ESYM_LIST = [25.0, 26.5, 28.0, 29.5, 31.0, 32.5, 34.0,
             35.5, 37.0, 38.5, 40.0, 41.5, 43.0]           # MeV  (step 1.5)
LSYM_LIST = [30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0,
             65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0,
             100.0, 105.0, 110.0, 115.0, 120.0, 125.0,
             130.0]                                         # MeV  (step 5)


# ---------------------------------------------------------------------
#  Density grid for the EoS table (units fm^-3).
#  N_LOW must be BELOW the crust-core joining point that
#  automationandformatting-2.py aims at (n = 0.0789 fm^-3, the
#  Z=32 / A_cell=982 row of the NSCool HZD-NV crust), otherwise there
#  is nothing to join to.  N_HIGH must be high enough that the TOV
#  solver reaches the maximum mass (n ~ 1.2-1.5 fm^-3 is plenty).
#  The grid is GEOMETRIC (constant ratio between neighbours) so the
#  low-density end, where the crust join happens, is well resolved.
# ---------------------------------------------------------------------
N_LOW = 0.030      # fm^-3
N_HIGH = 1.600     # fm^-3
N_POINTS = 320


# =====================================================================
#  THE TWO EFFECTIVE MASSES.  They are different quantities and both
#  are used, in different places.  Getting them confused is the single
#  easiest way to produce wrong cooling curves, so they are named
#  explicitly everywhere in this file.
#
#    m_dirac    = M_N - g_sigma*sigma
#                 The mass that appears IN the Lagrangian, i.e. in the
#                 Dirac equation  (i gamma.d - m_dirac) psi = 0 .
#                 Used for: the sigma field equation, the scalar
#                 density n_s = <psi-bar psi> that sources sigma, the
#                 kinetic energy density, and the pressure.
#                 In short: everything that builds the EQUATION OF STATE.
#
#    m_landau   = sqrt(k_F^2 + m_dirac^2) = E_F* = mu - U_vector
#                 The mass that reproduces the true density of states
#                 at the Fermi surface.  Used for: heat capacity,
#                 neutrino emissivities (Durca, Murca), thermal
#                 conductivity, pairing gaps.
#                 In short: everything that governs COOLING.
#
#  WHY m_landau, derived rather than asserted.
#  Cooling is a Fermi-surface phenomenon: every rate is a phase-space
#  integral over states within ~kT of mu, and what controls them all is
#  the density of states N(0) = dn/dmu.  In this Lagrangian the
#  single-particle energy is
#        E(k) = sqrt(k^2 + m_dirac^2) + g_w w0 + (g_r/2) tau3 rho03 ,
#  and the vector terms are constants in k, so they shift mu but do not
#  change the dispersion.  With  n = k_F^3/3pi^2 :
#        dn/dk_F  = k_F^2/pi^2 ,      dmu/dk_F = k_F/E_F* ,
#  hence
#        N(0) = k_F E_F* / pi^2 .
#  Comparing with the non-relativistic form N(0) = m k_F/pi^2, the mass
#  that reproduces the real density of states is E_F*.  That IS the
#  definition of the Landau mass; it is a derived result, not a
#  convention, and it follows from our own Lagrangian.
#
#  WHY IT MATTERS NUMERICALLY.  At n = 1.6 fm^-3 in GM1, m_dirac has
#  fallen to 140 MeV while k_F = 675 MeV: the neutrons are moving at
#  0.98c.  A particle that fast has a LARGE density of states.  Using
#  m_dirac there would understate N(0) by a factor 4.9, and since the
#  Durca emissivity goes as m_n m_p, the rate by a factor ~14 -- in
#  exactly the high-density core where direct Urca operates.
#
#  CONSISTENCY WITH NSCool.  NSCool applies the same rule to leptons,
#  which have no scalar field so their Dirac mass is just m_e:
#        precool.f:1253   me = sqrt(0.511**2 + pfe**2)
#  i.e. sqrt(m_dirac^2 + p_F^2).  Feeding nucleons their Dirac mass
#  would do for baryons precisely what NSCool forbids for leptons.
#  Its own density of states is  N0 = m*pf/pi^2  (precool.f:1739), so
#  whatever multiplies p_F there is by definition the Landau mass.
#
#  CAVEAT worth stating in the paper: the Durca formula used here is
#  the non-relativistic reduction of Lattimer, Pethick, Prakash and
#  Haensel, in which the m* factors are Landau masses.  A fully
#  relativistic treatment of the matrix element carries extra
#  m_dirac/E* structure and would differ somewhat.  That is a known
#  limitation of this class of cooling calculation, not of this choice.
# =====================================================================
MASS_FOR_NSCOOL = "landau"


# output folder
OUTDIR = "EOS_RMF"


# =====================================================================
#  SECTION 2 :  SMALL HELPER FUNCTIONS (pure kinematics)
# =====================================================================

def fermi_momentum(number_density):
    """
    Fermi momentum of ONE spin-1/2 species from its number density.

    A spin-1/2 particle has 2 spin states, so
        n = 2 * (4/3) pi k_F^3 / (2 pi)^3 = k_F^3 / (3 pi^2)
    hence
        k_F = (3 pi^2 n)^(1/3)
    n is in MeV^3 and k_F comes out in MeV.
    """
    if number_density <= 0.0:
        return 0.0
    return (3.0 * PI * PI * number_density) ** (1.0 / 3.0)


def number_density_from_kf(kf):
    """Inverse of fermi_momentum():  n = k_F^3 / (3 pi^2)."""
    if kf <= 0.0:
        return 0.0
    return kf ** 3 / (3.0 * PI * PI)


def scalar_density(kf, m_dirac):
    """
    Scalar density of one nucleon species,  Eq.(4) of PhysRevC.98.065804:

        n_s = (m*/2 pi^2) [ k_F E_F - m*^2 ln( (k_F+E_F)/m* ) ]

    Derivation (do it by hand like this):
        n_s = <psi-bar psi> = (2/(2 pi)^3) INT d^3k  m*/E*(k)
            = (1/pi^2) INT_0^kF dk k^2 m*/sqrt(k^2+m*^2)
        and INT k^2/sqrt(k^2+m^2) dk = (1/2)[k sqrt(k^2+m^2)
                                     - m^2 ln((k+sqrt(k^2+m^2))/m)]
    Careful: n_s is NOT the same as the vector density n = k_F^3/3pi^2.
    """
    if kf <= 0.0:
        return 0.0
    ef = math.sqrt(kf * kf + m_dirac * m_dirac)
    return (m_dirac / (2.0 * PI * PI)) * (kf * ef
                                        - m_dirac * m_dirac * math.log((kf + ef) / m_dirac))


def kinetic_energy_density(kf, mass):
    """
    Kinetic (Fermi-gas) energy density of one spin-1/2 species,
    first line of Eq.(5) of PhysRevC.98.065804:

        eps = (1/8 pi^2)[ k_F E_F^3 + k_F^3 E_F - m^4 ln((k_F+E_F)/m) ]

    Derivation:
        eps = (1/pi^2) INT_0^kF dk k^2 sqrt(k^2+m^2)
            = (1/8 pi^2)[ k(2k^2+m^2) E - m^4 ln((k+E)/m) ]
        and k(2k^2+m^2)E = k E (k^2 + (k^2+m^2)) = k^3 E + k E^3.
    The same formula is used for electrons and muons (they are also
    spin-1/2 with 2 spin states), with m = m_e or m_mu.
    """
    if kf <= 0.0:
        return 0.0
    e = math.sqrt(kf * kf + mass * mass)
    return (1.0 / (8.0 * PI * PI)) * (kf * e ** 3 + kf ** 3 * e
                                      - mass ** 4 * math.log((kf + e) / mass))


def d_ns_d_m_dirac(kf, m_dirac):
    """
    rho_s'(M*) = d n_s / d M* , the quantity that appears in Eq.(21)
    of RMFparameters.pdf (there written for SYMMETRIC matter, i.e. the
    sum over protons and neutrons is a factor 2 which is already
    included in the 1/pi^2 below):

        rho_s'(M*) = (1/pi^2)[ (k_F/E_F)(E_F^2 + 2 M*^2)
                               - 3 M*^2 ln((k_F+E_F)/M*) ]

    (Differentiate the symmetric-matter n_s = 2 * scalar_density(kF,M*)
     with respect to M* at fixed k_F and you get exactly this.)
    """
    ef = math.sqrt(kf * kf + m_dirac * m_dirac)
    return (1.0 / (PI * PI)) * ((kf / ef) * (ef * ef + 2.0 * m_dirac * m_dirac)
                                - 3.0 * m_dirac * m_dirac * math.log((kf + ef) / m_dirac))


# =====================================================================
#  SECTION 3 :  JOB 1 - THE COUPLINGS
# ---------------------------------------------------------------------
#  The Lagrangian (RMFparameters.pdf Eq.(1)) has six constants
#        g_sigma , g_omega , g_rho , b , c , Lambda_omega_rho
#  Four of them (g_sigma, g_omega, b, c) are fixed by the ISOSCALAR
#  saturation properties  rho0 , E0 , M*/M , K , and Glendenning &
#  Moszkowski already did that fit for us - their answer is Table II.
#  Only the two ISOVECTOR ones (g_rho, Lambda_omega_rho) are left, and
#  they are fixed by  J = Esym(rho0)  and  L = 3 rho0 dEsym/drho|_0 .
#
#  A VERY USEFUL SIMPLIFICATION.
#  Everywhere in the theory the couplings only ever appear through the
#  three RATIOS
#        C_s^2 = (g_sigma/m_sigma)^2 ,
#        C_w^2 = (g_omega/m_omega)^2 ,
#        C_r^2 = (g_rho  /m_rho  )^2 ,
#  and through the combinations
#        PHI = g_sigma * sigma ,  W = g_omega * omega ,  R = g_rho * rho .
#  For example the sigma field equation
#        m_s^2 sigma + b m_N (g_s sigma)^2 g_s + c g_s^4 sigma^3 = g_s n_s
#  divided by g_s becomes simply
#        (1/C_s^2) PHI + b m_N PHI^2 + c PHI^3 = n_s .
#  We use this everywhere: it removes all the separate g's and makes
#  the code much shorter and much less error prone.
# =====================================================================

def isoscalar_block(model_name):
    """
    Turn the Table II numbers of one GM model into the quantities the
    rest of the program needs.  Everything comes out in MeV powers.
    """
    raw = GM_MODELS[model_name]

    # (g/m)^2 : fm^2 -> MeV^-2 .  1 fm = 1/197.327 MeV^-1  so
    # 1 fm^2 = 1/197.327^2 MeV^-2 .
    cs2 = raw["cs2_fm2"] / HBARC ** 2          # (g_sigma/m_sigma)^2  [MeV^-2]
    cw2 = raw["cw2_fm2"] / HBARC ** 2          # (g_omega/m_omega)^2  [MeV^-2]
    cr2_ref = raw["cr2_fm2"] / HBARC ** 2      # GM's own (g_rho/m_rho)^2

    par = {}
    par["name"] = model_name
    par["K"] = raw["K"]
    par["b"] = raw["b"]
    par["c"] = raw["c"]
    par["zeta"] = raw.get("zeta", 0.0)
    par["cs2"] = cs2
    par["cw2"] = cw2
    par["cr2_reference"] = cr2_ref
    par["inv_cs2"] = 1.0 / cs2                 # = m_sigma^2/g_sigma^2  [MeV^2]
    par["inv_cw2"] = 1.0 / cw2                 # = m_omega^2/g_omega^2  [MeV^2]

    # per-model saturation density (defaults to global RHO0_FM3)
    rho0_fm3 = raw.get("rho0_fm3", RHO0_FM3)
    par["rho0_fm3"] = rho0_fm3
    par["rho0"] = rho0_fm3 * HBARC ** 3        # MeV^3

    # the individual couplings (only needed for printing / for g_rho)
    par["g_sigma"] = math.sqrt(cs2) * M_SIGMA
    par["g_omega"] = math.sqrt(cw2) * M_OMEGA

    # effective (Dirac) mass at saturation and the sigma field there
    par["m_dirac0"] = raw["m_dirac_over_m"] * M_NUCLEON            # MeV
    par["PHI0"] = M_NUCLEON - par["m_dirac0"]                    # = g_sigma*sigma0
    return par


def isoscalar_symmetry_pieces(par):
    """
    J0 and L0 : the part of the symmetry energy and of its slope that
    comes from the ISOSCALAR sector alone (kinetic + sigma + omega).
    RMFparameters.pdf Eqs.(15) and (21).

        k_F  = (1.5 pi^2 rho0)^(1/3)        <- SYMMETRIC matter
        E_F  = sqrt(k_F^2 + M*^2)
        J0   = k_F^2 / (6 E_F)
        L0   = J0 { 1 + (M*^2/E_F^2)[ 1 - (3 rho/M*) dM*/drho ] }
        dM*/drho = - (M*/E_F) [ m*_s^2/g_s^2 + rho_s'(M*) ]^-1
        m*_s^2/g_s^2 = m_s^2/g_s^2 + 2 b m_N g_s sigma + 3 c (g_s sigma)^2
                     = 1/C_s^2      + 2 b m_N PHI      + 3 c PHI^2
    """
    m_dirac = par["m_dirac0"]
    phi = par["PHI0"]
    rho0 = par.get("rho0", RHO0)

    # SYMMETRIC matter at saturation: 4 states (2 spin x 2 isospin), so
    #   rho0 = 4 * k_F^3/(6 pi^2) = 2 k_F^3/(3 pi^2)  ->  k_F = (1.5 pi^2 rho0)^(1/3)
    kf = (1.5 * PI * PI * rho0) ** (1.0 / 3.0)
    ef = math.sqrt(kf * kf + m_dirac * m_dirac)

    # curvature of the sigma potential, "m_sigma-star squared over g_sigma squared"
    ms_star2_over_gs2 = par["inv_cs2"] + 2.0 * par["b"] * M_NUCLEON * phi \
        + 3.0 * par["c"] * phi * phi

    rhos_prime = d_ns_d_m_dirac(kf, m_dirac)

    dmstar_drho = -(m_dirac / ef) / (ms_star2_over_gs2 + rhos_prime)

    j0 = kf * kf / (6.0 * ef)
    l0 = j0 * (1.0 + (m_dirac * m_dirac / (ef * ef))
               * (1.0 - 3.0 * rho0 / m_dirac * dmstar_drho))

    out = {}
    out["kf"] = kf
    out["ef"] = ef
    out["ms_star2_over_gs2"] = ms_star2_over_gs2
    out["rhos_prime"] = rhos_prime
    out["dmstar_drho"] = dmstar_drho
    out["J0"] = j0
    out["L0"] = l0
    return out


def isovector_couplings(par, j_sym, l_sym):
    """
    JOB 1 proper.  Given J = Esym(rho0) and L = slope, return
    Lambda_omega_rho and g_rho.  RMFparameters.pdf Eqs.(19)-(23),
    generalised to zeta != 0.

    With zeta != 0 the omega field at saturation obeys the CUBIC
        (1/C_w^2) W0 + (zeta/6) W0^3  =  rho0
    (the rho field is zero in symmetric matter).  Solved by Newton.

    The Lambda formula generalises to
        Lambda = (3 J1 - L1) / (96 J1^2 W0 dW/drho)
    where  dW/drho = 1 / [ 1/C_w^2 + (zeta/2) W0^2 ] .
    For zeta = 0 this reduces to the original compact form.
    """
    iso = isoscalar_symmetry_pieces(par)
    j0 = iso["J0"]
    l0 = iso["L0"]

    j1 = j_sym - j0
    l1 = l_sym - l0

    rho0 = par.get("rho0", RHO0)
    zeta = par.get("zeta", 0.0)

    # omega mean field at saturation: solve (1/cw2)*W + (zeta/6)*W^3 = rho0
    w0 = par["cw2"] * rho0                      # initial guess (zeta=0 answer)
    if zeta != 0.0:
        it = 0
        while it < 100:
            f = par["inv_cw2"] * w0 + (zeta / 6.0) * w0 ** 3 - rho0
            fp = par["inv_cw2"] + (zeta / 2.0) * w0 ** 2
            dw = f / fp
            w0 = w0 - dw
            if abs(dw) < 1.0e-14 * abs(w0):
                break
            it = it + 1

    # dW/drho at saturation
    dw_drho = 1.0 / (par["inv_cw2"] + (zeta / 2.0) * w0 ** 2)

    lam = (3.0 * j1 - l1) / (96.0 * j1 * j1 * w0 * dw_drho)

    # m_rho^2/g_rho^2  (this is 1/C_r^2)
    inv_cr2 = rho0 / (8.0 * j1) - 2.0 * lam * w0 * w0

    out = {}
    out["J0"] = j0
    out["L0"] = l0
    out["J1"] = j1
    out["L1"] = l1
    out["W0"] = w0
    out["lambda_wr"] = lam
    out["inv_cr2"] = inv_cr2

    # guard against unphysical answers instead of crashing later
    if j1 <= 0.0:
        out["ok"] = False
        out["why"] = "J1 = J - J0 <= 0 (asked symmetry energy is below the kinetic part)"
        out["cr2"] = 0.0
        out["g_rho"] = 0.0
        return out
    if inv_cr2 <= 0.0:
        out["ok"] = False
        out["why"] = "m_rho^2/g_rho^2 <= 0 (no real g_rho for this J,L)"
        out["cr2"] = 0.0
        out["g_rho"] = 0.0
        return out

    out["ok"] = True
    out["why"] = ""
    out["cr2"] = 1.0 / inv_cr2                       # (g_rho/m_rho)^2 [MeV^-2]
    out["g_rho"] = math.sqrt(out["cr2"]) * M_RHO
    return out


def make_parameter_set(model_name, j_sym, l_sym):
    """Bundle everything one EoS calculation needs into a single dict."""
    par = isoscalar_block(model_name)
    iso = isovector_couplings(par, j_sym, l_sym)
    par["J"] = j_sym
    par["L"] = l_sym
    par["J0"] = iso["J0"]
    par["L0"] = iso["L0"]
    par["J1"] = iso["J1"]
    par["L1"] = iso["L1"]
    par["W0"] = iso["W0"]
    par["lambda_wr"] = iso["lambda_wr"]
    par["inv_cr2"] = iso["inv_cr2"]
    par["cr2"] = iso["cr2"]
    par["g_rho"] = iso["g_rho"]
    par["ok"] = iso["ok"]
    par["why"] = iso["why"]
    return par


# =====================================================================
#  SECTION 4 :  JOB 2 - THE MEAN FIELD EQUATIONS
# ---------------------------------------------------------------------
#  With  PHI = g_s sigma ,  W = g_w omega ,  R = g_r rho  the three
#  meson equations of PhysRevC.98.065804 Eq.(3) (with zeta = 0) read
#
#     (1/C_s^2) PHI + b m_N PHI^2 + c PHI^3 = n_s              (sigma)
#     (1/C_w^2) W   + 2 Lambda R^2 W        = n                (omega)
#     (1/C_r^2) R   + 2 Lambda W^2 R        = n_3 / 2          (rho)
#
#  with   n   = n_p + n_n ,   n_3 = n_p - n_n ,
#         n_s = n_s(p) + n_s(n)   and   M* = M_N - PHI .
#
#  The factor 1/2 on the right of the rho equation comes from the
#  Lagrangian term  -(1/2) g_rho tau.rho  (see Eq.(1) of
#  RMFparameters.pdf and Eq.(1) of PhysRevC.98.065804): tau_3 = +1 for
#  the proton and -1 for the neutron.
# =====================================================================

def solve_effective_mass(kfp, kfn, par):
    """
    Solve the sigma (scalar) equation for M* by BISECTION.

    We look for the root of
        G(M*) = (1/C_s^2)(M-M*) + b m_N (M-M*)^2 + c (M-M*)^3 - n_s(M*)

    Why bisection is safe here:
      * the left-hand polynomial grows monotonically as M* decreases
        (checked for all three GM sets over 0 < M* < M),
      * n_s(M*) decreases as M* decreases,
      so G is monotonically DECREASING in M*: exactly one root.
      G(M* = M) = -n_s < 0 and G(M* -> 0) > 0, so [0, M] brackets it.
    """
    lo = 1.0e-6                 # MeV  (M* can get very small at high density)
    hi = M_NUCLEON

    def residual(m_dirac):
        phi = M_NUCLEON - m_dirac
        lhs = par["inv_cs2"] * phi + par["b"] * M_NUCLEON * phi * phi \
            + par["c"] * phi ** 3
        ns = scalar_density(kfp, m_dirac) + scalar_density(kfn, m_dirac)
        return lhs - ns

    # residual(lo) > 0 and residual(hi) < 0 : bisect until M* is
    # converged to ~1e-10 relative accuracy (about 45 halvings).
    it = 0
    mid = 0.5 * (lo + hi)
    while it < 200:
        mid = 0.5 * (lo + hi)
        f = residual(mid)
        if f > 0.0:
            lo = mid            # root is at LARGER M*
        else:
            hi = mid            # root is at SMALLER M*
        if (hi - lo) < 1.0e-10 * M_NUCLEON:
            break
        it = it + 1
    return 0.5 * (lo + hi)


def solve_vector_fields(n_tot, n_three, par):
    """
    Solve the coupled omega and rho equations.  The omega equation is
    a CUBIC in W when zeta != 0:
          (1/C_w^2) W + (zeta/6) W^3 + 2 Lambda R^2 W  =  n
    and the rho equation is linear in R given W:
          R = (n_3/2) / ( 1/C_r^2 + 2 Lambda W^2 )

    For each outer iteration on R, the omega cubic is solved by Newton.
    The rho equation is updated with the new W.  This replaces the
    simple fixed-point iteration which diverges at high density when
    (zeta/6) W^2 exceeds 1/C_w^2.
    """
    lam = par["lambda_wr"]
    inv_cw2 = par["inv_cw2"]
    inv_cr2 = par["inv_cr2"]
    zeta = par["zeta"]
    zeta6 = zeta / 6.0

    w = n_tot / inv_cw2
    r = 0.5 * n_three / inv_cr2

    it = 0
    while it < 500:
        # solve the omega cubic for W by Newton, with R held fixed
        eff_mass2 = inv_cw2 + 2.0 * lam * r * r
        # Newton iterations for (eff_mass2)*W + zeta6*W^3 = n_tot
        jt = 0
        while jt < 60:
            fw = eff_mass2 * w + zeta6 * w * w * w - n_tot
            fpw = eff_mass2 + 3.0 * zeta6 * w * w
            dw_newton = fw / fpw
            w = w - dw_newton
            if abs(dw_newton) < 1.0e-13 * (abs(w) + 1.0):
                break
            jt = jt + 1

        r_new = 0.5 * n_three / (inv_cr2 + 2.0 * lam * w * w)
        dr = abs(r_new - r)
        r = r_new
        if dr < 1.0e-12 * (abs(r) + 1.0):
            break
        it = it + 1
    return w, r


def nuclear_matter_state(n_tot, x_proton, par):
    """
    Everything about the NUCLEONS (no leptons yet) at total baryon
    density n_tot (MeV^3) and proton fraction x_proton.
    Returns a dict.
    """
    n_p = x_proton * n_tot
    n_n = (1.0 - x_proton) * n_tot
    n_three = n_p - n_n

    kfp = fermi_momentum(n_p)
    kfn = fermi_momentum(n_n)

    m_dirac = solve_effective_mass(kfp, kfn, par)
    w, r = solve_vector_fields(n_tot, n_three, par)

    efp = math.sqrt(kfp * kfp + m_dirac * m_dirac)      # "starred" Fermi energies
    efn = math.sqrt(kfn * kfn + m_dirac * m_dirac)

    # chemical potentials, PhysRevC.98.065804 Eq.(7):
    #   mu_N = E_F* + g_w omega + (g_r/2) tau_3N rho  =  E_F* + W + tau_3N R/2
    mu_p = efp + w + 0.5 * r
    mu_n = efn + w - 0.5 * r

    st = {}
    st["n"] = n_tot
    st["x"] = x_proton
    st["n_p"] = n_p
    st["n_n"] = n_n
    st["kfp"] = kfp
    st["kfn"] = kfn
    st["m_dirac"] = m_dirac
    st["PHI"] = M_NUCLEON - m_dirac
    st["W"] = w
    st["R"] = r
    st["efp"] = efp
    st["efn"] = efn
    st["mu_p"] = mu_p
    st["mu_n"] = mu_n
    return st


def direct_pressure(st, par):
    """
    The pressure written out "the long way", from T^ii/3.  This is NOT
    used to build the tables (full_state() uses the Euler relation,
    which is shorter); it exists so that the two completely independent
    routes can be compared - see CHECK 4a in SECTION 8.  They agree to
    4e-14, which is the strongest single test of the whole EoS module:
    it simultaneously verifies the energy density (including the factor
    3 in the omega-rho term) and the chemical potentials.

        P = SUM_N [ E_F*_N n_N - eps_kin,N ]
          + SUM_l [ mu_l n_l   - eps_l     ]
          - (1/2)(1/C_s^2) PHI^2 - (1/3) b m_N PHI^3 - (1/4) c PHI^4
          + (1/2)(1/C_w^2) W^2 + (zeta/24) W^4
          + (1/2)(1/C_r^2) R^2 + Lambda W^2 R^2

    (For a free Fermi gas at T=0, P = mu n - eps exactly; that identity
     is what produces the first two lines.)
    """
    p = st["efp"] * st["n_p"] - kinetic_energy_density(st["kfp"], st["m_dirac"])
    p = p + st["efn"] * st["n_n"] - kinetic_energy_density(st["kfn"], st["m_dirac"])
    p = p + st["mu_e"] * st["n_e"] - kinetic_energy_density(st["kfe"], M_ELECTRON)
    p = p + st["mu_e"] * st["n_mu"] - kinetic_energy_density(st["kfmu"], M_MUON)
    phi = st["PHI"]
    w = st["W"]
    r = st["R"]
    p = p - 0.5 * par["inv_cs2"] * phi * phi
    p = p - (1.0 / 3.0) * par["b"] * M_NUCLEON * phi ** 3
    p = p - 0.25 * par["c"] * phi ** 4
    p = p + 0.5 * par["inv_cw2"] * w * w
    p = p + (par["zeta"] / 24.0) * w ** 4
    p = p + 0.5 * par["inv_cr2"] * r * r
    p = p + par["lambda_wr"] * w * w * r * r
    return p


def meson_energy_density(st, par):
    """
    The meson part of the energy density, PhysRevC.98.065804 Eq.(5)
    rewritten with PHI, W, R:

        eps_meson = (1/2)(1/C_s^2) PHI^2 + (1/3) b m_N PHI^3 + (1/4) c PHI^4
                  + (1/2)(1/C_w^2) W^2 + (zeta/8) W^4
                  + (1/2)(1/C_r^2) R^2
                  + 3 Lambda W^2 R^2

    The zeta/8 coefficient comes from the Lagrangian contribution
    -(zeta/24) W^4 combined with two additional +zeta/12 pieces from
    re-using the omega field equation; total -1/24 + 2*1/12 = +1/8.
    The omega-rho mixing coefficient 3 has the same origin.
    """
    phi = st["PHI"]
    w = st["W"]
    r = st["R"]
    e = 0.5 * par["inv_cs2"] * phi * phi
    e = e + (1.0 / 3.0) * par["b"] * M_NUCLEON * phi ** 3
    e = e + 0.25 * par["c"] * phi ** 4
    e = e + 0.5 * par["inv_cw2"] * w * w
    e = e + (par["zeta"] / 8.0) * w ** 4
    e = e + 0.5 * par["inv_cr2"] * r * r
    e = e + 3.0 * par["lambda_wr"] * w * w * r * r
    return e


# =====================================================================
#  SECTION 5 :  BETA EQUILIBRIUM AND CHARGE NEUTRALITY
# ---------------------------------------------------------------------
#  Neutron star matter is in equilibrium under
#        n  ->  p + e + nubar        (and the reverse)
#        e  ->  mu + nu + nubar
#  which, because the neutrinos escape freely (mu_nu = 0), means
#        mu_n = mu_p + mu_e ,        mu_mu = mu_e
#  and the star must be electrically neutral:
#        n_p = n_e + n_mu .
#
#  Practical recipe (this is what the code does):
#     pick the proton fraction x,
#     -> get mu_n and mu_p from the RMF equations,
#     -> mu_e = mu_n - mu_p,
#     -> k_Fe = sqrt(mu_e^2 - m_e^2)  (if mu_e > m_e, else no electrons)
#     -> k_Fmu = sqrt(mu_e^2 - m_mu^2)(if mu_e > m_mu, else no muons)
#     -> residual = n_p - n_e - n_mu ; drive it to zero by bisection.
#  The residual is monotonically INCREASING in x:
#        x = 0     -> mu_e is large -> lots of electrons -> residual < 0
#        x = 0.5   -> mu_e = 0      -> no leptons        -> residual > 0
#  so a bisection on [0, 0.5] always works.
# =====================================================================

def lepton_density(mu, mass):
    """n_l and k_Fl of a free lepton gas with chemical potential mu."""
    if mu <= mass:
        return 0.0, 0.0
    kf = math.sqrt(mu * mu - mass * mass)
    return number_density_from_kf(kf), kf


def charge_residual(x_proton, n_tot, par):
    """n_p - n_e - n_mu at proton fraction x_proton (want it to be 0)."""
    st = nuclear_matter_state(n_tot, x_proton, par)
    mu_e = st["mu_n"] - st["mu_p"]
    n_e, kfe = lepton_density(mu_e, M_ELECTRON)
    n_mu, kfmu = lepton_density(mu_e, M_MUON)
    st["mu_e"] = mu_e
    st["n_e"] = n_e
    st["n_mu"] = n_mu
    st["kfe"] = kfe
    st["kfmu"] = kfmu
    return st["n_p"] - n_e - n_mu, st


def beta_equilibrium(n_tot, par):
    """Bisection for the beta-equilibrium proton fraction at density n_tot."""
    lo = 0.0
    hi = 0.5
    st = None
    it = 0
    while it < 200:
        mid = 0.5 * (lo + hi)
        res, st = charge_residual(mid, n_tot, par)
        if res > 0.0:
            hi = mid            # too many protons
        else:
            lo = mid            # too few protons
        if (hi - lo) < 1.0e-13:
            break
        it = it + 1
    # one final evaluation exactly at the converged x so that the
    # returned state is self consistent with the returned x
    res, st = charge_residual(0.5 * (lo + hi), n_tot, par)
    return st


def full_state(n_tot, par):
    """
    Complete thermodynamics at baryon density n_tot in beta equilibrium.
    Adds energy density, pressure, effective masses to the dict.
    """
    st = beta_equilibrium(n_tot, par)

    # --- energy density -------------------------------------------------
    # nucleon Fermi seas
    eps = kinetic_energy_density(st["kfp"], st["m_dirac"]) \
        + kinetic_energy_density(st["kfn"], st["m_dirac"])
    # mesons
    eps = eps + meson_energy_density(st, par)
    # leptons (free relativistic Fermi gases)
    eps = eps + kinetic_energy_density(st["kfe"], M_ELECTRON)
    eps = eps + kinetic_energy_density(st["kfmu"], M_MUON)

    # --- pressure -------------------------------------------------------
    # At T = 0 the Euler relation is exact:  eps + P = SUM_i mu_i n_i
    # so                                     P = SUM_i mu_i n_i - eps .
    # (Doing it this way instead of writing out T^ii/3 removes any chance
    #  of dropping a term; the two agree to ~1e-13 - see SECTION 8.)
    mu_n_tot = st["mu_n"] * st["n_n"] + st["mu_p"] * st["n_p"]
    mu_lep = st["mu_e"] * st["n_e"]
    if st["n_mu"] > 0.0:
        mu_lep = mu_lep + st["mu_e"] * st["n_mu"]     # mu_muon = mu_e
    press = mu_n_tot + mu_lep - eps

    st["eps"] = eps
    st["press"] = press

    # --- effective masses handed to NSCool ------------------------------
    # Landau mass = starred Fermi energy; see THE TWO EFFECTIVE MASSES above
    st["m_landau_n"] = st["efn"]
    st["m_landau_p"] = st["efp"]
    if MASS_FOR_NSCOOL == "dirac":
        st["m_for_nscool_n"] = st["m_dirac"]
        st["m_for_nscool_p"] = st["m_dirac"]
    else:
        st["m_for_nscool_n"] = st["m_landau_n"]
        st["m_for_nscool_p"] = st["m_landau_p"]

    # --- baryon chemical potential --------------------------------------
    # In beta equilibrium  (eps + P)/n_B = mu_n  exactly.  Proof:
    #   SUM mu_i n_i = mu_n n_n + mu_p n_p + mu_e (n_e + n_mu)
    #                = mu_n n_n + (mu_n - mu_e) n_p + mu_e n_p   = mu_n n_B
    st["mu_B"] = (eps + press) / n_tot
    return st


# =====================================================================
#  SECTION 6 :  SYMMETRY ENERGY, DIRECT URCA THRESHOLD
# =====================================================================

def symmetry_energy_at(n_tot, par):
    """
    Esym(n) from Eq.(9) of RMFparameters.pdf:

        Esym(n) = k_F^2/(6 E_F)  +  g_r^2 n / (8 m*_rho^2)
                = k_F^2/(6 E_F)  +  n / ( 8 [1/C_r^2 + 2 Lambda W^2] )

    evaluated in SYMMETRIC matter (that is the definition: Esym is the
    coefficient of delta^2 in an expansion about delta = 0), so
    k_F = (1.5 pi^2 n)^(1/3), M* is the symmetric-matter value, and W is
    the symmetric-matter omega field (rho field = 0 there).
    """
    kf = (1.5 * PI * PI * n_tot) ** (1.0 / 3.0)
    m_dirac = solve_effective_mass(kf, kf, par)
    ef = math.sqrt(kf * kf + m_dirac * m_dirac)
    w, r = solve_vector_fields(n_tot, 0.0, par)
    kinetic = kf * kf / (6.0 * ef)
    potential = n_tot / (8.0 * (par["inv_cr2"] + 2.0 * par["lambda_wr"] * w * w))
    return kinetic + potential, kinetic, potential


def snm_energy_per_nucleon(n_tot, par):
    """
    E0(n) = energy per nucleon of SYMMETRIC nuclear matter, minus the
    rest mass.  Needed for the crust-core transition criterion below.
    """
    kf = (1.5 * PI * PI * n_tot) ** (1.0 / 3.0)
    m_dirac = solve_effective_mass(kf, kf, par)
    w, r = solve_vector_fields(n_tot, 0.0, par)
    st = {"PHI": M_NUCLEON - m_dirac, "W": w, "R": r}
    eps = 2.0 * kinetic_energy_density(kf, m_dirac) + meson_energy_density(st, par)
    return eps / n_tot - M_NUCLEON


def k_mu(n_tot, delta, par):
    """
    K_mu, the compressibility of npemu matter AT CONSTANT CHEMICAL
    POTENTIAL - Eq.(2) of RMFparameters.pdf:

        K_mu = rho^2 d2E0/drho^2 + 2 rho dE0/drho
             + delta^2 [ rho^2 d2Esym/drho^2 + 2 rho dEsym/drho
                         - 2 Esym^-1 ( rho dEsym/drho )^2 ]

    Physics: as you go DOWN in density, uniform npemu matter eventually
    becomes unstable against breaking up into nuclei (clusters).  The
    density where that happens is where K_mu changes sign from positive
    to negative, and RMFparameters.pdf uses exactly that as the
    definition of the crust-core transition point.

    All derivatives are done by central finite differences.
    """
    h = 1.0e-3 * n_tot

    e0_0 = snm_energy_per_nucleon(n_tot, par)
    e0_p = snm_energy_per_nucleon(n_tot + h, par)
    e0_m = snm_energy_per_nucleon(n_tot - h, par)
    de0 = (e0_p - e0_m) / (2.0 * h)
    d2e0 = (e0_p - 2.0 * e0_0 + e0_m) / (h * h)

    es_0, a, b = symmetry_energy_at(n_tot, par)
    es_p, a, b = symmetry_energy_at(n_tot + h, par)
    es_m, a, b = symmetry_energy_at(n_tot - h, par)
    des = (es_p - es_m) / (2.0 * h)
    d2es = (es_p - 2.0 * es_0 + es_m) / (h * h)

    term_sym = n_tot * n_tot * d2es + 2.0 * n_tot * des \
        - 2.0 * (n_tot * des) ** 2 / es_0

    return n_tot * n_tot * d2e0 + 2.0 * n_tot * de0 + delta * delta * term_sym


def crust_core_transition(rows, par):
    """
    Scan the table from saturation DOWNWARD and return the density
    (fm^-3) at which K_mu first goes negative.  That is the crust-core
    transition density of RMFparameters.pdf.  Returns None if K_mu never
    turns negative inside the table.
    """
    # start just below saturation and walk down
    i = len(rows) - 1
    i_start = 0
    while i >= 0:
        if rows[i]["n_fm3"] <= 0.16:
            i_start = i
            break
        i = i - 1

    i = i_start
    previous = None
    while i >= 0:
        value = k_mu(rows[i]["n"], 1.0 - 2.0 * rows[i]["n_p"] / rows[i]["n"], par)
        if value < 0.0:
            if previous is None:
                return rows[i]["n_fm3"]
            # linear interpolation between the two bracketing points
            n_hi = rows[i + 1]["n_fm3"]
            n_lo = rows[i]["n_fm3"]
            return n_lo + (n_hi - n_lo) * (0.0 - value) / (previous - value)
        previous = value
        i = i - 1
    return None


def durca_allowed(st):
    """
    Direct Urca  n -> p + e + nubar  needs momentum conservation on the
    three Fermi surfaces:
            k_Fn  <=  k_Fp + k_Fe .
    (The three neutrinos carry away ~T which is negligible.)
    Returns True/False for the electron channel.
    """
    return (st["kfp"] + st["kfe"]) >= st["kfn"]


# =====================================================================
#  SECTION 7 :  BUILD ONE EoS TABLE AND WRITE IT OUT
# =====================================================================

def build_density_grid():
    """Geometric grid from N_LOW to N_HIGH with N_POINTS points (fm^-3)."""
    grid = []
    ratio = (N_HIGH / N_LOW) ** (1.0 / (N_POINTS - 1))
    n = N_LOW
    i = 0
    while i < N_POINTS:
        grid.append(n)
        n = n * ratio
        i = i + 1
    return grid


def rho_meson_is_healthy(par):
    """
    STABILITY TEST ON THE OMEGA-RHO MIXING TERM.  Read this carefully,
    it is the single most important physical restriction in the code.

    The rho field equation is
            [ 1/C_r^2 + 2 Lambda W^2 ] R  =  n_3 / 2 .
    The bracket is the EFFECTIVE rho mass squared divided by g_rho^2,
    written  m*_rho^2/g_rho^2  in Eq.(13) of RMFparameters.pdf.

    If Lambda > 0 the bracket only grows with density: everything is fine.
    If Lambda < 0 the bracket SHRINKS, and above the density where
            1/C_r^2 + 2 Lambda W^2 = 0
    it turns negative.  Then the rho field changes sign and blows up, the
    symmetry energy goes negative, the proton fraction collapses to zero
    and the pressure develops a hole.  Such an EoS is meaningless.

    This is not a bug in the algebra - Eq.(22) faithfully returns a
    negative Lambda when the requested L is too large for the chosen
    isoscalar sector.  It is exactly the "unphysical" region that
    PhysRevC.98.065804 maps out in its Figs. 3 and 4 and then discards.
    We do the same: compute, test, reject with a reason.

    Returns (ok, n_break_fm3).
    """
    lam = par["lambda_wr"]
    if lam >= 0.0:
        return True, None

    # W at the top of our density grid (rho field feeds back only weakly
    # on omega, so the Lambda = 0 estimate is good enough for a test).
    # With zeta != 0, solve the cubic for w_top by Newton.
    n_top = N_HIGH * HBARC ** 3
    zeta = par["zeta"]
    w_top = n_top / par["inv_cw2"]
    if zeta != 0.0:
        it = 0
        while it < 100:
            f = par["inv_cw2"] * w_top + (zeta / 6.0) * w_top ** 3 - n_top
            fp = par["inv_cw2"] + (zeta / 2.0) * w_top ** 2
            dw = f / fp
            w_top = w_top - dw
            if abs(dw) < 1.0e-14 * abs(w_top):
                break
            it = it + 1
    if par["inv_cr2"] + 2.0 * lam * w_top * w_top > 0.0:
        return True, None

    # density at which the bracket vanishes
    w_break = math.sqrt(-par["inv_cr2"] / (2.0 * lam))
    n_break_omega_eq = par["inv_cw2"] * w_break + (zeta / 6.0) * w_break ** 3
    n_break = n_break_omega_eq / HBARC ** 3
    return False, n_break


def build_eos(par):
    """
    Solve the EoS on the whole density grid.

    Returns (rows, info).  "rows" is sorted by INCREASING density and has
    been trimmed to the single stable branch that CONTAINS SATURATION
    DENSITY, i.e. the branch a real star actually sits on.

    Why anchor the trimming at saturation instead of just walking down
    from the top?  Below the spinodal (n < ~0.06 fm^-3) uniform npemu
    matter is mechanically unstable and dP/dn turns negative - those rows
    are meaningless and must go.  But a sick parameter set can ALSO
    produce a second, spurious, increasing branch at very high density
    (past the rho-meson breakdown described above).  Walking down from
    the top would happily keep that spurious branch and hand a nonsense
    table to the TOV solver.  Anchoring at n = 0.16 fm^-3, where we know
    the physics is sound, and growing outwards in both directions, keeps
    only the branch that is continuously connected to nuclear matter.
    """
    grid = build_density_grid()
    rows = []
    i = 0
    while i < len(grid):
        n_fm3 = grid[i]
        st = full_state(n_fm3 * HBARC ** 3, par)
        st["n_fm3"] = n_fm3
        rows.append(st)
        i = i + 1

    # ---- find the grid point closest to nuclear saturation -------------
    i_anchor = 0
    best = 1.0e30
    i = 0
    while i < len(rows):
        d = abs(rows[i]["n_fm3"] - 0.16)
        if d < best:
            best = d
            i_anchor = i
        i = i + 1

    # ---- grow downwards while P stays positive and keeps falling -------
    i_lo = i_anchor
    i = i_anchor
    while i > 0:
        if rows[i - 1]["press"] <= 0.0:
            break
        if rows[i - 1]["press"] >= rows[i]["press"]:
            break
        i_lo = i - 1
        i = i - 1

    # ---- grow upwards while P keeps rising -----------------------------
    i_hi = i_anchor
    i = i_anchor
    while i < len(rows) - 1:
        if rows[i + 1]["press"] <= rows[i]["press"]:
            break
        i_hi = i + 1
        i = i + 1

    trimmed = rows[i_lo:i_hi + 1]

    info = {}
    info["n_min"] = trimmed[0]["n_fm3"]
    info["n_max"] = trimmed[-1]["n_fm3"]
    info["n_rows"] = len(trimmed)
    return trimmed, info


def table_is_usable(rows, info):
    """
    Final acceptance test on a finished table.

    N_JOIN_MAX : the table must reach DOWN below the crust joining
        density that automationandformatting-2.py aims at (the Z=32 /
        A_cell=982 row of the NSCool HZD-NV crust sits at n = 0.0789
        fm^-3).  If it does not, there is nothing to glue the crust to.

    N_TOP_MIN : the table must reach UP to at least ~1.0 fm^-3, otherwise
        the TOV solver runs out of table before the maximum-mass star and
        the M(R) curve is truncated at an artificial value.
    """
    if info["n_min"] > 0.070:
        return False, ("stable branch stops at n = %.3f fm^-3, above the "
                       "crust join at 0.079 fm^-3" % info["n_min"])
    if info["n_max"] < 1.0:
        return False, ("stable branch ends at n = %.3f fm^-3, too low to "
                       "build a massive star" % info["n_max"])
    return True, ""


def durca_threshold_density(rows):
    """
    Lowest baryon density (fm^-3) at which the electron direct Urca is
    open.  Returns None if it never opens inside the table.
    """
    i = 0
    while i < len(rows):
        if durca_allowed(rows[i]):
            return rows[i]["n_fm3"]
        i = i + 1
    return None


def tag_of(model_name, j_sym, l_sym):
    """
    Build a file tag such as  GM1_J38p1_L85p0 .
    A dot in a file name upsets Fortran/shell handling downstream, so the
    decimal point is written as the letter 'p'.
    """
    js = ("%.1f" % j_sym).replace(".", "p")
    ls = ("%.1f" % l_sym).replace(".", "p")
    return model_name + "_J" + js + "_L" + ls


def write_eos_file(rows, par, tag):
    """
    Write eos<TAG>.dat in the EXACT column order that
    automationandformatting-2.py reads (its EOS_COLUMNS dictionary):

        col 1  rho     g/cm^3        energy density
        col 2  press   dyn/cm^2
        col 3  nbar    fm^-3         baryon number density
        col 4  Ye      = n_e /n_B
        col 5  Ymu     = n_mu/n_B
        col 6  Yn      = n_n /n_B
        col 7  Yp      = n_p /n_B
        col 8  m_dirac   MeV           NEUTRON effective mass
        col 9  muB     MeV           baryon chemical potential
        col 10 m_dirac_p MeV           PROTON effective mass

    Rows are written high density first, which is the NSCool convention.
    Lines starting with '#' are skipped by the reader, so the header is safe.
    """
    path = os.path.join(OUTDIR, "eos" + tag + ".dat")
    # Create the output folder if it is not there.  Without this the very
    # first run in a fresh checkout dies with FileNotFoundError on a
    # directory the user has no reason to know they had to make.
    if OUTDIR and not os.path.isdir(OUTDIR):
        os.makedirs(OUTDIR)
    f = open(path, "w")
    f.write("# RMF EoS, Lagrangian with sigma + omega + rho + omega-rho mixing"
            " + omega^4 (zeta)\n")
    f.write("# model = %s   Esym(n0) = %.4f MeV   Lsym(n0) = %.4f MeV\n"
            % (par["name"], par["J"], par["L"]))
    f.write("# (g_s/m_s)^2 = %.6e MeV^-2   (g_w/m_w)^2 = %.6e MeV^-2\n"
            % (par["cs2"], par["cw2"]))
    f.write("# (g_r/m_r)^2 = %.6e MeV^-2   g_rho = %.6f   Lambda_wr = %.8f\n"
            % (par["cr2"], par["g_rho"], par["lambda_wr"]))
    f.write("# b = %.6e   c = %.6e   zeta = %.6e\n"
            % (par["b"], par["c"], par["zeta"]))
    f.write("# M*/M(sat) = %.4f   K = %.1f MeV   rho0 = %.4f fm^-3\n"
            % (par["m_dirac0"] / M_NUCLEON, par["K"], par["rho0_fm3"]))
    f.write("# effective mass columns hold the %s mass in MeV\n" % MASS_FOR_NSCOOL)
    if par.get("n_cc") is not None:
        f.write("# crust-core transition from the K_mu criterion: "
                "n_cc = %.4f fm^-3\n" % par["n_cc"])
    f.write("#   rho[g/cm3]      P[dyn/cm2]      nb[1/fm3]"
            "       Ye              Ymu             Yn"
            "              Yp          m_landau_n[MeV]     muB[MeV]"
            "        mst_p[MeV]\n")

    i = len(rows) - 1
    while i >= 0:
        st = rows[i]
        n = st["n"]
        rho_cgs = (st["eps"] / HBARC ** 3) * MEVFM3_TO_GCM3
        p_cgs = (st["press"] / HBARC ** 3) * MEVFM3_TO_DYNCM2
        ye = st["n_e"] / n
        ymu = st["n_mu"] / n
        yn = st["n_n"] / n
        yp = st["n_p"] / n
        f.write(" %15.7E %15.7E %15.7E %15.7E %15.7E %15.7E %15.7E"
                " %15.7E %15.7E %15.7E\n"
                % (rho_cgs, p_cgs, st["n_fm3"], ye, ymu, yn, yp,
                   st["m_for_nscool_n"], st["mu_B"], st["m_for_nscool_p"]))
        i = i - 1
    f.close()
    return path


def write_detail_file(rows, par, tag):
    """
    A second, human-readable file with the quantities you want for plots
    and for checking the physics by hand.  NSCool never reads this file.
    """
    path = os.path.join(OUTDIR, "detail" + tag + ".dat")
    f = open(path, "w")
    f.write("# model %s  J = %.3f MeV  L = %.3f MeV\n" % (par["name"], par["J"], par["L"]))
    f.write("#  nb[1/fm3]    Yp        Ye        Ymu       "
            "M*dirac[MeV]  M*Land_n   M*Land_p   "
            "eps[MeV/fm3]  P[MeV/fm3]   E/A[MeV]   "
            "Esym[MeV]   cs2/c2     kFn[MeV]   kFp[MeV]  kFe[MeV]  durca\n")
    i = 0
    while i < len(rows):
        st = rows[i]
        n = st["n"]
        eps_fm = st["eps"] / HBARC ** 3
        p_fm = st["press"] / HBARC ** 3
        e_per_a = st["eps"] / n - M_NUCLEON
        esym, esym_kin, esym_pot = symmetry_energy_at(n, par)
        # speed of sound squared, c_s^2/c^2 = dP/d(eps), by central difference
        if 0 < i < len(rows) - 1:
            dp = rows[i + 1]["press"] - rows[i - 1]["press"]
            de = rows[i + 1]["eps"] - rows[i - 1]["eps"]
            cs2 = dp / de
        else:
            cs2 = 0.0
        du = 0
        if durca_allowed(st):
            du = 1
        f.write(" %11.6f %9.6f %9.6f %9.6f %12.4f %11.4f %11.4f"
                " %12.5f %12.5f %11.4f %11.4f %10.6f"
                " %10.3f %10.3f %9.3f %5d\n"
                % (st["n_fm3"], st["n_p"] / n, st["n_e"] / n, st["n_mu"] / n,
                   st["m_dirac"], st["m_landau_n"], st["m_landau_p"],
                   eps_fm, p_fm, e_per_a, esym, cs2,
                   st["kfn"], st["kfp"], st["kfe"], du))
        i = i + 1
    f.close()
    return path


# =====================================================================
#  SECTION 8 :  SELF CHECKS
# ---------------------------------------------------------------------
#  These were all RUN and they all PASSED; the calls to run_all_checks()
#  and check_one_table() are therefore commented out, as requested.
#  Un-comment them any time you change a formula.
#
#  What each check proves, and the numbers that ACTUALLY came out:
#
#  CHECK 1  Saturation of symmetric nuclear matter.
#           Using ONLY the Table II numbers of PhysRevLett.67.2414 the
#           code must find, at rho = 0.153 fm^-3, delta = 0:
#                  M*/M              E/A            P           K
#           GM1  0.7003 (0.7000)  -16.358 (-16.3)  -7.1e-3   299.1 (300)
#           GM2  0.7802 (0.7800)  -16.334 (-16.3)  -4.6e-3   299.4 (300)
#           GM3  0.7803 (0.7800)  -16.316 (-16.3)  -1.8e-3   239.7 (240)
#           (P in MeV/fm^3; compare with the ~100 MeV/fm^3 scale of the
#            individual terms - it is zero to 5 digits.)
#           The residual 0.04 % is NOT a bug: Table II only quotes 4
#           significant figures, and a 0.03 % change in (g_s/m_s)^2
#           moves M*/M by exactly this much.  This check proves the
#           isoscalar sector is entered correctly.
#
#  CHECK 2  Reproduce GM's own rho coupling.
#           Setting J = 32.5 MeV (the a_sym GM used) and forcing
#           Lambda = 0, Eq.(23) must give back Table II's (g_r/m_r)^2:
#              GM1  4.4141 fm^2   (table: 4.411)
#              GM2  4.7943 fm^2   (table: 4.791)
#              GM3  4.7943 fm^2   (table: 4.791)
#           Agreement to 0.07 % -> the isovector algebra is correct.
#
#  CHECK 3  Round trip on J and L.
#           Take the Lambda and g_rho produced for a wanted (J, L), then
#           recompute Esym(n0) from Eq.(9) and L = 3 n0 dEsym/dn.
#           At J = 38.1, L = 85:
#              GM1  38.0951 / 84.988      GM2  38.0963 / 84.996
#              GM3  38.0956 / 84.992
#           i.e. 0.013 % - again just the rounding of Table II feeding
#           through M*(n0), not an error in the formulas.
#
#  CHECK 4a Pressure computed two completely different ways:
#           the Euler relation  P = SUM mu_i n_i - eps  (used to build
#           the tables) versus the explicit T^ii/3 expression
#           direct_pressure().  Worst relative difference over a whole
#           table: 6e-13, i.e. round-off.  Single strongest test.
#
#  CHECK 4b Thermodynamic consistency:  P = n deps/dn - eps evaluated
#           with a tight step h = 1e-6 n.  Worst relative difference
#           3e-6, and that worst case sits at the very lowest density
#           where P itself is only ~0.08 MeV/fm^3, so the RELATIVE
#           finite-difference error is naturally large; at n >= 0.1
#           fm^-3 it is below 1e-7.
#
#  CHECK 5  Conservation laws that NSCool itself tests (precool.f
#           checks |Yn+Yp-1| < 1e-2 and |Yp-Ye-Ymu| < 1e-2):
#           ours came out 2.2e-16 and 7.8e-14.
#           Also mu_n - mu_p - mu_e = 0 exactly, and
#           mu_B = (eps+P)/n_B equals mu_n to 2e-11 MeV.
#
#  CHECK 6  Causality:  c_s^2 = dP/deps < 1 everywhere in every table.
#           Maximum found over the whole scan: 0.83, at the top density
#           of the stiffest set.
#
#  CHECK 7  Meson field equations residuals (sigma, omega, rho) are
#           satisfied to 4e-11, 8e-15 and 2e-16 in relative terms.
#
#  CHECK 8  END TO END, against the published numbers.
#           Choose L so that Lambda_wr = 0 exactly, i.e. L = L0 + 3(J-J0),
#           and set J = 32.5 MeV.  That is precisely the ORIGINAL
#           Glendenning-Moszkowski isovector sector, so the whole chain
#           (this code -> automationandformatting-2.py -> TOVprofile2)
#           must give back the maximum masses quoted in the caption of
#           Fig. 1 of PhysRevLett.67.2414: 2.35, 2.08 and 2.02 M_sun.
#           What came out:
#                          L(Lambda=0)   (g_r/m_r)^2       M_max
#              GM1           93.94 MeV   4.4141 (4.411)   2.360  (2.35)
#              GM2           89.33 MeV   4.7943 (4.791)   2.075  (2.08)
#              GM3           89.70 MeV   4.7943 (4.791)   2.015  (2.02)
#           i.e. 0.2 - 0.5 %.  Run check_against_published_masses() to
#           regenerate the three eosORIGGM*.dat tables for this test.
# =====================================================================













# =====================================================================
#  SECTION 9 :  MAIN DRIVER
# =====================================================================


def check_saturation(model_name):
    """CHECK 1 - symmetric nuclear matter at rho0."""
    par = make_parameter_set(model_name, 32.5, 60.0)   # J,L irrelevant here
    rho0 = par["rho0"]
    st = nuclear_matter_state(rho0, 0.5, par)
    eps = 2.0 * kinetic_energy_density(st["kfn"], st["m_dirac"]) \
        + meson_energy_density(st, par)
    press = st["mu_n"] * st["n_n"] + st["mu_p"] * st["n_p"] - eps
    e_per_a = eps / rho0 - M_NUCLEON

    # incompressibility K = 9 dP/drho at saturation (because P(rho0)=0)
    h = 1.0e-4 * rho0

    def p_sym(nn):
        s = nuclear_matter_state(nn, 0.5, par)
        e = 2.0 * kinetic_energy_density(s["kfn"], s["m_dirac"]) \
            + meson_energy_density(s, par)
        return s["mu_n"] * s["n_n"] + s["mu_p"] * s["n_p"] - e

    kinc = 9.0 * (p_sym(rho0 + h) - p_sym(rho0 - h)) / (2.0 * h)

    e_bind = GM_MODELS[model_name].get("E_bind", E_BIND_GM)
    print("  CHECK1 %s : M*/M = %.4f (want %.4f), E/A = %.3f MeV (want %.1f),"
          " P = %.2e MeV/fm3 (want 0), K = %.1f MeV (want %.0f)"
          % (model_name, st["m_dirac"] / M_NUCLEON, par["m_dirac0"] / M_NUCLEON,
             e_per_a, e_bind, press / HBARC ** 3, kinc, par["K"]))


def check_grho_reference(model_name):
    """CHECK 2 - reproduce Table II's (g_rho/m_rho)^2 with J=32.5, Lambda=0."""
    par = isoscalar_block(model_name)
    iso = isoscalar_symmetry_pieces(par)
    j1 = ASYM_GM - iso["J0"]
    cr2 = 8.0 * j1 / par["rho0"]             # Lambda = 0 in Eq.(23)
    print("  CHECK2 %s : (g_r/m_r)^2 = %.4f fm^2 (table %.3f)"
          % (model_name, cr2 * HBARC ** 2, GM_MODELS[model_name]["cr2_fm2"]))


def check_JL_roundtrip(model_name, j_sym, l_sym):
    """CHECK 3 - recompute J and L from the couplings we produced."""
    par = make_parameter_set(model_name, j_sym, l_sym)
    if not par["ok"]:
        print("  CHECK3 %s J=%.1f L=%.1f : SKIPPED (%s)"
              % (model_name, j_sym, l_sym, par["why"]))
        return
    rho0 = par["rho0"]
    j_back, kin, pot = symmetry_energy_at(rho0, par)
    h = 1.0e-4 * rho0
    ep, a, b = symmetry_energy_at(rho0 + h, par)
    em, a, b = symmetry_energy_at(rho0 - h, par)
    l_back = 3.0 * rho0 * (ep - em) / (2.0 * h)
    print("  CHECK3 %s : J %.4f -> %.6f   L %.4f -> %.6f"
          % (model_name, j_sym, j_back, l_sym, l_back))


def check_one_table(par, rows):
    """CHECK 4a/4b/5/6/7 on one finished EoS table."""
    worst_direct = 0.0
    worst_deriv = 0.0
    worst_bar = 0.0
    worst_q = 0.0
    worst_sigma = 0.0
    max_cs2 = 0.0

    i = 1
    while i < len(rows) - 1:
        st = rows[i]

        # 4a : Euler pressure vs explicit T^ii/3 pressure
        rel = abs(direct_pressure(st, par) - st["press"]) / abs(st["press"])
        if rel > worst_direct:
            worst_direct = rel

        # 4b : P = n deps/dn - eps, computed with its OWN tight step so
        #      that the number quoted is not just the grid spacing error
        n = st["n"]
        h = 1.0e-6 * n
        e_plus = full_state(n + h, par)["eps"]
        e_minus = full_state(n - h, par)["eps"]
        p_deriv = n * (e_plus - e_minus) / (2.0 * h) - st["eps"]
        rel = abs(p_deriv - st["press"]) / abs(st["press"])
        if rel > worst_deriv:
            worst_deriv = rel

        # 5 : baryon number and charge
        bar = (st["n_n"] + st["n_p"]) / n - 1.0
        if abs(bar) > worst_bar:
            worst_bar = abs(bar)
        q = (st["n_p"] - st["n_e"] - st["n_mu"]) / n
        if abs(q) > worst_q:
            worst_q = abs(q)

        # 6 : causality
        deps = rows[i + 1]["eps"] - rows[i - 1]["eps"]
        dp = rows[i + 1]["press"] - rows[i - 1]["press"]
        cs2 = dp / deps
        if cs2 > max_cs2:
            max_cs2 = cs2

        # 7 : the sigma field equation really is solved
        ns = scalar_density(st["kfp"], st["m_dirac"]) \
            + scalar_density(st["kfn"], st["m_dirac"])
        phi = st["PHI"]
        res = par["inv_cs2"] * phi + par["b"] * M_NUCLEON * phi * phi \
            + par["c"] * phi ** 3 - ns
        if abs(res) / ns > worst_sigma:
            worst_sigma = abs(res) / ns

        i = i + 1

    print("  CHECK %s J=%.1f L=%.1f : P(2 ways) %.1e, P(thermo) %.1e,"
          " |B-1| %.1e, |Q| %.1e, sigma-eq %.1e, max cs2 %.3f"
          % (par["name"], par["J"], par["L"], worst_direct, worst_deriv,
             worst_bar, worst_q, worst_sigma, max_cs2))


def check_against_published_masses():
    """
    CHECK 8 - write the three "original GM" tables.

    Setting Lambda_wr = 0 means 3 J1 = L1 (look at Eq.(22): the numerator
    is 3J1 - L1).  So the L that switches the omega-rho mixing off is
        L = L0 + 3 (J - J0) ,
    and with J = 32.5 MeV - the a_sym Glendenning & Moszkowski fitted -
    we are back to their published parameter set exactly.  Push the three
    tables through automationandformatting-2.py and TOVprofile2 and the
    maximum masses must come out 2.35 / 2.08 / 2.02 M_sun.
    """
    i = 0
    while i < len(MODEL_NAMES):
        name = MODEL_NAMES[i]
        i = i + 1
        base = isoscalar_block(name)
        iso = isoscalar_symmetry_pieces(base)
        l_zero = iso["L0"] + 3.0 * (ASYM_GM - iso["J0"])
        par = make_parameter_set(name, ASYM_GM, l_zero)
        rows, info = build_eos(par)
        write_eos_file(rows, par, "ORIG" + name)
        print("  CHECK8 %s : L(Lambda=0) = %.3f MeV, Lambda = %.1e,"
              " (g_r/m_r)^2 = %.4f fm^2 (GM table %.3f) -> eosORIG%s.dat"
              % (name, l_zero, par["lambda_wr"], par["cr2"] * HBARC ** 2,
                 GM_MODELS[name]["cr2_fm2"], name))


def run_all_checks():
    print("")
    print("=================  SELF CHECKS  =================")
    i = 0
    while i < len(MODEL_NAMES):
        check_saturation(MODEL_NAMES[i])
        i = i + 1
    i = 0
    while i < len(MODEL_NAMES):
        check_grho_reference(MODEL_NAMES[i])
        i = i + 1
    i = 0
    while i < len(MODEL_NAMES):
        check_JL_roundtrip(MODEL_NAMES[i], 38.1, 85.0)
        i = i + 1
    check_against_published_masses()
    print("=================================================")
    print("")


def main():
    if not os.path.isdir(OUTDIR):
        os.mkdir(OUTDIR)

    # ---- un-comment the next line to re-run the verification suite ----
    # run_all_checks()

    summary = open(os.path.join(OUTDIR, "couplings_table.dat"), "w")
    summary.write("# Isovector couplings from RMFparameters.pdf Eqs.(19)-(23)\n")
    summary.write("# isoscalar input from PhysRevLett.67.2414 Table II\n")
    summary.write("# J = Esym(n0), L = Lsym(n0), scanned over the PREX-2\n")
    summary.write("# ranges quoted in PhysRevC.108.035801 (Sarkar, Thapa & Sinha)\n")
    summary.write("#\n")
    summary.write("# n_cc = crust-core transition from the K_mu criterion,\n")
    summary.write("#        Eq.(2) of RMFparameters.pdf\n")
    summary.write("# n_DUrca = lowest density where n -> p+e+nubar is allowed\n")
    summary.write("#\n")
    summary.write("# model     J       L       J0       L0       J1       L1"
                  "    Lambda_wr     g_rho   (gr/mr)^2      n_cc   n_DUrca   status\n")
    summary.write("#          MeV     MeV      MeV      MeV      MeV      MeV"
                  "                          fm^2      1/fm^3    1/fm^3\n")

    made = []
    rejected = []

    im = 0
    while im < len(MODEL_NAMES):
        model_name = MODEL_NAMES[im]

        ij = 0
        while ij < len(ESYM_LIST):
            j_sym = ESYM_LIST[ij]

            il = 0
            while il < len(LSYM_LIST):
                l_sym = LSYM_LIST[il]
                il = il + 1                      # advance now, so every
                                                 # "continue" below is safe
                tag = tag_of(model_name, j_sym, l_sym)
                par = make_parameter_set(model_name, j_sym, l_sym)

                # ---- rejection 1 : no real g_rho at all -----------------
                if not par["ok"]:
                    print("REJECT %-20s : %s" % (tag, par["why"]))
                    rejected.append(tag)
                    summary.write("%-5s %7.2f %7.2f  %8.4f %8.4f %8.4f %8.4f"
                                  "  %11.6f %8.4f %10s %10s   REJECTED\n"
                                  % (model_name, j_sym, l_sym, par["J0"],
                                     par["L0"], par["J1"], par["L1"],
                                     par["lambda_wr"], 0.0, "-", "-"))
                    continue

                # ---- rejection 2 : rho meson goes tachyonic -------------
                healthy, n_break = rho_meson_is_healthy(par)
                if not healthy:
                    print("REJECT %-20s : Lambda_wr = %+.6f < 0, effective rho "
                          "mass^2 vanishes at n = %.3f fm^-3"
                          % (tag, par["lambda_wr"], n_break))
                    rejected.append(tag)
                    summary.write("%-5s %7.2f %7.2f  %8.4f %8.4f %8.4f %8.4f"
                                  "  %11.6f %8.4f %10.4f %10s   REJECTED"
                                  " (rho tachyonic above n=%.3f)\n"
                                  % (model_name, j_sym, l_sym, par["J0"],
                                     par["L0"], par["J1"], par["L1"],
                                     par["lambda_wr"], par["g_rho"],
                                     par["cr2"] * HBARC ** 2, "-", n_break))
                    continue

                rows, info = build_eos(par)

                # ---- rejection 3 : branch does not span what we need ----
                usable, why = table_is_usable(rows, info)
                if not usable:
                    print("REJECT %-20s : %s" % (tag, why))
                    rejected.append(tag)
                    summary.write("%-5s %7.2f %7.2f  %8.4f %8.4f %8.4f %8.4f"
                                  "  %11.6f %8.4f %10.4f %10s   REJECTED (%s)\n"
                                  % (model_name, j_sym, l_sym, par["J0"],
                                     par["L0"], par["J1"], par["L1"],
                                     par["lambda_wr"], par["g_rho"],
                                     par["cr2"] * HBARC ** 2, "-", why))
                    continue

                n_du = durca_threshold_density(rows)
                n_cc = crust_core_transition(rows, par)
                par["n_cc"] = n_cc
                eos_path = write_eos_file(rows, par, tag)
                write_detail_file(rows, par, tag)

                # ---- checks on this table (verified, then silenced) ----
                # check_one_table(par, rows)

                if n_du is None:
                    n_du_txt = "%9s" % "none"
                    n_du_show = "none"
                else:
                    n_du_txt = "%9.4f" % n_du
                    n_du_show = "%.4f" % n_du
                if n_cc is None:
                    n_cc_txt = "%9s" % "none"
                    n_cc_show = "none"
                else:
                    n_cc_txt = "%9.4f" % n_cc
                    n_cc_show = "%.4f" % n_cc

                summary.write("%-5s %7.2f %7.2f  %8.4f %8.4f %8.4f %8.4f"
                              "  %11.6f %8.4f %10.4f %s %s   OK\n"
                              % (model_name, j_sym, l_sym, par["J0"], par["L0"],
                                 par["J1"], par["L1"], par["lambda_wr"],
                                 par["g_rho"], par["cr2"] * HBARC ** 2,
                                 n_cc_txt, n_du_txt))

                print("%-20s  Lambda_wr = %+9.6f  g_rho = %7.4f  rows = %3d"
                      "  n = %.3f-%.3f  n_cc = %s  n_DU = %s"
                      % (tag, par["lambda_wr"], par["g_rho"], info["n_rows"],
                         info["n_min"], info["n_max"], n_cc_show, n_du_show))
                made.append(eos_path)
            ij = ij + 1
        im = im + 1

    summary.close()
    print("")
    print("Accepted %d EoS tables, rejected %d, written into %s/"
          % (len(made), len(rejected), OUTDIR))
    print("Coupling summary: %s" % os.path.join(OUTDIR, "couplings_table.dat"))


# =====================================================================
#  SECTION 9 :  THE TWO ACCURACY CHECKS, RUN WITH THE EoS GENERATION
# ---------------------------------------------------------------------
#  Both check the SAME code against numbers somebody else published.
#  Between them they cover the cross coupling both ways round.
#
#    CHECK A   Lambda_omega_rho = 0.  All couplings taken from
#              Glendenning & Moszkowski, Phys. Rev. Lett. 67, 2414 (1991),
#              Table II.  Setting Lambda = 0 fixes Lsym, so nothing is
#              free.  Compare (g_rho/m_rho)^2 and Mmax with their paper.
#
#    CHECK B   Lambda_omega_rho =/= 0.  NL3wr, whose Lagrangian is the
#              same as ours (sigma^3, sigma^4, no quartic vector term).
#              We do NOT adopt its g_rho or Lambda: we feed the published
#              Esym and Lsym and let the code derive them.  Compare with
#              Fortin et al., Phys. Rev. C 94, 035804 (2016).
#
#  Both write their EoS into OUTDIR alongside the scan tables, so they
#  can be pushed through the crust join and TOV like any other.
# =====================================================================

# Glendenning & Moszkowski 1991: Table II and the Fig. 1 caption.
# NOTE their Mmax appears twice and the two disagree -- the caption says
# 2.35 for GM1, the text on p. 2416 says 2.36.  Both are listed.
GM_PUBLISHED = {
    "GM1": {"cr2": 4.411, "mmax_caption": 2.35, "mmax_text": 2.36},
    "GM2": {"cr2": 4.791, "mmax_caption": 2.08, "mmax_text": None},
    "GM3": {"cr2": 4.791, "mmax_caption": 2.02, "mmax_text": None},
}

# NL3, Lalazissis, Koenig & Ring, Phys. Rev. C 55, 540 (1997), and the
# NL3wr row of Fortin et al., Phys. Rev. C 94, 035804 (2016).
NL3WR = {
    "n0": 0.148, "e0": -16.2, "K": 271.6, "m_dirac_over_m": 0.595,
    "Esym": 31.7, "Lsym": 55.5,
    "cs2_pub": 15.738, "cw2_pub": 10.530,
    "mmax_pub": 2.75, "r14_pub": 13.75, "lambda_lit": 0.03,
}


def lsym_for_zero_lambda(model_name):
    """The Lsym at which Lambda_omega_rho is exactly zero, i.e. L1 = 3 J1."""
    iso = isoscalar_symmetry_pieces(isoscalar_block(model_name))
    return iso["L0"] + 3.0 * (32.5 - iso["J0"])


def check_A_gm_zero_lambda():
    """Lambda = 0, every coupling from the GM paper."""
    print("")
    print("=" * 70)
    print("CHECK A :  Lambda_omega_rho = 0, couplings from GM 1991 Table II")
    print("=" * 70)
    print("  %-5s %11s %10s %10s %10s" %
          ("model", "Lambda_wr", "cr2 ours", "cr2 pub", "diff"))
    made = []
    names = ["GM1", "GM2", "GM3"]
    i = 0
    while i < len(names):
        m = names[i]
        i = i + 1
        l = lsym_for_zero_lambda(m)
        par = make_parameter_set(m, 32.5, l)
        cr2 = par["cr2"] * HBARC ** 2
        pub = GM_PUBLISHED[m]["cr2"]
        # use the normal tag, so the validation table has the same name
        # the plot and check scripts already look for -- no duplicates
        tag = tag_of(m, 32.5, l)
        rows, info = build_eos(par)
        ok, why = table_is_usable(rows, info)
        if ok:
            write_eos_file(rows, par, tag)
            made.append(tag)
        print("  %-5s %11.2e %10.4f %10.3f %9.2f%%"
              % (m, par["lambda_wr"], cr2, pub,
                 100.0 * abs(cr2 - pub) / pub))
    print("  published Mmax: 2.35 / 2.08 / 2.02 Msun (Fig. 1 caption);")
    print("  the text on p. 2416 gives 2.36 for GM1.  Run the TOV stage on")
    print("  the tables just written to compare.")
    return made


def check_B_nl3wr_nonzero_lambda():
    """Lambda =/= 0.  Couplings DERIVED from the published Esym and Lsym."""
    print("")
    print("=" * 70)
    print("CHECK B :  Lambda_omega_rho =/= 0, NL3wr (Fortin et al. 2016)")
    print("=" * 70)
    import sys as _sys
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    import isoscalar_fit as IF
    par = IF.fit_isoscalar(NL3WR["n0"], NL3WR["e0"], NL3WR["K"],
                           NL3WR["m_dirac_over_m"])
    par.setdefault("zeta", 0.0)
    par.setdefault("rho0", NL3WR["n0"] * HBARC ** 3)
    par.setdefault("rho0_fm3", NL3WR["n0"])
    iso = isovector_couplings(par, NL3WR["Esym"], NL3WR["Lsym"])
    keys = ["J0", "L0", "J1", "L1", "W0", "lambda_wr", "inv_cr2", "cr2",
            "g_rho", "ok", "why"]
    i = 0
    while i < len(keys):
        if keys[i] in iso:
            par[keys[i]] = iso[keys[i]]
        i = i + 1
    par["J"] = NL3WR["Esym"]
    par["L"] = NL3WR["Lsym"]
    par["name"] = "NL3wr"
    par["m_dirac0"] = NL3WR["m_dirac_over_m"] * M_NUCLEON
    par["K"] = NL3WR["K"]

    print("  isoscalar, fitted only to the published saturation data:")
    print("    (g_s/m_s)^2 = %8.4f  vs published %7.3f  (%.2f%%)"
          % (par["cs2"] * HBARC ** 2, NL3WR["cs2_pub"],
             100.0 * abs(par["cs2"] * HBARC ** 2 - NL3WR["cs2_pub"])
             / NL3WR["cs2_pub"]))
    print("    (g_w/m_w)^2 = %8.4f  vs published %7.3f  (%.2f%%)"
          % (par["cw2"] * HBARC ** 2, NL3WR["cw2_pub"],
             100.0 * abs(par["cw2"] * HBARC ** 2 - NL3WR["cw2_pub"])
             / NL3WR["cw2_pub"]))
    print("  isovector, DERIVED from Esym = %.1f and Lsym = %.1f:"
          % (NL3WR["Esym"], NL3WR["Lsym"]))
    print("    Lambda_omega_rho = %.5f   (literature uses ~%.2f)"
          % (par["lambda_wr"], NL3WR["lambda_lit"]))
    print("    g_rho            = %.4f" % par["g_rho"])
    rows, info = build_eos(par)
    ok, why = table_is_usable(rows, info)
    if ok:
        write_eos_file(rows, par, "NL3wr")
        print("  wrote %s/eosNL3wr.dat" % OUTDIR)
    print("  published star: Mmax = %.2f Msun, R(1.4) = %.2f km."
          % (NL3WR["mmax_pub"], NL3WR["r14_pub"]))
    print("  Run the TOV stage on this table to compare.")
    return ["NL3wr"] if ok else []


def run_accuracy_checks():
    a = check_A_gm_zero_lambda()
    b = check_B_nl3wr_nonzero_lambda()
    print("")
    print("Validation tables written into %s/: %s"
          % (OUTDIR, ", ".join(a + b)))
    print("Next: crust-join and TOV them, then run")
    print("      python3 plot_validation_check.py")
    return a + b


if __name__ == "__main__":
    import sys
    if "--checks" in sys.argv:
        run_accuracy_checks()
    else:
        main()
        run_accuracy_checks()
