#!/usr/bin/env python3
"""
rmf_finite_nucleus.py
=====================
Spherical RMF Hartree calculation for 208Pb and 48Ca neutron skins.

Solves coupled Dirac + meson-field equations self-consistently.
Same Lagrangian as rmf_lambda_and_eos.py.

All radial quantities use r in fm; energies/fields in MeV.
The Dirac equation coefficients are divided by hbarc to give fm^-1.
"""

import numpy as np
from scipy.optimize import brentq
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rmf_lambda_and_eos as RMF

HBARC   = 197.3269804   # MeV fm
M_N     = 939.0
M_SIGMA = 550.0
M_OMEGA = 782.5
M_RHO   = 763.0
ALPHA_EM= 1.0 / 137.036
PI      = np.pi

# ====================== SHELL STRUCTURE ======================

def _s(lab, nr, l, j2):
    kappa = -(j2+1)//2 if j2 == 2*l+1 else (j2+1)//2
    return (lab, nr, l, j2, kappa, j2+1)

PB208_P = [
    _s("1s1/2",1,0,1), _s("1p3/2",1,1,3), _s("1p1/2",1,1,1),
    _s("1d5/2",1,2,5), _s("2s1/2",2,0,1), _s("1d3/2",1,2,3),
    _s("1f7/2",1,3,7), _s("2p3/2",2,1,3), _s("1f5/2",1,3,5),
    _s("2p1/2",2,1,1), _s("1g9/2",1,4,9), _s("1g7/2",1,4,7),
    _s("2d5/2",2,2,5), _s("2d3/2",2,2,3), _s("3s1/2",3,0,1),
    _s("1h11/2",1,5,11),
]
PB208_N = PB208_P + [
    _s("2f7/2",2,3,7), _s("1h9/2",1,5,9), _s("3p3/2",3,1,3),
    _s("3p1/2",3,1,1), _s("2f5/2",2,3,5), _s("1i13/2",1,6,13),
]
CA48_P = [
    _s("1s1/2",1,0,1), _s("1p3/2",1,1,3), _s("1p1/2",1,1,1),
    _s("1d5/2",1,2,5), _s("2s1/2",2,0,1), _s("1d3/2",1,2,3),
]
CA48_N = CA48_P + [_s("1f7/2",1,3,7)]

NUCLEI = {
    "Pb208": {"Z":82,"N":126,"A":208,
              "p_shells":PB208_P,"n_shells":PB208_N,
              "r_max":15.0,"ngrid":300},
    "Ca48":  {"Z":20,"N":28,"A":48,
              "p_shells":CA48_P,"n_shells":CA48_N,
              "r_max":10.0,"ngrid":200},
}

# ====================== DIRAC SOLVER ======================
#
# Radial Dirac equations with r in fm, energies in MeV:
#
#   dG/dr = -kappa/r G + [(eps-V+M*)/hbarc] F
#   dF/dr =  kappa/r F - [(eps-V-M*)/hbarc] G
#
# The /hbarc converts MeV → fm^-1 to match d/dr [fm^-1].

def _integrate_out(r, h, n, kappa, alpha, beta, i_stop):
    """
    Outward RK4.
    alpha = (eps-V+M*)/hbarc  [fm^-1]
    beta  = (eps-V-M*)/hbarc  [fm^-1]
    """
    G = np.zeros(n)
    F = np.zeros(n)

    ak = abs(kappa)
    if kappa < 0:
        G[0] = r[0]**ak
        F[0] = alpha[0] / (2*ak+1) * r[0]**(ak+1)
    else:
        F[0] = r[0]**ak
        G[0] = -beta[0] / (2*ak+1) * r[0]**(ak+1)

    for i in range(min(i_stop, n-1)):
        ri = r[i]
        Gi, Fi = G[i], F[i]
        ai, bi = alpha[i], beta[i]

        k1g = -kappa/ri * Gi + ai*Fi
        k1f =  kappa/ri * Fi - bi*Gi

        rm = ri + 0.5*h
        am = 0.5*(alpha[i]+alpha[i+1])
        bm = 0.5*(beta[i]+beta[i+1])
        G2 = Gi + 0.5*h*k1g; F2 = Fi + 0.5*h*k1f
        k2g = -kappa/rm * G2 + am*F2
        k2f =  kappa/rm * F2 - bm*G2

        G3 = Gi + 0.5*h*k2g; F3 = Fi + 0.5*h*k2f
        k3g = -kappa/rm * G3 + am*F3
        k3f =  kappa/rm * F3 - bm*G3

        rn = r[i+1]
        G4 = Gi + h*k3g; F4 = Fi + h*k3f
        k4g = -kappa/rn * G4 + alpha[i+1]*F4
        k4f =  kappa/rn * F4 - beta[i+1]*G4

        G[i+1] = Gi + (h/6)*(k1g + 2*k2g + 2*k3g + k4g)
        F[i+1] = Fi + (h/6)*(k1f + 2*k2f + 2*k3f + k4f)

        if abs(G[i+1]) + abs(F[i+1]) > 1e100:
            return G, F, i+1

    return G, F, min(i_stop, n-1)


def _integrate_in(r, h, n, kappa, alpha, beta, eps, i_stop):
    """Inward RK4 from r_max to i_stop."""
    G = np.zeros(n)
    F = np.zeros(n)

    lam = np.sqrt(max(M_N**2 - eps**2, 1.0)) / HBARC  # fm^-1
    G[n-1] = np.exp(-lam * r[n-1])
    F[n-1] = -lam / ((eps+M_N)/HBARC) * G[n-1]

    for i in range(n-1, max(i_stop, 0), -1):
        ri = r[i]
        Gi, Fi = G[i], F[i]

        k1g = -kappa/ri * Gi + alpha[i]*Fi
        k1f =  kappa/ri * Fi - beta[i]*Gi

        rm = ri - 0.5*h
        am = 0.5*(alpha[i]+alpha[i-1])
        bm = 0.5*(beta[i]+beta[i-1])
        G2 = Gi - 0.5*h*k1g; F2 = Fi - 0.5*h*k1f
        k2g = -kappa/rm * G2 + am*F2
        k2f =  kappa/rm * F2 - bm*G2

        G3 = Gi - 0.5*h*k2g; F3 = Fi - 0.5*h*k2f
        k3g = -kappa/rm * G3 + am*F3
        k3f =  kappa/rm * F3 - bm*G3

        rp = r[i-1]
        G4 = Gi - h*k3g; F4 = Fi - h*k3f
        k4g = -kappa/rp * G4 + alpha[i-1]*F4
        k4f =  kappa/rp * F4 - beta[i-1]*G4

        G[i-1] = Gi - (h/6)*(k1g + 2*k2g + 2*k3g + k4g)
        F[i-1] = Fi - (h/6)*(k1f + 2*k2f + 2*k3f + k4f)

    return G, F


def _count_nodes(G, i_max):
    s = np.sign(G[:i_max])
    s = s[s != 0]
    if len(s) < 2:
        return 0
    return int(np.sum(s[:-1] * s[1:] < 0))


def find_eigenvalue(r, h, n, kappa, n_nodes, V, Mstar, eps_prev=None):
    """Find Dirac eigenvalue by matching outward/inward at r_match."""
    i_m = n // 2

    def make_ab(eps):
        return (eps - V + Mstar) / HBARC, (eps - V - Mstar) / HBARC

    E_LO = M_N - 70.0
    E_HI = M_N - 0.5

    def match_fn(eps):
        a, b = make_ab(eps)
        Go, Fo, _ = _integrate_out(r, h, n, kappa, a, b, i_m+1)
        Gi, Fi = _integrate_in(r, h, n, kappa, a, b, eps, i_m)
        norm_o = max(abs(Go[i_m]), abs(Fo[i_m]), 1e-30)
        norm_i = max(abs(Gi[i_m]), abs(Fi[i_m]), 1e-30)
        return (Go[i_m]*Fi[i_m] - Fo[i_m]*Gi[i_m]) / (norm_o * norm_i)

    def nodes_at(eps):
        a, b = make_ab(eps)
        G, _, _ = _integrate_out(r, h, n, kappa, a, b, i_m+1)
        return _count_nodes(G, i_m)

    # Strategy: scan for sign changes of the match function at energies
    # where the outward solution has the correct node count.
    # Use previous eigenvalue to set a narrow window first.
    if eps_prev is not None:
        e_lo = max(E_LO, eps_prev - 15.0)
        e_hi = min(E_HI, eps_prev + 15.0)
    else:
        e_lo, e_hi = E_LO, E_HI

    # Scan: evaluate match_fn and node count at several energies
    n_scan = 40
    eps_arr = np.linspace(e_lo, e_hi, n_scan)
    f_arr = np.empty(n_scan)
    nd_arr = np.empty(n_scan, dtype=int)

    for i in range(n_scan):
        a, b = make_ab(eps_arr[i])
        Go, Fo, _ = _integrate_out(r, h, n, kappa, a, b, i_m+1)
        Gi, Fi = _integrate_in(r, h, n, kappa, a, b, eps_arr[i], i_m)
        norm_o = max(abs(Go[i_m]), abs(Fo[i_m]), 1e-30)
        norm_i = max(abs(Gi[i_m]), abs(Fi[i_m]), 1e-30)
        f_arr[i] = (Go[i_m]*Fi[i_m] - Fo[i_m]*Gi[i_m]) / (norm_o*norm_i)
        nd_arr[i] = _count_nodes(Go, i_m)

    # Find sign changes with correct node count
    found = False
    e_a, e_b = e_lo, e_hi
    for i in range(n_scan - 1):
        if nd_arr[i] == n_nodes and nd_arr[i+1] == n_nodes:
            if f_arr[i] * f_arr[i+1] < 0:
                e_a, e_b = eps_arr[i], eps_arr[i+1]
                found = True
                break
        elif nd_arr[i] == n_nodes and nd_arr[i+1] != n_nodes:
            if f_arr[i] * f_arr[i+1] < 0:
                e_a, e_b = eps_arr[i], eps_arr[i+1]
                found = True
                break

    if not found:
        # Wider scan
        eps_arr2 = np.linspace(E_LO, E_HI, 80)
        for i in range(79):
            a, b = make_ab(eps_arr2[i])
            Go, Fo, _ = _integrate_out(r, h, n, kappa, a, b, i_m+1)
            Gi, Fi = _integrate_in(r, h, n, kappa, a, b, eps_arr2[i], i_m)
            nd = _count_nodes(Go, i_m)
            if nd != n_nodes:
                continue
            norm_o = max(abs(Go[i_m]), abs(Fo[i_m]), 1e-30)
            norm_i = max(abs(Gi[i_m]), abs(Fi[i_m]), 1e-30)
            f1 = (Go[i_m]*Fi[i_m] - Fo[i_m]*Gi[i_m]) / (norm_o*norm_i)

            a2, b2 = make_ab(eps_arr2[i+1])
            Go2, Fo2, _ = _integrate_out(r, h, n, kappa, a2, b2, i_m+1)
            Gi2, Fi2 = _integrate_in(r, h, n, kappa, a2, b2, eps_arr2[i+1], i_m)
            norm_o2 = max(abs(Go2[i_m]), abs(Fo2[i_m]), 1e-30)
            norm_i2 = max(abs(Gi2[i_m]), abs(Fi2[i_m]), 1e-30)
            f2 = (Go2[i_m]*Fi2[i_m] - Fo2[i_m]*Gi2[i_m]) / (norm_o2*norm_i2)

            if f1 * f2 < 0:
                e_a, e_b = eps_arr2[i], eps_arr2[i+1]
                found = True
                break

    if found:
        eps_sol = brentq(match_fn, e_a, e_b, xtol=1e-4, maxiter=80)
    else:
        eps_sol = 0.5*(E_LO + E_HI)

    # Build final wavefunction
    a, b = make_ab(eps_sol)
    Go, Fo, _ = _integrate_out(r, h, n, kappa, a, b, i_m+1)
    Gi, Fi = _integrate_in(r, h, n, kappa, a, b, eps_sol, i_m)

    scale = Go[i_m] / Gi[i_m] if abs(Gi[i_m]) > 1e-30 else 1.0

    G = np.copy(Go)
    F = np.copy(Fo)
    G[i_m:] = Gi[i_m:] * scale
    F[i_m:] = Fi[i_m:] * scale

    norm = np.trapz(G**2 + F**2, r)
    if norm > 1e-30:
        s = 1.0 / np.sqrt(norm)
        G *= s
        F *= s

    return eps_sol, G, F


# ====================== DENSITIES ======================

def compute_densities(r, p_st, n_st):
    nn = len(r)
    rho_s  = np.zeros(nn)
    rho_vp = np.zeros(nn)
    rho_vn = np.zeros(nn)
    f = 1.0 / (4*PI*r**2)

    for (sh, _, G, F) in p_st:
        d = sh[5]
        rho_vp += d*f*(G**2 + F**2)
        rho_s  += d*f*(G**2 - F**2)

    for (sh, _, G, F) in n_st:
        d = sh[5]
        rho_vn += d*f*(G**2 + F**2)
        rho_s  += d*f*(G**2 - F**2)

    return rho_s, rho_vp + rho_vn, rho_vp, rho_vn


# ====================== FIELD SOLVERS ======================
# Field equations in fm coordinates with fields in MeV.
# See derivation in the docstring: divide natural-unit equation by hbarc^2.

def _tridiag(r, h, n, m2, src):
    """Solve -[f''+(2/r)f']+m2*f = src."""
    ih2 = 1.0/(h*h)
    a_vec = np.empty(n)
    b_vec = np.empty(n)
    c_vec = np.empty(n)
    d = np.copy(src)

    irh = 1.0/(r*h)
    a_vec[:] = -ih2 + irh
    c_vec[:] = -ih2 - irh
    if np.ndim(m2) == 0:
        b_vec[:] = 2*ih2 + m2
    else:
        b_vec[:] = 2*ih2 + m2

    # Thomas algorithm
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = c_vec[0]/b_vec[0]
    dp[0] = d[0]/b_vec[0]
    for i in range(1, n):
        w = b_vec[i] - a_vec[i]*cp[i-1]
        if abs(w) < 1e-30: w = 1e-30
        cp[i] = c_vec[i]/w
        dp[i] = (d[i] - a_vec[i]*dp[i-1])/w

    phi = np.empty(n)
    phi[n-1] = dp[n-1]
    for i in range(n-2, -1, -1):
        phi[i] = dp[i] - cp[i]*phi[i+1]
    return phi


def sigma_fld(r, h, n, rho_s, sig_old, p):
    gs = p["g_sigma"]
    ms2 = (M_SIGMA/HBARC)**2       # fm^-2
    bc = p["b"]*M_N*gs**3/HBARC**2  # fm^-2 MeV^-1
    cc = p["c"]*gs**4/HBARC**2      # fm^-2 MeV^-2
    m2 = ms2 + 2*bc*sig_old + 3*cc*sig_old**2
    src = gs*HBARC*rho_s + bc*sig_old**2 + 2*cc*sig_old**3
    return _tridiag(r, h, n, m2, src)


def omega_fld(r, h, n, rho_B, omg_old, rho03, p):
    gw = p["g_omega"]
    gr = p.get("g_rho", 0.0)
    z  = p.get("zeta", 0.0)
    lm = p.get("lambda_wr", 0.0)
    mw2 = (M_OMEGA/HBARC)**2
    zc = z*gw**4/(6*HBARC**2)
    xc = 2*lm*gw**2*gr**2/HBARC**2
    m2 = mw2 + 3*zc*omg_old**2 + xc*rho03**2
    src = gw*HBARC*rho_B + 2*zc*omg_old**3
    return _tridiag(r, h, n, m2, src)


def rho_fld(r, h, n, rho_3, omega, p):
    gr = p.get("g_rho", 0.0)
    gw = p["g_omega"]
    lm = p.get("lambda_wr", 0.0)
    mr2 = (M_RHO/HBARC)**2
    xc = 2*lm*gw**2*gr**2/HBARC**2
    m2 = mr2 + xc*omega**2
    src = 0.5*gr*HBARC*rho_3
    return _tridiag(r, h, n, m2, src)


def coulomb_fld(r, h, n, rho_p):
    Q = np.cumsum(4*PI*r**2*rho_p)*h
    I_r = np.cumsum((4*PI*r*rho_p)[::-1])[::-1]*h
    return ALPHA_EM*HBARC*(Q/r + I_r)


# ====================== MAIN SOLVER ======================

def solve_nucleus(nuc_name, par, max_iter=80, mix=0.3, tol=1e-4,
                  verbose=True):
    nuc = NUCLEI[nuc_name]
    n = nuc["ngrid"]
    h = nuc["r_max"] / n
    r = np.arange(1, n+1) * h
    A = nuc["A"]

    if verbose:
        print(f"  {nuc_name}: {par.get('name','?')} J={par.get('J','?')} "
              f"L={par.get('L','?')}")

    # Initial Woods-Saxon fields
    R0 = 1.2 * A**(1.0/3.0)
    ws = 1.0 / (1.0 + np.exp((r - R0) / 0.5))

    gs = par["g_sigma"]
    gw = par["g_omega"]
    gr = par.get("g_rho", 0.0)

    sig0 = par["PHI0"] / gs   # sigma at center (MeV)
    W0 = par.get("W0", 0.0)
    omg0 = W0 / gw if gw > 0 else 0.0

    sigma = sig0 * ws
    omega = omg0 * ws
    rho03 = np.zeros(n)
    A0 = np.zeros(n)

    ep_p = [None]*len(nuc["p_shells"])
    ep_n = [None]*len(nuc["n_shells"])

    converged = False
    for it in range(max_iter):
        Mstar = M_N - gs*sigma
        Vp = gw*omega + 0.5*gr*rho03 + A0
        Vn = gw*omega - 0.5*gr*rho03

        pst = []
        for idx, sh in enumerate(nuc["p_shells"]):
            _, nr, _, _, kappa, _ = sh
            nn = nr if kappa > 0 else nr - 1
            eps, G, F = find_eigenvalue(r, h, n, kappa, nn,
                                        Vp, Mstar, ep_p[idx])
            ep_p[idx] = eps
            pst.append((sh, eps, G, F))

        nst = []
        for idx, sh in enumerate(nuc["n_shells"]):
            _, nr, _, _, kappa, _ = sh
            nn = nr if kappa > 0 else nr - 1
            eps, G, F = find_eigenvalue(r, h, n, kappa, nn,
                                        Vn, Mstar, ep_n[idx])
            ep_n[idx] = eps
            nst.append((sh, eps, G, F))

        rho_s, rho_B, rho_p, rho_n = compute_densities(r, pst, nst)

        if np.any(np.isnan(rho_s)):
            if verbose:
                print(f"    iter {it}: NaN — stopping")
            break

        sig_new = sigma_fld(r, h, n, rho_s, sigma, par)
        omg_new = omega_fld(r, h, n, rho_B, omega, rho03, par)
        r03_new = rho_fld(r, h, n, rho_p - rho_n, omg_new, par)
        A0_new  = coulomb_fld(r, h, n, rho_p)

        ds = np.max(np.abs(sig_new - sigma))/(np.max(np.abs(sigma))+1e-10)
        do = np.max(np.abs(omg_new - omega))/(np.max(np.abs(omega))+1e-10)
        dm = max(ds, do)

        if verbose and (it < 3 or it % 10 == 0 or dm < tol):
            print(f"    iter {it:3d}: Δ={dm:.2e}")

        sigma = mix*sig_new + (1-mix)*sigma
        omega = mix*omg_new + (1-mix)*omega
        rho03 = mix*r03_new + (1-mix)*rho03
        A0 = A0_new

        if dm < tol and it > 3:
            converged = True
            if verbose:
                print(f"    CONVERGED at iter {it}")
            break

    if not converged and verbose:
        print(f"    NOT CONVERGED (Δ={dm:.2e})")

    # Observables
    r2p = 0.0
    for (sh, _, G, F) in pst:
        r2p += sh[5] * np.trapz(r**2*(G**2+F**2), r)
    r2p /= nuc["Z"]

    r2n = 0.0
    for (sh, _, G, F) in nst:
        r2n += sh[5] * np.trapz(r**2*(G**2+F**2), r)
    r2n /= nuc["N"]

    rp = np.sqrt(max(r2p, 0))
    rn = np.sqrt(max(r2n, 0))
    skin = rn - rp

    rch = np.sqrt(max(r2p + 0.8775**2 + (nuc["N"]/nuc["Z"])*(-0.1149), 0))

    if verbose:
        print(f"    r_p={rp:.4f}  r_n={rn:.4f}  skin={skin:.4f}  "
              f"r_ch={rch:.4f}")
        # Print a few eigenvalues
        for (sh, eps, _, _) in pst[:3]:
            print(f"      proton {sh[0]:8s}: eps={eps:.2f}  "
                  f"B.E.={M_N-eps:.2f} MeV")
        for (sh, eps, _, _) in nst[:3]:
            print(f"      neutron {sh[0]:8s}: eps={eps:.2f}  "
                  f"B.E.={M_N-eps:.2f} MeV")

    return {"nuc": nuc_name, "converged": converged,
            "r_p": rp, "r_n": rn, "skin": skin, "r_ch": rch}


# ====================== MAIN ======================

def main():
    import time
    print("="*60)
    print("  Finite-nucleus RMF: neutron skin")
    print("="*60)

    cases = [
        ("BigApple",  25.0, 50.0),
        ("BigApple",  31.0, 70.0),
        ("BigApple",  37.0, 100.0),
        ("GM1",       32.5, 50.0),
        ("GM1",       32.5, 70.0),
        ("GM1",       32.5, 100.0),
        ("FSUGarnet", 31.0, 55.0),
        ("IOPB-I",    31.0, 65.0),
    ]

    results = []
    for model, J, L in cases:
        print(f"\n--- {model} J={J} L={L} ---")
        par = RMF.make_parameter_set(model, J, L)
        if not par.get("ok", False):
            print(f"  SKIP: {par.get('why','')}")
            continue
        row = {"model": model, "J": J, "L": L}
        t0 = time.time()
        for nuc in ["Pb208", "Ca48"]:
            res = solve_nucleus(nuc, par, verbose=True)
            row[f"{nuc}_skin"] = res["skin"]
            row[f"{nuc}_rch"] = res["r_ch"]
        row["time"] = time.time() - t0
        results.append(row)

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':12s} {'J':>5s} {'L':>5s}  "
          f"{'skin(Pb)':>9s} {'rch(Pb)':>8s}  "
          f"{'skin(Ca)':>9s} {'rch(Ca)':>8s}  {'t(s)':>5s}")
    for row in results:
        print(f"  {row['model']:12s} {row['J']:5.1f} {row['L']:5.0f}  "
              f"{row['Pb208_skin']:9.4f} {row['Pb208_rch']:8.4f}  "
              f"{row['Ca48_skin']:9.4f} {row['Ca48_rch']:8.4f}  "
              f"{row['time']:5.1f}")
    print(f"\n  Expt: Pb r_ch=5.5012  Ca r_ch=3.4776")
    print(f"  PREX-2: skin(Pb)=0.283+-0.071")
    print(f"  CREX:   skin(Ca)=0.121+-0.050")


if __name__ == "__main__":
    main()
