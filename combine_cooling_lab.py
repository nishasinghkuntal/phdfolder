#!/usr/bin/env python3
# =====================================================================
#  combine_cooling_lab.py
# ---------------------------------------------------------------------
#  1. Turns the cooling chi-square of every EoS into credible intervals
#     for the quantities cooling actually probes -- not only (Esym, Lsym)
#     but S(2 n0), the proton fraction x_p(2 n0) and the direct Urca
#     threshold mass M_DU -- and the posterior probability that direct
#     Urca opens at or below 1.4 Msun.
#  2. Combines cooling with the INDRA-FAZIA (isospin diffusion)
#     likelihood published for external use by Montefusco et al.,
#     arXiv:2609.24940, Appendix D, Table II: a Gaussian in
#     (Esym, Lsym, Ksym) with
#         mean  (28.9, 40.7, -182) MeV
#         cov   [[0.877, -0.951, -205], [-0.951, 200, 3380],
#                [-205, 3380, 1.56e5]] MeV^2
#
#  Conventions follow joint_likelihood.py: chi2 is divided by the PDG
#  scale factor S^2 = chi2_min/dof (dof = N_stars - 2) and every
#  parametrization gets equal total weight.
#
#  Inputs:
#     cooling chi2 csv  (Data/cooling_chi2_final.csv; columns model,J,L,
#                        chi2_total,n_compared,...)
#     reports/data/eos_physical_quantities.dat  (from this project)
#
#  Usage:
#     python3 combine_cooling_lab.py Data/cooling_chi2_final.csv
#     python3 combine_cooling_lab.py old.csv --mode durca_sf   # older files
# =====================================================================
import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHYS = os.path.join(HERE, "reports", "data", "eos_physical_quantities.dat")

IF_MEAN = np.array([28.9, 40.7, -182.0])
IF_COV = np.array([[0.877, -0.951, -205.0],
                   [-0.951, 200.0, 3380.0],
                   [-205.0, 3380.0, 1.56e5]])


def load_phys():
    d = {}
    for line in open(PHYS):
        if line.startswith("#"):
            continue
        p = line.split()
        key = (p[0], float(p[1]), float(p[2]))
        d[key] = {"Ksym": float(p[3]), "S2": float(p[4]), "S3": float(p[5]),
                  "xp2": float(p[6]), "xp3": float(p[7]), "nDU": float(p[8]),
                  "MDU": float(p[9]), "Mmax": float(p[10]), "R14": float(p[11])}
    return d


def load_chi2(path, mode=None):
    out = {}
    for r in csv.DictReader(open(path)):
        if mode is not None and r.get("mode") != mode:
            continue
        key = (r["model"], float(r["J"]), float(r["L"]))
        out[key] = (float(r["chi2_total"]), int(float(r["n_compared"])))
    return out


def interval(vals, w, level):
    vals = np.asarray(vals, float)
    w = np.asarray(w, float)
    k = np.isfinite(vals)
    vals, w = vals[k], w[k]
    if w.sum() <= 0:
        return None
    o = np.argsort(vals)
    v, w = vals[o], w[o] / w.sum()
    c = np.cumsum(w)
    best = None
    for i in range(len(v)):
        j = np.searchsorted(c, c[i] - w[i] + level)
        if j >= len(v):
            break
        if best is None or v[j] - v[i] < best[1] - best[0]:
            best = (v[i], v[j])
    return best


def weights(keys, chi2, extra=None):
    cmin = min(chi2[k][0] for k in keys)
    n = chi2[keys[0]][1]
    s2 = max(cmin / (n - 2), 1.0)
    w = np.array([math.exp(-0.5 * (chi2[k][0] - cmin) / s2) for k in keys])
    if extra is not None:
        w = w * extra
    models = sorted(set(k[0] for k in keys))
    for m in models:                      # equal weight per parametrization
        idx = [i for i, k in enumerate(keys) if k[0] == m]
        tot = w[idx].sum()
        if tot > 0:
            w[idx] /= tot
    return w / w.sum(), s2


def report(title, keys, phys, w):
    print("\n--- %s ---" % title)
    cols = [("Esym [MeV]", [k[1] for k in keys]),
            ("Lsym [MeV]", [k[2] for k in keys]),
            ("S(2n0) [MeV]", [phys[k]["S2"] for k in keys]),
            ("x_p(2n0)", [phys[k]["xp2"] for k in keys]),
            ("R1.4 [km]", [phys[k]["R14"] for k in keys])]
    for name, v in cols:
        a, b = interval(v, w, 0.6827), interval(v, w, 0.9545)
        print("  %-13s 68%%: [%7.3f, %7.3f]   95%%: [%7.3f, %7.3f]"
              % (name, a[0], a[1], b[0], b[1]))
    mdu = np.array([phys[k]["MDU"] for k in keys])
    known = ~np.isnan(mdu)
    wk = w[known] / w[known].sum()
    p14 = float(wk[mdu[known] <= 1.4].sum())
    pinf = float(wk[np.isinf(mdu[known])].sum())
    fin = np.isfinite(mdu)
    a = interval(mdu[fin], w[fin], 0.6827)
    print("  M_DU (finite)  68%%: [%.2f, %.2f] Msun;  P(M_DU <= 1.4) = %.3f;"
          "  P(no DUrca in stable stars) = %.3f   [M_DU known for %d of %d EoS]"
          % (a[0], a[1], p14, pinf, known.sum(), len(keys)))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    mode = None
    if "--mode" in sys.argv:
        mode = sys.argv[sys.argv.index("--mode") + 1]
        args = [a for a in args if a != mode]
    path = args[0] if args else os.path.join(HERE, "Data", "cooling_chi2_final.csv")
    phys = load_phys()
    chi2 = load_chi2(path, mode)
    keys = sorted(k for k in chi2 if k in phys)
    print("cooling file: %s%s   EoS used: %d"
          % (path, "" if mode is None else " (mode %s)" % mode, len(keys)))

    w, s2 = weights(keys, chi2)
    print("PDG scale factor S^2 = %.3f" % s2)
    report("cooling alone", keys, phys, w)

    ci = np.linalg.inv(IF_COV)
    lif = []
    for k in keys:
        x = np.array([k[1], k[2], phys[k]["Ksym"]]) - IF_MEAN
        lif.append(math.exp(-0.5 * float(x @ ci @ x)))
    lif = np.array(lif)
    w2, _ = weights(keys, chi2, extra=lif)
    report("cooling + INDRA-FAZIA (Montefusco et al. App. D)", keys, phys, w2)

    wif = lif / lif.sum()
    report("INDRA-FAZIA alone, restricted to this model family", keys, phys, wif)


if __name__ == "__main__":
    main()
