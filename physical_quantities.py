#!/usr/bin/env python3
# =====================================================================
#  physical_quantities.py
# ---------------------------------------------------------------------
#  For every surviving EoS, compute the quantities cooling really
#  probes and write them to reports/data/eos_physical_quantities.dat
#  (the input of combine_cooling_lab.py):
#
#     Ksym, S(2n0), S(3n0), x_p(2n0), x_p(3n0), n_DU   from the RMF model
#     M_DU, Mmax, R1.4                                 from the pipeline
#                                                      MRcurve_<TAG>_CAT.dat
#
#  M_DU is the mass of the star whose central energy density equals
#  eps(n_DU), read off the stable branch of the MR curve.  "inf" means
#  direct Urca never opens in a stable star; "nan" means the MR curve
#  file was not found.
#
#  Usage:
#     python3 physical_quantities.py --tagfile ../../Data/surviving_tags.txt
#     python3 physical_quantities.py --tagfile tags.txt --mrdir /path/NSCool/EOS
#  The tag file has one tag per line, e.g. GM2_J32p5_L80p0.
# =====================================================================
import argparse
import os
import sys

import numpy as np

import rmf_lambda_and_eos as R

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "reports", "data", "eos_physical_quantities.dat")


def find_mrdir():
    for d in (os.path.join(HERE, "..", "..", "NSCool", "EOS"),
              os.path.join(HERE, "..", "NSCool", "EOS"),
              os.path.join(HERE, "NSCool", "EOS")):
        if os.path.isdir(d):
            return os.path.normpath(d)
    return None


def parse_tag(tag):
    model, js, ls = tag.rsplit("_", 2)
    return model, float(js[1:].replace("p", ".")), float(ls[1:].replace("p", "."))


def read_mr(path):
    """Stable branch (up to Mmax) of an MRcurve file: M, R, rho_c."""
    a = np.loadtxt(path, comments="#")
    i = int(np.argmax(a[:, 0]))
    return a[:i + 1, 0], a[:i + 1, 1], a[:i + 1, 2]


def quantities(model, J, L, mrfile):
    par = R.make_parameter_set(model, J, L)
    if not par["ok"]:
        return None
    n0 = par["rho0"]
    h = 1e-3 * n0

    def S(n):
        return R.symmetry_energy_at(n, par)[0]

    ksym = 9 * n0 * n0 * (S(n0 + h) - 2 * S(n0) + S(n0 - h)) / h ** 2
    xp = []
    for f in (2, 3):
        b = R.beta_equilibrium(f * n0, par)
        xp.append(b["n_p"] / b["n"])
    rows, _ = R.build_eos(par)
    ndu = R.durca_threshold_density(rows)

    mdu = mmax = r14 = float("nan")
    if os.path.isfile(mrfile):
        m, r, rc = read_mr(mrfile)
        mmax = float(m[-1])
        r14 = float(np.interp(1.4, m, r)) if mmax >= 1.4 else float("nan")
        if ndu is None:
            mdu = float("inf")
        else:
            st = [x for x in rows if x["n_fm3"] == ndu][0]
            rho = st["eps"] / R.HBARC ** 3 * R.MEVFM3_TO_GCM3
            mdu = float(np.interp(rho, rc, m)) if rho <= rc[-1] else float("inf")
    return (ksym, S(2 * n0), S(3 * n0), xp[0], xp[1],
            ndu if ndu else float("inf"), mdu, mmax, r14)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tagfile", required=True,
                    help="one EoS tag per line (the surviving set)")
    ap.add_argument("--mrdir", default=None,
                    help="folder with MRcurve_<TAG>_CAT.dat (default: NSCool/EOS)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    mrdir = a.mrdir or find_mrdir()
    if mrdir is None:
        sys.exit("NSCool/EOS not found; pass --mrdir")
    tags = [t.strip() for t in open(a.tagfile) if t.strip()]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    n_missing = 0
    with open(a.out, "w") as out:
        out.write("# model J L Ksym S(2n0) S(3n0) xp(2n0) xp(3n0) n_DU M_DU "
                  "Mmax R14   (Ksym,S in MeV; n in fm-3; M in Msun; "
                  "M_DU/Mmax/R14 from pipeline MR curves)\n")
        for i, tag in enumerate(sorted(tags, key=parse_tag)):
            model, J, L = parse_tag(tag)
            q = quantities(model, J, L,
                           os.path.join(mrdir, "MRcurve_%s_CAT.dat" % tag))
            if q is None:
                print("  skipped (no parameter set): %s" % tag)
                continue
            if np.isnan(q[6]):
                n_missing += 1
            out.write("%-9s %5.1f %6.1f %8.1f %6.2f %6.2f %.4f %.4f %7.4f "
                      "%7.3f %6.3f %6.2f\n" % ((model, J, L) + q))
            if (i + 1) % 50 == 0:
                print("  %d / %d" % (i + 1, len(tags)), flush=True)
    print("wrote %s  (%d EoS; MR curve missing for %d)"
          % (a.out, len(tags), n_missing))


if __name__ == "__main__":
    main()
