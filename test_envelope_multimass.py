#!/usr/bin/env python3
"""
test_envelope_multimass.py — Multi-mass envelope test for a subset of EoS.

Runs a representative sample of EoS (spanning L=30-130) at all 4 masses
with accreted envelope (ETA=1.0) in durca_sf mode, then compares the
multi-mass chi2 with the existing iron-envelope results.

This tests whether the multi-mass profiling can compensate for the
steeper accreted-envelope cooling curves.
"""
import os
import sys
import subprocess
import shutil
import glob
import time
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NSCOOL_DIR = os.path.join(BASE, "NSCool")
COOL_OUT = os.path.join(BASE, "Data", "Cooling_Outputs")

OBS = [
    ("Cas A",       2.51, 6.26, 0.05),
    ("XMMU J1732",  3.48, 6.25, 0.10),
    ("RX J0822",    3.57, 6.24, 0.05),
    ("1E 1207",     3.85, 6.20, 0.10),
    ("PSR B0833",   4.05, 5.88, 0.10),
    ("PSR B1706",   4.24, 5.81, 0.10),
    ("PSR J0538",   4.48, 5.95, 0.10),
    ("PSR B2334",   4.61, 5.53, 0.15),
    ("PSR B0656",   5.05, 5.71, 0.10),
    ("Geminga",     5.53, 5.75, 0.10),
    ("RX J1856",    5.70, 5.70, 0.10),
    ("PSR B1055",   5.73, 5.59, 0.10),
    ("RX J0720",    5.78, 5.72, 0.10),
    ("PSR J2043",   5.88, 5.22, 0.15),
    ("PSR J0437",   6.85, 5.34, 0.15),
    ("PSR B0950",   7.24, 5.02, 0.20),
]

MASSES = [1.0, 1.4, 1.8, 2.0]

# Representative sample: best-fit EoS + spread across L values
SAMPLE_TAGS = [
    "BigApple_J25p0_L50p0",   # best fit
    "BigApple_J25p0_L30p0",   # low L
    "BigApple_J25p0_L70p0",   # medium L
    "BigApple_J25p0_L100p0",  # high L
    "GM1_J25p0_L50p0",        # different model, same L
    "GM1_J25p0_L70p0",
    "GM1_J25p0_L100p0",
    "FSUGarnet_J25p0_L50p0",
    "FSUGarnet_J25p0_L70p0",
    "IOPB-I_J25p0_L50p0",
    "IOPB-I_J25p0_L70p0",
    "BigApple_J31p0_L50p0",   # different J
    "BigApple_J37p0_L50p0",
    "BigApple_J43p0_L50p0",
    "BigApple_J31p0_L70p0",
    "BigApple_J37p0_L70p0",
]


def write_cool_input(eos_tag, tov_profile_path):
    eos_file = "EOS/%s_CAT.dat" % eos_tag
    tov_rel = os.path.relpath(tov_profile_path, NSCOOL_DIR)
    content = """  'NEW'
BASIC MODEL FILES:
  'EOS/Crust/Crust_EOS_Cat_HZD-NV.dat'
  '%s'
  '%s'
OTHER MODEL FILES:
  'I_Files/I_Struct_1.6e14-4.0e11-1e10_normal.dat'
  'I_Files/I_Bound_Acc.dat'
  'I_Files/I_Pairing_SFB-a-T73.dat'
  'I_Files/I_Neutrino_1.dat'
  'I_Files/I_Conduct_32.dat'
  'I_Files/I_Heat_0.dat'
  'I_Files/I_Bfield_0.dat'
  'I_Files/I_Accretion_0.dat'
OUTPUT FILES:
  'Model_1/I.dat'
  'Model_1/Teff_Try.dat'
  'Model_1/Temp_Try.dat'
  'Model_1/Star_Try.dat'


""" % (eos_file, tov_rel)
    inpath = os.path.join(NSCOOL_DIR, "Model_1", "Cool_Try.in")
    with open(inpath, "w") as f:
        f.write(content)


def find_tov_profile(eos_tag, target_mass):
    prof_dir = os.path.join(NSCOOL_DIR, "TOV", "Profile")
    target_int = int(round(target_mass * 10))
    pattern = os.path.join(prof_dir,
                           "TOVprofile_%s_CAT_targetM%d.dat" % (eos_tag, target_int))
    if os.path.isfile(pattern):
        return pattern
    return None


def run_nscool():
    for f in ("Teff_Try.dat", "Temp_Try.dat", "Star_Try.dat"):
        p = os.path.join(NSCOOL_DIR, "Model_1", f)
        if os.path.exists(p):
            os.remove(p)
    idat_dst = os.path.join(NSCOOL_DIR, "Model_1", "I.dat")
    idat_src = os.path.join(NSCOOL_DIR, "I_Files", "I.dat")
    shutil.copy2(idat_src, idat_dst)
    try:
        result = subprocess.run(
            ["./NSCool.out"],
            input="Cool_Try.in\n",
            capture_output=True, text=True, cwd=NSCOOL_DIR, timeout=300
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def read_teff(fname):
    t, T = [], []
    with open(fname) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                int(parts[0])
                ti, Ti = float(parts[1]), float(parts[2])
                if ti > 0 and Ti > 0:
                    t.append(ti)
                    T.append(Ti)
            except (ValueError, IndexError):
                continue
    return np.array(t), np.array(T)


def interp_T_at_age(log_t_arr, log_T_arr, log_age):
    if len(log_t_arr) < 5 or log_t_arr.max() < 5.0:
        return None
    if log_age < log_t_arr.min():
        return None
    if log_age > log_t_arr.max():
        return log_T_arr[-1]
    return np.interp(log_age, log_t_arr, log_T_arr)


def compute_multimass_chi2(curves_by_mass):
    chi2_total = 0.0
    n = 0
    for name, log_age, log_Tobs, sigma in OBS:
        best_chi2_i = 1e10
        for mass in sorted(curves_by_mass.keys()):
            log_t, log_T = curves_by_mass[mass]
            T_pred = interp_T_at_age(log_t, log_T, log_age)
            if T_pred is None:
                continue
            chi2_i = ((T_pred - log_Tobs) / sigma) ** 2
            if chi2_i < best_chi2_i:
                best_chi2_i = chi2_i
        if best_chi2_i < 1e10:
            chi2_total += best_chi2_i
            n += 1
    return chi2_total / n if n > 0 else None


def load_existing_iron_chi2(tag):
    """Load iron envelope cooling curves from existing data."""
    from collections import defaultdict
    mass_suffixes = {1.0: "_M10", 1.4: "", 1.8: "_M18", 2.0: "_M20"}
    curves = {}
    for mass, suffix in mass_suffixes.items():
        dir_name = "durca_sf" + suffix if suffix else "durca_sf"
        teff_file = os.path.join(COOL_OUT, dir_name, tag, "Teff_%s.dat" % tag)
        if not os.path.isfile(teff_file):
            continue
        t_arr, T_arr = read_teff(teff_file)
        if len(t_arr) < 10:
            continue
        curves[mass] = (np.log10(t_arr), np.log10(T_arr))
    if len(curves) < 2:
        return None
    return compute_multimass_chi2(curves)


def main():
    print("=" * 70)
    print("MULTI-MASS ENVELOPE COMPARISON TEST")
    print("Running %d representative EoS at 4 masses with ETA=1.0" % len(SAMPLE_TAGS))
    print("=" * 70)

    # Filter to tags that actually exist
    existing_tags = []
    for tag in SAMPLE_TAGS:
        eos_file = os.path.join(NSCOOL_DIR, "EOS", "%s_CAT.dat" % tag)
        if os.path.isfile(eos_file):
            existing_tags.append(tag)
        else:
            print("  SKIP (no EoS file): %s" % tag)

    print("  Running %d EoS × 4 masses = %d NSCool runs" %
          (len(existing_tags), len(existing_tags) * 4))

    t0 = time.time()
    results = []

    for tag in existing_tags:
        curves_acc = {}
        all_ok = True

        for mass in MASSES:
            profile = find_tov_profile(tag, mass)
            if profile is None:
                continue

            # Check if already run
            out_dir = os.path.join(BASE, "Data", "Envelope_Test", "acc_multimass",
                                   tag, "M%d" % int(round(mass * 10)))
            teff_dst = os.path.join(out_dir, "Teff_%s.dat" % tag)

            if os.path.isfile(teff_dst):
                t_arr, T_arr = read_teff(teff_dst)
                if len(t_arr) >= 10:
                    curves_acc[mass] = (np.log10(t_arr), np.log10(T_arr))
                    continue

            os.makedirs(out_dir, exist_ok=True)
            write_cool_input(tag, profile)
            ok = run_nscool()
            if ok:
                teff_src = os.path.join(NSCOOL_DIR, "Model_1", "Teff_Try.dat")
                if os.path.isfile(teff_src):
                    shutil.copy2(teff_src, teff_dst)
                    t_arr, T_arr = read_teff(teff_dst)
                    if len(t_arr) >= 10:
                        curves_acc[mass] = (np.log10(t_arr), np.log10(T_arr))

        if len(curves_acc) < 2:
            print("  %s: too few masses, skipping" % tag)
            continue

        chi2_acc = compute_multimass_chi2(curves_acc)
        chi2_fe = load_existing_iron_chi2(tag)

        parts = tag.split("_")
        for i, p in enumerate(parts):
            if p.startswith("J"):
                model = "_".join(parts[:i])
                j = float(parts[i][1:].replace("p", "."))
                l = float(parts[i+1][1:].replace("p", "."))
                break

        results.append({
            "tag": tag, "model": model, "J": j, "L": l,
            "chi2_acc": chi2_acc, "chi2_fe": chi2_fe,
        })
        print("  %s: chi2_acc=%.3f, chi2_fe=%.3f, ratio=%.2f" %
              (tag, chi2_acc, chi2_fe if chi2_fe else 0,
               chi2_acc / chi2_fe if chi2_fe and chi2_fe > 0 else 0))

    elapsed = time.time() - t0
    print("\nDone in %.0f seconds" % elapsed)

    # Summary
    print("\n" + "=" * 70)
    print("MULTI-MASS RESULTS: Iron vs Accreted Envelope")
    print("=" * 70)
    print("\n  %-35s  %10s  %10s  %8s" % ("EoS", "chi2_Fe", "chi2_Acc", "Ratio"))
    print("  " + "-" * 70)
    for r in sorted(results, key=lambda x: x["chi2_fe"] or 999):
        chi2_fe = r["chi2_fe"] if r["chi2_fe"] else float('nan')
        ratio = r["chi2_acc"] / chi2_fe if chi2_fe and chi2_fe > 0 else float('nan')
        print("  %-35s  %10.3f  %10.3f  %8.2f" %
              (r["tag"], chi2_fe, r["chi2_acc"], ratio))

    if results:
        fe_vals = [r["chi2_fe"] for r in results if r["chi2_fe"]]
        acc_vals = [r["chi2_acc"] for r in results]
        if fe_vals:
            print("\n  Iron envelope:    min chi2/N = %.3f" % min(fe_vals))
            print("  Accreted envelope: min chi2/N = %.3f" % min(acc_vals))
            print("  Average ratio (acc/fe): %.2f" %
                  np.mean([r["chi2_acc"] / r["chi2_fe"]
                           for r in results if r["chi2_fe"] and r["chi2_fe"] > 0]))


if __name__ == "__main__":
    main()
