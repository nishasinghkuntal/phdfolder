#!/usr/bin/env python3
"""
test_envelope_effect.py — Quick test of envelope model effect on cooling curves.

Runs the best-fit EoS (BigApple_J25p0_L50p0) with different envelope eta values
to see how the surface temperature changes and how chi2 is affected.

ETA parameter (Potekhin, Chabrier & Yakovlev 1999):
  ETA = 0   : pure iron envelope (heavy elements, current default)
  ETA = 0.1 : small amount of accreted light elements
  ETA = 0.5 : half accreted
  ETA = 1.0 : fully accreted (pure light elements H/He)

Light-element envelopes are more transparent -> surface is HOTTER at same T_b.
"""
import os
import sys
import subprocess
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NSCOOL_DIR = os.path.join(BASE, "NSCool")
FIGDIR = os.path.join(BASE, "Figures")
os.makedirs(FIGDIR, exist_ok=True)

TEST_TAG = "BigApple_J25p0_L50p0"
TEST_MASS = 14  # M = 1.4 Msun

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

ENVELOPE_CONFIGS = {
    "eta0.0": "I_Files/I_Bound_Fe.dat",
    "eta0.1": "I_Files/I_Bound_Eta0p1.dat",
    "eta0.5": "I_Files/I_Bound_Eta0p5.dat",
    "eta1.0": "I_Files/I_Bound_Acc.dat",
}


def write_cool_input(tov_profile_path, boundary_file):
    eos_file = "EOS/%s_CAT.dat" % TEST_TAG
    tov_rel = os.path.relpath(tov_profile_path, NSCOOL_DIR)
    content = """  'NEW'
BASIC MODEL FILES:
  'EOS/Crust/Crust_EOS_Cat_HZD-NV.dat'
  '%s'
  '%s'
OTHER MODEL FILES:
  'I_Files/I_Struct_1.6e14-4.0e11-1e10_normal.dat'
  '%s'
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


""" % (eos_file, tov_rel, boundary_file)
    inpath = os.path.join(NSCOOL_DIR, "Model_1", "Cool_Try.in")
    with open(inpath, "w") as f:
        f.write(content)


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


def compute_chi2(log_t, log_T):
    chi2 = 0.0
    n = 0
    for name, log_age, log_Tobs, sigma in OBS:
        T_pred = interp_T_at_age(log_t, log_T, log_age)
        if T_pred is None:
            continue
        chi2 += ((T_pred - log_Tobs) / sigma) ** 2
        n += 1
    return chi2 / n if n > 0 else None, n


def main():
    print("=" * 70)
    print("ENVELOPE MODEL EFFECT TEST")
    print("EoS: %s, M = %.1f Msun, mode = durca_sf" % (TEST_TAG, TEST_MASS / 10))
    print("=" * 70)

    tov_profile = os.path.join(
        NSCOOL_DIR, "TOV", "Profile",
        "TOVprofile_%s_CAT_targetM%d.dat" % (TEST_TAG, TEST_MASS))

    if not os.path.isfile(tov_profile):
        print("ERROR: TOV profile not found: %s" % tov_profile)
        sys.exit(1)

    all_curves = {}
    chi2_results = {}

    for label, boundary_file in sorted(ENVELOPE_CONFIGS.items()):
        bf_path = os.path.join(NSCOOL_DIR, boundary_file)
        if not os.path.isfile(bf_path):
            print("  SKIP %s: file not found" % label)
            continue

        print("\nRunning %s (%s)..." % (label, boundary_file))
        write_cool_input(tov_profile, boundary_file)
        ok = run_nscool()

        if not ok:
            print("  FAILED!")
            continue

        teff_file = os.path.join(NSCOOL_DIR, "Model_1", "Teff_Try.dat")
        t_arr, T_arr = read_teff(teff_file)
        log_t = np.log10(t_arr)
        log_T = np.log10(T_arr)
        all_curves[label] = (log_t, log_T)

        chi2, n = compute_chi2(log_t, log_T)
        chi2_results[label] = chi2
        print("  chi2/N = %.3f (N=%d)" % (chi2, n))

        # Save output
        out_dir = os.path.join(BASE, "Data", "Envelope_Test")
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy2(teff_file, os.path.join(out_dir, "Teff_%s_%s.dat" % (TEST_TAG, label)))

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("RESULTS: Effect of envelope ETA on chi2/N")
    print("=" * 70)
    print("\n  %-10s  %10s  %10s" % ("Envelope", "chi2/N", "Delta"))
    print("  " + "-" * 35)
    chi2_fe = chi2_results.get("eta0.0")
    for label in sorted(chi2_results.keys()):
        delta = chi2_results[label] - chi2_fe if chi2_fe else 0
        print("  %-10s  %10.3f  %+10.3f" % (label, chi2_results[label], delta))

    # ================================================================
    # Comparison figure
    # ================================================================
    if all_curves:
        print("\nGenerating comparison figure...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        colors_eta = {"eta0.0": "navy", "eta0.1": "blue",
                      "eta0.5": "green", "eta1.0": "red"}
        labels_eta = {"eta0.0": r"$\eta=0$ (iron)",
                      "eta0.1": r"$\eta=0.1$ (slightly accreted)",
                      "eta0.5": r"$\eta=0.5$ (half accreted)",
                      "eta1.0": r"$\eta=1.0$ (fully accreted, H/He)"}

        for label in sorted(all_curves.keys()):
            log_t, log_T = all_curves[label]
            c = colors_eta.get(label, "gray")
            lbl = labels_eta.get(label, label)
            chi2 = chi2_results[label]
            ax1.plot(log_t, log_T, '-', color=c, lw=2,
                     label=r"%s ($\chi^2/N=%.2f$)" % (lbl, chi2))

        # Plot observations
        for name, la, lt, sigma in OBS:
            ax1.errorbar(la, lt, yerr=sigma, fmt='ko', ms=5, capsize=3, alpha=0.6)

        ax1.set_xlabel(r"$\log_{10}(t/{\rm yr})$", fontsize=13)
        ax1.set_ylabel(r"$\log_{10}(T_s^\infty/{\rm K})$", fontsize=13)
        ax1.set_title("BigApple J=25 L=50, M=1.4: Envelope Comparison", fontsize=11)
        ax1.legend(fontsize=9, loc="lower left")
        ax1.set_xlim(1.5, 7.5)
        ax1.set_ylim(4.8, 6.5)
        ax1.grid(True, alpha=0.3)

        # Temperature difference plot
        if "eta0.0" in all_curves:
            ref_t, ref_T = all_curves["eta0.0"]
            for label in sorted(all_curves.keys()):
                if label == "eta0.0":
                    continue
                log_t, log_T = all_curves[label]
                common_t = np.linspace(max(ref_t.min(), log_t.min()),
                                       min(ref_t.max(), log_t.max()), 500)
                T_ref = np.interp(common_t, ref_t, ref_T)
                T_this = np.interp(common_t, log_t, log_T)
                diff = T_this - T_ref
                c = colors_eta.get(label, "gray")
                lbl = labels_eta.get(label, label)
                ax2.plot(common_t, diff, '-', color=c, lw=2, label=lbl)

            ax2.axhline(0, color='gray', ls='-', lw=0.5)
            ax2.set_xlabel(r"$\log_{10}(t/{\rm yr})$", fontsize=13)
            ax2.set_ylabel(r"$\Delta\log_{10} T_s$ (accreted $-$ iron)", fontsize=13)
            ax2.set_title("Temperature Shift from Light-Element Envelope", fontsize=11)
            ax2.legend(fontsize=9)
            ax2.set_xlim(1.5, 7.5)
            ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(FIGDIR, "fig_envelope_eta_comparison.png"), dpi=200)
        print("  Saved fig_envelope_eta_comparison.png")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
The Potekhin et al. (1999) envelope model uses parameter ETA:
  ETA = 0   : pure iron (Fe) envelope — heaviest, least transparent
  ETA = 1   : fully accreted light elements (H/He) — lightest, most transparent

Light-element envelopes have higher thermal conductivity, so:
  T_surface(accreted) > T_surface(iron)  at the same internal temperature

This means with a light-element envelope:
  - Cooling curves shift UP (hotter surface for same core temperature)
  - Young hot NSs (Cas A, RX J0822) may be better fit
  - Cold old NSs (PSR B2334, J2043) may be harder to fit

The key question: does the L_sym constraint change significantly?
If chi2_min changes but the RELATIVE ranking of EoS is preserved,
then the constraint is robust to envelope composition.
""")


if __name__ == "__main__":
    main()
