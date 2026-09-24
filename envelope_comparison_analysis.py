#!/usr/bin/env python3
"""
envelope_comparison_analysis.py — Compare iron vs accreted envelope chi-square results.

After running run_cooling_accreted.py, this script:
  1. Loads both iron (existing) and accreted envelope cooling curves
  2. Computes multi-mass chi2/N for both
  3. Compares the L_sym constraint under each envelope assumption
  4. Tests a mixed-envelope scenario (iron for isolated NSs, accreted for PSR J0437)
  5. Generates comparison figures

Usage:
  python3 envelope_comparison_analysis.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob
import os
from collections import defaultdict

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
COOL_DIR = os.path.join(BASE, "Data", "Cooling_Outputs")
EOSDIR = os.path.join(BASE, "NSCool", "EOS")
FIGDIR = os.path.join(BASE, "Figures")
os.makedirs(FIGDIR, exist_ok=True)

ALL_MODELS = ["GM1", "GM2", "FSUGarnet", "IOPB-I", "BigApple"]
MASSES = [1.0, 1.4, 1.8, 2.0]

OBS = [
    ("Cas A",       2.51, 6.26, 0.05, "isolated"),
    ("XMMU J1732",  3.48, 6.25, 0.10, "isolated"),
    ("RX J0822",    3.57, 6.24, 0.05, "isolated"),
    ("1E 1207",     3.85, 6.20, 0.10, "isolated"),
    ("PSR B0833",   4.05, 5.88, 0.10, "isolated"),
    ("PSR B1706",   4.24, 5.81, 0.10, "isolated"),
    ("PSR J0538",   4.48, 5.95, 0.10, "isolated"),
    ("PSR B2334",   4.61, 5.53, 0.15, "isolated"),
    ("PSR B0656",   5.05, 5.71, 0.10, "isolated"),
    ("Geminga",     5.53, 5.75, 0.10, "isolated"),
    ("RX J1856",    5.70, 5.70, 0.10, "isolated"),
    ("PSR B1055",   5.73, 5.59, 0.10, "isolated"),
    ("RX J0720",    5.78, 5.72, 0.10, "isolated"),
    ("PSR J2043",   5.88, 5.22, 0.15, "isolated"),
    ("PSR J0437",   6.85, 5.34, 0.15, "binary_recycled"),  # recycled MSP, likely accreted envelope
    ("PSR B0950",   7.24, 5.02, 0.20, "isolated"),
]

C_SQUARED_CGS = 8.9875e20


def parse_tag(tag):
    parts = tag.split("_")
    for i, p in enumerate(parts):
        if p.startswith("J") and "p" in p:
            model = "_".join(parts[:i])
            j = float(parts[i][1:].replace("p", "."))
            l = float(parts[i + 1][1:].replace("p", "."))
            return model, j, l
    raise ValueError("Cannot parse: %s" % tag)


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


def check_causality(eos_file):
    try:
        with open(eos_file) as f:
            lines = f.readlines()
        itext = int(lines[0].strip().split()[0])
        rho_list, P_list = [], []
        for line in lines[itext:]:
            stripped = line.strip()
            if not stripped:
                continue
            vals = stripped.split()
            try:
                rho_list.append(float(vals[0]))
                P_list.append(float(vals[1]))
            except (ValueError, IndexError):
                pass
        rho = np.array(rho_list)
        P = np.array(P_list)
    except Exception:
        return True, 0.0
    if len(rho) < 10:
        return True, 0.0
    idx = np.argsort(rho)
    rho, P = rho[idx], P[idx]
    core = rho > 1.5 * 2.8e14
    if core.sum() < 5:
        return True, 0.0
    eps = rho[core] * C_SQUARED_CGS
    dP = np.diff(P[core])
    deps = np.diff(eps)
    mask = deps > 0
    if mask.sum() == 0:
        return True, 0.0
    vs2 = dP[mask] / deps[mask]
    return float(vs2.max()) < 1.0, float(vs2.max())


def get_mmax_r14(mr_file):
    try:
        data = np.loadtxt(mr_file, comments="#")
        if data.ndim < 2 or len(data) < 5:
            return 0, 0
        masses = data[:, 0]
        radii = data[:, 1]
        immax = np.argmax(masses)
        mmax = masses[immax]
        stable_m = masses[:immax + 1]
        stable_r = radii[:immax + 1]
        r14 = np.interp(1.4, stable_m, stable_r) if mmax >= 1.4 else 0
        return mmax, r14
    except Exception:
        return 0, 0


def load_eos_data():
    eos_data = {}
    for model in ALL_MODELS:
        mr_files = sorted(glob.glob(os.path.join(
            EOSDIR, "MRcurve_%s_*_CAT.dat" % model)))
        for f in mr_files:
            tag = os.path.basename(f).replace("MRcurve_", "").replace("_CAT.dat", "")
            try:
                m, j, l = parse_tag(tag)
            except (IndexError, ValueError):
                continue
            if not (25.0 <= j <= 43.0 and 30.0 <= l <= 130.0):
                continue
            mmax, r14 = get_mmax_r14(f)
            cat_file = os.path.join(EOSDIR, "%s_CAT.dat" % tag)
            causal, vs2max = True, 0.0
            if os.path.isfile(cat_file):
                causal, vs2max = check_causality(cat_file)
            eos_data[tag] = {
                "model": m, "J": j, "L": l,
                "Mmax": mmax, "R14": r14,
                "causal": causal, "pass_all": mmax >= 2.0 and causal,
            }
    valid_tags = sorted(tag for tag, v in eos_data.items() if v["pass_all"])
    return eos_data, valid_tags


def load_curves(mode, envelope_type="fe"):
    """Load cooling curves for a given mode and envelope type.
    envelope_type: 'fe' for iron, 'acc' for accreted
    """
    mass_suffixes_fe = {1.0: "_M10", 1.4: "", 1.8: "_M18", 2.0: "_M20"}
    mass_suffixes_acc = {m: "_acc_M%d" % int(round(m * 10)) for m in MASSES}

    suffixes = mass_suffixes_acc if envelope_type == "acc" else mass_suffixes_fe
    curves = defaultdict(dict)

    for mass, suffix in suffixes.items():
        dir_name = mode + suffix if suffix else mode
        mode_dir = os.path.join(COOL_DIR, dir_name)
        if not os.path.isdir(mode_dir) and envelope_type == "fe" and mass == 1.4:
            dir_name = mode + "_M14"
            mode_dir = os.path.join(COOL_DIR, dir_name)
        if not os.path.isdir(mode_dir):
            continue
        for d in os.listdir(mode_dir):
            teff_file = os.path.join(mode_dir, d, "Teff_%s.dat" % d)
            if not os.path.isfile(teff_file):
                continue
            t_arr, T_arr = read_teff(teff_file)
            if len(t_arr) < 10:
                continue
            curves[d][mass] = (np.log10(t_arr), np.log10(T_arr))
    return curves


def compute_chi2(curves, eos_data, valid_tags, envelope_scenario="all_fe"):
    """Compute multi-mass chi2/N.

    envelope_scenario:
      'all_fe'   — iron envelope for all NSs (current approach)
      'all_acc'  — accreted envelope for all NSs
      'mixed'    — iron for isolated, accreted for PSR J0437
    """
    results = []
    for tag in valid_tags:
        if tag not in curves:
            continue
        available_masses = sorted(curves[tag].keys())
        if len(available_masses) < 2:
            continue

        chi2_total = 0.0
        n_compared = 0
        mass_assignments = {}

        for obs_name, log_age, log_Tobs, sigma, ns_type in OBS:
            best_chi2_i = 1e10
            best_mass_i = None

            for mass in available_masses:
                log_t_arr, log_T_arr = curves[tag][mass]
                T_pred = interp_T_at_age(log_t_arr, log_T_arr, log_age)
                if T_pred is None:
                    continue
                chi2_i = ((T_pred - log_Tobs) / sigma) ** 2
                if chi2_i < best_chi2_i:
                    best_chi2_i = chi2_i
                    best_mass_i = mass

            if best_mass_i is not None:
                chi2_total += best_chi2_i
                n_compared += 1
                mass_assignments[obs_name] = best_mass_i

        if n_compared < 10:
            continue

        chi2_per_n = chi2_total / n_compared
        v = eos_data[tag]
        results.append({
            "tag": tag, "model": v["model"], "J": v["J"], "L": v["L"],
            "Mmax": v["Mmax"], "R14": v["R14"],
            "chi2": chi2_per_n, "n_compared": n_compared,
            "mass_assignments": mass_assignments,
        })

    results.sort(key=lambda x: x["chi2"])
    return results


def compute_mixed_chi2(curves_fe, curves_acc, eos_data, valid_tags):
    """Mixed envelope: iron for 15 isolated NSs, accreted for PSR J0437."""
    results = []
    for tag in valid_tags:
        if tag not in curves_fe:
            continue
        available_masses_fe = sorted(curves_fe[tag].keys())
        available_masses_acc = sorted(curves_acc.get(tag, {}).keys())
        if len(available_masses_fe) < 2:
            continue

        chi2_total = 0.0
        n_compared = 0
        mass_assignments = {}

        for obs_name, log_age, log_Tobs, sigma, ns_type in OBS:
            best_chi2_i = 1e10
            best_mass_i = None

            if ns_type == "binary_recycled" and available_masses_acc:
                use_curves = curves_acc[tag]
                use_masses = available_masses_acc
            else:
                use_curves = curves_fe[tag]
                use_masses = available_masses_fe

            for mass in use_masses:
                if mass not in use_curves:
                    continue
                log_t_arr, log_T_arr = use_curves[mass]
                T_pred = interp_T_at_age(log_t_arr, log_T_arr, log_age)
                if T_pred is None:
                    continue
                chi2_i = ((T_pred - log_Tobs) / sigma) ** 2
                if chi2_i < best_chi2_i:
                    best_chi2_i = chi2_i
                    best_mass_i = mass

            if best_mass_i is not None:
                chi2_total += best_chi2_i
                n_compared += 1
                mass_assignments[obs_name] = best_mass_i

        if n_compared < 10:
            continue

        chi2_per_n = chi2_total / n_compared
        v = eos_data[tag]
        results.append({
            "tag": tag, "model": v["model"], "J": v["J"], "L": v["L"],
            "Mmax": v["Mmax"], "R14": v["R14"],
            "chi2": chi2_per_n, "n_compared": n_compared,
        })

    results.sort(key=lambda x: x["chi2"])
    return results


def bayesian_posterior(results, param="L"):
    """Compute Bayesian posterior P(param | data) by marginalizing."""
    if param == "L":
        vals = sorted(set(r["L"] for r in results))
    else:
        vals = sorted(set(r["J"] for r in results))

    posterior = {}
    for v in vals:
        weight = 0.0
        for r in results:
            rval = r["L"] if param == "L" else r["J"]
            if abs(rval - v) < 0.01:
                chi2_total = r["chi2"] * r["n_compared"]
                weight += np.exp(-chi2_total / 2.0)
        posterior[v] = weight

    total = sum(posterior.values())
    if total > 0:
        for v in posterior:
            posterior[v] /= total
    return posterior


def credible_interval(posterior, level=0.68):
    """Shortest credible interval at given level."""
    vals = sorted(posterior.keys())
    probs = np.array([posterior[v] for v in vals])
    dv = np.diff(vals)
    if len(dv) == 0:
        return vals[0], vals[0], vals[0]

    cdf = np.cumsum(probs)
    cdf /= cdf[-1]

    map_val = vals[np.argmax(probs)]

    best_width = 1e10
    best_lo, best_hi = vals[0], vals[-1]
    for i in range(len(vals)):
        for j in range(i, len(vals)):
            width = vals[j] - vals[i]
            prob_in = cdf[j] - (cdf[i - 1] if i > 0 else 0)
            if prob_in >= level and width < best_width:
                best_width = width
                best_lo, best_hi = vals[i], vals[j]
    return map_val, best_lo, best_hi


def main():
    sep = "=" * 80
    print(sep)
    print("ENVELOPE COMPARISON ANALYSIS")
    print("Iron (ETA=0) vs Accreted (ETA=1) vs Mixed")
    print(sep)

    eos_data, valid_tags = load_eos_data()
    print("Valid EoS: %d" % len(valid_tags))

    # Load iron envelope curves (existing)
    print("\nLoading iron envelope curves (durca_sf)...")
    curves_fe = load_curves("durca_sf", "fe")
    n_fe = sum(1 for t in valid_tags if t in curves_fe and len(curves_fe[t]) >= 2)
    print("  Iron: %d EoS with >= 2 masses" % n_fe)

    # Load accreted envelope curves
    print("Loading accreted envelope curves (durca_sf)...")
    curves_acc = load_curves("durca_sf", "acc")
    n_acc = sum(1 for t in valid_tags if t in curves_acc and len(curves_acc[t]) >= 2)
    print("  Accreted: %d EoS with >= 2 masses" % n_acc)

    if n_acc == 0:
        print("\n*** No accreted envelope curves found! ***")
        print("Run first: python3 Source_Code/run_cooling_accreted.py --modes durca_sf")
        print("\nProceeding with iron-only analysis and mixed-envelope placeholder...")

    # ================================================================
    # Chi-square analysis
    # ================================================================
    print("\n" + sep)
    print("CHI-SQUARE RESULTS")
    print(sep)

    # Iron envelope (existing)
    results_fe = compute_chi2(curves_fe, eos_data, valid_tags, "all_fe")
    if results_fe:
        chi2_min_fe = results_fe[0]["chi2"]
        best_fe = results_fe[0]
        print("\n--- IRON ENVELOPE (current, ETA=0) ---")
        print("  chi2_min/N = %.3f (%s, J=%.1f, L=%.0f)"
              % (chi2_min_fe, best_fe["model"], best_fe["J"], best_fe["L"]))
        d1_fe = [r for r in results_fe if r["chi2"] <= chi2_min_fe + 1.0]
        Ls_fe = [r["L"] for r in d1_fe]
        print("  1-sigma L_sym = [%.0f, %.0f] MeV (%d EoS)" %
              (min(Ls_fe), max(Ls_fe), len(d1_fe)))

        post_fe = bayesian_posterior(results_fe, "L")
        map_fe, lo_fe, hi_fe = credible_interval(post_fe, 0.68)
        print("  Bayesian 68%% CI: L = %.0f [%.0f, %.0f] MeV" % (map_fe, lo_fe, hi_fe))

    # Accreted envelope
    if n_acc > 0:
        results_acc = compute_chi2(curves_acc, eos_data, valid_tags, "all_acc")
        if results_acc:
            chi2_min_acc = results_acc[0]["chi2"]
            best_acc = results_acc[0]
            print("\n--- ACCRETED ENVELOPE (ETA=1) ---")
            print("  chi2_min/N = %.3f (%s, J=%.1f, L=%.0f)"
                  % (chi2_min_acc, best_acc["model"], best_acc["J"], best_acc["L"]))
            d1_acc = [r for r in results_acc if r["chi2"] <= chi2_min_acc + 1.0]
            Ls_acc = [r["L"] for r in d1_acc]
            print("  1-sigma L_sym = [%.0f, %.0f] MeV (%d EoS)" %
                  (min(Ls_acc), max(Ls_acc), len(d1_acc)))

            post_acc = bayesian_posterior(results_acc, "L")
            map_acc, lo_acc, hi_acc = credible_interval(post_acc, 0.68)
            print("  Bayesian 68%% CI: L = %.0f [%.0f, %.0f] MeV" % (map_acc, lo_acc, hi_acc))

        # Mixed envelope
        results_mix = compute_mixed_chi2(curves_fe, curves_acc, eos_data, valid_tags)
        if results_mix:
            chi2_min_mix = results_mix[0]["chi2"]
            best_mix = results_mix[0]
            print("\n--- MIXED ENVELOPE (Fe for isolated, Acc for PSR J0437) ---")
            print("  chi2_min/N = %.3f (%s, J=%.1f, L=%.0f)"
                  % (chi2_min_mix, best_mix["model"], best_mix["J"], best_mix["L"]))
            d1_mix = [r for r in results_mix if r["chi2"] <= chi2_min_mix + 1.0]
            Ls_mix = [r["L"] for r in d1_mix]
            print("  1-sigma L_sym = [%.0f, %.0f] MeV (%d EoS)" %
                  (min(Ls_mix), max(Ls_mix), len(d1_mix)))

        # ================================================================
        # Comparison figure
        # ================================================================
        print("\n--- Generating comparison figures ---")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        colors = {"GM1": "C0", "GM2": "C1", "FSUGarnet": "C2",
                  "IOPB-I": "C3", "BigApple": "C4"}

        datasets = [
            (results_fe, chi2_min_fe, "Iron envelope (ETA=0)", axes[0]),
            (results_acc, chi2_min_acc, "Accreted envelope (ETA=1)", axes[1]),
        ]
        if results_mix:
            datasets.append((results_mix, chi2_min_mix, "Mixed envelope", axes[2]))

        for res, chi2min, title, ax in datasets:
            for model in ALL_MODELS:
                ls_p = [r["L"] for r in res if r["model"] == model]
                chi2_p = [r["chi2"] for r in res if r["model"] == model]
                if ls_p:
                    ax.scatter(ls_p, chi2_p, c=colors[model], s=10, alpha=0.5, label=model)
            ax.axhline(y=chi2min + 1.0, color="green", ls="--", lw=1.5, alpha=0.7,
                       label=r"$\Delta\chi^2/N = 1$")
            ax.axhline(y=chi2min + 2.0, color="orange", ls="--", lw=1.5, alpha=0.7,
                       label=r"$\Delta\chi^2/N = 2$")
            ax.set_xlabel(r"$L_{\rm sym}$ [MeV]", fontsize=11)
            ax.set_ylabel(r"$\chi^2/N$", fontsize=11)
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=7, ncol=2)
            ax.set_ylim(0, 25)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Envelope Model Comparison: Iron vs Accreted", fontsize=13,
                     fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGDIR, "fig_envelope_comparison_chi2.png"), dpi=200)
        print("  Saved fig_envelope_comparison_chi2.png")

        # Bayesian posterior comparison
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        L_vals_fe = sorted(post_fe.keys())
        L_vals_acc = sorted(post_acc.keys())
        ax2.plot(L_vals_fe, [post_fe[v] for v in L_vals_fe], 'b-', lw=2.5,
                 label="Iron envelope (ETA=0)")
        ax2.plot(L_vals_acc, [post_acc[v] for v in L_vals_acc], 'r--', lw=2.5,
                 label="Accreted envelope (ETA=1)")
        ax2.axvline(map_fe, color='blue', ls=':', alpha=0.5)
        ax2.axvline(map_acc, color='red', ls=':', alpha=0.5)
        ax2.set_xlabel(r"$L_{\rm sym}$ [MeV]", fontsize=13)
        ax2.set_ylabel(r"$P(L_{\rm sym} | {\rm data})$", fontsize=13)
        ax2.set_title("Bayesian Posterior: Iron vs Accreted Envelope", fontsize=13)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(os.path.join(FIGDIR, "fig_envelope_posterior_comparison.png"), dpi=200)
        print("  Saved fig_envelope_posterior_comparison.png")

    else:
        print("\n*** Skipping accreted envelope comparison (no data yet) ***")
        print("  To generate accreted envelope cooling curves, run:")
        print("  cd %s" % BASE)
        print("  python3 Source_Code/run_cooling_accreted.py --modes durca_sf --masses 1.0 1.4 1.8 2.0")
        print("  Then re-run this script.")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + sep)
    print("SUMMARY: ENVELOPE SYSTEMATICS")
    print(sep)
    print("""
The 16 observed neutron stars in our sample:
  - 15 are isolated cooling NSs or radio pulsars -> iron envelope is appropriate
  - PSR J0437-4715 is a recycled millisecond pulsar in a binary with a WD companion
    It was accreted onto in the past -> likely has a light-element (accreted) envelope

Physical effect of envelope composition:
  - Accreted (light-element) envelope: higher thermal conductivity -> surface is HOTTER
    at the same internal temperature (by ~0.1-0.3 dex in log T_s)
  - This means the cooling curve shifts UP for accreted envelope
  - The shift affects the chi2 comparison with observations

Implications for the L_sym constraint:
  - If accreted envelope makes stars hotter, the model may better fit the hot young NSs
    (Cas A, RX J0822) but worse for the cold old NSs (PSR B2334, J2043)
  - The L_sym constraint should be relatively robust because:
    (a) The constraint comes from the RELATIVE ranking of EoS, not absolute T
    (b) The envelope shift is the same for all EoS at a given mass
    (c) Only 1 of 16 NSs (PSR J0437) clearly needs an accreted envelope
""")


if __name__ == "__main__":
    main()
