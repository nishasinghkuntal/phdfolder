#!/usr/bin/env python3
"""
run_cooling_accreted.py — Run cooling curves with accreted (light-element) envelope.

Same grid as run_cooling_prl.py but uses I_Bound_Acc.dat (ETA=1.0) instead of
I_Bound_Fe.dat (ETA=0.0). This tests the sensitivity of the L_sym constraint
to the envelope composition assumption.

Output directory structure:
  Data/Cooling_Outputs/{mode}_acc_M{mass*10}/{eos_tag}/Teff_{eos_tag}.dat

The "_acc" tag distinguishes accreted-envelope runs from iron-envelope runs.

Usage:
  python3 run_cooling_accreted.py --masses 1.0 1.4 1.8 2.0 --modes durca_sf
  python3 run_cooling_accreted.py --max 10  # test run with 10 EoS
"""
import os
import sys
import subprocess
import argparse
import shutil
import glob
import time
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NSCOOL_DIR = os.path.join(BASE, "NSCool")
EOS_DIR = os.path.join(NSCOOL_DIR, "EOS")
TOV_DIR = os.path.join(NSCOOL_DIR, "TOV")
COOL_OUT = os.path.join(BASE, "Data", "Cooling_Outputs")

PHYSICS_MODES = {
    "durca_sf": {
        "pairing": "I_Files/I_Pairing_SFB-a-T73.dat",
        "neutrino": "I_Files/I_Neutrino_1.dat",
        "conductivity": "I_Files/I_Conduct_32.dat",
        "heating": "I_Files/I_Heat_0.dat",
        "bfield": "I_Files/I_Bfield_0.dat",
        "accretion": "I_Files/I_Accretion_0.dat",
        "boundary": "I_Files/I_Bound_Acc.dat",
        "structure": "I_Files/I_Struct_1.6e14-4.0e11-1e10_normal.dat",
    },
    "durca_only": {
        "pairing": "I_Files/I_Pairing_0-0-0.dat",
        "neutrino": "I_Files/I_Neutrino_1.dat",
        "conductivity": "I_Files/I_Conduct_32.dat",
        "heating": "I_Files/I_Heat_0.dat",
        "bfield": "I_Files/I_Bfield_0.dat",
        "accretion": "I_Files/I_Accretion_0.dat",
        "boundary": "I_Files/I_Bound_Acc.dat",
        "structure": "I_Files/I_Struct_1.6e14-4.0e11-1e10_normal.dat",
    },
    "murca_sf": {
        "pairing": "I_Files/I_Pairing_SFB-a-T73.dat",
        "neutrino": "I_Files/I_Neutrino_no_durca.dat",
        "conductivity": "I_Files/I_Conduct_32.dat",
        "heating": "I_Files/I_Heat_0.dat",
        "bfield": "I_Files/I_Bfield_0.dat",
        "accretion": "I_Files/I_Accretion_0.dat",
        "boundary": "I_Files/I_Bound_Acc.dat",
        "structure": "I_Files/I_Struct_1.6e14-4.0e11-1e10_normal.dat",
    },
}


def write_cool_input(eos_tag, tov_profile_path, mode):
    phys = PHYSICS_MODES[mode]
    eos_file = "EOS/%s_CAT.dat" % eos_tag
    tov_rel = os.path.relpath(tov_profile_path, NSCOOL_DIR)
    content = """  'NEW'
BASIC MODEL FILES:
  'EOS/Crust/Crust_EOS_Cat_HZD-NV.dat'
  '%s'
  '%s'
OTHER MODEL FILES:
  '%s'
  '%s'
  '%s'
  '%s'
  '%s'
  '%s'
  '%s'
  '%s'
OUTPUT FILES:
  'Model_1/I.dat'
  'Model_1/Teff_Try.dat'
  'Model_1/Temp_Try.dat'
  'Model_1/Star_Try.dat'


""" % (eos_file, tov_rel,
       phys["structure"], phys["boundary"], phys["pairing"],
       phys["neutrino"], phys["conductivity"], phys["heating"],
       phys["bfield"], phys["accretion"])
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
    pat = os.path.join(prof_dir, "TOVprofile_%s_CAT_targetM*.dat" % eos_tag)
    profiles = sorted(glob.glob(pat))
    if not profiles:
        return None
    best, best_diff = None, 999.0
    for p in profiles:
        try:
            m_str = os.path.basename(p).split("targetM")[1].replace(".dat", "")
            if m_str == "max":
                continue
            m_val = int(m_str) / 10.0
        except (IndexError, ValueError):
            continue
        diff = abs(m_val - target_mass)
        if diff < best_diff:
            best_diff = diff
            best = p
    return best


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


def output_dir_name(mode, mass):
    mass_tag = "M%d" % int(round(mass * 10))
    return "%s_acc_%s" % (mode, mass_tag)


def run_one(eos_tag, mode, target_mass):
    dir_name = output_dir_name(mode, target_mass)
    out_dir = os.path.join(COOL_OUT, dir_name, eos_tag)
    teff_dst = os.path.join(out_dir, "Teff_%s.dat" % eos_tag)

    if os.path.isfile(teff_dst):
        return "skip"

    os.makedirs(out_dir, exist_ok=True)
    profile = find_tov_profile(eos_tag, target_mass)
    if profile is None:
        return "no_profile"

    write_cool_input(eos_tag, profile, mode)
    ok = run_nscool()
    if ok:
        teff_src = os.path.join(NSCOOL_DIR, "Model_1", "Teff_Try.dat")
        if os.path.exists(teff_src):
            shutil.copy2(teff_src, teff_dst)
            return "ok"
        return "no_output"
    return "fail"


def get_allowed_tags():
    pattern = os.path.join(EOS_DIR, "*_CAT.dat")
    tags = []
    for f in sorted(glob.glob(pattern)):
        base = os.path.basename(f)
        tag = base.replace("_CAT.dat", "")
        if "_J" in tag and "_L" in tag:
            tags.append(tag)
    return tags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--masses", type=float, nargs="+", default=[1.0, 1.4, 1.8, 2.0])
    parser.add_argument("--modes", nargs="+", default=["durca_sf"],
                        choices=list(PHYSICS_MODES.keys()))
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--models", nargs="+",
                        default=["GM1", "GM2", "FSUGarnet", "IOPB-I", "BigApple"])
    args = parser.parse_args()

    tags = get_allowed_tags()
    tags = [t for t in tags if any(t.startswith(m + "_") for m in args.models)]
    if args.max:
        tags = tags[:args.max]

    total_runs = len(tags) * len(args.masses) * len(args.modes)
    print("=" * 70)
    print("ACCRETED ENVELOPE Cooling Runs (ETA=1.0)")
    print("  EoS: %d" % len(tags))
    print("  Masses: %s Msun" % ", ".join("%.1f" % m for m in args.masses))
    print("  Modes: %s" % ", ".join(args.modes))
    print("  Total runs: %d" % total_runs)
    print("  Estimated time: %.0f min" % (total_runs * 5.5 / 60))
    print("=" * 70)

    t0 = time.time()
    counts = {"ok": 0, "skip": 0, "fail": 0, "no_profile": 0, "no_output": 0}
    run_idx = 0

    for mode in args.modes:
        for mass in args.masses:
            dir_name = output_dir_name(mode, mass)
            print("\n--- %s (M=%.1f, ACCRETED) ---" % (mode, mass))
            for i, tag in enumerate(tags):
                run_idx += 1
                status = run_one(tag, mode, mass)
                counts[status] += 1
                if status != "skip":
                    elapsed = time.time() - t0
                    rate = run_idx / elapsed if elapsed > 0 else 0
                    eta = (total_runs - run_idx) / rate if rate > 0 else 0
                    if (i + 1) % 50 == 0 or status != "ok":
                        print("  [%d/%d] %s: %s (%.0fs elapsed, ETA %.0fs)" %
                              (run_idx, total_runs, tag, status, elapsed, eta))

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("DONE in %.1f seconds (%.1f min)" % (elapsed, elapsed / 60))
    for k, v in counts.items():
        if v > 0:
            print("  %s: %d" % (k, v))
    print("=" * 70)


if __name__ == "__main__":
    main()
