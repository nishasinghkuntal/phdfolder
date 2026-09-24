#!/usr/bin/env python3
"""
run_cooling_prl.py — Extended cooling runs for PRL paper.

Runs NSCool at multiple masses and with an alternative pairing model.
Output directory structure:
  Data/Cooling_Outputs/{mode}_M{mass*10}/{eos_tag}/Teff_{eos_tag}.dat

Modes:
  durca_only     : DUrca + MUrca, no SF (I_Pairing_0-0-0.dat)
  durca_sf       : DUrca + MUrca + SF (I_Pairing_SFB-a-T73.dat)
  durca_sf_altB  : DUrca + MUrca + SF (I_Pairing_SFB-b-T73.dat) — alternative 3P2 gap

Usage:
  python3 run_cooling_prl.py --masses 1.0 1.4 1.8 2.0 --modes durca_only durca_sf
  python3 run_cooling_prl.py --masses 1.4 --modes durca_sf_altB
  python3 run_cooling_prl.py --masses 1.0 1.8 2.0 --modes durca_only durca_sf --max 10  # test run
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
#  NSCOOL_DIR is the directory the solver actually runs in.  It is a
#  module-level global because --workdir rebinds it at startup.
#
#  WHY THIS EXISTS: NSCool writes its scratch files to Model_1/ with
#  fixed names (Cool_Try.in, Teff_Try.dat, ...).  Two jobs sharing one
#  NSCool directory therefore overwrite each other's inputs and outputs
#  silently -- this happened once in this project and corrupted a whole
#  batch.  --workdir points the run at a private worker directory whose
#  Model_1/ is real and whose large read-only trees (EOS, TOV, I_Files,
#  Code) are symlinks back to the master.  Workers are then safe to run
#  in parallel.  Use Source_Code/make_workers.py to build them.
NSCOOL_DIR = os.path.join(BASE, "NSCool")
EOS_DIR = os.path.join(NSCOOL_DIR, "EOS")
TOV_DIR = os.path.join(NSCOOL_DIR, "TOV")


def set_workdir(path):
    """Point every NSCool path at `path` instead of the master tree."""
    global NSCOOL_DIR, EOS_DIR, TOV_DIR
    NSCOOL_DIR = os.path.abspath(path)
    EOS_DIR = os.path.join(NSCOOL_DIR, "EOS")
    TOV_DIR = os.path.join(NSCOOL_DIR, "TOV")
    for need in ("NSCool.out", "EOS", "TOV", "I_Files", "Model_1"):
        if not os.path.exists(os.path.join(NSCOOL_DIR, need)):
            raise SystemExit(
                "workdir %s is missing %s -- build it with "
                "Source_Code/make_workers.py" % (NSCOOL_DIR, need))
COOL_OUT = os.path.join(BASE, "Data", "Cooling_Outputs")

PHYSICS_MODES = {
    "durca_only": {
        "pairing": "I_Files/I_Pairing_0-0-0.dat",
        "neutrino": "I_Files/I_Neutrino_1.dat",
        "conductivity": "I_Files/I_Conduct_32.dat",
        "heating": "I_Files/I_Heat_0.dat",
        "bfield": "I_Files/I_Bfield_0.dat",
        "accretion": "I_Files/I_Accretion_0.dat",
        "boundary": "I_Files/I_Bound_Fe.dat",
        "structure": "I_Files/I_Struct_1.6e14-4.0e11-1e10_normal.dat",
    },
    "durca_sf": {
        "pairing": "I_Files/I_Pairing_SFB-a-T73.dat",
        "neutrino": "I_Files/I_Neutrino_1.dat",
        "conductivity": "I_Files/I_Conduct_32.dat",
        "heating": "I_Files/I_Heat_0.dat",
        "bfield": "I_Files/I_Bfield_0.dat",
        "accretion": "I_Files/I_Accretion_0.dat",
        "boundary": "I_Files/I_Bound_Fe.dat",
        "structure": "I_Files/I_Struct_1.6e14-4.0e11-1e10_normal.dat",
    },
    "durca_sf_altB": {
        "pairing": "I_Files/I_Pairing_SFB-b-T73.dat",
        "neutrino": "I_Files/I_Neutrino_1.dat",
        "conductivity": "I_Files/I_Conduct_32.dat",
        "heating": "I_Files/I_Heat_0.dat",
        "bfield": "I_Files/I_Bfield_0.dat",
        "accretion": "I_Files/I_Accretion_0.dat",
        "boundary": "I_Files/I_Bound_Fe.dat",
        "structure": "I_Files/I_Struct_1.6e14-4.0e11-1e10_normal.dat",
    },
    "murca_sf": {
        "pairing": "I_Files/I_Pairing_SFB-a-T73.dat",
        "neutrino": "I_Files/I_Neutrino_no_durca.dat",
        "conductivity": "I_Files/I_Conduct_32.dat",
        "heating": "I_Files/I_Heat_0.dat",
        "bfield": "I_Files/I_Bfield_0.dat",
        "accretion": "I_Files/I_Accretion_0.dat",
        "boundary": "I_Files/I_Bound_Fe.dat",
        "structure": "I_Files/I_Struct_1.6e14-4.0e11-1e10_normal.dat",
    },
    #  Magnetized iron envelope, Potekhin & Yakovlev A&A 374, 213 (2001),
    #  selected by IFTEFF=4 in I_Bound_FeB.dat with a static B = 1e12 G
    #  (a typical young-pulsar dipole field) from I_Bfield_1e12.dat.
    #  NOTE: this mode only does anything because NSCool.f was patched to
    #  populate bf_r from BFIELD0 -- see the comment there. Before that
    #  patch IFTEFF=4 silently reproduced the field-free iron envelope.
    "durca_sf_magB12": {
        "pairing": "I_Files/I_Pairing_SFB-a-T73.dat",
        "neutrino": "I_Files/I_Neutrino_1.dat",
        "conductivity": "I_Files/I_Conduct_32.dat",
        "heating": "I_Files/I_Heat_0.dat",
        "bfield": "I_Files/I_Bfield_1e12.dat",
        "accretion": "I_Files/I_Accretion_0.dat",
        "boundary": "I_Files/I_Bound_FeB.dat",
        "structure": "I_Files/I_Struct_1.6e14-4.0e11-1e10_normal.dat",
    },
    #  Direct Urca + superfluidity WITH late-time vortex creep heating.
    #  This is the physics that was missing: without heating every model
    #  cools below the microphysics floor by ~2 Myr, so no curve reaches
    #  PSR J0437 (log t 6.85) or PSR B0950 (7.24) and both were scored
    #  against a clamped frozen temperature. With j_44 = 1.0 the curves
    #  run to log t ~ 9.7, past every star in the sample, so nothing is
    #  extrapolated and all 16 observations re-enter the fit.
    "durca_sf_vc": {
        "pairing": "I_Files/I_Pairing_SFB-a-T73.dat",
        "neutrino": "I_Files/I_Neutrino_1.dat",
        "conductivity": "I_Files/I_Conduct_32.dat",
        "heating": "I_Files/I_Heat_vc1.0.dat",
        "bfield": "I_Files/I_Bfield_0.dat",
        "accretion": "I_Files/I_Accretion_0.dat",
        "boundary": "I_Files/I_Bound_Fe.dat",
        "structure": "I_Files/I_Struct_1.6e14-4.0e11-1e10_normal.dat",
    },
    #  Light-element (carbon-proxy) envelope, Potekhin, Chabrier &
    #  Yakovlev (1999), IFTEFF=3. Scored ONLY against Cas A and
    #  XMMU J1732, both of which have published carbon atmospheres.
    "durca_sf_lightA": {
        "pairing": "I_Files/I_Pairing_SFB-a-T73.dat",
        "neutrino": "I_Files/I_Neutrino_1.dat",
        "conductivity": "I_Files/I_Conduct_32.dat",
        "heating": "I_Files/I_Heat_0.dat",
        "bfield": "I_Files/I_Bfield_0.dat",
        "accretion": "I_Files/I_Accretion_0.dat",
        "boundary": "I_Files/I_Bound_Eta_1e-12.dat",
        "structure": "I_Files/I_Struct_1.6e14-4.0e11-1e10_normal.dat",
    },
    #  Light-element (carbon-proxy) envelope, Potekhin, Chabrier &
    #  Yakovlev (1999), IFTEFF=3. Scored ONLY against Cas A and
    #  XMMU J1732, both of which have published carbon atmospheres.
    "durca_sf_lightB": {
        "pairing": "I_Files/I_Pairing_SFB-a-T73.dat",
        "neutrino": "I_Files/I_Neutrino_1.dat",
        "conductivity": "I_Files/I_Conduct_32.dat",
        "heating": "I_Files/I_Heat_0.dat",
        "bfield": "I_Files/I_Bfield_0.dat",
        "accretion": "I_Files/I_Accretion_0.dat",
        "boundary": "I_Files/I_Bound_Eta_1e-8.dat",
        "structure": "I_Files/I_Struct_1.6e14-4.0e11-1e10_normal.dat",
    },
    #  Carbon envelope, WITH vortex-creep heating so it is directly
    #  comparable to durca_sf_vc. eta = 6e-10 is not fitted to the data:
    #  it is calibrated so the Potekhin-Chabrier-Yakovlev envelope
    #  reproduces the +0.135 dex surface enhancement that the
    #  Hernquist-Applegate factor (A^3/Z^4)^(1/4) predicts for C-12 over
    #  Fe-56. Measured: +0.140 dex.
    #
    #  Scored ONLY against Cas A and XMMU J1732, the two objects with
    #  published carbon atmospheres (Ho & Heinke 2009; Klochkov et al.
    #  2013). The other 14 stars keep the iron envelope. Letting every
    #  star choose its own envelope would fit 16 parameters to 16 points.
    "durca_sf_vc_carbon": {
        "pairing": "I_Files/I_Pairing_SFB-a-T73.dat",
        "neutrino": "I_Files/I_Neutrino_1.dat",
        "conductivity": "I_Files/I_Conduct_32.dat",
        "heating": "I_Files/I_Heat_vc1.0.dat",
        "bfield": "I_Files/I_Bfield_0.dat",
        "accretion": "I_Files/I_Accretion_0.dat",
        "boundary": "I_Files/I_Bound_Carbon.dat",
        "structure": "I_Files/I_Struct_1.6e14-4.0e11-1e10_normal.dat",
    },
    #  Alternative 3P2 gap, now WITH heating so its curves reach the old
    #  stars like every other mode. Run over all 864 surviving EoS to
    #  remove the uneven-coverage bias: previously it covered only 191,
    #  which skewed the minimum-over-modes profile.
    "durca_sf_altB_vc": {
        "pairing": "I_Files/I_Pairing_SFB-b-T73.dat",
        "neutrino": "I_Files/I_Neutrino_1.dat",
        "conductivity": "I_Files/I_Conduct_32.dat",
        "heating": "I_Files/I_Heat_vc1.0.dat",
        "bfield": "I_Files/I_Bfield_0.dat",
        "accretion": "I_Files/I_Accretion_0.dat",
        "boundary": "I_Files/I_Bound_Fe.dat",
        "structure": "I_Files/I_Struct_1.6e14-4.0e11-1e10_normal.dat",
    },
    #  ------------------------------------------------------------------
    #  MAGNETIZED IRON ENVELOPES, for the third paper.
    #
    #  Same physics as durca_sf_vc (direct Urca + modified Urca +
    #  superfluidity + vortex creep heating), but with the envelope
    #  replaced by the magnetized iron relation of Potekhin & Yakovlev,
    #  A&A 374, 213 (2001), selected by IFTEFF=4 in I_Bound_FeB.dat and
    #  fed a static dipole field of magnitude BFIELD0.
    #
    #  THESE MODES ONLY DO ANYTHING BECAUSE OF THE bf_r PATCH IN
    #  NSCool.f.  fteff() passes bf_r(imax) to fteff_field_iron, and
    #  bf_r was never assigned anywhere in this build -- the only code
    #  that sets it lives in Bfield/Bfield_2.inc.f, which is not
    #  included.  It therefore arrived as an uninitialised common-block
    #  value of zero.  In fteff_field_iron, bfield = 0 gives beta = 0,
    #  hence ratio = 1 exactly, hence fteff = f_zero: the field-free
    #  iron envelope, bit for bit.  Before the patch, IFTEFF=4 silently
    #  returned the IFTEFF=3 answer and looked like "the field has no
    #  effect".
    #
    #  Four field strengths spanning the observed range, from ordinary
    #  young pulsars (1e11-1e12 G) to the X-ray dim isolated neutron
    #  stars (1e13 G) and up to magnetar-strength fields (1e14 G).
    #  The B = 0 baseline is durca_sf_vc itself.
    #  ------------------------------------------------------------------
    "durca_sf_vc_magB11": {
        "pairing": "I_Files/I_Pairing_SFB-a-T73.dat",
        "neutrino": "I_Files/I_Neutrino_1.dat",
        "conductivity": "I_Files/I_Conduct_32.dat",
        "heating": "I_Files/I_Heat_vc1.0.dat",
        "bfield": "I_Files/I_Bfield_1e11.dat",
        "accretion": "I_Files/I_Accretion_0.dat",
        "boundary": "I_Files/I_Bound_FeB.dat",
        "structure": "I_Files/I_Struct_1.6e14-4.0e11-1e10_normal.dat",
    },
    "durca_sf_vc_magB12": {
        "pairing": "I_Files/I_Pairing_SFB-a-T73.dat",
        "neutrino": "I_Files/I_Neutrino_1.dat",
        "conductivity": "I_Files/I_Conduct_32.dat",
        "heating": "I_Files/I_Heat_vc1.0.dat",
        "bfield": "I_Files/I_Bfield_1e12.dat",
        "accretion": "I_Files/I_Accretion_0.dat",
        "boundary": "I_Files/I_Bound_FeB.dat",
        "structure": "I_Files/I_Struct_1.6e14-4.0e11-1e10_normal.dat",
    },
    "durca_sf_vc_magB13": {
        "pairing": "I_Files/I_Pairing_SFB-a-T73.dat",
        "neutrino": "I_Files/I_Neutrino_1.dat",
        "conductivity": "I_Files/I_Conduct_32.dat",
        "heating": "I_Files/I_Heat_vc1.0.dat",
        "bfield": "I_Files/I_Bfield_1e13.dat",
        "accretion": "I_Files/I_Accretion_0.dat",
        "boundary": "I_Files/I_Bound_FeB.dat",
        "structure": "I_Files/I_Struct_1.6e14-4.0e11-1e10_normal.dat",
    },
    "durca_sf_vc_magB14": {
        "pairing": "I_Files/I_Pairing_SFB-a-T73.dat",
        "neutrino": "I_Files/I_Neutrino_1.dat",
        "conductivity": "I_Files/I_Conduct_32.dat",
        "heating": "I_Files/I_Heat_vc1.0.dat",
        "bfield": "I_Files/I_Bfield_1e14.dat",
        "accretion": "I_Files/I_Accretion_0.dat",
        "boundary": "I_Files/I_Bound_FeB.dat",
        "structure": "I_Files/I_Struct_1.6e14-4.0e11-1e10_normal.dat",
    },
    #  Control: IFTEFF=4 with B = 0.  Must be bit-for-bit identical to
    #  durca_sf_vc, which uses IFTEFF=3.  This is the test that proves
    #  the bug above, and it is run as part of the verification suite.
    "durca_sf_vc_magB0": {
        "pairing": "I_Files/I_Pairing_SFB-a-T73.dat",
        "neutrino": "I_Files/I_Neutrino_1.dat",
        "conductivity": "I_Files/I_Conduct_32.dat",
        "heating": "I_Files/I_Heat_vc1.0.dat",
        "bfield": "I_Files/I_Bfield_0.dat",
        "accretion": "I_Files/I_Accretion_0.dat",
        "boundary": "I_Files/I_Bound_FeB.dat",
        "structure": "I_Files/I_Struct_1.6e14-4.0e11-1e10_normal.dat",
    },
    "sf_no3p2": {
        "pairing": "I_Files/I_Pairing_SFB-0-T73.dat",
        "neutrino": "I_Files/I_Neutrino_1.dat",
        "conductivity": "I_Files/I_Conduct_32.dat",
        "heating": "I_Files/I_Heat_0.dat",
        "bfield": "I_Files/I_Bfield_0.dat",
        "accretion": "I_Files/I_Accretion_0.dat",
        "boundary": "I_Files/I_Bound_Fe.dat",
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
    # Fallback: find closest
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
    """Generate output directory name encoding both mode and mass."""
    if abs(mass - 1.4) < 0.01:
        return mode
    mass_tag = "M%d" % int(round(mass * 10))
    return "%s_%s" % (mode, mass_tag)


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
    parser.add_argument("--modes", nargs="+", default=["durca_only", "durca_sf"],
                        choices=list(PHYSICS_MODES.keys()))
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--eos", default=None)
    #  Restrict the run to an explicit list of EoS tags, one per line.
    #  The analysis only ever uses the 864 equations of state that pass
    #  the Mmax and causality cuts; iterating all 1478 tables on disk
    #  spends about 40 per cent of the compute on points that are then
    #  discarded. Regeneration after the 2026-09-19 data loss uses this.
    parser.add_argument("--tagfile", default=None)
    parser.add_argument("--workdir", default=None,
                        help="private NSCool worker directory (see "
                             "make_workers.py); required for parallel runs")
    parser.add_argument("--models", nargs="+",
                        default=["GM1", "GM2", "FSUGarnet", "IOPB-I", "BigApple"],
                        help="RMF model prefixes to include")
    args = parser.parse_args()
    if args.workdir:
        set_workdir(args.workdir)

    if args.eos:
        tags = [args.eos]
    elif args.tagfile:
        tags = [t.strip() for t in open(args.tagfile) if t.strip()]
    else:
        tags = get_allowed_tags()
    tags = [t for t in tags if any(t.startswith(m + "_") for m in args.models)]
    if args.max:
        tags = tags[:args.max]

    total_runs = len(tags) * len(args.masses) * len(args.modes)
    print("=" * 70)
    print("PRL Cooling Runs")
    print("  EoS: %d" % len(tags))
    print("  Masses: %s Msun" % ", ".join("%.1f" % m for m in args.masses))
    print("  Modes: %s" % ", ".join(args.modes))
    print("  Total runs: %d" % total_runs)
    print("=" * 70)

    t0 = time.time()
    counts = {"ok": 0, "skip": 0, "fail": 0, "no_profile": 0, "no_output": 0}
    run_idx = 0

    for mode in args.modes:
        for mass in args.masses:
            dir_name = output_dir_name(mode, mass)
            print("\n--- %s (M=%.1f) ---" % (mode, mass))
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
