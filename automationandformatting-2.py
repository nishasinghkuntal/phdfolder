#!/usr/bin/env python3
"""
Build NSCool-compatible APR_EOS_Cat_*.dat files by replacing the core of
APR_EOS_Cat.dat with microphysical EOS tables (eos*.dat), keeping the
HZD-NV crust and matching the core–crust join.

Output names:
  BSR4.dat, FSUGarnet.dat, IOPB-I-2.dat, S271v6-2.dat
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# =============================================
# GLOBAL STORAGE FOR μ–P diagnostic plots
# =============================================
ALL_MU: dict[str, list[float]] = {}
ALL_P: dict[str, list[float]] = {}

# NSCool reference effective-mass ratio (APR_EOS_Cat at nuclear density)
REF_MSTP = 0.24542
REF_MSTN = 0.39065
MSTAR_MEV_THRESHOLD = 10.0
NUCLEON_MASS_MEV = 938.0

# Map eos base name → output suffix used by TOVprofile.f90
EOS_OUTPUT_ALIASES = {
    "IOPB-I": "IOPB-I-2",
    "S271v6": "S271v6-2",
}

# Where the finished table goes.  None means the default, NSCool/EOS/.
# Set from --outdir in the main block below.
OUTDIR_OVERRIDE = None

# Where to look for input EoS tables when none are named on the command
# line.  This used to be a hardcoded list of four files (eosBSR4.dat and
# friends).  Those live in Professor_EOS/, so running this script from
# anywhere else printed four "EOS file not found" failures and did
# nothing -- which looked like a broken script rather than an empty
# folder.  Now we look for what is actually there.
DEFAULT_SEARCH_DIRS = [
    "EOS_RMF",        # what rmf_lambda_and_eos.py writes
    ".",              # anything sitting beside this script
    "Professor_EOS",  # the four supplied tables
]


def discover_eos_files():
    """Return every eos*.dat we can find, in search-path order.

    Returns a list of paths.  Never raises: an empty list means the
    caller should explain what to do rather than fail four times.
    """
    found = []
    seen = set()
    i = 0
    while i < len(DEFAULT_SEARCH_DIRS):
        d = DEFAULT_SEARCH_DIRS[i]
        i = i + 1
        if not os.path.isdir(d):
            continue
        names = sorted(os.listdir(d))
        j = 0
        while j < len(names):
            n = names[j]
            j = j + 1
            if not n.startswith("eos") or not n.endswith(".dat"):
                continue
            path = os.path.normpath(os.path.join(d, n))
            # the same file reachable by two paths is still one file
            key = os.path.realpath(path)
            if key in seen:
                continue
            seen.add(key)
            found.append(path)
    return found

EOS_COLUMNS = {
    "rho": 0,
    "press": 1,
    "nbar": 2,
    "Ye": 3,
    "Ymu": 4,
    "Yn": 5,
    "Yp": 6,
    "m_eff": 7,
    "muB": 8,
}

APR_COLUMNS = {
    "Rho": 0,
    "Press": 1,
    "nbar": 2,
    "Ye": 3,
    "Ymu": 4,
    "Yn": 5,
    "Yp": 6,
    "Yla": 7,
    "Ysm": 8,
    "Ys0": 9,
    "Ysp": 10,
    "mstp": 11,
    "mstn": 12,
    "mstla": 13,
    "mstsm": 14,
    "msts0": 15,
    "mstsp": 16,
}

APR_TO_EOS_MAPPING = {
    "Rho": "rho",
    "Press": "press",
    "nbar": "nbar",
    "Ye": "Ye",
    "Ymu": "Ymu",
    "Yn": "Yn",
    "Yp": "Yp",
    "Yla": None,
    "Ysm": None,
    "Ys0": None,
    "Ysp": None,
    "mstp": None,
    "mstn": None,
    "mstla": None,
    "mstsm": None,
    "msts0": None,
    "mstsp": None,
}

# Reference core–crust join (APR_EOS_Cat.dat / NSCool HZD-NV crust)
JOIN_Z = 32
JOIN_A_CELL = 982
YE_REF_CORE = 3.1606e-2                  # reference APR last-core Ye
YE_TOL_GOOD = 0.005                      # |Ye_core - Ye_crust| like reference (~0.001)
PRESS_TOL_GOOD = 0.05                    # |dP|/P — primary join condition
RHO_TOL_SOFT = 0.20                      # |d rho|/rho — diagnostic only (jumps allowed)

# First crust row from reference APR_EOS_Cat.dat (only if join row missing)
EXTRA_CRUST_ROW = (
    "   1.334E+14        6.349E+32        7.890E-02       982     232     32\n"
)


def compute_mu(rho: float, press: float, nbar: float) -> float:
    """Chemical potential μ = (ε + P) / nB in MeV."""
    c2 = 1.7827e12
    c1 = 1.6022e33
    if nbar <= 0:
        return float("nan")
    eps = rho / c2
    return (eps + press / c1) / nbar


def column_m_eff_in_mev(m_eff_values: list[float]) -> bool:
    """
    Decide the units of the eos effective-mass column ONCE for the table.

    This routine is deliberately AGNOSTIC about which effective mass the
    column holds -- it only has to get the units right.  Tables written by
    rmf_lambda_and_eos.py carry the LANDAU mass sqrt(k_F^2 + m_dirac^2),
    which is what NSCool's Fermi-liquid formulas need.  Some older tables
    kept here (BSR4, IOPB-I, S271v6) instead carry the DIRAC mass, which
    collapses with density (e.g. BSR4: 765 MeV -> 2.35 MeV); those are
    legacy files and are not used by the current pipeline.
    Deciding units per-row with a fixed threshold is wrong: at high density
    M* legitimately drops below ~10 MeV, and a per-row test would suddenly
    treat those innermost core points as dimensionless ratios, producing an
    unphysical factor-~850 discontinuity in m*/m right at the stellar centre
    (which corrupts NSCool's Durca emissivity and specific heat there).

    Heuristic: if the column maximum is clearly MeV-scale (> threshold), the
    entire column is in MeV.
    """
    finite = [v for v in m_eff_values if v == v]
    if not finite:
        return False
    return max(finite) > MSTAR_MEV_THRESHOLD


def apr_effective_masses(m_eff: float, m_eff_in_mev: bool,
                         m_eff_p: float | None = None) -> tuple[float, float]:
    """
    Convert the eos effective-mass column to NSCool's mstp, mstn ratios.

    `m_eff_in_mev` is decided once per table by column_m_eff_in_mev() so the
    conversion is consistent across all rows (no centre discontinuity).

    Column 8 is the NEUTRON effective mass.  If the eos file also supplies a
    10th column it is the PROTON effective mass and is used directly.  RMF
    models give both Landau masses exactly and they do NOT share a fixed
    ratio, so using the real value matters: mstp enters the Durca emissivity
    and the proton heat capacity in NSCool.

    Files without a 10th column (the older BSR4/FSUGarnet/IOPB-I/S271v6
    tables) fall back to the APR proton/neutron ratio, as before.
    """
    if m_eff_in_mev:
        mstn = m_eff / NUCLEON_MASS_MEV
    else:
        mstn = m_eff

    if m_eff_p is None:
        mstp = mstn * (REF_MSTP / REF_MSTN)
    elif m_eff_in_mev:
        mstp = m_eff_p / NUCLEON_MASS_MEV
    else:
        mstp = m_eff_p
    return mstp, mstn


def parse_apr_metadata(first_line: str) -> tuple[int, int, int]:
    parts = first_line.split()
    if len(parts) < 3:
        raise ValueError("First line does not contain Itext Imax Icore metadata")
    return int(parts[0]), int(parts[1]), int(parts[2])


def read_apr_file(filename: str):
    with open(filename, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    itext, imax, icore = parse_apr_metadata(lines[0])
    header_end = itext + 1
    core_start = itext + 1 + 1
    core_end = itext + icore + 1
    crust_start = core_end + 1

    header = lines[0:header_end]
    doc_start = None
    for i in range(crust_start, len(lines)):
        if "CRUST:" in lines[i] or "CRUST-CORE" in lines[i]:
            doc_start = i
            break
    if doc_start is None:
        doc_start = min(crust_start + 65, len(lines))

    core_data = lines[core_start:core_end]
    crust_data = lines[crust_start:doc_start]
    documentation = lines[doc_start:]
    return header, core_data, crust_data, documentation


def read_eos_file(filename: str) -> list[str]:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"EOS file not found: {filename}")
    with open(filename, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return [
        line
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]


def format_apr_number(value: float) -> str:
    if abs(value) < 1e-99:
        return "0.0000E+00"
    s = f"{value:.4E}"
    if "E+" in s:
        parts = s.split("E+")
        return f"{parts[0]}E+{int(parts[1]):02d}"
    if "E-" in s:
        parts = s.split("E-")
        return f"{parts[0]}E-{int(parts[1]):02d}"
    return s


def convert_eos_to_apr_format(eos_lines: list[str]) -> list[str]:
    # Decide m_eff units once for the whole column (see column_m_eff_in_mev).
    mstar_col: list[float] = []
    for line in eos_lines:
        parts = line.split()
        if len(parts) >= 9:
            try:
                mstar_col.append(float(parts[EOS_COLUMNS["m_eff"]]))
            except ValueError:
                pass
    m_eff_in_mev = column_m_eff_in_mev(mstar_col)

    apr_data: list[tuple[float, str]] = []
    for line in eos_lines:
        parts = line.split()
        if len(parts) < 9:
            continue
        try:
            eos_values = {name: float(parts[idx]) for name, idx in EOS_COLUMNS.items()}
        except ValueError:
            continue

        # optional 10th column = proton effective mass (RMF tables supply it)
        m_eff_p = None
        if len(parts) >= 10:
            try:
                m_eff_p = float(parts[9])
            except ValueError:
                m_eff_p = None

        mstp, mstn = apr_effective_masses(eos_values["m_eff"], m_eff_in_mev,
                                          m_eff_p)
        apr_values: list[float] = []
        for apr_col_name in sorted(APR_COLUMNS.keys(), key=lambda x: APR_COLUMNS[x]):
            eos_col = APR_TO_EOS_MAPPING.get(apr_col_name)
            if apr_col_name == "mstp":
                apr_values.append(mstp)
            elif apr_col_name == "mstn":
                apr_values.append(mstn)
            else:
                apr_values.append(0.0 if eos_col is None else eos_values[eos_col])

        apr_line = " " + " ".join(format_apr_number(v) for v in apr_values) + "\n"
        apr_data.append((eos_values["rho"], apr_line))

    apr_data.sort(key=lambda x: x[0], reverse=True)
    return [line for _, line in apr_data]


def update_apr_metadata_line(new_icore: int, new_imax: int) -> str:
    itext = 6
    return f"      {itext:1d}    {new_imax:3d}    {new_icore:3d}          Itext Imax Icore\n"


def find_nscool_eos_dir() -> str | None:
    """
    Locate NSCool/EOS, looking beside this script and one level up.
    Returns None if NSCool is not installed here.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    for root in (here, parent, os.path.dirname(parent)):
        cand = os.path.join(root, "NSCool", "EOS")
        if os.path.isdir(cand):
            return cand
    return None


def find_apr_template(given: str) -> str:
    """
    Locate the crust template.

    The default is the bare name "APR_EOS_Cat.dat", which only works if
    you happen to be standing in a directory that has a copy.  The real
    one ships inside NSCool/EOS/, so look there too rather than failing
    with a bare "No such file or directory" that says nothing about
    where the file was expected to be.
    """
    if os.path.isfile(given):
        return given
    here = os.path.dirname(os.path.abspath(__file__))
    tries = [
        os.path.join(here, given),
        os.path.join(here, "NSCool", "EOS", "APR_EOS_Cat.dat"),
        os.path.join(os.path.dirname(here), "NSCool", "EOS",
                     "APR_EOS_Cat.dat"),
    ]
    eos_dir = find_nscool_eos_dir()
    if eos_dir is not None:
        tries.insert(0, os.path.join(eos_dir, "APR_EOS_Cat.dat"))
    i = 0
    while i < len(tries):
        if os.path.isfile(tries[i]):
            return tries[i]
        i = i + 1
    raise FileNotFoundError(
        "crust template %r not found.  Looked in the current directory, "
        "beside this script, and in NSCool/EOS/.  Pass one explicitly "
        "with --apr-template." % given)


def get_output_filename(eos_file: str, apr_file: str = "APR_EOS_Cat.dat",
                        outdir: str | None = None) -> str:
    """
    Build the output path.

    By default the file goes straight into NSCool/EOS/ named
    APR_EOS_Cat_<TAG>.dat, which is exactly what NSCool and TOVprofile
    expect, so nothing has to be copied or renamed afterwards.

    Pass --outdir to override.  The scan drivers use --outdir . because
    they run many models in parallel in private scratch directories and
    must not all write into the same shared folder.
    """
    base = os.path.splitext(os.path.basename(eos_file))[0]
    if base.startswith("eos"):
        base = base[3:]
    base = EOS_OUTPUT_ALIASES.get(base, base)
    if outdir is not None:
        return os.path.join(outdir, f"{base}_CAT.dat")
    name = f"{base}_CAT.dat"
    eos_dir = find_nscool_eos_dir()
    if eos_dir is None:
        return name
    return os.path.join(eos_dir, name)


def parse_crust_row(parts: list[str]) -> tuple[float, float, float, int, int, float] | None:
    """Return rho, press, nbar, A_cell, Z, Ye for a crust data row."""
    if len(parts) < 6 or not parts[0][0].isdigit():
        return None
    try:
        rho = float(parts[0].replace("D", "E"))
        press = float(parts[1].replace("D", "E"))
        nbar = float(parts[2].replace("D", "E"))
        a_cell = int(float(parts[3]))
        z = int(float(parts[5]))
    except ValueError:
        return None
    return rho, press, nbar, a_cell, z, z / a_cell


def find_crust_join_index(crust_lines: list[str]) -> int:
    """Index of the NSCool reference join row (Z=32, A_cell=982)."""
    for i, line in enumerate(crust_lines):
        parts = line.split()
        row = parse_crust_row(parts)
        if row is None:
            continue
        _, _, _, a_cell, z, _ = row
        if a_cell == JOIN_A_CELL and z == JOIN_Z:
            return i
    for i, line in enumerate(crust_lines):
        parts = line.split()
        if parse_crust_row(parts) is not None:
            return i
    return 0


def crust_already_has_extra_row(crust_lines: list[str]) -> bool:
    for line in crust_lines:
        parts = line.split()
        row = parse_crust_row(parts)
        if row is None:
            continue
        rho, _, _, a_cell, z, _ = row
        if a_cell == JOIN_A_CELL and z == JOIN_Z:
            return abs(rho - 1.334e14) < 1.0e12
    return False


def count_crust_rows(crust_lines: list[str]) -> int:
    n = 0
    for line in crust_lines:
        if parse_crust_row(line.split()) is not None:
            n += 1
    return n


def parse_core_row(line: str) -> tuple[float, float, float, float] | None:
    parts = line.split()
    if len(parts) < 4:
        return None
    try:
        return (
            float(parts[APR_COLUMNS["Rho"]].replace("D", "E")),
            float(parts[APR_COLUMNS["Press"]].replace("D", "E")),
            float(parts[APR_COLUMNS["nbar"]].replace("D", "E")),
            float(parts[APR_COLUMNS["Ye"]].replace("D", "E")),
        )
    except (ValueError, IndexError):
        return None


def _core_match_for_crust(new_core_lines, crust_row, max_core_skip):
    """
    Best achievable (dP/P, core index) for one candidate crust join row,
    subject to p_core >= p_crust and rho_core >= rho_crust.
    Returns None if no core row is compatible.
    """
    rho_crust, p_crust, nbar_crust, a_cell, z, ye_crust = crust_row
    best = None
    n_core = len(new_core_lines)
    for core_skip in range(0, min(max_core_skip, n_core - 1) + 1):
        idx_core = n_core - 1 - core_skip
        vals = parse_core_row(new_core_lines[idx_core])
        if vals is None:
            continue
        rho_core, p_core, nbar_core, ye_core = vals
        if p_core < p_crust or rho_core < rho_crust:
            continue
        press_rel = abs(p_core - p_crust) / max(abs(p_crust), 1.0)
        if best is None or press_rel < best[0]:
            best = (press_rel, idx_core)
    return best


def try_trim_match_core_crust(
    new_core_lines: list[str],
    crust_lines: list[str],
    max_core_skip: int = 300,
    max_crust_skip: int = 20,
):
    """
    Match core to the fixed NSCool crust join (Z=32, A_cell=982).

    Physics (core–crust interface):
      - Pressure must be continuous (mechanical equilibrium) — primary constraint.
      - Density must DECREASE from core to surface: the last core row is kept
        at rho >= rho_crust, so density only drops (a downward jump) at the join.
      - Ye_core should match crust Z/A ≈ 0.0326 when compatible with pressure.

    Selection:
      1. Among core rows with dP/P <= PRESS_TOL_GOOD, pick best Ye match.
      2. Otherwise minimize dP/P, then Ye, then mu.
    """
    c2 = 1.7827e12
    c1 = 1.6022e33

    idx_crust0 = find_crust_join_index(crust_lines)

    # The standard join row (Z=32, A_cell=982) assumes a core EoS of roughly
    # APR stiffness at n_b ~ 0.079 fm^-3.  A core with a stiff symmetry energy
    # is over-pressured there and no core row can match both the pressure and
    # the density ordering, giving joins with dP/P ~ 25%.  Physically the
    # crust-core transition moves to LOWER density as L increases, so we allow
    # the join to slide a few crust rows deeper and pick the best pressure
    # match.  For an APR-like core this loop stops immediately at idx_crust0
    # and the behaviour is unchanged.
    idx_crust = idx_crust0
    best_overall = None
    trial = 0
    while trial <= max_crust_skip and idx_crust0 + trial < len(crust_lines) - 1:
        cand = parse_crust_row(crust_lines[idx_crust0 + trial].split())
        if cand is not None:
            res = _core_match_for_crust(new_core_lines, cand, max_core_skip)
            if res is not None:
                press_rel_c, idx_c = res
                if best_overall is None or press_rel_c < best_overall[0]:
                    best_overall = (press_rel_c, idx_crust0 + trial)
                if press_rel_c <= PRESS_TOL_GOOD:
                    break
        trial = trial + 1

    if best_overall is not None:
        idx_crust = best_overall[1]

    crust_row = parse_crust_row(crust_lines[idx_crust].split())
    if crust_row is None:
        return new_core_lines, crust_lines, {"status": "no_crust_join"}

    rho_crust, p_crust, nbar_crust, a_cell, z, ye_crust = crust_row

    mu_crust = float("nan")
    if nbar_crust > 0:
        mu_crust = (rho_crust / c2 + p_crust / c1) / nbar_crust

    n_core = len(new_core_lines)
    if n_core == 0:
        return new_core_lines, crust_lines, {"status": "no_core"}

    best_press_ok: tuple[tuple[float, float, float], int] | None = None
    best_fallback: tuple[tuple[float, float, float], int] | None = None

    for core_skip in range(0, min(max_core_skip, n_core - 1) + 1):
        idx_core = n_core - 1 - core_skip
        core_vals = parse_core_row(new_core_lines[idx_core])
        if core_vals is None:
            continue

        rho_core, p_core, nbar_core, ye_core = core_vals
        if p_core < p_crust:
            continue
        # density must decrease from core to surface: keep the core only where
        # its density is >= the crust density, so it drops across the join
        if rho_core < rho_crust:
            continue

        mu_core = float("nan")
        if nbar_core > 0:
            mu_core = (rho_core / c2 + p_core / c1) / nbar_core

        press_rel = abs(p_core - p_crust) / max(abs(p_crust), 1.0)
        rho_rel = abs(rho_core - rho_crust) / max(abs(rho_crust), 1.0)
        ye_diff = abs(ye_core - ye_crust)

        mu_rel = 0.0
        if not np.isnan(mu_core) and not np.isnan(mu_crust) and abs(mu_core) > 0:
            mu_rel = abs(mu_core - mu_crust) / max(abs(mu_core), 1.0)

        fallback_key = (press_rel, ye_diff, mu_rel)
        if best_fallback is None or fallback_key < best_fallback[0]:
            best_fallback = (fallback_key, idx_core)

        if press_rel <= PRESS_TOL_GOOD:
            ye_key = (ye_diff, mu_rel, rho_rel)
            if best_press_ok is None or ye_key < best_press_ok[0]:
                best_press_ok = (ye_key, idx_core)

    if best_press_ok is not None:
        idx_core = best_press_ok[1]
        status = "matched"
    elif best_fallback is not None:
        idx_core = best_fallback[1]
        status = "best_found"
    else:
        return new_core_lines, crust_lines, {"status": "no_candidate"}

    core_vals = parse_core_row(new_core_lines[idx_core])
    assert core_vals is not None
    rho_core, p_core, _, ye_core = core_vals
    press_rel = abs(p_core - p_crust) / max(abs(p_crust), 1.0)
    rho_rel = abs(rho_core - rho_crust) / max(abs(rho_crust), 1.0)
    ye_diff = abs(ye_core - ye_crust)
    rho_jump = (rho_core - rho_crust) / max(abs(rho_crust), 1.0)

    mu_core = float("nan")
    nbar_core = core_vals[2]
    if nbar_core > 0:
        mu_core = (rho_core / c2 + p_core / c1) / nbar_core
    mu_rel = 0.0
    if not np.isnan(mu_core) and not np.isnan(mu_crust) and abs(mu_core) > 0:
        mu_rel = abs(mu_core - mu_crust) / max(abs(mu_core), 1.0)

    return (
        new_core_lines[: idx_core + 1],
        crust_lines[idx_crust:],
        {
            "status": status,
            "core_skip": n_core - 1 - idx_core,
            "crust_skip": idx_crust,
            "idx_core": idx_core,
            "idx_crust": idx_crust,
            "rho_core": rho_core,
            "rho_crust": rho_crust,
            "rho_rel": rho_rel,
            "rho_jump": rho_jump,
            "press_rel": press_rel,
            "p_core": p_core,
            "p_crust": p_crust,
            "ye_core": ye_core,
            "ye_crust": ye_crust,
            "ye_ref_core": YE_REF_CORE,
            "ye_diff": ye_diff,
            "mu_rel": mu_rel,
            "join_Z": z,
            "join_A": a_cell,
            "press_ok": press_rel <= PRESS_TOL_GOOD,
            "rho_jump_ok": abs(rho_jump) <= RHO_TOL_SOFT,
            "ye_ok": ye_diff <= YE_TOL_GOOD,
        },
    )


def print_join_report(match_diag: dict) -> None:
    if match_diag.get("status") not in ("matched", "best_found"):
        return
    print(
        f"  JOIN  Ye_core={match_diag['ye_core']:.6f}  "
        f"Ye_crust={match_diag['ye_crust']:.6f} "
        f"(Z={match_diag['join_Z']}/A={match_diag['join_A']})"
    )
    print(
        f"        rho_core={match_diag['rho_core']:.4e}  "
        f"rho_crust={match_diag['rho_crust']:.4e}  "
        f"drho/rho={match_diag['rho_rel']:.4f}  "
        f"(jump allowed)"
    )
    print(
        f"        P_core={match_diag['p_core']:.4e}  "
        f"P_crust={match_diag['p_crust']:.4e}  "
        f"dP/P={match_diag['press_rel']:.4f}"
    )
    checks = []
    if match_diag.get("press_ok"):
        checks.append("P OK")
    if match_diag.get("ye_ok"):
        checks.append("Ye OK")
    print(f"        dYe={match_diag['ye_diff']:.6f}  ref Ye_core={YE_REF_CORE:.6f}  [{', '.join(checks) or 'tolerances not all met'}]")


def main(eos_file: str, apr_file: str = "APR_EOS_Cat.dat", output_file: str | None = None):
    if output_file is None:
        output_file = get_output_filename(eos_file, apr_file, OUTDIR_OVERRIDE)

    header, _old_core, crust_data, documentation = read_apr_file(apr_file)
    eos_lines = read_eos_file(eos_file)
    new_core = convert_eos_to_apr_format(eos_lines)

    if not crust_already_has_extra_row(crust_data):
        crust_data = [EXTRA_CRUST_ROW] + crust_data

    chosen_core, chosen_crust, match_diag = try_trim_match_core_crust(
        new_core,
        crust_data,
        max_core_skip=300,
    )

    print(f"\n=== {eos_file} → {output_file} ===")
    print("Crust–core matching:", match_diag)
    print_join_report(match_diag)

    new_core = chosen_core
    crust_data = chosen_crust

    icore = len(new_core)
    ncrust = count_crust_rows(crust_data)
    imax = icore + ncrust
    header[0] = update_apr_metadata_line(icore, imax)

    mu_core_list: list[float] = []
    p_core_list: list[float] = []
    for line in new_core:
        parts = line.split()
        if len(parts) < 3:
            continue
        rho = float(parts[APR_COLUMNS["Rho"]])
        press = float(parts[APR_COLUMNS["Press"]])
        nbar = float(parts[APR_COLUMNS["nbar"]])
        mu_core_list.append(compute_mu(rho, press, nbar))
        p_core_list.append(press / 1.6022e33)

    mu_crust_list: list[float] = []
    p_crust_list: list[float] = []
    for line in crust_data:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            rho = float(parts[0])
            press = float(parts[1])
            nbar = float(parts[2])
        except ValueError:
            continue
        mu_crust_list.append(compute_mu(rho, press, nbar))
        p_crust_list.append(press / 1.6022e33)

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(header)
        f.writelines(new_core)
        f.writelines(crust_data)
        f.writelines(documentation)

    # Sanity check on effective masses at core–crust boundary
    if new_core:
        last = new_core[-1].split()
        print(
            f"  core rows={icore}  crust rows={ncrust}  imax={imax}  "
            f"join mstp={last[11]} mstn={last[12]}"
        )
    print(f"  written: {output_file}")

    ALL_MU[output_file] = mu_core_list + mu_crust_list
    ALL_P[output_file] = p_core_list + p_crust_list
    return output_file


def plot_all_mu_p(output_path: str = "mu_vs_P_all_eos.png") -> None:
    if not ALL_MU:
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    for label in sorted(ALL_MU):
        ax.plot(ALL_MU[label], ALL_P[label], linewidth=2, label=os.path.basename(label))
    ax.set_xlabel(r"Chemical potential $\mu$ (MeV)", fontsize=14)
    ax.set_ylabel(r"Pressure (MeV/fm$^3$)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Diagnostic plot saved: {output_path}")


def run_all(jobs: list[tuple[str, str | None]], apr_template: str, plot_file: str) -> int:
    failed = 0
    for eos_file, out_override in jobs:
        try:
            main(eos_file, apr_template, out_override)
        except Exception as exc:
            failed += 1
            print(f"FAILED {eos_file}: {exc}", file=sys.stderr)
    plot_all_mu_p(plot_file)
    return failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build APR_EOS_Cat_* files for NSCool/TOVprofile")
    parser.add_argument(
        "--apr-template",
        default="APR_EOS_Cat.dat",
        help="Reference APR file with crust (default: APR_EOS_Cat.dat)",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Write here using the plain <TAG>.dat name, instead of "
             "NSCool/EOS/APR_EOS_Cat_<TAG>.dat",
    )
    parser.add_argument(
        "--plot",
        default="mu_vs_P_all_eos.png",
        help="Save combined mu-P diagnostic plot here",
    )
    parser.add_argument(
        "eos_files",
        nargs="*",
        help="Input eos*.dat files (default: all four standard models)",
    )
    args = parser.parse_args()
    OUTDIR_OVERRIDE = args.outdir   # noqa: F811  (module-level default above)

    if args.eos_files:
        jobs = [(f, None) for f in args.eos_files]
    else:
        discovered = discover_eos_files()
        if not discovered:
            print("No EoS tables to work on.", file=sys.stderr)
            print("", file=sys.stderr)
            print("I looked for eos*.dat in: %s"
                  % ", ".join(DEFAULT_SEARCH_DIRS), file=sys.stderr)
            print("", file=sys.stderr)
            print("Generate some first:", file=sys.stderr)
            print("    python3 rmf_lambda_and_eos.py            "
                  "# the scan tables", file=sys.stderr)
            print("    python3 rmf_lambda_and_eos.py --checks   "
                  "# the GM and NL3wr checks", file=sys.stderr)
            print("", file=sys.stderr)
            print("or name one directly:", file=sys.stderr)
            print("    python3 %s EOS_RMF/eosGM1_J33p4_L69p0.dat"
                  % os.path.basename(__file__), file=sys.stderr)
            sys.exit(1)
        print("No files named; found %d in %s"
              % (len(discovered), os.path.dirname(discovered[0]) or "."))
        jobs = [(f, None) for f in discovered]

    try:
        apr_template = find_apr_template(args.apr_template)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    n_fail = run_all(jobs, apr_template, args.plot)
    sys.exit(1 if n_fail else 0)
