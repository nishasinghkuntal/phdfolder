#!/bin/zsh
# =====================================================================
#  Regenerate the cooling datasets lost on 2026-09-19 21:02.
#
#  Restricted to the 864 equations of state that pass the Mmax >= 2 and
#  causality cuts (Data/surviving_864_tags.txt). The other ~614 tables on
#  disk lie outside the analysis J/L window and are discarded by
#  full_multimass_5mode anyway, so running them would spend ~40% of the
#  compute on points that never reach a result.
#
#  Order is by value: the primary mode first, so a usable analysis exists
#  after the first block rather than only at the very end.
#
#  After EACH mode the chi-square is exported and copied OUTSIDE the
#  project tree. The raw Teff curves are bulky and regenerable; the
#  chi-square tables are what every result depends on, and last time they
#  were lost together with the curves. A second wipe should cost compute,
#  not work.
#
#  ONE job at a time: two concurrent NSCool runs share Model_1/ and
#  corrupt each other. That already happened once this session.
# =====================================================================
cd /Users/nishasingh/Documents/phdfolder
SC="/private/tmp/claude-501/-Users-nishasingh-Documents-phdfolder/32af815d-256b-4aef-a646-1ffbdf61b753/scratchpad"
TAGS=Data/surviving_864_tags.txt

run_mode () {
  local mode=$1; shift
  echo "[$(date '+%m-%d %H:%M')] ===== $mode  masses: $* ====="
  python3 -u Source_Code/run_cooling_prl.py --modes $mode --masses "$@" \
      --tagfile $TAGS 2>&1 | tail -4
  echo "[$(date '+%m-%d %H:%M')] $mode complete -- exporting chi2"
  ( cd Plotting_Scripts && python3 -u export_cooling_chi2.py 2>&1 | tail -7 )
  cp -p Data/cooling_chi2_by_eos.csv "$SC/regen_logs/chi2_after_$mode.csv" 2>/dev/null \
    && echo "[$(date '+%m-%d %H:%M')] chi2 snapshot saved outside project"
}

run_mode durca_sf      1.0 1.4 1.8 2.0
run_mode durca_only    1.0 1.4 1.8 2.0
run_mode sf_no3p2      1.0 1.4 1.8 2.0
run_mode murca_sf      1.0 1.4 1.8 2.0
run_mode durca_sf_altB 1.4
run_mode durca_sf_vc   1.0 1.4 1.8 2.0
echo "[$(date '+%m-%d %H:%M')] ALL REGENERATION COMPLETE"
