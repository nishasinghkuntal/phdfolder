#!/bin/zsh
# =====================================================================
#  Run durca_sf_altB for ALL 864 surviving EoS, after magB12 finishes.
#
#  WHY: the combined analysis takes the MINIMUM chi-square over the five
#  physics modes. Four modes cover all 864 equations of state, but
#  durca_sf_altB covered only 191. A mode available for a fifth of the
#  grid biases that minimum -- those EoS get an extra chance to score
#  low, the rest do not. Running it everywhere removes the objection.
#
#  TRAP: run_cooling_prl.output_dir_name(mode, 1.4) returns
#  "durca_sf_altB", but the existing results live in
#  "durca_sf_altB_M14". full_multimass_5mode.load_curves_for_mode reads
#  "durca_sf_altB" FIRST and only falls back to the _M14 name when the
#  former is absent. So a fresh run silently SHADOWS every existing
#  result. We merge into _M14 and delete the shadow directory.
# =====================================================================
cd /Users/nishasingh/Documents/phdfolder

echo "[$(date)] waiting for magB12 to finish..."
while pgrep -f "run_cooling_prl.py --modes durca_sf_magB12" > /dev/null; do sleep 60; done
echo "[$(date)] magB12 done. starting altB over all 864."

python3 -u Source_Code/run_cooling_prl.py \
    --modes durca_sf_altB --masses 1.4 \
    --models GM1 GM2 FSUGarnet IOPB-I BigApple

if [ -d Data/Cooling_Outputs/durca_sf_altB ]; then
  echo "[$(date)] merging shadow dir into durca_sf_altB_M14"
  for d in Data/Cooling_Outputs/durca_sf_altB/*/; do
    mv "$d" Data/Cooling_Outputs/durca_sf_altB_M14/ 2>/dev/null
  done
  rmdir Data/Cooling_Outputs/durca_sf_altB 2>/dev/null
fi
if [ -d Data/Cooling_Outputs/durca_sf_altB ]; then
  echo "[$(date)] WARNING: shadow dir survived - analysis will misread altB"
else
  echo "[$(date)] shadow dir clear"
fi

echo "[$(date)] re-exporting chi-square"
cd Plotting_Scripts && python3 -u export_cooling_chi2.py
echo "[$(date)] ALL DONE"
