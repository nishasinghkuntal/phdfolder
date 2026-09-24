#!/bin/zsh
# =====================================================================
#  Carbon-envelope runs for Cas A and XMMU J1732, queued after heating.
#
#  WHY: both objects have published CARBON atmospheres -- Cas A
#  (Ho & Heinke 2009) and XMMU J173203.3-344518 (Klochkov et al. 2013,
#  arXiv:1307.1230). They are also the two largest contributors to the
#  chi-square, and in both cases the iron-envelope model is too COLD:
#  Cas A needs +0.195 dex, XMMU J1732 needs +0.247 dex. A light-element
#  envelope is hotter at the same interior temperature, which is the
#  right direction. Using iron for them is simply wrong physics, and
#  correcting it is justified by spectroscopy, not by the fit.
#
#  HOW: NSCool's IFTEFF=5 (Hernquist & Applegate 1984) does take an
#  arbitrary (Z, A), but it reads them from the outermost zone of the
#  EoS table rather than from an input file, so it cannot be switched to
#  carbon without rewriting all 864 tables. We therefore use IFTEFF=3,
#  the Potekhin, Chabrier & Yakovlev (1999) light-element envelope,
#  which is better physics than the Hernquist-Applegate analytic formula
#  and is the standard treatment in the cooling literature.
#
#  The eta values below are NOT fitted to the data. They span the
#  plausible light-element column, and the analysis reports the
#  sensitivity across them rather than selecting the best one. For
#  scale, Hernquist-Applegate gives carbon a +0.135 dex surface
#  enhancement over iron.
#
#  IMPORTANT: only Cas A and XMMU J1732 will be scored against these
#  curves. The other twelve stars keep the iron envelope. Letting every
#  star choose its own envelope would be fitting 16 parameters to 16
#  data points and would mean nothing.
# =====================================================================
cd /Users/nishasingh/Documents/phdfolder

echo "[$(date)] waiting for the heating run to finish..."
while pgrep -f "durca_sf_vc" > /dev/null; do sleep 120; done
echo "[$(date)] heating done. starting carbon-envelope runs."

for MODE in durca_sf_lightA durca_sf_lightB; do
  echo "[$(date)] --- $MODE ---"
  python3 -u Source_Code/run_cooling_prl.py \
      --modes $MODE --masses 1.0 1.4 1.8 2.0 \
      --models GM1 GM2 FSUGarnet IOPB-I BigApple
done
echo "[$(date)] ALL DONE"
