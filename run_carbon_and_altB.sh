#!/bin/zsh
cd /Users/nishasingh/Documents/phdfolder
echo "[$(date '+%H:%M')] ===== carbon envelope (heated) ====="
python3 -u Source_Code/run_cooling_prl.py --modes durca_sf_vc_carbon \
    --masses 1.0 1.4 1.8 2.0 --tagfile Data/surviving_864_tags.txt 2>&1 | tail -4
echo "[$(date '+%H:%M')] ===== altB over all 864 (heated) ====="
python3 -u Source_Code/run_cooling_prl.py --modes durca_sf_altB_vc \
    --masses 1.0 1.4 1.8 2.0 --tagfile Data/surviving_864_tags.txt 2>&1 | tail -4
echo "[$(date '+%H:%M')] BOTH COMPLETE"
