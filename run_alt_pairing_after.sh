#!/bin/bash
# Wait for current run_cooling_prl.py to finish
echo "Waiting for multi-mass runs to complete..."
while pgrep -f "run_cooling_prl.py" > /dev/null 2>&1; do
    sleep 30
done
echo "Multi-mass runs done. Starting alternative pairing..."

cd /Users/nishasingh/Documents/phdfolder

# Run alternative pairing at M=1.4
python3 Source_Code/run_cooling_prl.py --masses 1.4 --modes durca_sf_altB 2>&1 | tee Data/Cooling_Outputs/altB_run.log

echo "Alternative pairing complete. Running analysis..."
python3 Plotting_Scripts/prl_multimass_analysis.py 2>&1 | tee Data/Cooling_Outputs/multimass_analysis.log

echo "ALL DONE"
