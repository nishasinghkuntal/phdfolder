#!/bin/bash
# Run all remaining cooling combinations sequentially
# Avoids race conditions by running one at a time

cd /Users/nishasingh/Documents/phdfolder

echo "=== Starting remaining cooling runs ==="
echo "$(date)"

# PRIORITY 1: durca_only M=1.4 for new models (best chi2 mechanism)
echo ""
echo ">>> durca_only M=1.4 new models <<<"
python3 Source_Code/run_cooling_prl.py --models FSUGarnet IOPB-I BigApple --masses 1.4 --modes durca_only

# PRIORITY 2: durca_sf M=1.8 and M=2.0 for new models
echo ""
echo ">>> durca_sf M=1.8 new models <<<"
python3 Source_Code/run_cooling_prl.py --models FSUGarnet IOPB-I BigApple --masses 1.8 --modes durca_sf

echo ""
echo ">>> durca_sf M=2.0 new models <<<"
python3 Source_Code/run_cooling_prl.py --models FSUGarnet IOPB-I BigApple --masses 2.0 --modes durca_sf

# PRIORITY 3: durca_only other masses for new models
echo ""
echo ">>> durca_only M=1.0 new models <<<"
python3 Source_Code/run_cooling_prl.py --models FSUGarnet IOPB-I BigApple --masses 1.0 --modes durca_only

echo ""
echo ">>> durca_only M=1.8 new models <<<"
python3 Source_Code/run_cooling_prl.py --models FSUGarnet IOPB-I BigApple --masses 1.8 --modes durca_only

echo ""
echo ">>> durca_only M=2.0 new models <<<"
python3 Source_Code/run_cooling_prl.py --models FSUGarnet IOPB-I BigApple --masses 2.0 --modes durca_only

# PRIORITY 4: murca_sf and sf_no3p2 at M=1.4 for mode comparison
echo ""
echo ">>> murca_sf M=1.4 all models <<<"
python3 Source_Code/run_cooling_prl.py --models GM1 GM2 FSUGarnet IOPB-I BigApple --masses 1.4 --modes murca_sf

echo ""
echo ">>> sf_no3p2 M=1.4 all models <<<"
python3 Source_Code/run_cooling_prl.py --models GM1 GM2 FSUGarnet IOPB-I BigApple --masses 1.4 --modes sf_no3p2

echo ""
echo "=== All runs complete ==="
echo "$(date)"
