#!/bin/bash
# Run all remaining cooling runs for FSUGarnet, IOPB-I, BigApple
# at M=1.0, 1.8, 2.0 (M=1.4 is already being handled by the current process)

cd /Users/nishasingh/Documents/phdfolder

echo "=== Starting remaining cooling runs ==="
echo "$(date)"

# M=1.0 for all three models
echo "--- M=1.0 ---"
python3 Source_Code/run_cooling_prl.py --models FSUGarnet IOPB-I BigApple --masses 1.0 --modes durca_sf

# M=1.8 for all three models
echo "--- M=1.8 ---"
python3 Source_Code/run_cooling_prl.py --models FSUGarnet IOPB-I BigApple --masses 1.8 --modes durca_sf

# M=2.0 for all three models
echo "--- M=2.0 ---"
python3 Source_Code/run_cooling_prl.py --models FSUGarnet IOPB-I BigApple --masses 2.0 --modes durca_sf

echo "=== All remaining cooling runs complete ==="
echo "$(date)"
