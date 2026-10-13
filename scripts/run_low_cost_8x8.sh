#!/usr/bin/env bash
# 8x8 PPO from scratch at the two ring-regime costs below the sweep, s = 0.04
# and 0.08, for the training-cost panel (Fig. 3b of the Scientific Reports
# version, collect.py onset). Same r10 settings as every other ladder stage;
# 1M frames is ample, since the sweep's runs reach 95% of the planner by 250k
# at s = 0.12. Logs use the ladder's naming so collect.py finds them.
set -u
cd "$(dirname "$0")/.."
source scripts/_env.sh
mkdir -p logs
for C in 0.02 0.04; do
  echo "=== 8x8 tc=tm=$C ==="
  STEERCA_N=8 STEERCA_TC="$C" STEERCA_TM="$C" STEERCA_FRAMES=1000000 STEERCA_INIT_FROM=none \
    python -u -m steerca.train > "logs/run_8x8_t${C}_${C}.log" 2>&1
  echo "=== 8x8 tc=tm=$C done ==="
done
