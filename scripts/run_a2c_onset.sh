#!/usr/bin/env bash
# 8x8 A2C from scratch at its best setting, for the difficulty figure
# (collect.py others; Fig. 3 in NCA, Fig. 2 in the Scientific Reports version).
# Entropy 0.2 is the only A2C configuration that learned anything in the
# robustness check (14.5% best at s = 0.12, 2M frames,
# baseline_a2c_8x8_t0.06_sweep2m_ent0.2.log); everything else is PPO's. These
# put it at the other costs on PPO's 4M budget, s = 0.08 first because PPO
# found it easiest. STEERCA_TAG=onset4m keeps them off every other checkpoint. About 65
# minutes each on an RTX 3090.
#
#   scripts/run_a2c_onset.sh              # s = 0.08, then the rest
#   scripts/run_a2c_onset.sh 0.12 0.2     # just these tc = tm values
set -u
cd "$(dirname "$0")/.."
source scripts/_env.sh
mkdir -p logs
COSTS=("$@")
[ "${#COSTS[@]}" -eq 0 ] && COSTS=(0.04 0.02 0.12 0.2 0.3 0.5 0.6)
for C in "${COSTS[@]}"; do
  echo "=== a2c ent0.2 8x8 tc=tm=$C ==="
  STEERCA_ALGO=a2c STEERCA_ENTROPY=0.2 STEERCA_TAG=onset4m STEERCA_N=8 STEERCA_TC="$C" STEERCA_TM="$C" \
    STEERCA_FRAMES=4000000 STEERCA_INIT_FROM=none \
    python -u -m steerca.train > "logs/baseline_a2c_8x8_t${C}_onset4m_ent0.2.log" 2>&1
  echo "=== a2c ent0.2 8x8 tc=tm=$C done ==="
done
echo "=== a2c onset runs complete ==="
