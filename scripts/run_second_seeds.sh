#!/usr/bin/env bash
# Second, seeded 8x8 PPO runs from scratch for the training-cost panel
# (collect.py extra / onset; Fig. 3b in the Scientific Reports version). Each
# fixes the global seed with STEERCA_SEED and carries STEERCA_TAG=seed2, so it never
# resumes the first run's checkpoint at the same cost. 4M frames each, about
# 5 hours on an RTX 3090.
#
#   scripts/run_second_seeds.sh              # every cost, in the order first run
#   scripts/run_second_seeds.sh 0.12 0.2     # just these tc = tm values
set -u
cd "$(dirname "$0")/.."
source scripts/_env.sh
mkdir -p logs
# tc = tm per run: s = 0.04, 1.2, 0.08, 0.12, then 0.24, 0.4, 0.6, 1.
COSTS=("$@")
[ "${#COSTS[@]}" -eq 0 ] && COSTS=(0.02 0.6 0.04 0.06 0.12 0.2 0.3 0.5)
for C in "${COSTS[@]}"; do
  echo "=== 8x8 tc=tm=$C seed 2 ==="
  STEERCA_SEED=2 STEERCA_TAG=seed2 STEERCA_N=8 STEERCA_TC="$C" STEERCA_TM="$C" \
    STEERCA_FRAMES=4000000 STEERCA_INIT_FROM=none \
    python -u -m steerca.train > "logs/run_8x8_t${C}_${C}_seed2.log" 2>&1
  echo "=== 8x8 tc=tm=$C seed 2 done ==="
done
