#!/usr/bin/env bash
# 8x8 SAC from scratch at its best setting, for the difficulty figure
# (collect.py others; Fig. 3 in NCA, Fig. 2 in the Scientific Reports version).
# The tuned variant -- reward scaled by area, initial alpha 0.05, target entropy
# 0.2 -- is the only SAC configuration that learned anything (14.8% at s = 0.4,
# 2M frames, baseline_sac_8x8_t0.2_sweep2m_tuned.log); the sweep has it at
# s = 0.12 and 0.4. These put it at the other costs on PPO's 4M budget, s = 0.08
# first because PPO found it easiest. STEERCA_TAG=onset4m keeps them off every
# other checkpoint. About 8 hours each on an RTX 3090, so it waits for the A2C
# queue to finish rather than share the GPU with it.
#
#   scripts/run_sac_onset.sh              # s = 0.08, then the rest
#   scripts/run_sac_onset.sh 0.12 0.3     # just these tc = tm values
#   STEERCA_FRAMES=2000000 STEERCA_TAG=onset2m scripts/run_sac_onset.sh 0.5 0.6
#
# s = 1 and 1.2 ran at 2M (onset2m): where SAC learns it shows by 152k frames
# and has 95% of its 4M best by 2M (s = 0.6). The learning rate anneals to zero
# at the budget, so finishing one at 4M means re-running it, not resuming it.
set -u
cd "$(dirname "$0")/.."
source scripts/_env.sh
mkdir -p logs
FRAMES="${STEERCA_FRAMES:-4000000}"
TAG="${STEERCA_TAG:-onset4m}"
# Anchored, so it matches the A2C launcher and not a shell that merely names it.
while pgrep -f "^bash scripts/run_a2c_onset.sh" >/dev/null; do sleep 60; done
COSTS=("$@")
[ "${#COSTS[@]}" -eq 0 ] && COSTS=(0.04 0.12 0.3 0.02 0.5 0.6)
for C in "${COSTS[@]}"; do
  echo "=== sac scaled 8x8 tc=tm=$C ==="
  STEERCA_ALGO=sac STEERCA_REWARD_SCALE=area STEERCA_SAC_ALPHA_INIT=0.05 STEERCA_SAC_TARGET_ENTROPY=0.2 \
    STEERCA_TAG="$TAG" STEERCA_N=8 STEERCA_TC="$C" STEERCA_TM="$C" \
    STEERCA_FRAMES="$FRAMES" STEERCA_INIT_FROM=none \
    python -u -m steerca.train > "logs/baseline_sac_8x8_t${C}_${TAG}_scaled.log" 2>&1
  echo "=== sac scaled 8x8 tc=tm=$C done ==="
done
echo "=== sac onset runs complete ==="
