#!/usr/bin/env bash
# Positive control for the A2C / SAC baselines. Both scored 0.0% in every cell
# of the containment sweep, which invites "the implementation is broken". This
# runs them on the easiest task we have -- the acyclic planted graph tier, where
# one cut wins the whole payoff and PPO goes from 5% to 65% of the oracle in
# 600k frames -- at the same budget. Each learner at PPO's settings and at a
# conventional learning rate of its own.
set -u
cd "$(dirname "$0")/.."
source scripts/_env.sh
mkdir -p logs
TIER="family=graph,rcyc=0,rtrap=0,cuts=1-6,keep=0.4,fdeg=1"
run() {  # tag algo lr
  echo "=== $1 ==="
  STEERCA_ALGO="$2" STEERCA_LR="$3" STEERCA_PLANTED="$TIER" STEERCA_N=32 STEERCA_TC=0.2 STEERCA_TM=0.2 \
    STEERCA_FRAMES=600000 STEERCA_TAG="posctl-$1" STEERCA_INIT_FROM=none \
    python -u -m steerca.train > "logs/posctl_$1.log" 2>&1
}
run a2c-lr1e4 a2c 1e-4
run a2c-lr7e4 a2c 7e-4
run sac-lr1e4 sac 1e-4
run sac-lr3e4 sac 3e-4
echo "=== positive control complete ==="
