#!/usr/bin/env bash
# Implementation sanity check for the A2C / SAC baselines. The positive control
# (run_positive_control.sh) could not tell "cannot learn this task" from "broken
# learner", because both look like 0%. This removes the credit-assignment
# problem entirely: with both tree charges at -1.5 a felled tree PAYS 1.0 on the
# step it is cut (a cut on dirt still costs 0.5), so "cut the tree you stand on"
# is optimal and immediately rewarded. Any working policy-gradient or Q learner
# should raise its eval return well above the untrained one within a few hundred
# thousand frames. Trees saved is meaningless here -- read the return.
set -u
cd "$(dirname "$0")/.."
source scripts/_env.sh
mkdir -p logs
run() {  # tag algo lr [extra env ...]
  echo "=== $1 ==="
  env "${@:4}" STEERCA_ALGO="$2" STEERCA_LR="$3" STEERCA_CUT_TREE_AHEAD=-1.5 STEERCA_CUT_TREE_IMMINENT=-1.5 \
    STEERCA_N=8 STEERCA_TC=0.06 STEERCA_TM=0.06 STEERCA_FRAMES=200000 \
    STEERCA_TAG="sanity-$1" STEERCA_INIT_FROM=none \
    python -u -m steerca.train > "logs/sanity_$1.log" 2>&1
}
run ppo ppo 1e-4
run a2c a2c 1e-4
run sac sac 1e-4
# The same learners at the settings the diagnosis above points to: A2C with an
# entropy bonus that stops its collapse, SAC with the reward in units of trees
# and a temperature that starts small. These are the settings the tuned sweep
# (run_baseline_tuned.sh) uses; they go in the experiment name.
run a2c-tuned a2c 1e-4 STEERCA_ENTROPY=0.05
run sac-tuned sac 1e-4 STEERCA_REWARD_SCALE=area STEERCA_SAC_ALPHA_INIT=0.05 STEERCA_SAC_TARGET_ENTROPY=0.2
echo "=== sanity check complete ==="
