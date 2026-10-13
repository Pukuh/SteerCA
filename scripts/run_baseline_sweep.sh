#!/usr/bin/env bash
# Baseline sweep: three learners x two grid sizes x two action costs.
#
#   scripts/run_baseline_sweep.sh
#
# The cells are chosen to span the difficulty axis rather than to fill it:
# s = 0.12 sits in the ring regime where a closed barrier exists and PPO
# saturates, s = 0.4 in the spiral regime where it does not. Grid sizes 8 and 16
# are the two cheapest rungs of the ladder.
#
# Every run starts from scratch, and STEERCA_TAG keeps them clear of the r10
# checkpoints that own the same names -- without it each PPO cell would resume a
# fully trained policy and report it as a result.
#
# 2M frames a cell, the same budget for every learner, so the comparison is on
# samples rather than on wall clock. Expect roughly 20 hours in total: A2C is
# the cheapest at one epoch a batch, SAC the dearest at 250 gradient steps.
set -u
cd "$(dirname "$0")/.."
source scripts/_env.sh
mkdir -p logs

FRAMES="${STEERCA_FRAMES:-2000000}"
TAG="${STEERCA_TAG:-sweep2m}"

for N in 8 16; do
  for C in 0.06 0.2; do
    for A in ppo a2c sac; do
      LOG="logs/baseline_${A}_${N}x${N}_t${C}_${TAG}.log"
      echo "=== $A  ${N}x${N}  tc=tm=$C  ${FRAMES} frames -> $LOG ==="
      STEERCA_ALGO="$A" STEERCA_N="$N" STEERCA_TC="$C" STEERCA_TM="$C" \
        STEERCA_FRAMES="$FRAMES" STEERCA_TAG="$TAG" STEERCA_INIT_FROM=none \
        python -u -m steerca.train > "$LOG" 2>&1
      echo "=== $A ${N}x${N} t$C done ==="
    done
  done
done
echo "=== baseline sweep complete ==="
