#!/usr/bin/env bash
# The baseline sweep re-run at settings under which each learner can learn at
# all. The sanity check (run_sanity_check.sh, the papers' verification task) found the
# original 0.0% cells were set up to fail even a task where a cut pays at once:
#
#   A2C  entropy 5e-3 collapses the policy within ten batches -> STEERCA_ENTROPY=0.05
#   SAC  alpha_init 1 and target entropy 0.98*log 9 swamp a 1/N^2 reward
#        -> reward scaled to area, alpha_init 0.05, target entropy weight 0.2
#
# Same cells, budget and tag as run_baseline_sweep.sh; the settings go in the
# experiment name, so no checkpoint is shared with the untuned runs. PPO is not
# re-run -- its settings are unchanged. A2C first (~3 h), then SAC (~33 h).
set -u
cd "$(dirname "$0")/.."
source scripts/_env.sh
mkdir -p logs

FRAMES="${STEERCA_FRAMES:-2000000}"
TAG="${STEERCA_TAG:-sweep2m}"

cell() {  # algo N C, extra env in the caller
  local LOG="logs/baseline_${1}_${2}x${2}_t${3}_${TAG}_tuned.log"
  echo "=== $1  ${2}x${2}  tc=tm=$3  ${FRAMES} frames -> $LOG ==="
  STEERCA_ALGO="$1" STEERCA_N="$2" STEERCA_TC="$3" STEERCA_TM="$3" \
    STEERCA_FRAMES="$FRAMES" STEERCA_TAG="$TAG" STEERCA_INIT_FROM=none \
    python -u -m steerca.train > "$LOG" 2>&1
  echo "=== $1 ${2}x${2} t$3 done ==="
}

for N in 8 16; do for C in 0.06 0.2; do
  STEERCA_ENTROPY=0.05 cell a2c "$N" "$C"
done; done
for N in 8 16; do for C in 0.06 0.2; do
  STEERCA_REWARD_SCALE=area STEERCA_SAC_ALPHA_INIT=0.05 STEERCA_SAC_TARGET_ENTROPY=0.2 \
    cell sac "$N" "$C"
done; done
echo "=== tuned baseline sweep complete ==="
