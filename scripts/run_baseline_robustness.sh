#!/usr/bin/env bash
# Robustness check for the baseline sweep, on its easiest cell (8x8, tc=tm=0.06,
# where PPO reaches 88% from scratch). The sweep ran A2C and SAC at PPO's
# settings; this asks whether their 0.0% survives settings of their own.
#   a2c-e10   A2C at ten epochs: the same gradient steps per frame as PPO
#   a2c-lr7e4 / a2c-lr1e3   A2C at conventional learning rates
#   sac-lr3e4 / sac-lr1e3   SAC at conventional learning rates
# Waits for the baseline sweep to finish so the two never share the GPU.
set -u
cd "$(dirname "$0")/.."
source scripts/_env.sh
mkdir -p logs
while pgrep -f run_baseline_sweep.sh >/dev/null; do sleep 60; done
run() {  # tag algo lr [epochs]
  local LOG="logs/robust_$1.log"
  echo "=== $1 -> $LOG ==="
  STEERCA_ALGO="$2" STEERCA_LR="$3" STEERCA_EPOCHS="${4:-}" STEERCA_N=8 STEERCA_TC=0.06 STEERCA_TM=0.06 \
    STEERCA_FRAMES=2000000 STEERCA_TAG="robust-$1" STEERCA_INIT_FROM=none \
    python -u -m steerca.train > "$LOG" 2>&1
}
run a2c-e10   a2c 1e-4 10
run a2c-lr7e4 a2c 7e-4
run a2c-lr1e3 a2c 1e-3
run sac-lr3e4 sac 3e-4
run sac-lr1e3 sac 1e-3
echo "=== robustness check complete ==="
