#!/usr/bin/env bash
# Train 8x8, then 16x16, then 32x32 at one or more action costs, each stage warm
# starting from the one below it. The stages are sequential by necessity: every
# run auto-derives init_from as the next size down with the SAME tc/tm, so 8x8
# must finish before 16x16 starts, and 16x16 before 32x32.
#
#   scripts/run_pipeline.sh 0.06                   # tc = tm = 0.06
#   scripts/run_pipeline.sh 0.06 0.12 0.2 0.3      # one full ladder per cost
#   scripts/run_pipeline.sh 0.05,0.03              # tc = 0.05, tm = 0.03
#   STEERCA_SIZES="8 16" scripts/run_pipeline.sh 0.05 # skip the 32x32 stage
#   STEERCA_FRAMES_32=2000000 scripts/run_pipeline.sh 0.2   # per-size frame budget
#   STEERCA_PLANTED=1 scripts/run_pipeline.sh 0.2           # the planted-instance ladder
#
# 32x32 is where the time goes -- roughly 6x the per-frame cost of 16x16, since
# the trunk runs over 4x the cells with a dilated tail on top.
#
# Runs in the foreground; use nohup/tmux for long jobs. Each stage resumes from
# its own checkpoint if interrupted, so re-running is safe.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_env.sh

if [ "$#" -eq 0 ]; then
  echo "usage: run_pipeline.sh <cost> [cost ...]   where cost is 'tc' or 'tc,tm'" >&2
  exit 1
fi
# Back-compat: the old two-argument form `run_pipeline.sh 0.05 0.03` meant
# tc=0.05 tm=0.03. Two bare numbers are now two separate cost settings, so that
# spelling moved to `0.05,0.03`. One argument means the same thing it always did.
COSTS=("$@")
SIZES="${STEERCA_SIZES:-8 16 32}"
FRAMES="${STEERCA_FRAMES:-4000000}"

mkdir -p logs models

# Report what shape of firebreak the chosen cost admits, per grid size in the
# ladder. Nothing is refused: even where containment is out of reach, limiting
# losses is worth studying.
report_feasibility() {
  python - "$1" "$2" "$SIZES" <<'PY'
import sys
tc, tm = float(sys.argv[1]), float(sys.argv[2])
sizes = [int(x) for x in sys.argv[3].split()]
s = tc + tm
print(f"=== tc={tc:g} tm={tm:g}  tm+tc={s:.3f}  actions/tick={1/tm:.0f} ===")
if s < 0.125:
    # Ringing the fire at radius r means cutting its perimeter, 8r cells at
    # tm+tc each, before the fire reaches them all at time r: 8r(tm+tc) < r, so
    # a closed ring needs tm+tc < 1/8, at any radius. Reaching the ring is what
    # costs: from d cells away it needs r >= d*tm/(1-8(tm+tc)), shown per size.
    print(f"  a closed ring works at any radius once the agent is on it; "
          f"starting d cells away it needs r >= {tm/(1-8*s):.2f} * d")
elif s < 1.0:
    # Above 1/8 no ring closes in time, but a spiral still contains the fire.
    # One turn costs 8r + dr cells, so staying ahead needs
    # dr >= 8r(tm+tc)/(1-(tm+tc)) -- the radius multiplies by (1+7s)/(1-s) per
    # turn. Geometric growth works for any tm+tc < 1; the price is radius, and
    # on a finite grid the edge both bounds that price and closes the curve
    # early. See steerca/firebreak/planner.py, which plans exactly this.
    print(f"  no closed ring is fast enough; containment needs an outward "
          f"spiral, whose radius grows {(1+7*s)/(1-s):.1f}x per turn -- so the "
          f"grid has to be big enough to hold that")
else:
    print(f"  WARNING: tm+tc >= 1 -- a barrier cell costs more than a tick, so "
          f"the agent cannot outbuild the fire at any radius; the best "
          f"available policy only limits losses.")

# Both bounds above assume the agent starts ON the barrier. It does not -- it
# spawns up to N-1 cells away and that travel eats the same deadline -- so read
# them as optimistic. They also ignore the grid edge, which is a free barrier
# and makes them pessimistic for a fire near a corner. The number to trust is
# what the planner actually achieves:
#     python -m steerca.evaluate --n <N> --costs <tc>,<tm>
for n in sizes:
    steps = int(1.4 * n / min(tc, tm))
    gamma = max(0.999, 1 - 1/(2*steps))
    # Median Chebyshev start distance under the env's placement rule (radius
    # uniform on 0..n-1 in a random direction, clamped into the grid); the
    # clamping makes it far shorter than (n-1)/2: 2, 5 and 9 at n = 8, 16, 32.
    import math, random
    rnd = random.Random(0)
    def start_distance():
        fx, fy, r, a = rnd.randrange(n), rnd.randrange(n), rnd.randrange(n), rnd.random() * 2 * math.pi
        dx = min(max(round(r * math.cos(a)), -fx), n - fx - 1)
        dy = min(max(round(r * math.sin(a)), -fy), n - fy - 1)
        return max(abs(dx), abs(dy))
    d_med = sorted(start_distance() for _ in range(20001))[10000]
    need = d_med * tm / (1 - 8 * s) if s < 0.125 else 0.0
    note = (f"  (from the median start, {d_med} cells away, the ring needs radius "
            f"{need:.0f}, more than this grid holds)" if need > (n - 1) // 2 else "")
    print(f"  N={n:<3d} max {steps:5d} steps/episode, gamma {gamma:.5f}{note}")
PY
}

for COST in "${COSTS[@]}"; do
  TC="${COST%%,*}"
  TM="${COST#*,}"; [ "$TM" = "$COST" ] && TM="$TC"

  report_feasibility "$TC" "$TM"

  for N in $SIZES; do
    # Per-size frame budget, e.g. STEERCA_FRAMES_32=2000000, falling back to STEERCA_FRAMES.
    eval "N_FRAMES=\${STEERCA_FRAMES_${N}:-\$FRAMES}"
    echo "=== N=${N}  tc=${TC}  tm=${TM}  frames=${N_FRAMES} ==="
    LOG="logs/run_${N}x${N}_t${TC}_${TM}${STEERCA_PLANTED:+_planted}.log"
    STEERCA_N="$N" STEERCA_TC="$TC" STEERCA_TM="$TM" STEERCA_FRAMES="$N_FRAMES" \
      python -u -m steerca.train 2>&1 | tee "$LOG"
    echo "=== N=${N} done -> ${LOG} ==="
  done
done
