#!/usr/bin/env bash
# A two-stage curriculum on the graph planted family, at 32x32 tc=tm=0.2.
#
#   scripts/run_curriculum.sh
#
# The target distribution needs four corridors cut before a single tree is
# saved. A flat run reaches 12.9% of the oracle in 2M frames and 51% in 2.8M,
# so it learns, slowly. This asks whether an easier tier first gets there
# faster, at the same 2M total budget.
#
# Picking the easy tier took three attempts, and the two failures are the
# instructive part:
#
#   * a NEAR agent gives two-cut answers but almost no slack, because the
#     corridors leaving the ignition point burn within a tick. Zero for 1.2M.
#   * a HIGH MARGIN gives slack but shrinks the payoff, because only cuts far
#     from the fire can satisfy it: 44 trees saved of 121 at risk, against 103
#     of 152 on the target. Zero for 600k, where the flat run had signal at 52k.
#
# Every knob trades slack against payoff EXCEPT the cycle requirement. Drop it
# and each region hangs by one corridor, so a single cut wins the whole payoff
# with the agent still far away. That is slack and payoff together, and it is
# what rcyc=0 buys: k=1, a 5.3 point stake, and 69 ms an instance.
#
# Stage 2 carries scuts=6, which is its own default and changes nothing, purely
# so its tag differs from earlier runs on this distribution. Without that the
# script finds a checkpoint of its own, resumes it, and ignores STEERCA_INIT_FROM.
set -u
cd "$(dirname "$0")/.."
source scripts/_env.sh
mkdir -p logs
B=${STEERCA_MODELS:-models}/ppo_equiv_c96_b9d2-4-8_512_64_1hot_32x32_t0.2_0.2_cut0.5
S1="$B""_planted-graph-cuts1-6-fdeg1-keep0.4-rcyc0-rtrap0_r10_policy.pth"

echo "=== stage 1: acyclic, one-cut answers, 600k frames ==="
STEERCA_PLANTED="family=graph,rcyc=0,rtrap=0,cuts=1-6,keep=0.4,fdeg=1" \
  STEERCA_N=32 STEERCA_TC=0.2 STEERCA_TM=0.2 STEERCA_FRAMES=600000 STEERCA_INIT_FROM=none \
  python -u -m steerca.train

echo "=== stage 2: target distribution, 1.4M frames, warm from stage 1 ==="
STEERCA_PLANTED="family=graph,cuts=3-6,scuts=6" \
  STEERCA_N=32 STEERCA_TC=0.2 STEERCA_TM=0.2 STEERCA_FRAMES=1400000 \
  STEERCA_INIT_FROM="$S1" STEERCA_KEEP_CRITIC=1 \
  python -u -m steerca.train
echo "=== curriculum done ==="
