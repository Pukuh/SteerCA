# Planted instances

The all-tree reset leaves only one thing for a policy to exploit: where the fire
is. **Planted instances** encode a harder problem in the initial condition and
come with their optimum known by construction, so a policy can be scored against
a certificate instead of only against the planner. The generator is
`steerca/firebreak/planted.py`, whose module docstring gives the geometry and
the proofs in full.

```bash
STEERCA_PLANTED=1 STEERCA_N=16 STEERCA_TC=0.2 STEERCA_TM=0.2 python -m steerca.train
```

## Two families

**Pocket** (`STEERCA_PLANTED=1`). The fire starts at the centre of a small square
pocket of dirt near the middle of the grid. A run of `k` tree cells in the pocket
wall, the **gap**, is its only way out. Cut the gap before the fire arrives and
the forest outside stands; miss it and everything burns. The gap is hidden by
dirt noise, decoy wall segments, decoy loops and dead-end openings, and a random
rotation or reflection of the whole grid.

**Graph** (`STEERCA_PLANTED=family=graph`, 32×32 at `tc = tm` of 0.2 or 0.3).
The trees form a graph: one-cell junctions joined by one-cell-wide corridors.
Protecting a region means severing every corridor on its boundary, typically
three to six of them, and nothing is saved until the whole cut set is in place.

## The certificate

Each reset is checked in closed form (numpy, under a millisecond apart from a
max-flow):

1. **Separation.** With the gap cut, the fire reaches no protected cell.
2. **Minimality.** A max-flow on the grid graph proves that no smaller set of
   cuts protects the same forest.
3. **Schedule.** The planner's exact clock model shows the agent can cut the gap,
   in order, before the fire arrives, with a tick to spare.

Every reset carries its answer in `params`:

| key | meaning |
|---|---|
| `gap` | `(8, 2)` int64: the cells to cut, in order, `-1` padded |
| `oracle_saved` | fraction of the grid the certified plan keeps |
| `noop_saved` | fraction kept if nobody acts |

`PlantedOraclePolicy` replays the certified plan and is a drop-in policy.
Evaluation on planted instances adds `oracle_fraction` (the mean of
`saved / oracle_saved` per episode) and `oracle_gain` (the share of what was at
stake, between doing nothing and the oracle). The certificate is a feasible
plan, not a proven optimum over all strategies, so a policy can exceed it:
`oracle_fraction > 1` is legitimate. Checkpoints are still selected on trees
saved.

## Commands

```bash
# Train; the ladder warm starts each planted stage from the planted run one size down
STEERCA_PLANTED=1 STEERCA_SIZES="16 32" scripts/run_pipeline.sh 0.2
# The graph tier: a flat run, and the two-stage curriculum of the papers
STEERCA_PLANTED=family=graph STEERCA_N=32 STEERCA_TC=0.2 STEERCA_TM=0.2 STEERCA_FRAMES=2000000 python -m steerca.train
scripts/run_curriculum.sh

# Planner, oracle and a checkpoint on the same planted episodes
python -m steerca.evaluate --planted --n 16 --costs 0.2,0.2 --ckpt <policy.pth>
# Check the generator: certificate checks, plus the planner must not beat the oracle
python -m steerca.validate_planted --n 16 --costs 0.2,0.2 --count 100 --planner-episodes 30
# Watch it
python -m steerca.render --planted --oracle --ckpt <policy.pth>
```

## Configuring the generator

`STEERCA_PLANTED` takes `key=value` pairs, for example
`STEERCA_PLANTED="k=2-4,noise=0.05-0.1,decoys=0"`. The keys are the fields of
`PlantedConfig` (see `PlantedConfig.from_string` for the short names), and
ranges are written `lo-hi`. Any change from the defaults goes into the
experiment name (`..._cut0.5_planted-decoys0-k2-4-noise0.05-0.1_r10_...` for the example above), so runs on different
instance distributions never share a checkpoint.

Two practical notes:

* Planted episodes are capped at 2.5× the usual length, because the fire waits
  at the gap and then winds through decoys.
* A planted run warm starts with the critic only from another planted run. A
  vanilla checkpoint passed in `STEERCA_INIT_FROM` loads policy-only, because
  planted returns are on a different scale (the pocket always burns). For a
  curriculum across planted distributions that share the reward, set
  `STEERCA_KEEP_CRITIC=1`, and give each stage its own `STEERCA_TAG` so it does
  not resume the previous stage's checkpoint.
