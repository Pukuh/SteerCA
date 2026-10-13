# The Firebreak environment

Firebreak is SteerCA's forest-fire suite. A fire spreads over an `n × n` forest
by a cellular automaton (CA), and an agent, a bulldozer, moves over the grid
cutting trees to build firebreaks. This page defines the task precisely. The
code is `steerca/firebreak/` and the generic step is `steerca/core/cs_env.py`.

## State

| key | type, shape | meaning |
|---|---|---|
| `grid` | bool `(3, n, n)` | one-hot cell state: plane 0 `DIRT`, 1 `TREE`, 2 `FIRE` |
| `agent_state` | int64 `(2,)` | the agent's `(row, col)` |
| `t` | float32 `()` | time left until the next CA update, in `(0, 1]` |
| `params` | TensorDict | `tc`, `tm`, `n`, `distance`, plus the planted keys on planted instances |

Every reset is an all-tree grid with one burning cell at a uniformly random
position. The agent starts at a random angle and a random distance of up to
`n - 1` cells from it, clamped to the grid.

## The automaton

Each update applies two rules:

* a `FIRE` cell burns out to `DIRT`;
* a `TREE` cell with at least one burning Moore neighbour (any of its 8 neighbours) becomes `FIRE`.

A front therefore advances one cell per update in every direction. The episode
ends at the CA's fixed point, when no fire is left.

## Actions and the hybrid time model

Nine discrete actions: `0..7` move one cell (E, NE, N, NW, W, SW, S, SE; moves
clamp at the border), and `8` cuts the tree under the agent (`TREE → DIRT`; a
cut on bare dirt does nothing but still costs).

Actions take time measured in CA updates: a move takes `tm`, a cut `tc`. The
agent's clock `t` counts down to the next update, and on each action the CA
advances `floor(1 + Δt)` updates, where `Δt` is how far the action's duration
overshoots the clock. Cheap actions fit several to an update; an action longer
than one update lets the CA advance several times. `tc` and `tm` set the
difficulty: the papers write `s = tc + tm` for the cost of one firebreak cell.

| `s` | what a barrier can do |
|---|---|
| `< 1/8` | a closed ring around the fire is fast enough at any radius once the agent is on it |
| `1/8 … 1` | no ring closes in time, but an outward spiral still contains the fire |
| `> 1` | containment is impossible; only losses can be limited |

`scripts/run_pipeline.sh` prints this analysis for the costs you give it.

## Reward

```
reward = -(trees burnt + 0.5 · cut actions) / n²
```

A tree the fire takes costs 1, and a cut costs a flat 0.5 wherever it lands.
Returns are area-normalised, so `-1` means the whole forest was lost at any
grid size.

**A cut is priced below a tree, so cutting a doomed tree pays.** A tree the fire
would take costs 1 if left alone and 0.5 if cut, so cutting it earns 0.5 whether
or not the cut is part of a closed barrier. That is a deliberate trade: it is
what lets PPO learn to cut at all. Pricing it away (a cut at 1.0 or 1.5) made PPO
converge to never cutting when trained from scratch at 8×8, `tc = tm = 0.06`,
where the planner saves 87%. The cost of the
trade is that **return is not the score.** Every evaluation reports both:

* **trees saved**: the fraction of the grid still forested at the fixed point.
  This is the task metric, the one the planner is compared on, and the one
  checkpoints are selected on.
* **return**: what the agent optimises. A rising return with flat trees saved
  means the agent is farming doomed trees, not learning.

Two settings re-price cuts for experiments (both default to 0):

| `STEERCA_CUT_TREE_AHEAD` | `STEERCA_CUT_TREE_IMMINENT` | a cut costs |
|---|---|---|
| 0 | 0 | 0.5 flat (default, all published runs) |
| 0 | 1 | 0.5 ahead of the fire, 1.5 next to it |
| 0.5 | 0.5 | 1.0 flat: the return is exactly −(trees lost)/n² |
| 1 | 1 | 1.5 flat |

"Next to it" means the tree had a burning Moore neighbour, so it would have
ignited on the next update. The price goes into the checkpoint name (`cut0.5`,
`cut0.5-1.5`, `cut1`…), so runs under different rewards never share a file.

## Episode length and discount

An episode is capped at `1.4 · n / min(tc, tm)` steps (2.5× on planted
instances), enough for the fire to reach the far corner and burn out, so
evaluation measures finished episodes. The discount is
`γ = max(0.999, 1 − 1/(2 · cap))`, so it only rises above 0.999 for long
32×32 episodes.

## The reference: the barrier planner

`steerca.firebreak.planner.BarrierPlannerPolicy` is a hand-designed planner. Once
per CA update it scores every rectangular barrier (corners may sit off the grid,
so the edge counts as free barrier) and a family of outward spirals against the
fire's exact arrival times. It then keeps the cheapest plan it can finish before
the fire arrives, or does nothing when nothing pays. It is the yardstick every learned policy
is scored against, on paired episodes (same ignition, same start):

```bash
python -m steerca.evaluate --n 16 --costs 0.2,0.2 --episodes 100 --ckpt <policy.pth>
```

It prices a felled tree as a lost tree (`CUT_COST = 1`), so it optimises trees
saved; its return is not comparable with an agent's.

## Planted instances

`STEERCA_PLANTED=1` replaces the all-tree reset with instances whose optimum is
known by construction. See [planted.md](planted.md).

## The network

`steerca.archs.ForestFireEquivariantNet` is fully convolutional: a 3×3 stem over
the state planes plus a one-hot agent plane, a stack of residual blocks, and three
readouts that need no parameters (the features at the agent's cell, and global
mean and max pooling), plus a time embedding. No parameter depends on `n`, so a
checkpoint trained at one size loads and runs at another. That is what the
8 → 16 → 32 warm-start ladder relies on.

What does depend on `n` is how far the trunk sees: the receptive field is
`3 + 4 · Σ dilations`. `arch_for_grid(n)` picks the configuration:

| `n` | blocks | tag | receptive radius |
|---|---|---|---|
| ≤ 16 | 6 dense | `c96_b6_512_64` | 13 |
| 32 | 6 dense + dilations 2, 4, 8 | `c96_b9d2-4-8_512_64` | 41 |

A new residual block starts as the identity, so a 16×16 checkpoint loaded into
the 32×32 trunk computes exactly what it did before and then learns the
long-range blocks (`load_grown_state_dict`).

## Checkpoint names

```
{algo}_equiv_{arch}_1hot_{n}x{n}_t{tc}_{tm}_{cut price}[_{planted}][_{hp}]_r10[_{tag}][_final]_{policy,value}.pth
```

* `algo` is `ppo`, `a2c` or `sac`.
* `hp` appears only for non-default learner settings (`_ent0.05`, `_rs64_a0.05-te0.2`).
* `tag` is `STEERCA_TAG`.
* `r10` is the reward revision every published run uses.
* A custom CA's runs are named `{algo}_{spec name}_equiv_...`.

The plain file is the best evaluation by trees saved and is what a run resumes
from. `_final` is written when a run completes. A run that never beats 0% keeps
its first evaluation as the plain file, so look at `_final` to see where it ended.

**Build a network from the checkpoint's tag, not from the grid you evaluate
at.** `parse_arch_tag(name)` recovers the architecture, and
`steerca.evaluate.make_ppo_policy(path, "equiv", device, n)` does the whole load.
