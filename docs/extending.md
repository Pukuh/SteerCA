# Extending SteerCA: variants and new automata

A SteerCA environment is three things: a **cellular automaton**, a **local
agent** that moves over it and acts on the cell it stands on, and the **hybrid
time model** that makes every action cost CA time. The agent and the time model
are shared. What you write is the automaton and the task, as a
`steerca.core.CASpec`: a handful of plain functions over batched tensors.
`SteerCAEnv` turns a spec into a batched TorchRL environment, and
`python -m steerca.train` trains on it unchanged.

There are two worked examples:

| example | what changes | lines |
|---|---|---|
| [`examples/firebreak_wind/`](../examples/firebreak_wind/spec.py) | a Firebreak **variant**: fire cannot burn upwind | ~20 |
| [`examples/excitable_medium/`](../examples/excitable_medium/spec.py) | a **new CA**: Greenberg–Hastings excitable waves, five planes, its own reward and score | ~110 |

## The contract

All tensors are batched: `B` environments, `C` grid planes, an `n × n` grid.

```python
from steerca.core import CASpec

CASpec(
    name="my-ca",              # goes in checkpoint names; use '-', never '_'
    n_states=C,                # number of bool planes in the grid
    ca_step=...,               # (B,C,n,n) bool -> (B,C,n,n) bool; one CA update, no mutation of the input
    action=...,                # (actions (B,), grid, agent (B,2)) -> (grid, new agent); may mutate grid
    reward_grid=...,           # (grid_before, grid_after_CA, grid_after_action) -> (B,) float32
    done=...,                  # grid -> (B,) bool: True at the CA's fixed point
    reset=...,                 # (batch shape, n, params, device) -> (grid, agent_state); use the torch RNG
    # optional
    reward_action=...,         # (actions, grid, agent) -> (B,), a per-action charge. Default 0
    time=...,                  # (actions, params) -> (B,) durations in updates. Default tc for action 8, tm otherwise
    score=...,                 # final grid -> (B,) float, the task metric (higher is better)
    extra_params=lambda n: {}, # extra scalars your reset reads from params
    episode_cap=...,           # (n, tc, tm) -> steps before truncation. Default 1.4·n/min(tc, tm)
    symmetric=True,            # rule invariant under the square's 8 symmetries? (enables augmentation)
)
```

What each piece is for:

* **`ca_step`** is called once per CA update, on the whole batch. Write it with
  tensor ops; one `conv2d` with a 3×3 kernel counts neighbours, as in both
  examples.
* **`action`** handles the agent. Actions `0..7` are moves and `8` is the cell
  action. `steerca.core.local_action(effect)` builds the whole function from
  `effect(grid, b, r, c, mask)`, which only says what the cell action does to
  the cell at `(r, c)` where `mask` holds. Use that unless your agent is unusual.
* **`reward_grid`** sees three grids: before the step, after the CA advance and
  after the action. That is enough to tell what the CA destroyed from what the
  agent did, and to price them differently.
* **`done`** must become true at the fixed point, or every episode runs to the
  cap.
* **`score`** is what evaluation reports and what checkpoint selection
  maximises. Define it whenever the reward is shaped: Firebreak's reward pays for
  cutting doomed trees, so its return can rise while its score does not. Without
  a score, selection falls back to the return. If the score depends on history,
  carry that history in a plane (the excitable example's `SCAR` plane records
  every cell that ever fired).
* **`symmetric`**: training augments minibatches with random rotations and
  reflections, which is only valid if your rule is the same under them. Set it
  to `False` for anything directional (wind, gravity, a one-way wire).

The observation is the grid plus a one-hot plane for the agent, so the network
(`ForestFireEquivariantNet`, which takes `in_states=C`) works for any `C` and
any `n`.

## A variant: change one function

Start from Firebreak's own spec and replace what differs.
`steerca.firebreak.spec.firebreak_spec()` is bit-for-bit the environment of the
papers (`tests/test_spec_equivalence.py` checks it against `ForestFireEnv`), so
whatever you change is the only difference:

```python
from dataclasses import replace
from steerca.firebreak.spec import firebreak_spec

def make_spec():
    return replace(firebreak_spec(), name="firebreak-wind",
                   ca_step=windy_step, symmetric=False)
```

Other variants that fit the same mould:

* **Terrain**: a fourth plane of "wet" cells that never ignite. Change
  `n_states`, `ca_step` and `reset`.
* **Stochastic spread**: ignite with probability `p` per burning neighbour.
  `ca_step` may draw from `torch.rand`. Keep it on the torch RNG so seeds still
  reproduce.
* **Several fires** or **a different start**: replace `reset`.
* **Another price for a cut**: replace `reward_grid` / `reward_action`, or for
  Firebreak just set `STEERCA_CUT_TREE_AHEAD` / `STEERCA_CUT_TREE_IMMINENT`.

Check what in the toolchain your change invalidates. The barrier planner
(`steerca.firebreak.planner`) assumes Firebreak's isotropic rule, so it is no
longer a reference for the wind variant. The `episode_cap` assumes a front
crosses the grid in about `n` updates.

## A new automaton

`examples/excitable_medium/spec.py` is a complete one. The pattern:

1. **Name the planes.** Use one-hot planes for the cell states, plus any flag
   planes the score needs.
2. **Write `ca_step`** with tensor ops, returning a new tensor.
3. **Decide what the agent's cell action does**, and wrap it with
   `local_action`.
4. **Write the reward** as the per-step change in what you care about, divided
   by `n²` so returns are comparable across grid sizes.
5. **Write `done`** (the fixed point) and **`score`** (the metric).
6. **Write `reset`**, drawing only from the torch RNG.

Then check it before training:

```python
from torchrl.envs.utils import check_env_specs
from steerca.core import SteerCAEnv, evaluate_policy

env = SteerCAEnv(make_spec(), n=16, tc=0.2, tm=0.2, batch_size=[64], seed=0)
check_env_specs(env)                  # shapes and dtypes agree with the specs
print(evaluate_policy(env, policy))   # {'return': ..., 'score': ...}
```

and write baselines (an idle agent, a random agent, a hand-written heuristic if
you have one), as `examples/excitable_medium/run.py` does. A trained policy
means nothing without a reference to beat.

## Training on your spec

Point `STEERCA_ENV` at a function that returns the spec, as `module:function` or
`path/to/file.py:function`:

```bash
STEERCA_ENV=examples.excitable_medium.spec:make_spec \
STEERCA_N=16 STEERCA_TC=0.2 STEERCA_TM=0.2 STEERCA_FRAMES=1000000 \
python -m steerca.train
```

Everything else works as for Firebreak: PPO, A2C or SAC (`STEERCA_ALGO`), seeds,
tags, resume, and the 8 → 16 → 32 warm-start ladder (`scripts/run_pipeline.sh`
honours `STEERCA_ENV`). Checkpoints are named
`ppo_my-ca_equiv_<arch>_1hot_<n>x<n>_t<tc>_<tm>_r10_policy.pth`.
Evaluation prints `eval score` (your `score`) instead of `eval saved`, and
selects checkpoints on it. The name does not encode anything about your reward
beyond the spec name, so give a spec with a different reward a different name.

To run a trained policy yourself:

```python
from steerca.evaluate import make_ppo_policy   # builds the net from the checkpoint's own tag
spec = make_spec()
policy = make_ppo_policy("models/ppo_my-ca_equiv_..._policy.pth", "equiv", "cuda",
                         n=16, in_states=spec.n_states)
env = SteerCAEnv(spec, n=16, tc=0.2, tm=0.2, batch_size=[100], device="cuda", seed=0)
print(evaluate_policy(env, policy))
```

## Checklist

- [ ] `check_env_specs` passes.
- [ ] `ca_step` does not mutate its input, and `done` becomes true.
- [ ] An idle agent's episode ends before `episode_cap`.
- [ ] `score` is defined, or the reward is unshaped.
- [ ] `symmetric` is right.
- [ ] `reset` uses only the torch RNG: two seeded resets are identical.
- [ ] There is at least one baseline to compare against.
- [ ] The name contains no `_`.
