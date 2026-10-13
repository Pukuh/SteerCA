from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import tqdm
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn

from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import (
    CatTensors,
    EnvBase,
    TransformedEnv,
    UnsqueezeTransform,
)

from torchrl.envs.transforms import Transform

from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp, ExplorationType, set_exploration_type

import sys
import os
from steerca.config import setting
import steerca.core.cs_env as cs_env
import steerca.firebreak.agents as agents
from steerca.firebreak.ca import DIRT, TREE, FIRE
from steerca.firebreak.planted import (
    K_MAX, PlantedConfig, generate_instance, instance_to_tensors,
)

from torchvision.transforms.v2.functional import rotate, vertical_flip, horizontal_flip

# Grid size and action costs are read from the environment so a run can be
# launched without editing this file:
#     STEERCA_N=8 STEERCA_TC=0.05 STEERCA_TM=0.05 python -m steerca.train
# They are read once at import, before the nets and envs are built.
N = int(setting("N", 16))  # Default grid size

# Action costs, in CA ticks. 1/tm actions fit in one tick, so tm sets how fast
# the agent moves relative to the fire: the fire advances one cell per tick, the
# agent 1/tm cells.
#
# These control the difficulty of the task, and there is a threshold. Ringing a
# fire at Chebyshev radius r means cutting its perimeter, the boundary of a
# (2r+1) x (2r+1) square: 8r cells, each costing a move plus a cut. The fire
# reaches every cell of that ring at time r. So containment is possible when
#
#     8r (tm + tc) < r      <=>   tm + tc < 1/8 = 0.125,   for any radius r.
#
# The radius cancels; what makes rings hard near the threshold is reaching them.
# An agent starting d cells away spends d*tm travelling first, so the ring needs
# r >= d*tm / (1 - 8(tm+tc)), which blows up as tm+tc approaches 0.125 -- near
# the threshold the ring must be large, and the grid big enough to hold it.
#
# tm + tc = 0.02 leaves a 6x margin and makes containment trivial at any radius.
TC = float(setting("TC", 0.05))  # time to cut a tree
TM = float(setting("TM", 0.05))  # time to move

# Planted instances. STEERCA_PLANTED=1 (defaults) or STEERCA_PLANTED="k=2-4,noise=0.05-0.1"
# makes every reset draw a certified planted instance from steerca.firebreak.planted
# instead of the all-tree grid: a hidden gap of k trees is the only passage
# between the fire and a large protected forest, and cutting it is the known
# optimum. Read at call time, so a script can set ff.PLANTED_CFG before building
# an env, exactly as it sets ff.N -- but before, not after: the spec gains the
# gap / oracle keys at construction.
def planted_config_from_env():
    value = setting("PLANTED")
    if value is None or value.strip().lower() in ("", "0", "false", "no", "none"):
        return None
    return PlantedConfig.from_string(value)

PLANTED_CFG = planted_config_from_env()


def episode_cap(n, tc, tm, planted=None):
    """Steps an episode can need before the fire burns out, with headroom.

    The fire needs up to ~n+2 CA ticks to reach the far corner of an all-tree
    grid and 1/min(tc, tm) actions fit in a tick; 1.4x is the headroom every
    run has used. A planted fire starts near the middle, waits d_f ticks at
    the gap, and then has to go round decoy loops and dead-end pockets, and an
    idle 16x16 episode measured 21-24 ticks against that cap's 22.4 -- so
    planted runs get 2.5x. Truncating an episode means the agent never sees
    the outcome of its own firebreak, and eval measures truncations.
    """
    if planted is None:
        planted = PLANTED_CFG
    factor = 1.4 if planted is None else planted.cap_factor()
    return int(factor * n / min(tc, tm))

_FUNCS = {
    "fun_ca_step": agents.ca_step,
    "fun_time": agents.time,
    "fun_action": agents.action,
    "fun_reward_grid": agents.reward_grid,
    "fun_reward_action": agents.reward_action,
    "fun_done": agents.done,
}

def _step(tensordict):
    """Update the grid state using forest fire rules."""
    return cs_env.step(tensordict, funcs=_FUNCS)


def all_tree_reset(shape, n, distance, device):
    """Firebreak's default initial condition: every cell a tree, one random fire.

    The agent starts at a random angle and a random radius of at most
    ``distance`` cells from the fire (``N - 1`` by default), clamped to the grid. Returns ``(grid, agent_state)`` with shapes
    ``(*shape, 3, n, n)`` and ``(*shape, 2)``, drawn from the global torch RNG.
    """

    # Initialize grid with all cells as trees
    grid = torch.zeros((*shape, 3, n, n), dtype=torch.bool, device=device)
    grid[..., TREE, :, :] = 1  # Set all cells to TREE (1)

    # generate a random coordinate for the initial fire
    fire_x = torch.randint(0, n, size=(*shape, 1), device=device)
    fire_y = torch.randint(0, n, size=(*shape, 1), device=device)

    # ignite one cell per sample (flat view pairs each sample with its own coordinate)
    flat_grid = grid.view(-1, 3, n, n)
    batch = torch.arange(flat_grid.shape[0], device=device)
    flat_grid[batch, FIRE, fire_x.view(-1), fire_y.view(-1)] = 1
    flat_grid[batch, TREE, fire_x.view(-1), fire_y.view(-1)] = 0

    # Position the agent in a random position within a "circle" of radius `distance` around the fire
    angle = torch.rand((*shape, 1), device=device) * 2 * torch.pi
    radius = torch.randint(0, distance+1, size=(*shape, 1), device=device)
    x_offset = (radius * torch.cos(angle)).round().long()
    y_offset = (radius * torch.sin(angle)).round().long()

    # Ensure the agent is within bounds
    x_offset = torch.clamp(x_offset, -fire_x, n - fire_x - 1)
    y_offset = torch.clamp(y_offset, -fire_y, n - fire_y - 1)
    # Calculate the agent's position relative to the fire
    pos_x = fire_x + x_offset
    pos_y = fire_y + y_offset

    pos = torch.cat((pos_x, pos_y), dim=-1)

    return grid, pos


def _reset(self, tensordict):

    if tensordict is None or tensordict.is_empty() or not "params" in tensordict :
        # if no ``tensordict`` is passed, we generate a single set of hyperparameters
        # Otherwise, we assume that the input ``tensordict`` contains all the relevant
        # parameters to get started.
        tensordict = self.gen_params(batch_size=self.batch_size)

    if "gap" in tensordict["params"].keys():
        return _reset_planted(self, tensordict)

    # On a partial reset ("_reset" mask present) we still regenerate the full
    # batch: torchrl's _update_during_reset keeps unmasked entries unchanged.
    # extract grid size n (same for all samples)
    n = int(tensordict["params","n"].reshape(-1)[0].item())
    distance = int(tensordict["params","distance"].reshape(-1)[0].item())
    grid, pos = all_tree_reset(tuple(tensordict.shape), n, distance, self.device)
    t = torch.ones((*tensordict.shape,), device=self.device)

    out = TensorDict(
        {
            "grid": grid,
            "t": t,
            "agent_state": pos,
            "params": tensordict["params"],
        },
        batch_size=tensordict.shape,
        device=self.device,
    )
    return out


def _reset_planted(self, tensordict):
    """Reset from the planted generator: one certified instance per slot.

    A partial reset hands this only the ``_reset`` mask (torchrl's
    ``maybe_reset`` selects the reset keys before calling ``_reset``), and its
    ``_update_during_reset`` then keeps every unmasked slot as it was. So only
    the masked slots are generated -- at 4-25 ms an instance, regenerating all
    eight collector envs each time one finished would be a real fraction of a
    training step -- and the rest are filled with a placeholder that is thrown
    away.

    The numpy generator is seeded from the global torch RNG, so the
    ``torch.manual_seed(s); env.set_seed(s); env.reset()`` idiom every script
    uses reproduces planted instances too.
    """
    params = tensordict["params"]
    shape = tuple(tensordict.shape)
    n = int(params["n"].reshape(-1)[0].item())
    tc = float(params["tc"].reshape(-1)[0].item())
    tm = float(params["tm"].reshape(-1)[0].item())
    b = int(np.prod(shape)) if shape else 1

    mask = tensordict.get("_reset", None)
    if mask is None:
        mask = np.ones(b, dtype=bool)
    else:
        mask = mask.reshape(-1).cpu().numpy().astype(bool)
        if mask.shape[0] != b:
            mask = np.ones(b, dtype=bool)

    seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    rng = np.random.default_rng(seed)
    cfg = PLANTED_CFG if PLANTED_CFG is not None else PlantedConfig()
    insts, last = [], None
    for i in range(b):
        if mask[i] or last is None:
            last = generate_instance(n, tc, tm, rng, cfg)
        insts.append(last)

    packed = [instance_to_tensors(inst, device="cpu") for inst in insts]
    def stack(key):
        return torch.stack([p[key] for p in packed]).to(self.device)
    grid = stack("grid").reshape(*shape, 3, n, n)
    agent = stack("agent_state").reshape(*shape, 2)
    params = params.clone()
    params["gap"] = stack("gap").reshape(*shape, K_MAX, 2)
    params["oracle_saved"] = stack("oracle_saved").reshape(shape)
    params["noop_saved"] = stack("noop_saved").reshape(shape)

    return TensorDict(
        {
            "grid": grid,
            "t": torch.ones(shape, device=self.device),
            "agent_state": agent,
            "params": params,
        },
        batch_size=shape,
        device=self.device,
    )


def _make_spec(self, td_params):
    # The planted keys exist iff gen_params put them there, so the spec and the
    # reset can never disagree about them.
    planted = ("params", "gap") in td_params.keys(include_nested=True)
    planted_spec = {}
    if planted:
        planted_spec = dict(
            gap=Bounded(
                low=-1,
                high=td_params["params", "n"] - 1,
                shape=(K_MAX, 2),
                dtype=torch.int64,
            ),
            oracle_saved=Bounded(low=0.0, high=1.0, shape=(), dtype=torch.float32),
            noop_saved=Bounded(low=0.0, high=1.0, shape=(), dtype=torch.float32),
        )
    observation_spec = Composite(
        grid=Bounded(
            low=0,
            high=2,
            shape=(3, td_params["params","n"], td_params["params","n"]),
            dtype=torch.bool,
        ),

        t=Bounded(
            low=0.0,
            high=1.0,
            shape=(), 
            dtype=torch.float32
        ),

        agent_state=Bounded(
            low=0,
            high=td_params["params","n"] - 1,
            shape=(2,),
            dtype=torch.int64,
        ),
        # Explicit agent spec instead of dynamic recursion
        params= Composite(            
            tc=Bounded(
                low=0.0,
                high=float("inf"),
                shape=(), 
                dtype=torch.float32
            ),
            tm=Bounded(
                low=0.0,
                high=float("inf"),
                shape=(), 
                dtype=torch.float32
            ),            
            n=Bounded(
                low=1,
                high=int(1e6),  # Arbitrary large number for grid size
                shape=(), 
                dtype=torch.int64
            ),
            distance=Bounded(
                low=0.0,
                high=td_params["params","n"] - 1,
                shape=(), 
                dtype=torch.int64
            ),
            **planted_spec,
            shape=(),
        ),
        
        shape = (),
    )

    action_spec = Bounded(
        low=0,
        high=8,
        shape=(),
        dtype=torch.int64,
    )

    reward_spec = Unbounded(
        shape=(*td_params.shape, 1),
        dtype=torch.float32,
    )

    # for a natively batched env, specs carry the leading batch dims
    if self.batch_size:
        observation_spec = observation_spec.expand(
            *self.batch_size, *observation_spec.shape)
        action_spec = action_spec.expand(
            *self.batch_size, *action_spec.shape)
        reward_spec = reward_spec.expand(
            *self.batch_size, *reward_spec.shape)

    self.observation_spec = observation_spec
    self.state_spec = observation_spec.clone()
    self.action_spec = action_spec
    self.reward_spec = reward_spec


def _set_seed(self, seed: Optional[int]) -> None:
    rng = torch.manual_seed(seed)
    self.rng = rng

def gen_params(batch_size=None) -> TensorDictBase:
    """Returns a ``tensordict`` containing the complex system parameters."""
    if batch_size is None:
        batch_size = []

    params = {
        "tc": TC,  # time to cut a tree
        "tm": TM,  # time to move
        "n": N,  # size of the grid
        "distance": N - 1,  # distance to the fire
    }
    if PLANTED_CFG is not None:
        # Filled in by _reset_planted: the gap cells in cutting order (-1
        # padded), and what the oracle / doing nothing saves on this instance.
        params["gap"] = torch.full((K_MAX, 2), -1, dtype=torch.int64)
        params["oracle_saved"] = torch.tensor(0.0)
        params["noop_saved"] = torch.tensor(0.0)
    agent_params = TensorDict({"params": TensorDict(params, [])}, [])
            
    if batch_size:
        agent_params = agent_params.expand(batch_size).contiguous()
    return agent_params


# Create a _render method for the environment

def render(self, tensordict, mode="rgb_array"):
    """Render the environment."""
    

    grid = tensordict["grid"]
    t = tensordict["t"].cpu().numpy()
    agent_state = tensordict["agent_state"].cpu().numpy()
    #step_counter = tensordict["step_counter"].cpu().numpy()

    img = agents.draw_forest_fire(grid, agent_state, t, step_counter=0, format="tensor")

    return img

class ForestFireEnv(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    batch_locked = False

    def __init__(self, td_params=None, seed=None, device="cpu", batch_size=None, **kwargs):
        if td_params is None:
            td_params = self.gen_params()

        super().__init__(device=device, batch_size=batch_size, **kwargs)
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    # Helpers: _make_step and gen_params
    gen_params = staticmethod(gen_params)
    _make_spec = _make_spec

    # Mandatory methods: _step, _reset and _set_seed
    _reset = _reset
    _step = staticmethod(_step)
    _set_seed = _set_seed
    # Optional methods: _render
    render = render


def make_env(n, tc, tm, batch, device, planted=None):
    """Build the env at a given grid size and action cost.

    N/TC/TM are module globals read by gen_params at call time, which is how the
    training script parameterises them too. ``planted`` is a STEERCA_PLANTED-style
    string (``"1"`` for the defaults) or None for the all-tree grid.
    """
    global N, TC, TM, PLANTED_CFG
    N, TC, TM = n, tc, tm
    PLANTED_CFG = PlantedConfig.from_string(planted) if planted else None
    return ForestFireEnv(device=device, batch_size=[batch])


class RandomRotate(Transform):
    def __init__(self, angles = [90, 180, 270, 360], N=None, device="cpu"):
        # N is accepted for backward compatibility but unused: the grid size is
        # derived from the data, so the transform works for any square grid
        super().__init__(in_keys=[], out_keys=[])  # overwrite same key
        self.angles = angles
        self.device = device

    def _apply_transform(self, tensordict):
        angle = self.angles[torch.randint(len(self.angles), (1,)).item()]
        # grid size is derived from the data so any square grid works
        n = tensordict["grid"].shape[-1]
        tensordict["grid"] = rotate(tensordict["grid"], angle=angle)
        new_pos = rotate_pixel_position(n, tensordict["agent_state"], angle)
        tensordict["agent_state"] = new_pos

        move_mask = tensordict["action"] != 8  # assuming 8 is the cut action
        if move_mask.any():
            idx = move_mask.nonzero(as_tuple=True)[0]
            # only rotate movements
            tensordict["action"][idx] = (tensordict["action"][idx] + angle // 90 * 2) % 8

        return tensordict

class RandomFlip(Transform):
    def __init__(self, N=None, device="cpu"):
        # N is accepted for backward compatibility but unused: the grid size is
        # derived from the data, so the transform works for any square grid
        super().__init__(in_keys=[], out_keys=[])

        self.device = device

        # Action permutations for the 8 move directions (E, NE, N, NW, W, SW, S, SE).
        # horizontal flip (left-right) negates the column component: E<->W, NE<->NW, ...
        self.horizontal_flip = torch.tensor([4, 3, 2, 1, 0, 7, 6, 5], device=self.device)
        # vertical flip (up-down) negates the row component: N<->S, NE<->SE, ...
        self.vertical_flip = torch.tensor([0, 7, 6, 5, 4, 3, 2, 1], device=self.device)

    def _flip(self, tensordict, flip_type):
        flip_grid = horizontal_flip if flip_type == "horizontal" else vertical_flip
        action_map = self.horizontal_flip if flip_type == "horizontal" else self.vertical_flip

        n = tensordict["grid"].shape[-1]
        tensordict["grid"] = flip_grid(tensordict["grid"])
        tensordict["agent_state"] = flip_pixel_position(n, tensordict["agent_state"], flip_type=flip_type)

        move_mask = tensordict["action"] != 8  # assuming 8 is the cut action
        if move_mask.any():
            idx = move_mask.nonzero(as_tuple=True)[0]
            # only remap movements
            tensordict["action"][idx] = action_map[tensordict["action"][idx]]

        return tensordict

    def _apply_transform(self, tensordict):
        # Identity half the time: combined with a uniform rotation this makes the
        # augmentation uniform over all 8 dihedral symmetries — always flipping
        # would make pure rotations (and the original orientation) unreachable.
        coin = torch.rand(1).item()
        if coin < 0.5:
            return tensordict
        if coin < 0.75:
            return self._flip(tensordict, "horizontal")
        return self._flip(tensordict, "vertical")

class RandomAugmentations(Transform):
    def __init__(self, device="cpu"):
        super().__init__(in_keys=[], out_keys=[])
        self.device = device
        self.transforms = [
            RandomRotate(device=self.device),
            RandomFlip(device=self.device)
        ]

    def _apply_transform(self, tensordict):
        for transform in self.transforms:
            tensordict = transform._apply_transform(tensordict)
        return tensordict
    
def rotate_pixel_position(n, position, angle_deg):
    # Convert input to a PyTorch tensor with a float dtype for calculations
    coords_tensor = position.float()
    
    # 1. Define the center of rotation
    center = (n - 1) / 2.0
    
    # 2. Convert the angle from degrees to radians
    angle_rad = torch.tensor(angle_deg * (torch.pi / 180.0), device=coords_tensor.device)
    
    # 3. Create the 2D rotation matrix as a PyTorch tensor
    cos_theta = torch.cos(angle_rad)
    sin_theta = torch.sin(angle_rad)
    rotation_matrix = torch.tensor([
        [cos_theta, -sin_theta],
        [sin_theta,  cos_theta]
    ], dtype=torch.float32, device=coords_tensor.device)

    # 4. Perform the transformation using tensor operations
    #    - Step 4a: Translate coordinates to the origin
    translated_coords = coords_tensor - center
    
    #    - Step 4b: Apply rotation using matrix multiplication (@)
    rotated_translated_coords = translated_coords @ rotation_matrix.T
    
    #    - Step 4c: Translate the coordinates back
    rotated_coords = rotated_translated_coords + center
    
    # 5. Round to the nearest integer and convert to a long tensor
    return torch.round(rotated_coords).round().long()

def flip_pixel_position(n, position, flip_type):
    # Get the original dimensions
    x, y = position[..., 0], position[..., 1]
    n = torch.tensor(n)

    if flip_type == 'horizontal':
        y = n - 1 - y
    elif flip_type == 'vertical':
        x = n - 1 - x

    new_position = torch.stack((x, y), dim=-1)

    return new_position.round().long()

# Create a function that tests a polcy on k episodes and returns the average reward
def test_policy(env, policy, logs, t, k=10, device="cpu", max_steps=2500):
    """Evaluate a policy, returning ``{"return": ..., "saved": ...}``.

    Both numbers matter and they are not the same thing. ``return`` is the mean
    total reward, which since r9 is a *shaped* quantity: felling a tree ahead of
    the fire is priced below what the fire would have charged, so it pays 0.5
    whether or not the cut ends up part of a closed barrier. ``saved`` is the
    mean fraction of the forest still standing at the fixed point, which is what
    the task is actually scored on and what the planner is measured against.

    Select checkpoints on ``saved``. An agent that farms the shaping raises
    ``return`` while ``saved`` stays flat, and selecting on ``return`` would
    then actively prefer it -- the failure mode that produced 0.25% of trees
    saved against the planner's 3.60% at tm+tc = 1.2.

    If ``env`` is batched, every episode runs in parallel as one set of tensor
    ops and ``k`` is taken from the env's batch size. That is far cheaper than
    the sequential path: the per-step cost becomes a single batched policy
    forward instead of k separate ones, so raising the episode count to get a
    trustworthy mean costs almost nothing.

    Episodes in a batch finish at different times, which is the whole difficulty.
    This steps manually and masks with ``alive`` rather than using rollout's
    break semantics: ``break_when_any_done`` would stop at the first episode to
    finish and truncate the rest, while ``break_when_all_done`` keeps stepping
    envs past their terminal state, where the agent goes on cutting trees and
    accruing reward that belongs to no episode. Masking makes a finished
    episode's total final.

    ``max_steps`` must exceed the longest possible episode (~1.4*N/min(tc,tm)),
    or the metric silently measures truncated episodes and is not comparable
    across grid sizes or action costs.
    """
    oracle = noop = None
    if env.batch_size:
        k = int(env.batch_size[0])
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            td = env.reset()
            if ("params", "oracle_saved") in td.keys(include_nested=True):
                oracle = td["params", "oracle_saved"].reshape(k).float()
                noop = td["params", "noop_saved"].reshape(k).float()
            total = torch.zeros(k, device=td.device, dtype=torch.float32)
            alive = torch.ones(k, dtype=torch.bool, device=td.device)
            area = td["grid"].shape[-2] * td["grid"].shape[-1]
            trees = td["grid"][..., TREE, :, :].sum((-2, -1)).reshape(k).float()
            for _ in range(max_steps):
                td = env.step(policy(td))
                nxt = td["next"]
                # credit reward only to episodes that had not already finished
                total += nxt["reward"].reshape(k) * alive.to(total.dtype)
                # freeze the tree count the same way: `alive` still says who was
                # running at the START of this step, so the count on the step an
                # episode ends is recorded, and nothing after it is
                now = nxt["grid"][..., TREE, :, :].sum((-2, -1)).reshape(k).float()
                trees = torch.where(alive, now, trees)
                alive &= ~nxt["done"].reshape(k)
                if not alive.any():
                    break
                td = step_mdp(td)
        avg = total.mean().item()
        per_episode = trees / area
        saved = per_episode.mean().item()
    else:
        total_rewards, saved_fracs, oracles, noops = [], [], [], []
        for _ in range(k):
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                # execute a rollout with the trained policy
                eval_rollout = env.rollout(max_steps, policy, break_when_any_done=True)
                total_rewards.append(eval_rollout["next", "reward"].sum().item())
                if ("params", "oracle_saved") in eval_rollout.keys(include_nested=True):
                    oracles.append(float(eval_rollout["params", "oracle_saved"][0]))
                    noops.append(float(eval_rollout["params", "noop_saved"][0]))
                # break_when_any_done stops at the terminal step, so the last
                # frame is the fixed point this episode reached
                final = eval_rollout["next", "grid"][-1]
                area = final.shape[-2] * final.shape[-1]
                saved_fracs.append(float(final[..., TREE, :, :].sum()) / area)

                del eval_rollout
        avg = float(np.mean(total_rewards))
        per_episode = torch.tensor(saved_fracs)
        saved = per_episode.mean().item()
        if oracles:
            oracle, noop = torch.tensor(oracles), torch.tensor(noops)

    logs.log_scalar("test/avg_total_reward", avg, step=t)
    logs.log_scalar("test/trees_saved", saved, step=t)
    out = {"return": avg, "saved": saved}
    if oracle is not None:
        # Planted instances carry their own optimum, so the score can be
        # normalised per episode: `oracle_fraction` is what share of the
        # oracle's trees the policy kept, and `oracle_gain` the share of what
        # was actually at stake -- the trees between doing nothing and the
        # oracle. Both are means of per-episode ratios, so a run of hard
        # instances does not read as a worse policy. Above 1 is legitimate: the
        # oracle only cuts the gap, and a policy may also save some of the fire
        # side. Checkpoints are still selected on `saved`.
        per_episode = per_episode.to(oracle.device)
        frac = per_episode / oracle.clamp_min(1e-6)
        gain = (per_episode - noop) / (oracle - noop).clamp_min(1e-6)
        out["oracle_saved"] = oracle.mean().item()
        out["oracle_fraction"] = frac.mean().item()
        out["oracle_gain"] = gain.mean().item()
        logs.log_scalar("test/oracle_saved", out["oracle_saved"], step=t)
        logs.log_scalar("test/oracle_fraction", out["oracle_fraction"], step=t)
        logs.log_scalar("test/oracle_gain", out["oracle_gain"], step=t)
    return out

# Named test_* for history, but it is an evaluation routine, not a pytest test;
# this stops pytest collecting it wherever a test module imports it.
test_policy.__test__ = False
