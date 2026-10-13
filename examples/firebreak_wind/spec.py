"""A Firebreak variant: wind.

Everything is Firebreak's (the agent, the time model, the reward, the reset and
the trees-saved score) except the ignition rule. With a wind blowing from the
west, a tree catches fire only from a burning neighbour to its west, north or
south: fire runs east and spreads sideways, but never burns back upwind.

The change is two lines of the spec: a new ``ca_step``, and ``symmetric=False``,
because the rule is no longer the same under rotations and reflections and the
training augmentation would otherwise teach the policy that fire can run in any
direction.

    STEERCA_ENV=examples.firebreak_wind.spec:make_spec \\
    STEERCA_N=16 STEERCA_TC=0.2 STEERCA_TM=0.2 python -m steerca.train

Two caveats. The barrier planner (``steerca.firebreak.planner``) assumes the
isotropic rule, so it is not a reference here, and the optional "imminent" cut
pricing (``STEERCA_CUT_TREE_IMMINENT``) still counts burning neighbours on every
side. Both are fine at the defaults, but they are what a careful variant would
revisit.

    python examples/firebreak_wind/run.py   # idle agent, calm vs windy
"""
from dataclasses import replace

import torch
import torch.nn.functional as F

from steerca.firebreak.ca import DIRT, FIRE, TREE
from steerca.firebreak.spec import firebreak_spec

# Which burning neighbours can ignite a cell. conv2d is a cross-correlation, so
# entry [i][j] looks at the neighbour at (row + i - 1, col + j - 1): the left
# column is the western neighbours, the right column (all zero) the eastern ones.
WIND_FROM_WEST = torch.tensor([[1., 1., 0.],
                               [1., 0., 0.],
                               [1., 1., 0.]]).view(1, 1, 3, 3)


def windy_step(grid):
    """Firebreak's rule with the wind kernel: fire burns out, trees downwind ignite."""
    fire = grid[..., FIRE, :, :]
    tree = grid[..., TREE, :, :]
    flat = fire.reshape(-1, 1, *fire.shape[-2:]).float()
    burning = F.conv2d(flat, WIND_FROM_WEST.to(grid.device), padding=1).view(fire.shape) >= 1
    new_fire = tree & burning
    return torch.stack([grid[..., DIRT, :, :] | fire, tree & ~new_fire, new_fire], dim=-3)


def make_spec():
    return replace(firebreak_spec(), name="firebreak-wind",
                   ca_step=windy_step, symmetric=False)

