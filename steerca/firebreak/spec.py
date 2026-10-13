"""Firebreak written as a :class:`~steerca.core.spec.CASpec`.

``ForestFireEnv`` (in :mod:`steerca.firebreak.env`) is the environment every
result in the papers was measured on, and it stays the default for training and
evaluation. This spec expresses the same task through the generic API.
``tests/test_spec_equivalence.py`` checks that ``SteerCAEnv(firebreak_spec())``
reproduces ``ForestFireEnv`` bit for bit, so the spec doubles as the reference
example of a complete environment, and as the starting point for a variant:

    from dataclasses import replace
    windy = replace(firebreak_spec(), name="firebreak-wind", ca_step=my_step, symmetric=False)

Planted instances are not part of the spec; they remain a ``ForestFireEnv``
feature (``STEERCA_PLANTED``).
"""
from __future__ import annotations

from steerca.core.spec import CASpec, default_episode_cap
from steerca.firebreak import agents, ca
from steerca.firebreak.ca import TREE
from steerca.firebreak.env import all_tree_reset


def _reset(shape, n, params, device):
    distance = int(params["distance"].reshape(-1)[0].item())
    return all_tree_reset(shape, n, distance, device)


def trees_saved(grid):
    """Fraction of the grid still forested: Firebreak's score."""
    area = grid.shape[-2] * grid.shape[-1]
    return grid[..., TREE, :, :].sum((-2, -1)).float() / area


def firebreak_spec() -> CASpec:
    return CASpec(
        name="firebreak",
        n_states=3,                       # DIRT, TREE, FIRE (one-hot)
        ca_step=ca.step,                  # a tree with a burning Moore neighbour ignites; fire burns out
        action=agents.action,             # 8 moves + cut (TREE -> DIRT at the agent's cell)
        reward_grid=agents.reward_grid,   # -(trees burnt)/n^2, plus the optional cut-tree charges
        reward_action=agents.reward_action,  # -0.5/n^2 per cut action
        done=agents.done,                 # no fire left
        reset=_reset,
        time=agents.time,
        score=trees_saved,
        extra_params=lambda n: {"distance": n - 1},
        episode_cap=default_episode_cap,
        symmetric=True,
    )
