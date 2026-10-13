"""The SteerCA agent: eight moves and one action on its own cell."""
from __future__ import annotations

import torch

# Row/column offset of each action. Actions 0..7 move the agent one cell
# (E, NE, N, NW, W, SW, S, SE, counter-clockwise from east); 8 acts on the cell
# it stands on and leaves it in place. The rotation/reflection augmentation in
# steerca.firebreak.env relies on this order.
DELTAS = torch.tensor([
    [0, 1], [-1, 1], [-1, 0], [-1, -1],
    [0, -1], [1, -1], [1, 0], [1, 1],
    [0, 0],
], dtype=torch.long)

_CACHE = {}


def deltas(device) -> torch.Tensor:
    d = _CACHE.get(device)
    if d is None:
        d = _CACHE[device] = DELTAS.to(device)
    return d


def local_action(cell_effect):
    """Build a spec's ``action`` from what the cell action does to one cell.

    ``cell_effect(grid, batch, row, col, mask)`` receives index tensors for the
    agent's cell in every batch element and the boolean ``mask`` of elements that
    chose the cell action, and must update ``grid`` in place at
    ``grid[batch, :, row, col]`` where ``mask`` holds. Moves clamp at the border.

    Example, an action that turns a ``TREE`` into ``DIRT``::

        def cut(grid, b, r, c, mask):
            hit = mask & grid[b, TREE, r, c]
            grid[b, TREE, r, c] &= ~hit
            grid[b, DIRT, r, c] |= hit
    """
    def action(actions, grids, agent_state):
        pos = agent_state.long()
        n = grids.shape[-1]
        batch = torch.arange(actions.shape[0], device=actions.device)
        cell_effect(grids, batch, pos[:, 0], pos[:, 1], actions == 8)
        new_pos = torch.clamp(pos + deltas(actions.device)[actions], 0, n - 1)
        return grids, new_pos

    return action


def random_position(shape, n, device):
    """A uniformly random cell per batch element, ``(*shape, 2)`` int64."""
    return torch.randint(0, n, size=(*shape, 2), device=device)
