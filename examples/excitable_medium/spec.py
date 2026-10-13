"""A new CA from scratch: steering waves in an excitable medium.

The automaton is Greenberg–Hastings, the classic cellular model of excitable
media (heart tissue, nerve fibre, the Belousov–Zhabotinsky reaction). A resting
cell becomes **excited** when at least ``THRESHOLD`` of its eight neighbours are
excited; an excited cell becomes **refractory** for one tick; a refractory cell
**recovers** to rest. Excitation therefore travels as waves that leave the
medium behind them able to fire again, and two waves that meet annihilate,
because each runs into the other's refractory tail. Neither happens to a fire.

The agent can make the resting cell it stands on **inert** (it never fires
again), at a cost. The task: keep as much of the medium as possible from ever
being excited. A fifth plane, the **scar**, records every cell that has fired,
so the score can be read from the final grid even though the medium recovers.

With ``THRESHOLD = 2`` a single excited cell cannot start a wave, and a front
cannot get through a one-cell gap in a wall of inert cells, since the cell in
the gap never sees two excited neighbours at once. A wall can therefore have
holes that a firebreak could not, which is the kind of difference between
automata that SteerCA is built to expose. ``python
examples/excitable_medium/run.py`` checks both on the live rule, and scores an
idle and a random agent as baselines.

    STEERCA_ENV=examples.excitable_medium.spec:make_spec \\
    STEERCA_N=16 STEERCA_TC=0.2 STEERCA_TM=0.2 STEERCA_FRAMES=1000000 \\
    python -m steerca.train
"""
import torch
import torch.nn.functional as F

from steerca.core import CASpec, local_action, random_position

# Grid planes. The first four are one-hot (each cell is in exactly one state);
# SCAR is a flag carried alongside them.
REST, EXCITED, REFRACTORY, INERT, SCAR = range(5)
N_PLANES = 5
THRESHOLD = 2          # excited neighbours needed to fire
BLOCK_COST = 0.5       # what making a cell inert costs, in cells of medium
SOURCES = (1, 3)       # 2x2 excited seeds per episode, min and max

_MOORE = torch.tensor([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]]).view(1, 1, 3, 3)


def ca_step(grid):
    """One Greenberg–Hastings tick. Returns a new tensor; ``grid`` is untouched."""
    rest, exc, ref, inert, scar = grid.unbind(dim=-3)
    flat = exc.reshape(-1, 1, *exc.shape[-2:]).float()
    neighbours = F.conv2d(flat, _MOORE.to(grid.device), padding=1).view(exc.shape)
    fire = rest & (neighbours >= THRESHOLD)
    return torch.stack([
        (rest & ~fire) | ref,     # REST: stays at rest unless it fires; refractory recovers
        fire,                     # EXCITED
        exc,                      # REFRACTORY: what was excited
        inert,                    # INERT never changes
        scar | fire,              # SCAR: ever excited
    ], dim=-3)


def _make_inert(grid, b, r, c, mask):
    """The cell action: a resting cell under the agent becomes inert."""
    hit = mask & grid[b, REST, r, c]
    grid[b, REST, r, c] &= ~hit
    grid[b, INERT, r, c] |= hit


def reward_grid(grid, mid_grid, next_grid):
    """Minus the medium newly scarred by the CA advance, per unit area."""
    area = grid.shape[-2] * grid.shape[-1]
    new = (mid_grid[..., SCAR, :, :] & ~grid[..., SCAR, :, :]).sum((-2, -1))
    return -new.float() / area


def reward_action(actions, next_grid, agent_state):
    area = next_grid.shape[-2] * next_grid.shape[-1]
    return torch.where(actions == 8, -BLOCK_COST / area, 0.0).float()


def done(grid):
    """Nothing excited means nothing can ever fire again: a fixed point."""
    return ~grid[..., EXCITED, :, :].any(dim=(-2, -1))


def score(grid):
    """Fraction of the medium never excited and not made inert."""
    area = grid.shape[-2] * grid.shape[-1]
    lost = grid[..., SCAR, :, :] | grid[..., INERT, :, :]
    return 1.0 - lost.sum((-2, -1)).float() / area


def reset(shape, n, params, device):
    """Resting medium with 1-3 excited 2x2 seeds; the agent anywhere."""
    b = 1
    for s in shape:
        b *= s
    grid = torch.zeros((b, N_PLANES, n, n), dtype=torch.bool, device=device)
    grid[:, REST] = True
    count = torch.randint(SOURCES[0], SOURCES[1] + 1, (b,), device=device)
    corners = torch.randint(0, n - 1, (b, SOURCES[1], 2), device=device)
    for k in range(SOURCES[1]):
        use = count > k
        idx = torch.nonzero(use).flatten()
        for dr in (0, 1):
            for dc in (0, 1):
                r, c = corners[idx, k, 0] + dr, corners[idx, k, 1] + dc
                grid[idx, REST, r, c] = False
                grid[idx, EXCITED, r, c] = True
                grid[idx, SCAR, r, c] = True
    pos = random_position((b,), n, device)
    return grid.view(*shape, N_PLANES, n, n), pos.view(*shape, 2)


def make_spec() -> CASpec:
    return CASpec(
        name="excitable",
        n_states=N_PLANES,
        ca_step=ca_step,
        action=local_action(_make_inert),
        reward_grid=reward_grid,
        reward_action=reward_action,
        done=done,
        reset=reset,
        score=score,
        # Fronts cross the grid at one cell per tick, as fire does, so
        # Firebreak's cap leaves the same headroom.
        symmetric=True,
    )
