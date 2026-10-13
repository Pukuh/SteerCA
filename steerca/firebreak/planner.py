"""A hand-designed planner for the forest-fire containment task.

This is the non-learned baseline PPO is measured against. It is not a solver:
choosing which cells to cut, in which order, under per-cell deadlines is a
travelling-salesman-with-time-windows problem. It is a *structured* planner that
exploits four exact facts about the environment, and within the structures it
searches it is close to optimal.

The four facts
--------------
1. A cut set blocks fire iff it is **4-connected**. Fire spreads diagonally, so a
   barrier with a diagonal step in it has a hole at the corner that step cuts
   across -- the same reason a diagonal firebreak is useless. Rectangle borders
   are 4-connected in walk order; the spiral walk is restricted to N/S/E/W to
   make it so.
2. Fire spreads one cell per CA tick in the Chebyshev (Moore) metric, so the tick
   at which it reaches each cell is a Moore-metric BFS from the current fire
   through tree cells. Dirt is absorbing, so that BFS already accounts for every
   cut made so far, and for the grid edge.
3. The agent covers one cell per ``tm`` ticks, also in the Chebyshev metric
   (diagonal moves cost the same as straight ones), so travel time between two
   cells is exactly ``chebyshev_distance * tm``. A plan's whole schedule is
   therefore a closed-form cumulative sum, not a simulation.
4. Reward is ``-(trees lost + 0.5 * cut actions) / N^2``, and a tree the agent
   fells is a tree lost like any other. So a cut costs **1.5** trees: 1 for the
   tree, 0.5 for the action. That is the exchange rate the planner optimises,
   and a firebreak is worth building only when it saves more than 1.5 trees per
   cell. (Before the reward was fixed, ``reward_grid`` differenced the grid
   *before* the action, so a felled tree never registered as lost and cutting a
   doomed one PAID 0.5 -- reward that could be farmed without saving anything.)

Two families of barrier
-----------------------
**Rectangles.** Every rectangle with a corner range of ``[-1, n]`` per axis,
scored in two roles at once: those strictly containing the fire (cut the border,
the outside is saved) and those strictly avoiding it (cut the border, the inside
is saved -- a refuge when containment is hopeless). Letting a corner sit off-grid
is what makes the grid edge free, so a fire in a corner needs only an L-shaped
arc and "wall off the left third" is a rectangle whose other three sides are
outside the grid. All ~30k of them are scored in a handful of vectorised
prefix-sum lookups.

**Spirals.** A rectangle can put each *side* at its own distance from the fire,
but it cannot change radius partway along a side. A spiral can: it hugs the fire
where the agent has time and drifts outward where it does not. This is also the
only family that works at all above ``tc + tm = 1/8``. Closing a ring of radius r
costs ``8r`` cells while buying only ``r`` ticks of warning, so rings need
``tc + tm < 1/8``; a spiral instead grows as it goes, needing
``dr >= 8r(tc+tm)/(1-(tc+tm))`` per turn -- geometric growth, achievable for any
``tc + tm < 1``. On a finite grid the edge both bounds the price of that growth
and closes the curve early. The walk is greedy, so it is not guaranteed to close;
it is scored by flood fill rather than by geometry, and a spiral that leaks
simply prices itself out against the rectangles.

The algorithm
-------------
Once per CA tick -- between ticks nothing about the fire changes, only the
agent's own position, which the plan already anticipates -- four candidates
compete on predicted loss, ``1.5 * cuts_still_needed + trees that still burn``:

  a. the plan already in progress, re-costed under the current state and given a
     hysteresis discount. Re-costed rather than re-derived: a spiral re-derived
     from a moved agent and a spread fire is a *different* walk each tick, so the
     agent would pay for half of one barrier and half of another;
  b. the best rectangle. Candidates are walked in increasing loss and the first
     *schedulable* one wins -- traversing its border must cut every cell strictly
     before fire arrives there -- and since loss does not depend on the schedule,
     the first feasible one is the best feasible one;
  c. the best spiral, over a few entry radii and both directions;
  d. doing nothing, so a hopeless fire is left to burn rather than paid for.

Only cells that are still trees *and* still reachable by fire ever count as cuts,
which is what lets a half-built barrier be finished cheaply. Because ``arrival``
assumes no further cuts, the feasibility test is conservative: a plan the planner
accepts always executes.

Use it as a drop-in policy -- ``policy(tensordict)`` writes ``action`` -- so the
same harness scores it and a PPO actor identically.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from steerca.firebreak.ca import DIRT, TREE, FIRE

# Large enough to mean "never", small enough that adding a margin cannot overflow.
INF = np.iinfo(np.int32).max // 4

CUT = 8

# What one felled tree costs, in trees. This is 1 and stays 1: a tree the agent
# cuts is a tree that does not survive, exactly like one the fire takes, so
# `CUT_COST * cuts + burnt` is the total the forest lost.
#
# It deliberately does NOT track the environment's reward, which since r9 prices
# a cut by whether the fire was about to reach it (0.5 ahead of the fire, 1.5 on
# top of it) to give a learning agent a signal. That shaping is a training aid;
# the planner is the yardstick for the metric everyone is actually scored on, so
# it optimises trees saved and nothing else. Return is consequently NOT
# comparable between the planner and a PPO agent any more -- compare trees saved.
CUT_COST = 1.0

# (d_row, d_col) -> action index, mirroring _DELTAS in steerca.firebreak.agents.
_ACTION_OF_DELTA = {
    (0, 1): 0, (-1, 1): 1, (-1, 0): 2, (-1, -1): 3,
    (0, -1): 4, (1, -1): 5, (1, 0): 6, (1, 1): 7,
}


# --------------------------------------------------------------------------- #
# Grid primitives
# --------------------------------------------------------------------------- #
def _dilate(mask: np.ndarray) -> np.ndarray:
    """One Moore-neighbourhood dilation (the CA's own spreading step)."""
    h, w = mask.shape
    padded = np.zeros((h + 2, w + 2), dtype=bool)
    padded[1:-1, 1:-1] = mask
    out = np.zeros_like(mask)
    for dr in range(3):
        for dc in range(3):
            out |= padded[dr:dr + h, dc:dc + w]
    return out


def fire_arrival(tree: np.ndarray, fire: np.ndarray) -> np.ndarray:
    """Ticks from now until fire reaches each cell; INF where it never does.

    A Moore-metric BFS from the current fire, through tree cells only. Dirt is
    absorbing -- fire neither enters nor crosses it -- so every cut already made
    is accounted for, and cells the fire can no longer reach come back INF and
    drop out of every plan.
    """
    arrival = np.full(tree.shape, INF, dtype=np.int64)
    arrival[fire] = 0
    frontier = fire.copy()
    tick = 0
    while frontier.any():
        tick += 1
        nxt = _dilate(frontier) & tree & (arrival == INF)
        if not nxt.any():
            break
        arrival[nxt] = tick
        frontier = nxt
    return arrival


def _prefix(a: np.ndarray) -> np.ndarray:
    """Inclusive 2D prefix sum, zero-padded, for O(1) rectangle sums."""
    return np.pad(a.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def _box_sum(P, r0, r1, c0, c1, n):
    """Sum over rows [r0, r1] x cols [c0, c1], clipped to the grid.

    Bounds may sit outside [0, n-1]: clipping (rather than rejecting) is what
    lets a rectangle side lie off-grid and cost nothing, so the grid edge acts
    as a free barrier. Empty ranges collapse to zero.
    """
    a = np.clip(r0, 0, n)
    b = np.maximum(np.clip(r1 + 1, 0, n), a)
    c = np.clip(c0, 0, n)
    d = np.maximum(np.clip(c1 + 1, 0, n), c)
    return P[b, d] - P[a, d] - P[b, c] + P[a, c]


# --------------------------------------------------------------------------- #
# Barrier geometry
# --------------------------------------------------------------------------- #
def _rect_perimeter(r0, r1, c0, c1):
    """The border cells of a rectangle, in cyclic (walkable) order.

    Consecutive cells are adjacent, so walking the list costs exactly one move
    per cell. Degenerate rectangles (a single row, column or cell) fall back to
    the segment itself.
    """
    if r0 == r1 and c0 == c1:
        return np.array([r0]), np.array([c0])
    if r0 == r1:
        cols = np.arange(c0, c1 + 1)
        return np.full_like(cols, r0), cols
    if c0 == c1:
        rows = np.arange(r0, r1 + 1)
        return rows, np.full_like(rows, c0)
    top_c = np.arange(c0, c1 + 1)
    right_r = np.arange(r0 + 1, r1 + 1)
    bot_c = np.arange(c1 - 1, c0 - 1, -1)
    left_r = np.arange(r1 - 1, r0, -1)
    rows = np.concatenate([np.full_like(top_c, r0), right_r,
                           np.full_like(bot_c, r1), left_r])
    cols = np.concatenate([top_c, np.full_like(right_r, c1),
                           bot_c, np.full_like(left_r, c0)])
    return rows, cols


def _clip_to_grid(rows, cols, n, closed):
    """Keep the in-grid part of a border, still in traversal order.

    ``closed`` says whether the border is a genuine loop (a degenerate rectangle
    is a segment, which has ends). Returns ``(rows, cols, cyclic)``, or None if
    none of it is on the grid. When
    the rectangle pokes outside, the survivors are rotated so the sequence starts
    at the beginning of an arc rather than in the middle of one, and the loop is
    no longer closed -- the grid edge closes it instead, for free.
    """
    inside = (rows >= 0) & (rows < n) & (cols >= 0) & (cols < n)
    if inside.all():
        return rows, cols, closed
    if not inside.any():
        return None
    starts = np.flatnonzero(inside & ~np.roll(inside, 1))
    idx = (starts[0] + np.arange(len(inside))) % len(inside)
    keep = idx[inside[idx]]
    return rows[keep], cols[keep], False


# --------------------------------------------------------------------------- #
# Scheduling
# --------------------------------------------------------------------------- #
def _schedule(order_r, order_c, agent, t0, tc, tm, arrival, margin):
    """Finish time of cutting `order` in sequence, or None if it is too slow.

    The environment's clock makes this exact rather than simulated. With ``t0``
    of the current tick left, cumulative action time ``tau`` has advanced the CA
    by ``floor(tau - t0) + 1`` ticks (zero while ``tau < t0``), and the cut is
    applied *after* that advance -- so a cell survives only if that tick count is
    still strictly below its fire-arrival time.
    """
    if len(order_r) == 0:
        return 0.0
    prev_r = np.concatenate(([agent[0]], order_r[:-1]))
    prev_c = np.concatenate(([agent[1]], order_c[:-1]))
    travel = np.maximum(np.abs(order_r - prev_r), np.abs(order_c - prev_c))
    tau = np.cumsum(travel * tm + tc)
    ticks = np.floor(tau - t0).astype(np.int64) + 1
    if np.any(ticks + margin >= arrival[order_r, order_c]):
        return None
    return float(tau[-1])


def _orderings(rows, cols, cyclic, agent, arrival):
    """Traversal orders worth trying for one barrier.

    An open arc must be walked end to end, so there are two. A closed loop can be
    entered anywhere: the entry points that matter are the one nearest the agent
    (least travel) and the one fire reaches first (tightest deadline served
    first), each in both directions.
    """
    if not cyclic:
        yield rows, cols
        yield rows[::-1], cols[::-1]
        return
    m = len(rows)
    dist = np.maximum(np.abs(rows - agent[0]), np.abs(cols - agent[1]))
    steps = np.arange(m)
    for start in {int(np.argmin(dist)), int(np.argmin(arrival[rows, cols]))}:
        yield rows[(start + steps) % m], cols[(start + steps) % m]
        yield rows[(start - steps) % m], cols[(start - steps) % m]


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #
def _corner_pairs(n):
    """All (lo, hi) side pairs with lo <= hi drawn from [-1, n].

    The off-grid values -1 and n are the "this side is the grid edge" options.
    """
    lo, hi = np.triu_indices(n + 2)
    return lo - 1, hi - 1


def _range_max_tables(deadline, n):
    """Dense range-max tables over row and column segments of ``deadline``.

    ``rows[r, i, j]`` is the max over ``deadline[r, i:j+1]`` and ``cols[c, i, j]``
    the max over ``deadline[i:j+1, c]``. A rectangle's border is exactly two row
    segments and two column segments, so these four lookups give the latest
    deadline on any border in O(1) -- which is what makes the feasibility
    pre-filter cheap enough to apply to every rectangle at once. O(n^3) entries,
    a few thousand at these grid sizes.
    """
    rows = np.full((n, n, n), -1, dtype=np.int64)
    cols = np.full((n, n, n), -1, dtype=np.int64)
    for i in range(n):
        row_run = np.full(n, -1, dtype=np.int64)
        col_run = np.full(n, -1, dtype=np.int64)
        for j in range(i, n):
            row_run = np.maximum(row_run, deadline[:, j])
            col_run = np.maximum(col_run, deadline[j, :])
            rows[:, i, j] = row_run
            cols[:, i, j] = col_run
    return rows, cols


def _border_deadline(table, line, lo, hi, n):
    """Latest deadline along one border side, or -1 when that side is off-grid."""
    on_grid = (line >= 0) & (line < n) & (hi >= 0) & (lo < n)
    idx = (np.clip(line, 0, n - 1), np.clip(lo, 0, n - 1), np.clip(hi, 0, n - 1))
    return np.where(on_grid, table[idx], -1)


def score_rectangles(doomed, fire, arrival, n, cut_cost, tc, tm):
    """Predicted loss of every rectangle barrier, and its role.

    Vectorised over all ~((n+2)(n+3)/2)^2 rectangles via prefix sums. Returns
    ``(loss, r0, r1, c0, c1, contains)`` flattened, restricted to rectangles that
    are both admissible and not provably too slow.

    Admissible means the fire is strictly inside the border (containment, the
    outside is saved) or strictly off the closed rectangle (a refuge, the inside
    is saved). Fire sitting *on* the border means it has already crossed.

    The speed filter is a necessary condition, so it never discards a plan that
    would have worked: ``k`` cuts take at least ``k*tc + (k-1)*tm`` even with the
    cells laid out consecutively and the agent starting on one, and *some* cell
    is cut last, whose deadline is at most the border's latest. Without it the
    hard settings -- where thousands of tight barriers are all just out of reach
    -- spend their whole budget on doomed schedule checks.
    """
    Pd, Pf = _prefix(doomed), _prefix(fire)
    total_d, total_f = int(doomed.sum()), int(fire.sum())

    r0, r1 = _corner_pairs(n)
    c0, c1 = _corner_pairs(n)
    r0, r1 = r0[:, None], r1[:, None]
    c0, c1 = c0[None, :], c1[None, :]

    box_d = _box_sum(Pd, r0, r1, c0, c1, n)
    int_d = _box_sum(Pd, r0 + 1, r1 - 1, c0 + 1, c1 - 1, n)
    cuts = box_d - int_d                        # doomed cells needing a cut

    contains = _box_sum(Pf, r0 + 1, r1 - 1, c0 + 1, c1 - 1, n) == total_f
    avoids = _box_sum(Pf, r0, r1, c0, c1, n) == 0

    # Containment saves the outside; a refuge saves the inside.
    loss = np.where(contains, cut_cost * cuts + int_d,
                    cut_cost * cuts + (total_d - box_d))

    deadline = np.where(doomed, arrival, -1)
    row_tab, col_tab = _range_max_tables(deadline, n)
    latest = np.maximum.reduce([
        _border_deadline(row_tab, r0, c0, c1, n),
        _border_deadline(row_tab, r1, c0, c1, n),
        _border_deadline(col_tab, c0, r0, r1, n),
        _border_deadline(col_tab, c1, r0, r1, n),
    ])
    work = cuts * tc + np.maximum(cuts - 1, 0) * tm
    # A tick has at most 1.0 left, so the last cut lands no earlier than this.
    earliest_last = np.floor(work - 1.0).astype(np.int64) + 1
    in_time = (cuts == 0) | (earliest_last < latest)

    keep = (contains | avoids) & in_time
    shape = np.broadcast_shapes(r0.shape, c0.shape)
    return [np.broadcast_to(x, shape)[keep]
            for x in (loss, r0, r1, c0, c1, contains)]



# --------------------------------------------------------------------------- #
# Spiral barriers
# --------------------------------------------------------------------------- #
# The spiral steps only N/S/E/W. Fire spreads diagonally, so a barrier with a
# diagonal step in it has a hole at the corner the step cuts across -- the same
# reason a diagonal firebreak does not hold. A 4-connected curve separates
# 8-connected regions, and a rectangle border is 4-connected in walk order,
# which is why that family never needed the restriction spelled out.
_STEPS = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _polar(n, origin):
    """Angle and Chebyshev radius of every cell about ``origin``.

    Precomputed per re-plan because the spiral walk queries them once per
    candidate neighbour, and ``atan2`` in the inner loop dominates otherwise.
    """
    dr = np.arange(n)[:, None] - origin[0]
    dc = np.arange(n)[None, :] - origin[1]
    return (np.arctan2(np.broadcast_to(dr, (n, n)), np.broadcast_to(dc, (n, n))),
            np.maximum(np.abs(dr), np.abs(dc)))


def _score_cut_set(tree, fire, cut, n_cuts, cut_cost):
    """Exact loss of a cut set: what still burns, plus what the cutting cost.

    A flood fill rather than a geometric argument, which is what makes an
    arbitrarily-shaped barrier safe to propose. A leaky spiral is not rejected
    by a special case -- the fill simply finds the trees it fails to save and
    the loss comes out bad, so it loses to the rectangles on its own merits.
    """
    remaining = tree & ~cut
    burn = int((remaining & (fire_arrival(remaining, fire) < INF)).sum())
    return cut_cost * n_cuts + burn


def spiral_barrier(doomed, tree, fire, arrival, angle, radius, start, sigma,
                   agent, t0, tc, tm, cut_cost, margin, max_steps):
    """Walk one outward spiral around the fire, returning its best prefix.

    This is the family rectangles cannot express: a barrier that sits tight
    where the agent has time and drifts outward where it does not, changing
    radius *within* a side rather than only between sides.

    It is also the only family that works at all above ``tc + tm = 1/8``. Closing
    a ring of radius r costs ``8r`` cells while buying only ``r`` ticks of
    warning, so rings need ``tc + tm < 1/8``. A spiral instead grows its radius
    as it goes: one turn costs ``8r + dr`` cells, so staying ahead needs
    ``dr >= 8r(tc+tm)/(1-(tc+tm))`` -- geometric growth, which is achievable for
    any ``tc + tm < 1``. The price is radius, and on a finite grid the grid edge
    both bounds that price and closes the curve early.

    The walk is greedy: at each step take the safe N/S/E/W neighbour that
    advances the angle about the fire while staying at the smallest radius.
    Safety is checked against the same clock the executor uses, so it only ever
    proposes cuts it can make, and ``visited`` guarantees termination. What the
    walk cannot guarantee is that the curve closes, so the result is scored by
    flood fill rather than by geometry: a spiral that leaks simply prices itself
    out against the rectangles.
    """
    n = doomed.shape[0]
    cut = np.zeros((n, n), dtype=bool)
    visited = np.zeros((n, n), dtype=bool)
    early = np.zeros((n, n), dtype=bool)
    order_r, order_c = [], []

    r, c = start
    tau = max(abs(r - agent[0]), abs(c - agent[1])) * tm
    best = None

    def commit(rr, cc, when):
        """Time after cutting (rr, cc) if it needs it, or None if fire wins."""
        if not doomed[rr, cc]:
            return when
        when += tc
        if math.floor(when - t0) + 1 + margin >= arrival[rr, cc]:
            return None
        return when

    tau = commit(r, c, tau)
    if tau is None:
        return None
    visited[r, c] = True
    if doomed[r, c]:
        cut[r, c] = True
        early[r, c] = True
        order_r.append(r)
        order_c.append(c)

    theta = angle[r, c]
    total = 0.0
    on_edge = r in (0, n - 1) or c in (0, n - 1)

    for _ in range(max_steps):
        pick = None
        for dr, dc in _STEPS:
            qr, qc = r + dr, c + dc
            if not (0 <= qr < n and 0 <= qc < n) or visited[qr, qc]:
                continue
            if radius[qr, qc] == 0:           # the ignition cell is not a barrier
                continue
            step = sigma * ((angle[qr, qc] - theta + math.pi) % (2 * math.pi) - math.pi)
            # Wind one way only, and never jump across the fire -- except that a
            # step *outward* is always allowed however the angle moves. On a grid
            # a radial step shifts atan2 slightly, backwards on half the
            # perimeter; forbidding it there strands the walk at the radius it
            # started on, which is the one move a spiral cannot do without.
            outward = radius[qr, qc] > radius[r, c]
            if not outward and (step < -1e-9 or step > math.pi / 2 + 1e-9):
                continue
            when = commit(qr, qc, tau + tm)
            if when is None:
                continue
            key = (radius[qr, qc], -step)     # hug the fire, then make progress
            if pick is None or key < pick[0]:
                pick = (key, qr, qc, when, step)
        if pick is None:
            break                             # boxed in; the prefix may still close

        _, r, c, tau, step = pick
        visited[r, c] = True
        if doomed[r, c]:
            cut[r, c] = True
            order_r.append(r)
            order_c.append(c)
            if total < math.pi / 2:
                early[r, c] = True
        total += step
        theta = angle[r, c]

        # Two ways a spiral closes: it comes back alongside where it began, or
        # it runs into the grid edge, which finishes the curve for free.
        looped = total >= 1.75 * math.pi and _touches(early, r, c, n)
        edge = r in (0, n - 1) or c in (0, n - 1)
        if order_r and (looped or (edge and not on_edge and total >= math.pi / 2)):
            loss = _score_cut_set(tree, fire, cut, len(order_r), cut_cost)
            if best is None or loss < best[0]:
                best = (loss, np.array(order_r), np.array(order_c))
            if looped:
                break
        on_edge = edge

    if best is None and order_r:
        loss = _score_cut_set(tree, fire, cut, len(order_r), cut_cost)
        best = (loss, np.array(order_r), np.array(order_c))
    return best


def _touches(mask, r, c, n):
    """True if any Moore neighbour of (r, c) is set."""
    return bool(mask[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2].any())


def _ring_entry(origin, rho, agent, n):
    """The cell at Chebyshev radius ``rho`` from ``origin`` nearest the agent."""
    rows, cols = _rect_perimeter(origin[0] - rho, origin[0] + rho,
                                 origin[1] - rho, origin[1] + rho)
    ok = (rows >= 0) & (rows < n) & (cols >= 0) & (cols < n)
    if not ok.any():
        return None
    rows, cols = rows[ok], cols[ok]
    j = int(np.argmin(np.maximum(np.abs(rows - agent[0]), np.abs(cols - agent[1]))))
    return int(rows[j]), int(cols[j])


def best_spiral(doomed, tree, fire, arrival, origin, agent, t0, tc, tm,
                cut_cost, margin, starts=6):
    """Best spiral over a few entry radii and both directions.

    Entry radii start just outside the burnt core -- nothing inside it is worth
    cutting -- and the two directions matter because the agent is on one side.

    Walking wider, more conservative spirals as well (requiring each cut to land
    a tick or two ahead of the fire, to leave slack against being boxed in) was
    tried and never once won: the extra cells always cost more than the slack
    bought. Hence the single greedy walk per (radius, direction).
    """
    n = doomed.shape[0]
    angle, radius = _polar(n, origin)
    burnt = int(radius[fire].max()) if fire.any() else 0
    max_steps = 6 * n

    best = None
    for rho in range(burnt + 1, burnt + 1 + starts):
        entry = _ring_entry(origin, rho, agent, n)
        if entry is None:
            break
        for sigma in (1.0, -1.0):
            found = spiral_barrier(doomed, tree, fire, arrival, angle, radius,
                                   entry, sigma, agent, t0, tc, tm, cut_cost,
                                   margin, max_steps)
            if found is not None and (best is None or found[0] < best[0]):
                best = (found[0], rho, sigma, found[1], found[2])
    return best


def plan_barrier(grid, agent, t0, tc, tm, cut_cost=CUT_COST, margin=0,
                 keep=None, hysteresis=1.0, origin=None, max_probe=None):
    """Choose the lowest-loss barrier that can actually be finished in time.

    Four candidates compete on predicted loss: the plan already in progress,
    every rectangle (exhaustive, scored in closed form), a handful of greedy
    outward spirals (the shapes a rectangle cannot express), and doing nothing --
    so a hopeless fire is left to burn rather than paid for. Returns
    ``(spec, rows, cols, predicted_loss)`` with ``spec`` None for "let it burn".

    ``keep`` is the plan in progress, ``(spec, rows, cols)``. It is re-costed
    under the current state rather than re-derived, and given a ``hysteresis``
    discount. Both matter: re-deriving a spiral from a moved agent and a spread
    fire produces a *different* walk each tick, so the agent pays for half of one
    barrier and half of another. Carrying the committed cell list forward is what
    makes the cuts already spent count for something.

    ``max_probe`` caps how many rectangles get the exact schedule check; leave it
    None -- capping it makes the planner give up on hard settings where the best
    barrier is deep in the loss ordering, which is what the pre-filter in
    ``score_rectangles`` is there to avoid instead.
    """
    tree, fire = grid[TREE], grid[FIRE]
    n = tree.shape[0]
    if not fire.any():
        return None, None, None, 0.0

    arrival = fire_arrival(tree, fire)
    doomed = tree & (arrival < INF)          # burns if the agent does nothing
    burn_all = float(doomed.sum())

    def loss_of(rows, cols):
        """Exact loss of a cut set, by flood fill: cuts plus what still burns."""
        cut = np.zeros((n, n), dtype=bool)
        cut[rows, cols] = True
        return _score_cut_set(tree, fire, cut, len(rows), cut_cost)

    best = (burn_all, burn_all, None, None, None)   # key, loss, spec, rows, cols

    # --- the plan already in progress ------------------------------------- #
    if keep is not None and keep[1] is not None:
        spec, rows, cols = keep
        still = doomed[rows, cols]           # drop cells already cut or burnt
        rows, cols = rows[still], cols[still]
        if _schedule(rows, cols, agent, t0, tc, tm, arrival, margin) is not None:
            loss = loss_of(rows, cols)
            best = (loss - hysteresis, loss, spec, rows, cols)

    # --- rectangles ------------------------------------------------------- #
    loss, r0, r1, c0, c1, contains = score_rectangles(
        doomed, fire, arrival, n, cut_cost, tc, tm)
    viable = np.flatnonzero(loss < best[0])
    order = viable[np.argsort(loss[viable], kind="stable")]
    for j in (order if max_probe is None else order[:max_probe]):
        a, b, c, d = int(r0[j]), int(r1[j]), int(c0[j]), int(c1[j])
        clipped = _clip_to_grid(*_rect_perimeter(a, b, c, d), n,
                                closed=(b > a and d > c))
        if clipped is None:
            continue
        rows, cols, cyclic = clipped
        need = doomed[rows, cols]
        rows, cols = rows[need], cols[need]

        chosen, soonest = None, None
        for o_r, o_c in _orderings(rows, cols, cyclic, agent, arrival):
            finish = _schedule(o_r, o_c, agent, t0, tc, tm, arrival, margin)
            if finish is not None and (soonest is None or finish < soonest):
                chosen, soonest = (o_r, o_c), finish
        if chosen is not None:
            spec = ("rect", a, b, c, d, bool(contains[j]))
            best = (float(loss[j]), float(loss[j]), spec, chosen[0], chosen[1])
            break

    # --- spirals ---------------------------------------------------------- #
    # Only worth the walk when the leader still leaves trees burning: if it
    # already saves everything the fire could reach, the most a spiral could add
    # is a slightly cheaper border.
    if best[2] is not None and best[1] <= cut_cost * len(best[3]) + 1e-9:
        return best[2], best[3], best[4], best[1]

    if origin is None:
        rc = np.argwhere(fire)
        origin = (int((rc[:, 0].min() + rc[:, 0].max()) // 2),
                  int((rc[:, 1].min() + rc[:, 1].max()) // 2))
    found = best_spiral(doomed, tree, fire, arrival, origin, agent, t0, tc, tm,
                        cut_cost, margin)
    if found is not None and found[0] < best[0]:
        loss, rho, sigma, rows, cols = found
        # The walk charges a move per cell it passes through; the executor jumps
        # straight between cuts, so re-cost it the way it will actually be run.
        if _schedule(rows, cols, agent, t0, tc, tm, arrival, margin) is not None:
            best = (loss, loss, ("spiral", rho, sigma), rows, cols)

    return best[2], best[3], best[4], best[1]


# --------------------------------------------------------------------------- #
# Policy wrapper
# --------------------------------------------------------------------------- #
class BarrierPlannerPolicy:
    """``plan_barrier`` wrapped as a torchrl-style policy.

    Call it with a tensordict carrying ``grid`` / ``t`` / ``agent_state`` /
    ``params`` and it writes ``action``, batched or not, so the same evaluation
    harness scores it and a PPO actor identically.
    """

    def __init__(self, cut_cost=CUT_COST, hysteresis=1.0, margin=0, max_probe=None):
        self.cut_cost = cut_cost
        self.hysteresis = hysteresis
        self.margin = margin
        self.max_probe = max_probe
        self._state: dict[int, dict] = {}

    def reset(self):
        self._state.clear()

    def _episode_state(self, i, grid, t):
        """Fetch (creating if needed) the state for batch element i.

        A reset is detected by the one thing that cannot happen inside an
        episode: **dirt shrinking**. Fire burns to dirt and a cut turns a tree
        to dirt, and neither ever reverses, so the dirt set only grows while an
        episode runs. A grid with a cell that was dirt and is not any more must
        therefore be a new episode.

        The tests that used to do this job are not general enough. ``t`` alone
        is not: the clock lands on exactly 1.0 mid-episode whenever an action
        ends on a tick boundary (tm = 0.2: 1, .8, .6, .4, .2, 1). "No dirt
        touching the fire" holds for the vanilla grid and for a fire planted
        inside a pocket, but a graph instance ignites a lone junction cell whose
        neighbours are mostly dirt, so that test would never fire there.

        Detecting a reset here is what lets one policy object serve a batch of
        episodes that end at different times, and it is the only moment the
        ignition cell is readable off the grid -- which the spiral search needs
        as the centre to wind around.
        """
        fire = grid[FIRE]
        dirt = grid[DIRT]
        st = self._state.get(i)
        prev = None if st is None else st["prev_dirt"]
        # A different grid size is certainly a different episode.
        fresh = prev is not None and (prev.shape != dirt.shape
                                      or bool((prev & ~dirt).any()))
        if st is None or fresh:
            lit = np.argwhere(fire)
            st = {"spec": None, "rows": None, "cols": None, "t_prev": -1.0,
                  "origin": (int(lit[0][0]), int(lit[0][1])) if len(lit) else None,
                  "prev_dirt": dirt.copy()}
            self._state[i] = st
        else:
            st["prev_dirt"] = dirt.copy()
        return st

    def _act(self, i, grid, t, agent, tc, tm):
        st = self._episode_state(i, grid, t)

        # t counts *down* within a tick and jumps back up when one fires, so a
        # rise is exactly the moment the fire moved and the plan is stale.
        if t > st["t_prev"] + 1e-9:
            spec, rows, cols, _ = plan_barrier(
                grid, agent, float(t), tc, tm, cut_cost=self.cut_cost,
                margin=self.margin, keep=(st["spec"], st["rows"], st["cols"]),
                hysteresis=self.hysteresis, origin=st["origin"],
                max_probe=self.max_probe,
            )
            st.update(spec=spec, rows=rows, cols=cols)
        st["t_prev"] = float(t)

        rows, cols = st["rows"], st["cols"]
        if rows is not None and len(rows):
            # Drop cells that stopped being useful: already cut, or burnt past.
            keep = grid[TREE][rows, cols]
            if not keep.all():
                rows, cols = rows[keep], cols[keep]
                st["rows"], st["cols"] = rows, cols

        if rows is None or len(rows) == 0:
            # Nothing left worth cutting. Moving is free, so drift rather than
            # cut, which costs half a tree even when it hits bare dirt.
            return 0

        tr, tcol = int(rows[0]), int(cols[0])
        if (agent[0], agent[1]) == (tr, tcol):
            return CUT
        return _ACTION_OF_DELTA[(int(np.sign(tr - agent[0])),
                                int(np.sign(tcol - agent[1])))]

    def __call__(self, tensordict):
        grid = tensordict["grid"]
        batch_shape = grid.shape[:-3]
        grids = grid.reshape(-1, *grid.shape[-3:]).cpu().numpy()
        ts = tensordict["t"].reshape(-1).cpu().numpy()
        agents = tensordict["agent_state"].reshape(-1, 2).cpu().numpy()
        tcs = tensordict["params", "tc"].reshape(-1).cpu().numpy()
        tms = tensordict["params", "tm"].reshape(-1).cpu().numpy()

        actions = np.empty(len(grids), dtype=np.int64)
        for i in range(len(grids)):
            actions[i] = self._act(i, grids[i], float(ts[i]),
                                   (int(agents[i, 0]), int(agents[i, 1])),
                                   float(tcs[i]), float(tms[i]))

        tensordict.set("action", torch.as_tensor(
            actions.reshape(batch_shape), device=grid.device, dtype=torch.int64))
        return tensordict
