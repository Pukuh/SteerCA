"""Planted instances: initial grids whose optimal strategy is known.

The vanilla environment starts every episode from an all-tree grid with one
ignition cell, so the only thing a policy can exploit is where the fire is, and
the best it can do is the ring / spiral / rectangle barrier the planner already
searches. A *planted* instance is built the other way round. The fire starts
inside a small closed loop of dirt -- the **pocket** -- near the middle of the
grid, and a run of ``k`` tree cells in that loop, the **gap**, is its only way
out. Cut the gap before the fire gets there and the whole forest outside the
pocket stands; miss it and everything burns. The pocket's own trees are lost
either way. The gap is then hidden: the loop is irregular, dirt is scattered
over the forest, decoy loops and dead-end openings are added, and the whole
thing gets a random dihedral symmetry.

Why a pocket, and not a wall across the grid
--------------------------------------------
The first version split the grid with an edge-to-edge wall and put the gap in
it. The planner beat that oracle on every instance: it cut the gap *and* a short
line from the wall out to the grid edge far from the fire, where the deadline
is late, and so saved the far end of the fire side as well. Any large fire side
bounded by free barriers (edges, the wall itself) can be partly saved that way.
A small pocket with the fire at its centre cannot, provided two things hold:

* **No profitable box closes in time.** Every rectangle around the fire with
  sides no further out than the pocket wall (whose cells are free) is
  scheduled in deadline order, charging ``tc + tm`` per cell and ``tm`` per
  cell of the agent's approach. A box is a threat if what it cuts plus what
  burns inside it is no more than the pocket -- a rim barrier that saves
  exactly what it cuts still beats the oracle, because the oracle also spends
  its ``k`` gap cuts. ``PlantedConfig.agent_fire_distance`` is the smallest
  starting distance at which no threat box is feasible: 0 for ``tc + tm >= 0.4``
  at every pocket size used, and large at low cost, where the agent covers
  many cells per tick. ``resolve`` picks the largest pocket whose distance
  fits on the grid; ``_attempt`` clamps it to what the pocket's position
  allows and records the clamp in ``stats``.
* **The pocket is a plain square.** A dent in the rim is a free cell for a
  rim barrier, a bump beyond it is a tree such a barrier saves; both hand the
  planner a couple of trees per instance. Hiding comes from the noise, the
  decoys and the dihedral element instead.

The residue is a few trees on the odd instance from shapes the box family
does not cover (a staircase across a pocket corner). Measured with
``validate_planted.py`` at ``tc = tm`` in {0.06, 0.12, 0.2, 0.3}: the planner's
mean edge over the oracle is under 0.5 pts at 16x16 and under 0.1 pts at
32x32, never more than 2 pts on any instance. 8x8 is too small to hold the
agent far enough at low cost, and there the planner keeps a one-tree edge at
``tc = tm <= 0.12``.

What is certified, and how
--------------------------
Everything at reset is numpy, well under a millisecond apart from the max-flow:

1. **Separation.** With the gap set to dirt, ``fire_arrival`` reaches no
   protected cell; with it standing, the fire reaches every gap cell.
2. **Min vertex cut == k.** A max-flow on the split-vertex grid graph (protected
   cells uncuttable) proves no set of fewer than ``k`` cuts keeps the fire off
   the protected forest. A dent in the loop or a decoy can open a narrower neck
   -- this catches it.
3. **Schedule.** The planner's exact clock model, ``_schedule``, says the agent
   can cut the gap in order with ``cfg.margin`` ticks to spare. ``fire_arrival``
   and ``_schedule`` are pinned to the CA and to ``cs_env``'s clock by
   ``tests/test_heuristic_agent.py``, so this is the oracle's run in the real env in
   closed form.
4. **Static optimum** (reported on request). The same flow graph with unit sink
   capacities gives the minimum of ``cuts + burnt`` over every static labelling
   of the grid -- what an infinitely fast agent could do. It is always far
   below the oracle's loss, since such an agent would just ring the ignition
   cell, so it is a statistic, not a gate.

The planner cross-check -- ``BarrierPlannerPolicy`` must not save more than the
oracle -- costs about a second per instance and lives in ``validate_planted.py``,
which reports its rate per ``(n, tc, tm)`` setting rather than gating each
reset.

What the certificate does not say: a strategy that cuts the gap *and* saves a
few pocket trees with extra cuts is excluded only by the geometry argument
above. ``oracle_saved`` is a lower bound on the optimum that the planner cannot
beat, and ``oracle_fraction > 1`` in an evaluation is legitimate.

Geometry, in canonical orientation (before the D4 element)
----------------------------------------------------------
The pocket interior is a ``(2R+1)``-square around a centre placed so that the
loop fits on the grid and the gap lands within ``max(window * n, R + 2)`` of
the grid's centre. The wall is the interior's Moore dilation minus the
interior, which is exactly the set an 8-connected fire cannot cross. The fire
sits at the centre, its whole Moore neighbourhood inside the pocket. The gap is
a chain of ``k`` adjacent wall cells whose two mouths are both wider than
``k`` (else the narrow mouth is another cut of size ``k`` that keeps the gap's
own trees), at tree-distance ``d_f >= d_min = floor(k*tc + (k-1)*tm) + margin
+ 1`` from the fire -- the schedule bound for an agent already standing on the
gap. The agent goes anywhere, inside or out, within the travel budget the
schedule leaves and no nearer the fire than ``agent_fire_distance``.
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field, fields

import numpy as np
import torch

from steerca.firebreak.ca import DIRT, FIRE, TREE
from steerca.firebreak.planner import (
    CUT,
    INF,
    _ACTION_OF_DELTA,
    _dilate,
    _rect_perimeter,
    _schedule,
    fire_arrival,
)

# Padding of ``params/gap``: the gap travels in the reset tensordict as a
# ``(K_MAX, 2)`` int64 tensor, unused rows -1. The oracle policy and the eval
# metrics read it; nothing in the PPO loss does.
K_MAX = 8

_NEIGHBOURS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def _parse_range(text, cast):
    """'2-4' -> (2, 4); '3' -> (3, 3)."""
    parts = text.split("-")
    if len(parts) == 1:
        v = cast(parts[0])
        return (v, v)
    if len(parts) != 2:
        raise ValueError(f"bad range {text!r}")
    return (cast(parts[0]), cast(parts[1]))


@dataclass
class PlantedConfig:
    """Knobs of the generator; ``from_string`` parses the STEERCA_PLANTED value.

    ``k_max``, ``pocket`` and ``offset`` default to ``None`` and are resolved per
    grid size and action cost by ``resolve(n, tc, tm)``: ``k_max = clip(n // 8,
    1, 4)``, the pocket radius range by grid size capped by the timing bound,
    and the fire's offset from the pocket centre by the same bound (see the
    module docstring).
    """
    family: str = "pocket"                      # "pocket" or "graph"
    k_min: int = 1
    k_max: int | None = None
    # --- graph family ---
    cuts: tuple[int, int] = (3, 6)              # boundary corridors of the optimum
    search_cuts: int | None = None              # search budget; None -> cuts[1]
    lattice: tuple[int, int] | None = None      # lattice side range; resolved by n
    patch: tuple[int, int] = (0, 0)             # junction half-size; 0 keeps the contraction exact
    jitter: int = 1                             # per-centre offset, in cells
    edge_keep: float = 1.00                     # share of lattice edges attempted
    agent_tries: int = 6                        # agent placements tried per layout
    corridor_min: int = 2                       # shortest corridor, in cells
    cut_sites: int = 1                          # candidate cut cells per corridor searched
    spur_room: int = 4                          # margin left for dead-end spurs
    fire_degree: int | None = None              # corridors at the fire node; None -> cuts[1] + 1
    trap_ratio: float = 0.75                    # cheapest region's payoff, as a share of the optimum
    require_trap: bool = True                   # demand a richer region that cannot be reached
    require_cycle: bool = True                  # demand a cycle, which is what forces a multi-cut answer
    agent_far: bool = True                      # place the agent far from the fire, not near it
    risk_min_frac: float = 0.25                 # optimum saves this share of at-risk trees
    burn_budget: float = 0.80                   # share of the episode cap the fire must finish in
    pocket: tuple[int, int] | None = None       # pocket Chebyshev radius range
    offset: int = 0                             # max fire offset from the pocket centre
    agent_fire_dist: int | None = None          # min agent distance from the fire; None -> derived
    bumps: int = 0                              # one-cell bumps on the loop, at most 4x this; 0 keeps it square
    window: float = 0.25                        # gap within max(window*n, R+2) of the grid centre
    margin: int = 1                             # ticks of slack the schedule must keep
    agent_min_dist: int = 1                     # Chebyshev distance from gap[0]
    noise: tuple[float, float] = (0.03, 0.10)   # dirt density outside the pocket
    noise_inside: float = 0.0                   # inside the pocket, as a fraction of that
    decoy_walls: tuple[int, int] = (1, 3)       # dirt segments outside
    decoy_loops: tuple[int, int] = (0, 2)       # closed loops elsewhere, with an opening
    decoy_gaps: tuple[int, int] = (0, 2)        # openings in the pocket wall into a dead-end pocket
    pocket_radius: tuple[int, int] = (1, 1)     # size of those dead-end pockets: 1 encloses nothing
    protected_min_frac: float = 0.35            # protected trees / n^2
    free_max_frac: float = 0.10                 # trees nobody needs to save / n^2
    require_static_optimal: bool = False
    max_attempts: int = 500
    overrides: dict = field(default_factory=dict)  # what from_string changed, for tag()

    # STEERCA_PLANTED="k=2-4,R=2-3,offset=1,noise=0.05-0.1,margin=2,decoys=0"
    _KEYS = {
        "k": ("k", lambda s: _parse_range(s, int)),
        "family": ("family", str),
        "cuts": ("cuts", lambda s: _parse_range(s, int)),
        "scuts": ("search_cuts", int),
        "L": ("lattice", lambda s: _parse_range(s, int)),
        "P": ("patch", lambda s: _parse_range(s, int)),
        "jitter": ("jitter", int),
        "keep": ("edge_keep", float),
        "cmin": ("corridor_min", int),
        "sites": ("cut_sites", int),
        "room": ("spur_room", int),
        "atries": ("agent_tries", int),
        "fdeg": ("fire_degree", int),
        "trap": ("trap_ratio", float),
        "rtrap": ("require_trap", lambda s: s not in ("0", "false", "")),
        "rcyc": ("require_cycle", lambda s: s not in ("0", "false", "")),
        "afar": ("agent_far", lambda s: s not in ("0", "false", "")),
        "risk": ("risk_min_frac", float),
        "burn": ("burn_budget", float),
        "R": ("pocket", lambda s: _parse_range(s, int)),
        "offset": ("offset", int),
        "afire": ("agent_fire_dist", int),
        "bumps": ("bumps", int),
        "noise": ("noise", lambda s: _parse_range(s, float)),
        "inside": ("noise_inside", float),
        "margin": ("margin", int),
        "m": ("margin", int),
        "window": ("window", float),
        "dw": ("decoy_walls", lambda s: _parse_range(s, int)),
        "dl": ("decoy_loops", lambda s: _parse_range(s, int)),
        "dg": ("decoy_gaps", lambda s: _parse_range(s, int)),
        "pocket": ("pocket_radius", lambda s: _parse_range(s, int)),
        "pmin": ("protected_min_frac", float),
        "free": ("free_max_frac", float),
        "amin": ("agent_min_dist", int),
        "static": ("require_static_optimal", lambda s: s not in ("0", "false", "")),
        "decoys": ("decoys", lambda s: s not in ("0", "false", "")),
        "attempts": ("max_attempts", int),
    }

    @classmethod
    def from_string(cls, text: str | None) -> "PlantedConfig":
        """``"1"``/``"true"``/``""`` are the defaults; otherwise ``key=value,...``."""
        cfg = cls()
        text = (text or "").strip()
        if text.lower() in ("", "1", "true", "yes", "planted"):
            return cfg
        for token in text.split(","):
            token = token.strip()
            if not token:
                continue
            if "=" not in token:
                raise ValueError(f"STEERCA_PLANTED token {token!r} is not key=value")
            key, value = token.split("=", 1)
            key, value = key.strip(), value.strip()
            if key not in cls._KEYS:
                raise ValueError(f"unknown STEERCA_PLANTED key {key!r}; "
                                 f"known: {sorted(cls._KEYS)}")
            attr, cast = cls._KEYS[key]
            parsed = cast(value)
            if attr == "k":
                cfg.k_min, cfg.k_max = parsed
            elif attr == "decoys":
                if not parsed:
                    cfg.decoy_walls = cfg.decoy_loops = cfg.decoy_gaps = (0, 0)
            else:
                setattr(cfg, attr, parsed)
            cfg.overrides[key] = value
        cfg.validate()
        return cfg

    def validate(self):
        if self.family not in ("pocket", "graph"):
            raise ValueError(f"unknown planted family {self.family!r}")
        if self.family == "graph":
            if not (1 <= self.cuts[0] <= self.cuts[1] <= K_MAX):
                raise ValueError(f"cuts must lie in 1..{K_MAX}")
            if max(self.patch) > 0:
                # A node of more than one cell can be SLICED: a line of cells
                # destroyed leaves everything beyond it standing, which is a cut
                # set the contraction cannot express, and it beat the certified
                # plan by 64% when nodes were 3x3. One-cell junctions have
                # nothing to slice, which is what makes the optimum provable.
                raise ValueError("node half-size > 0 re-admits patch slices, "
                                 "which the certificate does not cover")
            return
        if self.k_max is not None and self.k_max > K_MAX:
            raise ValueError(f"k_max={self.k_max} exceeds K_MAX={K_MAX}")
        if self.k_min < 1:
            raise ValueError("k_min must be >= 1")
        if self.pocket is not None and self.pocket[0] < 1:
            raise ValueError("pocket radius must be >= 1")

    def tag(self) -> str:
        """``planted`` for the defaults, else ``planted-k2-4-m2`` -- goes in the
        experiment name so runs under different instance distributions cannot
        share a checkpoint."""
        if not self.overrides:
            return "planted"
        rest = "-".join(f"{k}{v}" for k, v in sorted(self.overrides.items())
                        if k != "family")
        head = "planted" if self.family == "pocket" else f"planted-{self.family}"
        return f"{head}-{rest}" if rest else head

    @staticmethod
    def pocket_limits(s: float):
        """``(R_max, e_max_of_R)`` from the chord argument in the module docstring.

        No profitable chord is feasible while ``(2R+1) s > R - 2``, i.e.
        ``R < (2 + s) / (1 - 2s)`` (any R once ``s >= 1/2``); with the fire ``e``
        off-centre the pocket extends ``R + e`` past it, so ``e < (2R+1) s - R + 2``.
        The default keeps the fire centred (``offset = 0``) regardless: off
        centre, the planner walls off the far part of the pocket with a
        rectangle that borrows the pocket's own dirt, which the chord bound
        does not model.
        """
        eps = 1e-9
        r_max = 10 ** 6 if s >= 0.5 else int(math.ceil((2 + s) / (1 - 2 * s) - eps)) - 1

        def e_max(R):
            return max(0, int(math.ceil((2 * R + 1) * s - R + 2 - eps)) - 1)
        return r_max, e_max

    @staticmethod
    def agent_fire_distance(R: int, tc: float, tm: float, d_max: int = 64) -> int:
        """How far from the fire the agent must start so that no box around
        the fire, inside the pocket, can be closed in time.

        Below ``tc + tm = 1/8`` the fire can be contained at birth: a 4x4 box
        around it has 5 cells the fire reaches at tick 1 and 7 at tick 2, and
        at ``tc = tm = 0.06`` those 5 take 0.6 ticks, so an agent within ~6
        cells of them closes the box and saves the pocket. Every rectangle with
        the fire strictly inside and sides no further than the pocket wall
        (whose cells are free) is scheduled here in deadline order, charging
        ``tc + tm`` per cell and ``tm`` per cell of the agent's approach; the
        smallest starting distance at which none is feasible is returned. It is
        0 for ``tc + tm >= 1/8`` at every pocket size used.
        """
        s = tc + tm
        pocket = (2 * R + 1) ** 2 - 1                    # trees in the pocket
        boxes = []
        span = range(1, R + 2)
        for r0 in span:
            for r1 in span:
                for c0 in span:
                    for c1 in span:
                        cells = []
                        for r in range(-r0, r1 + 1):
                            for c in range(-c0, c1 + 1):
                                if r in (-r0, r1) or c in (-c0, c1):
                                    d = max(abs(r), abs(c))
                                    if d <= R:          # d == R+1 is the wall: free
                                        cells.append(d)
                        inside = (r0 + r1 - 1) * (c0 + c1 - 1) - 1   # trees inside
                        # The box loses what it cuts plus what burns inside;
                        # the oracle loses the whole pocket plus its k >= 1
                        # cuts. So the box is a threat unless it loses strictly
                        # more than the pocket -- a rim barrier that saves
                        # exactly what it cuts still beats the oracle by k.
                        if cells and len(cells) + inside <= pocket:
                            boxes.append(sorted(cells))
        for dist in range(0, d_max + 1):
            feasible = False
            for deadlines in boxes:
                # the approach ends on the first cell, so it costs travel + tc,
                # and every further cell a move plus a cut
                tau = max(0, dist - deadlines[0]) * tm + tc
                ok = tau < deadlines[0]
                for d in deadlines[1:]:
                    tau += s
                    if tau >= d:
                        ok = False
                        break
                if ok:
                    feasible = True
                    break
            if not feasible:
                return dist
        return d_max

    def cap_factor(self) -> float:
        """Episode-length headroom this family needs, as a multiple of n ticks."""
        return 2.5

    def resolve(self, n: int, tc: float, tm: float) -> "PlantedConfig":
        """A copy with the ``None`` defaults filled in for this grid and cost."""
        out = PlantedConfig(**{f.name: getattr(self, f.name) for f in fields(self)})
        if out.family == "graph":
            if out.fire_degree is None:
                # Enough corridors that sealing the ignition patch is a big,
                # early, tightly-scheduled job rather than the obvious answer.
                # Tying this to the cut budget instead makes it unreachable on a
                # lattice, where a node has at most four neighbours.
                out.fire_degree = 4
            if out.lattice is None:
                # The largest lattice whose corridors still clear corridor_min:
                # patches occupy 2p+1 cells and the margin leaves room for spurs.
                p = max(out.patch)
                m = p + 1 + out.spur_room
                need = 2 * p + out.corridor_min + 2
                # Capped at 4 a side: the region search is vectorised over all
                # 2**V node subsets, so V must stay at or below 16.
                hi = 2
                while hi < 4 and (n - 1 - 2 * m) // hi >= need:
                    hi += 1
                if (n - 1 - 2 * m) // max(hi - 1, 1) < need:
                    raise ValueError(
                        f"family=graph does not fit on {n}x{n}: a {2*p+1}-cell patch "
                        f"plus a {out.corridor_min}-cell corridor and {out.spur_room} "
                        f"cells of spur room need a larger grid")
                out.lattice = (max(2, hi - 1), hi)
            return out
        if out.k_max is None:
            out.k_max = int(np.clip(n // 8, 1, 4))
        out.k_max = max(out.k_min, min(out.k_max, K_MAX))
        r_max, _ = self.pocket_limits(tc + tm)
        if out.pocket is None:
            lo, hi = (1, 1) if n < 12 else (2, 3) if n < 24 else (3, 5)
            hi = min(hi, r_max)
            # The largest pocket the grid can certify: the agent must start
            # agent_fire_distance from the fire, and with the gap held inside
            # the central window the pocket centre is at most about
            # window*n + R + 1 off the grid's centre, so the farthest cell is
            # roughly (n-1)/2 + that. Smaller pockets need less distance.
            while hi > 1 and _agent_fire_distance_cached(hi, tc, tm) > self.max_agent_reach(n, hi):
                hi -= 1
            out.pocket = (max(1, min(lo, hi)), max(1, hi))
        return out

    def max_agent_reach(self, n: int, R: int) -> int:
        """Farthest an agent can start from a fire whose gap is in the window."""
        return int((n - 1) / 2 + max(self.window * n, R + 2) + R + 1)


# --------------------------------------------------------------------------- #
# Instances
# --------------------------------------------------------------------------- #
@dataclass
class PlantedInstance:
    grid: np.ndarray            # (3, n, n) bool one-hot, final orientation
    agent: tuple[int, int]
    fire: tuple[int, int]
    gap: np.ndarray             # (k, 2) int64, in the order the oracle cuts them
    protected: np.ndarray       # (n, n) bool: burns if nothing is done, saved by the gap cut
    fire_side: np.ndarray       # (n, n) bool: burns whatever the agent does
    oracle_saved: float         # fraction of the AREA still tree after the oracle plays
    noop_saved: float           # same if nobody cuts
    min_cut: int
    static_loss_gap: int | None  # oracle loss - static min loss; 0 => optimal at any speed
    stats: dict
    family: str = "pocket"

    @property
    def k(self) -> int:
        return int(len(self.gap))


def instance_to_tensors(inst: PlantedInstance, device="cpu") -> dict:
    """The reset keys of one instance, padded the way the env spec expects."""
    gap = np.full((K_MAX, 2), -1, dtype=np.int64)
    gap[: len(inst.gap)] = inst.gap
    return {
        "grid": torch.from_numpy(inst.grid).to(device),
        "agent_state": torch.tensor(inst.agent, dtype=torch.int64, device=device),
        "gap": torch.from_numpy(gap).to(device),
        "oracle_saved": torch.tensor(inst.oracle_saved, dtype=torch.float32, device=device),
        "noop_saved": torch.tensor(inst.noop_saved, dtype=torch.float32, device=device),
    }


# --------------------------------------------------------------------------- #
# Max-flow on the grid: min vertex cut and static minimum loss
# --------------------------------------------------------------------------- #
class _Dinic:
    """Dinic's algorithm on an explicit residual graph; iterative, no recursion.

    Pure Python on purpose: scipy and networkx are not installed, and the graphs
    here are tiny (a few thousand nodes) with flow values of a handful, so the
    whole thing is milliseconds.
    """

    def __init__(self, n_nodes):
        self.n = n_nodes
        self.adj = [[] for _ in range(n_nodes)]
        self.to = []
        self.cap = []

    def add(self, u, v, c):
        self.adj[u].append(len(self.to))
        self.to.append(v)
        self.cap.append(c)
        self.adj[v].append(len(self.to))
        self.to.append(u)
        self.cap.append(0)

    def _levels(self, s):
        level = [-1] * self.n
        level[s] = 0
        q = deque([s])
        to, cap, adj = self.to, self.cap, self.adj
        while q:
            u = q.popleft()
            lu = level[u] + 1
            for e in adj[u]:
                v = to[e]
                if cap[e] > 0 and level[v] < 0:
                    level[v] = lu
                    q.append(v)
        return level

    def _augment(self, s, t, level, it):
        """One augmenting path in the level graph, or 0 if there is none."""
        to, cap, adj = self.to, self.cap, self.adj
        stack, path = [s], []
        while stack:
            u = stack[-1]
            if u == t:
                f = min(cap[e] for e in path)
                for e in path:
                    cap[e] -= f
                    cap[e ^ 1] += f
                return f
            edges = adj[u]
            lu = level[u] + 1
            advanced = False
            while it[u] < len(edges):
                e = edges[it[u]]
                v = to[e]
                if cap[e] > 0 and level[v] == lu:
                    stack.append(v)
                    path.append(e)
                    advanced = True
                    break
                it[u] += 1
            if not advanced:
                level[u] = -1            # dead end for this phase
                stack.pop()
                if path:
                    path.pop()
                if stack:
                    it[stack[-1]] += 1
        return 0

    def max_flow(self, s, t, stop_above=None):
        flow = 0
        while True:
            level = self._levels(s)
            if level[t] < 0:
                return flow
            it = [0] * self.n
            while True:
                pushed = self._augment(s, t, level, it)
                if pushed == 0:
                    break
                flow += pushed
                if stop_above is not None and flow > stop_above:
                    return flow

    def reachable(self, s):
        return [lv >= 0 for lv in self._levels(s)]


def _build_flow(tree, fire, sink, unit_sink):
    """The split-vertex graph over the cells the fire can reach.

    Node ``2i`` is cell i's entry, ``2i+1`` its exit. ``in -> out`` has capacity
    1 (cutting the cell) or INF for a sink cell (it must be saved, not cut).
    Moore-adjacent trees are joined ``out -> in`` with INF. Trees touching the
    fire hang off the source. Sink cells hang off the sink with INF; with
    ``unit_sink`` every cell does with capacity 1, which turns the cut value
    into ``cuts + burnt``.
    """
    h, w = tree.shape
    arrival = fire_arrival(tree, fire)
    doomed = tree & (arrival < INF)
    cells = np.argwhere(doomed)
    m = len(cells)
    ids = np.full((h, w), -1, dtype=np.int64)
    ids[cells[:, 0], cells[:, 1]] = np.arange(m)
    S, T = 2 * m, 2 * m + 1
    g = _Dinic(2 * m + 2)

    sink_mask = np.zeros((h, w), dtype=bool) if sink is None else (sink & doomed)
    is_sink = sink_mask[cells[:, 0], cells[:, 1]]
    for i in range(m):
        g.add(2 * i, 2 * i + 1, INF if is_sink[i] else 1)
        if unit_sink:
            g.add(2 * i + 1, T, 1)
        elif is_sink[i]:
            g.add(2 * i + 1, T, INF)

    src = _dilate(fire) & doomed
    for i in ids[src]:
        g.add(S, 2 * int(i), INF)

    padded = np.full((h + 2, w + 2), -1, dtype=np.int64)
    padded[1:-1, 1:-1] = ids
    for dr, dc in _NEIGHBOURS:
        nb = padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        ok = (ids >= 0) & (nb >= 0)
        for u, v in zip(ids[ok].tolist(), nb[ok].tolist()):
            g.add(2 * u + 1, 2 * v, INF)
    return g, S, T, cells, doomed


def min_vertex_cut(tree, fire, sink, stop_above=None):
    """Fewest cells to cut so the fire never reaches any ``sink`` cell.

    Sink cells themselves are uncuttable: a cut set that fells a protected tree
    has not saved it. Returns ``(value, cut_mask)``; with ``stop_above`` the
    search stops once the flow exceeds it and the mask is then meaningless.
    """
    g, S, T, cells, _ = _build_flow(tree, fire, sink, unit_sink=False)
    value = g.max_flow(S, T, stop_above=stop_above)
    cut = np.zeros(tree.shape, dtype=bool)
    if stop_above is None or value <= stop_above:
        reach = g.reachable(S)
        for i, (r, c) in enumerate(cells):
            if reach[2 * i] and not reach[2 * i + 1]:
                cut[r, c] = True
    return int(value), cut


def static_min_loss(tree, fire):
    """min over cut sets C of ``|C| + |trees the fire still reaches|``.

    Every outcome an agent of any speed can produce is one such labelling
    (cut / burnt / saved), so this bounds every policy from below.
    """
    g, S, T, _, _ = _build_flow(tree, fire, None, unit_sink=True)
    return int(g.max_flow(S, T))


# --------------------------------------------------------------------------- #
# Geometry helpers (canonical orientation)
# --------------------------------------------------------------------------- #
def _erode(mask):
    return ~_dilate(~mask)


def _component(mask, seed_rc):
    """8-connected component of ``mask`` containing ``seed_rc``."""
    comp = np.zeros_like(mask)
    comp[seed_rc] = mask[seed_rc]
    while True:
        grown = _dilate(comp) & mask
        if grown.sum() == comp.sum():
            return comp
        comp = grown


def _pocket(n, rng, cfg, R, keep_radius):
    """A ``(2R+1)``-square pocket; returns ``(interior, wall, centre)``.

    The wall is the interior's Moore dilation minus the interior, so it is
    exactly what an 8-connected fire cannot cross. It is a plain square by
    default, and that is deliberate: a dent in the rim is a free cell for a
    barrier along the rim, and a bump beyond it is a tree such a barrier
    saves -- the planner found a couple of trees on every instance that way.
    On a plain square a rim barrier saves exactly what it cuts. ``cfg.bumps``
    adds one-cell bumps for experiments that want a rougher loop.
    """
    margin = R + 2
    if n - margin <= margin:
        return None
    cr, cc = (int(v) for v in rng.integers(margin, n - margin, size=2))
    interior = np.zeros((n, n), dtype=bool)
    interior[cr - R:cr + R + 1, cc - R:cc + R + 1] = True
    if cfg.bumps > 0:
        outer = np.argwhere(_dilate(interior) & ~interior)
        n_bumps = int(rng.integers(0, 4 * cfg.bumps + 1))
        for r, c in outer[rng.permutation(len(outer))[:n_bumps]]:
            interior[r, c] = True
    wall = _dilate(interior) & ~interior
    if wall[0, :].any() or wall[-1, :].any() or wall[:, 0].any() or wall[:, -1].any():
        return None
    return interior, wall, (cr, cc)


def _gap_chain(rng, wall, interior, k, window_mask, tries=12):
    """``k`` adjacent wall cells, each with a tree on both sides, in the window."""
    outside = ~wall & ~interior
    proper = wall & _dilate(interior) & _dilate(outside) & window_mask
    starts = np.argwhere(proper)
    if len(starts) == 0:
        return None
    for _ in range(tries):
        chain = [tuple(int(v) for v in starts[rng.integers(len(starts))])]
        used = {chain[0]}
        while len(chain) < k:
            r, c = chain[-1]
            nxt = [(r + dr, c + dc) for dr, dc in _NEIGHBOURS
                   if 0 <= r + dr < wall.shape[0] and 0 <= c + dc < wall.shape[1]
                   and proper[r + dr, c + dc] and (r + dr, c + dc) not in used]
            if not nxt:
                break
            pick = nxt[rng.integers(len(nxt))]
            chain.append(pick)
            used.add(pick)
        if len(chain) == k:
            return np.array(chain, dtype=np.int64)
    return None


def _decoy_wall(n, rng, forbidden, dirt):
    """One wobbly dirt segment that separates nothing on its own.

    Length is at most n/3, so it cannot span the grid; it stops at any cell in
    ``forbidden`` (the pocket zone, the gap's mouths).
    """
    length = int(rng.integers(max(2, n // 6), max(3, n // 3) + 1))
    r, c = int(rng.integers(0, n)), int(rng.integers(0, n))
    vertical = bool(rng.integers(0, 2))
    placed = 0
    for _ in range(length):
        if not (0 <= r < n and 0 <= c < n) or forbidden[r, c]:
            break
        dirt[r, c] = True
        placed += 1
        lateral = int(rng.integers(-1, 2))
        if vertical:
            c2 = c + lateral
            if lateral and 0 <= c2 < n and not forbidden[r, c2]:
                c = c2
                dirt[r, c] = True
            r += 1
        else:
            r2 = r + lateral
            if lateral and 0 <= r2 < n and not forbidden[r2, c]:
                r = r2
                dirt[r, c] = True
            c += 1
    return placed


def _decoy_loop(n, rng, cfg, R_max, forbidden, dirt, tree):
    """A closed loop elsewhere, with an opening so nothing inside is free."""
    R = int(rng.integers(1, max(1, R_max) + 1))
    margin = R + 1
    if n - margin <= margin:
        return False
    cr, cc = (int(v) for v in rng.integers(margin, n - margin, size=2))
    interior = np.zeros((n, n), dtype=bool)
    interior[cr - R:cr + R + 1, cc - R:cc + R + 1] = True
    zone = _dilate(_dilate(interior))
    if (zone & forbidden).any():
        return False
    wall = _dilate(interior) & ~interior
    cells = np.argwhere(wall)
    opening = cells[rng.permutation(len(cells))[:int(rng.integers(1, 3))]]
    wall[opening[:, 0], opening[:, 1]] = False
    dirt |= wall
    tree &= ~wall
    return True


def _dead_end_gap(n, rng, cfg, wall, interior, gap_mask, forbidden, dirt, tree):
    """Open a decoy gap in the pocket wall and wall off its outside mouth.

    The decoy cell turns to TREE and a rectangle perimeter grown
    ``pocket_radius`` around it, restricted to the outside, turns to dirt: the
    fire that comes through burns the little pocket inside and nothing else.
    """
    radius = int(rng.integers(cfg.pocket_radius[0], cfg.pocket_radius[1] + 1))
    outside = ~wall & ~interior
    rr, cc = np.indices((n, n))
    gap_cells = np.argwhere(gap_mask)
    far = np.ones((n, n), dtype=bool)
    for r, c in gap_cells:
        far &= np.maximum(np.abs(rr - r), np.abs(cc - c)) >= radius + 3
    proper = wall & _dilate(interior) & _dilate(outside) & far
    cand = np.argwhere(proper)
    if len(cand) == 0:
        return False
    r, c = (int(v) for v in cand[rng.integers(len(cand))])
    pr, pc = _rect_perimeter(r - radius, r + radius, c - radius, c + radius)
    on = (pr >= 0) & (pr < n) & (pc >= 0) & (pc < n)
    pr, pc = pr[on], pc[on]
    border = outside[pr, pc] & ~forbidden[pr, pc]
    dirt[pr[border], pc[border]] = True
    tree[pr[border], pc[border]] = False
    tree[r, c] = True
    dirt[r, c] = False
    return True


# --------------------------------------------------------------------------- #
# Dihedral symmetry
# --------------------------------------------------------------------------- #
def d4_apply(g: int, n: int, grids=(), points=(), masks=()):
    """Apply dihedral element ``g`` (0..7) to grids ``(..., n, n)``, point
    arrays ``(..., 2)`` of (row, col), and masks ``(n, n)``.

    Bits of ``g``: transpose, flip rows, flip cols. Points are mapped through a
    transformed index grid, so their formula can never drift from the arrays'.
    """
    def arr(a):
        if g & 1:
            a = np.swapaxes(a, -2, -1)
        if g & 2:
            a = np.flip(a, axis=-2)
        if g & 4:
            a = np.flip(a, axis=-1)
        return np.ascontiguousarray(a)

    idx = arr(np.arange(n * n).reshape(n, n))
    where = np.empty((n * n, 2), dtype=np.int64)
    rr, cc = np.indices((n, n))
    where[idx.ravel()] = np.stack([rr.ravel(), cc.ravel()], axis=1)

    def pts(p):
        p = np.asarray(p, dtype=np.int64)
        flat = p.reshape(-1, 2)
        out = where[flat[:, 0] * n + flat[:, 1]]
        return out.reshape(p.shape)

    return ([arr(a) for a in grids], [pts(p) for p in points], [arr(m) for m in masks])


# --------------------------------------------------------------------------- #
# Certificate
# --------------------------------------------------------------------------- #
def oracle_order(gap_rc, arrival, tc, tm, margin):
    """The cutting order with the most slack, and that slack in ticks.

    Candidates: the gap in walk order, reversed, and by fire arrival. Slack is
    ``min_i(arrival_i - margin - w_i)`` with ``w_i`` the cumulative work from the
    first cell on (the agent's travel to it is added by the caller), so the
    agent may start up to ``slack / tm`` cells away.
    """
    gap_rc = np.asarray(gap_rc, dtype=np.int64)
    by_arrival = gap_rc[np.argsort(arrival[gap_rc[:, 0], gap_rc[:, 1]], kind="stable")]
    best = None
    for order in (gap_rc, gap_rc[::-1], by_arrival):
        travel = np.maximum(np.abs(np.diff(order[:, 0])), np.abs(np.diff(order[:, 1])))
        work = np.cumsum(np.concatenate(([0.0], travel * tm)) + tc)
        slack = float(np.min(arrival[order[:, 0], order[:, 1]] - margin - work))
        if best is None or slack > best[1]:
            best = (order.copy(), slack)
    return best


def _certify(grid, gap_rc, tc, tm, cfg, rng, agent=None, want_static=False,
             agent_fire_dist=0):
    """Check one grid and, unless given, place the agent.

    Returns ``(fields, None)`` or ``(None, reason)``. ``fields`` carries
    everything the instance needs: arrival, sides, order, agent, min cut.
    """
    n = grid.shape[-1]
    tree, fire = grid[TREE], grid[FIRE]
    gap_rc = np.asarray(gap_rc, dtype=np.int64)
    k = len(gap_rc)
    gap_mask = np.zeros((n, n), dtype=bool)
    gap_mask[gap_rc[:, 0], gap_rc[:, 1]] = True
    if not tree[gap_mask].all():
        return None, "gap_not_tree"

    if (_dilate(fire) & grid[DIRT]).any():
        return None, "fire_touches_dirt"
    arrival0 = fire_arrival(tree, fire)
    if arrival0[gap_mask].max() >= INF:
        return None, "gap_unreachable"
    doomed0 = tree & (arrival0 < INF)
    tree_cut = tree & ~gap_mask
    doomed_cut = tree_cut & (fire_arrival(tree_cut, fire) < INF)
    protected = doomed0 & ~doomed_cut & ~gap_mask
    area = n * n
    if protected.sum() < cfg.protected_min_frac * area:
        return None, "protected_small"
    if (tree & ~doomed0).sum() > cfg.free_max_frac * area:
        return None, "free_trees"

    # Both mouths of the gap must be wider than the gap, or the cells of the
    # narrow mouth are another cut of size k that keeps the gap's own trees
    # standing -- a strictly better plan by up to k trees.
    mouth = _dilate(gap_mask) & tree & ~gap_mask
    if (mouth & doomed_cut).sum() <= k or (mouth & protected).sum() <= k:
        return None, "narrow_mouth"

    order, slack = oracle_order(gap_rc, arrival0, tc, tm, cfg.margin)
    if slack <= 0:
        return None, "no_slack"

    if agent is None:
        d_max = int(math.ceil(slack / tm)) - 1
        if d_max < cfg.agent_min_dist:
            return None, "no_agent_site"
        rr, cc = np.indices((n, n))
        dist = np.maximum(np.abs(rr - order[0, 0]), np.abs(cc - order[0, 1]))
        fr, fc = np.argwhere(fire)[0]
        from_fire = np.maximum(np.abs(rr - fr), np.abs(cc - fc))
        ok = ((dist >= cfg.agent_min_dist) & (dist <= d_max) & ~fire & ~gap_mask
              & (from_fire >= agent_fire_dist))
        cand = np.argwhere(ok)
        if len(cand) == 0:
            return None, "no_agent_site"
        rng.shuffle(cand)
        agent = None
        for r, c in cand[:64]:
            if _schedule(order[:, 0], order[:, 1], (int(r), int(c)), 1.0, tc, tm,
                         arrival0, cfg.margin) is not None:
                agent = (int(r), int(c))
                break
        if agent is None:
            return None, "no_agent_site"
    elif _schedule(order[:, 0], order[:, 1], tuple(agent), 1.0, tc, tm,
                   arrival0, cfg.margin) is None:
        return None, "schedule_infeasible"

    mc, _ = min_vertex_cut(tree, fire, protected, stop_above=k)
    if mc != k:
        return None, f"min_cut_{mc}"

    static_gap = None
    if want_static or cfg.require_static_optimal:
        oracle_loss = k + int(doomed_cut.sum())
        static_gap = oracle_loss - static_min_loss(tree, fire)
        if cfg.require_static_optimal and static_gap != 0:
            return None, "not_static_optimal"

    return {
        "arrival0": arrival0, "doomed0": doomed0, "doomed_cut": doomed_cut,
        "protected": protected, "order": order, "slack": slack, "agent": agent,
        "min_cut": mc, "static_loss_gap": static_gap,
        "oracle_saved": float(tree_cut.sum() - doomed_cut.sum()) / area,
        "noop_saved": float((tree & ~doomed0).sum()) / area,
        "d_f": int(arrival0[gap_mask].min()),
    }, None


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
def generate_instance(n, tc, tm, rng, cfg=None, want_static=False) -> PlantedInstance:
    """One certified planted instance for grid size ``n`` at costs ``tc``/``tm``.

    Resamples on every rejection, up to ``cfg.max_attempts``, and records the
    reasons in ``stats["rejections"]``. Raises ``RuntimeError`` with that
    histogram if the configuration cannot be satisfied.
    """
    cfg = (cfg or PlantedConfig()).resolve(n, tc, tm)
    build = _attempt_graph if cfg.family == "graph" else _attempt
    rejections = Counter()
    for attempt in range(1, cfg.max_attempts + 1):
        inst, reason = build(n, tc, tm, rng, cfg, want_static)
        if inst is not None:
            inst.stats["attempts"] = attempt
            inst.stats["rejections"] = dict(rejections)
            return inst
        rejections[reason] += 1
    raise RuntimeError(f"no planted instance for n={n} tc={tc} tm={tm} after "
                       f"{cfg.max_attempts} attempts: {dict(rejections)}")


def generate_batch(b, n, tc, tm, rng, cfg=None):
    return [generate_instance(n, tc, tm, rng, cfg) for _ in range(b)]


_AFIRE_CACHE = {}


def _agent_fire_distance_cached(R, tc, tm):
    key = (R, tc, tm)
    if key not in _AFIRE_CACHE:
        _AFIRE_CACHE[key] = PlantedConfig.agent_fire_distance(R, tc, tm)
    return _AFIRE_CACHE[key]


def _attempt(n, tc, tm, rng, cfg, want_static):
    R = int(rng.integers(cfg.pocket[0], cfg.pocket[1] + 1))
    e_max = min(cfg.offset, R - 1)
    afire = (cfg.agent_fire_dist if cfg.agent_fire_dist is not None
             else _agent_fire_distance_cached(R, tc, tm))
    afire_wanted = afire

    # a. pocket
    made = _pocket(n, rng, cfg, R, keep_radius=e_max + 1)
    if made is None:
        return None, "pocket_misfit"
    interior, wall, (cr, cc) = made
    # A grid too small for the distance gets what it can hold: the farthest
    # cell from this pocket's centre. Recorded, since the certificate is then
    # only as good as the planner check says (validate_planted.py).
    reach = max(cr, n - 1 - cr, cc, n - 1 - cc)
    afire_clamped = afire > reach
    afire = min(afire, reach)

    # b. gap: k adjacent wall cells near the grid centre
    k = int(rng.integers(cfg.k_min, cfg.k_max + 1))
    rr, cc_ = np.indices((n, n))
    centre = (n - 1) / 2.0
    window = np.maximum(np.abs(rr - centre), np.abs(cc_ - centre)) <= max(cfg.window * n, R + 2)
    gap_rc = _gap_chain(rng, wall, interior, k, window)
    if gap_rc is None:
        return None, "no_gap_window"
    gap_mask = np.zeros((n, n), dtype=bool)
    gap_mask[gap_rc[:, 0], gap_rc[:, 1]] = True

    tree = ~wall | gap_mask
    dirt = ~tree

    # c. fire: near the pocket centre, Moore neighbourhood all tree, at least
    # d_min ticks from the gap
    dist = fire_arrival(tree, gap_mask)
    work = k * tc + (k - 1) * tm
    d_min = int(math.floor(work)) + cfg.margin + 1
    near = np.maximum(np.abs(rr - cr), np.abs(cc_ - cc)) <= e_max
    sites = interior & near & _erode(interior) & (dist >= d_min) & (dist < INF)
    cand = np.argwhere(sites)
    if len(cand) == 0:
        return None, "no_fire_site"
    fr, fc = (int(v) for v in cand[rng.integers(len(cand))])
    fire = np.zeros((n, n), dtype=bool)
    fire[fr, fc] = True
    tree[fr, fc] = False

    # d. hiding. The pocket zone (interior, wall, and the ring outside it) is
    # kept clear of decoys; the gap's mouths keep their trees.
    zone = _dilate(_dilate(interior))
    mouths = _dilate(gap_mask)
    forbidden = zone | mouths
    p_noise = float(rng.uniform(cfg.noise[0], cfg.noise[1]))
    noise = (rng.random((n, n)) < p_noise) & tree & ~zone
    if cfg.noise_inside > 0:
        inner = ((rng.random((n, n)) < p_noise * cfg.noise_inside) & interior
                 & ~mouths & ~(np.maximum(np.abs(rr - fr), np.abs(cc_ - fc)) <= 1))
        noise |= inner
    dirt |= noise
    tree &= ~noise

    placed_walls = sum(_decoy_wall(n, rng, forbidden, dirt) > 0
                       for _ in range(int(rng.integers(cfg.decoy_walls[0], cfg.decoy_walls[1] + 1))))
    placed_loops = sum(_decoy_loop(n, rng, cfg, R, forbidden, dirt, tree)
                       for _ in range(int(rng.integers(cfg.decoy_loops[0], cfg.decoy_loops[1] + 1))))
    placed_gaps = sum(_dead_end_gap(n, rng, cfg, wall, interior, gap_mask, mouths, dirt, tree)
                      for _ in range(int(rng.integers(cfg.decoy_gaps[0], cfg.decoy_gaps[1] + 1))))
    tree &= ~dirt
    tree[fr, fc] = False
    dirt[fr, fc] = False

    grid = np.stack([dirt, tree, fire])

    # e. certificate, agent placement
    fields_, reason = _certify(grid, gap_rc, tc, tm, cfg, rng, want_static=want_static,
                               agent_fire_dist=afire)
    if fields_ is None:
        return None, reason

    # f. a random dihedral element on everything
    g = int(rng.integers(0, 8))
    (grid_t, protected_t, fire_side_t), (order_t, agent_t, fire_t), _ = d4_apply(
        g, n, grids=[grid, fields_["protected"], fields_["doomed0"]],
        points=[fields_["order"], np.array(fields_["agent"]), np.array([fr, fc])])

    stats = {
        "k": k, "R": R, "offset": int(max(abs(fr - cr), abs(fc - cc))),
        "d_f": fields_["d_f"], "slack": fields_["slack"], "agent_fire_dist": afire,
        "agent_fire_dist_clamped": afire_clamped, "agent_fire_dist_wanted": afire_wanted,
        "p_noise": p_noise, "decoy_walls": placed_walls, "decoy_loops": placed_loops,
        "decoy_gaps": placed_gaps, "d4": g, "min_cut": fields_["min_cut"],
    }
    return PlantedInstance(
        grid=grid_t, agent=(int(agent_t[0]), int(agent_t[1])),
        fire=(int(fire_t[0]), int(fire_t[1])), gap=order_t.astype(np.int64),
        protected=protected_t, fire_side=fire_side_t,
        oracle_saved=fields_["oracle_saved"], noop_saved=fields_["noop_saved"],
        min_cut=fields_["min_cut"], static_loss_gap=fields_["static_loss_gap"],
        stats=stats,
    ), None


def certify_instance(inst: PlantedInstance, tc, tm, cfg=None, want_static=False):
    """Re-run the certificate on a finished instance (any orientation).

    Used by the tests to show that hiding and the dihedral element change
    nothing the certificate measures.
    """
    cfg = (cfg or PlantedConfig()).resolve(inst.grid.shape[-1], tc, tm)
    if inst.family == "graph":
        return _certify_graph(inst.grid, inst.gap, tc, tm, cfg, inst.agent,
                              order=inst.gap)
    return _certify(inst.grid, inst.gap, tc, tm, cfg, np.random.default_rng(0),
                    agent=inst.agent, want_static=want_static)


# --------------------------------------------------------------------------- #
# The graph family: a network of corridors, and the cut you can reach
# --------------------------------------------------------------------------- #
# Nodes are single-cell junctions, edges are width-1 tree corridors, and the
# fire starts at a junction on the left. To protect a region the agent must
# sever EVERY corridor on that region's boundary -- a cut in the graph sense.
# The instance is built so the richest region is out of reach in time, so the
# agent has to find the most valuable cut it can actually finish. Boundaries
# come out at 3 to 6 corridors.
#
# Why junctions are ONE cell. An earlier version made them 3x3 patches, to carry
# more payoff. That is not certifiable: a patch can be SLICED, one line of cells
# destroyed leaving everything beyond it standing, and that is a cut set the
# contraction cannot express. The planner duly found them and beat the certified
# plan by 130%. A one-cell junction has nothing to slice. It also is not a cut
# vertex -- two corridors meeting there have their first cells diagonally
# adjacent, so fire flows around it without passing through -- which means
# cutting a junction achieves nothing and CORRIDOR CUTS ARE THE ENTIRE CUT-SET
# FAMILY. That is what makes the region search complete, and `_separation_ok`
# deliberately permits that adjacency rather than routing around it.
#
# Two more facts were measured rather than assumed:
#
# 1. Only a STRAIGHT interior cell severs a corridor. At an L-shaped bend the
#    corner's two neighbours are Moore-adjacent, so the fire steps diagonally
#    around a cut corner. `_cut_sites` finds the legal cells with a local flood
#    fill, and a test pins it against a full one.
# 2. Sealing the fire's own junction costs `deg(fire)` cuts, and on a bare
#    lattice that is 2 or 3 -- inside any sensible budget, so every instance
#    would have the same trivial answer. Dead-end spurs raise that price without
#    adding payoff worth chasing.
#
# What is certified. The oracle's plan saves exactly what it claims: its run in
# the real environment matches `oracle_saved` to the tree, with no truncation,
# and `min_vertex_cut` confirms no smaller cut set protects the same region. The
# search walks every region in order of a payoff bound, with a vectorised
# earliest-deadline filter and a Held-Karp feasibility gate, and stops when the
# bound falls to the best value already found.
#
# What is not. The search carries a cut budget (`cuts`, default up to 6) and
# tries one cell per corridor (`cut_sites`), because the feasibility check is
# exponential in the boundary size -- at a budget of 8 it costs 12 s an
# instance. So the plan is the best within that budget, not over all cut sets.
# Measured at 32x32, tc=tm=0.2 over 16 paired episodes, the planner saves 11.1%
# against this oracle's 9.7%, so `oracle_saved` is a tight floor rather than a
# proven ceiling. Read `oracle_gain`, and keep the planner as the yardstick.
#
# Deadlines come from `arrival0` on the untouched grid. With cycles an early cut
# can reroute the fire and buy time, so `oracle_fraction > 1` is legitimate.

_NODE_BASE = 1000


@dataclass
class _GraphLayout:
    """A contracted view of the grid: patches, corridors, and who touches whom."""
    n: int
    owner: np.ndarray            # (n,n) int64: -1 empty, v node, _NODE_BASE+e corridor
    nodes: list                  # (m,2) cells per node
    edges: list                  # (L,2) cells per corridor, ordered u -> v
    ends: list                   # (u, v) per corridor
    sites: list                  # legal cut-site indices into edges[e]
    adj: list                    # node -> incident corridor ids
    tree: np.ndarray
    fire_node: int = -1
    fire_rc: tuple = (-1, -1)
    n_core: int = 0              # lattice junctions; spur tips come after them
    table: tuple = ()            # cached region table, see _region_table


def _span(a, b):
    return range(a, b + 1) if b >= a else range(a, b - 1, -1)


def _route(a, b, rng, horizontal):
    """A 4-connected path from a to b inclusive, with one Z bend."""
    (r1, c1), (r2, c2) = a, b
    cells = []
    if horizontal:
        lo, hi = min(c1, c2), max(c1, c2)
        if hi - lo < 2:
            return None
        pivot = int(rng.integers(lo + 1, hi))
        for c in _span(c1, pivot):
            cells.append((r1, c))
        for r in (_span(r1, r2)[1:] if r1 != r2 else []):
            cells.append((r, pivot))
        for c in _span(pivot, c2)[1:]:
            cells.append((r2, c))
    else:
        lo, hi = min(r1, r2), max(r1, r2)
        if hi - lo < 2:
            return None
        pivot = int(rng.integers(lo + 1, hi))
        for r in _span(r1, pivot):
            cells.append((r, c1))
        for c in (_span(c1, c2)[1:] if c1 != c2 else []):
            cells.append((pivot, c))
        for r in _span(pivot, r2)[1:]:
            cells.append((r, c2))
    return np.array(cells, dtype=np.int64)


def _severs(x, a, b, tree, radius=3):
    """True if removing x disconnects a from b inside a local window.

    Width-1 corridors held apart mean the only way a cut fails to sever is a
    bend, where the corner's two neighbours are Moore-adjacent. A local fill
    catches that exactly and costs a 7x7 window rather than a grid BFS.
    """
    n = tree.shape[0]
    r0, r1 = max(0, x[0] - radius), min(n, x[0] + radius + 1)
    c0, c1 = max(0, x[1] - radius), min(n, x[1] + radius + 1)
    win = tree[r0:r1, c0:c1].copy()
    win[x[0] - r0, x[1] - c0] = False
    sa = (int(a[0] - r0), int(a[1] - c0))
    sb = (int(b[0] - r0), int(b[1] - c0))
    if not win[sa] or not win[sb]:
        return False
    return not _component(win, sa)[sb]


def _cut_sites(cells, tree, before, after):
    """Indices of the corridor cells whose removal actually severs it."""
    out = [i for i in range(len(cells))
           if _severs(cells[i],
                      before if i == 0 else cells[i - 1],
                      after if i == len(cells) - 1 else cells[i + 1], tree)]
    return np.array(out, dtype=np.int64)


def _build_layout(n, rng, cfg):
    """Lattice of patches joined by width-1 corridors. Returns (layout, reason)."""
    L = int(rng.integers(cfg.lattice[0], cfg.lattice[1] + 1))
    p = int(rng.integers(cfg.patch[0], cfg.patch[1] + 1))
    m = p + 1 + cfg.spur_room
    if L < 2 or n - 2 * m < L - 1:
        return None, "lattice_misfit"
    step = (n - 1 - 2 * m) // (L - 1)
    if step < 2 * p + cfg.corridor_min + 2:
        return None, "lattice_tight"

    centres = {}
    for i in range(L):
        for j in range(L):
            r = m + i * step + int(rng.integers(-cfg.jitter, cfg.jitter + 1))
            c = m + j * step + int(rng.integers(-cfg.jitter, cfg.jitter + 1))
            centres[(i, j)] = (int(np.clip(r, m, n - 1 - m)),
                               int(np.clip(c, m, n - 1 - m)))

    order = sorted(centres)
    index = {k: v for v, k in enumerate(order)}
    owner = np.full((n, n), -1, dtype=np.int64)
    tree = np.zeros((n, n), bool)
    nodes = []
    for v, key in enumerate(order):
        r, c = centres[key]
        cells = np.array([(rr, cc) for rr in range(r - p, r + p + 1)
                          for cc in range(c - p, c + p + 1)], dtype=np.int64)
        tree[cells[:, 0], cells[:, 1]] = True
        owner[cells[:, 0], cells[:, 1]] = v
        nodes.append(cells)

    edges, ends, sites = [], [], []
    for key in order:
        i, j = key
        for di, dj in ((0, 1), (1, 0)):
            b = (i + di, j + dj)
            if b not in centres or rng.random() > cfg.edge_keep:
                continue
            (r1, c1), (r2, c2) = centres[key], centres[b]
            if dj:
                cells = _route((r1, c1 + p + 1), (r2, c2 - p - 1), rng, True)
            else:
                cells = _route((r1 + p + 1, c1), (r2 - p - 1, c2), rng, False)
            if cells is None or len(cells) < cfg.corridor_min:
                continue
            if (owner[cells[:, 0], cells[:, 1]] != -1).any():
                continue
            e = len(edges)
            tree[cells[:, 0], cells[:, 1]] = True
            owner[cells[:, 0], cells[:, 1]] = _NODE_BASE + e
            edges.append(cells)
            ends.append((index[key], index[b]))
            sites.append(None)

    adj = [[] for _ in range(len(nodes))]
    for e, (u, v) in enumerate(ends):
        adj[u].append(e)
        adj[v].append(e)
    return _GraphLayout(n=n, owner=owner, nodes=nodes, edges=edges, ends=ends,
                        sites=sites, adj=adj, tree=tree, n_core=len(nodes)), None


def _separation_ok(layout):
    """Only elements that actually meet may have Moore-adjacent tree cells.

    This is the whole basis of the contraction: it is what makes "cut one cell
    of this corridor" mean "remove this edge", so it is checked rather than
    trusted to the routing.

    Two corridors that share a junction are allowed to touch, and with one-cell
    junctions they always do: their first cells sit either side of the junction,
    diagonally adjacent. That is not a flaw to be routed around, it is the
    reason this geometry is certifiable. It means fire flows around a junction
    without passing through it, so a junction is not a cut vertex and cutting it
    achieves nothing -- which leaves corridor cuts as the ENTIRE cut-set family,
    and the region search over corridor boundaries therefore complete. A
    multi-cell junction would be a cut vertex, and could also be sliced, which
    is the hole that made the earlier 3x3-node version uncertifiable.
    """
    owner, n = layout.owner, layout.n
    pad = np.full((n + 2, n + 2), -1, dtype=np.int64)
    pad[1:-1, 1:-1] = owner
    allowed = set()
    at_node = {}
    for e, (u, v) in enumerate(layout.ends):
        allowed.update({(u, _NODE_BASE + e), (_NODE_BASE + e, u),
                        (v, _NODE_BASE + e), (_NODE_BASE + e, v)})
        at_node.setdefault(u, []).append(e)
        at_node.setdefault(v, []).append(e)
    for es in at_node.values():
        for a in es:
            for b in es:
                allowed.add((_NODE_BASE + a, _NODE_BASE + b))
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nb = pad[1 + dr:1 + dr + n, 1 + dc:1 + dc + n]
            bad = (owner >= 0) & (nb >= 0) & (owner != nb)
            if not bad.any():
                continue
            for a, b in zip(owner[bad].tolist(), nb[bad].tolist()):
                if (a, b) not in allowed:
                    return False
    return True


def _finish_sites(layout):
    """Legal cut sites per corridor, once the whole grid exists."""
    for e, cells in enumerate(layout.edges):
        u, v = layout.ends[e]
        nu, nv = layout.nodes[u], layout.nodes[v]
        du = np.maximum(np.abs(nu[:, 0] - cells[0][0]), np.abs(nu[:, 1] - cells[0][1]))
        dv = np.maximum(np.abs(nv[:, 0] - cells[-1][0]), np.abs(nv[:, 1] - cells[-1][1]))
        layout.sites[e] = _cut_sites(cells, layout.tree,
                                     tuple(nu[int(np.argmin(du))]),
                                     tuple(nv[int(np.argmin(dv))]))
        if len(layout.sites[e]) == 0:
            return False
    return True


def _add_spurs(layout, node, rng, cfg, want):
    """Dead-end corridors off `node`, to raise the price of sealing it."""
    n = layout.n
    cells0 = layout.nodes[node]
    r0, c0 = int(cells0[:, 0].min()), int(cells0[:, 1].min())
    r1, c1 = int(cells0[:, 0].max()), int(cells0[:, 1].max())
    made = 0
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        if made >= want:
            break
        if dr:
            start = ((r1 + 1) if dr > 0 else (r0 - 1), (c0 + c1) // 2)
        else:
            start = ((r0 + r1) // 2, (c1 + 1) if dc > 0 else (c0 - 1))
        room = (start[0] if dr < 0 else n - 1 - start[0]) if dr else \
               (start[1] if dc < 0 else n - 1 - start[1])
        length = min(int(rng.integers(cfg.corridor_min + 1, cfg.corridor_min + 5)),
                     room + 1)
        if length < cfg.corridor_min + 1:
            continue
        cells = np.array([(start[0] + dr * i, start[1] + dc * i)
                          for i in range(length)], dtype=np.int64)
        if (cells < 0).any() or (cells >= n).any():
            continue
        if (layout.owner[cells[:, 0], cells[:, 1]] != -1).any():
            continue
        blocked = False
        for r, c in cells:
            win = layout.owner[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2]
            if ((win != -1) & (win != node)).any():
                blocked = True
                break
        if blocked:
            continue
        e, leaf = len(layout.edges), len(layout.nodes)
        layout.tree[cells[:, 0], cells[:, 1]] = True
        layout.owner[cells[:, 0], cells[:, 1]] = _NODE_BASE + e
        tip = cells[-1:].copy()
        layout.owner[tip[0][0], tip[0][1]] = leaf
        layout.nodes.append(tip)
        layout.edges.append(cells[:-1])
        layout.ends.append((node, leaf))
        layout.sites.append(None)
        layout.adj.append([e])
        layout.adj[node].append(e)
        made += 1
    return made


def _graph_value(layout, sel, cut_nodes=()):
    """Trees left standing, computed on the contracted graph, not the grid.

    `sel` maps a corridor to the index of the cell cut in it; `cut_nodes` lists
    junctions cut outright, which severs every corridor meeting them for the
    price of one cell. With one-cell junctions those two are the whole cut-set
    family, so this is exact rather than a restriction -- and it is O(V+E)
    instead of a grid flood fill, which is what makes the search affordable at
    reset.
    """
    V = len(layout.nodes)
    dead = set(cut_nodes)
    seen = np.zeros(V, bool)
    seen[layout.fire_node] = True
    stack = [layout.fire_node]
    while stack:
        x = stack.pop()
        for e in layout.adj[x]:
            if e in sel:
                continue
            u, v = layout.ends[e]
            y = v if u == x else u
            if y in dead or seen[y]:
                continue
            seen[y] = True
            stack.append(y)
    tree = layout.tree

    def live(cells):
        return int(tree[cells[:, 0], cells[:, 1]].sum())

    burnt = sum(live(layout.nodes[v]) for v in range(V) if seen[v])
    for e, cells in enumerate(layout.edges):
        u, v = layout.ends[e]
        u_lit = seen[u] and u not in dead
        v_lit = seen[v] and v not in dead
        if e not in sel:
            # a corridor whose junction was cut still burns from the other end
            if u_lit or v_lit:
                burnt += live(cells)
        else:
            i = sel[e]
            if u_lit and i:
                burnt += live(cells[:i])
            if v_lit and i + 1 < len(cells):
                burnt += live(cells[i + 1:])
    return int(tree.sum()) - burnt - len(sel) - len(dead)


def _region_table(layout, arrival, tc, margin):
    """Every node subset, with its boundary size and a payoff upper bound.

    Vectorised over all 2**V masks. The bound counts the nodes and corridors
    wholly inside the region plus, per boundary corridor, the most cells a legal
    cut could leave standing on the region's side -- tighter than the corridor
    length because only straight-interior cells sever, and it is the tightness
    that makes the branch-and-bound prune instead of walking every region.
    """
    # Only the lattice junctions are enumerated. A spur tip is a dead end whose
    # payoff is its own few cells, so letting it double the table would buy
    # nothing; tips are simply always outside the region.
    V, E = layout.n_core, len(layout.edges)
    masks = np.arange(1 << V, dtype=np.int64)
    core = ((masks[:, None] >> np.arange(V)[None, :]) & 1).astype(bool)
    bits = np.zeros((core.shape[0], len(layout.nodes)), dtype=bool)
    bits[:, :V] = core
    u = np.array([e[0] for e in layout.ends], dtype=np.int64)
    v = np.array([e[1] for e in layout.ends], dtype=np.int64)
    inu, inv = bits[:, u], bits[:, v]
    boundary = inu ^ inv
    internal = inu & inv
    tree = layout.tree
    node_w = np.array([int(tree[c[:, 0], c[:, 1]].sum()) for c in layout.nodes])
    edge_w = np.array([int(tree[c[:, 0], c[:, 1]].sum()) for c in layout.edges])
    suf_u = np.zeros(E, dtype=np.int64)
    suf_v = np.zeros(E, dtype=np.int64)
    for e, cells in enumerate(layout.edges):
        idx = layout.sites[e]
        if len(idx):
            suf_u[e] = int(idx.max())
            suf_v[e] = int(len(cells) - 1 - idx.min())
    gain = np.where(inu, suf_u[None, :], suf_v[None, :]) * boundary
    payoff_ub = (bits @ node_w) + (internal @ edge_w) + gain.sum(1)
    nb = boundary.sum(1)
    # An earliest-deadline filter, vectorised over every region at once. Even
    # with no travel at all the i-th cut cannot land before (i+1)*tc, so a
    # region whose i-th earliest boundary deadline has already passed is
    # infeasible for every order. Running it here, on the whole table, is what
    # makes a wide cut budget affordable: the Python loop below then walks
    # hundreds of regions instead of tens of thousands.
    edge_dead = np.full(E, -1.0)
    for e, cells in enumerate(layout.edges):
        idx_e = layout.sites[e]
        if len(idx_e):
            pts = cells[idx_e]
            edge_dead[e] = float(arrival[pts[:, 0], pts[:, 1]].max())
    dead = np.where(boundary, edge_dead[None, :], np.inf)
    dead.sort(axis=1)
    need = np.floor((np.arange(E) + 1) * tc - 1.0) + 1 + margin
    in_time = (need[None, :] < dead).all(1)

    live = (~bits[:, layout.fire_node]) & (nb >= 1) & (payoff_ub > 0) & in_time
    idx = np.flatnonzero(live)
    order = idx[np.argsort(-payoff_ub[idx], kind="stable")]
    return order, bits, boundary, payoff_ub, nb


def _regions_for(layout, arrival, tc, margin, cut_max):
    """The region table, computed once per layout and filtered per cut budget.

    Nothing in it depends on where the agent stands, and the agent is retried
    several times per layout, so computing it once saves most of the search.
    """
    if not layout.table:
        layout.table = _region_table(layout, arrival, tc, margin)
    order, bits, boundary, payoff_ub, nb = layout.table
    return order[nb[order] <= cut_max], bits, boundary, payoff_ub


def _picks(cand, cap=48):
    out = [()]
    for cells in cand:
        out = [p + (i,) for p in out for i in range(len(cells))][:cap]
    return out


def _any_feasible(cand, agent, t0, tc, tm, arrival, margin):
    """Can ANY choice of one cut site per corridor be finished in time?

    One Held-Karp over (corridors cut so far, last cell), where the cell ranges
    over every candidate site of every corridor. Most regions the search walks
    are infeasible, and without this gate each costs a scheduler run per site
    combination instead of one.
    """
    k = len(cand)
    if k == 0:
        return True
    flat = [(j, cell) for j, cells in enumerate(cand) for cell in cells]
    best = {}
    for fi, (j, cell) in enumerate(flat):
        tau = max(abs(cell[0] - agent[0]), abs(cell[1] - agent[1])) * tm + tc
        if math.floor(tau - t0) + 1 + margin < arrival[cell]:
            key = (1 << j, fi)
            if key not in best or tau < best[key]:
                best[key] = tau
    for _ in range(k - 1):
        nxt = {}
        for (mask, fi), tau in best.items():
            cell = flat[fi][1]
            for fj, (j2, c2) in enumerate(flat):
                if mask & (1 << j2):
                    continue
                t2 = tau + max(abs(c2[0] - cell[0]), abs(c2[1] - cell[1])) * tm + tc
                if math.floor(t2 - t0) + 1 + margin >= arrival[c2]:
                    continue
                key = (mask | (1 << j2), fj)
                if key not in nxt or t2 < nxt[key]:
                    nxt[key] = t2
        best = nxt
        if not best:
            return False
    full = (1 << k) - 1
    return any(mask == full for mask, _ in best)


def _schedule_plan(cells, agent, t0, tc, tm, arrival, margin):
    """The fastest order in which these cells can all be cut in time, or None.

    The recurrence is `_schedule`'s -- Chebyshev travel at `tm` a cell, `tc` a
    cut, and a cut lands legally only while `floor(tau - t0) + 1 + margin <
    arrival` -- so a plan this accepts is one the environment executes.
    """
    k = len(cells)
    if k == 0:
        return []
    pts = np.array(cells, dtype=np.int64)
    dead = arrival[pts[:, 0], pts[:, 1]]
    best = {}
    for i in range(k):
        tau = max(abs(pts[i, 0] - agent[0]), abs(pts[i, 1] - agent[1])) * tm + tc
        if math.floor(tau - t0) + 1 + margin < dead[i]:
            best[(1 << i, i)] = (tau, [i])
    for _ in range(k - 1):
        nxt = {}
        for (mask, last), (tau, path) in best.items():
            for j in range(k):
                if mask & (1 << j):
                    continue
                t2 = tau + max(abs(pts[j, 0] - pts[last, 0]),
                               abs(pts[j, 1] - pts[last, 1])) * tm + tc
                if math.floor(t2 - t0) + 1 + margin >= dead[j]:
                    continue
                key = (mask | (1 << j), j)
                if key not in nxt or t2 < nxt[key][0]:
                    nxt[key] = (t2, path + [j])
        best = nxt
        if not best:
            return None
    full = (1 << k) - 1
    done = [(tau, path) for (mask, _), (tau, path) in best.items() if mask == full]
    if not done:
        return None
    return [cells[i] for i in min(done)[1]]


def _best_plan(layout, arrival, agent, tc, tm, cfg, cut_max):
    """The most valuable cut set the agent can finish, and the richest it cannot.

    Regions are walked in order of their payoff bound and the walk stops once
    that bound falls to the best value already achieved, so the winner is
    optimal over the corridor-boundary family rather than merely the first
    feasible one. `rich` is the best strictly richer region that proved
    infeasible -- the trap the instance is built around.
    """
    order, bits, boundary, payoff_ub = _regions_for(layout, arrival, tc,
                                                    cfg.margin, cut_max)
    best = rich = None
    seen = set()
    for row in order:
        if best is not None and payoff_ub[row] <= best[0]:
            break
        edges = np.flatnonzero(boundary[row])
        key = tuple(edges.tolist())
        if key in seen:
            continue
        seen.add(key)
        inside = bits[row]
        cand, gains, chosen = [], [], []
        for e in edges:
            # Ordered by deadline, latest first: the schedule binds on the
            # deadline, so with a small site budget that is the cell to try.
            pts = layout.edges[e][layout.sites[e]]
            rank = np.argsort(-arrival[pts[:, 0], pts[:, 1]], kind="stable")
            idx = layout.sites[e][rank][:cfg.cut_sites]
            cells = layout.edges[e]
            u, _v = layout.ends[e]
            chosen.append([int(i) for i in idx])
            cand.append([tuple(cells[i]) for i in idx])
            gains.append([(int(i) if inside[u] else len(cells) - 1 - int(i))
                          for i in idx])
        if not _any_feasible(cand, agent, 1.0, tc, tm, arrival, cfg.margin):
            if rich is None or payoff_ub[row] > rich[0]:
                rich = (int(payoff_ub[row]), len(edges))
            continue
        # Site selections in descending value, so the first schedulable one is
        # the best schedulable one and no region needs a second pass.
        picks = sorted(_picks(cand),
                       key=lambda pk: -sum(gains[j][pk[j]] for j in range(len(edges))))
        for pick in picks:
            combo = [cand[j][pick[j]] for j in range(len(edges))]
            got = _schedule_plan(combo, agent, 1.0, tc, tm, arrival, cfg.margin)
            if got is None:
                continue
            # `pick` indexes the deadline-sorted candidates, so the cell index
            # must come from `chosen`, not from layout.sites. Taking it from the
            # unsorted array made the valued cut set differ from the scheduled
            # one, which the value_mismatch guard caught.
            sel = {int(edges[j]): chosen[j][pick[j]] for j in range(len(edges))}
            value = _graph_value(layout, sel)
            if best is None or value > best[0]:
                best = (value, got, len(edges), sel)
            break
    return best, rich


def _slice_threat(layout, fire, arrival, agent, tc, tm, cfg, floor_value):
    """Does slicing a single patch beat the certified plan?

    A slice is a cut set the corridor search cannot express: one line of patch
    cells destroyed leaves everything beyond it standing. At 3x3 it can never
    pay -- it costs 3 cells to protect what one corridor cut protects -- but the
    check is cheap and it is what keeps the certificate honest if the patch size
    is ever raised.
    """
    for cells in layout.nodes:
        if len(cells) < 9:
            continue
        rows = np.unique(cells[:, 0])
        cols = np.unique(cells[:, 1])
        for line, axis in [(r, 0) for r in rows] + [(c, 1) for c in cols]:
            pick = cells[cells[:, axis] == line]
            combo = [tuple(x) for x in pick]
            if len(combo) > cfg.cuts[1] + 2:
                continue
            if _schedule_plan(combo, agent, 1.0, tc, tm, arrival, cfg.margin) is None:
                continue
            cut = layout.tree.copy()
            cut[pick[:, 0], pick[:, 1]] = False
            saved = int((cut & (fire_arrival(cut, fire) >= INF)).sum())
            if saved > floor_value:
                return True
    return False


def _certify_graph(grid, cut_rc, tc, tm, cfg, agent, order=None):
    """Re-derive what a graph instance's cut set achieves, from the grid alone.

    No layout needed, so it works on any orientation of a finished instance --
    which is what lets the tests re-certify all eight dihedral images and
    confirm the certificate is a property of the grid, not of the bookkeeping.
    """
    n = grid.shape[-1]
    tree, fire = grid[TREE], grid[FIRE]
    cut_rc = np.asarray(cut_rc, dtype=np.int64)
    k = len(cut_rc)
    cut_mask = np.zeros((n, n), dtype=bool)
    cut_mask[cut_rc[:, 0], cut_rc[:, 1]] = True
    if not tree[cut_mask].all():
        return None, "gap_not_tree"
    arrival0 = fire_arrival(tree, fire)
    if arrival0[cut_mask].max() >= INF:
        return None, "gap_unreachable"
    doomed0 = tree & (arrival0 < INF)
    tree_cut = tree & ~cut_mask
    doomed_cut = tree_cut & (fire_arrival(tree_cut, fire) < INF)
    protected = tree_cut & ~doomed_cut & doomed0
    order = cut_rc if order is None else np.asarray(order, dtype=np.int64)
    if _schedule(order[:, 0], order[:, 1], tuple(agent), 1.0, tc, tm,
                 arrival0, cfg.margin) is None:
        return None, "schedule_infeasible"
    area = n * n
    return {
        "arrival0": arrival0, "doomed0": doomed0, "doomed_cut": doomed_cut,
        "protected": protected, "order": order, "agent": tuple(agent),
        "oracle_saved": float(tree_cut.sum() - doomed_cut.sum()) / area,
        "noop_saved": float((tree & ~doomed0).sum()) / area,
        "at_risk": int(doomed0.sum()),
        "saved_trees": int(tree_cut.sum() - doomed_cut.sum()),
        "idle_ticks": int(arrival0[doomed0].max()) if doomed0.any() else 0,
    }, None


def _attempt_graph(n, tc, tm, rng, cfg, want_static):
    layout, why = _build_layout(n, rng, cfg)
    if layout is None:
        return None, why

    # The fire goes in a left-hand patch, and gets enough corridors that sealing
    # its own patch is out of budget -- otherwise every instance has the same
    # trivial answer and nothing is learned about choosing a region.
    left = [v for v in range(len(layout.nodes))
            if layout.nodes[v][:, 1].mean() < n / 2]
    if not left:
        return None, "no_left_node"
    # Pick the LOWEST-degree left node that still meets the target, so that
    # fire_degree actually binds. Taking the highest-degree node regardless
    # meant the knob did nothing and every instance came out at four cuts.
    eligible = [v for v in left if len(layout.adj[v]) >= cfg.fire_degree]
    fire_node = (min(eligible, key=lambda v: len(layout.adj[v])) if eligible
                 else max(left, key=lambda v: len(layout.adj[v])))
    layout.fire_node = fire_node
    _add_spurs(layout, fire_node, rng, cfg,
               want=max(0, cfg.fire_degree - len(layout.adj[fire_node])))
    # Cheap structural gates first: `_finish_sites` walks every corridor cell
    # with a local flood fill, so it must not run on a layout already doomed.
    # A cycle is what forces a region to need several corridors cut. Without it
    # every region hangs by a single corridor, so one cut wins the whole payoff
    # with the agent still far away -- slack AND payoff together, which is the
    # easy tier a curriculum needs and which no other knob here provides. Every
    # other knob trades one against the other.
    if cfg.require_cycle and len(layout.edges) - len(layout.nodes) + 1 < 1:
        return None, "acyclic"
    if len(layout.adj[fire_node]) < cfg.fire_degree:
        return None, "fire_degree_low"
    if not _separation_ok(layout):
        return None, "elements_touch"
    if not _finish_sites(layout):
        return None, "no_cut_site"

    cells = layout.nodes[fire_node]
    fr, fc = (int(v) for v in cells[len(cells) // 2])
    layout.fire_rc = (fr, fc)
    fire = np.zeros((n, n), dtype=bool)
    fire[fr, fc] = True
    layout.tree[fr, fc] = False
    tree = layout.tree
    # No "fire must not touch dirt" gate here: a one-cell junction always has
    # dirt neighbours. The planner detects a reset by dirt shrinking instead,
    # which holds for every family.

    arrival0 = fire_arrival(tree, fire)
    doomed0 = tree & (arrival0 < INF)
    if not doomed0.any():
        return None, "nothing_at_risk"
    # The fire must burn out well inside the episode cap, or eval measures
    # truncations instead of outcomes. A long serpentine graph would not.
    if int(arrival0[doomed0].max()) + 2 > cfg.burn_budget * cfg.cap_factor() * n:
        return None, "slow_burn"
    if (tree & ~doomed0).sum() > cfg.free_max_frac * tree.sum():
        return None, "free_trees"

    # The agent starts far enough away that it cannot simply box in the fire
    # inside its own patch, the same threat scan the pocket family uses.
    afire = _agent_fire_distance_cached(1, tc, tm)
    rr, cc = np.indices((n, n))
    from_fire = np.maximum(np.abs(rr - fr), np.abs(cc - fc))
    ok = (from_fire >= max(afire, cfg.agent_min_dist)) & ~fire
    cand = np.argwhere(ok)
    if len(cand) == 0:
        return None, "no_agent_site"

    # Where the agent stands is what decides which regions are reachable, so it
    # is what creates the trap -- and it is far cheaper to retry than the whole
    # layout. Candidates are tried farthest-first, since a distant agent leaves
    # the richest region out of reach while a near one can have everything.
    rng.shuffle(cand)
    pool = cand[:cfg.agent_tries * 4]
    far = np.maximum(np.abs(pool[:, 0] - fr), np.abs(pool[:, 1] - fc))
    # Distance from the fire is the difficulty knob. Far, and the cheap region
    # around the ignition point has already burnt by the time the agent gets
    # there, leaving only a distant many-corridor boundary. Near, and sealing
    # the source is reachable, which is the easy tier a curriculum starts on.
    pool = pool[np.argsort(-far if cfg.agent_far else far, kind="stable")][:cfg.agent_tries]
    last = "no_region"
    chosen = None
    for row in pool:
        agent = (int(row[0]), int(row[1]))
        # Budget for the search. Raising it to K_MAX makes the plan better but
        # costs seconds per instance, because the Held-Karp over a large
        # boundary dominates; cfg.search_cuts is the knob. Whatever it is, the
        # plan is the best the agent can finish WITHIN it, which is why the
        # oracle is documented as a certified floor and the planner, which has
        # no budget, is the yardstick.
        best, rich = _best_plan(layout, arrival0, agent, tc, tm, cfg,
                                cfg.search_cuts or cfg.cuts[1])
        if best is None:
            last = "no_region"
            continue
        value, order, n_edges, sel = best
        if not (cfg.cuts[0] <= n_edges <= cfg.cuts[1]):
            last = "cuts_out_of_range"
            continue
        if value < cfg.risk_min_frac * int(doomed0.sum()):
            last = "risk_small"
            continue
        # Trap: a strictly richer region exists that cannot be severed in time.
        if cfg.require_trap and (rich is None or rich[0] <= value):
            last = "no_trap_rich"
            continue
        # Trap: the cheapest option -- a single corridor -- must not pay nearly
        # as well, or the agent can ignore the choice and cut what is easiest.
        cheap = None
        if n_edges >= 2:
            cheap, _ = _best_plan(layout, arrival0, agent, tc, tm, cfg, 1)
            if cheap is not None and cheap[0] > cfg.trap_ratio * value:
                last = "cheap_is_good"
                continue
        if _slice_threat(layout, fire, arrival0, agent, tc, tm, cfg, value):
            last = "slice_beats_plan"
            continue
        chosen = (agent, value, order, n_edges, rich, cheap)
        break
    if chosen is None:
        return None, last
    agent, value, order, n_edges, rich, cheap = chosen

    grid = np.stack([~tree & ~fire, tree, fire])
    fields_, reason = _certify_graph(grid, order, tc, tm, cfg, agent, order=order)
    if fields_ is None:
        return None, reason
    if fields_["saved_trees"] != value:
        return None, "value_mismatch"
    mc, _mask = min_vertex_cut(tree, fire, fields_["protected"], stop_above=n_edges)
    if mc != n_edges:
        return None, f"min_cut_{mc}"

    static_gap = None
    if want_static or cfg.require_static_optimal:
        static_gap = (n_edges + int(fields_["doomed_cut"].sum())
                      - static_min_loss(tree, fire))
        if cfg.require_static_optimal and static_gap != 0:
            return None, "not_static_optimal"

    g = int(rng.integers(0, 8))
    (grid_t, protected_t, fire_side_t), (order_t, agent_t, fire_t), _ = d4_apply(
        g, n, grids=[grid, fields_["protected"], fields_["doomed0"]],
        points=[np.asarray(order, dtype=np.int64), np.array(agent), np.array([fr, fc])])

    stats = {
        "k": n_edges, "d_f": int(fields_["arrival0"][
            np.asarray(order)[:, 0], np.asarray(order)[:, 1]].min()),
        "slack": 0.0, "p_noise": 0.0, "d4": g, "min_cut": mc,
        "nodes": len(layout.nodes), "edges": len(layout.edges),
        "cycles": len(layout.edges) - len(layout.nodes) + 1,
        "fire_degree": len(layout.adj[fire_node]),
        "at_risk": fields_["at_risk"], "saved_trees": value,
        "rich_payoff": -1 if rich is None else int(rich[0]),
        "cheap_payoff": 0 if cheap is None else int(cheap[0]),
        "idle_ticks": fields_["idle_ticks"], "agent_fire_dist": afire,
        "corridor_len": float(np.mean([len(c) for c in layout.edges])),
    }
    return PlantedInstance(
        grid=grid_t, agent=(int(agent_t[0]), int(agent_t[1])),
        fire=(int(fire_t[0]), int(fire_t[1])), gap=order_t.astype(np.int64),
        protected=protected_t, fire_side=fire_side_t,
        oracle_saved=fields_["oracle_saved"], noop_saved=fields_["noop_saved"],
        min_cut=mc, static_loss_gap=static_gap, stats=stats, family="graph",
    ), None


# --------------------------------------------------------------------------- #
# Oracle policy
# --------------------------------------------------------------------------- #
class PlantedOraclePolicy:
    """Cuts the planted gap in the certified order, then stays out of the way.

    Stateless: ``params/gap`` carries the cells in cutting order, so each step
    the policy heads for the first one still standing and cuts it on arrival.
    Once none is left it moves (moving is free) and never cuts again. Drop-in
    for the same harness as ``BarrierPlannerPolicy`` and a PPO actor.
    """

    def __init__(self, idle_action: int = 0):
        self.idle_action = idle_action

    def reset(self):
        pass

    @staticmethod
    def act(grid_np, agent_rc, gap_np, idle_action=0):
        for r, c in gap_np:
            if r < 0:
                break
            if grid_np[TREE, r, c]:
                if (int(agent_rc[0]), int(agent_rc[1])) == (int(r), int(c)):
                    return CUT
                return _ACTION_OF_DELTA[(int(np.sign(r - agent_rc[0])),
                                         int(np.sign(c - agent_rc[1])))]
        return idle_action

    def __call__(self, tensordict):
        grid = tensordict["grid"]
        batch_shape = grid.shape[:-3]
        grids = grid.reshape(-1, *grid.shape[-3:]).cpu().numpy()
        agents = tensordict["agent_state"].reshape(-1, 2).cpu().numpy()
        gaps = tensordict["params", "gap"].reshape(-1, K_MAX, 2).cpu().numpy()
        actions = np.empty(len(grids), dtype=np.int64)
        for i in range(len(grids)):
            actions[i] = self.act(grids[i], agents[i], gaps[i], self.idle_action)
        tensordict.set("action", torch.as_tensor(
            actions.reshape(batch_shape), device=grid.device, dtype=torch.int64))
        return tensordict
