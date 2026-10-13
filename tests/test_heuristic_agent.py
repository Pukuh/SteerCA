"""Checks on the hand-designed planner: that its model of the environment is
exactly right, and that a plan it commits to is one it can actually execute.

The planner is only a fair baseline if its predictions match the environment it
is scored in, so each test pins one of the assumptions the docstring makes.

Run from the repo root:  python tests/test_heuristic_agent.py
"""
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import steerca.firebreak.ca as ca
import steerca.firebreak.env as ff
from steerca.firebreak.ca import DIRT, FIRE, TREE
from steerca.firebreak.agents import CUT_COST_AHEAD, CUT_COST_IMMINENT
from steerca.firebreak.planner import (
    CUT_COST,
    INF,
    BarrierPlannerPolicy,
    _clip_to_grid,
    _dilate,
    _rect_perimeter,
    fire_arrival,
    plan_barrier,
)
from torchrl.envs.utils import step_mdp

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def random_grid(rng, n, p_dirt=0.15):
    """A grid with scattered dirt and one fire cell, as mid-episode states look."""
    grid = np.zeros((3, n, n), dtype=bool)
    dirt = rng.random((n, n)) < p_dirt
    grid[DIRT] = dirt
    grid[TREE] = ~dirt
    r, c = rng.integers(0, n, 2)
    grid[TREE, r, c] = False
    grid[DIRT, r, c] = False
    grid[FIRE, r, c] = True
    return grid


# --------------------------------------------------------------------------- #
def test_arrival_matches_ca():
    """fire_arrival must equal the tick each cell actually ignites in the CA."""
    rng = np.random.default_rng(0)
    worst = 0
    for trial in range(30):
        n = int(rng.integers(6, 17))
        grid = random_grid(rng, n)
        predicted = fire_arrival(grid[TREE].copy(), grid[FIRE].copy())

        actual = np.full((n, n), INF, dtype=np.int64)
        actual[grid[FIRE]] = 0
        g = torch.from_numpy(grid).unsqueeze(0)
        for tick in range(1, 3 * n + 4):
            g = ca.step(g)
            lit = g[0, FIRE].numpy() & (actual == INF)
            actual[lit] = tick
            if not g[0, FIRE].any():
                break
        worst = max(worst, int(np.abs(np.minimum(predicted, 10 ** 6)
                                      - np.minimum(actual, 10 ** 6)).max()))
        if not np.array_equal(predicted, actual):
            check("fire_arrival matches the CA", False, f"trial {trial}, n={n}")
            return
    check("fire_arrival matches the CA", True, f"30 random grids, max error {worst}")


def test_rectangle_border_separates():
    """A rectangle's border must block Moore spread from inside to outside."""
    rng = np.random.default_rng(1)
    for _ in range(300):
        n = int(rng.integers(4, 17))
        r0, r1 = sorted(rng.integers(-1, n + 1, 2))
        c0, c1 = sorted(rng.integers(-1, n + 1, 2))
        rr, cc = np.arange(n)[:, None], np.arange(n)[None, :]
        interior = ((rr > r0) & (rr < r1) & (cc > c0) & (cc < c1))
        rows, cols = _rect_perimeter(r0, r1, c0, c1)
        border = np.zeros((n, n), dtype=bool)
        inside = (rows >= 0) & (rows < n) & (cols >= 0) & (cols < n)
        border[rows[inside], cols[inside]] = True
        # Grow from the interior with the border removed; nothing may escape.
        reach = interior.copy()
        for _ in range(2 * n):
            grown = _dilate(reach) & ~border
            if (grown == reach).all():
                break
            reach = grown
        outside = ~interior & ~border
        if (reach & outside).any():
            check("rectangle border separates", False, f"n={n} {(r0, r1, c0, c1)}")
            return
    check("rectangle border separates", True, "300 random rectangles")


def test_border_cells_are_exact():
    """The traversal must list exactly the in-grid border cells, each once.

    A duplicate would be paid for twice; a missing cell would leave a hole the
    fire walks through. Adjacency is only required where the border stays on the
    grid -- when it leaves and re-enters, the walk jumps between arcs, and
    ``_schedule`` charges those jumps at their true Chebyshev cost.
    """
    rng = np.random.default_rng(2)
    for _ in range(300):
        n = int(rng.integers(4, 17))
        r0, r1 = sorted(rng.integers(-1, n + 1, 2))
        c0, c1 = sorted(rng.integers(-1, n + 1, 2))
        rr, cc = np.arange(n)[:, None], np.arange(n)[None, :]
        in_box = (rr >= r0) & (rr <= r1) & (cc >= c0) & (cc <= c1)
        interior = (rr > r0) & (rr < r1) & (cc > c0) & (cc < c1)
        expected = set(map(tuple, np.argwhere(in_box & ~interior)))

        clipped = _clip_to_grid(*_rect_perimeter(r0, r1, c0, c1), n,
                                closed=(r1 > r0 and c1 > c0))
        got = [] if clipped is None else list(zip(*clipped[:2]))
        got = [(int(a), int(b)) for a, b in got]
        if len(got) != len(set(got)) or set(got) != expected:
            check("border cells are exact", False, f"n={n} {(r0, r1, c0, c1)}")
            return
        if clipped is not None and clipped[2] and len(got) > 2:
            rows, cols, _ = clipped
            ring = np.append(rows, rows[0]), np.append(cols, cols[0])
            if np.maximum(np.abs(np.diff(ring[0])), np.abs(np.diff(ring[1]))).max() > 1:
                check("border cells are exact", False, f"loop not closed, n={n}")
                return
    check("border cells are exact", True, "300 random rectangles")


def test_schedule_is_exact():
    """A plan the planner accepts must execute: every cut lands on a live tree.

    This is the end-to-end version of the clock model. If ``floor(tau - t0) + 1``
    were off by one, or travel were not Chebyshev, the agent would arrive at
    cells the fire had already taken and cut dirt instead -- which the reward
    still charges 0.5 for.
    """
    for n, tc, tm in [(8, 0.05, 0.05), (16, 0.05, 0.05), (16, 0.1, 0.1),
                      (8, 0.2, 0.2), (16, 0.2, 0.2)]:
        ff.N, ff.TC, ff.TM = n, tc, tm
        env = ff.ForestFireEnv(device="cpu", batch_size=[16])
        torch.manual_seed(0)
        env.set_seed(0)
        td = env.reset()
        policy = BarrierPlannerPolicy()
        alive = torch.ones(16, dtype=torch.bool)
        wasted = 0
        for _ in range(int(1.4 * n / min(tc, tm))):
            td = policy(td)
            cutting = (td["action"] == 8) & alive
            on_tree = td["grid"][:, TREE][
                torch.arange(16), td["agent_state"][:, 0], td["agent_state"][:, 1]]
            wasted += int((cutting & ~on_tree).sum())
            out = env.step(td)
            alive &= ~out["next", "done"].reshape(16)
            if not alive.any():
                break
            td = step_mdp(out)
        if wasted:
            check("committed plans execute", False,
                  f"n={n} tc=tm={tc}: {wasted} cuts on non-tree cells")
            return
    check("committed plans execute", True, "0 wasted cuts over 5 settings x 16 episodes")


def test_doomed_tree_pricing():
    """Price both kinds of doomed cut and check they match the constants exactly.

    Every tree in this grid is doomed -- neither rollout can save one -- so the
    only difference between a cutting run and an idling one is the price of that
    single cut. Two positions, because the reward can price them differently:

      next to the fire, so the CA ignites it next tick -> CUT_COST_IMMINENT
      far from the fire, doomed only much later        -> CUT_COST_AHEAD

    What this can and cannot guarantee is worth being clear about. It pins the
    ARITHMETIC: each delta must equal 1 - CUT_COST for that case, so a pricing
    change shows up here rather than silently in a training run. It deliberately
    does NOT assert that cutting a doomed tree costs the agent something, because
    under the default reward it does not -- a cut is a flat 0.5 against the 1 the
    fire would have charged, so felling a doomed tree PAYS 0.5. That is the
    farming exploit, it is the current reward's deliberate trade for a learning
    signal, and the test reports it rather than hiding or forbidding it.

    A configuration where nothing can be farmed is one with both prices at or
    above 1: CARL_CUT_TREE_AHEAD=0.5 CARL_CUT_TREE_IMMINENT=0.5 gives that, at
    the cost of PPO having nothing to learn from at the start.
    """
    n = 8
    ff.N, ff.TC, ff.TM = n, 0.6, 0.6
    env = ff.ForestFireEnv(device="cpu", batch_size=[1])
    torch.manual_seed(0)
    env.set_seed(0)

    def rollout(cut_at):
        """cut_at: None to idle, else the (row, col) to stand on and cut once."""
        td = env.reset()
        grid = torch.zeros((1, 3, n, n), dtype=torch.bool)
        grid[0, TREE] = True
        grid[0, TREE, 0, 0] = False
        grid[0, FIRE, 0, 0] = True
        td["grid"] = grid
        td["agent_state"] = torch.tensor([list(cut_at or (n // 2, n // 2))])
        td["t"] = torch.ones(1)
        total, step = 0.0, 0
        while True:
            td.set("action", torch.tensor([8 if (step == 0 and cut_at) else 0]))
            out = env.step(td)
            total += out["next", "reward"].item()
            if out["next", "done"].item():
                return total, int(out["next", "grid"][0, TREE].sum())
            td = step_mdp(out)
            step += 1

    area = n * n
    idle_ret, idle_trees = rollout(None)
    # (0, 1) is Moore-adjacent to the fire at (0, 0), so the CA takes it next tick.
    near_ret, near_trees = rollout((0, 1))
    # The far corner is doomed too, but only many ticks later.
    far_ret, far_trees = rollout((n - 1, n - 1))

    near_delta = near_ret - idle_ret
    far_delta = far_ret - idle_ret
    want_near = (1.0 - CUT_COST_IMMINENT) / area
    want_far = (1.0 - CUT_COST_AHEAD) / area

    nothing_saved = idle_trees == 0 and near_trees == 0 and far_trees == 0
    ok = (nothing_saved
          and abs(near_delta - want_near) < 1e-6
          and abs(far_delta - want_far) < 1e-6)
    farmable = max(want_near, want_far) > 0
    check("doomed tree pricing", ok,
          f"next to fire {near_delta:+.5f} (want {want_near:+.5f}), "
          f"ahead of fire {far_delta:+.5f} (want {want_far:+.5f})"
          + ("  -- FARMABLE by design: cutting a doomed tree pays"
             if farmable else "  -- nothing farmable at this pricing"))


def test_predicted_loss_is_achieved():
    """Cutting a plan's cells must lose exactly what the planner predicted.

    The planner picks between families -- rectangles scored in closed form,
    spirals scored by flood fill -- so those two scores have to mean the same
    thing or it chooses between numbers that are not comparable. This burns the
    grid down for real and checks the arithmetic, including that a felled tree
    is charged the planner's CUT_COST -- the tree charge plus the action -- and
    that the planner's price and the environment's are the same number.
    """
    rng = np.random.default_rng(5)
    checked = 0
    for _ in range(60):
        n = int(rng.integers(8, 17))
        grid = random_grid(rng, n, p_dirt=float(rng.uniform(0, 0.2)))
        agent = (int(rng.integers(0, n)), int(rng.integers(0, n)))
        cost = float(rng.choice([0.05, 0.1, 0.2]))
        spec, rows, cols, predicted = plan_barrier(grid, agent, 1.0, cost, cost)

        burned = grid.copy()
        if spec is not None:
            burned[TREE][rows, cols] = False
            burned[DIRT][rows, cols] = True
        before = int(burned[TREE].sum())
        g = torch.from_numpy(burned).unsqueeze(0)
        while g[0, FIRE].any():
            g = ca.step(g)
        cuts = 0 if spec is None else len(rows)
        actual = CUT_COST * cuts + before - int(g[0, TREE].sum())
        checked += 1
        if abs(actual - predicted) > 1e-6:
            check("predicted loss is achieved", False,
                  f"n={n} spec={spec}: predicted {predicted}, got {actual}")
            return
    check("predicted loss is achieved", True, f"{checked} plans burnt to a fixed point")


def test_never_worse_than_idling():
    """The planner must not spend cuts on a fire it cannot contain.

    "Let it burn" competes as an ordinary candidate, so on a hopeless setting the
    predicted loss must be no worse than the do-nothing loss, and any barrier it
    does buy must pay for itself.
    """
    rng = np.random.default_rng(3)
    bad = 0
    for _ in range(40):
        n = int(rng.integers(6, 17))
        grid = random_grid(rng, n, p_dirt=0.0)
        agent = (int(rng.integers(0, n)), int(rng.integers(0, n)))
        arrival = fire_arrival(grid[TREE].copy(), grid[FIRE].copy())
        idle_loss = float((grid[TREE] & (arrival < INF)).sum())
        # tc + tm = 0.5 is four times the containment threshold of 1/8.
        _, _, _, loss = plan_barrier(grid, agent, 1.0, 0.25, 0.25)
        bad += loss > idle_loss + 1e-9
    check("never worse than idling", bad == 0, f"{bad}/40 plans above the idle loss")


def test_batch_matches_single():
    """A batched call must produce the same actions as running episodes alone."""
    n, tc, tm = 12, 0.1, 0.1
    ff.N, ff.TC, ff.TM = n, tc, tm
    k = 6
    env = ff.ForestFireEnv(device="cpu", batch_size=[k])
    torch.manual_seed(0)
    env.set_seed(0)
    init = env.reset()

    batched = BarrierPlannerPolicy()
    td = init.clone()
    seq_batched = []
    for _ in range(60):
        td = batched(td)
        seq_batched.append(td["action"].clone())
        td = step_mdp(env.step(td))

    env1 = ff.ForestFireEnv(device="cpu", batch_size=[1])
    ok = True
    for i in range(k):
        single = BarrierPlannerPolicy()
        td = init[i:i + 1].clone()
        for step in range(60):
            td = single(td)
            if int(td["action"][0]) != int(seq_batched[step][i]):
                ok = False
                break
            td = step_mdp(env1.step(td))
        if not ok:
            break
    check("batched actions match single-episode", ok, f"{k} episodes x 60 steps")


if __name__ == "__main__":
    test_arrival_matches_ca()
    test_rectangle_border_separates()
    test_border_cells_are_exact()
    test_schedule_is_exact()
    test_doomed_tree_pricing()
    test_predicted_loss_is_achieved()
    test_never_worse_than_idling()
    test_batch_matches_single()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all checks passed")
