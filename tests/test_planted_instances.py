"""Checks on the planted-instance generator: that the certificate it issues is
true in the real environment, that hiding changes nothing it measures, and that
the env wiring round-trips.

Run from the repo root:  python tests/test_planted_instances.py
"""
import itertools
import os
import sys
import time

import numpy as np
import torch
from torchrl.envs.utils import check_env_specs, step_mdp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import steerca.firebreak.env as ff
from steerca.evaluate import run
from steerca.firebreak.ca import DIRT, FIRE, TREE
from steerca.firebreak.planner import INF, BarrierPlannerPolicy, _dilate, _schedule, fire_arrival
from steerca.firebreak.planted import (
    K_MAX,
    PlantedConfig,
    PlantedOraclePolicy,
    certify_instance,
    d4_apply,
    generate_instance,
    min_vertex_cut,
    static_min_loss,
)

FAILURES = []
SETTINGS = [(n, c, c) for n in (8, 16, 32) for c in (0.06, 0.12, 0.2, 0.3)]


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def sample(rng, per_setting=3, settings=SETTINGS):
    for n, tc, tm in settings:
        for _ in range(per_setting):
            yield n, tc, tm, generate_instance(n, tc, tm, rng)


def planted_env(n, tc, tm, batch, seed, cfg=None):
    ff.N, ff.TC, ff.TM = n, tc, tm
    ff.PLANTED_CFG = cfg or PlantedConfig()
    env = ff.ForestFireEnv(device="cpu", batch_size=[batch] if batch else None)
    torch.manual_seed(seed)
    env.set_seed(seed)
    return env


class IdlePolicy:
    def __call__(self, td):
        td.set("action", torch.zeros(td["grid"].shape[:-3], dtype=torch.int64))
        return td


# --------------------------------------------------------------------------- #
def test_flow_solver():
    """The max-flow must agree with what can be read off by hand."""
    n = 10
    ok = True
    for w in (1, 2, 3):
        tree = np.ones((n, n), dtype=bool)
        tree[:, 4:6] = False
        tree[3:3 + w, 4:6] = True          # a corridor of width w
        fire = np.zeros((n, n), dtype=bool)
        fire[5, 0] = True
        tree[5, 0] = False
        sink = np.zeros((n, n), dtype=bool)
        sink[:, 7:] = True
        value, cut = min_vertex_cut(tree, fire, sink)
        remaining = tree & ~cut
        leak = (fire_arrival(remaining, fire) < INF)[sink].any()
        ok &= value == w and cut.sum() == w and not leak
    tree = np.ones((n, n), dtype=bool)
    tree[:, 5] = False
    tree[2, 5] = True
    tree[7, 5] = True                       # two holes in a straight wall
    fire = np.zeros((n, n), dtype=bool); fire[5, 0] = True; tree[5, 0] = False
    sink = np.zeros((n, n), dtype=bool); sink[:, 8:] = True
    ok &= min_vertex_cut(tree, fire, sink)[0] == 2
    tree[7, 5] = False
    ok &= min_vertex_cut(tree, fire, sink)[0] == 1
    tree[:, 5] = False                      # sealed
    ok &= min_vertex_cut(tree, fire, sink)[0] == 0
    check("min vertex cut on hand-built grids", ok)

    # static_min_loss against brute force over every cut set of a 3x4 grid
    rng = np.random.default_rng(0)
    worst = 0
    for _ in range(20):
        tree = rng.random((3, 4)) < 0.8
        fire = np.zeros((3, 4), dtype=bool)
        fire[1, 0] = True
        tree[1, 0] = False
        cells = np.argwhere(tree)
        best = INF
        for m in range(len(cells) + 1):
            for sub in itertools.combinations(range(len(cells)), m):
                keep = tree.copy()
                for j in sub:
                    keep[tuple(cells[j])] = False
                burnt = int((keep & (fire_arrival(keep, fire) < INF)).sum())
                best = min(best, m + burnt)
        worst = max(worst, abs(best - static_min_loss(tree, fire)))
    check("static min loss matches brute force", worst == 0, f"max error {worst}")


def test_wall_separates():
    """With the gap standing the fire reaches it; with it cut, nothing protected."""
    rng = np.random.default_rng(1)
    bad = 0
    total = 0
    for n, tc, tm, inst in sample(rng):
        total += 1
        tree, fire = inst.grid[TREE], inst.grid[FIRE]
        gap = inst.gap
        open_arrival = fire_arrival(tree, fire)
        cut = tree.copy()
        cut[gap[:, 0], gap[:, 1]] = False
        closed_arrival = fire_arrival(cut, fire)
        reaches_gap = (open_arrival[gap[:, 0], gap[:, 1]] < INF).all()
        protected_burns = (closed_arrival < INF)[inst.protected].any()
        protected_doomed = (open_arrival < INF)[inst.protected].all()
        if not (reaches_gap and not protected_burns and protected_doomed):
            bad += 1
    check("wall separates iff the gap stands", bad == 0, f"{bad}/{total} instances wrong")


def test_min_cut_is_the_gap():
    rng = np.random.default_rng(2)
    bad = 0
    total = 0
    for n, tc, tm, inst in sample(rng, per_setting=2):
        total += 1
        tree, fire = inst.grid[TREE], inst.grid[FIRE]
        value, cut = min_vertex_cut(tree, fire, inst.protected)
        remaining = tree & ~cut
        leak = (fire_arrival(remaining, fire) < INF)[inst.protected].any()
        if value != inst.k or cut.sum() != inst.k or leak or inst.min_cut != inst.k:
            bad += 1
    check("min vertex cut equals k and protects", bad == 0, f"{bad}/{total} wrong")


def test_schedule_feasible():
    """The certified order is feasible under the planner's exact clock, and the
    fire is placed no closer than the schedule bound."""
    rng = np.random.default_rng(3)
    bad = 0
    total = 0
    for n, tc, tm, inst in sample(rng):
        total += 1
        arrival = fire_arrival(inst.grid[TREE], inst.grid[FIRE])
        cfg = PlantedConfig().resolve(n, tc, tm)
        finish = _schedule(inst.gap[:, 0], inst.gap[:, 1], inst.agent, 1.0, tc, tm,
                           arrival, cfg.margin)
        d_min = int(np.floor(inst.k * tc + (inst.k - 1) * tm)) + cfg.margin + 1
        r_max, e_max = PlantedConfig.pocket_limits(tc + tm)
        R = inst.stats["R"]
        geometry = R <= r_max and inst.stats["offset"] <= e_max(R)
        if finish is None or inst.stats["d_f"] < d_min or not geometry:
            bad += 1
    check("oracle schedule feasible with margin; pocket within the timing bounds",
          bad == 0, f"{bad}/{total} wrong")


def _roll(env, init_td, policy, max_steps):
    n = init_td["grid"].shape[-1]
    return run(env, init_td, policy, max_steps, n)


def test_oracle_saves_in_env():
    """The oracle, run for real, saves exactly what the certificate says."""
    bad = []
    for cost in (0.06, 0.12, 0.2, 0.3):
        n, k = 16, 32
        env = planted_env(n, cost, cost, k, seed=4)
        init = env.reset()
        m = _roll(env, init, PlantedOraclePolicy(), ff.episode_cap(n, cost, cost))
        expect = init["params", "oracle_saved"]
        ks = (init["params", "gap"][..., 0] >= 0).sum(-1).float()
        exact = torch.allclose(m["per_episode"], expect, atol=1e-6)
        cuts_ok = m["cuts"] == ks.mean().item()
        if not (exact and cuts_ok and m["truncated"] == 0):
            bad.append((cost, (m["per_episode"] - expect).abs().max().item(), m["cuts"], ks.mean().item()))
    check("oracle in the real env saves oracle_saved exactly", not bad, str(bad) if bad else "4 costs x 32 episodes")


def test_noop_burns_fire_side():
    bad = []
    for cost in (0.12, 0.3):
        n, k = 16, 16
        env = planted_env(n, cost, cost, k, seed=5)
        init = env.reset()
        m = _roll(env, init, IdlePolicy(), ff.episode_cap(n, cost, cost))
        noop, oracle = init["params", "noop_saved"], init["params", "oracle_saved"]
        exact = torch.allclose(m["per_episode"], noop, atol=1e-6)
        stake = ((oracle - noop) >= PlantedConfig().protected_min_frac - 1e-6).all()
        if not (exact and stake):
            bad.append(cost)
    check("doing nothing saves noop_saved, the oracle saves the protected side on top",
          not bad, str(bad) if bad else "")


def test_planner_not_better_than_oracle():
    """The planner's own barriers must not beat the planted cut. Slow (~30 s).

    A distribution-level bound rather than per-episode exactness: a staircase
    cut across a pocket corner can pick up a few trees on the odd instance,
    and validate_planted.py reports the raw numbers. Measured at 16x16: the
    planner's mean edge is under 0.5 pts at every cost and it never took more
    than 2 pts on an instance.
    """
    worst, diffs = -1.0, []
    for cost in (0.12, 0.2):
        n, k = 16, 12
        env = planted_env(n, cost, cost, k, seed=6)
        init = env.reset()
        planner = _roll(env, init, BarrierPlannerPolicy(), ff.episode_cap(n, cost, cost))
        oracle = init["params", "oracle_saved"]
        diff = 100 * (planner["per_episode"] - oracle)
        worst = max(worst, diff.max().item())
        diffs.append(diff)
    mean = torch.cat(diffs).mean().item()
    check("planner does not beat the oracle", mean <= 1.0 and worst <= 3.0,
          f"planner - oracle {mean:+.2f} pts on average, worst +{worst:.1f} pts")


def test_hiding_preserves_certificate():
    rng = np.random.default_rng(7)
    bad, noisy, decoyed, total = 0, 0, 0, 0
    for n, tc, tm, inst in sample(rng, per_setting=2):
        total += 1
        noisy += inst.stats["p_noise"] > 0
        decoyed += (inst.stats["decoy_walls"] + inst.stats["decoy_loops"]
                    + inst.stats["decoy_gaps"]) > 0
        grid = inst.grid
        fire = grid[FIRE]
        centre = (n - 1) / 2
        in_window = (np.maximum(np.abs(inst.gap[:, 0] - centre),
                                np.abs(inst.gap[:, 1] - centre))
                     <= max(0.25 * n, inst.stats["R"] + 2)).all()
        fire_clean = not (_dilate(fire) & grid[DIRT]).any()
        gap_tree = grid[TREE][inst.gap[:, 0], inst.gap[:, 1]].all()
        same = True
        for g in range(8):
            (gt, pt), (gap_t, agent_t), _ = d4_apply(g, n, grids=[grid, inst.protected],
                                                     points=[inst.gap, np.array(inst.agent)])
            twin = type(inst)(grid=gt, agent=(int(agent_t[0]), int(agent_t[1])), fire=inst.fire,
                              gap=gap_t, protected=pt, fire_side=inst.fire_side,
                              oracle_saved=0, noop_saved=0, min_cut=0, static_loss_gap=None, stats={})
            fields_, reason = certify_instance(twin, tc, tm)
            if fields_ is None or fields_["min_cut"] != inst.k or \
                    fields_["protected"].sum() != inst.protected.sum() or \
                    abs(fields_["oracle_saved"] - inst.oracle_saved) > 1e-9 or \
                    not np.array_equal(fields_["protected"], pt):
                same = False
        if not (in_window and fire_clean and gap_tree and same):
            bad += 1
    check("noise, decoys and D4 leave the certificate unchanged", bad == 0,
          f"{bad}/{total} wrong; {noisy}/{total} noisy, {decoyed}/{total} with decoys")


def test_static_optimality_rate():
    """How often the planted cut is provably optimal at ANY agent speed."""
    rng = np.random.default_rng(8)
    hits, total = 0, 0
    for n, tc, tm in [(8, 0.2, 0.2), (16, 0.2, 0.2), (16, 0.06, 0.06)]:
        for _ in range(4):
            inst = generate_instance(n, tc, tm, rng, want_static=True)
            total += 1
            hits += inst.static_loss_gap == 0
    check("static optimum reported", True, f"{hits}/{total} instances statically optimal")


def test_reproducible_and_diverse():
    rng_a, rng_b = np.random.default_rng(9), np.random.default_rng(9)
    a = generate_instance(16, 0.2, 0.2, rng_a)
    b = generate_instance(16, 0.2, 0.2, rng_b)
    same = np.array_equal(a.grid, b.grid) and a.agent == b.agent and np.array_equal(a.gap, b.gap)
    env = planted_env(16, 0.2, 0.2, 16, seed=10)
    x = env.reset()
    torch.manual_seed(10); env.set_seed(10)
    y = env.reset()
    distinct = len({x["grid"][i].numpy().tobytes() for i in range(16)})
    check("same seed, same instance; batch slots all differ",
          same and torch.equal(x["grid"], y["grid"]) and distinct == 16,
          f"{distinct}/16 distinct")


def test_env_specs_and_partial_reset():
    env = planted_env(16, 0.2, 0.2, 4, seed=11)
    check_env_specs(env)
    check_env_specs(planted_env(16, 0.2, 0.2, None, seed=11))
    a = env.reset()
    mask = torch.tensor([True, False, False, True])
    c = env.reset(a.clone().set("_reset", mask))
    changed = [not torch.equal(a["grid"][i], c["grid"][i]) for i in range(4)]
    shape_ok = tuple(a["params", "gap"].shape) == (4, K_MAX, 2)
    # Fresh frames of an auto-resetting rollout carry a standing gap.
    roll = env.rollout(300, PlantedOraclePolicy(), break_when_any_done=False)
    ok = True
    for i in range(4):
        for j in range(roll.shape[1]):
            g, t = roll["grid"][i, j].numpy(), float(roll["t"][i, j])
            fire = g[FIRE]
            fresh = fire.sum() == 1 and t >= 1.0 and not (_dilate(fire) & g[DIRT]).any()
            if fresh:
                gap = roll["params", "gap"][i, j].numpy()
                gap = gap[gap[:, 0] >= 0]
                ok &= bool(g[TREE][gap[:, 0], gap[:, 1]].all())
    check("specs pass; partial reset touches only masked slots; params follow auto-resets",
          changed == [True, False, False, True] and shape_ok and ok, f"changed={changed}")


def test_planner_fresh_detection():
    """The planner must see a planted reset as a new episode, and nothing else."""
    env = planted_env(16, 0.2, 0.2, 1, seed=12)
    pol = BarrierPlannerPolicy()
    td = env.reset()
    fire0 = tuple(int(v) for v in np.argwhere(td["grid"][0, FIRE].numpy())[0])
    td = pol(td)
    origin0 = pol._state[0]["origin"]
    st0 = pol._state[0]
    # Play the episode to its end through the planner, then auto-reset.
    for _ in range(200):
        out = env.step(td)
        if bool(out["next", "done"].reshape(-1)[0]):
            break
        td = pol(step_mdp(out))
        stale = pol._state[0] is not st0
        if stale:
            break
    never_stale = pol._state[0] is st0
    td = env.reset()
    fire1 = tuple(int(v) for v in np.argwhere(td["grid"][0, FIRE].numpy())[0])
    pol(td)
    origin1 = pol._state[0]["origin"]
    # The rule is that dirt never shrinks inside an episode. Adding dirt, which
    # is what burning and cutting do, must NOT read as a new episode; taking
    # dirt away, which nothing can do, must.
    g = td["grid"][0].numpy().copy()
    burnt = g.copy()
    spot = np.argwhere(burnt[TREE])[0]
    burnt[TREE, spot[0], spot[1]] = False
    burnt[DIRT, spot[0], spot[1]] = True
    st = pol._episode_state(0, burnt, 1.0)
    not_fresh = st["origin"] == origin1
    grown = burnt.copy()
    back = np.argwhere(grown[DIRT])[0]
    grown[DIRT, back[0], back[1]] = False
    grown[TREE, back[0], back[1]] = True
    pol._episode_state(0, grown, 1.0)
    saw_reset = pol._state[0]["rows"] is None
    check("planner reset detection by dirt shrinking",
          origin0 == fire0 and never_stale and origin1 == fire1
          and not_fresh and saw_reset,
          f"origins {origin0}->{origin1}, fires {fire0}->{fire1}, "
          f"more dirt stays in-episode={not_fresh}, less dirt resets={saw_reset}")


# --------------------------------------------------------------------------- #
# The graph family
# --------------------------------------------------------------------------- #
GRAPH_CFG = PlantedConfig.from_string("family=graph")
GRAPH_SETTINGS = [(32, c, c) for c in (0.2, 0.3)]


def graph_sample(rng, per_setting=2):
    for n, tc, tm in GRAPH_SETTINGS:
        for _ in range(per_setting):
            yield n, tc, tm, generate_instance(n, tc, tm, rng, GRAPH_CFG)


def test_graph_cut_sites():
    """Only a straight interior cell severs a corridor; a bend does not.

    The fire steps diagonally around a cut corner, so this is the fact the whole
    geometry rests on. Checked against a full flood fill rather than the local
    one the generator uses.
    """
    n = 16
    tree = np.zeros((n, n), bool)
    tree[3, 2:8] = True          # run east
    tree[3:9, 7] = True          # then south: (3,7) is the bend
    fire = np.zeros((n, n), bool)
    fire[3, 2] = True
    tree[3, 2] = False
    beyond = np.zeros((n, n), bool)
    beyond[6:9, 7] = True
    bend_blocks = not (beyond & (fire_arrival(_without(tree, (3, 7)), fire) < INF)).any()
    straight_blocks = not (beyond & (fire_arrival(_without(tree, (3, 5)), fire) < INF)).any()
    check("a bend does not sever, a straight cell does",
          (not bend_blocks) and straight_blocks,
          f"cutting the bend blocks={bend_blocks}, cutting a straight cell blocks={straight_blocks}")


def _without(tree, rc):
    out = tree.copy()
    out[rc] = False
    return out


def test_graph_value_matches_grid():
    """The contracted value must equal the grid flood fill, or the search is
    optimising a different problem from the one the environment runs."""
    rng = np.random.default_rng(20)
    bad = total = 0
    for n, tc, tm, inst in graph_sample(rng):
        tree, fire = inst.grid[TREE], inst.grid[FIRE]
        cut = tree.copy()
        cut[inst.gap[:, 0], inst.gap[:, 1]] = False
        grid_val = int((cut & (fire_arrival(cut, fire) >= INF)).sum())
        total += 1
        bad += grid_val != inst.stats["saved_trees"]
    check("graph value equals the grid flood fill", bad == 0, f"{bad}/{total} wrong")


def test_graph_oracle_saves_in_env():
    """The oracle, run for real, saves exactly what the certificate claims."""
    bad = []
    for n, cost, _ in GRAPH_SETTINGS:
        env = planted_env(n, cost, cost, 12, seed=21, cfg=GRAPH_CFG)
        init = env.reset()
        m = _roll(env, init, PlantedOraclePolicy(), ff.episode_cap(n, cost, cost))
        expect = init["params", "oracle_saved"]
        ks = (init["params", "gap"][..., 0] >= 0).sum(-1).float()
        if not (torch.allclose(m["per_episode"], expect, atol=1e-6)
                and abs(m["cuts"] - ks.mean().item()) < 1e-6 and m["truncated"] == 0):
            bad.append((cost, (m["per_episode"] - expect).abs().max().item(),
                        m["cuts"], m["truncated"]))
    check("graph oracle saves oracle_saved exactly in the env", not bad,
          str(bad) if bad else f"{len(GRAPH_SETTINGS)} settings x 12 episodes")


def test_graph_structure():
    """Separation, cycles, cut budget, and the fire's degree."""
    rng = np.random.default_rng(22)
    bad = []
    for n, tc, tm, inst in graph_sample(rng):
        st = inst.stats
        if not (3 <= inst.k <= GRAPH_CFG.cuts[1]):
            bad.append(("cuts", inst.k))
        if st["cycles"] < 1:
            bad.append(("acyclic", st["cycles"]))
        if st["fire_degree"] < 4:
            bad.append(("fire degree", st["fire_degree"]))
        if inst.min_cut != inst.k:
            bad.append(("min cut", inst.min_cut, inst.k))
        if st["rich_payoff"] <= st["saved_trees"]:
            bad.append(("no trap", st["rich_payoff"], st["saved_trees"]))
        if st["idle_ticks"] + 2 > GRAPH_CFG.cap_factor() * n:
            bad.append(("slow burn", st["idle_ticks"]))
    check("graph instances are cyclic, 3+ cuts, minimal, trapped and quick",
          not bad, str(bad[:4]) if bad else "")


def test_graph_junction_is_not_a_cut_vertex():
    """Cutting a junction cell must not disconnect its corridors.

    This is what makes corridor cuts the entire cut-set family, and so what
    makes the region search complete. If a junction ever became a cut vertex
    the certificate would be missing a family.
    """
    rng = np.random.default_rng(23)
    bad = total = 0
    for n, tc, tm, inst in graph_sample(rng, per_setting=1):
        tree, fire = inst.grid[TREE], inst.grid[FIRE]
        arrival = fire_arrival(tree, fire)
        base = int((tree & (arrival >= INF)).sum())
        # every tree cell with 3+ tree neighbours is a junction
        for r, c in np.argwhere(tree):
            win = tree[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2]
            if int(win.sum()) - 1 < 3:
                continue
            cut = _without(tree, (r, c))
            saved = int((cut & (fire_arrival(cut, fire) >= INF)).sum())
            total += 1
            bad += saved > base
    check("cutting a junction saves nothing", bad == 0,
          f"{bad}/{total} junction cuts protected something")


def test_oracle_batched_matches_single():
    env = planted_env(16, 0.2, 0.2, 5, seed=13)
    init = env.reset()
    pol = PlantedOraclePolicy()
    batched = pol(init.clone())["action"]
    single = torch.stack([pol(init[i].clone())["action"] for i in range(5)])
    check("oracle batched matches single", torch.equal(batched, single))


def test_config_parsing():
    c = PlantedConfig.from_string("k=2-4,noise=0.05-0.1,m=2,R=2-3")
    ok = (c.k_min, c.k_max) == (2, 4) and c.noise == (0.05, 0.1) and c.margin == 2
    ok &= c.tag() == "planted-R2-3-k2-4-m2-noise0.05-0.1"
    ok &= c.pocket == (2, 3)
    ok &= PlantedConfig.from_string("1").tag() == "planted"
    ok &= PlantedConfig.from_string("decoys=0").decoy_walls == (0, 0)
    r_max, e_max = PlantedConfig.pocket_limits(0.4)
    ok &= r_max == 11 and e_max(3) == 1 and e_max(2) == 1
    ok &= PlantedConfig.pocket_limits(0.12)[0] == 2
    ok &= PlantedConfig.pocket_limits(0.6)[0] > 100
    try:
        PlantedConfig.from_string("k=3-9")
        ok = False
    except ValueError:
        pass
    check("config parsing and tags", ok, c.tag())


def test_generation_speed():
    rng = np.random.default_rng(14)
    out = []
    slow = False
    for n, cost in ((16, 0.2), (32, 0.2)):
        t0 = time.perf_counter()
        for _ in range(10):
            generate_instance(n, cost, cost, rng)
        ms = 1000 * (time.perf_counter() - t0) / 10
        out.append(f"n={n}: {ms:.1f} ms")
        slow |= ms > 500
    check("generation speed", not slow, ", ".join(out))


if __name__ == "__main__":
    test_flow_solver()
    test_wall_separates()
    test_min_cut_is_the_gap()
    test_schedule_feasible()
    test_oracle_saves_in_env()
    test_noop_burns_fire_side()
    test_hiding_preserves_certificate()
    test_static_optimality_rate()
    test_reproducible_and_diverse()
    test_env_specs_and_partial_reset()
    test_planner_fresh_detection()
    test_oracle_batched_matches_single()
    test_config_parsing()
    test_generation_speed()
    test_graph_cut_sites()
    test_graph_value_matches_grid()
    test_graph_oracle_saves_in_env()
    test_graph_structure()
    test_graph_junction_is_not_a_cut_vertex()
    test_planner_not_better_than_oracle()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all checks passed")
