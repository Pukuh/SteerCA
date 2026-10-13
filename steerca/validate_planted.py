"""Certify a planted-instance distribution the expensive way.

The generator certifies each instance with numpy at reset -- separation, min
vertex cut, the exact schedule. This script runs the two checks that are too
slow for a reset on a *sample* of instances per (n, tc, tm) setting:

  * the oracle, played in the real environment, saves exactly `oracle_saved`
    and makes exactly k cuts;
  * the planner (`BarrierPlannerPolicy`), given the same instances, never saves
    more than the oracle -- the check that would catch a barrier family beating
    the planted cut, e.g. an L-arc against the grid edge at low action cost.

It also reports generation time, the rejection histogram, and (with --static)
how often the planted cut is provably optimal for an agent of any speed.

    python -m steerca.validate_planted --n 16 --costs 0.06,0.06 0.12,0.12 0.2,0.2 0.3,0.3 \
        --count 200 --planner-episodes 50
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

import steerca.firebreak.env as ff
from steerca.evaluate import make_env, run
from steerca.firebreak.planner import BarrierPlannerPolicy
from steerca.firebreak.planted import (
    PlantedConfig,
    PlantedOraclePolicy,
    generate_instance,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--costs", nargs="+", default=["0.2,0.2"])
    ap.add_argument("--count", type=int, default=100, help="instances per setting")
    ap.add_argument("--planner-episodes", type=int, default=20,
                    help="how many of them the planner is scored on (slow)")
    ap.add_argument("--cfg", default="1", help="STEERCA_PLANTED-style config string")
    ap.add_argument("--static", action="store_true",
                    help="also compute the static optimality bound per instance")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = PlantedConfig.from_string(args.cfg)
    device = torch.device(args.device)
    print(f"grid {args.n}x{args.n}   config {cfg.tag()}   {args.count} instances per setting\n")

    for cost in args.costs:
        tc, tm = (float(x) for x in cost.split(","))
        rng = np.random.default_rng(args.seed)

        # --- generation ------------------------------------------------------
        t0 = time.perf_counter()
        insts = [generate_instance(args.n, tc, tm, rng, cfg, want_static=args.static)
                 for _ in range(args.count)]
        gen_ms = 1000 * (time.perf_counter() - t0) / args.count
        rejections = Counter()
        attempts = 0
        for inst in insts:
            rejections.update(inst.stats["rejections"])
            attempts += inst.stats["attempts"]
        ks = Counter(inst.k for inst in insts)
        oracle = np.array([inst.oracle_saved for inst in insts])
        noop = np.array([inst.noop_saved for inst in insts])
        print(f"=== tc={tc:g} tm={tm:g} ===")
        print(f"  generation {gen_ms:.1f} ms/instance, {attempts / args.count:.1f} attempts each; "
              f"rejections {dict(rejections)}")
        print(f"  k {dict(sorted(ks.items()))}   d_f mean {np.mean([i.stats['d_f'] for i in insts]):.1f}   "
              f"noise {np.mean([i.stats['p_noise'] for i in insts]):.3f}")
        print(f"  oracle saves {100 * oracle.mean():.1f}% (min {100 * oracle.min():.1f}, "
              f"max {100 * oracle.max():.1f}); doing nothing {100 * noop.mean():.1f}%")
        if args.static:
            gaps = np.array([inst.static_loss_gap for inst in insts])
            print(f"  statically optimal: {int((gaps == 0).sum())}/{args.count} "
                  f"(mean gap {gaps.mean():.1f} trees)")

        # --- the same instances through the real env -------------------------
        env = make_env(args.n, tc, tm, args.count, device, planted=args.cfg)
        torch.manual_seed(args.seed)
        env.set_seed(args.seed)
        init = env.reset()
        max_steps = ff.episode_cap(args.n, tc, tm)
        expect = init["params", "oracle_saved"].cpu()
        ks_env = (init["params", "gap"][..., 0] >= 0).sum(-1).float().cpu()

        m = run(env, init, PlantedOraclePolicy(), max_steps, args.n)
        mism = int((m["per_episode"] - expect).abs().gt(1e-6).sum())
        print(f"  oracle in env: {100 * m['saved']:.1f}% saved, {m['cuts']:.2f} cuts "
              f"(k mean {ks_env.mean():.2f}), {mism} mismatches with the certificate, "
              f"{m['truncated']} truncated")

        p = min(args.planner_episodes, args.count)
        if p > 0:
            t0 = time.perf_counter()
            env_p = make_env(args.n, tc, tm, p, device, planted=args.cfg)
            mp = run(env_p, init[:p], BarrierPlannerPolicy(), max_steps, args.n)
            dt = time.perf_counter() - t0
            diff = mp["per_episode"] - expect[:p]
            exceed = int((diff > 2.0 / (args.n * args.n)).sum())
            print(f"  planner on {p}: {100 * mp['saved']:.1f}% saved vs oracle "
                  f"{100 * expect[:p].mean():.1f}%; planner - oracle {100 * diff.mean():+.2f} pts, "
                  f"beats the oracle on {exceed}/{p} (worst {100 * diff.max():+.1f}), {dt:.0f} s")
        print()


if __name__ == "__main__":
    main()
