"""Render one episode per checkpoint as a GIF.

Run from the repo root:

    python -m steerca.render                       # every r10 policy in artifacts/models
    python -m steerca.render --ckpt <path> ...     # named ones
    python -m steerca.render --planner             # the baseline too
    python -m steerca.render --planted --oracle    # planted instances, with the oracle

A checkpoint whose name carries the `planted` tag is rendered on planted
instances automatically; --planted [CFG] forces it for any checkpoint.

Each GIF's caption carries the step, the reward, the running return and the tree
and fire counts, so the file is readable on its own without the run that made it.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import imageio
import numpy as np
import torch
from torchrl.envs import StepCounter, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

import steerca.firebreak.agents as ff_agents
import steerca.firebreak.env as ff
from steerca.evaluate import make_ppo_policy
from steerca.firebreak.ca import TREE
from steerca.firebreak.planner import BarrierPlannerPolicy
from steerca.firebreak.planted import PlantedConfig, PlantedOraclePolicy

CUT = 8


def settings_from_name(name):
    """Recover (n, tc, tm) from a checkpoint filename."""
    grid = re.search(r"_(\d+)x\d+", name)
    costs = re.search(r"_t([\d.]+)_([\d.]+)", name)
    if not (grid and costs):
        raise ValueError(f"cannot read grid size and costs from {name}")
    return int(grid.group(1)), float(costs.group(1)), float(costs.group(2))


def planted_from_name(name, override=None):
    """The planted config a checkpoint was trained on, or None for vanilla.

    Only the default tag can be read back; a customised tag needs --planted
    with the same config string it was trained with.
    """
    if override:
        return override
    tag = re.search(r"_(planted(?:-[^_]+)?)_r\d+", name)
    if not tag:
        return None
    known = {"planted": "1", "planted-graph": "family=graph"}
    cfg = known.get(tag.group(1))
    if cfg is None:
        raise SystemExit(f"{name} was trained on a customised planted config "
                         f"({tag.group(1)}); pass it with --planted <cfg>")
    return cfg


def rollout(policy, env, max_steps, frame_every, scale, title):
    """One episode, returning (frames, stats).

    Frames are 1100x1000 apiece, so a long episode is gigabytes if every step is
    kept. `frame_every` strides them -- the final frame is always kept, so the
    outcome is never the one dropped -- and `scale` subsamples pixels. Striding
    costs little: 1/tm actions fit in one CA tick, so consecutive steps usually
    differ only by where the bulldozer stands.
    """
    td = env.reset()
    n = int(td["params", "n"])
    trees_0 = int(td["grid"][TREE].sum())
    cum = torch.tensor(0.0)
    frames, cuts, steps, done = [], 0, 0, False

    def snap(source, reward=None):
        img = ff_agents.draw_forest_fire(
            source["grid"], source["agent_state"], source["t"],
            step_counter=steps, reward=reward, cum_reward=cum,
            format="img", title=title)
        frames.append(np.ascontiguousarray(img[::scale, ::scale]))

    snap(td)
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for _ in range(max_steps):
            td = policy(td)
            cuts += int(td["action"].reshape(-1)[0] == CUT)
            out = env.step(td)
            nxt = out["next"]
            steps += 1
            cum = cum + nxt["reward"].reshape(-1)[0].cpu()
            done = bool(nxt["done"].reshape(-1)[0])
            if steps % frame_every == 0 or done:
                snap(nxt, reward=nxt["reward"])
            if done:
                break
            td = step_mdp(out)

    trees_1 = int(nxt["grid"][TREE].sum())
    return frames, {"steps": steps, "cuts": cuts, "ret": float(cum),
                    "saved": trees_1 / (n * n), "trees": (trees_0, trees_1)}


def render(label, policy, n, tc, tm, out_path, seed, fps, scale, title, planted=None):
    ff.N, ff.TC, ff.TM = n, tc, tm
    ff.PLANTED_CFG = PlantedConfig.from_string(planted) if planted else None
    max_steps = ff.episode_cap(n, tc, tm)
    env = TransformedEnv(ff.ForestFireEnv(device="cpu"),
                         StepCounter(max_steps=max_steps))
    torch.manual_seed(seed)
    env.set_seed(seed)
    frame_every = max(1, max_steps // 100)

    frames, st = rollout(policy, env, max_steps, frame_every, scale, title)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Hold the final frame so the outcome is readable before the loop restarts.
    imageio.mimsave(out_path, frames + [frames[-1]] * 5, duration=1000 / fps, loop=0)
    size = os.path.getsize(out_path) / 1e6
    print(f"{label:<34} {100 * st['saved']:5.1f}% saved  {st['cuts']:3d} cuts  "
          f"{st['steps']:4d} steps  ->  {os.path.basename(out_path)} ({size:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="*", default=None,
                    help="checkpoints to render; default is every r10 policy in "
                         "$STEERCA_ARTIFACTS/models")
    ap.add_argument("--planner", action="store_true",
                    help="also render the planner on the same settings")
    ap.add_argument("--planted", nargs="?", const="1", default=None,
                    help="render on planted instances; optionally a generator config")
    ap.add_argument("--oracle", action="store_true",
                    help="also render the planted oracle (implies --planted)")
    ap.add_argument("--out", default="videos")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--scale", type=int, default=2, help="pixel subsampling factor")
    args = ap.parse_args()

    if args.oracle and not args.planted:
        args.planted = "1"
    models = os.path.join(os.environ.get("STEERCA_ARTIFACTS", "artifacts"), "models")
    ckpts = args.ckpt or sorted(glob.glob(os.path.join(models, "*_cut0.5_r10_policy.pth")))
    if not ckpts:
        raise SystemExit("no checkpoints matched")

    for path in ckpts:
        name = os.path.basename(path)[: -len("_policy.pth")]
        n, tc, tm = settings_from_name(name)
        planted = planted_from_name(name, args.planted)
        suffix = "_planted" if planted and "planted" not in name else ""
        where = "  planted" if planted else ""
        # The architecture comes from the checkpoint's own tag, so a policy
        # trained at one grid size can be rendered at the size it was trained at
        # without guessing.
        ff.N, ff.TC, ff.TM = n, tc, tm
        policy = make_ppo_policy(path, "equiv", torch.device("cpu"), n)
        render(f"ppo {n}x{n} tc=tm={tc:g}", policy, n, tc, tm,
               os.path.join(args.out, f"{name}{suffix}_episode.gif"),
               args.seed, args.fps, args.scale,
               f"PPO  {n}x{n}  tc=tm={tc:g}{where}", planted=planted)

        stem = f"{n}x{n}_t{tc:g}_{tm:g}{'_planted' if planted else ''}"
        if args.planner:
            render(f"planner {n}x{n} tc=tm={tc:g}", BarrierPlannerPolicy(), n, tc, tm,
                   os.path.join(args.out, f"planner_{stem}_episode.gif"),
                   args.seed, args.fps, args.scale,
                   f"Planner  {n}x{n}  tc=tm={tc:g}{where}", planted=planted)
        if args.oracle and planted:
            render(f"oracle {n}x{n} tc=tm={tc:g}", PlantedOraclePolicy(), n, tc, tm,
                   os.path.join(args.out, f"oracle_{stem}_episode.gif"),
                   args.seed, args.fps, args.scale,
                   f"Oracle  {n}x{n}  tc=tm={tc:g}{where}", planted=planted)


if __name__ == "__main__":
    main()
