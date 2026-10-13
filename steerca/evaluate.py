"""Score the hand-designed planner and a PPO checkpoint on identical episodes.

Both agents are run from the *same* initial states (same ignition cell, same
starting position) at each (tc, tm) setting, so the comparison is paired and the
large episode-to-episode variance from ignition luck cancels out.

    python -m steerca.evaluate --n 16 --episodes 100 --costs 0.2,0.2 \
        --ckpt artifacts/models/ppo_equiv_c96_b6_512_64_1hot_16x16_t0.2_0.2_cut0.5_r10_policy.pth

Omit --ckpt to score the planner alone.

    --planted            score on planted instances too, adding the oracle that
                         cuts the planted gap; --planted "k=2-4,noise=0.05-0.1"
                         passes a generator configuration (see steerca.firebreak.planted)
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import torch
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
from torchrl.modules import ProbabilisticActor

import steerca.firebreak.env as ff
from steerca.firebreak.ca import FIRE, TREE
from steerca.archs import (
    ForestFireCommonNetCNN,
    ForestFireEquivariantNet,
    ForestFirePolicyNet,
    arch_for_grid,
    parse_arch_tag,
)
from steerca.firebreak.planner import BarrierPlannerPolicy
from steerca.firebreak.planted import PlantedConfig, PlantedOraclePolicy
from steerca.firebreak.env import make_env  # noqa: F401  (re-exported)

CUT = 8


def make_ppo_policy(ckpt, arch, device, n, in_states=3):
    """Build the actor a checkpoint was trained as, and load it.

    Works for PPO, A2C and SAC checkpoints alike (the actor is the same
    network). ``in_states`` is the number of grid planes: 3 for Firebreak, the
    spec's ``n_states`` for a custom CA.

    The architecture comes from the checkpoint's own name, not from `n`. Those
    differ whenever a policy is evaluated at a grid size it was not trained at --
    the whole point of the equivariant trunk -- and building from `n` would then
    construct a net the checkpoint cannot load into. `arch_for_grid` is only the
    fallback for a name that predates the tag.
    """
    if arch == "equiv":
        cfg = parse_arch_tag(os.path.basename(ckpt)) or arch_for_grid(n)
        backbone = ForestFireEquivariantNet(grid_size=n, device=device, **cfg,
                                            in_states=in_states)
        hidden = cfg["hidden"]
    else:
        backbone = ForestFireCommonNetCNN(grid_size=n, filters_1=256, filters_2=256,
                                          hidden=512, tdim=64, device=device)
        hidden = 512
    net = ForestFirePolicyNet(backbone_net=backbone, n_actions=9, hidden=hidden,
                              device=device)
    actor = ProbabilisticActor(
        TensorDictModule(net, in_keys=["grid", "t", "agent_state"], out_keys=["logits"]),
        distribution_class=Categorical, in_keys=["logits"], out_keys=["action"],
        return_log_prob=False,
    )
    actor.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
    actor.eval()
    return actor


def run(env, init_td, policy, max_steps, n, track_edge=False):
    """Roll every episode in parallel, freezing each one's stats when it ends.

    Episodes in a batch finish at different times. Reward, cut count and the
    final tree count are all masked by ``alive`` so a finished episode's numbers
    stop moving -- otherwise the agent keeps acting in a terminal state and
    accrues reward that belongs to no episode.

    With ``track_edge``, also report ``edge_free``: the fraction of episodes the
    fire burned out without ever reaching the grid edge, i.e. was enclosed by a
    barrier that does not use the edge. A fire cell lives one CA update, so this
    is exact only when every action spans at most one update (t_c, t_m < 1);
    the caller decides.
    """
    k = init_td.shape[0]
    td = init_td.clone()
    dev = td.device
    total = torch.zeros(k, device=dev)
    cuts = torch.zeros(k, device=dev)
    steps = torch.zeros(k, device=dev)
    alive = torch.ones(k, dtype=torch.bool, device=dev)
    trees = td["grid"][:, TREE].sum((-2, -1)).float()
    edge = torch.zeros(n, n, dtype=torch.bool, device=dev)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    touched = (td["grid"][:, FIRE] & edge).flatten(1).any(1)

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for _ in range(max_steps):
            td = policy(td)
            cuts += ((td["action"].reshape(k) == CUT) & alive).float()
            steps += alive.float()
            out = env.step(td)
            nxt = out["next"]
            total += nxt["reward"].reshape(k) * alive
            now = nxt["grid"][:, TREE].sum((-2, -1)).float()
            trees = torch.where(alive, now, trees)
            touched |= (nxt["grid"][:, FIRE] & edge).flatten(1).any(1) & alive
            alive &= ~nxt["done"].reshape(k)
            if not alive.any():
                break
            td = step_mdp(out)

    per_episode = (trees / (n * n)).cpu()
    out = {
        "per_episode": per_episode,
        "saved": per_episode.mean().item(),
        "saved_std": per_episode.std().item(),
        "ret": total.mean().item(),
        "cuts": cuts.mean().item(),
        "steps": steps.mean().item(),
        "truncated": int(alive.sum().item()),
    }
    if track_edge:
        # An episode still burning at the cap was not enclosed.
        out["edge_free"] = (~touched & ~alive).float().mean().item()
    if ("params", "oracle_saved") in init_td.keys(include_nested=True):
        # Planted instances know their optimum: report the share of the
        # oracle's trees each episode kept. Above 1 is possible and honest --
        # the oracle only cuts the gap, a policy may also save fire-side trees.
        oracle = init_td["params", "oracle_saved"].reshape(k).float().cpu()
        out["oracle_frac"] = (per_episode / oracle.clamp_min(1e-6)).mean().item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--costs", nargs="+", default=["0.05,0.05"],
                    help='"tc,tm" pairs, e.g. 0.05,0.05 0.1,0.1 0.2,0.2')
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--arch", default="equiv", choices=["equiv", "cnn"])
    ap.add_argument("--planted", nargs="?", const="1", default=None,
                    help="score on planted instances; optionally a generator config")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--csv", default=None,
                    help="append one tidy row per (n, tc, tm, agent) to this file, "
                         "creating it with a header if it does not exist. The paired "
                         "margin against the reference agent rides on each row, so a "
                         "row is self-contained.")
    args = ap.parse_args()

    device = torch.device(args.device)
    costs = [tuple(float(x) for x in c.split(",")) for c in args.costs]

    print(f"grid {args.n}x{args.n}   episodes {args.episodes}   seed {args.seed}   "
          f"device {device}")
    if args.ckpt:
        print(f"ppo checkpoint: {args.ckpt} ({args.arch})")
    if args.planted:
        print(f"planted instances: {PlantedConfig.from_string(args.planted).tag()}")
    print()
    oracle_col = f" {'of oracle':>10}" if args.planted else ""
    print(f"{'tc':>6} {'tm':>6} {'agent':>10} {'trees saved':>16} {'return':>9} "
          f"{'cuts':>7} {'steps':>8} {'trunc':>6}{oracle_col}")
    print("-" * (76 + len(oracle_col)))

    rows = []
    for tc, tm in costs:
        env = make_env(args.n, tc, tm, args.episodes, device, planted=args.planted)
        # Same seed, same reset -> both agents see identical episodes.
        torch.manual_seed(args.seed)
        env.set_seed(args.seed)
        init_td = env.reset()
        max_steps = ff.episode_cap(args.n, tc, tm)

        agents = [("planner", BarrierPlannerPolicy())]
        if args.planted:
            agents.append(("oracle", PlantedOraclePolicy()))
        if args.ckpt:
            agents.append(("ppo", make_ppo_policy(args.ckpt, args.arch, device, args.n)))

        scored = {}
        for label, policy in agents:
            # Edge closure is exact only when no action spans two CA updates.
            m = run(env, init_td, policy, max_steps, args.n,
                    track_edge=max(tc, tm) < 1 and not args.planted)
            scored[label] = m
            extra = f" {100 * m['oracle_frac']:>9.1f}%" if "oracle_frac" in m else ""
            print(f"{tc:>6.2f} {tm:>6.2f} {label:>10} "
                  f"{100 * m['saved']:>10.1f}% ±{100 * m['saved_std']:<4.1f}"
                  f"{m['ret']:>9.3f} {m['cuts']:>7.1f} {m['steps']:>8.0f} "
                  f"{m['truncated']:>6d}{extra}")

        # The paired difference is the statistic that matters. Episode-to-episode
        # spread is huge -- where the fire ignites dominates -- so two means over
        # 100 episodes each carry a standard error of several points and routinely
        # disagree by more than the agents do. Both agents ran the *same*
        # episodes, so differencing per episode cancels the ignition luck.
        # On planted instances the reference is the oracle, and every other
        # agent is differenced against it; otherwise planner against PPO.
        if "oracle" in scored:
            pairs = [("oracle", other) for other in scored if other != "oracle"]
        elif len(scored) == 2:
            pairs = [("planner", "ppo")]
        else:
            pairs = []
        # Per-agent paired margin against the reference, so each CSV row stands
        # alone. `paired` is (reference - this agent) in points of area saved,
        # exactly the sign convention the printed row uses; it is blank on the
        # reference agent's own row.
        margin = {}
        for ref, other in pairs:
            diff = 100 * (scored[ref]["per_episode"] - scored[other]["per_episode"])
            se = diff.std().item() / math.sqrt(len(diff))
            wins = int((diff > 1e-6).sum())
            ties = int((diff.abs() <= 1e-6).sum())
            margin[other] = (ref, diff.mean().item(), se, wins, ties)
            print(f"{'':>13} {'paired':>10} {diff.mean().item():>+10.2f} "
                  f"± {se:<4.2f} ({ref} - {other}, pts)   "
                  f"{ref} better on {wins}/{len(diff)}, tied {ties}")
        print()

        for label, m in scored.items():
            ref, mean, se, wins, ties = margin.get(label, ("", "", "", "", ""))
            rows.append({
                "n": args.n, "tc": tc, "tm": tm, "s": round(tc + tm, 6),
                "agent": label,
                "episodes": args.episodes, "seed": args.seed,
                "planted": PlantedConfig.from_string(args.planted).tag()
                           if args.planted else "",
                "saved": m["saved"], "saved_std": m["saved_std"],
                "ret": m["ret"], "cuts": m["cuts"], "steps": m["steps"],
                "truncated": m["truncated"],
                "oracle_frac": m.get("oracle_frac", ""),
                "edge_free": m.get("edge_free", ""),
                "paired_ref": ref, "paired_mean": mean, "paired_se": se,
                "paired_ref_wins": wins, "paired_ties": ties,
                "ckpt": os.path.basename(args.ckpt) if args.ckpt and label == "ppo" else "",
            })

    if args.csv:
        fields = list(rows[0].keys())
        fresh = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
        d = os.path.dirname(args.csv)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            if fresh:
                w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows to {args.csv}\n")

    print("Return is area-normalised: -1.0 means the whole forest was lost.\n"
          "'trunc' counts episodes that hit the step cap without the fire "
          "burning out.\nJudge the comparison on the paired row: the +- on each "
          "agent's own mean is\nepisode spread, not uncertainty about the agent. "
          "Note also that PPO's actions\ndepend on float details, so its score "
          "shifts a few points between cpu and cuda.")


if __name__ == "__main__":
    main()
