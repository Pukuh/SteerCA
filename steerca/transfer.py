"""Zero-shot transfer: run one checkpoint, unchanged, at several grid sizes.

The backbone has no parameter that depends on the grid size, so a policy
trained at 8x8 loads verbatim at 16x16 or 32x32. This scores it there, paired
against an untrained network of the same architecture (the floor a from-scratch
run starts at) on identical episodes, at the action costs the checkpoint was
trained with (read from its name, or --costs).

    python -m steerca.transfer --ckpt artifacts/models/ppo_equiv_c96_b6_512_64_1hot_8x8_t0.2_0.2_cut0.5_r10_policy.pth
    python -m steerca.transfer --ckpt <path> --sizes 8 16 32 --episodes 100

Returns are normalised by the grid area, so they are comparable across sizes:
-1.0 means the whole forest was lost. Trees saved is the score.
The papers' zero-shot numbers come from ``paper/collect.py zeroshot``, which
also pairs every run with the planner.
"""
import argparse
import os

import torch

from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torchrl.modules import ProbabilisticActor

import steerca.firebreak.env as ff
from steerca.archs import (ForestFireEquivariantNet, ForestFirePolicyNet,
                           arch_for_grid, parse_arch_tag)
from steerca.evaluate import make_ppo_policy, run
from steerca.render import settings_from_name

# One of the published r10 policies (fetch with tools/fetch_artifacts.py).
DEFAULT_CKPT = os.path.join(
    os.environ.get("STEERCA_ARTIFACTS", "artifacts"), "models",
    "ppo_equiv_c96_b6_512_64_1hot_8x8_t0.2_0.2_cut0.5_r10_policy.pth")


def untrained_like(ckpt, device, n):
    """A freshly initialised actor with the checkpoint's architecture."""
    cfg = parse_arch_tag(os.path.basename(ckpt)) or arch_for_grid(n)
    torch.manual_seed(0)
    net = ForestFirePolicyNet(
        backbone_net=ForestFireEquivariantNet(grid_size=n, device=device, **cfg),
        n_actions=9, hidden=cfg["hidden"], device=device)
    actor = ProbabilisticActor(
        TensorDictModule(net, in_keys=["grid", "t", "agent_state"], out_keys=["logits"]),
        distribution_class=Categorical, in_keys=["logits"], out_keys=["action"])
    return actor.eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--costs", default=None, help='"tc,tm"; default: from the checkpoint name')
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.costs:
        tc, tm = (float(x) for x in args.costs.split(","))
    else:
        _, tc, tm = settings_from_name(os.path.basename(args.ckpt))
    device = torch.device(args.device)
    print(f"checkpoint: {args.ckpt}")
    print(f"tc={tc:g} tm={tm:g}   device {device}   episodes per cell {args.episodes}\n")
    print(f"{'grid':>6} {'policy':>12} {'trees saved':>14} {'return':>9} {'steps':>7}")
    print("-" * 54)
    for n in args.sizes:
        env = ff.make_env(n, tc, tm, args.episodes, device)
        torch.manual_seed(args.seed)
        env.set_seed(args.seed)
        init_td = env.reset()
        max_steps = ff.episode_cap(n, tc, tm)
        arms = (("trained", make_ppo_policy(args.ckpt, "equiv", device, n)),
                ("untrained", untrained_like(args.ckpt, device, n)))
        for label, policy in arms:
            m = run(env, init_td, policy, max_steps, n)
            print(f"{n:>4}x{n:<2} {label:>12} "
                  f"{100 * m['saved']:>9.1f}% ±{100 * m['saved_std']:<3.1f}"
                  f"{m['ret']:>9.3f} {m['steps']:>7.0f}")


if __name__ == "__main__":
    main()
