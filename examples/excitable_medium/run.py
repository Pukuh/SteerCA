"""Sanity checks and baselines for the excitable-medium example.

    python examples/excitable_medium/run.py

1. The rule: does a single excited cell start a wave, and which gap widths in a
   wall of inert cells does a wave get through?
2. Baselines on the environment: an agent that never acts and a uniformly
   random one, scored on the spec's metric, the same way training scores a
   policy (``evaluate_policy``). A trained policy should beat both.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from examples.excitable_medium.spec import (  # noqa: E402
    EXCITED, INERT, REST, SCAR, ca_step, done, make_spec)
from steerca.core import SteerCAEnv, evaluate_policy  # noqa: E402


def blank(n):
    g = torch.zeros(1, 5, n, n, dtype=torch.bool)
    g[:, REST] = True
    return g


def excite(g, cells):
    for r, c in cells:
        g[0, REST, r, c], g[0, EXCITED, r, c], g[0, SCAR, r, c] = False, True, True


def settle(g, cap=500):
    for _ in range(cap):
        if done(g).item():
            break
        g = ca_step(g)
    return g


def crosses(gap, n=24, wall=16):
    """Does a wave from a 2x2 seed cross a wall at column `wall` with a `gap`-cell opening?"""
    g = blank(n)
    mid = n // 2
    opening = range(mid - gap // 2, mid - gap // 2 + gap)
    for r in range(n):
        if r not in opening:
            g[0, REST, r, wall], g[0, INERT, r, wall] = False, True
    excite(g, [(mid - 1, mid - 1), (mid - 1, mid), (mid, mid - 1), (mid, mid)])
    return settle(g)[0, SCAR, :, wall + 1:].any().item()


class Idle:
    def __call__(self, td):
        td["action"] = torch.zeros(td.shape, dtype=torch.int64, device=td.device)
        return td


class Random:
    def __call__(self, td):
        td["action"] = torch.randint(0, 9, td.shape, device=td.device)
        return td


if __name__ == "__main__":
    g = blank(9)
    excite(g, [(4, 4)])
    print(f"single excited cell starts a wave: {bool(settle(g)[0, SCAR].sum() > 1)}")
    print("wave crosses a wall with a gap of width: "
          + ", ".join(f"{w}: {'yes' if crosses(w) else 'no'}" for w in range(1, 9)))

    spec = make_spec()
    for n, tc, tm in ((16, 0.2, 0.2), (16, 0.06, 0.06)):
        for label, policy in (("idle", Idle()), ("random", Random())):
            torch.manual_seed(0)
            env = SteerCAEnv(spec, n, tc, tm, batch_size=[200], seed=0)
            out = evaluate_policy(env, policy)
            print(f"{n}x{n} tc={tc} tm={tm}  {label:>6}: score {100 * out['score']:5.1f}%  "
                  f"return {out['return']:+.3f}")
