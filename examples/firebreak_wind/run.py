"""Calm vs windy Firebreak, with an agent that never acts.

    python examples/firebreak_wind/run.py

Wind stops the fire burning back upwind, so more of the forest survives even
when nobody cuts anything. That baseline is what a policy trained on the
variant has to beat.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from examples.firebreak_wind.spec import make_spec  # noqa: E402
from steerca.core import SteerCAEnv, evaluate_policy  # noqa: E402
from steerca.firebreak.spec import firebreak_spec  # noqa: E402


class Idle:
    def __call__(self, td):
        td["action"] = torch.zeros(td.shape, dtype=torch.int64, device=td.device)
        return td


if __name__ == "__main__":
    for label, spec in (("calm", firebreak_spec()), ("wind", make_spec())):
        torch.manual_seed(0)
        env = SteerCAEnv(spec, 16, 0.2, 0.2, batch_size=[200], seed=0)
        out = evaluate_policy(env, Idle())
        print(f"{label:>5}: an idle agent keeps {100 * out['score']:5.1f}% of the forest")
