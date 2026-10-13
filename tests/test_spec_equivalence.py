"""The generic API reproduces Firebreak exactly.

``SteerCAEnv(firebreak_spec(), ...)`` must step identically to ``ForestFireEnv``,
the environment every published number was measured on: same initial states from
the same seed, and the same grid, agent position, clock, reward and termination
after every action. If this holds, a variant built by editing the spec starts
from exactly the task in the papers.

Run from the repo root:  python tests/test_spec_equivalence.py
"""
import os
import sys

import torch
from torchrl.envs.utils import check_env_specs, step_mdp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import steerca.firebreak.env as ff  # noqa: E402
from steerca.core import SteerCAEnv  # noqa: E402
from steerca.firebreak.spec import firebreak_spec  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _pair(n, tc, tm, batch, device, seed):
    torch.manual_seed(seed)
    ref = ff.make_env(n, tc, tm, batch, device)
    ref.set_seed(seed)
    td_ref = ref.reset()
    torch.manual_seed(seed)
    gen = SteerCAEnv(firebreak_spec(), n, tc, tm, device=device, batch_size=[batch], seed=seed)
    td_gen = gen.reset()
    return ref, td_ref, gen, td_gen


def _same(a, b, keys, where):
    for k in keys:
        assert torch.equal(a[k], b[k]), f"{where}: {k} differs"


def test_rollouts_are_identical():
    for device in DEVICES:
        for n, tc, tm in ((8, 0.06, 0.06), (16, 0.2, 0.3), (12, 0.5, 1.5)):
            ref, td_ref, gen, td_gen = _pair(n, tc, tm, 16, device, seed=n)
            _same(td_ref, td_gen, ("grid", "agent_state", "t"), f"{device} reset")
            actions = torch.Generator().manual_seed(1)
            for step in range(3 * n):
                a = torch.randint(0, 9, (16,), generator=actions).to(device)
                td_ref["action"], td_gen["action"] = a, a.clone()
                td_ref, td_gen = ref.step(td_ref), gen.step(td_gen)
                _same(td_ref["next"], td_gen["next"],
                      ("grid", "agent_state", "t", "reward", "done"),
                      f"{device} n={n} tc={tc} tm={tm} step {step}")
                td_ref, td_gen = step_mdp(td_ref), step_mdp(td_gen)
    print(f"spec == ForestFireEnv .... ok  ({', '.join(DEVICES)})")


def test_specs_check():
    check_env_specs(SteerCAEnv(firebreak_spec(), 8, 0.2, 0.2, batch_size=[4], seed=0))
    print("SteerCAEnv specs ......... ok")


def test_score_is_trees_saved():
    env = SteerCAEnv(firebreak_spec(), 8, 0.2, 0.2, batch_size=[2], seed=0)
    td = env.reset()
    score = firebreak_spec().score(td["grid"])
    assert torch.allclose(score, torch.full((2,), 63 / 64))
    print("score = trees saved ...... ok")


if __name__ == "__main__":
    test_rollouts_are_identical()
    test_specs_check()
    test_score_is_trees_saved()
    print("\nall checks passed")
