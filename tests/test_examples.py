"""The worked examples in examples/ build, step, and say what they claim.

Run from the repo root:  python tests/test_examples.py
Set STEERCA_SLOW_TESTS=1 to also train each example for one batch.
"""
import os
import subprocess
import sys

import torch
from torchrl.envs.utils import check_env_specs

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from steerca.core import SteerCAEnv, evaluate_policy  # noqa: E402
from steerca.core.spec import CASpec, load_spec  # noqa: E402
from steerca.evaluate import make_ppo_policy  # noqa: E402
from steerca.firebreak import ca  # noqa: E402

import examples.excitable_medium.run as ex_run  # noqa: E402
import examples.firebreak_wind.spec as wind  # noqa: E402
from examples.excitable_medium.spec import make_spec as excitable_spec  # noqa: E402


def test_specs_are_valid_envs():
    for spec in (excitable_spec(), wind.make_spec()):
        env = SteerCAEnv(spec, 8, 0.2, 0.2, batch_size=[4], seed=0)
        check_env_specs(env)
        out = evaluate_policy(env, ex_run.Random())
        assert 0.0 <= out["score"] <= 1.0
    print("example envs ............. ok")


def test_excitable_rule():
    g = ex_run.blank(9)
    ex_run.excite(g, [(4, 4)])
    assert ex_run.settle(g)[0, ex_run.SCAR].sum() == 1, "a lone excited cell started a wave"
    assert not ex_run.crosses(1), "a wave crossed a one-cell gap"
    assert ex_run.crosses(4), "a wave failed to cross a four-cell gap"
    print("excitable rule ........... ok")


def test_wind_is_firebreak_with_another_kernel():
    torch.manual_seed(0)
    g = torch.nn.functional.one_hot(torch.randint(0, 3, (4, 10, 10)), 3).permute(0, 3, 1, 2).bool()
    kernel = wind.WIND_FROM_WEST
    try:
        wind.WIND_FROM_WEST = ca._fire_kernel(torch.device("cpu")).cpu()
        assert torch.equal(wind.windy_step(g), ca.step(g))
    finally:
        wind.WIND_FROM_WEST = kernel
    # A fire on the east edge of an all-tree grid never burns west.
    g = torch.zeros(1, 3, 6, 6, dtype=torch.bool)
    g[:, ca.TREE] = True
    g[0, ca.TREE, 3, 5], g[0, ca.FIRE, 3, 5] = False, True
    for _ in range(10):
        g = wind.windy_step(g)
    assert g[0, ca.TREE, :, :5].all()
    print("wind kernel .............. ok")


def test_load_spec():
    assert load_spec(None) is None and load_spec("firebreak") is None
    by_module = load_spec("examples.excitable_medium.spec:make_spec")
    by_file = load_spec(os.path.join(ROOT, "examples", "firebreak_wind", "spec.py") + ":make_spec")
    assert isinstance(by_module, CASpec) and by_module.name == "excitable"
    assert isinstance(by_file, CASpec) and by_file.name == "firebreak-wind"
    print("STEERCA_ENV resolution ... ok")


def test_train_one_batch():
    if os.environ.get("STEERCA_SLOW_TESTS") != "1":
        print("train one batch .......... skipped (set STEERCA_SLOW_TESTS=1)")
        return
    import tempfile
    for ref in ("examples.excitable_medium.spec:make_spec", "examples.firebreak_wind.spec:make_spec"):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, STEERCA_ENV=ref, STEERCA_MODELS=tmp,
                       STEERCA_LOG_DIR=os.path.join(tmp, "csv"), STEERCA_N="8",
                       STEERCA_TC="0.2", STEERCA_TM="0.2", STEERCA_FRAMES="2000")
            subprocess.run([sys.executable, "-m", "steerca.train"], cwd=ROOT, env=env,
                           check=True, capture_output=True)
            final = [f for f in os.listdir(tmp) if f.endswith("_final_policy.pth")]
            assert len(final) == 1
            # The checkpoint loads back through the public helper and runs.
            spec = load_spec(ref)
            policy = make_ppo_policy(os.path.join(tmp, final[0]), "equiv", "cpu", 8,
                                     in_states=spec.n_states)
            out = evaluate_policy(SteerCAEnv(spec, 8, 0.2, 0.2, batch_size=[8], seed=0), policy)
            assert 0.0 <= out["score"] <= 1.0
    print("train one batch .......... ok")


if __name__ == "__main__":
    test_specs_are_valid_envs()
    test_excitable_rule()
    test_wind_is_firebreak_with_another_kernel()
    test_load_spec()
    test_train_one_batch()
    print("\nall checks passed")
