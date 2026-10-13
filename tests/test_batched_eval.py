"""Checks that the batched path in test_policy measures the same thing as the
sequential one, and that a finished episode's total is final.

Run from the repo root:  python tests/test_batched_eval.py
"""
import os
import statistics
import sys
import time

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import steerca.firebreak.env as ff

N, TC = 16, 0.2
ff.N, ff.TC, ff.TM = N, TC, TC

from steerca.firebreak.env import test_policy  # noqa: E402
from torchrl.envs import StepCounter, TransformedEnv  # noqa: E402
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp  # noqa: E402
import steerca.transfer as E  # noqa: E402
from steerca.evaluate import make_ppo_policy  # noqa: E402

# The published 16x16 t0.2 policy when the artifacts are present; otherwise an
# untrained net of the same architecture, which exercises the same code paths
# (every comparison below is between two ways of running ONE policy).
CKPT = os.path.join(os.environ.get("STEERCA_ARTIFACTS",
                                   os.path.join(os.path.dirname(__file__), "..", "artifacts")),
                    "models", f"ppo_equiv_c96_b6_512_64_1hot_{N}x{N}_t{TC:g}_{TC:g}_cut0.5_r10_policy.pth")
CAP = int(1.4 * N / TC)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class _Logs:
    """Stands in for the WandbLogger; records what test_policy logs."""
    def __init__(self): self.calls = []
    def log_scalar(self, name, value, step=None): self.calls.append((name, value, step))


def _env(batch_size=None, seed=0):
    e = TransformedEnv(ff.ForestFireEnv(device=DEV, batch_size=batch_size),
                       StepCounter(max_steps=CAP))
    e.set_seed(seed)
    return e


def _policy():
    if os.path.exists(CKPT):
        return make_ppo_policy(CKPT, "equiv", DEV, N)
    return E.untrained_like(CKPT, DEV, N)


def test_agrees_with_sequential():
    """Both paths estimate the same quantities, so their means must agree.

    Episodes differ between the two (different RNG draw order), so this compares
    distributions over many episodes rather than demanding equality. Both fields
    are checked: trees saved is what checkpoints are selected on, so the two
    paths disagreeing about it would be as bad as disagreeing about return.
    """
    pol = _policy()
    seq, bat = [], []
    for seed in range(6):
        logs = _Logs()
        seq.append(test_policy(_env(seed=seed), pol, logs, 0, k=30,
                               device=DEV, max_steps=CAP))
        bat.append(test_policy(_env(batch_size=[30], seed=seed), pol, logs, 0,
                               device=DEV, max_steps=CAP))
    for field, fmt in (("return", "+.4f"), ("saved", ".4f")):
        s_vals = [r[field] for r in seq]
        b_vals = [r[field] for r in bat]
        ms, mb = statistics.mean(s_vals), statistics.mean(b_vals)
        # spread of the sequential estimator itself, as the scale to judge against
        sd = statistics.stdev(s_vals)
        print(f"agreement ({field:>6}) .... sequential {ms:{fmt}}  batched {mb:{fmt}}  "
              f"diff {abs(ms-mb):.4f}  (seq run-to-run sd {sd:.4f})")
        # The floor covers a policy that does the same thing every episode (the
        # untrained fallback), where sd is 0 and only float rounding differs.
        assert abs(ms - mb) < max(sd, 1e-4), f"batched {field} {mb} outside noise of sequential {ms}"


def test_k_comes_from_batch_size():
    """A batched env must evaluate batch_size episodes, whatever k says."""
    pol = _policy()
    logs = _Logs()
    test_policy(_env(batch_size=[12]), pol, logs, 0, k=999, device=DEV, max_steps=CAP)
    assert [c[0] for c in logs.calls] == ["test/avg_total_reward", "test/trees_saved"]
    print("k from batch size ..... ok  (k=999 ignored, batch of 12 used)")


def test_saved_is_frozen_at_termination():
    """Trees saved must be read at the fixed point, not after it.

    The same masking argument as the reward, and it matters more here: a finished
    episode keeps getting stepped while its batch-mates run, and the agent goes
    on cutting in it. Without the freeze those cuts would lower the tree count of
    an episode that had already ended, so the metric checkpoints are selected on
    would drift downward with nothing to do with the policy's real performance.
    """
    pol = _policy()
    env = _env(batch_size=[16])
    k = 16
    area = N * N
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        td = env.reset()
        alive = torch.ones(k, dtype=torch.bool, device=DEV)
        trees = td["grid"][..., ff.TREE, :, :].sum((-2, -1)).reshape(k).float()
        at_done = torch.full((k,), float("nan"), device=DEV)
        for _ in range(CAP):
            td = env.step(pol(td))
            nxt = td["next"]
            now = nxt["grid"][..., ff.TREE, :, :].sum((-2, -1)).reshape(k).float()
            trees = torch.where(alive, now, trees)
            just = alive & nxt["done"].reshape(k)
            at_done[just] = trees[just]          # snapshot at termination
            alive &= ~nxt["done"].reshape(k)
            td = step_mdp(td)
        assert not alive.any(), f"{int(alive.sum())} episodes never finished within {CAP}"
        assert not torch.isnan(at_done).any()
        drift = (trees - at_done).abs().max().item()
    print(f"saved frozen at done .. ok  (max drift {drift:.2e}, "
          f"mean saved {100 * (trees / area).mean().item():.1f}%)")
    assert drift == 0.0, f"tree count moved after done: {drift}"


def test_finished_episodes_stop_accruing():
    """The masking claim, checked directly.

    Replays the batched loop and records each env's running total at the step it
    terminates, then keeps stepping to max_steps. Totals must not move after
    termination -- without the mask the agent goes on cutting trees in a
    finished episode and its reward would leak into the metric.
    """
    pol = _policy()
    env = _env(batch_size=[16])
    k = 16
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        td = env.reset()
        total = torch.zeros(k, device=DEV)
        alive = torch.ones(k, dtype=torch.bool, device=DEV)
        at_done = torch.full((k,), float("nan"), device=DEV)
        for _ in range(CAP):
            td = env.step(pol(td))
            nxt = td["next"]
            total += nxt["reward"].reshape(k) * alive.to(total.dtype)
            just = alive & nxt["done"].reshape(k)
            at_done[just] = total[just]          # snapshot at termination
            alive &= ~nxt["done"].reshape(k)
            td = step_mdp(td)
        # every episode finished, and none moved after it did
        assert not alive.any(), f"{int(alive.sum())} episodes never finished within {CAP}"
        assert not torch.isnan(at_done).any()
        drift = (total - at_done).abs().max().item()
    print(f"post-done masking ..... ok  (max drift after termination {drift:.2e})")
    assert drift == 0.0, f"reward accrued after done: {drift}"


def test_is_faster():
    pol = _policy()
    logs = _Logs()
    t0 = time.perf_counter(); test_policy(_env(), pol, logs, 0, k=30, device=DEV, max_steps=CAP)
    seq = time.perf_counter() - t0
    t0 = time.perf_counter(); test_policy(_env(batch_size=[30]), pol, logs, 0, device=DEV, max_steps=CAP)
    bat = time.perf_counter() - t0
    print(f"speedup ............... sequential {seq:.2f}s  batched {bat:.2f}s  "
          f"({seq/bat:.1f}x)")
    assert bat < seq


if __name__ == "__main__":
    torch.manual_seed(0)
    if not os.path.exists(CKPT):
        print(f"(no {os.path.basename(CKPT)}; using an untrained policy)")
    test_k_comes_from_batch_size()
    test_finished_episodes_stop_accruing()
    test_saved_is_frozen_at_termination()
    test_agrees_with_sequential()
    test_is_faster()
    print("\nall checks passed")
