"""A TorchRL environment built from a :class:`~steerca.core.spec.CASpec`."""
from __future__ import annotations

from typing import Optional

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

from steerca.core import cs_env
from steerca.core.spec import CASpec, N_ACTIONS


class SteerCAEnv(EnvBase):
    """A natively batched SteerCA environment for any :class:`CASpec`.

    Observation keys, the same as Firebreak's so the same networks apply:
    ``grid`` (bool, ``(C, n, n)``), ``t`` (float, time left until the next CA
    tick, in ``(0, 1]``), ``agent_state`` (int64 ``(row, col)``) and ``params``
    (``tc``, ``tm``, ``n`` and the spec's ``extra_params``). The action is an
    int64 in ``0..8``.

    >>> env = SteerCAEnv(spec, n=16, tc=0.2, tm=0.2, batch_size=[8])
    >>> td = env.rollout(100)   # random policy
    """

    batch_locked = False

    def __init__(self, spec: CASpec, n: int, tc: float, tm: float,
                 device="cpu", batch_size=None, seed: Optional[int] = None, **kwargs):
        super().__init__(device=device, batch_size=batch_size, **kwargs)
        self.spec_ = spec
        self.n, self.tc, self.tm = int(n), float(tc), float(tm)
        self._funcs = spec.funcs()
        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def gen_params(self, batch_size=None) -> TensorDictBase:
        params = {"tc": self.tc, "tm": self.tm, "n": self.n,
                  **self.spec_.extra_params(self.n)}
        td = TensorDict({"params": TensorDict(params, [])}, [])
        if batch_size:
            td = td.expand(batch_size).contiguous()
        return td

    def _make_spec(self):
        n, c = self.n, self.spec_.n_states
        extra = {
            k: Unbounded(shape=(), dtype=torch.int64 if isinstance(v, int) else torch.float32)
            for k, v in self.spec_.extra_params(n).items()
        }
        observation_spec = Composite(
            grid=Bounded(low=0, high=1, shape=(c, n, n), dtype=torch.bool),
            t=Bounded(low=0.0, high=1.0, shape=(), dtype=torch.float32),
            agent_state=Bounded(low=0, high=n - 1, shape=(2,), dtype=torch.int64),
            params=Composite(
                tc=Bounded(low=0.0, high=float("inf"), shape=(), dtype=torch.float32),
                tm=Bounded(low=0.0, high=float("inf"), shape=(), dtype=torch.float32),
                n=Bounded(low=1, high=int(1e6), shape=(), dtype=torch.int64),
                **extra,
                shape=(),
            ),
            shape=(),
        )
        action_spec = Bounded(low=0, high=N_ACTIONS - 1, shape=(), dtype=torch.int64)
        reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        if self.batch_size:
            observation_spec = observation_spec.expand(*self.batch_size, *observation_spec.shape)
            action_spec = action_spec.expand(*self.batch_size, *action_spec.shape)
            reward_spec = reward_spec.expand(*self.batch_size, *reward_spec.shape)
        self.observation_spec = observation_spec
        self.state_spec = observation_spec.clone()
        self.action_spec = action_spec
        self.reward_spec = reward_spec

    def _reset(self, tensordict):
        if tensordict is None or tensordict.is_empty() or "params" not in tensordict:
            tensordict = self.gen_params(batch_size=self.batch_size)
        shape = tuple(tensordict.shape)
        # As in Firebreak, a partial reset regenerates the whole batch and
        # torchrl keeps the slots that were not asked to reset.
        grid, pos = self.spec_.reset(shape, self.n, tensordict["params"], self.device)
        return TensorDict(
            {
                "grid": grid,
                "t": torch.ones(shape, device=self.device),
                "agent_state": pos,
                "params": tensordict["params"],
            },
            batch_size=shape,
            device=self.device,
        )

    def _step(self, tensordict):
        return cs_env.step(tensordict, funcs=self._funcs)

    def _set_seed(self, seed: Optional[int]):
        self.rng = torch.manual_seed(seed)

    def max_episode_steps(self) -> int:
        return self.spec_.episode_cap(self.n, self.tc, self.tm)


def evaluate_policy(env: SteerCAEnv, policy, logger=None, step=0, max_steps=None):
    """Run one episode per batch slot to its end; return mean return and score.

    The batched counterpart of Firebreak's ``test_policy`` for any spec: every
    episode runs in parallel, a finished episode stops accruing reward, and its
    ``score`` is frozen at the step it ended. Returns ``{"return", "score"}``,
    with ``score`` ``None`` when the spec defines none.
    """
    spec = env.spec_
    k = int(env.batch_size[0])
    max_steps = max_steps or env.max_episode_steps() + 100
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        td = env.reset()
        total = torch.zeros(k, device=td.device)
        alive = torch.ones(k, dtype=torch.bool, device=td.device)
        score = spec.score(td["grid"]).reshape(k).float() if spec.score else None
        for _ in range(max_steps):
            td = env.step(policy(td))
            nxt = td["next"]
            total += nxt["reward"].reshape(k) * alive.float()
            if score is not None:
                score = torch.where(alive, spec.score(nxt["grid"]).reshape(k).float(), score)
            alive &= ~nxt["done"].reshape(k)
            if not alive.any():
                break
            td = step_mdp(td)
    out = {"return": total.mean().item(),
           "score": score.mean().item() if score is not None else None}
    if logger is not None:
        logger.log_scalar("test/avg_total_reward", out["return"], step=step)
        if out["score"] is not None:
            logger.log_scalar("test/score", out["score"], step=step)
    return out
