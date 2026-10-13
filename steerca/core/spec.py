"""What defines a SteerCA environment.

A SteerCA environment couples three things, and a :class:`CASpec` names each of
them as a plain function over batched tensors:

* a **cellular automaton**: a grid of ``n_states`` boolean planes, shape
  ``(B, n_states, n, n)``, advanced one tick at a time by ``ca_step``;
* a **local agent** at a cell of that grid, with nine actions: eight moves
  (``0..7``: E, NE, N, NW, W, SW, S, SE) and one action on the cell it stands on
  (``8``), described by ``action``;
* the **hybrid time model**: a move takes ``tm`` CA ticks and the cell action
  ``tc`` (``time``), so several actions fit in one tick when they are cheap and
  one action spans several ticks when it is not.

The generic step in :mod:`steerca.core.cs_env` composes them. On every action it
advances the CA by as many ticks as the action's duration covers, then applies
the action, then asks the spec for the reward and whether the CA has reached its
fixed point. :class:`steerca.core.env.SteerCAEnv` turns a spec into a TorchRL
environment, and ``python -m steerca.train`` trains on it when
``STEERCA_ENV=<module>:<factory>`` names a function returning one.

Firebreak, the forest-fire suite of the papers, is itself a spec:
:func:`steerca.firebreak.spec.firebreak_spec`. ``docs/extending.md`` walks
through writing a new one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

CELL_ACTION = 8
N_ACTIONS = 9


def default_time(actions: torch.Tensor, params) -> torch.Tensor:
    """Duration of each action in CA ticks: ``tc`` for the cell action, ``tm`` for a move."""
    return torch.where(actions == CELL_ACTION, params["tc"], params["tm"])


def no_action_reward(actions, next_grid, agent_state) -> torch.Tensor:
    """A reward term that charges nothing for acting."""
    return torch.zeros(actions.shape, dtype=torch.float32, device=actions.device)


def default_episode_cap(n: int, tc: float, tm: float) -> int:
    """Firebreak's cap: ~n ticks for a front to cross the grid, 1/min(tc, tm) actions per tick, 1.4x headroom."""
    return int(1.4 * n / min(tc, tm))


@dataclass
class CASpec:
    """A SteerCA environment, as functions over batched tensors.

    Shapes: ``B`` is the batch, ``C = n_states``, ``n`` the grid side. Grids are
    ``torch.bool``, ``(B, C, n, n)``; the agent state is ``int64`` ``(B, 2)``
    holding ``(row, col)``; actions are ``int64`` ``(B,)``.

    Required:
        name: short identifier, used in experiment and checkpoint names.
        n_states: number of boolean planes in the grid. They need not be one-hot:
            a plane can be a flag the CA carries along (see the
            ``excitable_medium`` example's scar plane).
        ca_step: ``grid -> grid``, one tick of the automaton. Must not mutate
            its input.
        action: ``(actions, grid, agent_state) -> (grid, new_agent_state)``.
            May mutate ``grid`` in place. Build one from a cell effect with
            :func:`steerca.core.agent.local_action`.
        reward_grid: ``(grid, mid_grid, next_grid) -> (B,) float32``, the reward
            for what changed: before the step, after the CA advance, after the
            action. Seeing all three lets it tell what the CA did from what the
            agent did.
        done: ``grid -> (B,) bool``, True once the CA can no longer change. An
            episode also ends at ``episode_cap`` steps.
        reset: ``(shape, n, params, device) -> (grid, agent_state)``, the
            initial condition, drawn with the global torch RNG so
            ``torch.manual_seed`` reproduces it. ``shape`` is the batch shape
            and ``params`` the env's parameter TensorDict.

    Optional:
        reward_action: ``(actions, next_grid, agent_state) -> (B,)``, a per-action
            charge added to ``reward_grid``. Default: nothing.
        time: ``(actions, params) -> (B,)`` durations in ticks. Default: ``tc``
            for the cell action, ``tm`` for moves.
        score: ``final_grid -> (B,) float``, the task metric, higher is better.
            Evaluation and checkpoint selection use it in place of the return,
            which a shaped reward can make misleading. Default: none, and
            evaluation falls back to the return.
        extra_params: ``n -> dict`` of extra scalar parameters the reset reads
            from ``params`` (Firebreak's agent-to-fire ``distance``).
        episode_cap: ``(n, tc, tm) -> int`` steps before truncation.
        symmetric: whether the dynamics are invariant under the 8 rotations and
            reflections of the square. If True, training augments minibatches
            with them (``RandomAugmentations``). Set False for anisotropic rules
            such as wind.
        render: optional ``(grid, agent_state) -> HxWx3 uint8`` for videos.
    """

    name: str
    n_states: int
    ca_step: Callable
    action: Callable
    reward_grid: Callable
    done: Callable
    reset: Callable
    reward_action: Callable = no_action_reward
    time: Callable = default_time
    score: Optional[Callable] = None
    extra_params: Callable = field(default=lambda n: {})
    episode_cap: Callable = default_episode_cap
    symmetric: bool = True
    render: Optional[Callable] = None

    def funcs(self) -> dict:
        """The hooks :func:`steerca.core.cs_env.step` expects."""
        return {
            "fun_ca_step": self.ca_step,
            "fun_time": self.time,
            "fun_action": self.action,
            "fun_reward_grid": self.reward_grid,
            "fun_reward_action": self.reward_action,
            "fun_done": self.done,
        }


def load_spec(ref: Optional[str]) -> Optional[CASpec]:
    """Resolve a ``STEERCA_ENV`` value to a spec; ``None`` means Firebreak's own env.

    ``ref`` is ``"firebreak"`` (or empty) for the default, or ``"<target>:<factory>"``
    where ``<target>`` is a dotted module (``examples.excitable_medium.spec``) or a
    path to a ``.py`` file, and ``<factory>`` a function returning a
    :class:`CASpec`. Modules are looked up from the current directory too, so the
    repository's ``examples/`` work when run from its root.
    """
    if ref is None or ref.strip().lower() in ("", "firebreak"):
        return None
    import importlib
    import importlib.util
    import os
    import sys

    target, sep, factory = ref.strip().rpartition(":")
    if not sep or not target or not factory:
        raise SystemExit(f"STEERCA_ENV={ref!r} must look like 'package.module:make_spec' "
                         f"or 'path/to/spec.py:make_spec'")
    if target.endswith(".py"):
        module_name = "steerca_user_spec_" + os.path.splitext(os.path.basename(target))[0]
        loader = importlib.util.spec_from_file_location(module_name, target)
        if loader is None:
            raise SystemExit(f"STEERCA_ENV: cannot load {target!r}")
        module = importlib.util.module_from_spec(loader)
        sys.modules[module_name] = module
        loader.loader.exec_module(module)
    else:
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        module = importlib.import_module(target)
    spec = getattr(module, factory)()
    if not isinstance(spec, CASpec):
        raise SystemExit(f"STEERCA_ENV: {ref} returned {type(spec).__name__}, not a CASpec")
    if "_" in spec.name:
        raise SystemExit(f"CASpec name {spec.name!r} must not contain '_': it is a "
                         f"field of the checkpoint file name (use '-')")
    return spec
