import torch
from tensordict import TensorDict, TensorDictBase

def step(tensordict, funcs) -> TensorDict:

    # retrieve the complex system and agent state from the tensordict
    grid = tensordict["grid"]
    t = tensordict["t"]
    action = tensordict["action"]

    # restrive functions from the particular environment
    fun_ca_step = funcs["fun_ca_step"]
    fun_time = funcs["fun_time"]
    fun_action = funcs["fun_action"]
    fun_reward_grid = funcs["fun_reward_grid"]
    fun_reward_action = funcs["fun_reward_action"]
    fun_done = funcs["fun_done"]

    agent_state = tensordict["agent_state"]
    params = tensordict["params"]


    batched = action.ndim > 0
    if batched==False:
        action = action.unsqueeze(0)
        grid = grid.unsqueeze(0)
        agent_state = agent_state.unsqueeze(0)
        t = t.unsqueeze(0)

    # update the complex system for each sample in the batch
    # compute steps per sample
    times = fun_time(action, params)  # [B]
    # One epsilon guards the tick boundary: in float32 the accumulated clock can
    # land within rounding error above `times`, in which case a bare floor would
    # skip the tick while the clock wraps — silently freezing the CA forever.
    # eps is far below any sane tc/tm and far above float32 noise.
    eps = 1e-4
    overshoot = times - t  # [B] time past the tick boundary (negative if before it)
    steps = torch.floor(1.0 + overshoot + eps).to(torch.int64)  # [B]
    max_steps = int(steps.max().item())
    # step the full batch each tick and keep updates only where still active;
    # torch.where returns a fresh tensor, so grid itself is never mutated
    next_grid = grid.clone() if max_steps == 0 else grid
    for i in range(max_steps):
        mask = (steps > i).view(-1, 1, 1, 1)  # [B,1,1,1]
        next_grid = torch.where(mask, fun_ca_step(next_grid), next_grid)

    # The state after the CA advance but BEFORE the agent acts. `fun_action`
    # mutates the grid in place, so this has to be a copy: it is what lets the
    # reward tell a tree the fire took from one the agent felled, and charge them
    # differently. A bool grid, so the copy is cheap next to a CA tick.
    mid_grid = next_grid.clone()

    # apply the agent's action
    # does not create a copy of the grid
    next_grid, new_agent_state = fun_action(action, next_grid, agent_state)

    # Reward for the transition, over the WHOLE step: the CA advance AND the
    # agent's action. The hook gets all three states -- before, after the CA,
    # after the action -- so it can attribute each loss to whichever caused it.
    #
    # No max_steps guard: a cut changes the grid even on a step that advances no
    # CA tick, and it has to be charged there too.
    reward_grid = fun_reward_grid(grid, mid_grid, next_grid)
    # calculate the reward related to the action
    reward_action = fun_reward_action(action, next_grid, new_agent_state)


    # update the time: remaining time until the next tick, derived from the same
    # `steps` used to advance the CA so clock and CA can never disagree; the
    # clamp keeps accumulated rounding error from pushing t above the 1.0 bound
    new_t = torch.clamp(steps.to(t.dtype) - overshoot, max=1.0)

    # Check if the complex system has reached a fixed point
    if max_steps > 0:
        done = fun_done(next_grid)
    else:
        done = torch.zeros_like(t, dtype=torch.bool)

    reward = reward_grid + reward_action

    if batched==False:
        next_grid = next_grid.squeeze(0)
        new_agent_state = new_agent_state.squeeze(0)
        action = action.squeeze(0)
        new_t = new_t.squeeze(0)
    else:
        # reward/done specs have a trailing singleton dim: [B] -> [B, 1]
        reward = reward.unsqueeze(-1)
        done = done.unsqueeze(-1)

    out = TensorDict({
        "grid": next_grid,
        "t": new_t,
        "agent_state": new_agent_state,
        "params": params,
        "reward": reward,
        "done": done,
        },
        tensordict.shape,
    )

    return out
