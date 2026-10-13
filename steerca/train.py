"""Train PPO, A2C or SAC on Firebreak, or on any CASpec.

    STEERCA_N=16 STEERCA_TC=0.2 STEERCA_TM=0.2 python -m steerca.train
    STEERCA_ENV=examples.excitable_medium.spec:make_spec python -m steerca.train

Everything is configured by STEERCA_* environment variables (CARL_* also
accepted); the README lists them. The run's name encodes every setting that
changes the task or the learner, and its checkpoints go to
$STEERCA_MODELS/<name>_{policy,value}.pth (best evaluation by trees saved, or
by the spec's score) and <name>_final_{policy,value}.pth (end of training). A
run resumes from its own checkpoint; otherwise at 16x16 and 32x32 it warm
starts from the ladder's stage one size down.

Importing this module builds the environments, networks and losses without
training; paper/collect.py does that to reuse them.
"""
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)

from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torch.distributions import Categorical
from torchrl.objectives import A2CLoss, ClipPPOLoss, DiscreteSACLoss, SoftUpdate
from torchrl.objectives.value import GAE
from torchrl.data.replay_buffers.samplers import RandomSampler

from torchrl.record.loggers.csv import CSVLogger
from tqdm import tqdm

import multiprocessing as mp
import sys
import os
from steerca.config import setting, env_flag, models_dir
from steerca.core.env import SteerCAEnv, evaluate_policy
from steerca.core.spec import load_spec
from steerca.firebreak.env import (
    ForestFireEnv, N, TC, TM, PLANTED_CFG, RandomAugmentations, episode_cap, test_policy,
)
from steerca.firebreak.agents import cut_price_tag
from steerca.archs import ForestFirePolicyNet, ForestFireValueNet, ForestFireCommonNetPlain
from steerca.archs import ForestFireCommonNetCNN, ForestFireEquivariantNet
from steerca.archs import (
    arch_for_grid, arch_tag, parse_arch_tag, load_grown_state_dict,
)


is_fork = mp.get_start_method() == "fork"
# Every published run used CUDA. CPU works for smoke tests and small grids, but a
# 4M-frame run is days there. STEERCA_DEVICE overrides the choice.
_device = setting("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(_device)
if device.type == "cpu" and __name__ == "__main__":
    print("WARNING: training on CPU; expect it to be far slower than on a GPU")

# Which environment. Unset (or "firebreak") is the forest-fire suite of the
# papers, through ForestFireEnv exactly as every published run used it.
# STEERCA_ENV=<module or file.py>:<factory> instead names a function returning a
# CASpec (see steerca/core/spec.py and examples/), trained through SteerCAEnv.
ENV_SPEC = load_spec(setting("ENV"))
if ENV_SPEC is not None and PLANTED_CFG is not None:
    raise SystemExit("STEERCA_PLANTED applies to Firebreak only; unset it or STEERCA_ENV")
# Which learner. PPO is the default and its path is untouched; A2C is the same
# actor-critic without the clipped ratio or the epochs of reuse, and SAC is
# off-policy over a large replay buffer with twin Q heads and a tuned entropy
# temperature. The algorithm goes in the experiment name, so the three cannot
# collide on a checkpoint path and each warm starts up its own ladder.
# Training is unseeded by default: torch draws a seed at startup and every
# environment's seed is drawn from it, so two runs differ but neither can be
# repeated. STEERCA_SEED fixes the global seed before any network or environment
# is built. Give a seeded re-run of an existing cell its own STEERCA_TAG too, or it
# resumes that cell's checkpoint.
_seed = setting("SEED", "").strip()
SEED = int(_seed) if _seed else None
if SEED is not None:
    torch.manual_seed(SEED)

ALGO = setting("ALGO", "ppo").lower()
if ALGO not in ("ppo", "a2c", "sac"):
    raise SystemExit(f"STEERCA_ALGO={ALGO!r} must be one of ppo, a2c, sac")

num_cells = 256  # number of cells in each layer i.e. output dim.
lr = float(setting("LR", 1e-4))
# A warm start begins from a policy that already works, so the same learning rate
# that trains one from scratch is far too large for refining one: the 32x32 r10
# run peaked at 97.2% and then collapsed to ~1% at frame 304k, never recovering.
# Applied only when this run actually warm starts -- a from-scratch run keeps lr.
warm_start_lr_scale = float(setting("WARM_LR_SCALE", 0.3))
max_grad_norm = 1.0

frames_per_batch = 2000
# Measured end-to-end (collection + optimisation) on a 3090: ~195 frames/s at
# N=8 and N=16 with tm=0.01. Larger tm means shorter episodes and cheaper
# collection, so expect this to be no slower. 4M frames is ~5.6 h plus eval.
total_frames = int(setting("FRAMES", 4_000_000))

sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
if ALGO == "a2c":
    # A2C has no importance ratio, so a second pass over the same batch is
    # off-policy with nothing correcting for it. One epoch, one pass.
    num_epochs = 1
# Override for controlled comparisons, e.g. A2C at PPO's ten epochs so the two
# see the same number of gradient steps per frame.
if setting("EPOCHS"):
    num_epochs = int(setting("EPOCHS"))
# SAC is off-policy: a replay buffer that outlives the batch, and a fixed number
# of gradient steps per batch rather than epochs over it. 250 x 256 is one
# update per 8 frames collected, which keeps the target networks moving without
# the replay ratio that makes small-batch SAC diverge.
sac_buffer_size = 200_000
sac_batch_size = 256
sac_updates_per_batch = 250
sac_init_random_frames = 20_000
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
# Horizon settings are tied to the env's time resolution: 1/tm env steps fit in
# one CA tick, and an episode runs ~N+2 ticks. These were set when tm=0.01 gave
# 100 steps/tick, where gamma=0.99 discounted a single fire tick by 0.99^100 =
# 0.37 and a firebreak's payoff was invisible by the time it arrived.
# gamma=0.999 gives a 1000-step horizon; at tm=0.05 (20 steps/tick) that is ~50
# ticks, comfortably longer than an episode, so returns are near-undiscounted --
# which suits a task scored on total trees saved. Re-check this if tm changes by
# another order of magnitude.
# Set below, once max_episode_steps is known: it has to scale with the episode.
gamma = None  # discount factor for the reward
# GAE trace length is 1/(1 - gamma*lmbda) = ~91 steps: one fire tick at tm=0.01,
# ~4.5 ticks at tm=0.05, so a cut can be credited with the tree loss it causes.
# At the old lmbda=0.95 the trace was ~20 steps, a fifth of a tick at tm=0.01,
# and the advantage was almost pure critic bootstrap.
lmbda = 0.99 # GAE parameter for advantage estimation
entropy_eps = 5e-3
# The sanity check (scripts/run_sanity_check.sh; the papers' verification task) found A2C collapses its entropy at
# 5e-3 -- H from 2.14 to ~0 within ten batches, even on a task where a cut pays
# on the step it is made -- and learns that task at 0.05. PPO's clip is what
# lets it live with the small coefficient. Default unchanged for every PPO run.
if setting("ENTROPY"):
    entropy_eps = float(setting("ENTROPY"))
# Rewards are normalised by area, so a tree is worth 1/N^2. Harmless for PPO and
# A2C, whose advantages are normalised, but SAC adds an entropy bonus of order
# alpha * log(9) to every step, and at alpha ~ 1 that is ~140x the largest reward
# at 8x8. STEERCA_REWARD_SCALE multiplies the reward the LEARNER sees ("area" means
# N^2, so a tree is worth 1); eval returns and the logged average reward stay in
# the env's units, so they remain comparable with every other run.
_rs = setting("REWARD_SCALE", "").strip()
reward_scale = (float(N * N) if _rs == "area" else float(_rs)) if _rs else 1.0
# SAC's temperature. torchrl's defaults are alpha_init=1.0 and a target entropy
# of 0.98*log(9) = 2.15 nats, nearly uniform, which pins the policy there; the
# sanity check learned with 0.05 and 0.2 (with the reward scaled to area).
sac_alpha_init = float(setting("SAC_ALPHA_INIT", 1.0))
sac_target_entropy_weight = float(setting("SAC_TARGET_ENTROPY", 0.98))

num_envs = 8
# 2000 batches / 25 gives 80 eval points, plenty for checkpoint selection.
# Eval used to run its episodes sequentially and cost ~50% of wall clock, which
# is why this was ever a tradeoff; test_policy batches now, so 100 episodes take
# 0.46s against 222s of training between evals -- 0.2% of the cycle.
eval_every = 25
# Score depends heavily on where the fire ignites -- at tc=tm=0.2 a 16x16
# episode ranges from 24% to 81% of trees saved -- so a small-sample mean is
# dominated by ignition luck, and "best by eval" then selects lucky draws rather
# than better policies. Since test_policy batches, all these episodes run as one
# set of tensor ops and 100 costs barely more than 10.
eval_episodes = 100
# Episode length scales with N and with the action cost: the fire needs up to
# ~N+2 CA ticks to reach the far corner and burn out, and 1/min(tc,tm) actions
# fit in a tick. The 1.4 factor is headroom over that worst case, so episodes
# always run to the fixed point -- truncating them means the agent never
# observes the outcome of its own firebreak.
# (At N=16, tm=0.01 this gives 2240, matching the 1200-1900 measured there.)
# Planted instances get 2.5x instead of 1.4x: see episode_cap.
max_episode_steps = (episode_cap(N, TC, TM) if ENV_SPEC is None
                     else ENV_SPEC.episode_cap(N, TC, TM))

# The discount has to cover the episode, or the agent cannot see the outcome of
# its own firebreak: at N=32 with tm=0.04 an episode runs 1120 steps, and
# gamma=0.999 discounts its end to 0.33 -- exactly the failure the note above
# describes for the old tm=0.01 setting.
#
# The floor at 0.999 is what keeps every previous run untouched. The longest
# episode in any configuration trained so far is 448 steps (N=16, tm=0.05),
# which this formula would give 0.99888 -- below the floor, so every existing
# run keeps precisely the gamma it had. Only grids or costs beyond anything run
# so far raise it. gamma stays a pure function of (N, TC, TM), all three of
# which are already in the experiment name, so two runs cannot collide on one
# path with different horizons.
gamma = max(0.999, 1 - 1 / (2 * max_episode_steps))

def make_env(batch_size=None):
    if ENV_SPEC is None:
        env = ForestFireEnv(device=device, batch_size=batch_size)
    else:
        env = SteerCAEnv(ENV_SPEC, N, TC, TM, device=device, batch_size=batch_size)
    return TransformedEnv(env, StepCounter(max_steps=max_episode_steps))

# one natively batched env: all sub-envs step in a single set of tensor ops
base_env = make_env(batch_size=[num_envs])
# batched too, so test_policy runs all eval episodes in parallel
eval_env = make_env(batch_size=[eval_episodes])

hidden1 = 1024
hidden2 = 512
hidden3 = 256
tdim = 64
# N is imported from steerca.firebreak.env so the nets always match the env's grid size

""" backbone_net_plain = ForestFireCommonNetPlain(
    grid_size=N,
    hidden1=hidden1,
    hidden2=hidden2,
    hidden3=hidden3,
    tdim=tdim,
    device=device,
) """ 

""" filters_1 = 256
filters_2 = 256
backbone_net_cnn = ForestFireCommonNetCNN(
    grid_size=N,
    device=device,
    filters_1=filters_1,
    filters_2=filters_2,
    tdim=tdim,
    hidden=hidden,
) """

# Fully-convolutional backbone: readouts are a gather at the agent's cell (all
# 9 actions are local) plus global pooling, so no parameter depends on N -- the
# same 1.18M parameters at any grid size, against 9.1M at N=8 / 34.3M at N=16 for
# the flatten+Linear CNN, which put >90% of its capacity in one position-specific
# layer.
#
# What DOES depend on N is how far the trunk can see, and arch_for_grid picks the
# configuration for the grid: dense blocks up to N=16 (receptive field 27, frozen
# so every checkpoint on disk stays loadable), plus a tail of dilated blocks above
# that. See steerca.archs.arch_for_grid. STEERCA_ARCH overrides it with an
# explicit tag, e.g. STEERCA_ARCH=c96_b6_512_64 to run the narrow trunk at 32x32 as
# a control -- the tag IS the experiment name, so an override cannot collide with
# the run it is being compared against.
_arch_override = setting("ARCH")
arch = (parse_arch_tag(f"ppo_equiv_{_arch_override}_1hot") if _arch_override
        else arch_for_grid(N))
if arch is None:
    raise SystemExit(f"STEERCA_ARCH={_arch_override!r} is not a valid architecture tag, "
                     f"e.g. c96_b6_512_64 or c96_b9d2-4-8_512_64")
channels, blocks, hidden, tdim = arch["channels"], arch["blocks"], arch["hidden"], arch["tdim"]
# A custom CA brings its own number of grid planes; Firebreak's is the default 3.
_in_states = {} if ENV_SPEC is None else {"in_states": ENV_SPEC.n_states}
backbone_net = ForestFireEquivariantNet(grid_size=N, device=device, **arch, **_in_states)

net = ForestFirePolicyNet(backbone_net=backbone_net, n_actions=9, hidden=hidden, device=device)

policy = TensorDictModule(
    net,
    in_keys=["grid", "t", "agent_state"],
    out_keys=["logits"],
)

# Create a distribution from the policy output using softmax
policy_module = ProbabilisticActor(
    policy,
    spec=base_env.action_spec,
    distribution_class=Categorical,
    in_keys=["logits"],
    out_keys=["action"],
    return_log_prob=True,
)


value_net = ForestFireValueNet(backbone_net=backbone_net, hidden=hidden, device=device)

value_module = ValueOperator(
    module = value_net,
    in_keys=["grid", "t", "agent_state"],
)

if ALGO == "sac":
    # The Q head scores all nine actions at once, so it is the policy net's
    # shape with a different meaning. It gets its OWN trunk rather than sharing
    # the actor's: SAC keeps a delayed copy of the Q network, and sharing would
    # put the actor's features inside the target and feed the Q loss straight
    # back into them. 1.18M parameters either way.
    q_backbone = ForestFireEquivariantNet(grid_size=N, device=device, **arch, **_in_states)
    q_net = ForestFirePolicyNet(backbone_net=q_backbone, n_actions=9,
                                hidden=hidden, device=device)
    qvalue_module = ValueOperator(
        module=q_net,
        in_keys=["grid", "t", "agent_state"],
        out_keys=["action_value"],
    )


collector = SyncDataCollector(
    base_env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
    init_random_frames=sac_init_random_frames if ALGO == "sac" else None,
)

if ALGO == "sac":
    # Outlives the batch, and sampled with replacement: that is the whole point
    # of an off-policy learner here.
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=sac_buffer_size, device=device),
        sampler=RandomSampler(),
        batch_size=sac_batch_size,
    )
else:
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement()
    )

# D4 augmentation. A CA whose rule is not invariant under rotations and
# reflections (wind, say) must not be augmented, or the policy learns the wrong
# dynamics; its spec says so with symmetric=False.
if ENV_SPEC is None or ENV_SPEC.symmetric:
    rr = RandomAugmentations(device=device)
else:
    class _NoAugmentation:
        @staticmethod
        def _apply_transform(td):
            return td
    rr = _NoAugmentation()

advantage_module = None
target_updater = None
if ALGO == "ppo":
    advantage_module = GAE(
        gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True, device=device
    )
    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coeff=entropy_eps,
        # halved so the value loss doesn't dominate the shared backbone's gradients
        critic_coeff=0.5,
        loss_critic_type="smooth_l1",
        device=device,
    )
    loss_keys = ("loss_objective", "loss_critic", "loss_entropy")
elif ALGO == "a2c":
    # Same actor-critic and the same GAE, without the clipped ratio and without
    # reusing the batch. The comparison against PPO is therefore exactly the
    # value of the trust region on this task, nothing else.
    advantage_module = GAE(
        gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True, device=device
    )
    loss_module = A2CLoss(
        actor_network=policy_module,
        critic_network=value_module,
        entropy_bonus=bool(entropy_eps),
        entropy_coeff=entropy_eps,
        critic_coeff=0.5,
        loss_critic_type="smooth_l1",
    )
    loss_keys = ("loss_objective", "loss_critic", "loss_entropy")
else:
    # Discrete SAC: twin Q heads, a delayed target, and a temperature tuned
    # against a target entropy rather than the fixed coefficient PPO uses.
    loss_module = DiscreteSACLoss(
        actor_network=policy_module,
        qvalue_network=qvalue_module,
        action_space="categorical",
        num_actions=9,
        num_qvalue_nets=2,
        loss_function="smooth_l1",
        alpha_init=sac_alpha_init,
        target_entropy_weight=sac_target_entropy_weight,
    )
    loss_module.make_value_estimator(gamma=gamma)
    target_updater = SoftUpdate(loss_module, eps=0.995)
    loss_keys = ("loss_actor", "loss_qvalue", "loss_alpha")

optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)



if __name__ == "__main__":
    #print("Running policy:", policy_module(base_env.reset()))
    #print("Running value:", value_module(base_env.reset()))
    # r10: back to the original reward -- a flat 0.5 per cut, no charge for the
    # tree and no imminent test. The most learnable of the four and the most
    # exploitable; the cut price now goes in the name, so an r10 run under
    # different pricing cannot resume this one's checkpoint.
    #
    # r9: a felled tree is priced by whether the fire was about to reach it --
    # 0.5 ahead of the fire, 1.5 when it is already adjacent. Deliberate shaping,
    # not the trees-saved objective: cutting a tree ahead of the fire PAYS 0.5
    # over letting it burn, which is the signal r7 and r8 lacked (both converged
    # to never cutting from scratch: 0.0 cuts, 0.0% saved against the planner's
    # 87.1% at 8x8 tc=tm=0.06). Score these runs on TREES SAVED; a rising return
    # with flat trees-saved means it is farming that 0.5, not learning.
    #
    # r8: a felled tree is charged CUT_TREE_CHARGE (0.5) rather than the full 1,
    # so a cut costs 1.0 instead of r7's 1.5. That is the cheapest a cut can be
    # made without changing which policy is optimal: a doomed tree costs 1 if it
    # burns and 1 if it is felled, so nothing can be farmed, and the return is now
    # exactly -(trees lost)/area -- the quantity the task is scored on, where r7
    # priced a cut above the tree it destroyed and so declined barriers that
    # trees-saved would build. Returns are strictly higher than r7's at the same
    # behaviour, so every r7 VALUE head is mis-calibrated here. See the CUT_COST
    # note in steerca.firebreak.agents.
    #
    # r7: a tree the agent fells is now charged like any other tree lost.
    # reward_grid used to be differenced BEFORE the action, so a cut tree never
    # registered as a loss while the fire taking that same tree cost -1 --
    # meaning cutting a doomed tree PAID 0.5, reward an agent could farm without
    # saving anything. At tm+tc = 1.2 that was most of what a from-scratch run
    # learned to do: 0.25% of trees saved against the planner's 3.60%, with the
    # return still climbing. A cut now costs 1.5 (the tree, plus the 0.5 action),
    # so returns are strictly lower than r6's and every r6 VALUE head is
    # mis-calibrated -- hence the name bump. r6 POLICIES are still sensible
    # starting points, and `init_policy_only` keeps only those.
    #
    # r6: reward normalised by N*N (grid area) instead of N. Returns are now
    # ~[-1, 0] at every grid size and comparable across them, but every r5
    # checkpoint's VALUE head is calibrated to targets N times larger -- hence
    # the name bump, so no r5 checkpoint is silently resumed here.
    #
    # r5: equivariant fully-convolutional trunk with the r4 horizon settings
    # (gamma 0.999, lmbda 0.99, max_episode_steps = 140*N). Different
    # architecture from r3/r4, so those checkpoints are not loadable here.
    # At N=16 this run warm starts from the completed 8x8 r5 run -- see
    # `init_from` below.
    # The reward revision. Together with cut_price_tag() it identifies the reward
    # a checkpoint was trained under, which is what decides whether that
    # checkpoint's critic can be reused -- see init_policy_only below.
    REVISION = "r10"
    # Checkpoints are written to and resumed from here (STEERCA_MODELS).
    MODELS = models_dir().rstrip("/")
    os.makedirs(MODELS, exist_ok=True)
    # Planted instances (STEERCA_PLANTED) are a different task on the same reward:
    # the tag goes in the name so a planted run never resumes or warm starts
    # from a vanilla checkpoint by accident, and the ladder below stays within
    # planted runs. Empty for the all-tree grid, so every existing name holds.
    planted_infix = f"_{PLANTED_CFG.tag()}" if PLANTED_CFG is not None else ""
    # arch_tag renders the frozen N<=16 architecture as the exact `c96_b6_512_64`
    # every checkpoint on disk already carries, so those names are unchanged.
    # A free-form suffix, for running an experiment alongside one that already
    # owns its natural name. Without it a controlled re-run of an existing cell
    # silently resumes that cell's checkpoint instead of starting over.
    _tag = setting("TAG", "").strip()
    tag_suffix = f"_{_tag}" if _tag else ""
    # Learner settings that differ from the defaults, so a tuned baseline never
    # shares a checkpoint with an untuned one. Empty at the defaults, so every
    # existing name holds.
    hp_infix = ""
    if entropy_eps != 5e-3:
        hp_infix += f"_ent{entropy_eps:g}"
    if reward_scale != 1.0:
        hp_infix += f"_rs{reward_scale:g}"
    if ALGO == "sac" and (sac_alpha_init, sac_target_entropy_weight) != (1.0, 0.98):
        hp_infix += f"_a{sac_alpha_init:g}-te{sac_target_entropy_weight:g}"
    # A custom CA is named by its spec and has no cut pricing or planted tag;
    # Firebreak's name is exactly the one every published checkpoint carries.
    if ENV_SPEC is None:
        name_prefix = f"{ALGO}_equiv"
        reward_infix = f"_{cut_price_tag()}{planted_infix}"
    else:
        name_prefix = f"{ALGO}_{ENV_SPEC.name}_equiv"
        reward_infix = ""
    experiment_name = (
        f"{name_prefix}_{arch_tag(arch)}_1hot_{N}x{N}"
        f"_t{TC:g}_{TM:g}{reward_infix}{hp_infix}_{REVISION}{tag_suffix}"
    )
    #experiment_name = f"ppo_plain_h3_4x4_{hidden1}_{hidden2}_{tdim}_r2_1hot_trans"

    # Load the policy and value function modules.
    #
    # Precedence: this run's own checkpoint (so an interrupted run resumes), then
    # `init_from` (warm start from a smaller grid), then scratch.
    #
    # Warm starting across grid sizes works because ForestFireEquivariantNet has
    # no parameter that depends on N -- the trunk is all shared conv kernels and
    # the readouts (gather at the agent's cell, global pooling) are
    # parameter-free. The 8x8 checkpoint therefore loads at 16x16 with
    # strict=True. Measured zero-shot at 16x16: 96.0% of trees saved over 150
    # episodes, against 0.0% from random init.
    #
    # At N > 8 this defaults to the 8x8 run with the same architecture AND the
    # same action costs -- tc/tm change the task, so an 8x8 policy trained at a
    # different tc/tm is not a valid starting point. Run 8x8 first, then 16x16.
    #   STEERCA_INIT_FROM=<path>  to override
    #   STEERCA_INIT_FROM=none    to train from scratch
    #
    # Note the r5 and r6 checkpoints on disk are NOT picked up here: r5 predates
    # the N*N reward normalisation, and r6 predates charging the agent for the
    # trees it fells. Point STEERCA_INIT_FROM at an r6 policy deliberately if you
    # want it -- init_policy_only leaves the mis-calibrated value head behind.
    # One doubling at a time, and the source path is built from the SOURCE grid's
    # architecture, not this run's: at N=32 the trunk has a dilated tail the
    # 16x16 run did not, so naming it with this run's tag would point at a file
    # that cannot exist. 8x8 has no predecessor and starts from scratch.
    _ladder = {16: 8, 32: 16}
    _src_n = _ladder.get(N)
    _auto_init = None if _src_n is None else (
        f"{MODELS}/{name_prefix}_{arch_tag(arch_for_grid(_src_n))}_1hot_{_src_n}x{_src_n}"
        f"_t{TC:g}_{TM:g}{reward_infix}_{REVISION}{tag_suffix}_policy.pth"
    )
    _env_init = setting("INIT_FROM")
    if _env_init is None:
        init_from = _auto_init
    elif _env_init.lower() == "none":
        init_from = None
    else:
        init_from = _env_init
    # Whether to leave the value head at its fresh init on a warm start.
    #
    # This is decided, not fixed, because it was wrong in both directions. It
    # exists to survive a reward REVISION: an old critic predicts returns on a
    # scale that no longer exists, and feeding that into a policy which already
    # works is what cost the 16x16 r5 fine-tune (94.4%, having started at 96.0%).
    #
    # But leaving it fresh has its own failure, and a worse one. policy_module
    # and value_module share a single backbone, so a randomly initialised value
    # head sends its large early critic loss straight into the trunk the policy
    # depends on; critic_coeff=0.5 halves that, it does not prevent it. The 32x32
    # r10 run warm started policy-only, peaked at 97.2% -- the planner's level --
    # and collapsed to ~1% at frame 304k, never recovering over the next 3.7M.
    #
    # So: keep the critic when the source was trained under the SAME reward, and
    # only drop it when it was not. The reward is identified by the cut price and
    # the revision, both of which are in the filename, and returns are
    # area-normalised, so a same-reward critic is already calibrated for a
    # different grid size (16x16 returns -0.025 against 32x32's -0.018).
    # "Same reward" also means the same task: a vanilla critic predicts returns
    # around -0.02, while on planted instances the fire side always burns and
    # returns sit around -0.4, so a vanilla checkpoint named via STEERCA_INIT_FROM
    # warm starts a planted run policy-only.
    _src_name = os.path.basename(init_from) if init_from else ""
    _same_reward = f"{reward_infix}_{REVISION}_" in _src_name and (
        ENV_SPEC is None or f"_{ENV_SPEC.name}_equiv_" in _src_name)
    # A curriculum warm starts across instance DISTRIBUTIONS while the reward
    # stays put, so the tag differs but the critic is still calibrated. Dropping
    # it there is what the 32x32 collapse came from, so allow keeping it
    # explicitly. Only set this when the source really shares the reward.
    if setting("KEEP_CRITIC", "0") not in ("0", "", "false"):
        _same_reward = True
        print("Keeping the critic across a distribution change (STEERCA_KEEP_CRITIC)")
    _value_src = init_from.replace("policy", "value") if init_from else None
    _has_value = bool(_value_src) and os.path.exists(_value_src)
    init_policy_only = not (_same_reward and _has_value)
    if init_from and init_policy_only:
        why = ("a different reward" if not _same_reward
               else f"no value checkpoint at {os.path.basename(_value_src)}")
        print(f"Value head starts fresh: source is {why}")

    SAVE_PATH = f"{MODELS}/{experiment_name}_policy.pth"
    # What gets saved beside the policy. For PPO and A2C that is the value
    # network. For SAC it is the LOSS MODULE: torchrl copies the Q network into
    # its own TensorDictParams, so `qvalue_module`'s own tensors never move and
    # saving them would write an untrained Q head. The loss module's state dict
    # carries the live Q, its delayed target, and log_alpha.
    second_module = loss_module if ALGO == "sac" else value_module

    def _load(path, policy_only=False, grow=False):
        """Load a checkpoint pair, optionally keeping the value head fresh.

        Note what `policy_only` does and does not skip. policy_module and
        value_module share one ForestFireEquivariantNet instance, so the policy
        state dict carries the shared conv trunk as well as the policy head:
        loading it updates the features the critic sees too. The only thing left
        at its fresh init is the value head itself, a single Linear(hidden, 1).

        That is exactly what a warm start across a reward rescale wants -- keep
        the learned features, relearn the output scale from zero rather than
        inheriting one that is N times too large.

        `grow` permits the source to have fewer residual blocks than this run,
        which is how a 16x16 checkpoint warm starts a 32x32 run whose trunk has a
        dilated tail. It is off when resuming our own checkpoint: a missing key
        there means the file belongs to a different architecture, and that has to
        be a hard error rather than a silent regrow.
        """
        src = parse_arch_tag(os.path.basename(path))

        def _one(module, p):
            sd = torch.load(p, map_location=device)
            if not grow:
                module.load_state_dict(sd)  # strict
                return []
            return load_grown_state_dict(
                module, sd, what=os.path.basename(p),
                src_dilations=src["dilations"] if src else None,
                dst_dilations=arch["dilations"],
            )

        appended = _one(policy_module, path)
        if not policy_only:
            # For SAC this is the loss module rather than a state value, but it
            # lives at the same path so every script that pairs _policy.pth with
            # _value.pth keeps working. Growing a trunk across grid sizes has no
            # meaning for it, so that combination loads the policy only.
            if ALGO == "sac" and grow:
                print("  SAC critic starts fresh: its state cannot be grown "
                      "across grid sizes")
            else:
                _one(second_module, path.replace("policy", "value"))
        return appended

    if os.path.exists(SAVE_PATH):
        # Resuming our own run: load both, so training continues exactly.
        _load(SAVE_PATH)
        print(f"Resuming this run's checkpoint: {SAVE_PATH}")
        if _env_init is not None:
            # This precedence is deliberate, so an interrupted run resumes. But
            # it silently beats an explicit STEERCA_INIT_FROM, and a curriculum
            # stage whose config happens to render the same tag as an earlier
            # run will continue THAT run instead of warm starting -- which cost
            # a 20 hour experiment. Say so loudly; give the stage its own tag.
            what = ("start from scratch" if _env_init.lower() == "none"
                    else f"warm start from {os.path.basename(_env_init)}")
            print(f"  WARNING: STEERCA_INIT_FROM={_env_init} was IGNORED, so this run did "
                  f"NOT {what}.\n"
                  f"  This run's own checkpoint takes precedence. Move "
                  f"{os.path.basename(SAVE_PATH)} aside, or give the run a tag of its "
                  f"own, if that is not what you wanted.")
    elif init_from and os.path.exists(init_from):
        appended = _load(init_from, policy_only=init_policy_only, grow=True)
        what = "policy only" if init_policy_only else "policy and value"
        print(f"Warm starting ({what}) from: {init_from}")
        if appended:
            src_arch = parse_arch_tag(os.path.basename(init_from))
            src_rf = 3 + 4 * sum(src_arch["dilations"]) if src_arch else None
            print(f"  trunk grown {blocks - len(appended)} -> {blocks} blocks "
                  f"(dilations {arch['dilations']}), receptive field "
                  f"{src_rf} -> {backbone_net.receptive_field}")
            print(f"  blocks {appended} start at identity, so the net begins by "
                  f"computing exactly what the source did")
        # Refining a policy that already works is not the same problem as
        # training one, and the same learning rate does not serve both. The
        # scheduler is rebuilt rather than nudged: CosineAnnealingLR reads its
        # base rate from the optimizer when constructed, so simply lowering
        # `lr` on the param groups would be undone on its first step().
        if warm_start_lr_scale != 1.0:
            for group in optim.param_groups:
                group["lr"] = lr * warm_start_lr_scale
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, total_frames // frames_per_batch, 0.0
            )
            print(f"  fine-tuning learning rate {lr:g} -> {lr * warm_start_lr_scale:g}")
    else:
        print("No saved model found, starting from scratch.")

    
    run_config = {
        "arch": "equivariant",
        "arch_tag": arch_tag(arch),
        "channels": channels,
        "blocks": blocks,
        "dilations": list(arch["dilations"]),
        "receptive_field": backbone_net.receptive_field,
        # What the policy can actually see, in cells, from the agent's cell.
        "receptive_radius": (backbone_net.receptive_field - 1) // 2,
        "grid_size": N,
        # The GAE trace in CA ticks: beyond this the advantage is pure critic
        # bootstrap, and the critic's only whole-grid view is the pooled
        # readout. At N=32 the fire takes ~32 ticks to cross, so a short trace
        # here is the likelier failure than the discount.
        "ticks_per_gae_trace": TM / (1 - gamma * lmbda),
        "tc": TC,
        "tm": TM,
        "tm_plus_tc": TM + TC,
        "algo": ALGO,
        "planted": PLANTED_CFG is not None,
        "planted_cfg": PLANTED_CFG.tag() if PLANTED_CFG is not None else "",
        "max_episode_steps": max_episode_steps,
        "hidden": hidden,
        #"hidden1": hidden1,
        #"hidden2": hidden2,
        "tdim": tdim,
        "frames_per_batch": frames_per_batch,
        "total_frames": total_frames,
        "sub_batch_size": sub_batch_size,
        "num_epochs": num_epochs,
        "max_episode_steps": max_episode_steps,
        "clip_epsilon": clip_epsilon,
        "gamma": gamma,
        "lmbda": lmbda,
        "entropy_eps": entropy_eps,
        "reward_scale": reward_scale,
        "sac_alpha_init": sac_alpha_init,
        "sac_target_entropy_weight": sac_target_entropy_weight,
        "lr": lr,
        "seed": SEED,
        "env": "firebreak" if ENV_SPEC is None else ENV_SPEC.name,
    }
    # Metrics go to CSV files under logs/csv/<experiment name>/ by default.
    # STEERCA_WANDB=1 sends them to Weights & Biases instead, to
    # STEERCA_WANDB_PROJECT under STEERCA_WANDB_ENTITY (your default entity if unset).
    USE_WANDB = env_flag("WANDB")
    if USE_WANDB:
        from torchrl.record.loggers.wandb import WandbLogger
        _env_name = "Firebreak" if ENV_SPEC is None else ENV_SPEC.name
        logger = WandbLogger(
            project=setting("WANDB_PROJECT", f"SteerCA_{_env_name}_{N}x{N}"),
            exp_name=experiment_name,
            entity=setting("WANDB_ENTITY"),
            config=run_config,
        )
    else:
        logger = CSVLogger(exp_name=experiment_name, log_dir=setting("LOG_DIR", "logs/csv"))
        logger.log_hparams(run_config)

    logs = defaultdict(list)
    pbar = tqdm(total=total_frames)
    eval_str = ""
    # Checkpoints are selected on TREES SAVED, not on return. Since r9 the reward
    # is shaped -- felling a tree ahead of the fire is priced below what the fire
    # would have charged, so it pays whether or not the cut ends up part of a
    # closed barrier. An agent farming that raises return while saving nothing,
    # and selecting on return would prefer it. Trees saved is the metric the task
    # is scored on and the one the planner is compared against.
    best_saved = -1e6  # Initialize to a large negative value
    for i, tensordict_data in enumerate(collector):
    # we now have a batch of data to work with. Let's learn something from it.
        # Losses summed over every minibatch of this batch, as tensors: calling
        # .item() per minibatch would force ~310 GPU syncs per batch. One sync at
        # the end costs nothing and gives the critic curve that the 32x32
        # collapse could not be diagnosed without.
        loss_sums, n_updates = None, 0
        # Logged in the env's units, before any scaling the learner sees.
        raw_reward = tensordict_data["next", "reward"].mean().item()
        if reward_scale != 1.0:
            tensordict_data["next", "reward"] = tensordict_data["next", "reward"] * reward_scale

        def _step(subdata):
            """One gradient step on an augmented minibatch; returns its losses."""
            # D4 augmentation: rotates/flips grid, agent position and action
            # together (env dynamics are symmetric, advantage is invariant)
            subdata = rr._apply_transform(subdata)
            loss_vals = loss_module(subdata)
            loss_value = sum(loss_vals[k] for k in loss_keys)
            loss_value.backward()
            # this is not strictly mandatory but it's good practice to keep
            # your gradient norm bounded
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()
            return torch.stack([loss_vals[k].detach() for k in loss_keys])

        if ALGO == "sac":
            # Off-policy: the batch joins a buffer that outlives it, and the
            # number of gradient steps is fixed rather than a number of passes
            # over this batch. The target networks are nudged after each step.
            replay_buffer.extend(tensordict_data.reshape(-1))
            if len(replay_buffer) >= sac_batch_size:
                for _ in range(sac_updates_per_batch):
                    batch = _step(replay_buffer.sample())
                    loss_sums = batch if loss_sums is None else loss_sums + batch
                    n_updates += 1
                    target_updater.step()
        else:
            for _ in range(num_epochs):
                # We'll need an "advantage" signal to make PPO work.
                # We re-compute it at each epoch as its value depends on the value
                # network which is updated in the inner loop.
                advantage_module(tensordict_data)
                data_view = tensordict_data.reshape(-1)
                replay_buffer.extend(data_view)
                for _ in range(frames_per_batch // sub_batch_size):
                    batch = _step(replay_buffer.sample(sub_batch_size))
                    loss_sums = batch if loss_sums is None else loss_sums + batch
                    n_updates += 1

        # One sync per batch, logged against the same step as every other metric.
        means = (loss_sums / max(n_updates, 1)).cpu()
        for key, value in zip(loss_keys, means):
            logger.log_scalar(f"loss/{key}", float(value), step=i)
        logger.log_scalar("loss/total", float(means.sum()), step=i)

        logs["reward"].append(raw_reward)
        logger.log_scalar("metrics/avg_reward", logs["reward"][-1], step=i)

        pbar.update(tensordict_data.numel())
        cum_reward_str = (
            f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        )
        #logs["step_count"].append(tensordict_data["step_count"].max().item())
        #stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.6f}"
        
        if i % eval_every == 0:
            if ENV_SPEC is None:
                evaluation = test_policy(
                    eval_env,
                    policy_module,
                    logger,
                    i,
                    k=eval_episodes,
                    # must exceed the longest episode or eval measures truncations
                    max_steps=max_episode_steps + 100,
                    device=device,
                )
            else:
                # A custom CA is selected on its spec's score, or on the return
                # when it defines none. "saved" is the key the selection reads.
                evaluation = evaluate_policy(eval_env.base_env, policy_module, logger, i,
                                             max_steps=max_episode_steps + 100)
                evaluation["saved"] = (evaluation["score"] if evaluation["score"] is not None
                                       else evaluation["return"])
            # Both are printed: return is what the agent optimises, trees saved
            # is what it is judged on, and the two coming apart is the farming
            # signature rather than progress.
            eval_str = (f"eval {'saved' if ENV_SPEC is None else 'score'} "
                        f"{100 * evaluation['saved']:5.1f}% "
                        f"(return {evaluation['return']: 4.4f})")
            if "oracle_fraction" in evaluation:
                eval_str += f" {100 * evaluation['oracle_fraction']:5.1f}% of oracle"
            if evaluation["saved"] > best_saved:
                best_saved = evaluation["saved"]

                # Save the model if it has improved
                file = f"{MODELS}/{experiment_name}_policy.pth"
                torch.save(policy_module.state_dict(), file)
                file = f"{MODELS}/{experiment_name}_value.pth"
                torch.save(second_module.state_dict(), file)

        pbar.set_description(", ".join([eval_str, cum_reward_str, lr_str]))

        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        scheduler.step()

    # The final policy, beside the best one. The checkpoint above is saved on
    # the best trees-saved evaluation, so a run that never beats 0.0% keeps the
    # one from its FIRST evaluation, one batch into training -- probing that
    # file says nothing about where the run ended, which is how the paper once
    # reported batch-1 networks as final policies. Named `..._final_policy.pth`,
    # so the resume path above never picks it up.
    torch.save(policy_module.state_dict(), f"{MODELS}/{experiment_name}_final_policy.pth")
    torch.save(second_module.state_dict(), f"{MODELS}/{experiment_name}_final_value.pth")

    # Closed here rather than by tqdm's __del__ at interpreter exit, which can
    # segfault once the modules it formats with are already torn down.
    pbar.close()

    # The last batch's losses, in the same `key value` form as W&B's run
    # summary, so paper/collect.py reads the final policy entropy from any log.
    print()
    for key, value in zip(loss_keys, means):
        print(f"summary: loss/{key} {float(value):.6g}")

    collector.shutdown()
    if USE_WANDB:
        import wandb
        wandb.finish()
