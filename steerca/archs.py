"""Policy and value networks.

``ForestFireEquivariantNet`` is the backbone every published result uses: fully
convolutional, read out at the agent's cell plus global mean/max pooling, with
no parameter that depends on the grid size, so a checkpoint trained at one size
runs at another. ``arch_for_grid(n)`` picks its depth and dilations, and
``parse_arch_tag`` rebuilds it from a checkpoint's file name. Despite the name it
is not specific to the forest fire: ``in_states`` sets the number of grid
planes, which is all another CA needs.

The ``ForestFireCommonNet*`` classes are legacy: they exist only so the early
(pre-r5) checkpoints still load, and are not used for training.
"""
from torch import nn
import re
import torch
import torch.nn.functional as F

N = 8  # Size of the grid

# Create a common backbone for the policy and value networks
class ForestFireCommonNetCNN_Std(nn.Module):
    def __init__(self, grid_size=N, hidden=16, device='cpu'):
        super().__init__()
        self.grid_size = grid_size
        self.edim = 16
        self.device = device

        self.conv1 = nn.Conv2d(4, hidden, kernel_size=3, stride=1, padding=1, device=self.device)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(hidden, hidden // 2, kernel_size=3, stride=1, padding=1, device=self.device)
        self.conv3 = nn.Conv2d(hidden // 2, hidden // 4, kernel_size=3, stride=1, padding=1, device=self.device)
        self.flatten = nn.Flatten(start_dim=-3)
        self.fc1 = nn.Linear((grid_size // 1) * (grid_size // 1) * hidden // 4 + self.edim, hidden*20, device=self.device)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden*20, 128, device=self.device)
        
        self.fc3 = nn.Linear(3, self.edim, device=self.device)  # For logits output

    def forward(self, grid, t, agent_state):
        agent_pos = torch.zeros_like(grid[..., 0, :,:])  # Initialize agent position tensor
        agent_pos[..., agent_state[..., 0], agent_state[..., 1]] = 1  # One-hot encoding of agent position
        grid = torch.cat([grid, agent_pos.unsqueeze(-3)], dim=-3)  # Add agent position to the grid

        grid = grid.float()

        grid = self.conv1(grid)  # (batch, hidden, n, n)
        grid = self.relu(grid)  # Uncomment if you want to apply ReLU after conv1
        #grid = self.pool(grid)  # (batch, hidden, n/2, n/2)
        grid = self.conv2(grid)  # (batch, hidden, n/2, n/2)
        grid = self.relu(grid)
        grid = self.conv3(grid)
        grid = self.relu(grid)
        grid_flat = self.flatten(grid)  # (batch, hidden * n/2 * n/2)
        #t = t.view(batch, 1) if t.ndim == 1 else t
        #pos = agent_state.view(batch, 2) if agent_state.ndim == 1 else agent_state  # (batch, 2)

        tpos = torch.cat([t.unsqueeze(-1).float(), agent_state.float()/self.grid_size], dim=-1)
        #tpos = t.unsqueeze(-1).float()  # (batch, 1)
        tpos = self.fc3(tpos)

        features = torch.cat([grid_flat, tpos], dim=-1)  # (batch, hidden * n/2 * n/2 + 3)
        out = self.fc1(features)  # (batch, hidden)
        out = self.relu(out)
        out = self.fc2(out)

        out = self.relu(out)
        return out

class ForestFireCommonNetCNN(nn.Module):
    def __init__(self, grid_size=N, filters_1=32, filters_2=16, hidden=256, tdim=16, device='cpu'):
        super().__init__()
        self.grid_size = grid_size
        self.edim = tdim
        self.device = device

        self.conv1 = nn.Conv2d(4, filters_1, kernel_size=3, stride=1, padding=1, device=self.device)
        self.conv2 = nn.Conv2d(filters_1, filters_1, kernel_size=1, stride=1, padding=0, device=self.device)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(filters_1, filters_2, kernel_size=3, stride=1, padding=1, device=self.device)

        
        self.flatten = nn.Flatten(start_dim=-3)
        self.fc1 = nn.Linear((grid_size//1) * (grid_size//1) * filters_2 + self.edim, hidden, device=self.device)
        self.relu = nn.ReLU()
        

        self.fc_extra = nn.Linear(1, self.edim, device=self.device)  # For logits output

    def forward(self, grid, t, agent_state):
        batch_shape = grid.shape[:-3]  # e.g. (), (B,), or (N_envs, T)

        # One-hot encode agent position as 4th channel.
        # F.one_hot keeps agent_pos derived from vmapped agent_state (vmap-safe).
        lin_idx = (agent_state[..., 0] * self.grid_size + agent_state[..., 1]).long()
        agent_pos = F.one_hot(lin_idx, self.grid_size * self.grid_size).to(dtype=grid.dtype)
        agent_pos = agent_pos.view(*batch_shape, self.grid_size, self.grid_size)
        grid = torch.cat([grid, agent_pos.unsqueeze(-3)], dim=-3).float()

        # Conv2d requires 4D input: flatten all leading dims into one batch dim
        grid = grid.view(-1, 4, self.grid_size, self.grid_size)
        grid = self.conv1(grid)
        grid = self.relu(grid)
        grid = self.conv2(grid)
        grid = self.relu(grid)
        grid = self.conv3(grid)
        grid = self.relu(grid)
        grid_flat = self.flatten(grid)  # (-1, filters_2 * N * N)

        tpos = self.fc_extra(t.reshape(-1, 1).float())  # (-1, edim)

        features = torch.cat([grid_flat, tpos], dim=-1)
        out = self.fc1(features)
        out = self.relu(out)

        return out.view(*batch_shape, out.shape[-1])

#--------------------------------------------------------------------------------

class ForestFireResidualBlock(nn.Module):
    """3x3 -> 3x3 residual block, adding 4*dilation cells to the receptive field.

    GroupNorm rather than BatchNorm: PPO's minibatches are small (64) and highly
    correlated within a trajectory, so batch statistics are a poor estimator and
    they differ between collection and the replayed update.

    `dilation` spaces the kernel taps out, which buys receptive field without
    extra parameters or extra depth. `padding=dilation` keeps the spatial size,
    so a dilated block is a drop-in for a dense one. Note the parameter shapes do
    not depend on it at all: a block's weights load into a block of any dilation.
    """

    def __init__(self, channels, groups=8, dilation=1, device='cpu'):
        super().__init__()
        g = min(groups, channels)
        d = int(dilation)
        self.dilation = d
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=d, dilation=d, bias=False, device=device)
        self.norm1 = nn.GroupNorm(g, channels, device=device)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=d, dilation=d, bias=False, device=device)
        self.norm2 = nn.GroupNorm(g, channels, device=device)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.relu(x + h)


class ForestFireEquivariantNet(nn.Module):
    """Fully-convolutional backbone whose parameter count is independent of N.

    Replaces the flatten -> Linear(N*N*filters, hidden) bottleneck of
    ``ForestFireCommonNetCNN``, which held >90% of that network's parameters in a
    single position-specific layer and so had to learn every (fire pattern, agent
    position) pair independently -- quadrupling in size from N=8 to N=16 with no
    extra data.

    Structure:
      stem 3x3 -> `blocks` residual blocks -> three O(1) readouts -> MLP
      receptive field = 3 + 4 * sum(dilations)  (27 with the default six dense
      blocks, a radius of 13)

    Two things set how far the policy can see, and they differ by a factor of two.
    The gather at the agent's cell reaches one receptive radius, 13 by default.
    But the agent's position is an *input plane*, so a unit anywhere can encode a
    relation between the agent and a fire that both fall inside its own window --
    "agent 13 west of me, fire 13 east" -- and the global max pool surfaces that
    unit wherever it fires. Since the relation is expressed relative to the agent,
    knowing it holds somewhere is enough. That gives a relational range of about
    RF - 1 = 26, which is why 16x16 works comfortably at the default.

    32x32, where the agent and the fire can be 31 apart, is past even that, which
    is what `arch_for_grid` widens the field for. Note the pooled path is lossy in
    a way the gather is not -- one scalar per feature over the whole grid reports
    presence, not how much or how many -- so it is a weaker channel, not a
    substitute for reaching the fire directly.

    The readouts are what make the size independence work. All 9 actions act at
    the agent's cell, so the policy needs the trunk's feature vector *at that
    cell*; that is taken by an exact gather, which costs no parameters and no
    knowledge of N. The value head needs the whole grid, taken by global mean and
    max pooling, which are likewise parameter-free and N-independent.

    Because every parameter is either a shared conv kernel or acts on a
    fixed-width readout, a checkpoint trained at one grid size loads and runs at
    another -- so 8x8 can be used to pretrain 16x16.

    A caveat worth being precise about: translation equivariance is exact only
    away from the boundary. With zero padding and a receptive field of 27, every
    cell of a 16x16 grid sees the edge, so what this buys in practice is weight
    sharing and an N-independent parameterization rather than a literal
    invariance. That is still the property that fixes the scaling problem, and
    the boundary genuinely matters here (the agent clamps against it and fire
    cannot spread past it), so it is information the trunk should have.
    """

    def __init__(self, grid_size=None, channels=96, blocks=6, hidden=512, tdim=64,
                 dilations=None, device='cpu', in_states=3):
        super().__init__()
        # grid_size is accepted for signature compatibility with the other
        # backbones and is deliberately unused: N is read from the input tensor.
        self.grid_size = grid_size
        self.channels = channels
        self.tdim = tdim
        self.device = device
        # `dilations` is the authority when given, and `blocks` follows from it.
        # None means every block is dense, which is bit-identical to the original
        # net -- dilation=1, padding=1 is what these convs already did.
        self.dilations = (1,) * blocks if dilations is None else tuple(int(d) for d in dilations)
        self.blocks_count = len(self.dilations)
        self.receptive_field = 3 + 4 * sum(self.dilations)

        # in_states grid planes (Firebreak: DIRT / TREE / FIRE) + agent position.
        # Other CAs pass their own plane count; the rest of the net is unchanged.
        self.in_states = in_states
        self.stem = nn.Conv2d(in_states + 1, channels, 3, padding=1, bias=False, device=device)
        self.stem_norm = nn.GroupNorm(min(8, channels), channels, device=device)
        self.blocks = nn.ModuleList(
            [ForestFireResidualBlock(channels, dilation=d, device=device)
             for d in self.dilations]
        )
        self.relu = nn.ReLU()

        self.fc_time = nn.Linear(1, tdim, device=device)
        # local (gather) + global mean + global max + time
        self.fc_out = nn.Linear(3 * channels + tdim, hidden, device=device)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero the last norm of each block so the trunk starts as the identity
        # and signal reaches the readouts undistorted at step 0.
        for block in self.blocks:
            nn.init.zeros_(block.norm2.weight)

    def forward(self, grid, t, agent_state):
        batch_shape = grid.shape[:-3]  # e.g. (), (B,), or (N_envs, T)
        c, n = grid.shape[-3], grid.shape[-1]

        # Agent position as a one-hot plane. F.one_hot (rather than index
        # assignment) keeps this safe under the vmap torchrl applies to the loss.
        # The clamp is defensive: the D4 augmentation remaps positions by
        # rounding, so guard against a rounding step landing off-grid.
        pos = agent_state.long()
        lin_idx = pos[..., 0].clamp(0, n - 1) * n + pos[..., 1].clamp(0, n - 1)
        agent_plane = F.one_hot(lin_idx, n * n).to(dtype=torch.float32)
        agent_plane = agent_plane.view(-1, 1, n, n)

        # Conv2d needs 4D input: fold all leading dims into one batch dim
        x = grid.reshape(-1, c, n, n).float()
        x = torch.cat([x, agent_plane], dim=-3)

        x = self.relu(self.stem_norm(self.stem(x)))
        for block in self.blocks:
            x = block(x)

        # Readout at the agent's cell. Multiplying by the one-hot plane and
        # summing is an exact gather that stays vmap-safe and differentiable.
        local = (x * agent_plane).sum(dim=(-2, -1))  # (-1, C)
        # Translation-invariant global context, mainly for the value head.
        gmean = x.mean(dim=(-2, -1))                 # (-1, C)
        gmax = x.amax(dim=(-2, -1))                  # (-1, C)

        temb = self.fc_time(t.reshape(-1, 1).float())  # (-1, tdim)

        features = torch.cat([local, gmean, gmax, temb], dim=-1)
        out = self.relu(self.fc_out(features))

        return out.view(*batch_shape, out.shape[-1])


# --- Choosing the backbone for a grid size --------------------------------------

BASE_ARCH = dict(channels=96, blocks=6, hidden=512, tdim=64)

# At or below this grid size the architecture is frozen at BASE_ARCH. It is not a
# claim that BASE_ARCH is right there -- at N=16 the readout's radius of 13 is
# short of the 15 it would need -- but every checkpoint on disk was trained with
# it and every experiment name encodes it, so changing it would orphan them all.
FROZEN_UPTO = 16


def arch_for_grid(n):
    """Backbone kwargs for an n x n grid.

    Above FROZEN_UPTO, append power-of-two dilated blocks until the readout at
    the agent's cell can see the far corner. The readout is a gather at that
    cell, so what the policy sees is a window of `receptive_field` centred on it;
    from a corner the far corner is n-1 away, hence

        radius = (receptive_field - 1) // 2 = 1 + 2*sum(dilations) >= n - 1

    Dilating rather than stacking more dense blocks is what makes this cheap: at
    n=32 a tail of (2, 4, 8) reaches radius 41 with 9 blocks, where 15 dense
    blocks reach only 31 and measured 1.7x the time per sample. Keeping the first
    six blocks dense also means a smaller grid's checkpoint keeps the meaning of
    its local features instead of having them reinterpreted at a wider spacing.
    """
    dilations = [1] * BASE_ARCH["blocks"]
    if n > FROZEN_UPTO:
        d = 2
        while 1 + 2 * sum(dilations) < n - 1:
            dilations.append(d)
            d *= 2
        # A dilation of d taps +-d cells away, so past n/2 the taps land in zero
        # padding for most of the grid and the block stops carrying information.
        assert 2 * max(dilations) < n, f"dilation {max(dilations)} too wide for a {n}x{n} grid"
    return dict(BASE_ARCH, blocks=len(dilations), dilations=tuple(dilations))


# `c96_b6_512_64`, or `c96_b9d2-4-8_512_64` when there is a dilated tail. `b` is
# always the TOTAL block count, so it keeps meaning what it means in every name
# already on disk; the `d` group lists only the non-unit tail, and the dense
# blocks before it are whatever is left over.
#
# Anchored on `{algo}_equiv_` and `_1hot` so the search cannot wander into the
# cost fields, and the `-` separator is not cosmetic: run the digits together and
# `b10d24816` reads equally as (2,4,8,16) and (2,4,8,1,6). The algorithm prefix is
# whatever STEERCA_ALGO allows; anchoring on `ppo_` alone once made every A2C and
# SAC name parse to None, which quietly disabled the dilation guard on their
# warm starts.
ARCH_TAG_RE = re.compile(r"(?:ppo|a2c|sac)_equiv_c(\d+)_b(\d+)(?:d([\d-]+))?_(\d+)_(\d+)_1hot")


def arch_tag(arch):
    """The filename segment naming an architecture, e.g. `c96_b9d2-4-8_512_64`."""
    dilations = tuple(arch.get("dilations") or (1,) * arch["blocks"])
    tail = tuple(d for d in dilations if d != 1)
    if dilations != (1,) * (len(dilations) - len(tail)) + tail:
        raise ValueError(f"non-unit dilations must form a suffix, got {dilations}")
    suffix = ("d" + "-".join(str(d) for d in tail)) if tail else ""
    return f"c{arch['channels']}_b{len(dilations)}{suffix}_{arch['hidden']}_{arch['tdim']}"


def parse_arch_tag(name):
    """Recover the arch kwargs from a checkpoint or experiment name, or None.

    Names without a dilation segment parse to an all-dense trunk, so every
    checkpoint trained before dilations existed round-trips unchanged.

    This -- not the grid size being evaluated -- is what every consumer should
    build from. The architecture is a property of the checkpoint: evaluating a
    16x16 policy at 32x32 must still build the trunk the 16x16 run trained.
    """
    m = ARCH_TAG_RE.search(name)
    if m is None:
        return None
    channels, blocks, tail, hidden, tdim = m.groups()
    tail = [int(d) for d in tail.split("-")] if tail else []
    dilations = [1] * (int(blocks) - len(tail)) + tail
    return dict(channels=int(channels), blocks=len(dilations), hidden=int(hidden),
                tdim=int(tdim), dilations=tuple(dilations))


def load_grown_state_dict(module, state_dict, what="checkpoint", src_dilations=None,
                          dst_dilations=None):
    """Load a state dict that may be missing blocks appended since it was saved.

    A freshly initialised ForestFireResidualBlock is exactly the identity -- the
    residual is `relu(x + norm2(...))`, `_init_weights` zeroes every norm2 weight,
    and block inputs are always post-ReLU -- so a trunk grown by appending blocks
    starts out computing exactly what the smaller trunk computed, then learns the
    long-range blocks from there. That makes a 16x16 checkpoint an exact warm
    start for a 32x32 net.

    `strict=False` is what permits that, and the assertions here are what stop it
    from also hiding a genuine mismatch: the only keys allowed to be missing are
    whole appended blocks, and nothing may be left over. (Shape mismatches need
    no guard -- load_state_dict raises on those even when strict is False.)

    Pass `src_dilations` / `dst_dilations` to catch the one mismatch a state dict
    cannot show. Dilation is not a parameter, so a trunk trained dense loads
    silently into a dilated one and every kernel is quietly reinterpreted at a
    spacing it never saw. The target's dilations must start with the source's.
    """
    if src_dilations is not None and dst_dilations is not None:
        src_d, dst_d = tuple(src_dilations), tuple(dst_dilations)
        if dst_d[:len(src_d)] != src_d:
            raise RuntimeError(
                f"{what} was trained with dilations {src_d}, which is not a prefix of this "
                f"model's {dst_d}: its kernels would be reinterpreted at a different spacing"
            )
    result = module.load_state_dict(state_dict, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            f"{what} has {len(result.unexpected_keys)} key(s) this model does not: "
            f"{sorted(result.unexpected_keys)[:5]}"
        )
    if not result.missing_keys:
        return []

    grown = sorted({int(m.group(1)) for k in result.missing_keys
                    if (m := re.search(r"\bblocks\.(\d+)\.", k))})
    ckpt_blocks = min(grown) if grown else 0
    accounted = {k for k in result.missing_keys
                 if (m := re.search(r"\bblocks\.(\d+)\.", k)) and int(m.group(1)) >= ckpt_blocks}
    if set(result.missing_keys) != accounted or grown != list(range(ckpt_blocks, ckpt_blocks + len(grown))):
        raise RuntimeError(
            f"{what} is missing keys that are not an appended block tail: "
            f"{sorted(set(result.missing_keys) - accounted)[:5]}"
        )

    # The appended blocks are only the identity while their norm2 weight is zero,
    # which is what makes the grown net function-preserving. Check it rather than
    # trusting that nothing has touched _init_weights.
    live = dict(module.state_dict())
    for k in accounted:
        if k.endswith("norm2.weight") and live[k].abs().max() > 0:
            raise RuntimeError(
                f"appended block {k} is not at identity init, so growing the trunk would "
                f"change what the {what} computes"
            )
    return grown


#--------------------------------------------------------------------------------

class ForestFireCommonNetPlain(nn.Module):
    def __init__(self, grid_size=N, hidden1=256, hidden2=256, hidden3=128, tdim=16, device='cpu'):
        super().__init__()
        self.grid_size = grid_size
        self.tdim = tdim
        self.device = device

        
        self.flatten = nn.Flatten(start_dim=-3)
        flatten_size = grid_size * grid_size * 4  # Assuming 4 channels in the grid
        self.fc1 = nn.Linear(flatten_size + self.tdim, hidden1, device=self.device)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2, device=self.device)
        self.fc3 = nn.Linear(hidden2, hidden3, device=self.device)

        self.fc_time = nn.Linear(1, self.tdim, device=self.device)  # Embedding of time

    def forward(self, grid, t, agent_state):
        batch_shape = grid.shape[:-3]  # e.g. (), (B,), or (N_envs, T)

        # One-hot encode agent position as 4th channel.
        # F.one_hot keeps agent_pos derived from vmapped agent_state (vmap-safe).
        lin_idx = (agent_state[..., 0] * self.grid_size + agent_state[..., 1]).long()
        agent_pos = F.one_hot(lin_idx, self.grid_size * self.grid_size).to(dtype=grid.dtype)
        agent_pos = agent_pos.view(*batch_shape, self.grid_size, self.grid_size)
        grid = torch.cat([grid, agent_pos.unsqueeze(-3)], dim=-3).float()

        # nn.Flatten(start_dim=-3) and nn.Linear handle arbitrary leading dims natively
        grid_flat = self.flatten(grid)

        tpos = self.fc_time(t.unsqueeze(-1).float())

        features = torch.cat([grid_flat, tpos], dim=-1)
        out = self.fc1(features)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        return out

# Create the policy and value networks using the common backbone
class ForestFirePolicyNet(nn.Module):
    def __init__(self, backbone_net, n_actions=9, hidden=128, device='cpu'):
        super().__init__()
        self.device = device
        self.backbone_net = backbone_net
        self.fc4 = nn.Linear(hidden, n_actions, device=self.device)  # For logits output

    def forward(self, grid, t, agent_state):
        out = self.backbone_net(grid, t, agent_state)
        out = self.fc4(out)  # (batch, n_actions)
        return out

class ForestFireValueNet(nn.Module):
    def __init__(self, backbone_net, hidden=128, device='cpu'):
        super().__init__()
        self.device = device
        self.backbone_net = backbone_net
        self.fc4 = nn.Linear(hidden, 1, device=self.device)  # For value output

    def forward(self, grid, t, agent_state):
        out = self.backbone_net(grid, t, agent_state)
        out = self.fc4(out)  # (batch, 1)
        return out
