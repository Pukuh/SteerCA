"""Checks on ForestFireEquivariantNet: shape handling, N-independence,
cross-grid-size checkpoint transfer, translation equivariance in the interior,
and that it survives the vmap torchrl applies inside ClipPPOLoss.

Run from the repo root:  python tests/test_equivariant_arch.py
"""
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from steerca.archs import (
    ForestFireEquivariantNet,
    ForestFirePolicyNet,
    ForestFireValueNet,
)

HIDDEN = 128
KW = dict(channels=32, blocks=4, hidden=HIDDEN, tdim=16)


def _inputs(batch_shape, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    grid = torch.zeros((*batch_shape, 3, n, n), dtype=torch.bool)
    cls = torch.randint(0, 3, (*batch_shape, n, n), generator=g)
    grid.scatter_(-3, cls.unsqueeze(-3), True)
    t = torch.rand(batch_shape, generator=g)
    pos = torch.randint(0, n, (*batch_shape, 2), generator=g)
    return grid, t, pos


def test_shapes():
    """Leading dims (), (B,) and (B, T) must all pass through intact."""
    net = ForestFireEquivariantNet(**KW)
    for batch_shape in [(), (5,), (4, 7)]:
        out = net(*_inputs(batch_shape, 16))
        assert out.shape == (*batch_shape, HIDDEN), (batch_shape, out.shape)
    print("shapes ................ ok")


def test_param_count_independent_of_n():
    """The whole point: parameters must not grow with the grid size."""
    counts = {}
    for n in (8, 16, 32):
        net = ForestFireEquivariantNet(grid_size=n, **KW)
        counts[n] = sum(p.numel() for p in net.parameters())
    assert len(set(counts.values())) == 1, counts
    print(f"N-independence ........ ok  ({counts[8]:,} params at N=8, 16 and 32)")


def test_checkpoint_transfers_across_grid_sizes():
    """An 8x8 checkpoint must load and run at 16x16 (pretrain small, scale up)."""
    small = ForestFirePolicyNet(
        backbone_net=ForestFireEquivariantNet(grid_size=8, **KW),
        n_actions=9, hidden=HIDDEN,
    )
    big = ForestFirePolicyNet(
        backbone_net=ForestFireEquivariantNet(grid_size=16, **KW),
        n_actions=9, hidden=HIDDEN,
    )
    missing, unexpected = big.load_state_dict(small.state_dict(), strict=True), None
    with torch.no_grad():
        logits = big(*_inputs((3,), 16))
    assert logits.shape == (3, 9), logits.shape
    print("checkpoint transfer ... ok  (8x8 weights run at 16x16)")


def test_translation_equivariance_in_interior():
    """Shifting the world and the agent together must not change the output.

    Only exact away from the boundary, so this uses a grid much larger than the
    receptive field with all content confined to the middle.
    """
    net = ForestFireEquivariantNet(**KW).eval()
    n = 48
    rf = net.receptive_field
    assert n > 2 * rf + 8, "grid must clear the receptive field on both sides"

    def world(shift):
        grid = torch.zeros((1, 3, n, n), dtype=torch.bool)
        grid[0, 1] = True                       # all trees
        cx = cy = n // 2 + shift
        grid[0, 1, cx - 2:cx + 3, cy - 2:cy + 3] = False
        grid[0, 2, cx - 2:cx + 3, cy - 2:cy + 3] = True   # fire blob
        pos = torch.tensor([[cx + 4, cy + 3]])
        return grid, torch.tensor([0.5]), pos

    with torch.no_grad():
        a = net(*world(0))
        b = net(*world(3))
    err = (a - b).abs().max().item()
    assert err < 1e-4, f"max deviation {err}"
    print(f"equivariance .......... ok  (max deviation {err:.2e} under a 3-cell shift)")


def test_survives_ppo_loss():
    """Integration: GAE + ClipPPOLoss vmap the actor/critic; the gather and the
    F.one_hot plane must both stay vmap-safe."""
    from tensordict import TensorDict
    from tensordict.nn import TensorDictModule
    from torch.distributions import Categorical
    from torchrl.modules import ProbabilisticActor, ValueOperator
    from torchrl.objectives import ClipPPOLoss
    from torchrl.objectives.value import GAE

    n, b = 16, 6
    backbone = ForestFireEquivariantNet(grid_size=n, **KW)
    policy = ProbabilisticActor(
        TensorDictModule(
            ForestFirePolicyNet(backbone_net=backbone, n_actions=9, hidden=HIDDEN),
            in_keys=["grid", "t", "agent_state"], out_keys=["logits"],
        ),
        distribution_class=Categorical, in_keys=["logits"], out_keys=["action"],
        return_log_prob=True,
    )
    value = ValueOperator(
        module=ForestFireValueNet(backbone_net=backbone, hidden=HIDDEN),
        in_keys=["grid", "t", "agent_state"],
    )

    grid, t, pos = _inputs((b,), n)
    ngrid, nt, npos = _inputs((b,), n, seed=1)
    td = TensorDict({
        "grid": grid, "t": t, "agent_state": pos,
        "next": TensorDict({
            "grid": ngrid, "t": nt, "agent_state": npos,
            "reward": torch.randn(b, 1),
            "done": torch.zeros(b, 1, dtype=torch.bool),
            "terminated": torch.zeros(b, 1, dtype=torch.bool),
        }, [b]),
    }, [b])
    with torch.no_grad():  # the collector stores log-probs detached
        policy(td)

    GAE(gamma=0.999, lmbda=0.99, value_network=value, average_gae=True)(td)
    loss = ClipPPOLoss(actor_network=policy, critic_network=value,
                       entropy_bonus=True, entropy_coeff=5e-3, critic_coeff=0.5,
                       loss_critic_type="smooth_l1")(td)
    total = loss["loss_objective"] + loss["loss_critic"] + loss["loss_entropy"]
    total.backward()

    grads = [p.grad for p in backbone.parameters() if p.grad is not None]
    assert grads, "no gradient reached the backbone"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"
    print(f"ppo loss + backward ... ok  (loss {total.item():.4f}, "
          f"{len(grads)} backbone tensors received gradient)")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_shapes()
    test_param_count_independent_of_n()
    test_checkpoint_transfers_across_grid_sizes()
    test_translation_equivariance_in_interior()
    test_survives_ppo_loss()
    print("\nall checks passed")
