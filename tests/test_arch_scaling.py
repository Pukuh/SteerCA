"""Checks on the grid-size-dependent architecture: that the configuration used by
every run so far is frozen, that the new dilation machinery is inert at its
defaults, and that growing a trunk to a bigger grid preserves what it computes.

The contract these pin is narrow and worth stating plainly: adding an
architecture that depends on N must not change the architecture at N <= 16, or
every checkpoint on disk becomes unloadable and every experiment name changes.

Run from the repo root:  python tests/test_arch_scaling.py
"""
import glob
import os
import re
import sys

import torch

# The published checkpoints, when fetched (tools/fetch_artifacts.py).
MODELS = os.path.join(os.environ.get("STEERCA_ARTIFACTS",
                                     os.path.join(os.path.dirname(__file__), "..", "artifacts")),
                      "models")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from steerca.archs import (
    ForestFireEquivariantNet,
    ForestFirePolicyNet,
    arch_for_grid,
    arch_tag,
    load_grown_state_dict,
    parse_arch_tag,
)

# What every equivariant run to date was trained with, written out as literals so
# this file does not read the answer from the code it is checking.
TODAY = dict(channels=96, blocks=6, hidden=512, tdim=64)
TODAY_TAG = "c96_b6_512_64"


def _inputs(batch, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    grid = torch.zeros((batch, 3, n, n), dtype=torch.bool)
    grid.scatter_(-3, torch.randint(0, 3, (batch, n, n), generator=g).unsqueeze(-3), True)
    return grid, torch.rand(batch, generator=g), torch.randint(0, n, (batch, 2), generator=g)


def test_small_grids_are_frozen():
    """At every size trained so far, the architecture must be exactly today's."""
    for n in range(1, 17):
        cfg = arch_for_grid(n)
        assert cfg["dilations"] == (1,) * 6, (n, cfg)
        assert {k: cfg[k] for k in TODAY} == TODAY, (n, cfg)
        assert arch_tag(cfg) == TODAY_TAG, (n, arch_tag(cfg))
    print("small grids frozen ....... ok  (n<=16 -> " + TODAY_TAG + ")")


def test_built_net_matches_todays_literals():
    """arch_for_grid(16) must build the same module as the old hardcoded call.

    Same parameter names and same count -- which together are what decide whether
    a state dict saved before this change still loads.
    """
    old = ForestFireEquivariantNet(grid_size=16, **TODAY)
    new = ForestFireEquivariantNet(grid_size=16, **arch_for_grid(16))
    assert set(old.state_dict()) == set(new.state_dict())
    n_old = sum(p.numel() for p in old.parameters())
    assert n_old == sum(p.numel() for p in new.parameters())
    assert old.receptive_field == new.receptive_field == 27
    print(f"matches old literals ..... ok  ({n_old:,} params, receptive field 27)")


def test_dilations_default_is_inert():
    """dilations=None and dilations=(1,)*blocks must be the same net, bit for bit.

    Same seed either way: this also pins that the new argument consumed no extra
    randomness during init, which would silently change every future run.
    """
    kw = dict(channels=32, blocks=4, hidden=64, tdim=16)
    torch.manual_seed(0)
    a = ForestFireEquivariantNet(**kw)
    torch.manual_seed(0)
    b = ForestFireEquivariantNet(**kw, dilations=(1, 1, 1, 1))
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k]), k
    with torch.no_grad():
        x = _inputs(3, 12)
        assert torch.equal(a(*x), b(*x))
    print("dilations default inert .. ok")


def test_receptive_field_is_measured_not_asserted():
    """Check the 3 + 4*sum(dilations) formula against an actual gradient trace.

    Probe the convolutions the module actually built, which is what the formula
    describes, and nothing else. Two other paths reach further and would swamp
    the measurement: GroupNorm takes its statistics over the whole spatial
    extent, and the readout concatenates global pooling -- so a gradient trace
    through the full module reports the grid size at any depth. (A fresh module
    measures 3, in the other direction: every block starts at identity.)

    What `receptive_field` therefore means is the span of grid a trunk feature
    can be shaped by, which is the quantity that decides whether the gather at
    the agent's cell can relate the agent to a fire some distance away.
    """
    for dilations, expected in [((1,) * 4, 19), ((1, 1, 2, 4), 35)]:
        net = ForestFireEquivariantNet(channels=8, blocks=len(dilations), hidden=8,
                                       tdim=4, dilations=dilations).eval()
        assert net.receptive_field == expected, (dilations, net.receptive_field)
        n = 2 * expected + 9
        x = torch.zeros((1, 4, n, n), requires_grad=True)
        h = net.stem(x)
        for block in net.blocks:
            h = block.conv2(block.conv1(h))
        h[0, :, n // 2, n // 2].sum().backward()
        touched = (x.grad.abs().sum((0, 1)) > 0).nonzero()
        span = int(touched[:, 0].max() - touched[:, 0].min()) + 1
        assert span == expected, (dilations, span, expected)
    print("receptive field measured . ok  (19 and 35 cells, matching the formula)")


def test_grid_sizes_get_a_field_that_covers_them():
    for n in (8, 32, 64):
        cfg = arch_for_grid(n)
        radius = 1 + 2 * sum(cfg["dilations"])
        assert radius >= n - 1, (n, radius)
    # 16 is the documented exception: frozen at radius 13 where it would want 15.
    assert 1 + 2 * sum(arch_for_grid(16)["dilations"]) == 13
    print("field covers the grid .... ok  (n=16 frozen short, by design)")


def test_tag_roundtrips():
    for n in (8, 16, 32, 64):
        cfg = arch_for_grid(n)
        name = f"ppo_equiv_{arch_tag(cfg)}_1hot_{n}x{n}_t0.2_0.2_r7_policy.pth"
        assert parse_arch_tag(name) == cfg, (n, parse_arch_tag(name), cfg)
    # A dilation of 10+ must not be ambiguous with the digits that follow it.
    wide = dict(channels=96, blocks=10, hidden=512, tdim=64,
                dilations=(1,) * 6 + (2, 4, 8, 16))
    assert parse_arch_tag(f"ppo_equiv_{arch_tag(wide)}_1hot_") == wide
    # Every learner the training script names a checkpoint after, including the
    # learner-settings infix and the `_final` checkpoint.
    for algo in ("ppo", "a2c", "sac"):
        cfg = arch_for_grid(32)
        name = (f"{algo}_equiv_{arch_tag(cfg)}_1hot_32x32_t0.2_0.2_cut0.5"
                f"_rs1024_a0.05-te0.2_r10_sweep2m_final_policy.pth")
        assert parse_arch_tag(name) == cfg, (algo, parse_arch_tag(name))
    assert parse_arch_tag("dqn_equiv_c96_b6_512_64_1hot_8x8") is None
    print("tag round-trips .......... ok  (" + arch_tag(arch_for_grid(32)) + ")")


def test_every_checkpoint_on_disk_still_parses():
    """Every name in models/ must round-trip, and small grids must be frozen.

    Two separate claims, and only the second is about the frozen tier. Every
    checkpoint's name has to parse and render back identically, or the file
    cannot be loaded into the network that produced it. Separately, any
    checkpoint at a grid size of 16 or below must carry the base config -- that
    is the promise that adding an N-dependent architecture did not orphan the
    runs that came before it. Larger grids legitimately carry a dilated tail.
    """
    frozen, wide = 0, 0
    for path in glob.glob(os.path.join(MODELS, "*_equiv_*_policy.pth")):
        name = os.path.basename(path)
        cfg = parse_arch_tag(name)
        assert cfg is not None, path
        assert arch_tag(cfg) in name, (path, arch_tag(cfg))
        grid = re.search(r"_(\d+)x\d+", name)
        if grid and int(grid.group(1)) <= 16:
            assert cfg["dilations"] == (1,) * 6, (path, cfg)
            assert arch_tag(cfg) == TODAY_TAG, (path, arch_tag(cfg))
            frozen += 1
        else:
            wide += 1
    total = frozen + wide
    print(f"checkpoint names parse ... ok  ({total} files: {frozen} frozen at "
          f"{TODAY_TAG}, {wide} above n=16)"
          if total else "checkpoint names parse ... skipped (no artifacts/models on disk)")


def test_real_checkpoint_still_loads_strict():
    """The strongest regression guard available: a real artifact, loaded strict."""
    hits = sorted(glob.glob(os.path.join(MODELS, "ppo_equiv_*_1hot_16x16_*_policy.pth")))
    if not hits:
        print("real checkpoint loads .... skipped (no artifacts/models; see tools/fetch_artifacts.py)")
        return
    ckpt = hits[0]
    cfg = parse_arch_tag(os.path.basename(ckpt))
    net = ForestFirePolicyNet(
        backbone_net=ForestFireEquivariantNet(grid_size=16, **cfg),
        n_actions=9, hidden=cfg["hidden"],
    )
    from tensordict.nn import TensorDictModule
    from torch.distributions import Categorical
    from torchrl.modules import ProbabilisticActor
    actor = ProbabilisticActor(
        TensorDictModule(net, in_keys=["grid", "t", "agent_state"], out_keys=["logits"]),
        distribution_class=Categorical, in_keys=["logits"], out_keys=["action"],
        return_log_prob=False,
    )
    actor.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    print(f"real checkpoint loads .... ok  ({os.path.basename(ckpt)[:46]}...)")


def test_grow_load_is_function_preserving():
    """A 16x16 checkpoint grown to the 32x32 trunk must compute the same thing.

    This is what makes the warm start honest: at step 0 the bigger net is not an
    approximation of the smaller one, it is the smaller one.
    """
    small_cfg, big_cfg = arch_for_grid(16), arch_for_grid(32)
    torch.manual_seed(0)
    small = ForestFirePolicyNet(
        backbone_net=ForestFireEquivariantNet(grid_size=16, **small_cfg),
        n_actions=9, hidden=small_cfg["hidden"])
    big = ForestFirePolicyNet(
        backbone_net=ForestFireEquivariantNet(grid_size=32, **big_cfg),
        n_actions=9, hidden=big_cfg["hidden"])
    with torch.no_grad():  # a trained checkpoint is not at its init
        for p in small.parameters():
            p.add_(0.02 * torch.randn_like(p))

    grown = load_grown_state_dict(big, small.state_dict(), src_dilations=small_cfg["dilations"],
                                  dst_dilations=big_cfg["dilations"])
    assert grown == [6, 7, 8], grown
    with torch.no_grad():
        x = _inputs(4, 32)
        diff = (small(*x) - big(*x)).abs().max().item()
    assert diff == 0.0, diff
    print(f"grow-load preserves ...... ok  (blocks {grown} appended, max diff {diff})")


def test_grow_load_rejects_real_mismatches():
    big_cfg = arch_for_grid(32)

    def fresh():
        return ForestFireEquivariantNet(grid_size=32, **big_cfg)

    # A different width is a shape mismatch, which torch raises on regardless.
    other = ForestFireEquivariantNet(grid_size=32, **dict(big_cfg, channels=64))
    try:
        load_grown_state_dict(fresh(), other.state_dict())
    except RuntimeError:
        pass
    else:
        raise AssertionError("a different channel count must not load")

    # A source with MORE blocks leaves keys over, which means it is not a source.
    bigger = ForestFireEquivariantNet(grid_size=32, **dict(
        big_cfg, blocks=11, dilations=big_cfg["dilations"] + (16, 32)))
    try:
        load_grown_state_dict(fresh(), bigger.state_dict())
    except RuntimeError as e:
        assert "does not" in str(e), e
    else:
        raise AssertionError("a larger source must not load")

    # Dilation is not a parameter, so this one is invisible in the state dict:
    # a dense trunk loading into a net whose FIRST blocks are dilated would have
    # every kernel silently reinterpreted at a spacing it never saw.
    small = ForestFireEquivariantNet(grid_size=16, **arch_for_grid(16))
    try:
        load_grown_state_dict(fresh(), small.state_dict(), src_dilations=(1,) * 6,
                              dst_dilations=(2, 4, 8) + (1,) * 6)
    except RuntimeError as e:
        assert "spacing" in str(e), e
    else:
        raise AssertionError("a dilation prefix mismatch must not load")

    # And an appended block that is not at identity would change the function.
    target = fresh()
    with torch.no_grad():
        target.blocks[7].norm2.weight.fill_(0.5)
    try:
        load_grown_state_dict(target, small.state_dict())
    except RuntimeError as e:
        assert "identity" in str(e), e
    else:
        raise AssertionError("a non-identity appended block must not load")

    print("grow-load rejects junk ... ok  (width, extra blocks, dilation, non-identity)")


if __name__ == "__main__":
    test_small_grids_are_frozen()
    test_built_net_matches_todays_literals()
    test_dilations_default_is_inert()
    test_receptive_field_is_measured_not_asserted()
    test_grid_sizes_get_a_field_that_covers_them()
    test_tag_roundtrips()
    test_every_checkpoint_on_disk_still_parses()
    test_real_checkpoint_still_loads_strict()
    test_grow_load_is_function_preserving()
    test_grow_load_rejects_real_mismatches()
    print("\nall checks passed")
