import steerca.firebreak.ca as forest_ca
import steerca.core.cs_env as cs_env
from steerca.firebreak.ca import DIRT, TREE, FIRE
import os
from steerca.config import setting

import torch
from tensordict import TensorDict, TensorDictBase


def ca_step(grid):
    """Update the grid state using forest fire rules."""
    return forest_ca.step(grid)

_DELTAS = torch.tensor([
    [ 0,  1],  # right      0
    [-1,  1],  # up-right   1
    [-1,  0],  # up         2
    [-1, -1],  # up-left    3
    [ 0, -1],  # left       4
    [ 1, -1],  # down-left  5
    [ 1,  0],  # down       6
    [ 1,  1],  # down-right 7
    [ 0,  0],  # cut        8
], dtype=torch.long)

_DELTAS_CACHE = {}

def _deltas(device):
    """Per-device cache of _DELTAS to avoid a host-to-device copy every step."""
    deltas = _DELTAS_CACHE.get(device)
    if deltas is None:
        deltas = _DELTAS.to(device)
        _DELTAS_CACHE[device] = deltas
    return deltas

def time(actions, agent_params):
    """Return the time taken for each action in the batch."""
    tc = agent_params["tc"]  # shape [B]
    tm = agent_params["tm"]  # shape [B]
    # vectorized selection: cut uses tc, moves use tm
    return torch.where(actions == 8, tc, tm)
    
# Does not create a copy of the grid
def action(actions, grids, agent_state):
    """Perform batch actions on grids."""


    pos = agent_state.long()  # shape [B,2]
    n = grids.shape[-1]  # grid size

    # apply cut on grid for action eight: only where there is a tree,
    # fully vectorized so no host/device sync is needed
    batch = torch.arange(actions.shape[0], device=actions.device)
    do_cut = (actions == 8) & grids[batch, forest_ca.TREE, pos[:, 0], pos[:, 1]]
    # cut the tree
    grids[batch, forest_ca.TREE, pos[:, 0], pos[:, 1]] &= ~do_cut
    # set dirt where the tree was cut
    grids[batch, forest_ca.DIRT, pos[:, 0], pos[:, 1]] |= do_cut

    delta = _deltas(actions.device)[actions]  # [B,2]

    new_pos = torch.clamp(pos + delta, 0, n - 1)

    return grids, new_pos  # return updated grid and new agent state


def done(grids):
    """Fixed point reached iff no FIRE cells remain: a FIRE cell always burns
    to DIRT, so any grid containing fire must still change under the CA."""
    return ~grids[..., FIRE, :, :].any(dim=(-2, -1))



# --- What a cut costs -----------------------------------------------------------
#
# A tree the FIRE takes always costs 1. A tree the AGENT fells is charged
# CUT_TREE_CHARGE_AHEAD, or CUT_TREE_CHARGE_IMMINENT when the fire was already on
# top of it -- a burning Moore neighbour, so the CA would ignite it next tick.
# CUT_ACTION_COST is added to either, and is charged for the cut action whether
# or not it lands on a tree.
#
# By default both tree charges are 0, so a cut costs a flat 0.5 wherever it lands
# and the imminent test never changes the price. Raising CUT_TREE_CHARGE_IMMINENT
# to 1 makes felling a tree the fire is about to take cost 1.5, while a cut ahead
# of the fire -- what a firebreak is made of -- stays at 0.5.
#
# None of these is the trees-saved objective, and the gap is the point: it is
# traded for a learning signal. What matters is what each choice earns the agent
# relative to leaving the tree alone, since that is what the gradient sees.
# Writing s = CUT_COST for the case in question:
#
#   adjacent to fire (burns next tick)   cut s vs burn 1.0   -> pays 1.0 - s
#   burns in k ticks, not yet adjacent   cut s vs burn 1.0   -> pays 1.0 - s
#   fire can never reach it              cut s vs burn 0.0   -> costs s
#
# So any price below 1 pays the agent to cut trees the fire would have taken --
# whether or not the cut ends up part of a closed barrier. That is the farming
# exploit, and it cuts both ways. It is what lets an agent learn to cut at all:
# the r6 8x8 policy at tc=tm=0.05 scored 88.8%, tying the planner, and the r9
# shaping reached 70.5% from scratch in 600k frames. It is also what PPO farmed
# at tm+tc = 1.2, where nothing can be saved, to reach 0.25% of trees saved
# against the planner's 3.60% with its return climbing throughout.
#
# Pricing it away does not work either: r7 (flat 1.5) and r8 (flat 1.0, where the
# return is exactly -(trees lost)/area) both had PPO converge to never cutting
# from scratch -- 0.0 cuts, 0.0% saved, against the planner's 87.1%.
#
# So: score runs on TREES SAVED, never on return, and treat a rising return with
# flat trees-saved as the signature of farming rather than progress. test_policy
# reports both and checkpoints are selected on trees saved for exactly this
# reason.
#
# The default charges nothing for the tree in EITHER case, so a cut costs a flat
# 0.5 wherever it lands -- the original reward, from before the whole sequence
# above. It is the most learnable and the most exploitable of the four, and both
# halves of that are established rather than guessed:
#
#   the r6 8x8 policy at tc=tm=0.05 scored 88.8%, tying the planner
#   at tm+tc = 1.2 PPO farmed it to 0.25% saved against the planner's 3.60%
#
# Set STEERCA_CUT_TREE_IMMINENT=1 for the r9 shaping, which keeps the cheap cut
# ahead of the fire but charges 1.5 for felling a tree the fire already has --
# removing the most wasteful farming without removing the signal. Set both to
# 1 for the r7 objective, or AHEAD=0.5/IMMINENT=0.5 for r8's flat 1.0.
#
# Whatever is chosen ends up in the experiment name via cut_price_tag(), so runs
# under different pricing cannot land on the same checkpoint path.
CUT_ACTION_COST = 0.5
CUT_TREE_CHARGE_IMMINENT = float(setting("CUT_TREE_IMMINENT", 0.0))
CUT_TREE_CHARGE_AHEAD = float(setting("CUT_TREE_AHEAD", 0.0))

# What felling a tree costs the agent in each case, action included.
CUT_COST_IMMINENT = CUT_TREE_CHARGE_IMMINENT + CUT_ACTION_COST
CUT_COST_AHEAD = CUT_TREE_CHARGE_AHEAD + CUT_ACTION_COST


def cut_price_tag():
    """Short name for the cut pricing, to go in experiment names.

    `cut0.5` for a flat price, `cut0.5-1.5` when a tree the fire is about to take
    is charged more. Without this two runs under different rewards would share a
    checkpoint path and silently resume each other.
    """
    if CUT_COST_AHEAD == CUT_COST_IMMINENT:
        return f"cut{CUT_COST_AHEAD:g}"
    return f"cut{CUT_COST_AHEAD:g}-{CUT_COST_IMMINENT:g}"


def reward_grid(grid, mid_grid, next_grid):
    """Trees lost across a whole step, charged by what took them.

    `cs_env.step` passes three states: before the step, after the CA advance
    (`mid_grid`), and after the agent's action. The fire's losses are the first
    difference and the agent's cuts the second, which is what lets the two be
    priced separately -- and within the cuts, a tree the fire was about to take
    is charged more than one felled ahead of the fire. See the note above.
    """
    trees0 = grid[..., TREE, :, :].sum(dim=(-2, -1))
    trees_mid = mid_grid[..., TREE, :, :].sum(dim=(-2, -1))

    burnt = (trees0 - trees_mid).to(torch.float32)  # taken by the fire

    # Cells the agent felled: a tree in mid_grid that is gone from next_grid.
    felled_mask = mid_grid[..., TREE, :, :] & ~next_grid[..., TREE, :, :]
    # "About to burn" is asked of the CA's own ignition rule, evaluated on the
    # state the agent actually cut into.
    imminent_mask = felled_mask & (forest_ca.fire_neighbors(mid_grid) >= 1)

    felled = felled_mask.sum(dim=(-2, -1)).to(torch.float32)
    imminent = imminent_mask.sum(dim=(-2, -1)).to(torch.float32)
    ahead = felled - imminent

    # Normalized by grid area so reward scale (and thus value targets / critic
    # loss) stays invariant to grid size. This is N*N, not N: the tree count
    # itself scales with the area, so dividing by N left the return still
    # growing linearly with the grid size and made returns from different N
    # incomparable.
    area = grid.shape[-2] * grid.shape[-1]
    charge = (burnt
              + CUT_TREE_CHARGE_IMMINENT * imminent
              + CUT_TREE_CHARGE_AHEAD * ahead)
    return -charge / area

def reward_action(actions, next_grid, agent_state):
    """Calculate the reward based on the action taken."""
    # CUT_ACTION_COST per cut ACTION, charged even when it hits bare dirt, which
    # is what makes pointless cutting cost something. The tree it destroys is
    # charged separately by reward_grid. Scaled by the same area factor so their
    # relative weighting is size-independent.
    area = next_grid.shape[-2] * next_grid.shape[-1]
    cut_reward = -CUT_ACTION_COST / area
    move_penalty = 0.0

    rewards = torch.where(actions == 8, cut_reward, move_penalty)

    return rewards.to(torch.float32)  # Ensure rewards are float32




import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch, Circle, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Cell colors — CVD-validated (protan/deutan/tritan ΔE >= 21 for every pair,
# Machado 2009 at severity 1.0), so the map stays readable for colorblind viewers.
DIRT_COLOR = "#D9B38C"   # dry earth tan
TREE_COLOR = "#2E7D32"   # forest green
FIRE_COLOR = "#FF6D00"   # flame orange

_SURFACE = "#FCFCFB"     # chart surface / cell-gap color
_INK = "#1A1A19"         # primary text
_MUTED = "#5F6368"       # secondary text
_CHARCOAL = "#263238"    # bulldozer dark parts
_DOZER_YELLOW = "#F9A825"  # bulldozer body


def _fmt(value, fmt=".2f"):
    """Format a scalar that may be a tensor, numpy value or float."""
    return format(float(value), fmt)


def _draw_bulldozer(ax, row: int, col: int):
    """Draw a small bulldozer pictogram centered on cell (row, col).

    Coordinates are grid data coords with origin='upper', so +y points down:
    the tracks sit at row + offset, the cab above them at row - offset.
    """
    # White halo so the icon reads on any cell color
    ax.add_patch(Circle((col, row), 0.44, facecolor=_SURFACE, edgecolor=_CHARCOAL,
                        linewidth=1.2, alpha=0.95, zorder=4))
    # Blade (front-right) and arm
    ax.add_patch(FancyBboxPatch((col + 0.18, row - 0.16), 0.10, 0.34,
                                boxstyle="round,pad=0,rounding_size=0.04",
                                facecolor=_CHARCOAL, edgecolor="none", zorder=6))
    ax.add_patch(Rectangle((col + 0.02, row - 0.03), 0.18, 0.07,
                           facecolor=_CHARCOAL, edgecolor="none", zorder=5))
    # Body and cab window
    ax.add_patch(FancyBboxPatch((col - 0.28, row - 0.22), 0.36, 0.28,
                                boxstyle="round,pad=0,rounding_size=0.05",
                                facecolor=_DOZER_YELLOW, edgecolor=_CHARCOAL,
                                linewidth=1.0, zorder=5))
    ax.add_patch(Rectangle((col - 0.23, row - 0.17), 0.14, 0.12,
                           facecolor=_SURFACE, edgecolor=_CHARCOAL,
                           linewidth=0.8, zorder=6))
    # Track with two wheels
    ax.add_patch(FancyBboxPatch((col - 0.30, row + 0.08), 0.42, 0.16,
                                boxstyle="round,pad=0,rounding_size=0.08",
                                facecolor=_CHARCOAL, edgecolor="none", zorder=5))
    for wheel_x in (col - 0.21, col + 0.03):
        ax.add_patch(Circle((wheel_x, row + 0.16), 0.035,
                            facecolor=_SURFACE, edgecolor="none", zorder=6))


def draw_forest_fire(grid: torch.Tensor, bulldozer_pos: tuple[int, int], t: torch.Tensor, step_counter = 0,
                     reward: torch.Tensor = None, cum_reward: torch.Tensor = None,
                     format: str = "tensor", title: str = "Forest Fire — Bulldozer Agent"):
    """
    Render the forest fire grid with the bulldozer agent as an RGB image.

    Args:
        grid (torch.Tensor): One-hot encoded (3, N, N) tensor with DIRT / TREE / FIRE channels.
        bulldozer_pos: (row, col) position of the bulldozer.
        t: fractional time within the current CA tick.
        step_counter: current step number, shown in the header.
        reward / cum_reward: optional scalars shown in the header.
        format: "tensor" returns a uint8 torch tensor (H, W, 3), anything else a numpy array.
        title (str): The title for the plot.
    """
    grid = grid.detach().cpu()
    grid2D = torch.zeros((grid.shape[-2], grid.shape[-1]), dtype=torch.int64)
    grid2D = torch.where(grid[DIRT, :, :] == 1, 0, grid2D)
    grid2D = torch.where(grid[TREE, :, :] == 1, 1, grid2D)
    grid2D = torch.where(grid[FIRE, :, :] == 1, 2, grid2D)

    grid2D = np.array(grid2D)
    rows, cols = grid2D.shape

    cmap = mcolors.ListedColormap([DIRT_COLOR, TREE_COLOR, FIRE_COLOR])
    norm = mcolors.BoundaryNorm(np.arange(-0.5, 3), cmap.N)

    # --- Layout: fixed 10 in width; header band above a margin-padded grid ---
    margin_in, header_in = 0.35, 1.35
    fig_w = 10.0
    ax_w_in = fig_w - 2 * margin_in
    ax_h_in = ax_w_in * rows / cols
    fig_h = ax_h_in + margin_in + header_in
    # Use Figure directly (not pyplot) so rendering is thread-safe and leak-free
    fig = Figure(figsize=(fig_w, fig_h), facecolor=_SURFACE)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([margin_in / fig_w, margin_in / fig_h,
                       ax_w_in / fig_w, ax_h_in / fig_h])
    ax.set_facecolor(_SURFACE)

    ax.imshow(grid2D, cmap=cmap, norm=norm, interpolation="nearest")

    # Thin surface-colored gaps between cells instead of heavy black lines.
    # Boundaries go on the MAJOR ticks: minor ticks that coincide with the
    # auto-chosen majors are silently dropped (remove_overlapping_locs),
    # which erased the grid for small N.
    ax.set_xticks(np.arange(-.5, cols, 1))
    ax.set_yticks(np.arange(-.5, rows, 1))
    # Gap width scales down with cell size so dense grids don't turn into mesh
    cell_pt = ax_w_in / cols * 72
    ax.grid(color=_SURFACE, linewidth=min(2.0, max(0.4, 0.08 * cell_pt)))
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_color("#D6D4CF")
        spine.set_linewidth(1.2)

    bulldozer_row, bulldozer_col = int(bulldozer_pos[0]), int(bulldozer_pos[1])
    if 0 <= bulldozer_row < rows and 0 <= bulldozer_col < cols:
        _draw_bulldozer(ax, bulldozer_row, bulldozer_col)

    # --- Header: title, stats line, legend ---
    fig.text(margin_in / fig_w, 1 - 0.38 / fig_h, title,
             ha="left", va="center", fontsize=18, fontweight="bold", color=_INK)

    stats = [f"Step {step_counter}", f"t = {_fmt(t)}"]
    if reward is not None:
        stats.append(f"Reward {_fmt(reward)}")
    if cum_reward is not None:
        stats.append(f"Return {_fmt(cum_reward)}")
    stats.append(f"Trees {int((grid2D == 1).sum())}")
    stats.append(f"Fire {int((grid2D == 2).sum())}")
    fig.text(margin_in / fig_w, 1 - 0.80 / fig_h, "   ·   ".join(stats),
             ha="left", va="center", fontsize=17, color=_MUTED)

    legend_handles = [
        Patch(facecolor=DIRT_COLOR, edgecolor="none", label="Dirt"),
        Patch(facecolor=TREE_COLOR, edgecolor="none", label="Tree"),
        Patch(facecolor=FIRE_COLOR, edgecolor="none", label="Fire"),
        Line2D([], [], marker="o", linestyle="none", markersize=9,
               markerfacecolor=_DOZER_YELLOW, markeredgecolor=_CHARCOAL, label="Bulldozer"),
    ]
    fig.legend(handles=legend_handles, loc="upper right",
               bbox_to_anchor=(1 - margin_in / fig_w, 1 - 0.20 / fig_h),
               ncol=4, frameon=False, fontsize=13, handlelength=1.2,
               labelcolor=_MUTED, borderaxespad=0)

    # --- Render the figure to a numpy RGB array ---
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()

    if format == "tensor":
        return torch.from_numpy(img).to(torch.uint8)

    return img  # numpy array of shape (H, W, 3)