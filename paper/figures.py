"""Draw every figure in the paper from ``paper/data/*.csv``.

    python paper/figures.py            # all of them
    python paper/figures.py difficulty ppo_vs_planner

Nothing here touches a model, a log or the environment -- ``paper/collect.py``
does that, once, and writes the CSVs. So a figure can be restyled in a second
without a GPU, and every number in the paper traces to a row on disk.

The two exceptions are `episode` and `regimes`, which need a rollout to draw: they
import the environment and reuse ``steerca.firebreak.agents.draw_forest_fire``, the same
renderer the episode GIFs are made from.

Style notes, since these are print figures in a two-column journal:
  * Colours are Okabe-Ito, the published colour-vision-deficiency-safe set. Every
    series also carries its own marker and dash pattern, so identity survives
    greyscale printing and the figures never rely on hue alone.
  * One y-axis per panel, never two.
  * Grid and spines are recessive; series are labelled directly where there is
    room, with a legend as well.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(HERE, "data")
IMAGES = os.path.join(HERE, "figures")
# The few figures that roll out a policy (episode, regimes, planted_episodes)
# load checkpoints from the SteerCA-artifacts archive; see paper/collect.py.
MODELS = os.path.join(os.path.abspath(os.environ.get(
    "STEERCA_ARTIFACTS", os.path.join(ROOT, "artifacts"))), "models")

# IEEEtran: \columnwidth is 3.5in, \textwidth 7.16in.
COL, TEXT = 3.5, 7.16

# Okabe-Ito. Grid size is the categorical dimension in every figure, and a size
# keeps its colour, marker and dash across all of them -- colour follows the
# entity, so the reader learns the mapping once.
GRID_STYLE = {
    8:  dict(color="#0072B2", marker="o", ls="-",  label=r"$8\times8$"),
    16: dict(color="#D55E00", marker="s", ls="--", label=r"$16\times16$"),
    32: dict(color="#009E73", marker="^", ls="-.", label=r"$32\times32$"),
}
AGENT_COLOR = {"planner": "#56B4E9", "ppo": "#E69F00"}
INK, MUTED = "#222222", "#666666"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})


def load(name):
    with open(os.path.join(DATA, name)) as fh:
        return list(csv.DictReader(fh))


# Set by springer(): the suffix every saved figure takes during a Springer build.
SAVE_SUFFIX = ""


def save(fig, name):
    os.makedirs(IMAGES, exist_ok=True)
    stem, ext = os.path.splitext(name)
    path = os.path.join(IMAGES, stem + SAVE_SUFFIX + ext)
    fig.savefig(path)
    plt.close(fig)
    print(f"  {os.path.relpath(path, ROOT)}")


def tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)


# ==========================================================================
# 1. The difficulty axis: what the planner saves as the action cost rises
# ==========================================================================

# Feldheim and Hod's firefighter bound: from s = 1/3 the bulldozer protects at
# most 3t cells before update t, so no barrier of any shape closes without the
# grid edge. A theorem rather than a count, so it is drawn unlike the two
# regime boundaries (dotted): dashed, inside the spiral band, without a band of
# its own -- below it the question is open, not answered the other way.
S_EDGE = 1 / 3
S_TICKS = [0.04, 0.08, 0.125, S_EDGE, 0.5, 1.0]
S_TICKLABELS = ["0.04", "0.08", "1/8", "1/3", "0.5", "1"]


def _edge_bound(ax, note=None, fontsize=6.5):
    """The s = 1/3 line, optionally with a note just right of it at the top."""
    ax.axvline(S_EDGE, color=MUTED, lw=0.8, ls=(0, (4, 2)), zorder=0.5)
    if note:
        ax.text(S_EDGE * 1.05, 0.97, note, ha="left", va="top", fontsize=fontsize,
                color=MUTED, linespacing=1.1, transform=ax.get_xaxis_transform())


def fig_difficulty():
    """Trees saved against s = t_c + t_m, with the three barrier regimes marked.

    This is the figure the difficulty section is built on. The closed-form
    bounds say a ring closes only below s = 1/8 and that no barrier keeps pace
    with the fire at s >= 1 (and, from the firefighter bound, that none closes
    without the grid edge at s >= 1/3); what the measurement shows is that no
    boundary is a cliff, because the grid edge is a free barrier.
    """
    rows = [r for r in load("planner_ceiling.csv") if r["agent"] == "planner"]
    # The twelve sweep cells plus the ladders past s = 1, where no barrier keeps pace
    # (s = 1 and 1.2), so the learned agent is shown in all three regimes.
    ppo = [r for r in load("sweep.csv") + load("beyond.csv") if r["agent"] == "ppo"]

    fig, ax = plt.subplots(figsize=(COL, 2.5))

    # Regime bands, drawn first so they sit under everything.
    ax.axvspan(0, 0.125, color="#0072B2", alpha=0.055, lw=0)
    ax.axvspan(0.125, 1.0, color="#E69F00", alpha=0.055, lw=0)
    ax.axvline(0.125, color=MUTED, lw=0.7, ls=":")
    ax.axvline(1.0, color=MUTED, lw=0.7, ls=":")
    _edge_bound(ax, note="every barrier\nneeds the edge")
    ax.text(0.055, 1.02, "closed ring", ha="center", va="bottom",
            fontsize=6.5, color=MUTED, transform=ax.get_xaxis_transform())
    ax.text(0.36, 1.02, "edge-closed", ha="center", va="bottom",
            fontsize=6.5, color=MUTED, transform=ax.get_xaxis_transform())
    ax.text(1.16, 1.02, "refuge", ha="center", va="bottom",
            fontsize=6.5, color=MUTED, transform=ax.get_xaxis_transform())

    by_grid = defaultdict(list)
    for r in rows:
        by_grid[int(r["n"])].append((float(r["s"]), 100 * float(r["saved"])))
    for n, style in GRID_STYLE.items():
        pts = sorted(by_grid.get(n, []))
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, **style, markerfacecolor="white", markeredgewidth=1.0)

    # PPO's eighteen cells on the same axes: the learned agent sits on the
    # planner's curve, which is the result, so it is drawn as bare markers
    # rather than a second family of lines.
    pp = sorted((float(r["s"]), 100 * float(r["saved"])) for r in ppo)
    if pp:
        xs, ys = zip(*pp)
        ax.plot(xs, ys, ls="none", marker="x", ms=4.5, mew=1.1,
                color=INK, label="PPO (all sizes)")

    ax.set_xscale("log")
    ax.set_xlabel(r"action cost $s = t_c + t_m$  (CA ticks)")
    ax.set_ylabel(r"trees saved (\% of grid)" if plt.rcParams["text.usetex"]
                  else "trees saved (% of grid)")
    ax.set_ylim(0, 100)
    ax.set_xticks(S_TICKS)
    ax.set_xticklabels(S_TICKLABELS)
    ax.minorticks_off()
    ax.set_xlim(0.033, 1.35)
    ax.legend(loc="lower left", frameon=False, handlelength=2.2)
    tidy(ax)
    save(fig, "carl_difficulty.pdf")


# The Springer Nature text block (31pc) and the paper's body font. Matplotlib
# ships Computer Modern (cmr10, and the "cm" math set), so the Springer-version
# figures match the text without usetex, which needs cm-super installed.
SN_TEXT = 31 / 6.0225  # 31pc in inches
SN_RC = {"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
         "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
         "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7.5,
         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5}


# Planner curves on the difficulty figure: thinner than the default and
# unmarked, so they read as the reference the agents' markers sit on.
PLANNER_LW = 0.9

# The other learners on the difficulty figure, at their best settings.
OTHER_MARKER = {"a2c": "D", "sac": "v"}
OTHER_LABEL = {"a2c": r"A2C ($8\times8$)", "sac": r"SAC ($8\times8$)"}


# Schematic grids for the three regimes, drawn above Figure 2. Cells are
# (row, col) with row 0 on top. Each barrier is 4-connected -- a diagonal step
# leaks, since fire spreads diagonally -- and the path lists the cells the
# bulldozer cuts, in order; arrows follow it. Trees the fire will still reach
# are white, trees the barrier saves green.
REGIME_SKETCHES = {
    "ring": dict(
        n=9,
        burnt=[(4, 4)],
        front=[(r, c) for r in range(3, 6) for c in range(3, 6) if (r, c) != (4, 4)],
        path=[(1, 1), (1, 7), (7, 7), (7, 1), (2, 1)],
    ),
    "spiral": dict(
        n=12,
        burnt=[(r, c) for r in range(8, 11) for c in range(2, 5)],
        front=[(r, c) for r in range(7, 12) for c in range(1, 6)
               if max(abs(r - 9), abs(c - 3)) == 2],
        # The staircase PPO learns in this regime (Fig. C2): anchored on both
        # side edges and stepping away from the fire as it is built, so each
        # cell is cut before the fire, which reaches the far end later, arrives.
        path=[(3, 0), (3, 3), (4, 3), (4, 6), (5, 6), (5, 9), (6, 9), (6, 11)],
        saved=[(r, c) for r in range(0, 6) for c in range(12)
               if r < {0: 3, 1: 3, 2: 3, 3: 4, 4: 4, 5: 4, 6: 5, 7: 5, 8: 5,
                       9: 6, 10: 6, 11: 6}[c]],
    ),
    "refuge": dict(
        n=10,
        burnt=[(r, c) for r in range(0, 5) for c in range(0, 5)],
        front=[(r, c) for r in range(0, 6) for c in range(0, 6)
               if max(r, c) == 5],
        # Walled off against the grid edge: two sides cut, two supplied free.
        path=[(6, 9), (6, 6), (9, 6)],
        refuge=[(r, c) for r in range(7, 10) for c in range(7, 10)],
    ),
}


def _path_cells(corners):
    """Every cell along a path of horizontal and vertical runs."""
    cells = [corners[0]]
    for (r0, c0), (r1, c1) in zip(corners, corners[1:]):
        dr, dc = (r1 > r0) - (r1 < r0), (c1 > c0) - (c1 < c0)
        r, c = r0, c0
        while (r, c) != (r1, c1):
            r, c = r + dr, c + dc
            cells.append((r, c))
    return cells


def _regime_sketch(ax, kind):
    """One regime's strategy: the fire, the barrier the bulldozer cuts, its
    path as arrows, and which trees it saves (green) or concedes (white)."""
    from matplotlib.patches import Rectangle as R
    sk = REGIME_SKETCHES[kind]
    n = sk["n"]
    def cell(rc, **kw):
        r, c = rc
        ax.add_patch(R((c, n - 1 - r), 1, 1, lw=0, **kw))
    # Which trees the fire will still reach: a Moore flood fill from the fire
    # through every cell the barrier leaves uncut. Those end burnt and are
    # drawn white; the trees the barrier saves keep the episode figures' green.
    cut = set(_path_cells(sk["path"]))
    fire = set(sk["burnt"]) | set(sk["front"])
    doomed, stack = set(fire), list(fire)
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                q = (r + dr, c + dc)
                if 0 <= q[0] < n and 0 <= q[1] < n and q not in cut and q not in doomed:
                    doomed.add(q)
                    stack.append(q)
    for r in range(n):
        for c in range(n):
            cell((r, c), facecolor="white" if (r, c) in doomed else CELL_COLORS[1])
    for rc in sk["burnt"]:
        cell(rc, facecolor="#EAD7BF")
    for rc in sk["front"]:
        cell(rc, facecolor=CELL_COLORS[2])
    for rc in _path_cells(sk["path"]):
        cell(rc, facecolor=CELL_COLORS[0])
    for k in range(n + 1):
        ax.plot([k, k], [0, n], color="#BBBBBB", lw=0.3, alpha=0.7, zorder=1)
        ax.plot([0, n], [k, k], color="#BBBBBB", lw=0.3, alpha=0.7, zorder=1)
    # The bulldozer's path through cell centres, an arrow on each straight run.
    pts = [(c + 0.5, n - 1 - r + 0.5) for r, c in sk["path"]]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8,
                                    mutation_scale=6, shrinkA=0, shrinkB=0))
    ax.plot(*pts[0], marker="o", ms=3.6, markerfacecolor=DOZER,
            markeredgecolor="#333333", markeredgewidth=0.5, zorder=5)
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#999999"); sp.set_linewidth(0.6)


CELL_COLORS = ["#D9B38C", "#2E7D32", "#FF6D00"]   # dirt, tree, fire
XMAX = 3.0   # right end of Figure 2's action-cost axis
# The refuge regime's band: the tree green of the episode figures and of the
# refuge sketch, as faint as the other two bands.
REFUGE_BAND = dict(color=CELL_COLORS[1], alpha=0.07, lw=0)


def fig_difficulty_sn():
    """Figure 2 of the Springer version: fig_difficulty, redrawn to be read.

    Same data, plus PPO at 8x8 s = 0.04 and 0.08 and the other learners.
    Drawn at its printed width in Computer Modern. PPO's twenty cells take their grid size's colour and marker, filled, so each sits on the
    planner curve it is compared against (a bare line); in the
    original the eighteen there are one black cross. The third regime is named as the
    paper names it, and the dashed line at s = 1/3 is the firefighter bound. carl_difficulty.pdf, which the Scientific Reports version
    uses, is left as it was.
    """
    rows = [r for r in load("planner_ceiling.csv") if r["agent"] == "planner"]
    ppo = [r for r in load("sweep.csv") + load("beyond.csv") if r["agent"] == "ppo"]
    # 8x8 below the sweep's costs (s = 0.04, 0.08): the second-seed runs, the
    # only ones there with the full 4M-frame budget of every other cell.
    have = {float(r["s"]) for r in ppo if int(r["n"]) == 8}
    ppo += [r for r in load("extra_8x8.csv") if r["agent"] == "ppo"
            and r["ckpt"].endswith("_r10_seed2_policy.pth") and float(r["s"]) not in have]
    # A2C and SAC at their best settings, 8x8 only; absent until collected.
    try:
        others = [r for r in load("others_8x8.csv") if r["agent"] != "planner"]
    except FileNotFoundError:
        others = []
    with plt.rc_context(SN_RC):
        W, H, TOP = SN_TEXT, 4.1, 1.35   # TOP: inches above the plot for the sketches
        fig, ax = plt.subplots(figsize=(W, H))
        fig.subplots_adjust(left=0.13, right=0.985, bottom=0.40 / H, top=1 - TOP / H)
        ax.axvspan(0.01, 0.125, color="#0072B2", alpha=0.055, lw=0)
        ax.axvspan(0.125, 1.0, color="#E69F00", alpha=0.055, lw=0)
        ax.axvspan(1.0, XMAX, **REFUGE_BAND)
        ax.axvline(0.125, color=MUTED, lw=0.7, ls=":")
        ax.axvline(1.0, color=MUTED, lw=0.7, ls=":")
        _edge_bound(ax, note="every barrier\nneeds the edge", fontsize=7)

        by_grid = defaultdict(list)
        for r in rows:
            by_grid[int(r["n"])].append((float(r["s"]), 100 * float(r["saved"])))
        cells = defaultdict(list)
        for r in ppo:
            cells[int(r["n"])].append((float(r["s"]), 100 * float(r["saved"])))
        learners = [(a, m) for a, m in OTHER_MARKER.items()
                    if any(r["agent"] == a for r in others)]
        # The curves are drawn under the paired agents, so they must be the
        # planner on the agents' own 100 starts: then the gap from a marker to
        # its line is the paired margin. planner_ceiling.csv is collected that
        # way (collect.py ceiling: 100 episodes, cuda); a curve that disagrees
        # with the planner's paired scores means it was not, so say so.
        curve = {(n, round(x, 3)): y for n, pts in by_grid.items() for x, y in pts}
        for f in ("sweep.csv", "beyond.csv", "others_8x8.csv", "extra_8x8.csv"):
            for r in load(f):
                k = (int(r["n"]), round(float(r["s"]), 3))
                if (r["agent"] == "planner" and k in curve
                        and abs(curve[k] - 100 * float(r["saved"])) > 0.05):
                    print(f"  WARNING: planner curve {curve[k]:.2f}% at n={k[0]}, "
                          f"s={k[1]} is not its paired score "
                          f"{100 * float(r['saved']):.2f}% in {f}", file=sys.stderr)

        def draw(a, ms=4.6, clip=False, planner=by_grid):
            """The data, on the main axes or (clipped to its window) the inset."""
            for n, style in GRID_STYLE.items():
                pts = sorted(planner.get(n, []))
                if pts:
                    xs, ys = zip(*pts)
                    # A bare line, so the trained agents' markers are the only ones.
                    a.plot(xs, ys, color=style["color"], ls=style["ls"], lw=PLANNER_LW,
                           zorder=2)
                if cells.get(n):
                    xs, ys = zip(*sorted(cells[n]))
                    a.plot(xs, ys, ls="none", marker=style["marker"], ms=ms,
                           color=style["color"], markeredgecolor=INK,
                           markeredgewidth=0.5, zorder=3)
            # The other learners in the 8x8 colour, told apart from PPO by marker.
            # Unclipped, so a cell at 0% sits on the axis rather than half under it.
            for algo, marker in learners:
                # At its own s: no offset, though at s >= 1 A2C and SAC overlap
                # near 0%.
                xs, ys = zip(*sorted((float(r["s"]), 100 * float(r["saved"]))
                                     for r in others if r["agent"] == algo))
                a.plot(xs, ys, ls="none", marker=marker, ms=ms - 0.4,
                       color=GRID_STYLE[8]["color"], markeredgecolor=INK,
                       markeredgewidth=0.5, zorder=3, clip_on=clip)

        draw(ax)

        ax.set_xscale("log")
        ax.set_xlabel(r"action cost $s = t_c + t_m$  (CA updates)")
        ax.set_ylabel("trees saved (% of grid)")
        ax.set_ylim(0, 100)
        ax.set_xticks(S_TICKS + [2.0])
        ax.set_xticklabels(S_TICKLABELS + ["2"])
        ax.minorticks_off()
        # Out to s = 3, past the data, so the refuge band is wide enough to hold
        # its sketch.
        ax.set_xlim(0.033, XMAX)

        # One key for grid size, each entry showing the planner's line and PPO's
        # marker for that size; a second for the other learners.
        sizes = ax.legend(handles=[Line2D([], [], color=st["color"], ls=st["ls"],
                                          lw=PLANNER_LW, marker=st["marker"], ms=4.6,
                                          markeredgecolor=INK, markeredgewidth=0.5,
                                          label=st["label"])
                                   for st in GRID_STYLE.values()],
                          title="line: planner, marker: PPO", title_fontsize=7,
                          loc="upper right", handlelength=2.6, frameon=True,
                          facecolor="white", edgecolor="none", framealpha=0.85)
        sizes._legend_box.align = "left"
        ax.add_artist(sizes)
        ax.legend(handles=[Line2D([], [], color=GRID_STYLE[8]["color"], marker=m,
                                  markeredgecolor=INK, markeredgewidth=0.5, ls="none",
                                  label=OTHER_LABEL[a]) for a, m in learners],
                  loc="upper right", bbox_to_anchor=(1.0, 0.70), handlelength=1.6,
                  frameon=True, facecolor="white", edgecolor="none", framealpha=0.85)
        tidy(ax)

        # s >= 1 enlarged: every score there is below 10%, too low to read on the
        # main axes, and the refuge band's lower right is otherwise empty.
        zoom = ax.inset_axes([0.815, 0.07, 0.175, 0.36])
        zoom.set_facecolor("white")             # opaque over the main grid lines
        zoom.set_zorder(5)
        zoom.axvspan(0.8, 1.6, **REFUGE_BAND)
        draw(zoom, ms=4.0, clip=True)
        zoom.set_xscale("log")
        zoom.set_xlim(0.9, 1.32)
        zoom.set_ylim(-0.6, 10)
        zoom.set_xticks([1.0, 1.2])
        zoom.set_xticklabels(["1", "1.2"])
        zoom.minorticks_off()
        zoom.set_yticks([0, 5, 10])
        zoom.tick_params(labelsize=6.5, length=2, pad=1.5)
        zoom.grid(True, color="#DDDDDD", lw=0.4)
        for sp in zoom.spines.values():
            sp.set_edgecolor("#999999"); sp.set_linewidth(0.6)
        ax.indicate_inset_zoom(zoom, edgecolor="#999999", lw=0.6, alpha=1.0)

        # The strategy each regime calls for, drawn above its band. A sketch that
        # would overhang the plot is pulled back inside its width.
        pos = ax.get_position()
        lo, hi = math.log(0.033), math.log(XMAX)
        def fx(x):
            return pos.x0 + pos.width * (math.log(x) - lo) / (hi - lo)
        side = 0.95                          # sketch side, inches
        w, h = side / W, side / H
        y0 = pos.y1 + 0.10 / H
        band_x = {"ring": math.sqrt(0.033 * 0.125), "spiral": math.sqrt(0.125 * 1.0),
                  "refuge": math.sqrt(1.0 * XMAX)}
        # Each sketch centred over its band, kept inside the plot's width. The
        # middle band is named for what every barrier in it does -- close
        # against the grid edge -- not for the outward spiral of the bound.
        lefts = {k: min(max(fx(x) - w / 2, pos.x0), pos.x1 - w) for k, x in band_x.items()}
        titles = {"ring": "closed ring", "spiral": "edge-closed staircase",
                  "refuge": "refuge"}
        # No leader lines: each sketch is centred over its own band, and the
        # middle one's leader would land beside the s = 1/3 line and read as
        # pointing at it.
        for kind in ("ring", "spiral", "refuge"):
            sx = fig.add_axes([lefts[kind], y0, w, h])
            _regime_sketch(sx, kind)
            sx.set_title(titles[kind], fontsize=7.5, color=MUTED, pad=2.5)
        save(fig, "carl_difficulty_sn.pdf")


def _regime_axes(ax):
    """Log-s axis with the three regime bands, shared by the s-sweep figures."""
    ax.axvspan(0, 0.125, color="#0072B2", alpha=0.055, lw=0)
    ax.axvspan(0.125, 1.0, color="#E69F00", alpha=0.055, lw=0)
    if SAVE_SUFFIX:
        ax.axvspan(1.0, 1.62, **REFUGE_BAND)
    ax.axvline(0.125, color=MUTED, lw=0.7, ls=":")
    ax.axvline(1.0, color=MUTED, lw=0.7, ls=":")
    _edge_bound(ax)
    # A Springer build uses CA updates, not ticks, as the unit, and keeps the
    # labels to one word so they fit a half-width panel.
    right = 1.62 if SAVE_SUFFIX else 1.35
    labels = (((0.057, "ring", "center"), (0.36, "edge-closed", "center"),
               (1.27, "refuge", "center"))
              if SAVE_SUFFIX else
              ((0.055, "closed ring", "center"), (0.36, "edge-closed", "center"),
               (1.16, "refuge", "center")))
    for x, label, ha in labels:
        ax.text(x, 1.02, label, ha=ha, va="bottom",
                fontsize=7 if SAVE_SUFFIX else 6.5,
                color=MUTED, transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    unit = "CA updates" if SAVE_SUFFIX else "CA ticks"
    ax.set_xlabel(rf"action cost $s = t_c + t_m$  ({unit})")
    ax.set_xticks(S_TICKS)
    ax.set_xticklabels(S_TICKLABELS)
    ax.minorticks_off()
    ax.set_xlim(0.033, right)


def _draw_runtime(ax):
    """The planner's search time per decision, normalized by each grid's own
    cost at the cheapest action cost, one curve per grid size.

    Per decision rather than per episode: the planner replans once per CA update,
    so per-episode time is per-decision cost times how long the fire burns, and
    that tracks the score rather than the difficulty. Normalized because the
    planner's absolute cost grows with n in every regime (it enumerates ~n^4
    rectangles); what the regimes change is the cost relative to the ring regime.
    """
    _regime_axes(ax)
    by_grid = defaultdict(dict)
    for r in load("planner_runtime.csv"):
        by_grid[int(r["n"])][float(r["s"])] = float(r["ms_per_call"])
    for n, style in GRID_STYLE.items():
        pts = by_grid.get(n)
        if not pts:
            continue
        base = pts[min(pts)]
        xs = sorted(pts)
        ax.plot(xs, [pts[x] / base for x in xs], **style,
                markerfacecolor="white", markeredgewidth=1.0)
    # Above the grid lines in the Springer build, where it would otherwise
    # vanish into the one at y = 1.
    ax.axhline(1.0, color=INK if SAVE_SUFFIX else MUTED, lw=0.8 if SAVE_SUFFIX else 0.7,
               ls="--", zorder=1.5 if SAVE_SUFFIX else 0)
    ax.set_ylim(0, None)
    ax.set_ylabel("search time per decision\n(relative to $s = 0.04$)")
    ax.legend(loc="upper right", frameon=False, handlelength=2.2,
              bbox_to_anchor=(1.0, 0.93))
    tidy(ax)


def _draw_onset(ax, unit="frames"):
    """Training needed by 8x8 PPO from scratch to reach 95% of the planner.

    `unit` is "frames" (exact, from the logs) or "episodes" (frames divided by
    the trained policy's mean episode length). A run that never reaches the
    threshold is drawn hollow at its budget with an arrow: the true value lies
    above it, not at it.
    """
    _regime_axes(ax)
    rows = sorted(load("onset_8x8.csv"), key=lambda r: float(r["s"]))
    key = "frames" if unit == "frames" else "episodes"
    style = GRID_STYLE[8]
    hit = [(float(r["s"]), float(r[key])) for r in rows if r["reached"] == "1"]
    miss = [(float(r["s"]), float(r[key])) for r in rows if r["reached"] != "1"]
    if hit:
        # Several runs per cost: a marker for each, and the line through the
        # geometric mean of the runs that reached the threshold at each cost
        # (the axis is logarithmic), so two runs at one s do not zigzag.
        by_s = defaultdict(list)
        for x, y in hit:
            by_s[x].append(y)
        xs = sorted(by_s)
        means = [math.exp(sum(math.log(y) for y in by_s[x]) / len(by_s[x])) for x in xs]
        ax.plot(xs, means, color=style["color"], ls=style["ls"], lw=1.4,
                label=r"PPO, $8\times8$ (mean of runs)")
        hx, hy = zip(*hit)
        # Filled: reached the threshold. Censored runs, below, are hollow.
        ax.plot(hx, hy, ls="none", marker=style["marker"], ms=4.5,
                markerfacecolor=style["color"], markeredgecolor=style["color"],
                markeredgewidth=1.0)
        hit = list(zip(xs, means))   # censored runs attach to the mean line
    for x, y in miss:
        # Joined, dotted, to the nearest reached point in s -- not to the last
        # one -- so a censored run at either end of the axis attaches to its
        # own neighbor.
        if hit:
            nx, ny = min(hit, key=lambda p: abs(math.log(p[0] / x)))
            # A second run at the same cost that did reach the threshold is
            # its own point, not a neighbor to join with a vertical line.
            if nx != x:
                ax.plot([nx, x], [ny, y], color=style["color"], ls=":", lw=1.0)
        ax.plot([x], [y], ls="none", marker=style["marker"], ms=5,
                markerfacecolor="white", markeredgecolor=style["color"], markeredgewidth=1.2)
        ax.annotate("", xy=(x, y * 2.2), xytext=(x, y * 1.15),
                    arrowprops=dict(arrowstyle="->", color=style["color"], lw=0.9))
        # Label on the side facing the middle of the axis, so it stays inside.
        left_end = x < math.sqrt(0.033 * 1.35)
        ax.text(x * (1.08 if left_end else 0.93), y * 1.6, "not reached\nin budget",
                ha="left" if left_end else "right", va="center", fontsize=6, color=MUTED,
                **(dict(bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none",
                                  alpha=0.85)) if SAVE_SUFFIX else {}))
    ax.set_yscale("log")
    ax.set_ylabel(f"{unit} to reach 95% of planner")
    lo = min(y for _, y in hit + miss)
    # Lower limit on a power of ten, so the lowest point has a labelled tick
    # beneath it rather than floating above an unlabelled axis.
    top = 8.0 if SAVE_SUFFIX else 3.6   # room for the note under the regime labels
    ax.set_ylim(10 ** math.floor(math.log10(lo / 1.5)), max(y for _, y in hit + miss) * top)
    ax.legend(loc="upper left", frameon=False, handlelength=2.2,
              bbox_to_anchor=(0.0, 0.93))
    tidy(ax)


def fig_runtime():
    """The planner's search time per episode against s, one curve per grid size.

    The companion to fig_difficulty: same x-axis, same regime bands. There the
    score falls smoothly across s = 1/8; here the cost of finding the barrier
    jumps by more than an order of magnitude at the same boundary.
    """
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    _draw_runtime(ax)
    save(fig, "carl_runtime.pdf")


def fig_onset():
    """Frames 8x8 PPO needs, from scratch, to come within 5% of the planner."""
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    _draw_onset(ax, unit="frames")
    save(fig, "carl_onset.pdf")


def fig_structure():
    """Two panels against s: what finding a barrier costs the planner (a) and
    what learning one costs PPO (b). Both rise where the score does not jump."""
    if SAVE_SUFFIX:
        # Stacked at full text width on one shared s axis: each panel is twice
        # as wide as side by side, and the regime boundaries line up from (a)
        # to (b). The regime names and the s label are given once.
        fig, (a, b) = plt.subplots(2, 1, figsize=(TEXT, 4.3), sharex=True,
                                   gridspec_kw=dict(hspace=0.12))
        _draw_runtime(a)
        _draw_onset(b, unit="frames")
        a.set_xlabel("")
        a.tick_params(labelbottom=False)
        for t in list(b.texts):
            if t.get_text() in ("ring", "edge-closed", "refuge"):
                t.remove()
        # The headroom _draw_onset leaves under the regime names is not needed.
        b.set_ylim(b.get_ylim()[0], b.get_ylim()[1] / 2.5)
        for ax, tag in ((a, "a"), (b, "b")):
            ax.text(-0.135, 1.0, tag, transform=ax.transAxes, fontsize=10,
                    fontweight="bold", va="top", ha="right")
        # White legend boxes, as in the trees-saved figure: a regime boundary
        # runs under (b)'s.
        for ax in (a, b):
            leg = ax.get_legend()
            leg.set_frame_on(True)
            leg.get_frame().set(facecolor="white", edgecolor="none", alpha=0.85)
        fig.align_ylabels((a, b))
        save(fig, "carl_structure.pdf")
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXT, 2.6))
    _draw_runtime(a)
    _draw_onset(b, unit="frames")
    for ax, tag in ((a, "a"), (b, "b")):
        # High enough to clear the top of the rotated y-axis label.
        ax.text(-0.19, 1.13, tag, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="bottom")
    fig.tight_layout(w_pad=2.0)
    save(fig, "carl_structure.pdf")


# ==========================================================================
# 2. PPO against the planner, twelve cells
# ==========================================================================

def fig_ppo_vs_planner():
    """Grouped bars plus the paired margin, which is the statistic that matters.

    The two agents ran the same episodes -- same ignition cell, same starting
    position -- so the per-episode difference cancels the very large variance
    that comes from where the fire starts. The bars give the absolute level; the
    lower panel gives the margin with its standard error.
    """
    rows = load("sweep.csv")
    costs = sorted({float(r["tc"]) for r in rows})
    grids = sorted({int(r["n"]) for r in rows})

    saved = {(int(r["n"]), float(r["tc"]), r["agent"]): 100 * float(r["saved"])
             for r in rows}
    margin = {(int(r["n"]), float(r["tc"])): (float(r["paired_mean"]),
                                              float(r["paired_se"]))
              for r in rows if r["agent"] == "ppo" and r["paired_mean"] != ""}

    fig, axes = plt.subplots(2, 1, figsize=(TEXT * 0.86, 3.2), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.3, 1], hspace=0.16))
    top, bot = axes

    # One slot per (cost, grid) with a gap between cost groups, so the reader
    # compares agents within a bar pair and grids within a group.
    width, xs, labels, ticks = 0.38, [], [], []
    x = 0.0
    for cost in costs:
        group_start = x
        for n in grids:
            for i, agent in enumerate(("planner", "ppo")):
                v = saved.get((n, cost, agent))
                if v is None:
                    continue
                top.bar(x + i * width, v, width * 0.94,
                        color=AGENT_COLOR[agent],
                        edgecolor="white", linewidth=0.8,
                        hatch="" if agent == "planner" else "///",
                        label=({"planner": "planner", "ppo": "PPO"}[agent]
                               if (cost == costs[0] and n == grids[0]) else None))
            m = margin.get((n, cost))
            if m is not None:
                bot.errorbar(x + width / 2, -m[0], yerr=m[1], fmt="o", ms=3.5,
                             lw=1.2, capsize=2,
                             color=GRID_STYLE[n]["color"])
            ticks.append(x + width / 2)
            labels.append(rf"${n}$")
            x += 2 * width + 0.12
        xs.append((group_start, x - 0.12))
        x += 0.35

    top.set_xticks(ticks)
    top.set_ylabel("trees saved (%)")
    top.set_ylim(0, 105)
    top.legend(loc="upper right", frameon=False, ncol=2, handlelength=1.6)
    tidy(top)

    bot.axhline(0, color=MUTED, lw=0.8)
    bot.set_ylabel("PPO $-$ planner\n(points)")
    bot.set_xticks(ticks)
    bot.set_xticklabels(labels)
    tidy(bot)

    # Cost group labels under the axis.
    for (a, b), cost in zip(xs, costs):
        bot.annotate(rf"$t_c=t_m={cost:g}$", xy=((a + b) / 2, -0.38 if SAVE_SUFFIX else -0.28),
                     xycoords=("data", "axes fraction"), ha="center",
                     va="top", fontsize=7.5, color=INK)
    bot.set_xlabel("")
    bot.annotate("grid size", xy=(-0.012, -0.12), xycoords="axes fraction",
                 ha="right", va="top", fontsize=6.5, color=MUTED)
    save(fig, "carl_ppo_vs_planner.pdf")


# ==========================================================================
# 3. Training curves
# ==========================================================================

def fig_margins():
    """PPO minus planner on paired episodes, every grid size and action cost.

    The paired margin is the statistic that matters, and the scores themselves
    are already in the trees-saved figure, so this draws the margin alone, by
    s = t_c + t_m as the rest of the paper does, over every cost with all three
    grid sizes (the sweep and the two ladders past s = 1). The band is the tie
    zone of the tables (a win must clear half a point as well as twice its
    standard error); the error bars are episode-to-episode, one training run per
    setting, so the second 8x8 seed is drawn beside its first to show what a
    seed moves.
    """
    def margins(rows):
        # paired_mean is planner minus agent; plotted the other way round.
        return {(int(r["n"]), round(float(r["s"]), 3)):
                (-float(r["paired_mean"]), float(r["paired_se"]))
                for r in rows if r["agent"] == "ppo" and r["paired_mean"] != ""}

    first = margins(load("sweep.csv") + load("beyond.csv"))
    second = margins([r for r in load("extra_8x8.csv")
                      if r["ckpt"].endswith("_r10_seed2_policy.pth")])
    costs = sorted({s for _, s in first})
    grids = sorted({n for n, _ in first})
    off = {8: -0.27, 16: 0.0, 32: 0.27}

    fig, ax = plt.subplots(figsize=(TEXT, 2.3))
    ax.axhspan(-0.5, 0.5, color="#000000", alpha=0.06, lw=0, zorder=0)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    for i in range(1, len(costs)):
        ax.axvline(i - 0.5, color="#DDDDDD", lw=0.6, zorder=0)
    for n in grids:
        st = GRID_STYLE[n]
        pts = [(i + off[n], *first[(n, s)]) for i, s in enumerate(costs) if (n, s) in first]
        xs, ys, es = zip(*pts)
        ax.errorbar(xs, ys, yerr=es, ls="none", marker=st["marker"], ms=4.6,
                    color=st["color"], markeredgecolor=INK, markeredgewidth=0.5,
                    elinewidth=0.9, capsize=1.8, label=st["label"], zorder=3)
    pts = [(i + off[8] - 0.13, *second[(8, s)]) for i, s in enumerate(costs)
           if (8, s) in second]
    if pts:
        xs, ys, es = zip(*pts)
        ax.errorbar(xs, ys, yerr=es, ls="none", marker=GRID_STYLE[8]["marker"],
                    ms=4.2, markerfacecolor="white", color=GRID_STYLE[8]["color"],
                    markeredgewidth=0.9, elinewidth=0.8, capsize=1.6,
                    label=r"$8\times8$, second seed", zorder=3)
    ax.set_xticks(range(len(costs)))
    ax.set_xticklabels([f"{s:g}" for s in costs])
    ax.set_xlim(-0.6, len(costs) - 0.4)
    ax.set_xlabel(r"action cost $s = t_c + t_m$  (CA updates)")
    ax.set_ylabel("PPO $-$ planner\n(points of grid area)")
    ax.legend(loc="lower left", ncol=4, frameon=False, handlelength=1.2,
              columnspacing=1.0, fontsize=7)
    lo = min(y - e for _, (y, e) in first.items())
    ax.set_ylim(min(lo, -3.5) - 1.2, 4.6)
    tidy(ax)
    ax.grid(False, axis="x")
    save(fig, "carl_margins.pdf")


def fig_training_curves():
    """Trees saved against frames, one panel per grid size.

    Only the runs whose checkpoint survived are drawn solid; the two 32x32
    stages that collapsed and had to be repeated are drawn faint, because the
    collapse is part of what the warm-start ladder costs and hiding it would
    misrepresent the method.
    """
    rows = load("training_curves.csv")
    grids = sorted({int(r["n"]) for r in rows})
    costs = sorted({float(r["tc"]) for r in rows})
    # Cost is the within-panel dimension here, so it gets its own fixed ramp --
    # one hue, light to dark, because cost is an ordered magnitude.
    ramp = {c: shade for c, shade in
            zip(costs, ["#9ecae1", "#4292c6", "#2171b5", "#08306b"])}

    series = defaultdict(list)
    for r in rows:
        series[(int(r["n"]), float(r["tc"]), r["run"], r["kept"] == "1")].append(
            (int(r["frames"]), 100 * float(r["eval_saved"])))

    fig, axes = plt.subplots(1, len(grids), figsize=(TEXT, 1.95), sharey=True)
    for ax, n in zip(axes, grids):
        for (m, cost, run, kept), pts in sorted(series.items()):
            if m != n:
                continue
            xs, ys = zip(*sorted(pts))
            xs = [v / 1e6 for v in xs]
            ax.plot(xs, ys, color=ramp[cost], lw=1.3 if kept else 0.8,
                    alpha=1.0 if kept else 0.35,
                    ls="-" if kept else ":",
                    label=f"{cost:g}" if kept else None)
        ax.set_title(rf"${n}\times{n}$", pad=3)
        ax.set_xlabel("frames (M)")
        ax.set_ylim(0, 100)
        tidy(ax)
    axes[0].set_ylabel("trees saved (%)")
    # One figure-level legend below the panels: an in-panel box would have to
    # sit on top of a curve in every panel.
    handles = [Line2D([], [], color=ramp[c], lw=1.3, label=f"$t_c=t_m={c:g}$")
               for c in costs]
    handles.append(Line2D([], [], color=MUTED, lw=0.9, ls=":", alpha=0.6,
                          label="stage that collapsed and was repeated"))
    fig.legend(handles=handles, loc="lower center",
               ncol=3 if SAVE_SUFFIX else len(handles), frameon=False,
               bbox_to_anchor=(0.5, -0.33 if SAVE_SUFFIX else -0.30), handlelength=1.8,
               columnspacing=1.4)
    save(fig, "carl_training_curves.pdf")


# ==========================================================================
# 4. The backbone
# ==========================================================================

def fig_architecture():
    """The equivariant backbone, drawn to make the N-independence visible.

    The point the diagram has to carry is that nothing downstream of the trunk
    knows N: the three readouts are a gather and two global pools, all
    parameter-free, so the same weights run at any grid size.
    """
    fig, ax = plt.subplots(figsize=(TEXT * 0.92, 2.45))
    # The axes fill the figure, so one x unit is a known fraction of the width
    # and the boxes below can be sized against their text rather than guessed.
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 14.0)
    ax.set_ylim(-0.6, 4.0)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#FFFFFF", fs=6.8):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=MUTED,
                               linewidth=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=INK, linespacing=1.4)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8,
                                    shrinkA=1, shrinkB=1))

    MID = 1.75  # the trunk's centre line

    # Input: four planes, offset so they read as a stack.
    for i, col in enumerate(["#E8E8E8", "#C7E9C0", "#FDBE85", "#C6DBEF"]):
        ax.add_patch(Rectangle((0.15 + 0.13 * i, 0.85 + 0.26 * i), 0.95, 0.95,
                               facecolor=col, edgecolor=MUTED, linewidth=0.7))
    ax.text(0.9, 3.35, "dirt / tree / fire\n+ agent plane", ha="center",
            fontsize=6.4, color=MUTED, linespacing=1.4)
    ax.text(0.9, 0.5, r"$4 \times N \times N$", ha="center", fontsize=6.4)

    arrow(1.75, MID, 2.25, MID)
    box(2.25, MID - 0.55, 0.9, 1.1, "stem\n" r"$3{\times}3$")
    arrow(3.15, MID, 3.6, MID)
    box(3.6, MID - 0.6, 4.0, 1.2,
        "residual blocks\n"
        r"6 dense $+$ dilations 2, 4, 8" "\n"
        r"receptive field $= 3 + 4\sum d$", fs=6.5)
    ax.text(5.6, MID - 0.85, r"$C \times N \times N$", ha="center", va="top",
            fontsize=6.4, color=MUTED)

    # Three readouts plus the time embedding, each on its own row. The readouts
    # carry no weights, which is the claim the figure exists to make, so they
    # get a bracket saying so rather than per-box captions that would crowd
    # into the row beneath.
    reads = [(3.35, "gather at agent cell"),
             (2.45, "global mean pool"),
             (1.55, "global max pool"),
             (0.55, r"time $\phi$ $\rightarrow$ embedding")]
    for y, label in reads:
        box(8.3, y - 0.3, 2.6, 0.6, label, fc="#EAF2FA", fs=6.4)
        arrow(10.9, y, 11.5, MID)
    for y in (3.35, 2.45, 1.55):
        arrow(7.6, MID, 8.3, y)
    ax.text(9.6, 3.78, "three readouts, no parameters", ha="center", va="bottom",
            fontsize=6.0, color=MUTED)

    box(11.5, MID - 0.5, 0.85, 1.0, "MLP")
    arrow(12.35, MID + 0.25, 12.8, 3.05)
    arrow(12.35, MID - 0.25, 12.8, 0.8)
    box(12.8, 2.8, 1.0, 0.5, r"$\pi(a \,|\, s)$", fc="#FDE0C5")
    box(12.8, 0.55, 1.0, 0.5, r"$V(s)$", fc="#D6E9D6")

    ax.text(7.0, -0.42, "every weight is a shared kernel or acts on a "
            "fixed-width readout: none depends on $N$",
            ha="center", va="bottom", fontsize=6.4, color=INK)
    save(fig, "carl_arch_equiv.pdf")


# ==========================================================================
# Rollout figures -- these need the environment
# ==========================================================================

def _rollout(n, tc, tm, ckpt, seed, planner=False, planted=None, oracle=False):
    """Roll one greedy episode and return its grids and agent positions.

    `planted` is a STEERCA_PLANTED config string; with it the reset draws a planted
    instance, and the episode also returns the certified cut cells and the
    oracle's score. `oracle=True` plays the certified cut instead of a policy.

    Imports torch lazily: the CSV-driven figures must stay runnable without it.
    """
    sys.path.insert(0, ROOT)
    import torch
    from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
    import steerca.firebreak.env as ff
    from steerca.evaluate import make_ppo_policy
    from steerca.firebreak.planner import BarrierPlannerPolicy

    device = torch.device("cpu")
    from steerca.firebreak.planted import PlantedConfig, PlantedOraclePolicy
    ff.N, ff.TC, ff.TM = n, tc, tm
    ff.PLANTED_CFG = PlantedConfig.from_string(planted) if planted else None
    env = ff.ForestFireEnv(device=device, batch_size=[1])
    torch.manual_seed(seed)
    env.set_seed(seed)
    td = env.reset()
    if oracle:
        policy = PlantedOraclePolicy()
    elif planner:
        policy = BarrierPlannerPolicy()
    else:
        policy = make_ppo_policy(ckpt, "equiv", device, n)
    gap, oracle_saved = None, None
    if planted:
        g = td["params", "gap"][0]
        gap = g[g[:, 0] >= 0].tolist()
        oracle_saved = float(td["params", "oracle_saved"][0])

    grids = [td["grid"][0].clone()]
    pos = [td["agent_state"][0].clone()]
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for _ in range(ff.episode_cap(n, tc, tm)):
            out = env.step(policy(td))
            nxt = out["next"]
            grids.append(nxt["grid"][0].clone())
            pos.append(nxt["agent_state"][0].clone())
            if bool(nxt["done"].reshape(-1)[0]):
                break
            td = step_mdp(out)
    if planted:
        return grids, pos, gap, oracle_saved
    return grids, pos


# The environment's own palette, so a still frame in the paper matches the
# rendered episodes exactly. draw_forest_fire itself is not reused: it composes
# a GIF frame with a header band of step and reward readouts, which is caption
# furniture a figure panel does not want.
DOZER = "#F5C518"
# Two kinds of dirt that the environment does not tell apart but a reader must:
# a cell the fire burned, and a cell the agent cut. Only the planted figure
# passes them, so the episode figures keep the environment's own palette.
BURNT, CUT_CELL = "#7A7A7A", "#141414"


def _history(grids, k):
    """Burnt and cut masks after step k of an episode, from its frames.

    A fire cell lasts one CA update and becomes dirt; a cut turns a tree to dirt
    without it ever burning. Telling them apart frame by frame is exact while no
    action spans two updates (t_c, t_m < 1), which holds for every planted run.
    """
    import torch
    fire_seen = torch.zeros_like(grids[0][2])
    cut = torch.zeros_like(grids[0][2])
    for a, b in zip(grids[:k], grids[1:k + 1]):
        fire_seen |= a[2]
        cut |= a[1] & ~b[1] & ~b[2] & ~a[2]
    fire_seen |= grids[k][2]
    dirt = grids[k][0]
    return fire_seen & dirt, cut & dirt


def _panel(ax, grid, pos, marks=None, burnt=None, cut=None):
    import numpy as np
    from matplotlib.colors import BoundaryNorm, ListedColormap
    g = grid.detach().cpu().numpy()
    flat = np.zeros(g.shape[-2:], dtype=int)
    for idx in (0, 1, 2):
        flat[g[idx] == 1] = idx
    colors = list(CELL_COLORS)
    if burnt is not None:
        flat[burnt.detach().cpu().numpy()] = 3
        flat[cut.detach().cpu().numpy()] = 4
        colors += [BURNT, CUT_CELL]
    ax.imshow(flat, cmap=ListedColormap(colors),
              norm=BoundaryNorm(np.arange(-0.5, len(colors)), len(colors)),
              interpolation="nearest")
    if marks:
        # The certified cut cells, as hollow squares: visible whether the cell
        # is still tree, already cut to dirt, or burning.
        for r, c in marks:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                       edgecolor="#1F3A93", linewidth=0.9))
    ax.plot(int(pos[1]), int(pos[0]), marker="o", ms=3.2,
            markerfacecolor=DOZER, markeredgecolor="#333333", markeredgewidth=0.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_edgecolor("#BBBBBB"); s.set_linewidth(0.6)


def _trees(grid):
    return int(grid[1].sum()) / (grid.shape[-1] * grid.shape[-2])


def fig_episode():
    """A filmstrip of one 16x16 episode at t_c = t_m = 0.2.

    The middle of the difficulty range, and the most informative place to look:
    no ring closes, so what the agent builds is a partial barrier anchored on
    the grid edge, and the still frames show that decision being made.
    """
    ckpt = os.path.join(MODELS, "ppo_equiv_c96_b6_512_64_1hot_16x16_t0.2_0.2_cut0.5_r10_policy.pth")
    grids, pos = _rollout(16, 0.2, 0.2, ckpt, seed=3)
    last = len(grids) - 1
    picks = sorted({0, last // 5, 2 * last // 5, 3 * last // 5, last})
    # Sized for the full text width: this goes in a figure* spanning both
    # columns, and a panel drawn at column width would have its labels
    # magnified twofold on the page.
    fig, axes = plt.subplots(1, len(picks),
                             figsize=(TEXT, TEXT / len(picks) + 0.40))
    for ax, p in zip(axes, picks):
        _panel(ax, grids[p], pos[p])
        ax.set_title(f"step {p}\n{100 * _trees(grids[p]):.0f}% tree",
                     fontsize=7.5, pad=2.5, linespacing=1.3)
    # A Springer build gives the panels the full width.
    fig.subplots_adjust(wspace=0.12, **(dict(left=0.01, right=0.99) if SAVE_SUFFIX else {}))
    save(fig, "carl_episode_strip.pdf")


# What the planted figure draws, shared with collect.py's `planted` target so
# the instances it shows are the ones that target scored. Both tiers at
# t_c = t_m = 0.2 and 32 x 32, the size the text reports them at.
PLANTED_COST = 0.2
PLANTED_N = 32
POCKET_CKPT = "ppo_equiv_c96_b9d2-4-8_512_64_1hot_32x32_t0.2_0.2_cut0.5_planted_r10_policy.pth"
GRAPH_CFG = "family=graph,cuts=3-6,scuts=6"
# Stage 2 of scripts/run_curriculum.sh: the four-corridor target, warm-started
# from the acyclic stage. Its plain checkpoint is the best evaluation.
GRAPH_CKPT = ("ppo_equiv_c96_b9d2-4-8_512_64_1hot_32x32_t0.2_0.2_cut0.5"
              "_planted-graph-cuts3-6-scuts6_r10_policy.pth")


def planted_modes(rows):
    """The two graph instances the figure shows, one from each group.

    The curriculum policy's score on an instance, as a fraction of the
    certificate, falls in two loose groups with a dip between them, not around
    one mean. The sorted scores are split where the within-group variance is
    least (Otsu's split, in one dimension: robust to a stray outlier, which a
    largest-gap split is not), and each group contributes its median instance
    (the lower one of an even group), so each is typical of its own group
    rather than of a mean that falls between them.
    """
    pts = sorted((float(r["frac"]), int(r["seed"])) for r in rows
                 if r["set"] == "graph" and r["agent"] == "ppo")

    def sse(xs):
        m = sum(x for x, _ in xs) / len(xs)
        return sum((x - m) ** 2 for x, _ in xs)

    cut = min(range(1, len(pts)), key=lambda i: sse(pts[:i]) + sse(pts[i:]))
    low, high = pts[:cut], pts[cut:]
    return high[(len(high) - 1) // 2][1], low[(len(low) - 1) // 2][1], (low, high)


def planted_pocket_seed(rows):
    """The first pocket instance on which PPO matches the certificate: what it
    does on the 55 of 60 paired episodes it ties."""
    return min(int(r["seed"]) for r in rows if r["set"] == "pocket"
               and r["agent"] == "ppo" and float(r["frac"]) >= 0.999)


def fig_planted():
    """Planted instances: the pocket tier, and two graph instances, one from
    each mode of the curriculum policy's score, each played by the certificate
    and by PPO.

    Burnt cells are drawn apart from bare dirt and cut cells apart from both,
    because on a graph instance most of the grid is dirt from the start and the
    fire's path and the agent's cuts would otherwise vanish into it. Rows of one
    instance share their column steps, taken from the longer episode; a row
    that has already ended repeats its final frame and says so.
    """
    seeds = load("planted_seeds.csv")
    hi_seed, lo_seed, _ = planted_modes(seeds)
    frac = {(r["set"], r["agent"], int(r["seed"])): float(r["frac"]) for r in seeds}
    M = MODELS
    n, c = PLANTED_N, PLANTED_COST
    pocket = planted_pocket_seed(seeds)

    blocks = [[("pocket\nPPO", _rollout(n, c, c, os.path.join(M, POCKET_CKPT),
                                        seed=pocket, planted="1"))]]
    for tag, seed in (("A", hi_seed), ("B", lo_seed)):
        blocks.append([
            (f"{tag}: certificate", _rollout(n, c, c, None, seed=seed,
                                              planted=GRAPH_CFG, oracle=True)),
            (f"{tag}: PPO, {100 * frac[('graph', 'ppo', seed)]:.0f}%",
             _rollout(n, c, c, os.path.join(M, GRAPH_CKPT), seed=seed,
                      planted=GRAPH_CFG))])

    ncol = 5
    nrow = sum(len(b) for b in blocks)
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(TEXT, nrow * (TEXT / ncol + 0.27)))
    i = 0
    for block in blocks:
        span = max(len(run[0]) for _, run in block) - 1
        picks = [round(k * span / (ncol - 1)) for k in range(ncol)]
        for label, (grids, pos, gap, cert) in block:
            for j, step in enumerate(picks):
                ax = axes[i][j]
                k = min(step, len(grids) - 1)
                burnt, cut = _history(grids, k)
                _panel(ax, grids[k], pos[k], marks=gap, burnt=burnt, cut=cut)
                when = f"step {step}" if step == k else f"ended, step {k}"
                sep = "\n" if SAVE_SUFFIX else "  "
                title = f"{when}{sep}{100 * _trees(grids[k]):.1f}% tree"
                if j == ncol - 1:
                    title += f"\n{100 * _trees(grids[k]) / cert:.0f}% of certificate"
                ax.set_title(title, fontsize=6.6, pad=2, linespacing=1.25)
                if j == 0:
                    ax.set_ylabel(label, fontsize=7.2)
            i += 1
    if SAVE_SUFFIX:
        fig.subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.01,
                            wspace=0.08, hspace=0.55)
    else:
        fig.subplots_adjust(wspace=0.1, hspace=0.42)
    save(fig, "carl_planted_episodes.pdf")


def fig_regimes():
    """Final frames at three action costs: the difficulty axis, seen.

    Same grid size, same seed, same initial state -- only s differs, so the
    three panels differ in exactly the parameter the difficulty section is
    about.
    """
    cells = [
        (0.06, "ppo_equiv_c96_b6_512_64_1hot_16x16_t0.06_0.06_cut0.5_r10_policy.pth"),
        (0.2, "ppo_equiv_c96_b6_512_64_1hot_16x16_t0.2_0.2_cut0.5_r10_policy.pth"),
        (0.3, "ppo_equiv_c96_b6_512_64_1hot_16x16_t0.3_0.3_cut0.5_r10_policy.pth"),
    ]
    fig, axes = plt.subplots(1, len(cells),
                             figsize=(COL, COL / len(cells) + 0.42))
    for ax, (cost, name) in zip(axes, cells):
        grids, pos = _rollout(16, cost, cost, os.path.join(MODELS, name),
                              seed=3)
        _panel(ax, grids[-1], pos[-1])
        ax.set_title(rf"$s = {2 * cost:g}$" "\n"
                     f"{100 * _trees(grids[-1]):.0f}% saved",
                     fontsize=6.4, pad=2, linespacing=1.3)
    fig.subplots_adjust(wspace=0.1, **(dict(left=0.01, right=0.99) if SAVE_SUFFIX else {}))
    save(fig, "carl_regimes.pdf")


# ==========================================================================
# Tables -- generated into sections/tab_*.tex so the LaTeX never holds a
# transcribed number. \input'ed from the section that discusses them.
# ==========================================================================

SECTIONS = os.path.join(HERE, "tables")


def write_table(name, body):
    os.makedirs(SECTIONS, exist_ok=True)
    path = os.path.join(SECTIONS, f"tab_{name}.tex")
    with open(path, "w") as fh:
        fh.write("% Generated by paper/figures.py -- do not edit by hand.\n")
        fh.write(body.rstrip() + "\n")
    print(f"  {os.path.relpath(path, ROOT)}")


def tab_difficulty():
    rows = [r for r in load("planner_ceiling.csv") if r["agent"] == "planner"]
    grids = sorted({int(r["n"]) for r in rows})
    ss = sorted({float(r["s"]) for r in rows})
    val = {(int(r["n"]), float(r["s"])): 100 * float(r["saved"]) for r in rows}
    # The share of episodes the fire burned out without ever reaching the grid
    # edge: enclosed by a barrier that does not use it. Blank in a CSV collected
    # before steerca.evaluate recorded it.
    free = {(int(r["n"]), float(r["s"])): 100 * float(r["edge_free"])
            for r in rows if r.get("edge_free", "") != ""}

    nl = r" \\"
    g = len(grids)
    sizes = [rf"${n}\times{n}$" for n in grids]
    body = [r"\begin{tabular}{l" + "c" * (2 * g) + "}", r"\toprule",
            rf"& \multicolumn{{{g}}}{{c}}{{trees saved (\%)}} & "
            rf"\multicolumn{{{g}}}{{c}}{{enclosed without the edge (\%)}}" + nl,
            rf"\cmidrule(lr){{2-{1 + g}}} \cmidrule(lr){{{2 + g}-{1 + 2 * g}}}",
            " & ".join([r"$s = t_c + t_m$"] + sizes * 2) + nl, r"\midrule"]
    # The regime boundaries fall between sampled costs, so they are drawn as
    # ruled separators rather than as a note on a row that does not sit on them.
    # s = 1/3 is Feldheim and Hod's firefighter bound: from there on the fire
    # reaches the grid edge whatever the agent does.
    marks = [(0.125, "0.125", "no square ring closes above here"),
             (1 / 3, "1/3", "no barrier closes without the grid edge above here"),
             (1.0, "1", "no barrier keeps pace with the fire above here")]
    prev = None
    for s_ in ss:
        for at, shown, label in marks:
            if prev is not None and prev < at <= s_:
                body += [r"\midrule",
                         rf"\multicolumn{{{1 + 2 * g}}}{{l}}"
                         rf"{{\footnotesize\itshape $s = {shown}$: {label}}}" + nl,
                         r"\midrule"]
        cells = [f"{val[(n, s_)]:.1f}" if (n, s_) in val else "---" for n in grids]
        cells += [f"{free[(n, s_)]:.0f}" if (n, s_) in free else "---" for n in grids]
        body.append(f"${s_:g}$ & " + " & ".join(cells) + nl)
        prev = s_
    body += [r"\bottomrule", r"\end{tabular}"]
    write_table("difficulty", "\n".join(body))


def tab_runtime():
    """The planner's raw cost per decision and the number of decisions per
    episode, the two factors of its per-episode search time, across s."""
    rows = load("planner_runtime.csv")
    grids = sorted({int(r["n"]) for r in rows})
    ss = sorted({float(r["s"]) for r in rows})
    ms = {(int(r["n"]), float(r["s"])): float(r["ms_per_call"]) for r in rows}
    calls = {(int(r["n"]), float(r["s"])): float(r["calls_per_episode"]) for r in rows}
    nl = r" \\"
    g = len(grids)
    head1 = (r"& \multicolumn{%d}{c}{ms per decision} & \multicolumn{%d}{c}{plans per episode}"
             % (g, g)) + nl
    rule = rf"\cmidrule(lr){{2-{1 + g}}} \cmidrule(lr){{{2 + g}-{1 + 2 * g}}}"
    head2 = " & ".join([r"$s$"] + [rf"${n}\times{n}$" for n in grids] * 2) + nl
    body = [r"\begin{tabular}{l" + "c" * (2 * g) + "}", r"\toprule", head1, rule,
            head2, r"\midrule"]
    marks = [(0.125, "spiral search becomes necessary"),
             (1.0, "no barrier keeps pace with the fire above here")]
    prev = None
    for s_ in ss:
        for at, label in marks:
            if prev is not None and prev < at <= s_:
                body += [r"\midrule",
                         rf"\multicolumn{{{1 + 2 * g}}}{{l}}"
                         rf"{{\footnotesize\itshape $s = {at:g}$: {label}}}" + nl,
                         r"\midrule"]
        c1 = [(f"{ms[(n, s_)]:.1f}" if ms[(n, s_)] < 100 else f"{ms[(n, s_)]:.0f}")
              if (n, s_) in ms else "---" for n in grids]
        c2 = [f"{calls[(n, s_)]:.1f}" if (n, s_) in calls else "---" for n in grids]
        body.append(f"${s_:g}$ & " + " & ".join(c1 + c2) + nl)
        prev = s_
    body += [r"\bottomrule", r"\end{tabular}"]
    write_table("runtime", "\n".join(body))


def _paired_table(csv_name, table_name):
    """PPO and planner side by side per (cost, grid), bolding a clear margin."""
    rows = load(csv_name)
    grids = sorted({int(r["n"]) for r in rows})
    costs = sorted({float(r["tc"]) for r in rows})
    saved = {(int(r["n"]), float(r["tc"]), r["agent"]): 100 * float(r["saved"])
             for r in rows}
    margin = {(int(r["n"]), float(r["tc"])): (float(r["paired_mean"]),
                                              float(r["paired_se"]))
              for r in rows if r["agent"] == "ppo" and r["paired_mean"] != ""}

    cols = "l" + "cc" * len(grids)
    head1 = r"& " + " & ".join(rf"\multicolumn{{2}}{{c}}{{${n}\times{n}$}}"
                              for n in grids)
    rule = " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}"
                    for i in range(len(grids)))
    head2 = r"$t_c = t_m$ & " + " & ".join(["PPO & planner"] * len(grids))
    body = [rf"\begin{{tabular}}{{{cols}}}", r"\toprule", head1 + r" \\", rule,
            head2 + r" \\", r"\midrule"]
    for c in costs:
        cells = []
        for n in grids:
            p = saved.get((n, c, "ppo"))
            h = saved.get((n, c, "planner"))
            # Bold whichever agent is ahead, on two conditions: the margin
            # clears twice its paired standard error, and it is at least half a
            # point of grid area. The second is not statistical -- with 100
            # paired episodes a 0.15-point gap can clear its error bar while
            # meaning nothing -- so a margin that small is reported as a tie.
            m = margin.get((n, c))
            bold_p = bold_h = False
            if m and abs(m[0]) > 2 * m[1] and abs(m[0]) >= 0.5:
                bold_p, bold_h = m[0] < 0, m[0] > 0
            cells.append(r"\textbf{%.1f}" % p if bold_p else
                         ("---" if p is None else f"{p:.1f}"))
            cells.append(r"\textbf{%.1f}" % h if bold_h else
                         ("---" if h is None else f"{h:.1f}"))
        body.append(f"${c:g}$ & " + " & ".join(cells) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write_table(table_name, "\n".join(body))


def tab_sweep():
    _paired_table("sweep.csv", "sweep")


def tab_beyond():
    """The same comparison past s = 1, from the r10 ladders at s = 1 and 1.2."""
    _paired_table("beyond.csv", "beyond")



def _pct(x):
    return f"{100 * float(x):.1f}" if x != "" else "---"


def _best_final(r):
    """The best evaluation over training, with the final one in parentheses
    where they differ by half a point or more: the best of many noisy
    evaluations is optimistic, and the pair shows by how much."""
    best, final = r["best_saved"], r.get("final_saved", "")
    if best == "" or final == "" or abs(float(best) - float(final)) < 0.005:
        return _pct(best)
    return f"{_pct(best)} ({_pct(final)})"


def tab_baselines():
    """Best trees saved on the four sweep cells: PPO, then A2C and SAC at the
    settings the sanity check found each one needs. Their runs at PPO's own
    settings are in baselines.csv but not shown: another algorithm's
    hyperparameters say nothing about these two."""
    rows = load("baselines.csv")
    get = {(r["set"], r["algo"], r["variant"], int(r["n"]), float(r["tc"])): r
           for r in rows}
    cells = sorted({(int(r["n"]), float(r["tc"])) for r in rows if r["set"] == "sweep"})
    cols = (("sweep", "ppo", "default"), ("tuned", "a2c", "ent0.05"),
            ("tuned", "sac", "scaled"))
    nl = r" \\"
    body = [r"\begin{tabular}{lcccc}", r"\toprule",
            r"grid & $t_c = t_m$ & PPO & A2C & SAC" + nl,
            r"\midrule"]
    for n, c in cells:
        sv = [_best_final(get[k + (n, c)]) if k + (n, c) in get else "---"
              for k in cols]
        body.append(rf"${n}\times{n}$ & ${c:g}$ & " + " & ".join(sv) + nl)
    body += [r"\bottomrule", r"\end{tabular}"]
    write_table("baselines", "\n".join(body))


def _num(x, fmt):
    return "---" if x == "" else format(float(x), fmt)


def tab_robust():
    """The tuned settings tried on the easiest cell, and PPO without its clip
    (A2C's update at PPO's ten epochs, the one run at PPO's settings kept, as a
    controlled ablation): what each saved, and whether its final policy kept
    any entropy. A2C and SAC at PPO's settings, and at those settings with only
    the learning rate changed, are in baselines.csv but not shown."""
    rows = load("baselines.csv")
    nl = r" \\"
    # Two header rows so no column heading is wider than its numbers: the
    # single-row version ran 46pt past the column and into the text beside it.
    body = [r"\begin{tabular}{llccc}", r"\toprule",
            r"& & greedy & final & sampled" + nl,
            r"learner & variant & saved & entropy & saved" + nl,
            r"\midrule"]
    # (learner, variant) -> (learner column, variant column), in table order.
    shown = {("ppo", "default"): ("PPO", "clipped, as trained"),
             ("a2c", "a2c-e10"): ("PPO", "without the clip"),
             ("a2c", "ent0.05"): ("A2C", r"entropy $0.05$"),
             ("a2c", "ent0.2"): ("A2C", r"entropy $0.2$"),
             ("sac", "scaled"): ("SAC", "reward, temperature")}
    pick = {(r["algo"], r["variant"]): r for r in rows
            if int(r["n"]) == 8 and float(r["tc"]) == 0.06
            and r["set"] in ("sweep", "robust", "tuned")}
    last = None
    for key, (who, what) in shown.items():
        if key not in pick:
            continue
        if last is not None and who != last:
            body.append(r"\midrule")
        last = who
        r = pick[key]
        body.append(rf"{who} & {what} & "
                    rf"{_best_final(r)} & {_num(r['final_entropy'], '.2f')} & "
                    rf"{_pct(r['ckpt_saved'])}" + nl)
    body.append(r"\midrule")
    for r in rows:
        if r["set"] == "reference":
            what = {"untrained": "untrained network",
                    "uniform": "uniform random actions"}[r["algo"]]
            body.append(rf"\multicolumn{{2}}{{l}}{{\itshape {what}}} & --- & "
                        rf"{_num(r['ckpt_entropy'], '.2f')} & {_pct(r['ckpt_saved'])}" + nl)
    body += [r"\bottomrule", r"\end{tabular}"]
    write_table("robust", "\n".join(body))


def tab_sanity():
    """The same learners on a task where a felled tree pays at once."""
    # A2C and SAC at PPO's settings are left out, as in the other tables.
    rows = [r for r in load("baselines.csv") if r["set"] == "sanity"
            and (r["algo"] == "ppo" or r["variant"] != "default")]
    label = {"default": "as trained", "ent0.05": r"entropy $0.05$",
             "scaled": "reward, temperature"}
    nl = r" \\"
    body = [r"\begin{tabular}{llcc}", r"\toprule",
            r"& & \multicolumn{2}{c}{eval return}" + nl,
            r"\cmidrule(lr){3-4}",
            r"learner & settings & first & last" + nl, r"\midrule"]
    for r in rows:
        body.append(rf"{r['algo'].upper()} & {label[r['variant']]} & "
                    rf"${float(r['first_return']):+.2f}$ & ${float(r['final_return']):+.2f}$" + nl)
    body += [r"\bottomrule", r"\end{tabular}"]
    write_table("sanity", "\n".join(body))


def tab_warmstart():
    """One solved policy trained on with and without PPO's clip -- a controlled
    ablation at matched settings -- and the gradient check. warmstart.csv calls
    the unclipped update A2C, which is what it is (PPO's actor, critic and
    advantage estimator without the clipped ratio); the table names it by the
    one thing it changes."""
    relabel = {"PPO": "clipped (PPO)", "A2C, 1 epoch": "unclipped, 1 epoch (A2C)",
               "A2C, 10 epochs": "unclipped, 10 epochs"}
    rows = load("warmstart.csv")
    frames = sorted({int(r["frames"]) for r in rows})
    labels, val = [], {}
    for r in rows:
        if r["label"] not in labels:
            labels.append(r["label"])
        val[(r["label"], int(r["frames"]))] = 100 * float(r["eval_saved"])
    g = load("grad_check.csv")[0]
    nl = r" \\"
    head = "update & " + " & ".join(rf"${f / 1000:g}$k" for f in frames)
    body = [r"\begin{tabular}{l" + "c" * len(frames) + "}", r"\toprule",
            r"& \multicolumn{" + str(len(frames)) + r"}{c}{trees saved (\%) after"
            r" $\ldots$ frames}" + nl,
            rf"\cmidrule(lr){{2-{1 + len(frames)}}}", head + nl, r"\midrule"]
    for lab in labels:
        cells = [f"{val[(lab, f)]:.1f}" if (lab, f) in val else "---" for f in frames]
        body.append(f"{relabel.get(lab, lab)} & " + " & ".join(cells) + nl)
    body += [r"\bottomrule", r"\end{tabular}",
             # The note sits BELOW the tabular, not in a spanning cell: any cell
             # wider than the table's natural width hands the excess to the last
             # column, which is what pushed "152k" away from its numbers.
             r"\par\smallskip{\footnotesize\itshape First-update gradients, clipped"
             rf" against unclipped: cosine ${float(g['cosine']):+.4f}$, norms"
             rf" ${float(g['norm_ppo']):.4f}$ and ${float(g['norm_a2c']):.4f}$.\par}}"]
    write_table("warmstart", "\n".join(body))

def tab_throughput():
    """Environment transitions per second, thousands, by batch size, grid size
    and device: the full TorchRL step with random actions."""
    rows = load("env_throughput.csv")
    grids = sorted({int(r["n"]) for r in rows})
    batches = sorted({int(r["batch"]) for r in rows})
    val = {(r["device"], int(r["n"]), int(r["batch"])): float(r["transitions_per_s"]) / 1e3
           for r in rows}
    nl = r" \\"
    head1 = "& " + " & ".join(rf"\multicolumn{{2}}{{c}}{{${n}\times{n}$}}" for n in grids) + nl
    rule = " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(grids)))
    head2 = " & ".join(["environments"] + ["GPU", "CPU"] * len(grids)) + nl

    def fmt(x):
        return "---" if x is None else (f"{x:.1f}" if x < 10 else f"{x:,.0f}".replace(",", "{,}"))

    body = [r"\begin{tabular}{l" + "rr" * len(grids) + "}", r"\toprule", head1, rule,
            head2, r"\midrule"]
    for b in batches:
        cells = []
        for n in grids:
            cells += [fmt(val.get(("cuda", n, b))), fmt(val.get(("cpu", n, b)))]
        body.append(f"${b:,}$".replace(",", "{,}") + " & " + " & ".join(cells) + nl)
    body += [r"\bottomrule", r"\end{tabular}"]
    write_table("throughput", "\n".join(body))


def springer(fn):
    """Run a figure function as a Springer Nature figure.

    The functions are written for IEEEtran (COL = 3.5in, TEXT = 7.16in), and the
    Springer version used to include them shrunk to its 31pc text block, so a
    7.5pt label printed at about 5.4pt. Here the same function draws at the
    Springer widths, in Computer Modern (SN_RC), and saves under an _sn suffix;
    the unsuffixed files stay as they were for paper/scirep/.
    """
    def run():
        global COL, TEXT, SAVE_SUFFIX
        saved = COL, TEXT, SAVE_SUFFIX
        COL, TEXT, SAVE_SUFFIX = 0.78 * SN_TEXT, SN_TEXT, "_sn"
        try:
            with plt.rc_context(SN_RC):
                fn()
        finally:
            COL, TEXT, SAVE_SUFFIX = saved
    run.__doc__ = f"{fn.__name__}, as a Springer Nature figure."
    return run


TABLES = {"difficulty": tab_difficulty, "runtime": tab_runtime, "sweep": tab_sweep,
          "beyond": tab_beyond,
          "baselines": tab_baselines, "robust": tab_robust, "sanity": tab_sanity,
          "warmstart": tab_warmstart, "throughput": tab_throughput}


FIGURES = {
    "difficulty": fig_difficulty,
    "difficulty_sn": fig_difficulty_sn,
    "structure_sn": springer(fig_structure),
    "ppo_vs_planner_sn": springer(fig_ppo_vs_planner),
    "margins_sn": springer(fig_margins),
    "training_curves_sn": springer(fig_training_curves),
    "episode_sn": springer(fig_episode),
    "regimes_sn": springer(fig_regimes),
    "planted_sn": springer(fig_planted),
    "runtime": fig_runtime,
    "onset": fig_onset,
    "structure": fig_structure,
    "ppo_vs_planner": fig_ppo_vs_planner,
    "margins": fig_margins,
    "training_curves": fig_training_curves,
    "architecture": fig_architecture,
    "episode": fig_episode,
    "regimes": fig_regimes,
    "planted": fig_planted,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="*", default=None,
                    help="figure or table names; default is everything")
    args = ap.parse_args()
    names = args.which or list(FIGURES) + ["tables"]
    for name in names:
        if name == "tables":
            print("tables")
            for fn in TABLES.values():
                fn()
        elif name in FIGURES or name in TABLES:
            # A name can be both ("runtime" is a figure and a table); build both.
            print(name)
            try:
                if name in FIGURES:
                    FIGURES[name]()
                if name in TABLES:
                    TABLES[name]()
            except FileNotFoundError as err:
                # Only the rollout figures read checkpoints; without the
                # artifacts the CSV-only figures still build.
                if args.which:
                    raise
                print(f"  skipped: {err.filename} is missing "
                      f"(fetch it with tools/fetch_artifacts.py)")
        else:
            raise SystemExit(f"unknown target {name!r}; have "
                             f"{sorted(set(FIGURES) | set(TABLES))} or 'tables'")


if __name__ == "__main__":
    main()
