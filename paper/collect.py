"""Collect every number the paper quotes into tidy CSVs under ``paper/data/``.

Three products, each independently runnable:

    python paper/collect.py curves    # parse logs/ -> training_curves.csv
    python paper/collect.py sweep     # the 12 r10 cells -> sweep.csv        (GPU)
    python paper/collect.py ceiling   # planner vs action cost -> planner_ceiling.csv
    python paper/collect.py runtime   # planner time per decision -> planner_runtime.csv
    python paper/collect.py beyond    # PPO vs planner at s = 1, 1.2 -> beyond.csv  (GPU)
    python paper/collect.py extra     # extra 8x8 runs (s = 0.04, 0.08, second seeds) vs planner -> extra_8x8.csv  (GPU)
        python paper/collect.py onset     # 8x8 PPO frames/episodes to 95% of planner -> onset_8x8.csv
    python paper/collect.py others    # 8x8 A2C / SAC at their best settings vs planner -> others_8x8.csv  (GPU)
    python paper/collect.py throughput  # environment steps per second, GPU vs CPU -> env_throughput.csv
    python paper/collect.py planted   # per-instance planted scores for the planted figure -> planted_seeds.csv  (CPU)

    python paper/collect.py all

Nothing in ``paper/figures.py`` touches the models, the logs or the environment:
it reads these CSVs only. That is the point -- a figure can be redrawn in a
second without a GPU, and every number in the LaTeX traces to a row here.

`curves` is free and reads only files already on disk. `sweep` needs the r10
checkpoints in ``models/`` and a GPU; `ceiling` is planner-only but runs on the
GPU by default, at 100 episodes, because that is what draws the same starts as
the paired runs (the reset's random draws differ between cpu and cuda), so its
curve passes through the planner's paired scores.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
# The CSVs this writes, and that figures.py reads.
DATA = os.path.join(HERE, "data")
# Checkpoints and training logs come from the SteerCA-artifacts archive, fetched
# with tools/fetch_artifacts.py. STEERCA_ARTIFACTS points elsewhere; a directory
# holding your own models/ and logs/ works too.
ARTIFACTS = os.path.abspath(os.environ.get("STEERCA_ARTIFACTS",
                                           os.path.join(ROOT, "artifacts")))
MODELS = os.path.join(ARTIFACTS, "models")
LOGS = os.path.join(ARTIFACTS, "logs")

# The twelve cells of the r10 sweep: grid size x action cost, tc = tm throughout.
GRIDS = (8, 16, 32)
COSTS = (0.06, 0.12, 0.2, 0.3)

# Architecture tag per grid size, as `arch_for_grid` renders it. Above 16 the
# trunk carries a dilated tail, so the checkpoint name differs.
ARCH = {8: "c96_b6_512_64", 16: "c96_b6_512_64", 32: "c96_b9d2-4-8_512_64"}


def ckpt_path(n, cost):
    return os.path.join(
        MODELS, f"ppo_equiv_{ARCH[n]}_1hot_{n}x{n}_t{cost:g}_{cost:g}_cut0.5_r10_policy.pth")


# --------------------------------------------------------------------------
# Training curves, parsed out of the tqdm stdout streams in logs/
# --------------------------------------------------------------------------

# One tqdm record, `\r`-separated from its neighbours:
#   eval saved  41.1% (return -0.5462), average reward=-0.0103 (init=-0.0103),
#   lr policy:  0.000030:   0%|  | 2000/4000000 [00:12<6:16:28, 177.00it/s]
RECORD = re.compile(
    r"eval saved\s+([\d.]+)%\s+\(return\s+(-?[\d.]+)\)"
    r"(?:\s+([\d.]+)% of oracle)?"
    r",\s+average reward=\s*(-?[\d.]+)\s+\(init=\s*(-?[\d.]+)\)"
    r",\s+lr policy:\s+([\d.]+):.*?(\d+)/(\d+)\s*\[")

# Which log holds which cell. Where a stage was run twice, both are kept and
# `kept` marks the run whose checkpoint survived -- the 32x32 collapses are
# worth showing, not hiding, and the figure code can filter on the flag.
#   (n, cost): [(log basename, kept), ...]
CURVE_LOGS = {}
for _n in GRIDS:
    for _c in COSTS:
        base = f"run_{_n}x{_n}_t{_c:g}_{_c:g}"
        runs = []
        if os.path.exists(os.path.join(LOGS, f"{base}.log")):
            runs.append((f"{base}.log", True))
        if os.path.exists(os.path.join(LOGS, f"{base}_retry.log")):
            # A retry exists only because the first attempt collapsed, so the
            # retry is the run that produced the checkpoint in models/.
            runs = [(name, False) for name, _ in runs]
            runs.append((f"{base}_retry.log", True))
        CURVE_LOGS[(_n, _c)] = runs


def collect_curves(out):
    rows = []
    for (n, cost), runs in sorted(CURVE_LOGS.items()):
        for name, kept in runs:
            path = os.path.join(LOGS, name)
            if not os.path.exists(path):
                print(f"  missing {name}, skipped", file=sys.stderr)
                continue
            # newline="" is load-bearing: universal-newline translation rewrites
            # the `\r`s tqdm separates its records with, collapsing the whole
            # stream into a single line that matches exactly once.
            with open(path, errors="replace", newline="") as fh:
                text = fh.read()
            seen = None
            n_rows = 0
            for rec in re.split(r"[\r\n]", text):
                m = RECORD.search(rec)
                if not m:
                    continue
                saved, ret, oracle, avg, init, lr, frames, total = m.groups()
                frames = int(frames)
                # tqdm reprints its description every batch, so the eval fields
                # are stale between evals (one every 25 batches). Keep the first
                # batch carrying each new (saved, return) pair: that row is the
                # eval itself, and two consecutive evals agreeing to four
                # decimals on both numbers does not happen.
                if (saved, ret) == seen:
                    continue
                seen = (saved, ret)
                rows.append({
                    "n": n, "tc": cost, "tm": cost, "s": round(2 * cost, 6),
                    "run": name[:-4], "kept": int(kept),
                    "frames": frames, "total_frames": int(total),
                    "eval_saved": float(saved) / 100.0,
                    "eval_return": float(ret),
                    "oracle_fraction": float(oracle) / 100.0 if oracle else "",
                    "avg_reward": float(avg), "lr": float(lr),
                })
                n_rows += 1
            print(f"  {name:38s} {n_rows:6d} points")
    write(out, rows)


# --------------------------------------------------------------------------
# Measurements, driven through steerca.evaluate's --csv
# --------------------------------------------------------------------------

def compare(out, n, costs, episodes, device, ckpt=None):
    cmd = [sys.executable, "-m", "steerca.evaluate",
           "--n", str(n), "--episodes", str(episodes), "--device", device,
           "--costs", *[f"{c:g},{c:g}" for c in costs], "--csv", out]
    if ckpt:
        cmd += ["--ckpt", ckpt]
    print("  $ " + " ".join(os.path.relpath(x, ROOT) if x.startswith(ROOT) else x
                            for x in cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def collect_sweep(out, episodes, device):
    """PPO against the planner on identical episodes, all twelve cells.

    One process per (n, cost) rather than one per grid size: the checkpoint is
    per-cell, and --ckpt is a single argument.
    """
    fresh(out)
    for n in GRIDS:
        for cost in COSTS:
            ckpt = ckpt_path(n, cost)
            if not os.path.exists(ckpt):
                print(f"  no checkpoint for {n}x{n} @ {cost:g}, planner only",
                      file=sys.stderr)
                ckpt = None
            compare(out, n, [cost], episodes, device, ckpt)


# Spans all three regimes: closed rings below s = 1/8, spirals up to s = 1, and
# damage limitation above it. Shared by `ceiling` and `runtime`.
CEILING_COSTS = (0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6)


def collect_ceiling(out, episodes, device):
    """The planner alone across a dense cost grid -- the difficulty curve.

    The planner is the ceiling this task has: it optimises trees lost directly
    (CUT_COST = 1.0) and searches its two barrier families exhaustively. Running
    it across cost is what turns t_c + t_m from a pair of numbers into a
    measured difficulty axis, and unlike a training run it costs minutes.
    """
    fresh(out)
    for n in GRIDS:
        compare(out, n, CEILING_COSTS, episodes, device)


# Past s = 1: the r10 ladders at s = 1 and s = 1.2 (scripts/run_pipeline.sh 0.5 0.6).
BEYOND_COSTS = (0.5, 0.6)


def collect_beyond(out, onset_out, episodes, device):
    """PPO against the planner past the containment boundary, paired, plus
    how long each stage's evaluation sat at exactly zero before it learned.

    A stage is scored only once its log shows the full budget; one still
    training has a best-so-far checkpoint that is not its result. Re-run this
    target when the remaining stages finish.
    """
    fresh(out)
    onset = []
    for n in GRIDS:
        for cost in BEYOND_COSTS:
            log = os.path.join(LOGS, f"run_{n}x{n}_t{cost:g}_{cost:g}.log")
            ckpt = ckpt_path(n, cost)
            if not (os.path.exists(log) and os.path.exists(ckpt)):
                print(f"  {n}x{n} s={2 * cost:g}: no run, skipped", file=sys.stderr)
                continue
            with open(log, errors="replace", newline="") as fh:
                evals = [m.groups() for rec in re.split(r"[\r\n]", fh.read())
                         if (m := RECORD.search(rec))]
            if not evals or int(evals[-1][6]) < int(evals[-1][7]):
                print(f"  {n}x{n} s={2 * cost:g}: still training, skipped", file=sys.stderr)
                continue
            compare(out, n, [cost], episodes, device, ckpt)
            first = next((int(g[6]) for g in evals if float(g[0]) > 0), None)
            onset.append({"n": n, "tc": cost, "tm": cost, "s": round(2 * cost, 6),
                          "first_eval_saved": float(evals[0][0]) / 100.0,
                          "frames_at_zero": first if first is not None else int(evals[-1][6]),
                          "best_saved": max(float(g[0]) for g in evals) / 100.0,
                          "frames": int(evals[-1][6])})
    write(onset_out, onset)


# Below the sweep: 8x8 only, trained for the training-cost panel
# (scripts/run_low_cost_8x8.sh). Scored like the sweep, paired with the planner.
# 8x8 runs outside the ladders, scored here because the training-cost panel
# needs each one's paired reference: the two ring-regime costs below the sweep,
# and second seeds. (cost, STEERCA_TAG) per run; the empty tag is the first,
# unseeded run at a cost, "seed2" a run with STEERCA_SEED=2 and a 4M-frame budget.
EXTRA_RUNS = ((0.02, ""), (0.02, "seed2"), (0.04, ""), (0.04, "seed2"), (0.06, "seed2"),
              (0.12, "seed2"), (0.2, "seed2"), (0.3, "seed2"), (0.5, "seed2"),
              (0.6, "seed2"))


def _tagged(base_path, tag):
    """A checkpoint or log path with STEERCA_TAG's suffix spliced in before the end."""
    if not tag:
        return base_path
    if base_path.endswith("_policy.pth"):
        return base_path[:-len("_policy.pth")] + f"_{tag}_policy.pth"
    return base_path[:-len(".log")] + f"_{tag}.log"


def collect_extra(out, episodes, device):
    """PPO against the planner for the extra 8x8 runs (EXTRA_RUNS), paired.

    These runs exist only to place more points on the training-cost panel,
    which needs the planner's paired score as its reference and each trained
    policy's episode length. A run still training is skipped. Each row carries
    its checkpoint, which is how `onset` tells two runs at one cost apart.
    """
    fresh(out)
    for cost, tag in EXTRA_RUNS:
        log = _tagged(os.path.join(LOGS, f"run_8x8_t{cost:g}_{cost:g}.log"), tag)
        ckpt = _tagged(ckpt_path(8, cost), tag)
        label = f"8x8 s={2 * cost:g}{' ' + tag if tag else ''}"
        if not (os.path.exists(log) and os.path.exists(ckpt)):
            print(f"  {label}: no run, skipped", file=sys.stderr)
            continue
        with open(log, errors="replace", newline="") as fh:
            evals = [m.groups() for rec in re.split(r"[\r\n]", fh.read())
                     if (m := RECORD.search(rec))]
        if not evals or int(evals[-1][6]) < int(evals[-1][7]):
            print(f"  {label}: still training, skipped", file=sys.stderr)
            continue
        compare(out, 8, [cost], episodes, device, ckpt)


def collect_zeroshot(out, episodes, device):
    """The 8x8 sweep policies run unchanged at 16x16, paired with the planner.

    This is the transfer the warm-start ladder starts from: the 16x16 stage is
    initialised with exactly these weights. An untrained network of the same
    architecture is scored too, as the floor a from-scratch run starts at; its
    rows are told apart from the trained ones by the `ckpt` column.
    """
    import tempfile
    import torch
    sys.path.insert(0, ROOT)
    from steerca.evaluate import (Categorical, ForestFireEquivariantNet,
                                  ForestFirePolicyNet, ProbabilisticActor,
                                  TensorDictModule, parse_arch_tag)
    fresh(out)
    for cost in COSTS:
        compare(out, 16, [cost], episodes, device, ckpt_path(8, cost))
    with tempfile.TemporaryDirectory() as tmp:
        name = f"ppo_equiv_{ARCH[8]}_1hot_8x8_untrained_policy.pth"
        cfg = parse_arch_tag(name)
        torch.manual_seed(0)
        net = ForestFirePolicyNet(
            backbone_net=ForestFireEquivariantNet(grid_size=8, device="cpu", **cfg),
            n_actions=9, hidden=cfg["hidden"], device="cpu")
        actor = ProbabilisticActor(
            TensorDictModule(net, in_keys=["grid", "t", "agent_state"], out_keys=["logits"]),
            distribution_class=Categorical, in_keys=["logits"], out_keys=["action"],
            return_log_prob=False)
        ckpt = os.path.join(tmp, name)
        torch.save(actor.state_dict(), ckpt)
        for cost in COSTS:
            compare(out, 16, [cost], episodes, device, ckpt)


# 8x8 A2C and SAC from scratch at their best settings, one run per action cost,
# for the difficulty figure: A2C at entropy 0.2, SAC tuned (reward scaled by
# area, alpha 0.05, target entropy 0.2), everything else PPO's. The 2M-frame
# runs are the robustness check's (A2C, s = 0.12) and the tuned sweep's (SAC,
# s = 0.12 and 0.4); scripts/run_{a2c,sac}_onset.sh add the rest at 4M, except
# SAC at s = 1 and 1.2 (2M). SAC at s = 0.04 was stopped at 2.25M frames, still at
# 0%, so its log never completes and it is skipped.
#   algo: [(cost, log, checkpoint), ...]
# A2C's entropy coefficient in these runs, to read the policy's entropy back
# out of the logged entropy loss (-coefficient x entropy). SAC logs none.
OTHER_ENT = {"a2c": 0.2}
_A2C = "a2c_equiv_c96_b6_512_64_1hot_8x8_t{c:g}_{c:g}_cut0.5_ent0.2_r10_{tag}_policy.pth"
_SAC = "sac_equiv_c96_b6_512_64_1hot_8x8_t{c:g}_{c:g}_cut0.5_rs64_a0.05-te0.2_r10_{tag}_policy.pth"
OTHER_RUNS = {
    "a2c": [(c, f"baseline_a2c_8x8_t{c:g}_onset4m_ent0.2.log", _A2C.format(c=c, tag="onset4m"))
            for c in (0.02, 0.04, 0.12, 0.2, 0.3, 0.5, 0.6)]
           + [(0.06, "baseline_a2c_8x8_t0.06_sweep2m_ent0.2.log", _A2C.format(c=0.06, tag="sweep2m"))],
    "sac": [(c, f"baseline_sac_8x8_t{c:g}_onset4m_scaled.log", _SAC.format(c=c, tag="onset4m"))
            for c in (0.02, 0.04, 0.12, 0.3)]
           + [(c, f"baseline_sac_8x8_t{c:g}_onset2m_scaled.log", _SAC.format(c=c, tag="onset2m"))
              for c in (0.5, 0.6)]
           + [(c, f"baseline_sac_8x8_t{c:g}_sweep2m_tuned.log", _SAC.format(c=c, tag="sweep2m"))
              for c in (0.06, 0.2)],
}


def collect_planted(out, graph_seeds=40, pocket_seeds=6):
    """Per-instance scores behind the planted figure, one row per (instance,
    player), with the figure's own rollout so the seeds it picks reproduce.

    `frac` is the share of the certificate's trees the player keeps. The figure
    shows the first pocket instance PPO solves and, from the graph instances,
    the median instance of each mode of the curriculum policy's score
    (figures.planted_modes). CPU, one episode at a time, as the figure draws it.
    """
    sys.path.insert(0, HERE)
    import figures as F
    M = MODELS
    n, c = F.PLANTED_N, F.PLANTED_COST
    plan = ([("pocket", s, "1", F.POCKET_CKPT) for s in range(pocket_seeds)]
            + [("graph", s, F.GRAPH_CFG, F.GRAPH_CKPT) for s in range(graph_seeds)])
    rows = []
    for set_, seed, cfg, ckpt in plan:
        for agent in ("certificate", "ppo"):
            grids, pos, gap, cert = F._rollout(
                n, c, c, None if agent == "certificate" else os.path.join(M, ckpt),
                seed=seed, planted=cfg, oracle=agent == "certificate")
            _, cut = F._history(grids, len(grids) - 1)
            saved = F._trees(grids[-1])
            rows.append({"set": set_, "seed": seed, "agent": agent, "n": n,
                         "tc": c, "tm": c, "saved": saved, "certificate": cert,
                         "frac": saved / cert, "cuts": int(cut.sum()),
                         "certificate_cuts": len(gap), "steps": len(grids) - 1})
            print(f"  {set_} seed {seed} {agent}: {100 * saved / cert:.1f}% of "
                  f"certificate, {int(cut.sum())} cuts", flush=True)
    write(out, rows)


def collect_others(out, episodes, device):
    """A2C and SAC against the planner on identical episodes, 8x8, by cost.

    Scored exactly as the PPO cells are (steerca.evaluate, paired, from each
    run's best-evaluation checkpoint), so the difficulty figure can put them on
    the same axes. steerca.evaluate labels any checkpoint "ppo"; the label is
    rewritten to the learner here. A run still training is skipped.

    Each learner row also carries what the run's log says about it: its frame
    budget, its best and last greedy evaluation during training, and (A2C only)
    the final policy's entropy. The paired score is of the best checkpoint, so
    the last evaluation is what says whether the run kept what it found.
    """
    fresh(out)
    tmp = out + ".part"
    rows = []
    for algo, runs in OTHER_RUNS.items():
        for cost, log, ckpt in sorted(runs):
            label = f"{algo} 8x8 s={2 * cost:g}"
            log, ckpt = os.path.join(LOGS, log), os.path.join(MODELS, ckpt)
            if not (os.path.exists(log) and os.path.exists(ckpt)):
                print(f"  {label}: no run, skipped", file=sys.stderr)
                continue
            with open(log, errors="replace", newline="") as fh:
                text = fh.read()
            evals = [m.groups() for rec in re.split(r"[\r\n]", text)
                     if (m := RECORD.search(rec))]
            if not evals or int(evals[-1][6]) < int(evals[-1][7]):
                print(f"  {label}: still training, skipped", file=sys.stderr)
                continue
            summary = dict(SUMMARY.findall(text))
            saved = [float(g[0]) / 100.0 for g in evals]
            run = {"frames": int(evals[-1][6]), "train_best_saved": max(saved),
                   "train_final_saved": saved[-1],
                   "final_entropy": (max(0.0, -float(summary["loss_entropy"]) / OTHER_ENT[algo])
                                     if algo in OTHER_ENT and "loss_entropy" in summary
                                     else "")}
            fresh(tmp)
            compare(tmp, 8, [cost], episodes, device, ckpt)
            with open(tmp, newline="") as fh:
                for r in csv.DictReader(fh):
                    if r["agent"] == "ppo":
                        r["agent"] = algo
                        r.update(run)
                    else:
                        # Every row needs the same columns; the planner has no log.
                        r.update(dict.fromkeys(run, ""))
                    rows.append(r)
    fresh(tmp)
    write(out, rows)


# 8x8 PPO trained from scratch, one run per action cost, all r10. The r10 ladder's
# own 8x8 run at t = 0.06 resumed an earlier checkpoint, so the baseline sweep's
# from-scratch PPO run at the same settings stands in for it.
ONSET_RUNS = ((0.02, "run_8x8_t0.02_0.02.log", ""),
              (0.02, "run_8x8_t0.02_0.02_seed2.log", "seed2"),
              (0.04, "run_8x8_t0.04_0.04.log", ""),
              (0.04, "run_8x8_t0.04_0.04_seed2.log", "seed2"),
              (0.06, "baseline_ppo_8x8_t0.06_sweep2m.log", "sweep2m"),
              (0.06, "run_8x8_t0.06_0.06_seed2.log", "seed2"),
              (0.12, "run_8x8_t0.12_0.12.log", ""),
              (0.12, "run_8x8_t0.12_0.12_seed2.log", "seed2"),
              (0.2, "run_8x8_t0.2_0.2.log", ""),
              (0.2, "run_8x8_t0.2_0.2_seed2.log", "seed2"),
              (0.3, "run_8x8_t0.3_0.3.log", ""),
              (0.3, "run_8x8_t0.3_0.3_seed2.log", "seed2"),
              (0.5, "run_8x8_t0.5_0.5.log", ""),
              (0.5, "run_8x8_t0.5_0.5_seed2.log", "seed2"),
              (0.6, "run_8x8_t0.6_0.6.log", ""),
              (0.6, "run_8x8_t0.6_0.6_seed2.log", "seed2"))
ONSET_FRACTION = 0.95


def collect_onset(out):
    """How long 8x8 PPO trained from scratch takes to come within 5% of the planner.

    The reference is the planner's score on the paired evaluation of the same
    cost (sweep.csv / beyond.csv, 100 episodes). The training evaluation runs on
    its own 100 episodes, so the comparison is between two means over different
    instance sets, not a paired one. Episodes are estimated as frames divided by
    the trained policy's mean episode length on that paired evaluation; a run
    that never reaches the threshold is reported as censored at its budget.
    """
    planner, ppo = {}, defaultdict(list)
    for name in ("sweep.csv", "beyond.csv", "extra_8x8.csv"):
        path = os.path.join(DATA, name)
        if not os.path.exists(path):
            continue
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                if int(r["n"]) != 8:
                    continue
                if r["agent"] == "planner":
                    planner[float(r["tc"])] = r
                else:
                    ppo[float(r["tc"])].append(r)

    def own_row(cost, tag):
        """This run's paired PPO row, matched on its checkpoint name; the cost's
        first PPO row if the run has none of its own (the s = 0.12 stand-in)."""
        want = f"_r10_{tag}_policy.pth" if tag else "_r10_policy.pth"
        rows_ = ppo.get(cost, [])
        return next((r for r in rows_ if r.get("ckpt", "").endswith(want)),
                    rows_[0] if rows_ else None)

    rows = []
    for cost, log, tag in ONSET_RUNS:
        path = os.path.join(LOGS, log)
        mine = own_row(cost, tag)
        if not os.path.exists(path) or cost not in planner or mine is None:
            print(f"  s={2 * cost:g} {tag}: missing log or paired reference, skipped",
                  file=sys.stderr)
            continue
        with open(path, errors="replace", newline="") as fh:
            evals = [m.groups() for rec in re.split(r"[\r\n]", fh.read())
                     if (m := RECORD.search(rec))]
        if not evals or int(evals[-1][6]) < int(evals[-1][7]):
            # A run still training has no budget yet to be censored at.
            print(f"  s={2 * cost:g} {tag}: still training, skipped", file=sys.stderr)
            continue
        ref = float(planner[cost]["saved"])
        steps = float(mine["steps"])
        target = ONSET_FRACTION * ref
        hit = next((int(g[6]) for g in evals if float(g[0]) / 100.0 >= target), None)
        budget = int(evals[-1][6])
        frames = hit if hit is not None else budget
        rows.append({"n": 8, "tc": cost, "tm": cost, "s": round(2 * cost, 6),
                     "run": tag, "log": log, "planner_saved": ref, "target": target,
                     "best_saved": max(float(g[0]) for g in evals) / 100.0,
                     "reached": int(hit is not None), "frames": frames,
                     "budget_frames": budget, "mean_episode_steps": steps,
                     "episodes": frames / steps})
        print(f"  s={2 * cost:g} {tag or '-'}: target {100 * target:.1f}%, "
              + (f"reached at {hit} frames" if hit is not None else f"not reached in {budget}")
              + f" (budget {budget})")
    write(out, rows)


def collect_runtime(out, episodes):
    """What the planner spends on each decision, across the same cost grid.

    The planner re-plans once per CA update, so per-call time is the cost of one
    search over its barrier families at the current state. Wall clock, single
    process, CPU: absolute numbers depend on the machine and on what else is
    running, so the shape across s is the result, not the milliseconds.
    """
    import platform
    import time
    sys.path.insert(0, ROOT)
    import torch
    import steerca.firebreak.planner as H
    import steerca.firebreak.env as ff
    from steerca.evaluate import make_env, run

    stats = {"calls": 0, "secs": 0.0}
    plan = H.plan_barrier

    def timed(*a, **k):
        t0 = time.perf_counter()
        try:
            return plan(*a, **k)
        finally:
            stats["secs"] += time.perf_counter() - t0
            stats["calls"] += 1

    H.plan_barrier = timed  # BarrierPlannerPolicy looks it up as a module global
    rows = []
    try:
        for n in GRIDS:
            # One untimed episode first: imports and caches land in the first
            # call and would otherwise inflate the cheapest cell.
            env = make_env(n, 0.1, 0.1, 1, "cpu")
            run(env, env.reset(), H.BarrierPlannerPolicy(), ff.episode_cap(n, 0.1, 0.1), n)
            for cost in CEILING_COSTS:
                # One episode at a time. `run` keeps calling the policy on every
                # episode of a batch until the LAST one ends, so a batched
                # timing would pad the count with calls on finished episodes.
                stats.update(calls=0, secs=0.0)
                for ep in range(episodes):
                    env = make_env(n, cost, cost, 1, "cpu")
                    torch.manual_seed(ep)
                    env.set_seed(ep)
                    run(env, env.reset(), H.BarrierPlannerPolicy(),
                        ff.episode_cap(n, cost, cost) + 100, n)
                rows.append({"n": n, "tc": cost, "tm": cost, "s": round(2 * cost, 6),
                             "episodes": episodes,
                             "calls_per_episode": stats["calls"] / episodes,
                             "ms_per_call": 1000 * stats["secs"] / max(stats["calls"], 1),
                             "secs_per_episode": stats["secs"] / episodes,
                             "machine": platform.processor() or platform.machine()})
                print(f"  {n}x{n} s={2 * cost:g}: {rows[-1]['ms_per_call']:.2f} ms/call, "
                      f"{rows[-1]['calls_per_episode']:.1f} calls/ep")
    finally:
        H.plan_barrier = plan
    write(out, rows)


# Batch sizes timed per device. One environment is the conventional Gym loop;
# the training script runs 8; the larger batches are what evaluation and
# vectorized collection can use; both devices go to the same largest batch.
THROUGHPUT_BATCHES = {"cuda": (1, 8, 64, 512, 4096), "cpu": (1, 8, 64, 512, 4096)}


def collect_throughput(out, seconds=3.0):
    """Environment transitions per second, by device, grid size and batch size.

    Times the full TorchRL ``env.step`` the training loop calls -- CA advance,
    action, reward and TorchRL's own bookkeeping -- with uniformly random
    actions at ``t_c = t_m = 0.2``. The CA's cost does not depend on the policy,
    and episodes are not reset, since a reset is rare next to a step. Each
    configuration is warmed up first (``torch.compile`` specializes per shape),
    then stepped for ``seconds`` of wall clock.
    """
    import platform
    import time
    sys.path.insert(0, ROOT)
    import torch
    from steerca.evaluate import make_env

    devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    rows = []
    for device in devices:
        name = (torch.cuda.get_device_name(0) if device == "cuda"
                else platform.processor() or platform.machine())
        sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
        for n in GRIDS:
            for batch in THROUGHPUT_BATCHES[device]:
                torch.manual_seed(0)
                env = make_env(n, 0.2, 0.2, batch, device)
                td = env.reset()

                def advance(td, k):
                    for _ in range(k):
                        td["action"] = torch.randint(0, 9, (batch,), device=device)
                        td = env.step(td)["next"]
                    return td

                td = advance(td, 50)  # warm-up: compilation and caches
                sync()
                steps, t0 = 0, time.perf_counter()
                while time.perf_counter() - t0 < seconds:
                    td = advance(td, 20)
                    steps += 20
                sync()
                secs = time.perf_counter() - t0
                rows.append({"device": device, "n": n, "batch": batch,
                             "steps": steps * batch, "secs": secs,
                             "transitions_per_s": steps * batch / secs,
                             "hardware": name, "torch_threads": torch.get_num_threads()})
                print(f"  {device} {n}x{n} batch {batch}: "
                      f"{rows[-1]['transitions_per_s']:,.0f} transitions/s")
    write(out, rows)


def fresh(path):
    """steerca.evaluate --csv appends, so clear the file before a full sweep."""
    if os.path.exists(path):
        os.remove(path)


def write(path, rows):
    if not rows:
        print(f"  nothing to write to {path}", file=sys.stderr)
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {os.path.relpath(path, ROOT)}")


# --------------------------------------------------------------------------
# Other learners: the A2C / SAC sweep and its robustness check
# --------------------------------------------------------------------------

# One dict per run. `set` groups them for the tables:
#   sweep      every learner at PPO's settings, four cells
#   robust     A2C / SAC at learning rates and epochs of their own, easiest cell
#   tuned      the settings the sanity check found each learner needs to learn
#              at all -- A2C at entropy 0.05 (0.2 on the easiest cell as well),
#              SAC with the reward in units of trees and a small temperature
#   sanity     a task where a felled tree PAYS on the step it is cut, so there
#              is no credit to assign: separates a broken learner from a hard task
# `hp` is the experiment-name infix the training script adds for non-default
# learner settings, `ent` the entropy coefficient, needed to read the final
# policy entropy back out of the logged entropy loss. See
# scripts/run_baseline_{sweep,robustness,tuned}.sh and run_sanity_check.sh.
SAC_TUNED = "_rs{area}_a0.05-te0.2"
BASELINE_RUNS = []


def _run(set_, algo, variant, n, cost, log, tag, lr=1e-4, epochs=None, hp="",
         ent=5e-3, cut="cut0.5"):
    if epochs is None:
        epochs = {"ppo": 10, "a2c": 1, "sac": ""}[algo]
    BASELINE_RUNS.append(dict(set=set_, algo=algo, variant=variant, n=n, cost=cost,
                              lr=lr, epochs=epochs, log=log, tag=tag,
                              hp=hp.format(area=n * n), ent=ent, cut=cut))


for _n in (8, 16):
    for _c in (0.06, 0.2):
        for _a in ("ppo", "a2c", "sac"):
            _run("sweep", _a, "default", _n, _c,
                 f"baseline_{_a}_{_n}x{_n}_t{_c:g}_sweep2m.log", "sweep2m")
        _run("tuned", "a2c", "ent0.05", _n, _c,
             f"baseline_a2c_{_n}x{_n}_t{_c:g}_sweep2m_tuned.log", "sweep2m",
             hp="_ent0.05", ent=0.05)
        _run("tuned", "sac", "scaled", _n, _c,
             f"baseline_sac_{_n}x{_n}_t{_c:g}_sweep2m_tuned.log", "sweep2m",
             hp=SAC_TUNED)
_run("tuned", "a2c", "ent0.2", 8, 0.06, "baseline_a2c_8x8_t0.06_sweep2m_ent0.2.log",
     "sweep2m", hp="_ent0.2", ent=0.2)
for _tag, _a, _lr, _ep in (("a2c-e10", "a2c", 1e-4, 10),
                           ("a2c-lr7e4", "a2c", 7e-4, 1),
                           ("a2c-lr1e3", "a2c", 1e-3, 1),
                           ("sac-lr3e4", "sac", 3e-4, ""),
                           ("sac-lr1e3", "sac", 1e-3, "")):
    _run("robust", _a, _tag, 8, 0.06, f"robust_{_tag}.log", f"robust-{_tag}",
         lr=_lr, epochs=_ep)
for _v, _a, _hp, _ent in (("default", "ppo", "", 5e-3), ("default", "a2c", "", 5e-3),
                          ("default", "sac", "", 5e-3),
                          ("ent0.05", "a2c", "_ent0.05", 0.05),
                          ("scaled", "sac", SAC_TUNED, 5e-3)):
    _tag = _a if _v == "default" else f"{_a}-tuned"
    _run("sanity", _a, _v, 8, 0.06, f"sanity_{_tag}.log", f"sanity-{_tag}",
         hp=_hp, ent=_ent, cut="cut-1")

# The last value W&B prints in its run summary, e.g. `wandb:  loss/loss_entropy -0.00033`,
# or the `summary: loss/...` lines steerca.train prints at the end of every run.
SUMMARY = re.compile(r"(?:wandb|summary):\s+loss/(loss_entropy|loss_actor)\s+(-?[\d.e+-]+)\s*$",
                     re.M)


def _probe(policy, n, cost, device, episodes=24, seed=1):
    """P(cut) and entropy on fresh states, then a stochastic rollout.

    Only meaningful on a checkpoint that learned something: a run that never
    beats 0.0% keeps the checkpoint written at its FIRST evaluation, one batch
    into training, so probing it measures that network rather than the final
    policy. collect_baselines probes only runs with a non-zero best evaluation.
    """
    import torch
    sys.path.insert(0, ROOT)
    import steerca.firebreak.env as ff
    from steerca.firebreak.ca import TREE
    from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
    ff.N, ff.TC, ff.TM = n, cost, cost
    ff.PLANTED_CFG = None
    env = ff.ForestFireEnv(device=device, batch_size=[episodes])
    torch.manual_seed(seed); env.set_seed(seed)
    td = env.reset()
    k = episodes
    with torch.no_grad():
        if hasattr(policy, "forward"):
            p = torch.softmax(policy(td.clone())["logits"], -1)
            p_cut = float(p[..., 8].mean())
            entropy = float(-(p * p.clamp_min(1e-12).log()).sum(-1).mean())
        else:
            p_cut, entropy = 1.0 / 9.0, float(torch.log(torch.tensor(9.0)))
    alive = torch.ones(k, dtype=torch.bool)
    cuts = torch.zeros(k)
    trees = td["grid"][:, TREE].sum((-2, -1)).float().cpu()
    with set_exploration_type(ExplorationType.RANDOM), torch.no_grad():
        for _ in range(ff.episode_cap(n, cost, cost)):
            td = policy(td)
            cuts += ((td["action"].reshape(k).cpu() == 8) & alive).float()
            out = env.step(td)
            nxt = out["next"]
            trees = torch.where(alive, nxt["grid"][:, TREE].sum((-2, -1)).float().cpu(), trees)
            alive &= ~nxt["done"].reshape(k).cpu()
            if not alive.any():
                break
            td = step_mdp(out)
    return p_cut, entropy, float(cuts.mean()), float((trees / (n * n)).mean())


def collect_baselines(out, device):
    import torch
    sys.path.insert(0, ROOT)
    from steerca.evaluate import make_ppo_policy
    dev = torch.device(device)
    rows = []
    for r in BASELINE_RUNS:
        path = os.path.join(LOGS, r["log"])
        if not os.path.exists(path):
            print(f"  missing {r['log']}, skipped", file=sys.stderr)
            continue
        with open(path, errors="replace", newline="") as fh:
            text = fh.read()
        evals = [m.groups() for rec in re.split(r"[\r\n]", text) if (m := RECORD.search(rec))]
        saved = [float(g[0]) / 100.0 for g in evals]
        returns = [float(g[1]) for g in evals]
        summary = dict(SUMMARY.findall(text))
        # The final policy's entropy, read back from the last batch's entropy
        # loss (-coefficient x entropy). SAC logs no entropy; its actor loss is
        # kept instead, because at the default temperature it diverges.
        final_entropy = (max(0.0, -float(summary["loss_entropy"]) / r["ent"])
                         if "loss_entropy" in summary else "")
        n, cost = r["n"], r["cost"]
        row = {"set": r["set"], "algo": r["algo"], "variant": r["variant"], "n": n,
               "tc": cost, "tm": cost, "lr": r["lr"], "epochs": r["epochs"],
               "frames": int(evals[-1][6]) if evals else 0,
               "best_saved": max(saved) if saved else "",
               "final_saved": saved[-1] if saved else "",
               "first_return": returns[0] if returns else "",
               "final_return": returns[-1] if returns else "",
               "final_entropy": final_entropy,
               "final_actor_loss": summary.get("loss_actor", ""),
               "ckpt_p_cut": "", "ckpt_entropy": "", "ckpt_cuts": "", "ckpt_saved": ""}
        if r["set"] != "sanity" and saved and max(saved) > 0:
            # The checkpoint is the best greedy evaluation, a learned policy.
            ckpt = os.path.join(MODELS, f"{r['algo']}_equiv_{ARCH[n]}_1hot_{n}x{n}_t{cost:g}_{cost:g}"
                                f"_{r['cut']}{r['hp']}_r10_{r['tag']}_policy.pth")
            p_cut, h, cuts, st_saved = _probe(make_ppo_policy(ckpt, "equiv", dev, n),
                                              n, cost, dev, episodes=100)
            row.update(ckpt_p_cut=p_cut, ckpt_entropy=h, ckpt_cuts=cuts, ckpt_saved=st_saved)
        rows.append(row)
        h = f"{final_entropy:.3f}" if final_entropy != "" else "---"
        print(f"  {r['set']:6s} {r['algo']} {r['variant']:10s} {n}x{n} t{cost:g}: best "
              f"{100 * row['best_saved']:.1f}%  final H {h}")

    # References on the easiest cell: an untrained network and uniform actions.
    # Without them 0.0% reads as a score rather than as the floor it is.
    from steerca.archs import (ForestFireEquivariantNet,
                                               ForestFirePolicyNet, arch_for_grid)
    from tensordict.nn import TensorDictModule
    from torch.distributions import Categorical
    from torchrl.modules import ProbabilisticActor

    class Uniform:
        def __call__(self, td):
            td.set("action", torch.randint(0, 9, td["grid"].shape[:-3], device=td.device))
            return td

    torch.manual_seed(0)
    cfg = arch_for_grid(8)
    net = ForestFirePolicyNet(ForestFireEquivariantNet(grid_size=8, device=dev, **cfg),
                              9, cfg["hidden"], dev)
    untrained = ProbabilisticActor(
        TensorDictModule(net, in_keys=["grid", "t", "agent_state"], out_keys=["logits"]),
        distribution_class=Categorical, in_keys=["logits"], out_keys=["action"])
    for name, pol in (("untrained", untrained), ("uniform", Uniform())):
        p_cut, h, cuts, st_saved = _probe(pol, 8, 0.06, dev, episodes=100)
        rows.append({"set": "reference", "algo": name, "variant": "", "n": 8,
                     "tc": 0.06, "tm": 0.06, "lr": "", "epochs": "", "frames": 0,
                     "best_saved": "", "final_saved": "", "first_return": "",
                     "final_return": "", "final_entropy": h, "final_actor_loss": "",
                     "ckpt_p_cut": p_cut, "ckpt_entropy": h, "ckpt_cuts": cuts,
                     "ckpt_saved": st_saved})
    write(out, rows)


# --------------------------------------------------------------------------
# A2C warm-started from a solved PPO policy, and the gradient check behind it
# --------------------------------------------------------------------------

WARMSTART_RUNS = (("ppo", "PPO", "warmchk-ppo.log"),
                  ("a2c", "A2C, 1 epoch", "warmchk-a2c.log"),
                  ("a2c-e10", "A2C, 10 epochs", "warmchk-a2c-e10.log"))
WARMSTART_SRC = "ppo_equiv_c96_b6_512_64_1hot_8x8_t0.06_0.06_cut0.5_r10_policy.pth"


def collect_warmstart(out, grad_out, device):
    """Evaluation curves of three learners warm-started from one solved policy,
    plus the check that PPO's and A2C's policy gradients agree at ratio 1.

    Both halves are needed for the claim. The curves show A2C destroying a
    policy PPO holds; the gradient check shows that is not a wiring error --
    at the first update the clipped objective and A2C's are the same function,
    so identical gradients mean the divergence comes from what the trust region
    does afterwards.
    """
    rows = []
    for key, label, log in WARMSTART_RUNS:
        path = os.path.join(LOGS, log)
        if not os.path.exists(path):
            print(f"  missing {log}, skipped", file=sys.stderr)
            continue
        with open(path, errors="replace", newline="") as fh:
            at = {}
            for rec in re.split(r"[\r\n]", fh.read()):
                m = RECORD.search(rec)
                if not m:
                    continue
                frames = int(m.group(7))
                # One eval every 25 batches of 2000 frames. tqdm prints twice at
                # each: once on update, still carrying the PREVIOUS eval, then
                # again once the new eval is in the description. The last
                # record at each eval frame is therefore the eval itself.
                if (frames - 2000) % 50000 == 0:
                    at[frames] = m
            for frames, m in sorted(at.items()):
                rows.append({"run": key, "label": label, "frames": frames,
                             "eval_saved": float(m.group(1)) / 100.0,
                             "eval_return": float(m.group(2))})
    write(out, rows)

    import torch
    os.environ.update(STEERCA_N="8", STEERCA_TC="0.06", STEERCA_TM="0.06",
                      STEERCA_FRAMES="2000", STEERCA_ALGO="ppo")
    sys.path.insert(0, ROOT)
    import steerca.train as m
    from torchrl.objectives import A2CLoss, ClipPPOLoss
    from torchrl.objectives.value import GAE
    src = os.path.join(MODELS, WARMSTART_SRC)
    m.policy_module.load_state_dict(torch.load(src, map_location=m.device))
    m.value_module.load_state_dict(
        torch.load(src.replace("policy", "value"), map_location=m.device))
    torch.manual_seed(0)
    with torch.no_grad():
        td = m.base_env.rollout(48, m.policy_module, break_when_any_done=False)
        GAE(gamma=m.gamma, lmbda=m.lmbda, value_network=m.value_module,
            average_gae=True)(td)
    batch = td.reshape(-1)
    params = list(m.policy_module.parameters())

    def grad(loss):
        for p_ in params:
            p_.grad = None
        loss(batch.clone())["loss_objective"].backward()
        return torch.cat([(p_.grad if p_.grad is not None else torch.zeros_like(p_))
                          .flatten() for p_ in params])

    g_ppo = grad(ClipPPOLoss(m.policy_module, m.value_module, clip_epsilon=0.2,
                             entropy_bonus=False, critic_coeff=0.0))
    g_a2c = grad(A2CLoss(m.policy_module, m.value_module,
                         entropy_bonus=False, critic_coeff=0.0))
    write(grad_out, [{
        "batch": batch.numel(),
        "norm_ppo": float(g_ppo.norm()), "norm_a2c": float(g_a2c.norm()),
        "cosine": float(torch.nn.functional.cosine_similarity(g_ppo, g_a2c, dim=0)),
    }])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all",
                    choices=["all", "curves", "sweep", "ceiling", "baselines", "warmstart",
                                                          "runtime", "beyond", "extra", "onset", "others",
                             "throughput", "planted", "zeroshot"])
    ap.add_argument("--episodes", type=int, default=100,
                    help="paired episodes per cell for `sweep`")
    # The ceiling curve is drawn under the paired agents in Figure 2, so it is
    # scored on the same 100 starts: same count, same seed, same device.
    ap.add_argument("--ceiling-episodes", type=int, default=100)
    ap.add_argument("--runtime-episodes", type=int, default=32,
                    help="episodes timed one at a time for `runtime`")
    ap.add_argument("--device", default=None,
                    help="default: cuda if available, except cpu for `baselines`")
    args = ap.parse_args()

    import torch
    gpu = "cuda" if torch.cuda.is_available() else "cpu"

    if args.what in ("all", "curves"):
        print("== training curves ==")
        collect_curves(os.path.join(DATA, "training_curves.csv"))
    if args.what in ("all", "sweep"):
        print("== PPO vs planner, 12 cells ==")
        collect_sweep(os.path.join(DATA, "sweep.csv"),
                      args.episodes, args.device or gpu)
    if args.what in ("all", "baselines"):
        print("== other learners: A2C / SAC sweep and robustness ==")
        collect_baselines(os.path.join(DATA, "baselines.csv"), args.device or "cpu")
    if args.what in ("all", "warmstart"):
        print("== A2C warm-started from a solved policy ==")
        collect_warmstart(os.path.join(DATA, "warmstart.csv"),
                          os.path.join(DATA, "grad_check.csv"), args.device or gpu)
    if args.what in ("all", "ceiling"):
        print("== planner ceiling vs action cost ==")
        collect_ceiling(os.path.join(DATA, "planner_ceiling.csv"),
                        args.ceiling_episodes, args.device or gpu)
    if args.what in ("all", "beyond"):
        print("== PPO vs planner past s = 1 ==")
        collect_beyond(os.path.join(DATA, "beyond.csv"),
                       os.path.join(DATA, "beyond_onset.csv"),
                       args.episodes, args.device or gpu)
    if args.what in ("all", "extra"):
        print("== extra 8x8 runs vs planner ==")
        collect_extra(os.path.join(DATA, "extra_8x8.csv"), args.episodes, args.device or gpu)
    if args.what in ("all", "onset"):
        print("== 8x8 PPO: time to come within 5% of the planner ==")
        collect_onset(os.path.join(DATA, "onset_8x8.csv"))
    if args.what in ("all", "others"):
        print("== 8x8 A2C / SAC at their best settings vs planner ==")
        collect_others(os.path.join(DATA, "others_8x8.csv"), args.episodes, args.device or gpu)
    if args.what in ("all", "throughput"):
        print("== environment throughput, GPU vs CPU ==")
        collect_throughput(os.path.join(DATA, "env_throughput.csv"))
    if args.what in ("all", "runtime"):
        print("== planner time per decision vs action cost ==")
        collect_runtime(os.path.join(DATA, "planner_runtime.csv"), args.runtime_episodes)
    if args.what in ("all", "zeroshot"):
        print("== 8x8 policies run unchanged at 16x16 vs planner ==")
        collect_zeroshot(os.path.join(DATA, "zeroshot_16x16.csv"), args.episodes,
                         args.device or gpu)
    if args.what in ("all", "planted"):
        print("== planted instances, one at a time, for the planted figure ==")
        collect_planted(os.path.join(DATA, "planted_seeds.csv"))


if __name__ == "__main__":
    main()
