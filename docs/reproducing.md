# Reproducing the papers

Every number, figure and generated table in the papers comes out of two scripts:

* **`paper/collect.py`** does the measurements: it parses training logs, re-scores
  checkpoints against the planner on paired episodes, and times things. It writes
  tidy CSVs to `paper/data/`.
* **`paper/figures.py`** reads those CSVs, and nothing else, and draws every
  figure (`paper/figures/`) and table body (`paper/tables/`).

The CSVs are in the repository, so you can enter at any of three depths.

| tier | what you rerun | needs | time |
|---|---|---|---|
| **1. Redraw** | `paper/figures.py` from the shipped CSVs | CPU | about a minute |
| **2. Re-measure** | `paper/collect.py` from the published checkpoints and logs | the artifacts, a GPU for most targets | minutes to an hour per target |
| **3. Retrain** | the training runs themselves, then tiers 2 and 1 | a GPU, and time | about 6 h per 8×8 or 16×16 stage, 22 h per 32×32 stage (RTX 3090) |

## Tier 1: redraw from the shipped CSVs

```bash
python paper/figures.py                 # everything
python paper/figures.py difficulty_sn   # one target
```

Three figures (the episode strip, the regimes panel and the planted episodes)
roll a policy out to draw it. They need the checkpoints (tier 2's
`fetch_artifacts.py`) and are skipped with a message without them. The two
diagrams that carry no data (the time model and the network) are TikZ drawings
and are not generated here.

## Tier 2: re-measure from the published checkpoints

```bash
python tools/fetch_artifacts.py         # checkpoints + logs -> ./artifacts, sha256-verified
python paper/collect.py sweep           # one target
python paper/collect.py all             # everything
python paper/figures.py
```

`collect.py` reads `./artifacts/models` and `./artifacts/logs`, or the folder in
`STEERCA_ARTIFACTS`. Its targets are independent:

| target | writes | what it does | cost |
|---|---|---|---|
| `curves` | `training_curves.csv` | parses the 12 ladder logs | seconds, no GPU |
| `sweep` | `sweep.csv` | the 12 r10 cells (8/16/32 × s = 0.12, 0.24, 0.4, 0.6), PPO vs planner, 100 paired episodes | GPU, ~10 min |
| `beyond` | `beyond.csv`, `beyond_onset.csv` | the same at s = 1 and 1.2 (tc = tm = 0.5, 0.6) | GPU, minutes |
| `extra` | `extra_8x8.csv` | the extra 8×8 runs: low costs and second seeds | GPU, minutes |
| `onset` | `onset_8x8.csv` | frames for 8×8 PPO to reach 95% of the planner (logs + the CSVs above) | seconds |
| `others` | `others_8x8.csv` | A2C and SAC at every 8×8 cost, paired with the planner | GPU, minutes |
| `baselines` | `baselines.csv` | the A2C/SAC sweep, tuning, robustness and sanity runs (logs + checkpoint probes) | CPU, minutes |
| `warmstart` | `warmstart.csv`, `grad_check.csv` | the clip ablation, plus PPO/A2C gradient agreement at ratio 1 | GPU, a minute |
| `zeroshot` | `zeroshot_16x16.csv` | the 8×8 policies run unchanged at 16×16, plus an untrained floor | GPU, minutes |
| `planted` | `planted_seeds.csv` | certificate vs PPO on 40 graph + 6 pocket instances | CPU, minutes |
| `ceiling` | `planner_ceiling.csv` | the planner alone across 13 costs × 3 grid sizes (the difficulty curve) | GPU, the longest target |
| `runtime` | `planner_runtime.csv` | planner time per decision | CPU, under an hour |
| `throughput` | `env_throughput.csv` | environment steps per second, GPU vs CPU | minutes |

**What reproduces exactly.** The paired evaluations (`sweep`, `beyond`, `extra`,
`others`, `zeroshot`) are deterministic on a given device: rerunning them on CUDA
reproduces the shipped CSVs. Two caveats:

* The reset's random draws differ between CPU and CUDA, so a CPU run scores
  different episodes. That is also why `ceiling` defaults to the GPU: its 100
  episodes are the paired runs' episodes, so its curve passes through the
  planner's paired scores.
* PPO's greedy actions depend on float details, so a policy can shift by a
  point or so between devices.

`runtime` and `throughput` are timings and track your hardware.

## Tier 3: retrain

Training runs write to `./models` and `./logs`. To measure your own runs with
`collect.py`, point it at the repository root: `STEERCA_ARTIFACTS=. python
paper/collect.py sweep`.

| command | produces |
|---|---|
| `scripts/run_pipeline.sh 0.06 0.12 0.2 0.3 0.5 0.6` | one 8 → 16 → 32 warm-start ladder per action cost (the sweep and beyond-s=1 cells) |
| `scripts/run_low_cost_8x8.sh`, `scripts/run_second_seeds.sh` | the extra 8×8 runs: s = 0.04, 0.08 and a second seed at every cost |
| `scripts/run_baseline_sweep.sh`, `run_baseline_tuned.sh`, `run_baseline_robustness.sh`, `run_sanity_check.sh` | the A2C and SAC baselines, their tuning, and the verification task |
| `scripts/run_a2c_onset.sh`, `scripts/run_sac_onset.sh` | A2C and SAC at every 8×8 cost |
| `STEERCA_N=8 STEERCA_TC=0.06 STEERCA_TM=0.06 STEERCA_FRAMES=600000 STEERCA_INIT_FROM=none STEERCA_TAG=pricing STEERCA_CUT_TREE_AHEAD=a STEERCA_CUT_TREE_IMMINENT=b python -m steerca.train` | the cut-pricing runs, with (a, b) = (0, 0), (0, 1), (0.5, 0.5), and (1, 1) at `STEERCA_FRAMES=4000000` |
| `STEERCA_N=8 STEERCA_TC=0.06 STEERCA_TM=0.06 STEERCA_FRAMES=200000 STEERCA_INIT_FROM=<the 8×8 tc=tm=0.06 PPO policy> STEERCA_ALGO=x STEERCA_TAG=warmchk-x python -m steerca.train` | the warm-start (clip) ablation, x = `ppo`, `a2c`, and `a2c` again with `STEERCA_EPOCHS=10` and tag `warmchk-a2c-e10` |
| `STEERCA_PLANTED=1 STEERCA_SIZES="16 32" scripts/run_pipeline.sh 0.2` | the planted pocket ladder |
| `STEERCA_PLANTED=family=graph STEERCA_N=32 STEERCA_TC=0.2 STEERCA_TM=0.2 STEERCA_FRAMES=2000000 python -m steerca.train` | the flat graph-tier run |
| `scripts/run_curriculum.sh` | the two-stage graph curriculum |

The scripts write each run's log to `logs/` under the name `collect.py` expects.

**Training is not bit-reproducible on a GPU.** Even with `STEERCA_SEED` fixed,
two runs of the same command diverge after the first gradient steps, because
some CUDA kernels in the backward pass are nondeterministic. The initial
networks and the first batch of experience are identical across runs, but the
weights are not after training. Expect retrained numbers to match the published
ones statistically, not to the digit. The papers' own second seeds
(`extra_8x8.csv`) show how much one cell varies between runs.

## Where each figure and table comes from

Both versions of the paper (Scientific Reports and the Springer/NCA manuscript)
draw from the same targets. `_sn` targets are the Springer-width figures that
both versions include.

| content | `figures.py` target | output | data |
|---|---|---|---|
| one episode, as a filmstrip | `episode_sn` | `carl_episode_strip_sn.pdf` | rollout of the 16×16 tc=tm=0.2 policy |
| score vs the cost of a firebreak cell | `difficulty_sn` | `carl_difficulty_sn.pdf` | `planner_ceiling`, `sweep`, `beyond`, `extra_8x8`, `others_8x8` |
| barrier structure and training cost | `structure_sn` | `carl_structure_sn.pdf` | `onset_8x8`, `planner_runtime` |
| planted episodes | `planted_sn` | `carl_planted_episodes_sn.pdf` | `planted_seeds` + rollouts |
| one grid at three costs | `regimes_sn` | `carl_regimes_sn.pdf` | rollouts of the 16×16 policies |
| PPO vs planner, per cell | `ppo_vs_planner_sn` | `carl_ppo_vs_planner_sn.pdf` | `sweep` |
| paired margins | `margins_sn` | `carl_margins_sn.pdf` | `sweep`, `beyond`, `extra_8x8` |
| evaluation through training | `training_curves_sn` | `carl_training_curves_sn.pdf` | `training_curves` |
| the score curve (table) | `difficulty` (table) | `tables/tab_difficulty.tex` | `planner_ceiling` |
| planner search time | `runtime` | `tables/tab_runtime.tex` | `planner_runtime` |
| the twelve sweep cells | `sweep` | `tables/tab_sweep.tex` | `sweep` |
| PPO past s = 1 | `beyond` | `tables/tab_beyond.tex` | `beyond` |
| A2C/SAC baselines | `baselines` | `tables/tab_baselines.tex` | `baselines` |
| verification task | `sanity` | `tables/tab_sanity.tex` | `baselines` |
| settings tried on the easiest cell | `robust` | `tables/tab_robust.tex` | `baselines` |
| the clip ablation | `warmstart` | `tables/tab_warmstart.tex` | `warmstart`, `grad_check` |
| environment throughput | `throughput` | `tables/tab_throughput.tex` | `env_throughput` |

`python paper/figures.py tables` builds every table.

## Commands as printed in the papers

The papers' reproducibility tables use the code's original names. All of them
work here with these substitutions. The `CARL_*` variables are accepted as they
are, so only the paths change.

| in the papers | here |
|---|---|
| `python src_torchrl/vanilla_ppo_forest_fire.py` | `python -m steerca.train` |
| `python src_torchrl/compare_heuristic.py` | `python -m steerca.evaluate` |
| `python src_torchrl/render_episodes.py` | `python -m steerca.render` |
| `python src_torchrl/validate_planted.py` | `python -m steerca.validate_planted` |
| `python paper/data/collect.py` | `python paper/collect.py` |
| `CARL_N`, `CARL_TC`, … | `STEERCA_N`, `STEERCA_TC`, … (either works) |
| `models/`, `logs/` (published files) | `artifacts/models/`, `artifacts/logs/` |
