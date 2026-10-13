"""Assemble the SteerCA-artifacts archive from a training tree (maintainers only).

Collects every checkpoint and training log that ``paper/collect.py`` and
``paper/figures.py`` read, plus the value head beside each policy so a published
policy can also warm start a run, and writes ``MANIFEST.sha256``:

    python tools/build_artifacts.py --src /path/to/training/tree --dest ../SteerCA-artifacts

``--src`` holds ``models/`` and ``logs/`` as training wrote them. Checkpoints
are hard-linked when possible (they are large and never edited). Logs are
copied with personal details scrubbed: the W&B login line, W&B run URLs and
absolute local paths. The evaluation records and run summaries that
``collect.py`` parses are left untouched.
"""
import argparse
import hashlib
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SCRUB = [
    (re.compile(r"(wandb: Currently logged in as: )[^\n\r]*"), r"\1<redacted>"),
    (re.compile(r"https://wandb\.ai/\S+"), "<wandb-url>"),
    (re.compile(r"/home/[^/\s]+/MyCode/CARL/"), "./"),
    (re.compile(r"/home/[^/\s]+/(?:anaconda3|miniconda3)/envs/[^/\s]+/"), "<python>/"),
    (re.compile(r"/home/[^/\s]+/"), "~/"),
]


def wanted(src):
    """(models, logs) basenames the paper pipeline reads, given a source tree."""
    os.environ["STEERCA_ARTIFACTS"] = src
    sys.path.insert(0, os.path.join(ROOT, "paper"))
    import collect as C
    import figures as F

    models, logs = set(), set()
    for n in C.GRIDS:
        for c in C.COSTS + C.BEYOND_COSTS:
            models.add(os.path.basename(C.ckpt_path(n, c)))
            logs.add(f"run_{n}x{n}_t{c:g}_{c:g}.log")
    for runs in C.CURVE_LOGS.values():
        logs.update(name for name, _ in runs)
    for cost, tag in C.EXTRA_RUNS:
        models.add(os.path.basename(C._tagged(C.ckpt_path(8, cost), tag)))
        logs.add(os.path.basename(C._tagged(f"run_8x8_t{cost:g}_{cost:g}.log", tag)))
    for runs in C.OTHER_RUNS.values():
        for _, log, ckpt in runs:
            logs.add(log)
            models.add(ckpt)
    for _, log, _ in C.ONSET_RUNS:
        logs.add(log)
    for r in C.BASELINE_RUNS:
        n, cost = r["n"], r["cost"]
        logs.add(r["log"])
        models.add(f"{r['algo']}_equiv_{C.ARCH[n]}_1hot_{n}x{n}_t{cost:g}_{cost:g}"
                   f"_{r['cut']}{r['hp']}_r10_{r['tag']}_policy.pth")
    for _, _, log in C.WARMSTART_RUNS:
        logs.add(log)
    models.add(C.WARMSTART_SRC)
    models.update({F.POCKET_CKPT, F.GRAPH_CKPT})
    # Each policy's value head, for warm starts.
    models |= {m.replace("_policy.pth", "_value.pth") for m in models}
    return sorted(models), sorted(logs)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--dest", required=True)
    args = ap.parse_args()
    src, dest = os.path.abspath(args.src), os.path.abspath(args.dest)
    models, logs = wanted(src)

    missing = []
    for sub, names in (("models", models), ("logs", logs)):
        os.makedirs(os.path.join(dest, sub), exist_ok=True)
        for name in names:
            a, b = os.path.join(src, sub, name), os.path.join(dest, sub, name)
            if not os.path.exists(a):
                missing.append(f"{sub}/{name}")
                continue
            if os.path.exists(b):
                os.remove(b)
            if sub == "models":
                try:
                    os.link(a, b)
                except OSError:
                    shutil.copy2(a, b)
            else:
                with open(a, encoding="utf-8", errors="surrogateescape", newline="") as fh:
                    text = fh.read()
                for pattern, repl in SCRUB:
                    text = pattern.sub(repl, text)
                with open(b, "w", encoding="utf-8", errors="surrogateescape", newline="") as fh:
                    fh.write(text)

    entries = []
    for sub in ("models", "logs"):
        for name in sorted(os.listdir(os.path.join(dest, sub))):
            entries.append(f"{sha256(os.path.join(dest, sub, name))}  {sub}/{name}")
    with open(os.path.join(dest, "MANIFEST.sha256"), "w") as fh:
        fh.write("\n".join(entries) + "\n")

    size = sum(os.path.getsize(os.path.join(dest, e.split("  ", 1)[1])) for e in entries)
    print(f"{len(entries)} files, {size / 1e6:.0f} MB -> {dest}")
    if missing:
        # Value heads of SAC runs and runs that never finished are expected here.
        print(f"{len(missing)} referenced files not in the source tree:")
        for m in missing:
            print(f"  {m}")


if __name__ == "__main__":
    main()
