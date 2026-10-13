"""Fetch the published checkpoints and training logs into ./artifacts.

    python tools/fetch_artifacts.py                  # download, unpack, verify
    python tools/fetch_artifacts.py --from DIR       # link a copy you already have
    python tools/fetch_artifacts.py --verify         # re-check ./artifacts

The archive holds ``models/`` (every checkpoint the paper pipeline reads, each
policy with its value head) and ``logs/`` (the training logs it parses), with a
``MANIFEST.sha256`` that every file is checked against. ``paper/collect.py``,
``paper/figures.py``, ``steerca.render`` and ``steerca.transfer`` look for it at
``./artifacts`` unless ``STEERCA_ARTIFACTS`` says otherwise.
"""
import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEST = os.path.join(ROOT, "artifacts")

# The archive's download URL (a .tar, .tar.gz or .zip of the SteerCA-artifacts
# folder). TODO: set once the archive is deposited; STEERCA_ARTIFACTS_URL
# overrides it.
ARTIFACTS_URL = None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(root):
    manifest = os.path.join(root, "MANIFEST.sha256")
    if not os.path.exists(manifest):
        raise SystemExit(f"no MANIFEST.sha256 in {root}")
    bad = 0
    with open(manifest) as fh:
        entries = [line.rstrip("\n").split("  ", 1) for line in fh if line.strip()]
    for digest, rel in entries:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            print(f"  missing  {rel}")
            bad += 1
        elif sha256(path) != digest:
            print(f"  corrupt  {rel}")
            bad += 1
    if bad:
        raise SystemExit(f"{bad} of {len(entries)} files failed verification")
    print(f"verified {len(entries)} files in {root}")


def link_from(src):
    src = os.path.abspath(src)
    if not os.path.exists(os.path.join(src, "MANIFEST.sha256")):
        raise SystemExit(f"{src} has no MANIFEST.sha256; is it the artifacts folder?")
    if os.path.lexists(DEST):
        raise SystemExit(f"{DEST} already exists; remove it first")
    os.symlink(src, DEST, target_is_directory=True)
    print(f"linked {DEST} -> {src}")


def download(url):
    if os.path.lexists(DEST):
        raise SystemExit(f"{DEST} already exists; remove it first")
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
        archive = os.path.join(tmp, os.path.basename(url.split("?")[0]) or "artifacts")
        print(f"downloading {url}")
        with urllib.request.urlopen(url) as resp, open(archive, "wb") as out:
            shutil.copyfileobj(resp, out)
        unpacked = os.path.join(tmp, "unpacked")
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as z:
                z.extractall(unpacked)
        else:
            with tarfile.open(archive) as t:
                t.extractall(unpacked, filter="data")
        # Accept the folder at the top of the archive or one level down.
        top = unpacked
        entries = os.listdir(unpacked)
        if "MANIFEST.sha256" not in entries and len(entries) == 1:
            top = os.path.join(unpacked, entries[0])
        shutil.move(top, DEST)
    print(f"unpacked to {DEST}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="src", help="link an existing artifacts folder instead")
    ap.add_argument("--url", default=os.environ.get("STEERCA_ARTIFACTS_URL", ARTIFACTS_URL))
    ap.add_argument("--verify", action="store_true", help="only verify ./artifacts")
    args = ap.parse_args()

    if not args.verify:
        if args.src:
            link_from(args.src)
        elif args.url:
            download(args.url)
        else:
            sys.exit("No archive URL is configured yet. Pass --url <archive URL>, set "
                     "STEERCA_ARTIFACTS_URL, or use --from <folder> for a local copy.")
    verify(DEST)


if __name__ == "__main__":
    main()
