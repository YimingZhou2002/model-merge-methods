#!/usr/bin/env python3
"""Prepare the pinned evaluation source without modifying another checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "runtime_manifest.json").read_text())


def verify(source):
    failures = []
    for entry in MANIFEST["files"]:
        path = source / entry["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            failures.append(entry["path"])
    if failures:
        raise RuntimeError("Runtime source differs from the evaluated version: " + ", ".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="New checkout on your data disk")
    parser.add_argument("--repo", default=MANIFEST["upstream_repo"], help="Git remote or local clone")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    dest = args.destination.expanduser().resolve()
    if args.verify_only:
        verify(dest)
    elif dest.exists() and any(dest.iterdir()):
        verify(dest)  # Idempotent only for an already correct runtime.
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", "--no-recurse-submodules",
                        args.repo, str(dest)], check=True)
        commit = MANIFEST["upstream_commit"]
        subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", commit], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "--detach", commit], check=True)
        # Normalize legacy credential fields before applying the sanitized diff.
        # Both fields are removed by the patch; the resulting source is unchanged.
        entry = dest / "agentevolver/main_ppo.py"
        text = entry.read_text()
        text = re.sub(r'("WANDB_API_KEY":\s*")[^"\n]*(")', r'\1EMPTY\2', text)
        text = re.sub(r'("WANDB_BASE_URL":\s*")[^"\n]*(")', r'\1http://localhost\2', text)
        entry.write_text(text)
        patch = str(HERE / "runtime.patch")
        subprocess.run(["git", "-C", str(dest), "apply", "--check", patch], check=True)
        subprocess.run(["git", "-C", str(dest), "apply", patch], check=True)
        verify(dest)
    print("Verified evaluation source:", dest)
    print("AppWorld data and Python environments must be supplied separately; see README.md.")


if __name__ == "__main__":
    main()
