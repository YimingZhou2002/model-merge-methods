#!/usr/bin/env python3
"""Check pinned pool sizes or download the pretrained base needed by MMR-TIES."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check-pool", "download-base"])
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    root = args.workspace.resolve()
    if args.action == "check-pool":
        manifest = json.loads((root/"model-pool-manifest.json").read_text())
        if len(manifest["checkpoints"]) != 8 or not manifest.get("completed_at"):
            raise ValueError("The pinned eight-checkpoint pool has not completed")
        for entry in manifest["files"]:
            path = root/"model-pool"/entry["path"]
            if not path.is_file() or path.stat().st_size != entry["size"]:
                raise ValueError(f"Missing or truncated pool file: {path}")
        print(f"Verified {len(manifest['files'])} files, {manifest['total_bytes']} bytes, revision={manifest['revision']}")
        return
    os.environ.setdefault("HF_HOME", str(root/".hf-download-cache"))
    os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    from huggingface_hub import HfApi, snapshot_download
    repo = "Qwen/Qwen3-14B"
    receipt = root/"base-model"/"download.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    if receipt.exists():
        revision = json.loads(receipt.read_text())["revision"]
    else:
        revision = HfApi().model_info(repo).sha
        receipt.write_text(json.dumps({"repo_id": repo, "revision": revision}, indent=2)+"\n")
    print(f"Downloading raw base {repo}@{revision}", flush=True)
    snapshot_download(repo, revision=revision, local_dir=root/"base-model/Qwen3-14B",
                      allow_patterns=["*.safetensors", "*.json", "*.jinja", "LICENSE", "README.md"], max_workers=4)
    print("Base download finished", flush=True)


if __name__ == "__main__":
    main()
