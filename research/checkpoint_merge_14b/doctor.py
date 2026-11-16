"""Validate the actual GPU/runtime/data prerequisites before spending rollouts."""
import json
import os
from pathlib import Path
import subprocess

from research.model_merge.weighted_merge import _sha256_file


def capture(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    if result.returncode:
        raise RuntimeError(f"Environment check failed ({result.returncode}): {result.stderr[-6000:]}")
    return result


def inspect_environment(config, repo):
    tools = Path(repo)/"research/appworld_checkpoint_eval"
    subprocess.run([config["eval_python"], str(tools/"prepare_source.py"), config["source"], "--verify-only"], check=True)
    code = """
import importlib.metadata as m, json, torch, vllm, transformers, ray, verl
names=['torch','vllm','transformers','ray','verl','triton','flash-attn']
versions={name:m.version(name) for name in names}
print('INVENTORY_JSON='+json.dumps({'versions':versions,'cuda':torch.version.cuda,
 'cuda_available':torch.cuda.is_available(),'gpu_count':torch.cuda.device_count(),
 'torch_file':torch.__file__,'vllm_file':vllm.__file__}))
"""
    env = dict(os.environ, PYTHONPATH=config["source"], HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    result = capture([config["eval_python"], "-c", code], env=env)
    line = next(line for line in result.stdout.splitlines() if line.startswith("INVENTORY_JSON="))
    inventory = json.loads(line.split("=", 1)[1])
    expected = {"torch": "2.13.0", "vllm": "0.27.1", "transformers": "5.12.1", "ray": "2.57.0", "verl": "0.9.0"}
    for name, version in expected.items():
        if inventory["versions"][name].split("+")[0] != version:
            raise ValueError(f"Different inference dependency: {name}={inventory['versions'][name]}, expected {version}")
    if config["dtype"] == "float32" and config["attention_backend"] == "FLASH_ATTN":
        raise ValueError("The installed FlashAttention backend does not support FP32; use TRITON_ATTN")
    if not inventory["cuda_available"] or inventory["gpu_count"] < len(config["devices"].split(",")):
        raise ValueError("Insufficient visible CUDA devices")
    app_code = """
import json, os, appworld, importlib.metadata as m
from pathlib import Path
from appworld import load_task_ids
ids=load_task_ids('dev')
assert len(ids)==57 and len(set(ids))==57
instructions={tid:json.loads((Path(os.environ['APPWORLD_ROOT'])/'data/tasks'/tid/'specs.json').read_text())['instruction'] for tid in ids}
print('INVENTORY_JSON='+json.dumps({'appworld_version':m.version('appworld'),'task_ids':sorted(ids),'task_instructions':instructions,'module':appworld.__file__}))
"""
    app_env = dict(os.environ, PYTHONPATH=str(Path(config["appworld_repo"])/"src"),
                   APPWORLD_ROOT=str(Path(config["source"])/"env_service/environments/appworld"))
    app = capture([config["appworld_python"], "-c", app_code], env=app_env)
    inventory["appworld"] = json.loads(next(line.split("=", 1)[1] for line in app.stdout.splitlines() if line.startswith("INVENTORY_JSON=")))
    workspace = Path(repo).parent
    manifest_file = workspace/"runtime/appworld-data-manifest.json"
    manifest = json.loads(manifest_file.read_text())
    for rel, expected_sha in manifest["files"].items():
        path = workspace/"runtime/appworld_data/data"/rel
        if not path.is_file() or _sha256_file(path) != expected_sha:
            raise ValueError(f"AppWorld data snapshot changed: {path}")
    inventory["appworld_data_manifest_sha256"] = _sha256_file(manifest_file)
    inventory["runtime_manifest_sha256"] = _sha256_file(tools/"runtime_manifest.json")
    inventory["gpu_inventory"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv"], text=True)
    return inventory


def require_idle_gpus():
    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    if rows.strip():
        raise RuntimeError("GPU compute processes are present; wait for free GPUs, never terminate other workloads")
