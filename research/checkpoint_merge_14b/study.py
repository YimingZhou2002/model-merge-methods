#!/usr/bin/env python3
"""Run eight checkpoint baselines, freeze recipes, merge, then compare dev@8."""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from research.checkpoint_merge_14b.core import read_rollouts, select_bank, paired_comparison
from research.model_merge.weighted_merge import _atomic_write_json as write_json, _sha256_file

HERE = Path(__file__).resolve().parent
PATH_KEYS = ("pool", "base_model", "run_root", "source", "eval_python", "appworld_python", "appworld_repo")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_config(path, workspace):
    config = json.loads(Path(path).read_text())
    for key in PATH_KEYS:
        p = Path(config[key])
        config[key] = str(p if p.is_absolute() else workspace/p)
    if config["samples"] != 8 or config["tasks"] != 57 or config["dtype"] != "float32":
        raise ValueError("This study requires 57 dev tasks x8 and FP32")
    steps = config["checkpoints"]
    if steps != sorted(set(steps)) or len(steps) != 8 or config["anchor"] != steps[-1]:
        raise ValueError("Need eight chronological sources and the last source as anchor")
    return config


def state(config, phase, **extra):
    root = Path(config["run_root"])
    root.mkdir(parents=True, exist_ok=True)
    write_json({"phase": phase, "updated_at": now(), **extra}, root/"status.json")
    print(now(), phase, json.dumps(extra), flush=True)


def freeze_inputs(config):
    from research.checkpoint_merge_14b.merge import source, source_identity
    root = Path(config["run_root"])
    files = [HERE/"config.json", HERE/"core.py", HERE/"merge.py", HERE/"study.py", HERE/"doctor.py", HERE/"prepare_assets.py",
             REPO/"research/appworld_checkpoint_eval/rollout_appworld.sh",
             REPO/"research/appworld_checkpoint_eval/runtime_manifest.json"]
    frozen = {"config": config, "code_sha256": {str(p): _sha256_file(p) for p in files},
              "models": {str(s): source_identity(source(Path(config['pool'])/f'checkpoint-{s}'))
                         for s in config["checkpoints"]}}
    path = root/"inputs.json"
    if path.exists():
        if json.loads(path.read_text()) != frozen:
            raise ValueError("Frozen inputs/config/code changed; use a new study run_root")
    else:
        write_json(frozen, path)
    return frozen


def command(config, model, output, index, dry=False):
    workspace = REPO.parent
    ray = workspace/"r"/f"{config.get('ray_namespace', 0)+index:03d}"
    if len(os.fsencode(ray)) > 40:
        raise ValueError("Ray path exceeds 40 bytes")
    args = ["bash", str(REPO/"research/appworld_checkpoint_eval/rollout_appworld.sh"),
            "--source", config["source"], "--python", config["eval_python"],
            "--model-path", str(model), "--output-dir", str(output),
            "--samples", "8", "--temperature", str(config["temperature"]),
            "--devices", config["devices"], "--tp", str(config["tp"]),
            "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
            "--env-url", config["env_url"], "--ray-temp-dir", str(ray),
            "--attention-backend", config["attention_backend"],
            "--max-num-batched-tokens", str(config["max_num_batched_tokens"]),
            "--dtype", config["dtype"], "--seed", str(config["seed"]+index*32)]
    return args+["--dry-run"] if dry else args


def validate_config(config, model, output, index):
    import yaml
    resolved = yaml.safe_load((output/"resolved_config.yaml").read_text())
    rollout = resolved["actor_rollout_ref"]["rollout"]
    if rollout["dtype"] != config["dtype"] or rollout["seed"] != config["seed"]+index*32:
        raise ValueError("Resolved dtype/seed differs from frozen protocol")
    if rollout["engine_kwargs"]["vllm"]["attention_backend"] != config["attention_backend"]:
        raise ValueError("Resolved attention backend differs from frozen protocol")
    if rollout["max_num_batched_tokens"] != config["max_num_batched_tokens"]:
        raise ValueError("Resolved scheduler token budget differs from frozen protocol")
    if resolved["actor_rollout_ref"]["model"]["path"] != str(model):
        raise ValueError("Resolved model path differs")
    return resolved


def wait_for(process, config, phase, **extra):
    while process.poll() is None:
        state(config, phase, pid=process.pid, **extra)
        time.sleep(30)
    if process.returncode:
        raise RuntimeError(f"Child process failed, code={process.returncode}; inspect {extra}")


def evaluate(config, name, model, index, *, dry=False):
    root = Path(config["run_root"])
    output = root/("preflight" if dry else "evaluations")/name
    task_instructions = json.loads((root/"environment.json").read_text())["appworld"]["task_instructions"]
    receipt = output/"VERIFIED.json"
    if not dry and receipt.exists():
        validate_config(config, model, output, index)
        result = read_rollouts(output/"validation_log/0.jsonl", task_instructions=task_instructions)
        if result != json.loads(receipt.read_text()):
            raise ValueError("Previously verified evaluation changed")
        return result
    if output.exists():
        raise FileExistsError(f"Existing incomplete run needs inspection: {output}")
    if not dry:
        from research.checkpoint_merge_14b.doctor import require_idle_gpus
        require_idle_gpus()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (root/f"launcher-{name}{'-dry' if dry else ''}.log").open("x") as log:
        process = subprocess.Popen(command(config, model, output, index, dry), stdout=log, stderr=subprocess.STDOUT)
        wait_for(process, config, "preflight" if dry else "evaluating", job=name, output=str(output))
    validate_config(config, model, output, index)
    if dry:
        return None
    result = read_rollouts(output/"validation_log/0.jsonl", task_instructions=task_instructions)
    write_json(result, receipt)
    return result


def freeze_recipes(config, baselines):
    keys = baselines[str(config["checkpoints"][0])]["task_keys"]
    if any(r["task_keys"] != keys for r in baselines.values()):
        raise ValueError("Checkpoint baselines have different task prompts")
    mmr = config["mmr"]
    selection = select_bank({s: r["vector"] for s, r in baselines.items()}, k=mmr["k"], lam=mmr["lambda"], tau=mmr["tau"])
    frozen = {"selection": selection, "baseline_log_sha256": {s: r["log_sha256"] for s, r in baselines.items()},
              "anchor": config["anchor"], "trajectory_steps": config["checkpoints"],
              "primary_merges": ["mmr_ties", "tsg_tv"], "controls": ["linear", "tsg_tv_norm"],
              "selection_note": "8B checkpoint-290 exclusion is not transferred to the 14B pool. TSG-TV was chosen before seeing 14B results."}
    path = Path(config["run_root"])/"recipes.json"
    if path.exists() and json.loads(path.read_text()) != frozen:
        raise ValueError("Frozen recipes changed")
    write_json(frozen, path)
    return selection


def freeze_base(config):
    from research.checkpoint_merge_14b.merge import source, source_identity
    identity = source_identity(source(config["base_model"]))
    path = Path(config["run_root"])/"base-input.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Pretrained base changed after recipe freezing")
    write_json(identity, path)


def report(config, results):
    anchor = f"ck{config['anchor']}"
    baselines = [f"ck{s}" for s in config["checkpoints"]]
    best = max(baselines, key=lambda name: results[name]["mean_at_8"])
    pairs = [(name, reference) for name in config["merge_arms"] for reference in [anchor, best]]
    pairs += [("mmr_ties", "linear"), ("tsg_tv", "linear"), ("tsg_tv", "tsg_tv_norm"), ("tsg_tv", "mmr_ties")]
    comparisons = {f"{a} - {b}": paired_comparison(results[a], results[b], iterations=config["bootstrap_samples"])
                   for a, b in pairs if a != b}
    payload = {"updated_at": now(), "results": results, "best_observed_source": best, "comparisons": comparisons,
               "caveat": "One session per model; source selection and merge assessment reuse dev. Best-source selection and unadjusted comparisons are exploratory, not independent validation."}
    root = Path(config["run_root"])
    write_json(payload, root/"results.json")
    lines = ["# Qwen3-14B AppWorld dev@8", "", f"T={config['temperature']}, FP32, 57 tasks x8, eight GPUs.", "",
             "| Model | Mean@8 | Pass@8 |", "|---|---:|---:|"]
    for name, result in results.items():
        lines.append(f"| {name} | {result['mean_at_8']:.6f} | {result['pass_at_8']:.6f} |")
    lines += ["", "| Comparison | Delta (pp) | Primary bootstrap 95% interval (pp) | Method |", "|---|---:|---|---|"]
    for name, c in comparisons.items():
        lo, hi = c.get("family_bootstrap_95_pp", c["task_bootstrap_95_pp"])
        method = "task-family clusters" if "family_bootstrap_95_pp" in c else "tasks (family mapping unavailable)"
        lines.append(f"| {name} | {c['delta_pp']:.3f} | [{lo:.3f}, {hi:.3f}] | {method} |")
    lines += ["", payload["caveat"], "Intervals resample task families when identified from frozen AppWorld task instructions, otherwise tasks; never rollout rows or independent sessions.", ""]
    (root/"RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["preflight", "run", "report"])
    parser.add_argument("--config", type=Path, default=HERE/"config.json")
    parser.add_argument("--workspace", type=Path, default=REPO.parent)
    args = parser.parse_args()
    config = load_config(args.config, args.workspace.resolve())
    root = Path(config["run_root"])
    root.mkdir(parents=True, exist_ok=True)
    with (root/"study.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            from research.checkpoint_merge_14b.doctor import inspect_environment
            environment = inspect_environment(config, REPO)
            environment_path = root/"environment.json"
            if environment_path.exists() and json.loads(environment_path.read_text()) != environment:
                raise ValueError("The validated environment differs from the frozen study environment")
            write_json(environment, environment_path)
            freeze_inputs(config)
            if args.action == "preflight":
                evaluate(config, "ck300", Path(config["pool"])/"checkpoint-300", 7, dry=True)
                state(config, "preflight_passed")
                return
            if args.action == "report":
                old = json.loads((root/"results.json").read_text())
                report(config, old["results"])
                return
            baselines, results = {}, {}
            for index, step in enumerate(config["checkpoints"]):
                model = Path(config["pool"])/f"checkpoint-{step}"
                value = evaluate(config, f"ck{step}", model, index)
                baselines[str(step)] = value
                results[f"ck{step}"] = value
                write_json(baselines, root/"baselines.json")
            selection = freeze_recipes(config, baselines)
            freeze_base(config)
            state(config, "building_merges", selection=selection["weights"])
            from research.checkpoint_merge_14b.merge import build_primary, build_norm_control
            models = root/"models"
            if not all((models/n/"EXPORT_READY.json").exists() for n in ["mmr_ties", "linear", "tsg_tv"]):
                build_primary(config, selection, models)
            if not (models/"tsg_tv_norm"/"EXPORT_READY.json").exists():
                build_norm_control(config, models)
            for index, name in enumerate(config["merge_arms"], len(baselines)):
                results[name] = evaluate(config, name, models/name, index)
            report(config, results)
            state(config, "complete", results=str(root/"RESULTS.md"))
        except Exception as exc:
            state(config, "failed", error=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
