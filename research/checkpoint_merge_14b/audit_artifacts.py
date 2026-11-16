"""Independently audit completed cells of the frozen 14B dev@8 study.

Uses original JSONL, resolved YAML, frozen inputs and AppWorld instance logs.
Does not import the study's statistics, merge code or receipt readers. Source
weight contents and exported tensors require separate integrity checks.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import yaml


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def service_instances(path):
    pattern = re.compile(
        r"Creating instance with env_type: appworld, task_id: ([^,\s]+), "
        r"instance_id: ([0-9a-f]{32})"
    )
    instances = {}
    with path.open() as stream:
        for line in stream:
            match = pattern.search(line)
            if match:
                task, instance = match.groups()
                require(instances.get(instance, task) == task,
                        f"Instance {instance} was associated with different tasks")
                instances[instance] = task
    return list(instances.values())


def audit_cell(root, name, index, model_path, config, environment, instances):
    cell = root / "evaluations" / name
    receipt = read_json(cell / "VERIFIED.json")
    raw = cell / "validation_log" / "0.jsonl"
    groups = defaultdict(list)
    prompts = {}
    score_counts = Counter()
    with raw.open() as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            score, prompt = row["score"], row["input"]
            require(type(score) in (int, float) and math.isfinite(score)
                    and score in (0, 1), f"{name}:{number}: nonbinary score")
            require(isinstance(prompt, str), f"{name}:{number}: invalid input")
            key = hashlib.sha256(prompt.encode()).hexdigest()
            groups[key].append(float(score))
            prompts[key] = prompt
            score_counts[str(float(score))] += 1
    samples, tasks = config["samples"], config["tasks"]
    count = samples * tasks
    require(len(groups) == tasks and all(len(v) == samples for v in groups.values()),
            f"{name}: expected {tasks} tasks with {samples} samples each")
    keys = sorted(groups)
    vector = [sum(groups[key]) / samples for key in keys]
    mean = sum(sum(v) for v in groups.values()) / count
    passed = sum(any(v) for v in groups.values()) / tasks
    families = []
    for key in keys:
        matches = {task.split("_")[0]
                   for task, instruction in environment["appworld"]["task_instructions"].items()
                   if instruction in prompts[key]}
        require(len(matches) == 1, f"{name}: ambiguous task family for {key}")
        families.append(matches.pop())
    expected_receipt = {
        "log_sha256": digest(raw), "mean_at_8": mean, "pass_at_8": passed,
        "rollouts": count, "task_families": families,
        "task_grouping": "initial_input_sha256", "task_keys": keys, "vector": vector,
    }
    require(receipt == expected_receipt, f"{name}: receipt does not match raw data")
    reference = read_json(root / "evaluations" / "ck230" / "VERIFIED.json")
    require(keys == reference["task_keys"] and families == reference["task_families"],
            f"{name}: task alignment differs from ck230")
    resolved = yaml.safe_load((cell / "resolved_config.yaml").read_text())
    rollout = resolved["actor_rollout_ref"]["rollout"]
    expected_rollout = {
        "dtype": config["dtype"], "temperature": config["temperature"],
        "seed": config["seed"] + index * 32, "n": samples,
        "max_num_batched_tokens": config["max_num_batched_tokens"],
        "tensor_model_parallel_size": config["tp"],
        "gpu_memory_utilization": config["gpu_memory_utilization"],
        "max_model_len": 25580, "prompt_length": 20480, "response_length": 4096,
        "top_p": 1.0, "top_k": -1, "do_sample": True,
        "use_qwen3": False, "context_template": "linear_tokens",
    }
    for key, value in expected_rollout.items():
        require(rollout[key] == value, f"{name}: rollout {key} mismatch")
    require(rollout["engine_kwargs"]["vllm"]["attention_backend"]
            == config["attention_backend"], f"{name}: attention backend mismatch")
    require(rollout["multi_turn"]["max_steps"] == 30, f"{name}: action limit mismatch")
    require(all(rollout["val_kwargs"][key] == expected_rollout[key]
                for key in ("temperature", "n", "top_p", "top_k", "do_sample")),
            f"{name}: validation sampling mismatch")
    require(resolved["actor_rollout_ref"]["model"]["path"] == str(model_path),
            f"{name}: actual model path mismatch")
    require(resolved["trainer"]["val_only"] is True, f"{name}: not validation only")
    require(all(resolved["data"][key] == value for key, value in {
        "seed": 2027, "max_prompt_length": 4000, "max_response_length": 21580,
    }.items()), f"{name}: trajectory data configuration mismatch")
    start, end = index * count, (index + 1) * count
    coverage = Counter(instances[start:end])
    require(len(instances) >= end and coverage == {
        task: samples for task in environment["appworld"]["task_ids"]
    }, f"{name}: service task coverage mismatch")
    return {
        "raw_log_sha256": expected_receipt["log_sha256"],
        "resolved_config_sha256": digest(cell / "resolved_config.yaml"),
        "rows": count, "tasks": tasks, "samples_per_task": samples,
        "score_counts": dict(score_counts), "mean_at_8": mean, "pass_at_8": passed,
        "task_families": len(set(families)), "same_task_keys_as_ck230": True,
        "same_task_families_as_ck230": True, "receipt_matches_raw_data": True,
        "actual_model_path": str(model_path), "actual_rollout": expected_rollout,
        "actual_attention_backend": config["attention_backend"],
        "validation_sampling_matches": True, "max_actions": 30,
        "data_seed": 2027, "data_max_prompt_length": 4000,
        "data_max_response_length": 21580, "validation_only": True,
        "service_coverage": {
            "unique_tasks": len(coverage), "instances": sum(coverage.values()),
            "instances_per_task": samples, "all_frozen_dev_ids_present": True,
            "unique_instance_ordinal_range": [start + 1, end],
        },
    }


def audit(root, service_log, include_merges=False, require_complete=False):
    frozen = read_json(root / "inputs.json")
    config = frozen["config"]
    require(config["samples"] == 8 and config["tasks"] == 57,
            "This auditor targets the frozen 57-task dev@8 protocol")
    for name, expected in frozen["code_sha256"].items():
        require(digest(Path(name)) == expected, f"Frozen code changed: {name}")
    environment = read_json(root / "environment.json")
    require(len(set(environment["appworld"]["task_ids"])) == config["tasks"],
            "Frozen task IDs do not match task count")
    instances = service_instances(service_log)
    cells = [(f"ck{step}", Path(config["pool"]) / f"checkpoint-{step}")
             for step in config["checkpoints"]]
    if include_merges:
        cells.extend((name, root / "models" / name) for name in config["merge_arms"])
    complete, pending = {}, []
    baselines = read_json(root / "baselines.json")
    for index, (name, model_path) in enumerate(cells):
        path = root / "evaluations" / name / "VERIFIED.json"
        if not path.exists():
            pending.append(name)
            continue
        require(not pending, f"{name}: completed cells are not a sequential prefix")
        complete[name] = audit_cell(root, name, index, model_path, config,
                                    environment, instances)
        if name.startswith("ck"):
            require(baselines[name[2:]] == read_json(path),
                    f"{name}: baseline matrix does not match receipt")
    require(complete, "No completed cells to audit")
    require(not require_complete or not pending, f"Incomplete cells: {pending}")
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "scope": "all" if include_merges else "source_baselines",
        "run_root": str(root), "inputs_sha256": digest(root / "inputs.json"),
        "frozen_operational_files_checked": len(frozen["code_sha256"]),
        "service_log": str(service_log),
        "service_block_basis": (
            "Cells run sequentially in frozen checkpoint/merge-arm order, each consuming "
            "456 unique instances. The documented v1 OOM, first MMR IPC, first "
            "linear cache-space, norm-control multiprocess IPC and norm-control TCP bind failures "
            "created no instances. Retries or unrelated calls creating additional "
            "instances invalidate this ordinal attribution; "
            "check launch/service logs before using it for a different run."
        ),
        "unique_service_instances_observed": len(instances),
        "pending_cells": pending, "verified_cells": complete,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--service-log", type=Path, required=True)
    parser.add_argument("--include-merges", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.run_root, args.service_log, args.include_merges,
                   args.require_complete)
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
