"""Freeze recipes, export round-1 models and execute a resumable dev queue."""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import torch

from research.checkpoint_merge_14b import study
from research.checkpoint_merge_14b.core import read_rollouts, paired_comparison
from research.checkpoint_merge_14b.merge import source, source_identity, export_models
from research.model_merge import weighted_merge as wm
from .methods import calibrate, transform

REPO = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("MODEL_MERGE_WORKSPACE", REPO.parent)).resolve()
OLD = Path(os.environ.get("MMR_TSG_RUN_ROOT", WORK / "runs/qwen3-14b-dev8-v2")).resolve()
ROOT = Path(os.environ.get("BCM_RUN_ROOT", WORK / "runs/temporal-search14b-r1")).resolve()
HERE = Path(__file__).resolve().parent
EXPORTS = ["bcm_linear", "qis_05", "qis_10", "drr_05", "drr_10", "bcm_05", "bcm_10"]
ORDER = ["tsg_tv", *EXPORTS]
SERVICE = WORK / "appworld-service.log"
EVENT = re.compile(r"Creating instance with env_type: appworld, task_id: ([^,\s]+), instance_id: ([0-9a-f]{32})")


def sha(path):
    return wm._sha256_file(Path(path))


def write_new(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+"\n"
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError(f"Refusing to change frozen artifact: {path}")
    else:
        with path.open("x") as handle:
            handle.write(encoded)


def protected_files():
    files = [p for p in HERE.iterdir() if p.suffix in {".py", ".md"}]
    files += list(Path(p) for p in json.loads((OLD/"inputs.json").read_text())["code_sha256"])
    files += [REPO/"research/model_merge/weighted_merge.py"]
    return {str(p): sha(p) for p in sorted(set(files))}


def verify_frozen(frozen):
    for path, expected in frozen["code_sha256"].items():
        if sha(path) != expected:
            raise ValueError(f"Operational code changed after freeze: {path}")
    if sha(ROOT/"recipes.json") != frozen["recipes_sha256"]:
        raise ValueError("Recipes changed after freeze")


def prepare():
    if json.loads((OLD/"status.json").read_text())["phase"] != "complete":
        raise ValueError("Original study is not complete")
    frozen_old = json.loads((OLD/"inputs.json").read_text())
    for path, expected in frozen_old["code_sha256"].items():
        if sha(path) != expected:
            raise ValueError(f"Original frozen file changed: {path}")
    environment = json.loads((OLD/"environment.json").read_text())
    instructions = environment["appworld"]["task_instructions"]
    results = []
    steps = frozen_old["config"]["checkpoints"]
    for step in steps:
        path = OLD/f"evaluations/ck{step}"
        result = read_rollouts(path/"validation_log/0.jsonl", task_instructions=instructions)
        if result != json.loads((path/"VERIFIED.json").read_text()):
            raise ValueError("Historical calibration receipt no longer matches raw scores")
        if results and any(result[k] != results[0][k] for k in ["task_keys", "task_families"]):
            raise ValueError("Calibration tasks/families differ")
        results.append(result)
    calibration = calibrate([r["vector"] for r in results], results[0]["task_families"])
    config = dict(frozen_old["config"], study="temporal-search14b-r1", run_root=str(ROOT),
                  seed=20261110, ray_namespace=300)
    original_weights = json.loads((OLD/"recipes.json").read_text())["selection"]["weights"]
    historical = [original_weights.get(str(s), 0.) for s in steps]
    recipe = {"calibration": calibration, "historical_weights": historical,
              "steps": steps, "order": ORDER, "exports": EXPORTS,
              "calibration_logs": {str(s): r["log_sha256"] for s, r in zip(steps, results)},
              "calibration_split": "previously examined dev; not holdout",
              "all_cells_seed": 20261110, "strengths": [.5, 1.],
              "coordinate_cap_multiple": 2., "novelty": "proposed hypotheses; priority and effectiveness unestablished"}
    ROOT.mkdir(parents=True, exist_ok=True)
    physical = Path(os.environ.get("BCM_EVAL_PHYSICAL_DIR", WORK/"r/s14")).resolve()
    if not (ROOT/"evaluations").exists():
        if physical.exists():
            raise FileExistsError("Unattached existing short output directory")
        physical.mkdir()
        (ROOT/"evaluations").symlink_to(physical)
    if (ROOT/"evaluations").resolve() != physical:
        raise ValueError("Evaluation physical path differs from protocol")
    write_new(environment, ROOT/"environment.json")
    write_new(config, ROOT/"config.json")
    write_new(recipe, ROOT/"recipes.json")
    identities = {str(s): source_identity(source(WORK/f"model-pool/checkpoint-{s}")) for s in steps}
    for s in steps:
        if identities[str(s)] != frozen_old["models"][str(s)]:
            raise ValueError(f"Source identity changed since original study: {s}")
    write_new({"config": config, "code_sha256": protected_files(), "source_identities": identities,
               "recipes_sha256": sha(ROOT/"recipes.json"), "historical_commit": "45bb1210b33abcae2bece92731393c1e1269455a",
               "plan_sha256": sha(HERE/"PLAN.md")}, ROOT/"inputs.json")
    print(json.dumps({"phase": "frozen", "run_root": str(ROOT),
                      "source_quality": calibration["quality"],
                      "negative_interval_confidence": calibration["negative_interval_confidence"],
                      "bcm_weights": calibration["portfolio_weights"]}), flush=True)


def export():
    frozen = json.loads((ROOT/"inputs.json").read_text())
    verify_frozen(frozen)
    recipe = json.loads((ROOT/"recipes.json").read_text())
    sources = [source(WORK/f"model-pool/checkpoint-{s}") for s in recipe["steps"]]
    for step, model in zip(recipe["steps"], sources):
        if source_identity(model) != frozen["source_identities"][str(step)]:
            raise ValueError("Source changed before export")
    wm._load_and_validate_config(sources)
    wm._validate_tokenizer_assets(sources)
    torch.set_num_threads(16)
    statistics = {name: {"anchor_update_norm2": 0., "coordinates": 0} for name in EXPORTS}

    def apply(values):
        output = transform(values, recipe["calibration"], recipe["historical_weights"])
        for name, value in output.items():
            statistics[name]["anchor_update_norm2"] += float((value-values[-1]).double().square().sum())
            statistics[name]["coordinates"] += value.numel()
        return output

    def final_check():
        verify_frozen(frozen)
        if any(source_identity(m) != frozen["source_identities"][str(s)] for s, m in zip(recipe["steps"], sources)):
            raise ValueError("Source identity changed during export")

    export_models(sources, sources[-1], {n: ROOT/"models"/n for n in EXPORTS}, apply,
                  {"recipes_sha256": frozen["recipes_sha256"], "plan_sha256": frozen["plan_sha256"],
                   "methods": recipe}, chunk_bytes=8_000_000, final_check=final_check)
    write_new(statistics, ROOT/"export-statistics.json")
    print("All seven round-1 exports ready", flush=True)


def audit_cell(name, offset, service_stat):
    path = ROOT/"evaluations"/name
    config = json.loads((ROOT/"config.json").read_text())
    environment = json.loads((ROOT/"environment.json").read_text())
    result = read_rollouts(path/"validation_log/0.jsonl", task_instructions=environment["appworld"]["task_instructions"])
    if any(isinstance(row.get("score"), bool) or row.get("score") not in [0, 1]
           for row in map(json.loads, (path/"validation_log/0.jsonl").read_text().splitlines()) if row):
        raise ValueError("Nonbinary raw scores")
    reference = json.loads((OLD/"evaluations/tsg_tv/VERIFIED.json").read_text())
    if any(result[k] != reference[k] for k in ["task_keys", "task_families"]):
        raise ValueError("Task/prompt alignment differs from the original protocol")
    current = SERVICE.stat()
    if (current.st_dev, current.st_ino) != tuple(service_stat) or current.st_size < offset:
        raise ValueError("Service log rotated/truncated during evaluation")
    instances = {}
    with SERVICE.open("rb") as handle:
        handle.seek(offset)
        for line in handle:
            match = EVENT.search(line.decode(errors="replace"))
            if match:
                task, instance = match.groups()
                if instance in instances and instances[instance] != task:
                    raise ValueError("Instance identity reused across tasks")
                instances[instance] = task
    counts = Counter(instances.values())
    if counts != Counter({task: 8 for task in environment["appworld"]["task_ids"]}):
        raise ValueError(f"Service task counts differ: {counts}")
    import yaml
    resolved = yaml.safe_load((path/"resolved_config.yaml").read_text())
    if resolved["trainer"]["n_gpus_per_node"] != 8 or resolved["trainer"]["nnodes"] != 1:
        raise ValueError("Actual GPU protocol differs")
    return {"name": name, "result": result, "resolved_config_sha256": sha(path/"resolved_config.yaml"),
            "service_start_offset": offset, "service_end_offset": current.st_size,
            "service_log_identity": list(service_stat), "service_instances": len(instances),
            "service_task_counts": dict(sorted(counts.items())), "all_cells_seed": config["seed"]}


def report(results):
    comparisons = {name: paired_comparison(value, results["tsg_tv"], seed=20261110)
                   for name, value in results.items() if name != "tsg_tv"}
    output = {"complete": list(results), "pending": [n for n in ORDER if n not in results],
              "results": results, "vs_concurrent_tsg_tv": comparisons,
              "interpretation": "Exploratory reused-dev search; unadjusted intervals conditional on these sessions. No confirmed winners yet."}
    wm._atomic_write_json(output, ROOT/"results.json")
    lines = ["# Temporal merge search round 1 (development only)", "",
             "All cells: 57 tasks x8, T=0.9, FP32, seed=20261110. Original dev has already been examined.", "",
             "| Model | Mean@8 % | Pass@8 % | Delta vs concurrent TV pp | Family 95% pp |",
             "|---|---:|---:|---:|---|"]
    for name, value in results.items():
        comparison = comparisons.get(name)
        delta = f"{comparison['delta_pp']:+.3f}" if comparison else "reference"
        interval = str([round(x, 3) for x in comparison["family_bootstrap_95_pp"]]) if comparison else ""
        lines.append(f"| {name} | {100*value['mean_at_8']:.3f} | {100*value['pass_at_8']:.3f} | {delta} | {interval} |")
    lines += ["", "Pending: "+", ".join(output["pending"]), "", output["interpretation"]]
    (ROOT/"RESULTS.md").write_text("\n".join(lines)+"\n")


def run():
    frozen = json.loads((ROOT/"inputs.json").read_text())
    config = json.loads((ROOT/"config.json").read_text())
    with (ROOT/"study.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_frozen(frozen)
        results = {}
        for name in ORDER:
            verify_frozen(frozen)
            cell_config = dict(config, ray_namespace=config["ray_namespace"]+ORDER.index(name))
            model = OLD/"models/tsg_tv" if name == "tsg_tv" else ROOT/"models"/name
            receipt = ROOT/"evaluations"/name/"CELL_AUDIT.json"
            if name != "tsg_tv":
                export_audit = json.loads((ROOT/"independent-export-audit.json").read_text())
                if export_audit["inputs_sha256"] != sha(ROOT/"inputs.json") or export_audit["recipes_sha256"] != frozen["recipes_sha256"]:
                    raise ValueError("Independent export audit refers to different inputs/recipes")
            if receipt.exists():
                audit = json.loads(receipt.read_text())
                result = study.evaluate(cell_config, name, model, 0)
                if audit["result"] != result or audit["resolved_config_sha256"] != sha(ROOT/"evaluations"/name/"resolved_config.yaml"):
                    raise ValueError("Completed cell changed")
            else:
                if not (model/"EXPORT_READY.json").is_file():
                    raise FileNotFoundError(f"Model not ready: {model}; export first")
                info = SERVICE.stat()
                offset, identity = info.st_size, (info.st_dev, info.st_ino)
                write_new({"start_offset": offset, "service_log_identity": list(identity), "model": str(model),
                           "seed": config["seed"], "started_at": study.now()}, ROOT/f"cell-start-{name}.json")
                # index 0 gives every arm the same seed; the namespace above gives
                # each arm independent Ray logs and sockets.
                result = study.evaluate(cell_config, name, model, 0)
                write_new(audit_cell(name, offset, identity), receipt)
            results[name] = result
            report(results)
        study.state(config, "round1_complete", results=str(ROOT/"RESULTS.md"),
                    next_phase="Freeze finalists or preregister a bounded second development round; objective remains active")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "export", "run"])
    args = parser.parse_args()
    {"prepare": prepare, "export": export, "run": run}[args.action]()


if __name__ == "__main__":
    main()
