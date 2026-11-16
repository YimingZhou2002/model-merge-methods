"""CPU, sliced, FP32 exports; never load a whole input model or overwrite output."""
from contextlib import ExitStack
import gc
import json
import math
from pathlib import Path

import torch

from research.model_merge import weighted_merge as wm
from .core import weighted_ties


def source(path):
    path = Path(path)
    return wm.ModelSource(wm.ModelEntry(path.name, path, 1.))


def source_identity(model):
    shards = sorted({s.file_name for s in model.specs.values()})
    return {"path": str(model.entry.path), "tensor_count": len(model.specs),
            "shards": {name: {"bytes": (model.entry.path/name).stat().st_size,
                              "mtime_ns": (model.entry.path/name).stat().st_mtime_ns}
                       for name in shards},
            "config_sha256": wm._sha256_file(model.entry.path/"config.json"),
            "tokenizer_sha256": wm._tokenizer_fingerprint(model.entry.path),
            "hash_scope": "config/tokenizer SHA256 and shard size/mtime; not full source tensor hashing"}


def export_models(sources, anchor, destinations, transform, metadata, *, chunk_bytes=16_000_000, final_check=None):
    """Write all arms in one input pass; retained .building dirs make failure visible."""
    keys = sorted(anchor.specs)
    for model in sources:
        if set(model.specs) != set(keys) or any(model.specs[k].shape != anchor.specs[k].shape for k in keys):
            raise ValueError(f"Incompatible tensor keys/shapes: {model.entry.path}")
    specs = {key: wm.TensorSpec(anchor.specs[key].file_name, anchor.specs[key].shape, "F32") for key in keys}
    for model in sources:
        if any(not wm._torch_dtype(s.dtype_code).is_floating_point for s in model.specs.values()):
            raise ValueError("Only floating point Qwen3 parameters are supported")
    shards = wm._partition_shards(keys, specs, 2_000_000_000)
    stages = {name: Path(str(dest)+".building") for name, dest in destinations.items()}
    for name, destination in destinations.items():
        if Path(destination).exists() or stages[name].exists():
            raise FileExistsError(f"Existing output requires inspection: {destination}")
    identities = [source_identity(s) for s in sources]
    for stage in stages.values():
        stage.mkdir(parents=True)
        wm._copy_hf_assets(anchor.entry.path, stage)
        config = json.loads((stage/"config.json").read_text())
        config["dtype"] = "float32"
        if "torch_dtype" in config:
            config["torch_dtype"] = "float32"
        wm._atomic_write_json(config, stage/"config.json")
    weight_map = {}
    ready = {name: [] for name in destinations}
    with ExitStack() as stack:
        for model in sources:
            model.attach(stack)
        for index, shard_keys in enumerate(shards, 1):
            filename = f"model-{index:05d}-of-{len(shards):05d}.safetensors"
            output = {name: {} for name in destinations}
            for key in shard_keys:
                buffers = {name: torch.empty(specs[key].shape, dtype=torch.float32) for name in destinations}
                for selection in wm._iter_slices(specs[key].shape, torch.float32, chunk_bytes):
                    values = [model.read(key, selection).float() for model in sources]
                    if any(not torch.isfinite(v).all() for v in values):
                        raise ValueError(f"Nonfinite source tensor: {key}")
                    transformed = transform(values)
                    if set(transformed) != set(destinations):
                        raise ValueError("Transform returned different output arms")
                    for name, tensor in transformed.items():
                        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                            raise ValueError(f"Invalid output tensor: {name}:{key}")
                        if selection is None:
                            buffers[name].copy_(tensor)
                        else:
                            buffers[name][selection].copy_(tensor)
                for name in destinations:
                    output[name][key] = buffers[name]
                weight_map[key] = filename
            for name, tensors in output.items():
                path = stages[name]/filename
                wm._atomic_save_safetensors(tensors, path)
                ready[name].append({"file": filename, "bytes": path.stat().st_size,
                                    "sha256": wm._sha256_file(path)})
            print(f"EXPORT {index}/{len(shards)} {filename}", flush=True)
            del output, buffers, values, transformed
            gc.collect()
    if final_check is not None:
        final_check()
    total = sum(wm._tensor_nbytes(spec) for spec in specs.values())
    for name, stage in stages.items():
        wm._atomic_write_json({"metadata": {"total_size": total}, "weight_map": weight_map}, stage/"model.safetensors.index.json")
        wm._atomic_write_json({"arm": name, "sources": identities, "metadata": metadata,
                               "tensor_count": len(keys), "output_dtype": "float32"}, stage/"merge_audit.json")
        wm._atomic_write_json({"weights": ready[name]}, stage/"EXPORT_READY.json")
        stage.rename(destinations[name])


def build_primary(config, selection, root):
    pool = Path(config["pool"])
    steps = config["checkpoints"]
    models = [source(pool/f"checkpoint-{s}") for s in steps]
    base = source(config["base_model"])
    anchor = models[steps.index(config["anchor"])]
    wm._load_and_validate_config(models)
    wm._validate_tokenizer_assets(models)
    weights = selection["weights"]
    bank = [steps.index(int(s)) for s in sorted(weights, key=int)]
    w = [weights[str(steps[i])] for i in bank]
    raw2 = gated2 = 0.

    def transform(values):
        nonlocal raw2, gated2
        checkpoints, raw_base = values[:-1], values[-1]
        a = checkpoints[steps.index(config["anchor"])]
        linear = torch.zeros_like(a)
        for i, weight in zip(bank, w):
            linear.add_(checkpoints[i], alpha=weight)
        u = linear-a
        travel = torch.zeros_like(a)
        for left, right in zip(checkpoints, checkpoints[1:]):
            travel.add_((right-left).abs())
        gate = torch.where(travel > 0, 1-(checkpoints[-1]-checkpoints[0]).abs()/travel.clamp_min(1e-30), 0.).clamp_(0, 1)
        update = gate*u
        raw2 += float(u.double().square().sum())
        gated2 += float(((a+update)-a).double().square().sum())
        return {"linear": linear, "mmr_ties": weighted_ties(raw_base, [checkpoints[i] for i in bank], w),
                "tsg_tv": a+update}

    export_models(models+[base], anchor, {name: Path(root)/name for name in ["linear", "mmr_ties", "tsg_tv"]},
                  transform, {"selection": selection, "trajectory_steps": steps,
                              "ties": {"density": 1., "normalize": True, "lambda": 1.}})
    eta = math.sqrt(gated2/raw2) if raw2 else 0.
    wm._atomic_write_json({"raw_update_norm2": raw2, "tv_update_norm2": gated2, "eta": eta}, Path(root)/"norm_control.json")
    return eta


def build_norm_control(config, root):
    root = Path(root)
    eta = json.loads((root/"norm_control.json").read_text())["eta"]
    anchor = source(Path(config["pool"])/f"checkpoint-{config['anchor']}")
    linear = source(root/"linear")
    actual2 = 0.

    def transform(values):
        nonlocal actual2
        a, l = values
        value = a+eta*(l-a)
        actual2 += float((value-a).double().square().sum())
        return {"tsg_tv_norm": value}

    expected = json.loads((root/"norm_control.json").read_text())["tv_update_norm2"]
    def check_norm():
        if abs(math.sqrt(actual2)-math.sqrt(expected)) > .002*math.sqrt(expected)+1e-6:
            raise ValueError("Exported global control failed update norm matching")
    export_models([anchor, linear], anchor, {"tsg_tv_norm": root/"tsg_tv_norm"}, transform,
                  {"eta": eta, "control": "global interpolation at the same anchor-relative update norm as TSG-TV"},
                  final_check=check_norm)
