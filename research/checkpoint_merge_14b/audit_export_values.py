"""Independent export audit: file hashes, formula probes, full finite/norm scan.

Reads safetensors headers and NumPy memory maps directly, without importing the
merge implementation. Formula probes cover up to 32 coordinates in each tensor;
they are not a full independent recomputation of the merge formulas.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np


ARMS = ("linear", "mmr_ties", "tsg_tv", "tsg_tv_norm")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class Model:
    def __init__(self, path):
        self.path = Path(path)
        self.index = read(self.path / "model.safetensors.index.json")
        self.specs = {}
        for filename in sorted(set(self.index["weight_map"].values())):
            file = self.path / filename
            with file.open("rb") as stream:
                size = struct.unpack("<Q", stream.read(8))[0]
                header = json.loads(stream.read(size))
            for key, spec in header.items():
                if key == "__metadata__":
                    continue
                require(key not in self.specs, f"Duplicate tensor: {file}:{key}")
                require(spec["dtype"] in ("F32", "BF16"), f"Unsupported dtype: {file}:{key}")
                require(self.index["weight_map"].get(key) == filename,
                        f"Incorrect index location: {file}:{key}")
                count = math.prod(spec["shape"])
                start, end = spec["data_offsets"]
                itemsize = 4 if spec["dtype"] == "F32" else 2
                require(end - start == count * itemsize and 8 + size + end <= file.stat().st_size,
                        f"Incorrect tensor byte range: {file}:{key}")
                self.specs[key] = {**spec, "file": file, "offset": 8 + size + start, "count": count}
        require(set(self.specs) == set(self.index["weight_map"]), f"Index key mismatch: {path}")

    def array(self, key):
        spec = self.specs[key]
        return np.memmap(spec["file"], mode="r", offset=spec["offset"],
                         dtype="<f4" if spec["dtype"] == "F32" else "<u2",
                         shape=(spec["count"],))

    def samples(self, key, indices):
        values = self.array(key)[indices].copy()
        if self.specs[key]["dtype"] == "BF16":
            values = (values.astype(np.uint32) << 16).view(np.float32)
        return values.astype(np.float64)


def audit(root):
    frozen = read(root / "inputs.json")
    config = frozen["config"]
    recipes = read(root / "recipes.json")
    norm = read(root / "models" / "norm_control.json")
    for filename, expected in frozen["code_sha256"].items():
        require(sha(Path(filename)) == expected, f"Frozen code changed: {filename}")
    steps = [str(s) for s in config["checkpoints"]]
    sources = {step: Model(Path(config["pool"]) / f"checkpoint-{step}") for step in steps}
    base = Model(config["base_model"])
    anchor = sources[str(config["anchor"])]
    outputs = {name: Model(root / "models" / name) for name in ARMS}
    shapes = {key: spec["shape"] for key, spec in anchor.specs.items()}
    require(len(shapes) == 443, "Expected 443 Qwen3-14B tensors")
    for model in [*sources.values(), base, *outputs.values()]:
        require({k: s["shape"] for k, s in model.specs.items()} == shapes,
                f"Tensor coordinates differ: {model.path}")
    integrity = {}
    for name, model in outputs.items():
        require(all(s["dtype"] == "F32" for s in model.specs.values()), f"Output is not FP32: {name}")
        ready = read(model.path / "EXPORT_READY.json")["weights"]
        filenames = [item["file"] for item in ready]
        require(len(filenames) == len(set(filenames)) and set(filenames)
                == set(model.index["weight_map"].values())
                == {p.name for p in model.path.glob("*.safetensors")}, f"Shard inventory differs: {name}")
        for item in ready:
            path = model.path / item["file"]
            require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"],
                    f"Full shard SHA256 mismatch: {path}")
        require(model.index["metadata"]["total_size"] == sum(s["count"] * 4 for s in model.specs.values()),
                f"Tensor byte count differs: {name}")
        integrity[name] = {"shards": len(ready), "shard_bytes": sum(x["bytes"] for x in ready),
                           "tensor_count": len(model.specs), "all_full_shard_sha256s_match": True,
                           "all_tensor_shapes_match_anchor": True, "dtype": "F32",
                           "export_ready_sha256": sha(model.path / "EXPORT_READY.json")}
        print(f"INTEGRITY {name}: {len(ready)} full shard SHA256s match", flush=True)

    rng = np.random.default_rng(20261109)
    bank = sorted(recipes["selection"]["weights"], key=int)
    weights = np.array([recipes["selection"]["weights"][s] for s in bank])[:, None]
    probes = {name: {"coordinates": 0, "max_absolute_error": 0., "max_fraction_of_tolerance": 0.}
              for name in ARMS}
    atol, rtol = 5e-7, 2e-6
    for key in sorted(shapes):
        count = anchor.specs[key]["count"]
        indices = np.unique(np.r_[0, count // 2, count - 1, rng.integers(count, size=29)])
        checkpoint = {s: model.samples(key, indices) for s, model in sources.items()}
        actual = {name: model.samples(key, indices) for name, model in outputs.items()}
        a = checkpoint[str(config["anchor"])]
        raw_base = base.samples(key, indices)
        weighted = (np.stack([checkpoint[s] for s in bank]) - raw_base) * weights
        elected = np.where(weighted.sum(axis=0) >= 0, 1., -1.)
        aligned = np.sign(weighted) == elected
        divisor = (weights * aligned).sum(axis=0)
        divisor = np.where(divisor == 0, 1., divisor)
        trajectory = np.stack([checkpoint[s] for s in steps])
        travel = np.abs(np.diff(trajectory, axis=0)).sum(axis=0)
        gate = np.zeros_like(travel)
        np.divide(np.abs(trajectory[-1] - trajectory[0]), travel, out=gate, where=travel > 0)
        gate = np.where(travel > 0, np.clip(1 - gate, 0, 1), 0.)
        expected = {
            "linear": (np.stack([checkpoint[s] for s in bank]) * weights).sum(axis=0),
            "mmr_ties": raw_base + (weighted * aligned).sum(axis=0) / divisor,
            "tsg_tv": a + gate * (actual["linear"] - a),
            "tsg_tv_norm": a + norm["eta"] * (actual["linear"] - a),
        }
        for name in ARMS:
            difference = np.abs(actual[name] - expected[name])
            fraction = difference / (atol + rtol * np.abs(expected[name]))
            require(np.isfinite(actual[name]).all() and (fraction <= 1).all(),
                    f"Numerical formula probe failed: {name}:{key}, fraction={fraction.max()}")
            probes[name]["coordinates"] += len(indices)
            probes[name]["max_absolute_error"] = max(probes[name]["max_absolute_error"], float(difference.max()))
            probes[name]["max_fraction_of_tolerance"] = max(probes[name]["max_fraction_of_tolerance"], float(fraction.max()))
    print("FORMULA_PROBES all four arms passed", flush=True)

    norms2 = {name: 0. for name in ("linear", "tsg_tv", "tsg_tv_norm")}
    scanned = 0
    for number, key in enumerate(sorted(shapes), 1):
        arrays = {name: model.array(key) for name, model in outputs.items()}
        anchor_array = anchor.array(key)
        count = anchor.specs[key]["count"]
        for start in range(0, count, 4_000_000):
            stop = min(start + 4_000_000, count)
            a = anchor_array[start:stop].astype(np.float64)
            require(np.isfinite(a).all(), f"Nonfinite anchor: {key}")
            for name, array in arrays.items():
                values = array[start:stop]
                require(np.isfinite(values).all(), f"Nonfinite output: {name}:{key}")
                if name in norms2:
                    delta = values.astype(np.float64) - a
                    norms2[name] += float(np.sum(delta * delta, dtype=np.float64))
            scanned += stop - start
        if number % 50 == 0:
            print(f"FINITE_NORM_SCAN {number}/443 tensors", flush=True)
    require(math.isclose(norms2["linear"], norm["raw_update_norm2"], rel_tol=2e-6, abs_tol=1e-12),
            "Independent raw update norm differs")
    require(math.isclose(norms2["tsg_tv"], norm["tv_update_norm2"], rel_tol=2e-6, abs_tol=1e-12),
            "Independent TV update norm differs")
    eta = math.sqrt(norms2["tsg_tv"] / norms2["linear"]) if norms2["linear"] else 0.
    require(math.isclose(eta, norm["eta"], rel_tol=2e-6, abs_tol=1e-12), "Independent eta differs")
    tv_norm, control_norm = math.sqrt(norms2["tsg_tv"]), math.sqrt(norms2["tsg_tv_norm"])
    require(abs(control_norm - tv_norm) <= .002 * tv_norm + 1e-6, "Control update norm does not match TV")
    return {"checked_at": datetime.now(timezone.utc).isoformat(), "run_root": str(root),
            "inputs_sha256": sha(root / "inputs.json"), "recipes_sha256": sha(root / "recipes.json"),
            "integrity": integrity, "formula_probes": probes,
            "probe_protocol": {"tensors": 443, "up_to_coordinates_per_tensor": 32,
                               "seed": 20261109, "absolute_tolerance": atol, "relative_tolerance": rtol,
                               "reference_precision": "float64", "full_formula_recomputation": False,
                               "tv_and_norm_targets": "actual stored FP32 linear average"},
            "full_finite_norm_scan": {"coordinates_per_model": scanned, "models_checked": list(ARMS),
                                      "all_output_values_finite": True, "anchor_relative_update_norm2": norms2,
                                      "independent_eta": eta, "recorded_eta": norm["eta"],
                                      "tv_control_norm_relative_difference": abs(control_norm - tv_norm) / tv_norm
                                      if tv_norm else 0., "matching_tolerance": ".002 * tv_norm + 1e-6"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"Audit output already exists: {args.output}")
    result = audit(args.run_root)
    result["auditor_sha256"] = sha(Path(__file__))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("EXPORT_VALUE_AUDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
