#!/usr/bin/env python3
"""Memory-bounded weighted merge for Hugging Face safetensors checkpoints.

The implementation deliberately never calls ``torch.load`` and never materializes
an input model.  Inputs are memory-mapped with safetensors, one output tensor is
built at a time, and floating-point inputs are read and accumulated in slices.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file


class MergeError(RuntimeError):
    """Raised when inputs are incompatible or a merge cannot be completed safely."""


@dataclass(frozen=True)
class TensorSpec:
    file_name: str
    shape: tuple[int, ...]
    dtype_code: str


@dataclass(frozen=True)
class ModelEntry:
    name: str
    path: Path
    weight: float


_DTYPE_ATTRS = {
    "BOOL": "bool",
    "U8": "uint8",
    "I8": "int8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "U16": "uint16",
    "U32": "uint32",
    "U64": "uint64",
    "F16": "float16",
    "BF16": "bfloat16",
    "F32": "float32",
    "F64": "float64",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
}


def _torch_dtype(dtype_code: str) -> torch.dtype:
    attr = _DTYPE_ATTRS.get(dtype_code)
    dtype = getattr(torch, attr, None) if attr else None
    if dtype is None:
        raise MergeError(
            f"unsupported safetensors dtype {dtype_code!r} in this torch version"
        )
    return dtype


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise MergeError(f"cannot read JSON file {path}: {exc}") from exc


def _safe_child(directory: Path, relative_name: str) -> Path:
    relative = Path(relative_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise MergeError(f"unsafe shard path in index: {relative_name!r}")
    candidate = (directory / relative).resolve()
    try:
        candidate.relative_to(directory)
    except ValueError as exc:
        raise MergeError(f"shard escapes model directory: {relative_name!r}") from exc
    if not candidate.is_file():
        raise MergeError(f"missing safetensors shard: {candidate}")
    return candidate


class ModelSource:
    """Metadata plus lazily-opened safetensors shards for one HF directory."""

    def __init__(self, entry: ModelEntry) -> None:
        self.entry = entry
        self.specs = self._scan()
        self._stack: ExitStack | None = None
        self._handles: dict[str, Any] = {}

    def _scan(self) -> dict[str, TensorSpec]:
        directory = self.entry.path
        index_path = directory / "model.safetensors.index.json"
        indexed_map: dict[str, str] | None = None

        if index_path.is_file():
            index = _read_json(index_path)
            if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
                raise MergeError(f"invalid weight_map in {index_path}")
            indexed_map = {}
            for key, value in index["weight_map"].items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise MergeError(f"non-string weight_map entry in {index_path}")
                indexed_map[key] = value
            shard_names = sorted(set(indexed_map.values()))
        elif (directory / "model.safetensors").is_file():
            shard_names = ["model.safetensors"]
        else:
            shard_names = sorted(path.name for path in directory.glob("*.safetensors"))

        if not shard_names:
            raise MergeError(f"no HF safetensors weights found in {directory}")

        specs: dict[str, TensorSpec] = {}
        actual_map: dict[str, str] = {}
        for shard_name in shard_names:
            shard_path = _safe_child(directory, shard_name)
            try:
                with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                    for key in handle.keys():
                        if key in specs:
                            raise MergeError(
                                f"duplicate tensor {key!r} across shards in {directory}"
                            )
                        view = handle.get_slice(key)
                        specs[key] = TensorSpec(
                            file_name=shard_name,
                            shape=tuple(view.get_shape()),
                            dtype_code=str(view.get_dtype()),
                        )
                        actual_map[key] = shard_name
            except MergeError:
                raise
            except Exception as exc:
                raise MergeError(f"cannot inspect safetensors shard {shard_path}: {exc}") from exc

        if indexed_map is not None and indexed_map != actual_map:
            missing = sorted(set(indexed_map) - set(actual_map))[:5]
            extra = sorted(set(actual_map) - set(indexed_map))[:5]
            wrong_shard = sorted(
                key
                for key in set(indexed_map) & set(actual_map)
                if indexed_map[key] != actual_map[key]
            )[:5]
            raise MergeError(
                f"index does not match shard contents in {directory}; "
                f"missing={missing}, extra={extra}, wrong_shard={wrong_shard}"
            )
        return specs

    def attach(self, stack: ExitStack) -> None:
        self._stack = stack

    def _handle(self, file_name: str) -> Any:
        if self._stack is None:
            raise RuntimeError("ModelSource must be attached to an ExitStack before reading")
        if file_name not in self._handles:
            shard = _safe_child(self.entry.path, file_name)
            self._handles[file_name] = self._stack.enter_context(
                safe_open(str(shard), framework="pt", device="cpu")
            )
        return self._handles[file_name]

    def read(self, key: str, selection: tuple[slice, ...] | None) -> torch.Tensor:
        spec = self.specs[key]
        handle = self._handle(spec.file_name)
        if selection is None:
            return handle.get_tensor(key)
        return handle.get_slice(key)[selection]


def _parse_manifest(path: Path, normalize_weights: bool) -> tuple[list[ModelEntry], dict[str, Any]]:
    payload = _read_json(path)
    if isinstance(payload, list):
        raw_models = payload
        metadata: dict[str, Any] = {}
    elif isinstance(payload, dict) and isinstance(payload.get("models"), list):
        raw_models = payload["models"]
        raw_metadata = payload.get("metadata", {})
        if not isinstance(raw_metadata, dict):
            raise MergeError("manifest metadata must be a JSON object")
        metadata = raw_metadata
    else:
        raise MergeError("manifest must be a model-entry list or an object with a models list")

    if not raw_models:
        raise MergeError("manifest models list is empty")

    entries: list[ModelEntry] = []
    seen_names: set[str] = set()
    seen_paths: set[Path] = set()
    for index, raw in enumerate(raw_models):
        if not isinstance(raw, dict):
            raise MergeError(f"models[{index}] must be an object")
        raw_path = raw.get("path")
        raw_weight = raw.get("weight")
        if not isinstance(raw_path, str) or not raw_path:
            raise MergeError(f"models[{index}].path must be a non-empty string")
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
            raise MergeError(f"models[{index}].weight must be numeric")
        weight = float(raw_weight)
        if not math.isfinite(weight):
            raise MergeError(f"models[{index}].weight must be finite")

        model_path = Path(raw_path).expanduser()
        if not model_path.is_absolute():
            model_path = path.parent / model_path
        model_path = model_path.resolve()
        if not model_path.is_dir():
            raise MergeError(f"model directory does not exist: {model_path}")
        name = raw.get("name", model_path.name)
        if not isinstance(name, str) or not name:
            raise MergeError(f"models[{index}].name must be a non-empty string")
        if name in seen_names:
            raise MergeError(f"duplicate model name: {name!r}")
        if model_path in seen_paths:
            raise MergeError(f"duplicate model path: {model_path}")
        seen_names.add(name)
        seen_paths.add(model_path)
        entries.append(ModelEntry(name=name, path=model_path, weight=weight))

    total = math.fsum(entry.weight for entry in entries)
    if normalize_weights:
        if abs(total) < 1e-12:
            raise MergeError("cannot normalize weights whose sum is zero")
        entries = [
            ModelEntry(name=entry.name, path=entry.path, weight=entry.weight / total)
            for entry in entries
        ]
    elif not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise MergeError(
            f"model weights must sum to 1 (got {total:.12g}); "
            "pass --normalize-weights to normalize explicitly"
        )
    return entries, metadata


def _load_and_validate_config(sources: Sequence[ModelSource]) -> tuple[Any, str]:
    reference_path = sources[0].entry.path / "config.json"
    if not reference_path.is_file():
        raise MergeError(f"missing config.json in {sources[0].entry.path}")
    reference = _read_json(reference_path)
    canonical = _canonical_json(reference)
    for source in sources[1:]:
        candidate_path = source.entry.path / "config.json"
        if not candidate_path.is_file():
            raise MergeError(f"missing config.json in {source.entry.path}")
        if _canonical_json(_read_json(candidate_path)) != canonical:
            raise MergeError(
                f"config.json mismatch between {sources[0].entry.name!r} "
                f"and {source.entry.name!r}"
            )
    return reference, hashlib.sha256(canonical).hexdigest()


def _is_tokenizer_asset(relative: Path) -> bool:
    name = relative.name.lower()
    return (
        "tokenizer" in name
        or "special_tokens" in name
        or "added_tokens" in name
        or "chat_template" in name
        or name in {"vocab.json", "vocab.txt", "merges.txt", "spiece.model"}
        or name.endswith(".model")
    )


def _tokenizer_fingerprint(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): _sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and _is_tokenizer_asset(path.relative_to(directory))
    }


def _validate_tokenizer_assets(sources: Sequence[ModelSource]) -> str | None:
    reference = _tokenizer_fingerprint(sources[0].entry.path)
    for source in sources[1:]:
        candidate = _tokenizer_fingerprint(source.entry.path)
        if candidate != reference:
            reference_keys = set(reference)
            candidate_keys = set(candidate)
            changed = sorted(
                key
                for key in reference_keys & candidate_keys
                if reference[key] != candidate[key]
            )[:5]
            raise MergeError(
                f"tokenizer assets mismatch between {sources[0].entry.name!r} and "
                f"{source.entry.name!r}; missing={sorted(reference_keys-candidate_keys)[:5]}, "
                f"extra={sorted(candidate_keys-reference_keys)[:5]}, changed={changed}"
            )
    if not reference:
        return None
    return hashlib.sha256(_canonical_json(reference)).hexdigest()


def _validate_tensor_specs(sources: Sequence[ModelSource]) -> list[str]:
    reference = sources[0]
    keys = set(reference.specs)
    for source in sources[1:]:
        candidate_keys = set(source.specs)
        if candidate_keys != keys:
            raise MergeError(
                f"tensor keys mismatch between {reference.entry.name!r} and "
                f"{source.entry.name!r}; missing={sorted(keys-candidate_keys)[:5]}, "
                f"extra={sorted(candidate_keys-keys)[:5]}"
            )
        for key in sorted(keys):
            expected = reference.specs[key]
            actual = source.specs[key]
            if actual.shape != expected.shape:
                raise MergeError(
                    f"shape mismatch for {key!r}: {reference.entry.name}={expected.shape}, "
                    f"{source.entry.name}={actual.shape}"
                )
            if actual.dtype_code != expected.dtype_code:
                raise MergeError(
                    f"dtype mismatch for {key!r}: "
                    f"{reference.entry.name}={expected.dtype_code}, "
                    f"{source.entry.name}={actual.dtype_code}"
                )
    return sorted(keys)


def _tensor_nbytes(spec: TensorSpec) -> int:
    elements = math.prod(spec.shape) if spec.shape else 1
    return elements * torch.empty((), dtype=_torch_dtype(spec.dtype_code)).element_size()


def _iter_slices(
    shape: tuple[int, ...], dtype: torch.dtype, chunk_bytes: int
) -> Iterator[tuple[slice, ...] | None]:
    if not shape:
        yield None
        return
    if math.prod(shape) == 0:
        return

    reference_item_size = torch.empty((), dtype=dtype).element_size()
    # For FP16/BF16 inputs the FP32 accumulator is the largest per-element
    # working buffer, so make chunk_bytes constrain that buffer rather than the
    # smaller serialized input slice.
    item_size = max(reference_item_size, 4) if dtype.is_floating_point else reference_item_size
    total_elements = math.prod(shape)
    total_bytes = total_elements * item_size
    if total_bytes <= chunk_bytes:
        yield tuple(slice(None) for _ in shape)
        return

    # Slice along the axis with the smallest cross-section.  Unlike always using
    # dim 0, this keeps [1, very_large] tensors bounded too.
    candidates = []
    for axis, extent in enumerate(shape):
        if extent > 1:
            unit_bytes = (total_elements // extent) * item_size
            candidates.append((unit_bytes, -extent, axis))
    if not candidates:
        yield tuple(slice(None) for _ in shape)
        return
    unit_bytes, _, axis = min(candidates)
    units_per_chunk = max(1, chunk_bytes // unit_bytes)
    for start in range(0, shape[axis], units_per_chunk):
        selection = [slice(None) for _ in shape]
        selection[axis] = slice(start, min(shape[axis], start + units_per_chunk))
        yield tuple(selection)


def _merge_tensor(
    key: str,
    sources: Sequence[ModelSource],
    chunk_bytes: int,
) -> torch.Tensor:
    spec = sources[0].specs[key]
    dtype = _torch_dtype(spec.dtype_code)
    output = torch.empty(spec.shape, dtype=dtype, device="cpu")
    if output.numel() == 0:
        return output

    is_floating = torch.is_floating_point(output)
    for selection in _iter_slices(spec.shape, dtype, chunk_bytes):
        if is_floating:
            accumulator: torch.Tensor | None = None
            for source in sources:
                value = source.read(key, selection)
                value_fp32 = value.to(dtype=torch.float32)
                if accumulator is None:
                    accumulator = value_fp32.mul(source.entry.weight)
                else:
                    accumulator.add_(value_fp32, alpha=source.entry.weight)
                del value, value_fp32
            assert accumulator is not None
            converted = accumulator.to(dtype=dtype)
            if selection is None:
                output.copy_(converted)
            else:
                output[selection].copy_(converted)
            del accumulator, converted
        else:
            reference = sources[0].read(key, selection)
            for source in sources[1:]:
                candidate = source.read(key, selection)
                if not torch.equal(candidate, reference):
                    raise MergeError(
                        f"non-floating tensor {key!r} differs between "
                        f"{sources[0].entry.name!r} and {source.entry.name!r}"
                    )
                del candidate
            if selection is None:
                output.copy_(reference)
            else:
                output[selection].copy_(reference)
            del reference
    return output.contiguous()


def _partition_shards(
    keys: Sequence[str], specs: Mapping[str, TensorSpec], max_shard_size: int
) -> list[list[str]]:
    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for key in keys:
        size = _tensor_nbytes(specs[key])
        if current and current_size + size > max_shard_size:
            shards.append(current)
            current = []
            current_size = 0
        current.append(key)
        current_size += size
    if current:
        shards.append(current)
    return shards


def _atomic_save_safetensors(tensors: Mapping[str, torch.Tensor], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        save_file(dict(tensors), str(temporary), metadata={"format": "pt"})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(payload: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _is_copyable_hf_asset(relative: Path) -> bool:
    name = relative.name.lower()
    if name in {"model.safetensors.index.json", "pytorch_model.bin.index.json", "merge_manifest.json"}:
        return False
    if relative.suffix.lower() in {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}:
        return False
    if name.startswith(("optimizer", "scheduler", "rng_state", "trainer_state")):
        return False
    return (
        name == "config.json"
        or name == "generation_config.json"
        or _is_tokenizer_asset(relative)
        or relative.suffix.lower() == ".py"
    )


def _copy_hf_assets(reference_dir: Path, output_dir: Path) -> list[str]:
    copied: list[str] = []
    for source in sorted(reference_dir.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(reference_dir)
        if not _is_copyable_hf_asset(relative):
            continue
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(relative.as_posix())
    return copied


def merge_from_manifest(
    manifest_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    chunk_bytes: int = 256 * 1000 * 1000,
    max_shard_size: int = 5 * 1000 * 1000 * 1000,
    normalize_weights: bool = False,
    quiet: bool = False,
) -> Path:
    """Merge the checkpoints in ``manifest_path`` into a new atomic HF directory."""

    if chunk_bytes <= 0:
        raise MergeError("chunk_bytes must be positive")
    if max_shard_size <= 0:
        raise MergeError("max_shard_size must be positive")

    manifest = Path(manifest_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise MergeError(f"output path already exists (refusing to overwrite): {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    entries, user_metadata = _parse_manifest(manifest, normalize_weights)
    sources = [ModelSource(entry) for entry in entries]
    _, config_sha256 = _load_and_validate_config(sources)
    tokenizer_sha256 = _validate_tokenizer_assets(sources)
    keys = _validate_tensor_specs(sources)
    shards = _partition_shards(keys, sources[0].specs, max_shard_size)
    total_size = sum(_tensor_nbytes(sources[0].specs[key]) for key in keys)

    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=str(destination.parent))
    )
    try:
        copied_assets = _copy_hf_assets(sources[0].entry.path, stage)
        weight_map: dict[str, str] = {}
        output_shards: list[dict[str, Any]] = []
        with ExitStack() as stack:
            for source in sources:
                source.attach(stack)
            shard_count = len(shards)
            for shard_index, shard_keys in enumerate(shards, start=1):
                file_name = f"model-{shard_index:05d}-of-{shard_count:05d}.safetensors"
                if not quiet:
                    print(
                        f"[{shard_index}/{shard_count}] merging {len(shard_keys)} tensors -> {file_name}",
                        file=sys.stderr,
                        flush=True,
                    )
                tensors: dict[str, torch.Tensor] = {}
                for key in shard_keys:
                    tensors[key] = _merge_tensor(key, sources, chunk_bytes)
                    weight_map[key] = file_name
                shard_path = stage / file_name
                _atomic_save_safetensors(tensors, shard_path)
                output_shards.append(
                    {
                        "file": file_name,
                        "size": shard_path.stat().st_size,
                        "sha256": _sha256_file(shard_path),
                    }
                )
                del tensors
                gc.collect()

        index_payload = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }
        _atomic_write_json(index_payload, stage / "model.safetensors.index.json")
        merge_manifest = {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_manifest": str(manifest),
            "models": [
                {
                    "name": source.entry.name,
                    "path": str(source.entry.path),
                    "weight": source.entry.weight,
                }
                for source in sources
            ],
            "metadata": user_metadata,
            "validation": {
                "config_sha256": config_sha256,
                "tokenizer_sha256": tokenizer_sha256,
                "tensor_count": len(keys),
                "total_tensor_bytes": total_size,
            },
            "merge": {
                "floating_accumulator_dtype": "float32",
                "chunk_bytes": chunk_bytes,
                "max_shard_size": max_shard_size,
                "non_floating_policy": "require_identical",
            },
            "copied_assets": copied_assets,
            "output_shards": output_shards,
        }
        _atomic_write_json(merge_manifest, stage / "merge_manifest.json")
        os.replace(stage, destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return destination


def _parse_size(value: str) -> int:
    text = value.strip().upper()
    suffixes = {
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "B": 1,
    }
    for suffix, multiplier in suffixes.items():
        if text.endswith(suffix):
            number = text[: -len(suffix)].strip()
            break
    else:
        number = text
        multiplier = 1
    try:
        result = int(float(number) * multiplier)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid byte size: {value!r}") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("size must be positive")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSON model/weight manifest")
    parser.add_argument("--output-dir", required=True, help="new HF output directory")
    parser.add_argument(
        "--chunk-bytes",
        type=_parse_size,
        default=_parse_size("256MB"),
        help="maximum target size per input/FP32 accumulation slice (default: 256MB)",
    )
    parser.add_argument(
        "--max-shard-size",
        type=_parse_size,
        default=_parse_size("5GB"),
        help="maximum target tensor bytes per output shard (default: 5GB)",
    )
    parser.add_argument(
        "--normalize-weights",
        action="store_true",
        help="normalize manifest weights instead of requiring them to sum to one",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = merge_from_manifest(
            args.manifest,
            args.output_dir,
            chunk_bytes=args.chunk_bytes,
            max_shard_size=args.max_shard_size,
            normalize_weights=args.normalize_weights,
            quiet=args.quiet,
        )
    except MergeError as exc:
        print(f"merge failed: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
