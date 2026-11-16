"""Register the shared low-memory execution settings for a fresh BCM round 1."""
from __future__ import annotations

import json
import os
from pathlib import Path

from research.model_merge import weighted_merge as wm

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORK = Path(os.environ.get("MODEL_MERGE_WORKSPACE", REPO.parent)).resolve()
ROOT = Path(os.environ.get("BCM_RUN_ROOT", WORK / "runs/temporal-search14b-r1")).resolve()


def main() -> None:
    frozen_path = ROOT / "inputs.json"
    frozen = json.loads(frozen_path.read_text())
    target = ROOT / "execution-amendment.json"
    amendment = {
        "applies_to": "All eight BCM round-1 cells including concurrent TSG-TV",
        "completed_valid_cells_before_amendment": 0,
        "execution_config": dict(
            frozen["config"],
            max_num_batched_tokens=4096,
            gpu_memory_utilization=0.80,
            ray_namespace=310,
        ),
        "original_inputs_sha256": wm._sha256_file(frozen_path),
        "recovery_runner_sha256": wm._sha256_file(HERE / "recover_execution.py"),
        "reason": "Use a common scheduler/memory amendment after the original prefill OOM; keep all model recipes, tasks, samples, dtype, temperature, and seed unchanged.",
    }
    encoded = json.dumps(amendment, indent=2, sort_keys=True) + "\n"
    if target.exists():
        if target.read_text() != encoded:
            raise FileExistsError(f"Existing amendment differs; refusing to replace {target}")
    else:
        with target.open("x") as handle:
            handle.write(encoded)
    print(target)


if __name__ == "__main__":
    main()
