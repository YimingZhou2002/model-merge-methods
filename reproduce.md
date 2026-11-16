# Reproduce MMR, TSG-TV, and BCM on AppWorld Qwen3-14B

This document records the exact 14B development setup and the commands for
rebuilding the merges and running the evaluation. It distinguishes the frozen
historical result from a fresh replication: the historical result used a
development set already examined during method selection, so it is exploratory
evidence rather than an untouched test result.

## 1. Workspace and inputs

The scripts expect this repository to be a child of a workspace containing the
data and evaluation runtime, for example:

```text
workspace/
├── model-pool/checkpoint-{230,240,250,260,270,280,290,300}/
├── base-model/Qwen3-14B/                 # used for TIES
├── runtime/source/                       # pinned AgentEvolver source
├── runtime/appworld/                     # AppWorld source
├── runtime/appworld_data/                # AppWorld DB/data snapshot
├── envs/eval/                             # PyTorch/vLLM evaluation environment
├── envs/appworld/                         # AppWorld environment
├── appworld-service.log
└── model-merge-methods/                  # this repository
```

The current workspace at `/mnt/infini/zym-tmp/merge` already has these inputs.
Never add model weights or AppWorld task/database data to this Git repository.
Verify the downloaded checkpoint pool against the manifest before a new run.

The required evaluation protocol is 57 AppWorld dev tasks × 8 samples, FP32,
temperature 0.9, top-p 1, top-k -1, TP=1 over eight GPUs, context 25,580,
prompt limit 20,480, response limit 4,096, and at most 30 environment actions.
Use `TRITON_ATTN` for FP32. The original MMR/TSG study used scheduler budget
8192 and GPU memory utilization 0.85. BCM round 1 had a common prefill-memory
amendment applied to every cell: scheduler budget 4096 and GPU memory
utilization 0.80. Keep those protocols distinct when reporting the results.

## 2. What each method computes

### MMR selection and MMR-TIES

For each checkpoint, the calibration vector contains its per-task success rate
over the 57 dev tasks. MMR greedily selects five sources using

```text
score(i) = lambda * quality(i)
           - (1 - lambda) * max_cosine(i, selected_sources)
```

with `k=5`, `lambda=0.7`, and softmax temperature `tau=0.02`. The selection
starts from the highest-quality checkpoint. The selected source weights are a
softmax over their mean task success rates. MMR-TIES applies weighted TIES to
those sources relative to the pretrained Qwen3-14B base (`density=1`,
`normalize=true`, `lambda=1`). The exact selection, weights, source identities,
and evaluation settings are in [`results/tsg-mmr-recipes.json`](results/tsg-mmr-recipes.json)
and [`results/tsg-mmr-inputs.json`](results/tsg-mmr-inputs.json).

### TSG-TV

Checkpoint 300 is the anchor. Let `L` be the same MMR-bank weighted linear
average. For every parameter coordinate, compute chronological total variation
across checkpoints 230–300 and the end-to-end displacement:

```text
travel = sum_t abs(theta[t+1] - theta[t])
gate   = clamp(1 - abs(theta[300] - theta[230]) / travel, 0, 1)
TSG-TV = theta[300] + gate * (L - theta[300])
```

Coordinates with a flat trajectory receive gate 0. The implementation is in
[`research/checkpoint_merge_14b/merge.py`](research/checkpoint_merge_14b/merge.py),
function `build_primary`.

### BCM (bootstrap consensus merge)

BCM fits a KL-regularized log-utility portfolio over checkpoint success
probabilities. It refits the portfolio on task-family bootstrap samples and
uses the mean weights as the merge target. The bootstrap covariance estimates
coefficient instability; for each coordinate it attenuates the anchor-relative
mean update `u` using

```text
gate = u^2 / (u^2 + beta * update_variance)
BCM  = theta[300] + gate * u
```

Round 1 tested `beta=0.5` and `beta=1.0`, plus the exact BCM-mean linear target
as a control. The frozen calibration/weights and the execution amendment are
in [`results/bcm-r1-recipes.json`](results/bcm-r1-recipes.json) and
[`results/bcm-r1-execution-amendment.json`](results/bcm-r1-execution-amendment.json).
The formula and calibration code are in
[`research/temporal_merge_search14b/methods.py`](research/temporal_merge_search14b/methods.py).

## 3. Test the merge code

From this repository root, use the workspace's evaluation Python environment:

```bash
../envs/eval/bin/python -m unittest discover -s tests -p 'test_*merge*.py'
```

The tests cover MMR selection parity, TSG-TV tensor formulas, BCM calibration
invariants, independent numerical references, and safetensors exports. They do
not require GPUs or launch AppWorld.

## 4. Reproduce MMR-TIES and default TSG-TV

From the repository root, set short IPC/cache paths and run the complete frozen
study. It evaluates all eight source checkpoints first, freezes the MMR bank,
exports the four merge/control models, and evaluates them sequentially. The
checked-in config points at the preserved completed run, whose receipts are
intentionally verified/reused by the controller rather than sampled again.

```bash
WORKSPACE="$(dirname "$PWD")"
NEW_RUN="runs/qwen3-14b-dev8-v2-repro"
python - "$PWD/research/checkpoint_merge_14b/config.json" "/tmp/tsg-mmr-repro.json" "$NEW_RUN" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
data["run_root"] = sys.argv[3]
data["study"] = "qwen3-14b-mmr-tsg-dev8-repro"
data["ray_namespace"] = 1400
json.dump(data, open(sys.argv[2], "w"), indent=2)
PY

mkdir -p "$WORKSPACE/r/e15" "$WORKSPACE/$NEW_RUN"
ln -s "$WORKSPACE/r/e15" "$WORKSPACE/$NEW_RUN/evaluations"
export VLLM_RPC_BASE_PATH="$WORKSPACE/r/ipc15"
export VLLM_CACHE_ROOT="$WORKSPACE/runtime/cache/vllm-14b-v2-repro"

../envs/eval/bin/python -m research.checkpoint_merge_14b.study preflight \
  --config /tmp/tsg-mmr-repro.json --workspace "$WORKSPACE"
../envs/eval/bin/python -m research.checkpoint_merge_14b.study run \
  --config /tmp/tsg-mmr-repro.json --workspace "$WORKSPACE"
```

The checked-in config is [`research/checkpoint_merge_14b/config.json`](research/checkpoint_merge_14b/config.json).
Preserve all settings except the unique run directory and Ray namespace to
match the original protocol. The short physical `r/e15` output and symlink
avoid the IPC path-length failure observed in the original study. A fresh run
costs 12 evaluation cells; do not reuse a completed output directory.

After all 12 cells finish, verify every raw score and the service-instance
accounting:

```bash
../envs/eval/bin/python -m research.checkpoint_merge_14b.audit_artifacts \
  --run-root "$WORKSPACE/$NEW_RUN" \
  --service-log "$(dirname "$PWD")/appworld-service.log" \
  --include-merges --require-complete
```

The full run uses 5,472 trajectories (12 models × 456 rollouts). It takes a
substantial multi-GPU allocation. A successful dry-run alone is not an
evaluation result.

## 5. Reproduce BCM round 1

The first round is calibrated from the already completed eight source-model
dev@8 result vectors. It evaluates a concurrent TSG-TV reference, BCM linear,
QIS/DRR controls, and BCM β=0.5/1.0. The total is 3,648 trajectories.

The runner supports separate workspace and output paths. Start the local
AppWorld service and ensure the eight GPUs are idle, then:

```bash
export MODEL_MERGE_WORKSPACE="$(dirname "$PWD")"
export MMR_TSG_RUN_ROOT="$MODEL_MERGE_WORKSPACE/runs/qwen3-14b-dev8-v2"
export BCM_RUN_ROOT="$MODEL_MERGE_WORKSPACE/runs/temporal-search14b-r1-repro"
export BCM_EVAL_PHYSICAL_DIR="$MODEL_MERGE_WORKSPACE/r/s15"
export VLLM_RPC_BASE_PATH="$MODEL_MERGE_WORKSPACE/r/ipc15"
export VLLM_CACHE_ROOT="$MODEL_MERGE_WORKSPACE/runtime/cache/vllm-bcm-r1-repro"

mkdir -p "$(dirname "$BCM_RUN_ROOT")"

../envs/eval/bin/python -m research.temporal_merge_search14b.run prepare
../envs/eval/bin/python -m research.temporal_merge_search14b.amend_execution
../envs/eval/bin/python -m research.temporal_merge_search14b.run export
../envs/eval/bin/python -m research.temporal_merge_search14b.audit_exports \
  --root "$BCM_RUN_ROOT"
../envs/eval/bin/python -m research.temporal_merge_search14b.recover_execution
../envs/eval/bin/python -m research.temporal_merge_search14b.audit_results \
  --root "$BCM_RUN_ROOT" --require-complete
```

`prepare` freezes the source checkpoint identities, calibration inputs, code
hashes, formulas, and run settings before evaluation. `export` builds all seven
round-1 candidates and controls; `audit_exports` verifies their shard hashes
and formula probes; `recover_execution` runs the queue sequentially and records
the amended settings. Existing completed output cells are never overwritten.
Use a new output root and physical directory for every fresh run.

To reproduce the equal-norm BCM-relative result, follow the frozen protocol in
[`results/bcm-r2-evaluation-plan.json`](results/bcm-r2-evaluation-plan.json).
Round 2 requires all round-1 audits and exports, then adds a concurrent TSG-TV,
best-source and equal-norm controls. TV scale checks are separately documented
in `research/temporal_merge_search14b/TV_SCALE_PLAN.md`.

## 6. Recorded results and interpretation

| Experiment | Model | Mean@8 | Pass@8 | Interpretation |
|---|---|---:|---:|---|
| Original 14B dev@8 | TSG-TV | 78.51% | 96.49% | Highest point estimate in the original 12-cell study; family interval vs best source includes zero. |
| Original 14B dev@8 | MMR-TIES | 76.97% | 96.49% | Does not establish a reliable gain over the best source or simple controls. |
| BCM round 1 | BCM β=1.0 | 78.95% | 92.98% | Highest BCM round-1 point estimate; BCM-vs-linear mechanism contrast is inconclusive. |
| BCM round 1 | BCM β=0.5 | 78.73% | 94.74% | Exploratory reused-dev result. |
| Equal-norm round 2 | BCM-relative-NM | 77.85% | 94.74% | Paired family interval vs that round's TSG-TV includes zero. |
| TV scale development | TSG-TV-double | 79.17% | 94.74% | Scale chosen on reused dev; family interval vs default begins at zero and task interval includes zero. |

The first two studies use different seeds and execution settings. These numbers
are not a single matched comparison and do not establish that BCM or a TV scale
is a confirmed SOTA. The dev set was reused for source selection, recipe search,
and scale selection. Use a prospectively frozen, untouched confirmation split
and matched controls before making a generalization claim.

Full result tables and frozen audit inputs are under [`results/`](results/).
