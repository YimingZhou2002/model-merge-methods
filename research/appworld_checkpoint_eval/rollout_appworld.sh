#!/usr/bin/env bash
set -Eeuo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
SOURCE= PY= MODEL= OUTPUT=
SAMPLES=8 TEMPERATURE=0.9 ENV_URL=http://127.0.0.1:18129
DEVICES=0,1,2,3,4,5,6,7 TP=1 GPU_MEMORY=0.45 DRY_RUN=0 RAY_DIR=
DTYPE= SEED= ATTENTION_BACKEND=FLASH_ATTN
MAX_BATCHED_TOKENS=65536

usage() {
  cat <<'EOF'
Usage: rollout_appworld.sh --source DIR --python PATH --model-path DIR
                           --output-dir NEW_DIR [options]

AppWorld dev evaluation (57 tasks), preserving raw token history and CoT.
Options:
  --samples N               Samples per task (default: 8)
  --temperature T           Sampling temperature (default: 0.9)
  --env-url URL             Running AppWorld service (default: localhost:18129)
  --devices LIST            CUDA device list (default: 0,1,2,3,4,5,6,7)
  --tp N                    Tensor parallel size (default: 1)
  --gpu-memory-utilization F vLLM GPU memory fraction (default: 0.45)
  --dtype DTYPE             Explicit rollout dtype (float32/bfloat16/float16)
  --seed N                  Explicit vLLM rollout seed
  --attention-backend NAME  vLLM backend (default FLASH_ATTN; FP32 needs TRITON_ATTN)
  --max-num-batched-tokens N Scheduler token budget; long prompts use chunked prefill
  --ray-temp-dir DIR        Short Ray directory on data disk (use <=40 bytes)
  --dry-run                 Check source, model and resolved config without GPU rollout
EOF
}

while (($#)); do
  case "$1" in
    --source|--python|--model-path|--output-dir|--samples|--temperature|--env-url|--devices|--tp|--gpu-memory-utilization|--ray-temp-dir|--dtype|--seed|--attention-backend|--max-num-batched-tokens)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --source) SOURCE=$2;; --python) PY=$2;; --model-path) MODEL=$2;;
        --output-dir) OUTPUT=$2;; --samples) SAMPLES=$2;; --temperature) TEMPERATURE=$2;;
        --env-url) ENV_URL=$2;; --devices) DEVICES=$2;; --tp) TP=$2;;
        --gpu-memory-utilization) GPU_MEMORY=$2;;
        --dtype) DTYPE=$2;; --seed) SEED=$2;;
        --attention-backend) ATTENTION_BACKEND=$2;;
        --max-num-batched-tokens) MAX_BATCHED_TOKENS=$2;;
        --ray-temp-dir) RAY_DIR=$2;;
      esac
      shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2;;
  esac
done
[[ -n "$SOURCE" && -n "$PY" && -n "$MODEL" && -n "$OUTPUT" ]] || { usage >&2; exit 2; }
[[ -x "$PY" && -d "$SOURCE" && -d "$MODEL" ]] || { echo 'Check --source, --python and --model-path.' >&2; exit 2; }
[[ "$SAMPLES" =~ ^[0-9]+$ ]] && ((SAMPLES >= 1 && SAMPLES <= 32)) || { echo 'Samples must be 1..32.' >&2; exit 2; }
[[ "$TP" =~ ^[0-9]+$ ]] && ((TP >= 1)) || { echo 'TP must be positive.' >&2; exit 2; }
[[ "$DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Devices must be a comma-separated list of GPU indices.' >&2; exit 2; }
[[ -z "$DTYPE" || "$DTYPE" =~ ^(float32|bfloat16|float16)$ ]] || { echo 'Invalid rollout dtype.' >&2; exit 2; }
[[ -z "$SEED" || "$SEED" =~ ^[0-9]+$ ]] || { echo 'Seed must be nonnegative.' >&2; exit 2; }
[[ "$ATTENTION_BACKEND" =~ ^[A-Z_]+$ ]] || { echo 'Invalid attention backend.' >&2; exit 2; }
[[ "$MAX_BATCHED_TOKENS" =~ ^[0-9]+$ ]] && ((MAX_BATCHED_TOKENS >= 128)) || { echo 'Invalid token batch budget.' >&2; exit 2; }
IFS=, read -r -a GPU_LIST <<< "$DEVICES"
GPUS=${#GPU_LIST[@]}
((GPUS % TP == 0)) || { echo 'Number of GPUs must be divisible by TP.' >&2; exit 2; }
SOURCE=$(cd "$SOURCE" && pwd -P)
MODEL=$(cd "$MODEL" && pwd -P)
PY_DIR=$(cd "$(dirname "$PY")" && pwd -P)
PY="$PY_DIR/$(basename "$PY")"

"$PY" "$HERE/prepare_source.py" "$SOURCE" --verify-only
"$PY" - "$MODEL" "$TEMPERATURE" "$GPU_MEMORY" <<'PY'
import json, math, pathlib, sys
model = pathlib.Path(sys.argv[1])
temperature, memory = map(float, sys.argv[2:])
assert math.isfinite(temperature) and temperature > 0, 'Temperature must be positive; do_sample is enabled.'
assert math.isfinite(memory) and 0 < memory < 1, 'GPU memory fraction must be between 0 and 1.'
for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
    assert (model/name).is_file() and (model/name).stat().st_size > 0, name
index = model/'model.safetensors.index.json'
weights = set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
assert weights, 'No weight files in index'
for name in weights:
    assert (model/name).is_file() and (model/name).stat().st_size > 0, 'Missing weight shard: '+name
ready = model/'EXPORT_READY.json'
if ready.exists():
    for item in json.loads(ready.read_text())['weights']:
        assert (model/item['file']).stat().st_size == item['bytes'] > 0, item
print('Model files verified:', model, flush=True)
PY

# Never mix two evaluations or their Ray/cache state in the same directory.
mkdir -p "$(dirname "$OUTPUT")"
mkdir "$OUTPUT"
OUTPUT=$(cd "$OUTPUT" && pwd -P)
NAME="appworld_dev_n${SAMPLES}_$(date +%Y%m%d_%H%M%S)_$$"
VAL_DIR="$OUTPUT/validation_log"
RAY_DIR=${RAY_DIR:-$OUTPUT/ray}
mkdir -p "$VAL_DIR" "$RAY_DIR" "$OUTPUT/tmp" "$OUTPUT/triton"
RAY_DIR=$(cd "$RAY_DIR" && pwd -P)
export PYTHONPATH="$SOURCE" PATH="$PY_DIR:$PATH"
export CUDA_VISIBLE_DEVICES="$DEVICES" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline DASHSCOPE_API_KEY=EMPTY VLLM_USE_FLASHINFER_SAMPLER=0
export RAY_TMPDIR="$RAY_DIR" TMPDIR="$OUTPUT/tmp" TRITON_CACHE_DIR="$OUTPUT/triton"
unset RAY_ADDRESS DEBUG_ARG APPWORLD_ENABLE_RETHINK AE_TARGET_MINI_STEPS
cd "$SOURCE"

# Use the same AgentEvolver validation path as the 300-step original-base runs.
CMD=("$PY" -m agentevolver.main_ppo --config-path="$SOURCE/config" --config-name=script_config
  "env_service.env_url=$ENV_URL" env_service.env_type=appworld
  attribution_driven_credit_assignment.enable=false algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=false
  data.train_batch_size=32 data.seed=2027 data.max_prompt_length=4000 data.max_response_length=21580
  data.return_raw_chat=true data.truncation=error data.train_files=null data.val_files=null
  "actor_rollout_ref.model.path=$MODEL"
  actor_rollout_ref.model.use_remove_padding=true actor_rollout_ref.model.enable_gradient_checkpointing=true
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_epochs=1 actor_rollout_ref.actor.ppo_mini_batch_size=32
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 actor_rollout_ref.actor.use_dynamic_bsz=true
  actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.fsdp_config.param_offload=false actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=25580
  actor_rollout_ref.rollout.use_qwen3=false actor_rollout_ref.rollout.enable_request_id=false
  actor_rollout_ref.rollout.context_template=linear_tokens actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.prompt_length=20480 actor_rollout_ref.rollout.response_length=4096
  actor_rollout_ref.rollout.max_model_len=25580 "actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_BATCHED_TOKENS"
  actor_rollout_ref.rollout.max_num_seqs=128 actor_rollout_ref.rollout.enable_prefix_caching=true
  "actor_rollout_ref.rollout.temperature=$TEMPERATURE" actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1
  "actor_rollout_ref.rollout.n=$SAMPLES" "actor_rollout_ref.rollout.tensor_model_parallel_size=$TP"
  "actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY" actor_rollout_ref.rollout.max_env_worker=64
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768
  actor_rollout_ref.ref.fsdp_config.param_offload=true
  "++actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=$ATTENTION_BACKEND"
  ++actor_rollout_ref.rollout.engine_kwargs.vllm.logprobs_mode=processed_logprobs
  ++actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2
  "actor_rollout_ref.rollout.val_kwargs.n=$SAMPLES" actor_rollout_ref.rollout.val_kwargs.do_sample=true
  "actor_rollout_ref.rollout.val_kwargs.temperature=$TEMPERATURE"
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 actor_rollout_ref.rollout.val_kwargs.top_k=-1
  actor_rollout_ref.rollout.multi_turn.max_steps=30 task_manager.n=0 task_manager.mixture.synthetic_data_ratio=0.0
  task_manager.mixture.use_original_tasks=true exp_manager.train_rollout_mode=woexp exp_manager.val_rollout_mode=woexp
  exp_manager.init_exp_before_training=false exp_manager.reme.enable_summarizer=false exp_manager.reme.enable_context_generator=false
  ray_kwargs.ray_init.num_cpus=64 "trainer.n_gpus_per_node=$GPUS" trainer.nnodes=1
  'trainer.logger=[console]' trainer.project_name=appworld_qwen3_original_cot "trainer.experiment_name=$NAME"
  trainer.total_training_steps=300 trainer.total_epochs=200 trainer.save_freq=10 trainer.test_freq=10
  trainer.val_only=true trainer.val_before_train=true trainer.resume_mode=disable trainer.max_actor_ckpt_to_keep=2
  "trainer.default_local_dir=$OUTPUT/checkpoints"
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]'
  'actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]'
  "+trainer.export_model_dir=$OUTPUT/exports" "trainer.validation_data_dir=$VAL_DIR"
  "trainer.rollout_data_dir=$OUTPUT/rollout_log"
)
[[ -z "$DTYPE" ]] || CMD+=("actor_rollout_ref.rollout.dtype=$DTYPE")
[[ -z "$SEED" ]] || CMD+=("actor_rollout_ref.rollout.seed=$SEED")

if ! "${CMD[@]}" --cfg job > "$OUTPUT/config_stdout.txt" 2> "$OUTPUT/config_check.log"; then
  echo "Config check failed; see $OUTPUT/config_check.log" >&2; exit 1
fi
"$PY" "$HERE/validate_and_summarize.py" config "$OUTPUT" "$MODEL" "$SAMPLES" "$TEMPERATURE" "$GPUS" "$TP" "$ENV_URL"
echo "Run directory: $OUTPUT"
if ((DRY_RUN)); then
  echo "Dry run passed. Resolved config: $OUTPUT/resolved_config.yaml"
  exit 0
fi

curl -fsS --max-time 5 "${ENV_URL%/}/docs" > /dev/null || { echo 'AppWorld service is unavailable; see README.md.' >&2; exit 1; }
"$PY" - "$GPUS" "$RAY_DIR" <<'PY'
import os, sys, torch
assert len(os.fsencode(sys.argv[2])) <= 40, 'Ray socket path is too long. Supply --ray-temp-dir with a short data-disk path.'
assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.cuda.device_count() == int(sys.argv[1]), 'Visible GPU count differs from --devices'
PY
printf 'Generating AppWorld dev rollouts; log: %s\n' "$OUTPUT/rollout.log"
if ! "${CMD[@]}" > "$OUTPUT/rollout.log" 2>&1; then
  tail -n 30 "$OUTPUT/rollout.log" >&2; exit 1
fi
"$PY" "$HERE/validate_and_summarize.py" summary "$OUTPUT" "$MODEL" "$SAMPLES" "$TEMPERATURE"
