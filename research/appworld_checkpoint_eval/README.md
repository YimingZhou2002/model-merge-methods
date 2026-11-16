# Qwen3 original-base checkpoint AppWorld rollout

这是 2026-09/10 从官方原始 Qwen3-8B / Qwen3-14B 权重开始、分别训练 300 step 的推理入口。使用与训练 validation 相同的 AgentEvolver 多轮 rollout，保留 CoT 和原始 token 历史。`--model-path` 接受 Hugging Face 格式的任意 checkpoint，包括 8B step 220/300 和 14B 分片导出。

本目录的配置针对这两次 300-step original-base 实验。根目录文档中的 7 月 no-rethink SFT → 200-step GRPO / model-merge 实验有自己的运行入口和环境。请按本文件准备本次推理环境。

## 固定的评测协议

| 项目 | 默认值 |
|---|---|
| split / 任务数 | AppWorld `dev` / 57 |
| 每题采样数 | 8，共 456 条轨迹 |
| temperature / top_p / top_k | 0.9 / 1.0 / -1 |
| do_sample | true |
| 最大环境动作数 | 30 |
| context / prompt / 每次生成长度 | 25580 / 20480 / 4096 tokens |
| context_template | `linear_tokens` |
| use_qwen3 | false：关闭旧代码注入的禁用思考提示；这里并不表示关闭 Qwen3 模型 |
| experience / synthetic tasks | `woexp` / 0 |
| GPU / TP / vLLM memory fraction | 8 / 1 / 0.45 |
| sampler | `VLLM_USE_FLASHINFER_SAMPLER=0` |

`mean@8` 是 57 题 × 8 次 rollout 的 score 均值；每题任一次成功的 `pass@8` 是另一项指标。本脚本检查 57 组任务各有 8 条结果后才输出汇总。原实验 JSONL 未保存 `task_id`，因此按初始 `input` 的 SHA-256 分组；已核对历史 57 题、456 条结果恰好形成 57 组、每组 8 条。若输入文件含 `task_id`，则优先按 ID 分组。采样评测存在随机波动，不保证逐条重现历史轨迹。

## 环境与源码

评测 Python 环境记录：Python 3.12.13、PyTorch 2.13.0+cu130、vLLM 0.27.1、VERL 0.9.0、Transformers 5.12.1、Ray 2.57.0、Triton 3.7.1、FlashAttention 2.8.3.post1。版本快照见 `runtime_pip_freeze.txt`，其中 pip 对 torch 的版本记录省略了 `+cu130`，机器上的本地可编辑安装路径也未写入。这是依赖快照，安装时仍需选择与驱动、GPU 架构匹配的 CUDA wheel。

AppWorld 使用独立 CPU 环境：Python 3.11.16、AppWorld 0.2.0.dev0；完整包版本见 `appworld_packages.json`。运行时还需要已使用的 AppWorld 源码目录及数据，两个 Python 环境请分别提供。若节点已有这套环境，直接复用。

源码基线为 AgentEvolver `a5a8db8689d6493b028107dcb4c27415441581dc`。`runtime.patch` 包含本次训练和复测使用的 CoT/token 历史修复、VERL 0.9/vLLM 兼容、AppWorld 内存管理等改动。准备脚本在独立目录应用补丁，并检查 `runtime_manifest.json` 中 127 个文件的 SHA-256。补丁中的历史凭证字段已归一化；应用后的文件与实际评测文件一致。

所有源码、模型、缓存和结果路径都指定在数据盘。例如：

```bash
export DATA_ROOT=/nvme1/your_user/appworld_eval
export AE_SOURCE="$DATA_ROOT/source"
export EVAL_PY=/path/to/existing/rethink/bin/python
export APPWORLD_PY=/path/to/existing/appworld/bin/python
export APPWORLD_REPO=/path/on/data_disk/appworld_repo
export TOOLS="$PWD/research/appworld_checkpoint_eval"

# 已有同版 patched checkout 时，使用 --verify-only 即可。
python3 "$TOOLS/prepare_source.py" "$AE_SOURCE"

ln -s "$APPWORLD_REPO/data" \
  "$AE_SOURCE/env_service/environments/appworld/data"
```

准备脚本不安装 Python 环境、模型或 AppWorld 数据。完整的 AppWorld 源码/数据快照未随仓库发布；仅任务数相同不足以证明环境版本、prompt 和数据库一致。应复用原实验快照，或先核对自己的 AppWorld 环境。

## 下载 8B checkpoint

30 份 8B 权重在 `JunjieMeng/agentevolver-appworld-qwen3-8b-base-cot`。只下载需要的子目录，例如 final checkpoint：

```bash
"$EVAL_PY" - "$DATA_ROOT/hf/8b" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="JunjieMeng/agentevolver-appworld-qwen3-8b-base-cot",
    revision="9e48e4dc9a486c6b5a165e42cf14a663ba1948bd",
    allow_patterns=["checkpoint-300/*"],
    local_dir=sys.argv[1],
)
PY
export MODEL_PATH="$DATA_ROOT/hf/8b/checkpoint-300"
```

下载 step 220 时，把两处 `checkpoint-300` 改为 `checkpoint-220`。14B 使用已导出的本地 Hugging Face 模型目录。

8B step 300 的 `model.safetensors`：32,762,987,816 bytes，SHA-256 `a35dc0f8f63d879df1903f5fb1401bb032ceb75bf3e4c4c0a0108d1709b353ae`。可以在数据盘运行 `sha256sum "$MODEL_PATH/model.safetensors"` 核对。

## 启动 AppWorld 并评测

如果已有同版本 AppWorld 服务，直接使用其 URL。否则在一个独立终端启动：

```bash
bash "$TOOLS/start_appworld.sh" \
  --source "$AE_SOURCE" --python "$APPWORLD_PY" \
  --appworld-repo "$APPWORLD_REPO" \
  --output-dir "$DATA_ROOT/service" --port 18129 \
  --ray-temp-dir /nvme1/your_user/ae_svc_r
```

另一个终端先检查配置：

```bash
bash "$TOOLS/rollout_appworld.sh" \
  --source "$AE_SOURCE" --python "$EVAL_PY" \
  --model-path "$MODEL_PATH" \
  --output-dir "$DATA_ROOT/results/config_check" --dry-run
```

正式推理使用新的结果目录：

```bash
bash "$TOOLS/rollout_appworld.sh" \
  --source "$AE_SOURCE" --python "$EVAL_PY" \
  --model-path "$MODEL_PATH" \
  --output-dir "$DATA_ROOT/results/8b_step300_n8" \
  --env-url http://127.0.0.1:18129 \
  --ray-temp-dir /nvme1/your_user/ae_eval_r
```

产物：`resolved_config.yaml`、`rollout.log`、`validation_log/0.jsonl` 和 `rollout_summary.json`。已有结果目录会报错，避免复跑覆盖和混入旧轨迹。每条 JSONL 带 `score`、初始输入及生成轨迹；summary 还记录使用的任务分组方式。

Ray 的 Unix socket 有路径长度限制。`--ray-temp-dir` 必须使用数据盘上的短绝对路径（脚本检查不超过 40 bytes），服务和评测使用不同目录。结果目录可以保留较长的实验名称。

`--samples`、`--temperature`、`--devices`、`--tp`、`--gpu-memory-utilization` 可调整，最终值都会写入 resolved config。历史配置在 8 卡上验证过；其他卡数/TP 组合需要自行验证显存与吞吐。A800 的复现结果尚未由本目录验证。CUDA 工具链路径按本机环境设置，例如需要时显式设置 `LD_LIBRARY_PATH` 和 `TRITON_PTXAS_PATH`；脚本不会假设节点 117 的安装路径。

H20 复测采用 CUDA 13 用户态兼容库及容器提供 glibc；宿主驱动 570.133.20、宿主 CUDA 12.8，不能仅凭宿主 CUDA 版本安装另一套推理包。已记录的 H20 两次 8B step 300 `mean@8` 分别为 0.635965 和 0.616228（第二次从 Hugging Face 清缓存重新下载）。8B step 300 原训练 validation 为 0.653508；这些值是历史观测，不能视为新机器的保证分数。

## 发布验证

发布前在 117 现有环境、实际 8B / 14B step 300 权重上运行了新入口的 `--dry-run`，两次均通过。独立源码恢复及 127 文件 hash 校验通过，29 项原始 CoT 回归检查全部通过、无跳过；直接复用 AppWorld 环境内的 pytest，没有安装新环境。

结果汇总读取历史 8B step 300 的 456 条真实轨迹，核对出 57 个输入组、每组 8 条、`mean@8=0.620614`；还检查了总条数相同但每题采样数不齐的文件会被拒绝。本次发布没有重新运行整套 GPU 评测。
