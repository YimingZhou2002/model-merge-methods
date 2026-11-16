#!/usr/bin/env bash
set -Eeuo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
SOURCE= PY= APP_REPO= OUTPUT= PORT=18129 RAY_DIR=
while (($#)); do
  case "$1" in
    --source|--python|--appworld-repo|--output-dir|--port|--ray-temp-dir)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --source) SOURCE=$2;; --python) PY=$2;; --appworld-repo) APP_REPO=$2;;
        --output-dir) OUTPUT=$2;; --port) PORT=$2;;
        --ray-temp-dir) RAY_DIR=$2;;
      esac
      shift 2;;
    -h|--help)
      echo 'Usage: start_appworld.sh --source DIR --python APPWORLD_PYTHON --appworld-repo DIR --output-dir DIR [--port 18129] [--ray-temp-dir SHORT_DIR]'
      exit 0;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done
[[ -d "$SOURCE" && -x "$PY" && -d "$APP_REPO/src/appworld" && -n "$OUTPUT" ]] || { echo 'Supply source, AppWorld Python, AppWorld repository and output directory.' >&2; exit 2; }
[[ "$PORT" =~ ^[0-9]+$ ]] && ((PORT > 0 && PORT < 65536)) || { echo 'Invalid port.' >&2; exit 2; }
SOURCE=$(cd "$SOURCE" && pwd -P)
APP_REPO=$(cd "$APP_REPO" && pwd -P)
PY_DIR=$(cd "$(dirname "$PY")" && pwd -P)
PY="$PY_DIR/$(basename "$PY")"
"$PY" "$HERE/prepare_source.py" "$SOURCE" --verify-only
test -d "$SOURCE/env_service/environments/appworld/data" || { echo 'Link the AppWorld data directory first; see README.md.' >&2; exit 2; }
mkdir -p "$OUTPUT"
OUTPUT=$(cd "$OUTPUT" && pwd -P)
export APPWORLD_ROOT="$SOURCE/env_service/environments/appworld"
export PYTHONPATH="$SOURCE:$APP_REPO/src" PATH="$PY_DIR:$PATH"
export APPWORLD_STEP_MEMORY_HEADROOM_MB=4096 ENV_SERVICE_NUM_CPUS=128
RAY_DIR=${RAY_DIR:-$OUTPUT/ray}
mkdir -p "$RAY_DIR"
RAY_DIR=$(cd "$RAY_DIR" && pwd -P)
"$PY" - "$RAY_DIR" <<'PY'
import os, sys
assert len(os.fsencode(sys.argv[1])) <= 40, 'Supply --ray-temp-dir with a short data-disk path (<=40 bytes) for Ray sockets.'
PY
export RAY_TMPDIR="$RAY_DIR" TMPDIR="$OUTPUT/tmp"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset RAY_ADDRESS APPWORLD_ENABLE_RETHINK AE_TARGET_MINI_STEPS
mkdir -p "$RAY_TMPDIR" "$TMPDIR"
cd "$SOURCE"
"$PY" - <<'PY'
from appworld import load_task_ids
tasks = load_task_ids('dev')
assert len(tasks) == 57, ('Expected the evaluated 57-task dev set', len(tasks))
print('AppWorld dev tasks:', len(tasks), flush=True)
PY
exec "$PY" -m env_service.env_service --env appworld --portal 127.0.0.1 --port "$PORT"
