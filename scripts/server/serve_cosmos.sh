#!/usr/bin/env bash
# Start the Cosmos3 Reasoner locally (PLAN section 12 step 4; pip-installed vLLM, no docker 2026-10-02).
# Runs in a separate systemd user unit nebius-cosmos-<port>: when oomd kills a process only this unit dies, not the terminal and other sessions.
# CUDA_HOME points to 13.1: FlashInfer JIT needs nvcc >= 12.9 to recognize sm_120 (/usr/local/cuda is 12.8).
# Holds flock /tmp/nebius_gpu.lock, to avoid running a second heavy GPU task at the same time.
#
#   scripts/server/serve_cosmos.sh                     # Cosmos3-Nano → 8001, GPU 2
#   MODEL=nvidia/Cosmos3-Edge PORT=8002 scripts/server/serve_cosmos.sh
#   VENV=.venv-vllm HF_OVERRIDES='{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}' \
#     scripts/server/serve_cosmos.sh                   # the Nano README setup: vllm 0.21 + vllm-cosmos3 plugin
# vLLM 0.30.0 (.venv-vllm30) natively supports Cosmos3EdgeForConditionalGeneration and Cosmos3ForConditionalGeneration,
# no plugin needed; vllm 0.21 + plugin only recognizes Nano's cosmos3_omni and cannot start Edge (cosmos3_edge).
#   systemctl --user stop nebius-cosmos-8001           # stop
#   journalctl --user -u nebius-cosmos-8001 -f         # follow the log (also written to runs/cosmos/)
#   FOREGROUND=1 scripts/server/serve_cosmos.sh        # inside a cluster job (the cluster PBS, no systemd user units)
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL=${MODEL:-nvidia/Cosmos3-Nano}
VENV=${VENV:-.venv-vllm30}
HF_OVERRIDES=${HF_OVERRIDES:-}
PORT=${PORT:-8001}
GPU_IN=${GPU:-}
GPU=${GPU:-2}                    # GPU 0/1 are used by other tasks; GPU 2 drives the monitor, but it is the only one idle
MEM_UTIL=${MEM_UTIL:-0.5}        # use only half the card (about 48 GB), leaving headroom for other tasks on the machine
MAX_LEN=${MAX_LEN:-32768}        # a single verifier image + prompt does not need the 262k context; shorten it to reduce the KV cache
TP=${TP:-1}                      # number of GPUs for tensor parallelism: a single A100-40G on an HPC cluster cannot hold the Nano weights + KV cache, so use TP=2
EXTRA_ARGS=${EXTRA_ARGS:-}       # extra vllm serve arguments (for troubleshooting, e.g. --enforce-eager)
HF_HUB_CACHE=${HF_HUB_CACHE:-./store/hf/hub}
UNIT=nebius-cosmos-$PORT
# CUDA with nvcc >= 12.9 (FlashInfer JIT). Do not read CUDA_HOME: in the dev server's login environment CUDA_HOME=12.8; on the cluster set COSMOS_CUDA_HOME explicitly
CUDA_HOME=${COSMOS_CUDA_HOME:-/usr/local/cuda-13.1}
LOG=runs/cosmos/${UNIT}-$(date +%Y%m%d-%H%M%S).log

# COSMOS_SIF: the driver 570 on an HPC cluster is below the 580 required by the cu130 wheels, so use the vLLM 0.30.0 apptainer image + the CUDA forward-compatibility libraries inside the image
# Only used with FOREGROUND=1.
if [ -n "${COSMOS_SIF:-}" ]; then
  [ -f "$COSMOS_SIF" ] || { echo "image $COSMOS_SIF is missing"; exit 1; }
  export APPTAINERENV_LD_LIBRARY_PATH=/usr/local/cuda/compat PYTHONNOUSERSITE=1
  VLLM="apptainer exec --nv -B "${COSMOS_BIND:-$PWD}" $COSMOS_SIF vllm"
else
  [ -x "$VENV/bin/vllm" ] || { echo "$VENV is missing (see the install commands in the README)"; exit 1; }
  VLLM=$VENV/bin/vllm
fi
mkdir -p runs/cosmos

wait_ready() {   # $1 = command that checks the process is still alive
  for _ in $(seq 1 900); do
    if curl -fsS --noproxy '*' "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      # check that it is really the model we started (someone else's vLLM on the same node held the port, /health passed but the model was wrong, and all checks fell back to classical vision)
      if curl -fsS --noproxy '*' "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q "\"$MODEL\""; then
        echo "ready: http://127.0.0.1:$PORT/v1 ($MODEL)"; return 0
      fi
      echo "the service on port $PORT is not $MODEL: $(curl -fsS --noproxy '*' "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | head -c 300)"; return 1
    fi
    eval "$1" || { echo "vLLM has exited, last log lines:"; tail -30 "$LOG"; return 1; }
    sleep 2
  done
  echo "not ready within 30 minutes, see $LOG"; return 1
}

# FOREGROUND=1: inside a cluster job (the cluster PBS) there are no systemd user units. Start vLLM in the background directly in the job, return once it is ready, and exit with the job when it ends.
# The card is assigned by the scheduler: CUDA_VISIBLE_DEVICES is only changed when GPU=<card number> is given explicitly.
if [ "${FOREGROUND:-0}" = 1 ]; then
  if curl -fsS --noproxy '*' "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "port $PORT is already in use (another service on the same node): change it with PORT=... and set CP_BASE_URL_LOCAL_VLLM=http://127.0.0.1:<port>/v1/"; exit 1
  fi
  [ -n "$GPU_IN" ] && export CUDA_VISIBLE_DEVICES="$GPU_IN"
  export HF_HUB_CACHE HF_HUB_OFFLINE=1
  [ -n "${COSMOS_SIF:-}" ] || export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"   # the image has its own CUDA, do not use the host nvcc
  # shellcheck disable=SC2086
  $VLLM serve "$MODEL" ${HF_OVERRIDES:+--hf-overrides "$HF_OVERRIDES"} \
    --tensor-parallel-size "$TP" --mm-encoder-tp-mode data --async-scheduling \
    --gpu-memory-utilization "$MEM_UTIL" --max-model-len "$MAX_LEN" \
    --media-io-kwargs '{"video": {"num_frames": -1}}' $EXTRA_ARGS \
    --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 &
  echo $! > "runs/cosmos/vllm-$PORT.pid"
  echo "started vLLM inside the job ($MODEL, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}, port $PORT, pid $!), log $LOG; waiting for /health ..."
  wait_ready "kill -0 $! 2>/dev/null"; exit $?
fi

systemctl --user is-active -q "$UNIT" && { echo "$UNIT is already running"; exit 1; }

systemd-run --user --unit="$UNIT" --collect --expand-environment=no -p WorkingDirectory="$PWD" \
  -E CUDA_VISIBLE_DEVICES="$GPU" -E HF_HUB_CACHE="$HF_HUB_CACHE" -E HF_HUB_OFFLINE=1 -E HF_OVERRIDES="$HF_OVERRIDES" \
  -E CUDA_HOME="$CUDA_HOME" -E PATH="$CUDA_HOME/bin:/usr/bin:/bin" \
  /bin/bash -c "exec flock -n /tmp/nebius_gpu.lock $VENV/bin/vllm serve '$MODEL' \${HF_OVERRIDES:+--hf-overrides \"\$HF_OVERRIDES\"} \
    --tensor-parallel-size $TP --mm-encoder-tp-mode data --async-scheduling \
    --gpu-memory-utilization $MEM_UTIL --max-model-len $MAX_LEN \
    --media-io-kwargs '{\"video\": {\"num_frames\": -1}}' \
    --host 127.0.0.1 --port $PORT > '$LOG' 2>&1"

echo "started $UNIT ($MODEL, GPU $GPU, port $PORT), log $LOG; waiting for /health ..."
wait_ready "systemctl --user is-active -q '$UNIT'"
