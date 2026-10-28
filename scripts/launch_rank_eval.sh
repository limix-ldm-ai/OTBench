#!/usr/bin/env bash
# Scheduler entrypoint: one invocation per node, independent GPU workers.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# Activate the required Python environment before launching. Optional conda
# initialization requires both CONDA_SH and ENV_NAME to be supplied explicitly.
if [[ -n "${CONDA_SH:-}" ]]; then
  : "${ENV_NAME:?set ENV_NAME when CONDA_SH is supplied}"
  if [[ ! -f "${CONDA_SH}" ]]; then
    echo "ERROR: conda initialization script not found: ${CONDA_SH}" >&2
    exit 2
  fi
  source "${CONDA_SH}"
  conda activate "${ENV_NAME}"
fi

PYTHON="${PYTHON:-python}"
source "${SCRIPT_DIR}/_workspace.sh"
NNODES="${NNODES:?set the number of allocated nodes}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
# Keep all global-rank assignments but admit them in bounded local waves. The
# default 4 is used for ordinary methods. Memory-heavy methods are drained and
# run alone on the node instead of competing with another Python stack.
MAX_LOCAL_WORKERS="${MAX_LOCAL_WORKERS:-4}"
EXCLUSIVE_METHODS="${EXCLUSIVE_METHODS:-forest_vp forest_vp_joint}"
NODE_RANK="${NODE_RANK:-${RANK:-}}"
if [[ -z "${NODE_RANK}" ]]; then
  echo "ERROR: the scheduler must provide RANK=0..$((NNODES - 1)), or set NODE_RANK." >&2
  exit 2
fi
if ! [[ "${NNODES}" =~ ^[1-9][0-9]*$ && "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ \
        && "${MAX_LOCAL_WORKERS}" =~ ^[1-9][0-9]*$ \
        && "${NODE_RANK}" =~ ^[0-9]+$ ]] || (( NODE_RANK >= NNODES )); then
  echo "ERROR: invalid NNODES=${NNODES}, GPUS_PER_NODE=${GPUS_PER_NODE}, MAX_LOCAL_WORKERS=${MAX_LOCAL_WORKERS}, NODE_RANK=${NODE_RANK}" >&2
  exit 2
fi
if (( MAX_LOCAL_WORKERS > GPUS_PER_NODE )); then
  echo "ERROR: MAX_LOCAL_WORKERS=${MAX_LOCAL_WORKERS} exceeds GPUS_PER_NODE=${GPUS_PER_NODE}." >&2
  exit 2
fi

WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
PLAN_COUNT="$("${PYTHON}" "${SCRIPT_DIR}/protocol_catalog.py" rank-plan --count)"
if (( WORLD_SIZE < PLAN_COUNT )); then
  echo "ERROR: ${WORLD_SIZE} ranks cannot cover ${PLAN_COUNT} method-task assignments." >&2
  exit 2
fi
if [[ " ${EXTRA:-} " == *" --force-retrain "* ]]; then
  echo "ERROR: --force-retrain is forbidden in a shared multi-rank run." >&2
  exit 2
fi

visible_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
if [[ "${visible_gpus}" =~ ^[0-9]+$ ]] && (( visible_gpus < GPUS_PER_NODE )); then
  echo "ERROR: requested ${GPUS_PER_NODE} local GPUs, but nvidia-smi reports ${visible_gpus}." >&2
  exit 2
fi

RUN_ID="${RUN_ID:?set one shared, unique RUN_ID for this launch}"
SCHEDULER_ROOT="$(workspace_path "${SCHEDULER_ROOT:-${WORKSPACE_ROOT}/outputs/rank_scheduler}")"
RUN_ROOT="${SCHEDULER_ROOT}/${RUN_ID}"
"${PYTHON}" "${SCRIPT_DIR}/scheduler_state.py" freeze-rank \
  --run-root "${RUN_ROOT}" --run-id "${RUN_ID}" --world-size "${WORLD_SIZE}" >/dev/null
LOG_DIR="${RUN_ROOT}/scheduler_logs"
mkdir -p "${LOG_DIR}"

echo ">>> run id       : ${RUN_ID}"
echo ">>> node          : $((NODE_RANK + 1))/${NNODES} (node_rank=${NODE_RANK})"
echo ">>> global ranks  : $((NODE_RANK * GPUS_PER_NODE))-$((NODE_RANK * GPUS_PER_NODE + GPUS_PER_NODE - 1))/${WORLD_SIZE}"
echo ">>> assignments   : ${PLAN_COUNT} (remaining ranks are idle)"
echo ">>> local workers : ${MAX_LOCAL_WORKERS}/${GPUS_PER_NODE} (bounded waves)"
echo ">>> exclusive     : ${EXCLUSIVE_METHODS}"
echo ">>> scheduler root: ${RUN_ROOT}"

pids=()
ranks=()
failed=0

wait_wave() {
  local i
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "[FAIL] rank=${ranks[$i]} (see scheduler log)" >&2
      failed=$((failed + 1))
    fi
  done
  pids=()
  ranks=()
}

is_exclusive_method() {
  local wanted="$1" candidate
  for candidate in ${EXCLUSIVE_METHODS}; do
    [[ "${candidate}" == "${wanted}" ]] && return 0
  done
  return 1
}

for ((local_rank = 0; local_rank < GPUS_PER_NODE; local_rank++)); do
  global_rank=$((NODE_RANK * GPUS_PER_NODE + local_rank))
  log="${LOG_DIR}/rank_$(printf '%02d' "${global_rank}").log"
  rank_method=""
  if (( global_rank < PLAN_COUNT )); then
    rank_assignment="$("${PYTHON}" "${SCRIPT_DIR}/scheduler_state.py" rank-assignment --plan "${RUN_ROOT}/plan.json" --rank "${global_rank}")"
    IFS=$'\t' read -r rank_method _ <<< "${rank_assignment}"
  fi
  if is_exclusive_method "${rank_method}" && (( ${#pids[@]} > 0 )); then
    echo "[drain] before exclusive rank=${global_rank} method=${rank_method}"
    wait_wave
  fi
  env RANK="${global_rank}" LOCAL_RANK="${local_rank}" WORLD_SIZE="${WORLD_SIZE}" \
      LOCAL_WORLD_SIZE="${GPUS_PER_NODE}" RUN_ID="${RUN_ID}" \
      SCHEDULER_ROOT="${SCHEDULER_ROOT}" PYTHON="${PYTHON}" \
      SCHEDULER_PLAN_ONLY="${SCHEDULER_PLAN_ONLY:-0}" \
      bash "${SCRIPT_DIR}/run_rank_worker.sh" >"${log}" 2>&1 &
  pids+=("$!")
  ranks+=("${global_rank}")
  echo "[spawn] rank=${global_rank} gpu=${local_rank} log=${log}"
  if is_exclusive_method "${rank_method}"; then
    echo "[exclusive] rank=${global_rank} method=${rank_method}"
    wait_wave
  elif (( ${#pids[@]} >= MAX_LOCAL_WORKERS )); then
    wait_wave
  fi
done
if (( ${#pids[@]} > 0 )); then
  wait_wave
fi
if (( failed > 0 )); then
  echo ">>> node ${NODE_RANK}: ${failed}/${GPUS_PER_NODE} rank worker(s) failed." >&2
  echo ">>> failures are isolated; other ranks/nodes continue. Inspect with rank_status.py --strict." >&2
  # A nonzero node entrypoint can make a gang scheduler terminate healthy
  # nodes.  Status files, not this launcher exit code, are the run verdict.
  exit 0
fi
echo ">>> node ${NODE_RANK}: all ${GPUS_PER_NODE} rank workers completed."
