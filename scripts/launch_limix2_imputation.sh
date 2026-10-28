#!/usr/bin/env bash
# Scheduler entrypoint for four-node, multi-GPU LimiX2 imputation.
# The scheduler runs this same command once on every allocated node and must
# provide RANK (node rank), MASTER_ADDR, and MASTER_PORT.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

NNODES="${NNODES:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
LAUNCH_NODE_RANK="${RANK:?scheduler must provide RANK=0..$((NNODES - 1))}"
MASTER_ADDR="${MASTER_ADDR:?scheduler must provide MASTER_ADDR}"
MASTER_PORT="${MASTER_PORT:?scheduler must provide MASTER_PORT}"
RUN_ID="${RUN_ID:?set one shared, unique RUN_ID for this launch}"
CONTROL_STALE_MINUTES="${CONTROL_STALE_MINUTES:-5}"

# Use the active environment unless conda initialization is explicitly requested.
if [[ -n "${CONDA_SH:-}" ]]; then
  : "${ENV_NAME:?set ENV_NAME when CONDA_SH is supplied}"
  if [[ ! -f "${CONDA_SH}" ]]; then
    echo "ERROR: conda initialization script not found: ${CONDA_SH}" >&2
    exit 2
  fi
  source "${CONDA_SH}"
  conda activate "${ENV_NAME}"
fi
source "${SCRIPT_DIR}/_workspace.sh"
CONTROL_BASE="$(workspace_path "${CONTROL_ROOT:-${WORKSPACE_ROOT}/outputs/multinode/imputation/limix2}")"

if ! [[ "${NNODES}" =~ ^[1-9][0-9]*$ && "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ \
        && "${LAUNCH_NODE_RANK}" =~ ^[0-9]+$ && "${MASTER_PORT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: invalid topology/rendezvous values." >&2
  exit 2
fi
if (( LAUNCH_NODE_RANK >= NNODES || MASTER_PORT > 65535 )); then
  echo "ERROR: node rank or master port is outside its legal range." >&2
  exit 2
fi
if (( NNODES > 1 )) && [[ "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" ]]; then
  echo "ERROR: multi-node MASTER_ADDR must not be loopback." >&2
  exit 2
fi
if ! [[ "${RUN_ID}" =~ ^[A-Za-z0-9_.-]+$ ]] || [[ "${RUN_ID}" == "." || "${RUN_ID}" == ".." ]]; then
  echo "ERROR: RUN_ID may contain only letters, digits, dot, underscore, and dash." >&2
  exit 2
fi
if ! [[ "${CONTROL_STALE_MINUTES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: CONTROL_STALE_MINUTES must be a positive integer." >&2
  exit 2
fi

# A reused RUN_ID must start with empty status/done directories: otherwise old
# completion files can satisfy the file barrier before the new workers finish.
# Rank 0 performs the move before joining torchrun rendezvous, so no worker on
# any node can start until the shared control directory is ready.  Completed or
# stale attempts are preserved under _previous; a fresh incomplete attempt is
# treated as active and is never moved out from under another job.
archive_previous_control_run() {
  RUN_CONTROL_DIR="${CONTROL_BASE}/${RUN_ID}"
  if [[ -e "${RUN_CONTROL_DIR}" ]]; then
    if [[ ! -d "${RUN_CONTROL_DIR}" ]]; then
      echo "ERROR: control path exists but is not a directory: ${RUN_CONTROL_DIR}" >&2
      exit 2
    fi
    if [[ ! -f "${RUN_CONTROL_DIR}/SUMMARY.json" ]]; then
      RECENT_CONTROL_FILE="$(
        find "${RUN_CONTROL_DIR}" -maxdepth 2 -type f \
          -mmin "-${CONTROL_STALE_MINUTES}" -print -quit 2>/dev/null
      )"
      if [[ -z "${RECENT_CONTROL_FILE}" ]]; then
        RECENT_CONTROL_FILE="$(
          find "${RUN_CONTROL_DIR}" -maxdepth 0 \
            -mmin "-${CONTROL_STALE_MINUTES}" -print -quit 2>/dev/null
        )"
      fi
      if [[ -n "${RECENT_CONTROL_FILE}" ]]; then
        echo "ERROR: existing control directory has no SUMMARY.json and was updated" >&2
        echo "       within ${CONTROL_STALE_MINUTES} minute(s); refusing to disrupt a possibly active run:" >&2
        echo "       ${RUN_CONTROL_DIR}" >&2
        exit 2
      fi
    fi
    ARCHIVE_ROOT="${CONTROL_BASE}/_previous"
    ARCHIVE_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
    ARCHIVE_DIR="${ARCHIVE_ROOT}/${RUN_ID}.${ARCHIVE_STAMP}.pid$$"
    mkdir -p "${ARCHIVE_ROOT}"
    mv -- "${RUN_CONTROL_DIR}" "${ARCHIVE_DIR}"
    echo ">>> previous control run archived: ${ARCHIVE_DIR}"
  fi
}

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" \
      && "${CUDA_VISIBLE_DEVICES}" != "-1" ]]; then
  IFS=',' read -r -a VISIBLE_GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
  VISIBLE_GPU_COUNT="${#VISIBLE_GPU_LIST[@]}"
else
  VISIBLE_GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
fi
if ! [[ "${VISIBLE_GPU_COUNT}" =~ ^[0-9]+$ ]] || (( VISIBLE_GPU_COUNT < NPROC_PER_NODE )); then
  echo "ERROR: requested ${NPROC_PER_NODE} workers/node, but only ${VISIBLE_GPU_COUNT:-0} GPU(s) are visible." >&2
  exit 2
fi

TARGET_ACCESS="$("${PYTHON}" -c 'import sys,tomllib; print(tomllib.load(open(sys.argv[1],"rb"))["impute"]["target_access"])' "${PROJECT_ROOT}/methods/limix2/config.toml")"
if [[ "${TARGET_ACCESS}" != "target_observed" ]]; then
  echo "ERROR: LimiX2 imputation config is '${TARGET_ACCESS}', expected 'target_observed'." >&2
  exit 2
fi

read -r -a SEED_ARGS <<< "${SEEDS:-0 1 2}"
read -r -a SCENARIO_ARGS <<< "${SCENARIOS:-MCAR MAR MNAR MIXED_MCAR_MAR_MNAR}"
read -r -a RATE_ARGS <<< "${RATES:-0.1 0.3 0.5}"

WORKER_ARGS=(
  "${SCRIPT_DIR}/run_limix2_imputation_multinode.py"
  --run-id "${RUN_ID}"
  --dataset-type "${DATASET_TYPE:-all}"
  --workspace "${WORKSPACE_ROOT}"
  --control-root "${CONTROL_BASE}"
  --seeds "${SEED_ARGS[@]}"
  --scenarios "${SCENARIO_ARGS[@]}"
  --rates "${RATE_ARGS[@]}"
)
if [[ -n "${STORE:-}" ]]; then
  WORKER_ARGS+=(--store "${STORE}")
fi
if [[ -n "${DATASETS:-}" ]]; then
  read -r -a DATASET_ARGS <<< "${DATASETS}"
  WORKER_ARGS+=(--datasets "${DATASET_ARGS[@]}")
fi
if [[ "${SKIP_EXISTING:-1}" == "0" ]]; then
  WORKER_ARGS+=(--no-skip-existing)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  WORKER_ARGS+=(--dry-run)
fi
if [[ "${PLAN_ONLY:-0}" == "1" ]]; then
  WORKER_ARGS+=(--plan-only)
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-1}"

# Run every non-mutating preflight first. Other node agents may already be
# waiting at rendezvous, but workers cannot start until rank 0 joins it.
if (( LAUNCH_NODE_RANK == 0 )); then
  archive_previous_control_run
fi

echo ">>> run_id          : ${RUN_ID}"
echo ">>> node             : $((LAUNCH_NODE_RANK + 1))/${NNODES} (node_rank=${LAUNCH_NODE_RANK})"
echo ">>> topology         : ${NNODES} node(s) x ${NPROC_PER_NODE} GPU worker(s)"
echo ">>> rendezvous       : ${MASTER_ADDR}:${MASTER_PORT}"
echo ">>> environment      : ${ENV_NAME:-active environment}"
echo ">>> method/protocol  : limix2 / imputation / target_observed"
echo ">>> unit             : dataset x seed; each unit owns ${#SCENARIO_ARGS[@]} x ${#RATE_ARGS[@]} settings"
echo ">>> seeds            : ${SEED_ARGS[*]}"

# Capture scheduler RANK above: torchrun replaces RANK with each worker's global
# rank.  The explicit static rendezvous is exactly the contract supplied by the
# scheduler.
exec torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${LAUNCH_NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --max_restarts=0 \
  "${WORKER_ARGS[@]}"
