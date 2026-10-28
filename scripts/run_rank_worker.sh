#!/usr/bin/env bash
# Execute exactly one registry-derived method/task assignment on one local GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
PYTHON="${PYTHON:-python}"
source "${SCRIPT_DIR}/_workspace.sh"
GLOBAL_RANK="${RANK:?RANK is required}"
LOCAL_GPU="${LOCAL_RANK:?LOCAL_RANK is required}"
GLOBAL_WORLD_SIZE="${WORLD_SIZE:?WORLD_SIZE is required}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
SCHEDULER_ROOT="$(workspace_path "${SCHEDULER_ROOT:-${WORKSPACE_ROOT}/outputs/rank_scheduler}")"
RUN_ROOT="${SCHEDULER_ROOT}/${RUN_ID}"
STATUS_DIR="${RUN_ROOT}/status"
if [[ ! -f "${RUN_ROOT}/plan.json" ]]; then
  "${PYTHON}" "${SCRIPT_DIR}/scheduler_state.py" freeze-rank \
    --run-root "${RUN_ROOT}" --run-id "${RUN_ID}" --world-size "${GLOBAL_WORLD_SIZE}" >/dev/null
fi
PLAN_COUNT="$("${PYTHON}" "${SCRIPT_DIR}/scheduler_state.py" rank-assignment --plan "${RUN_ROOT}/plan.json" \
  --run-id "${RUN_ID}" --world-size "${GLOBAL_WORLD_SIZE}" --count)"
METHOD=""
TASK=""

write_status() {
  local state="$1" exit_code="${2:-}" reason="${3:-}"
  local cmd=("${PYTHON}" "${SCRIPT_DIR}/scheduler_state.py" write-status \
    --plan "${RUN_ROOT}/plan.json" --slot "${GLOBAL_RANK}" \
    --local-rank "${LOCAL_GPU}" --state "${state}" --reason "${reason}")
  if [[ -n "${exit_code}" ]]; then cmd+=(--exit-code "${exit_code}"); fi
  "${cmd[@]}"
}

if (( GLOBAL_RANK >= PLAN_COUNT )); then
  write_status idle 0 "rank is outside the ${PLAN_COUNT}-assignment plan"
  echo "[idle] rank=${GLOBAL_RANK}: ${PLAN_COUNT} method-task assignments"
  exit 0
fi

assignment="$("${PYTHON}" "${SCRIPT_DIR}/scheduler_state.py" rank-assignment --plan "${RUN_ROOT}/plan.json" --rank "${GLOBAL_RANK}")"
IFS=$'\t' read -r METHOD TASK <<< "${assignment}"
if [[ -z "${METHOD}" || -z "${TASK}" ]]; then
  write_status failed 2 "catalog returned an invalid assignment"
  exit 2
fi

readiness="$("${PYTHON}" "${SCRIPT_DIR}/smoke_plan.py" --readiness "${METHOD}")"
if [[ "${readiness}" != "READY" ]]; then
  write_status skipped 0 "${readiness}"
  echo "[skip] rank=${GLOBAL_RANK} ${METHOD}/${TASK}: ${readiness}"
  exit 0
fi

if [[ "${SCHEDULER_PLAN_ONLY:-0}" == "1" ]]; then
  write_status planned 0 "plan-only validation"
  echo "[plan] rank=${GLOBAL_RANK} gpu=${LOCAL_GPU} ${METHOD}/${TASK}"
  exit 0
fi

write_status running

# RANK/WORLD_SIZE describe independent scheduler slots, not a distributed
# model process group.  Do not leak them into method backends (Accelerate,
# Lightning, and torch.distributed would otherwise try to rendezvous).
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK \
  ROLE_WORLD_SIZE NODE_RANK MASTER_ADDR MASTER_PORT TORCHELASTIC_RUN_ID \
  TORCHELASTIC_RESTART_COUNT TORCHELASTIC_MAX_RESTARTS

export OTBENCH_GPUS="${LOCAL_GPU}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-1}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"

case "${TASK}" in
  generation) driver="${SCRIPT_DIR}/run_generation_all.sh" ;;
  imputation) driver="${SCRIPT_DIR}/run_imputation_all.sh" ;;
  prediction) driver="${SCRIPT_DIR}/run_prediction_all.sh" ;;
  *)
    write_status failed 2 "unknown task: ${TASK}"
    exit 2
    ;;
esac

echo "[run] rank=${GLOBAL_RANK} gpu=${LOCAL_GPU} ${METHOD}/${TASK}"
set +e
bash "${driver}" "${METHOD}" &
driver_pid=$!
"${PYTHON}" "${SCRIPT_DIR}/scheduler_state.py" heartbeat \
  --plan "${RUN_ROOT}/plan.json" --slot "${GLOBAL_RANK}" --local-rank "${LOCAL_GPU}" \
  --pid "${driver_pid}" --interval "${STATUS_HEARTBEAT_SECONDS:-60}" &
heartbeat_pid=$!
wait "${driver_pid}"
rc=$?
kill "${heartbeat_pid}" 2>/dev/null || true
wait "${heartbeat_pid}" 2>/dev/null || true
set -e
if (( rc == 0 )); then
  write_status succeeded 0
else
  write_status failed "${rc}" "task driver exited nonzero"
fi
exit "${rc}"
