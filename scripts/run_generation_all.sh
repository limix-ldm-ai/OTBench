#!/usr/bin/env bash
# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
# Generate and evaluate each dataset under independent checkpoint seeds.
# Usage: [SEEDS="0 1 2" | N_TRIALS=3] bash scripts/run_generation_all.sh METHOD [DATASET ...]
# OTBENCH_WORKSPACE controls writable roots; STORE, OUTPUTS and CHECKPOINTS override them.
# OTBENCH_GPUS selects GPUs. METRIC, EXTRA, SKIP_EXISTING=1 and DRY_RUN=1 are optional.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
if (( $# < 1 )); then
  echo "usage: bash scripts/run_generation_all.sh METHOD [DATASET ...]" >&2
  exit 2
fi
METHOD="$1"; shift
source "${SCRIPT_DIR}/_batch_common.sh"

N_TRIALS="${N_TRIALS:-3}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
METRIC="${METRIC:-all}"
if ! [[ "${N_TRIALS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: N_TRIALS must be a positive integer." >&2
  exit 2
fi
TARGET_SEEDS=()
if [[ -n "${SEEDS:-}" ]]; then
  read -r -a TARGET_SEEDS <<< "${SEEDS}"
else
  for ((s = 0; s < N_TRIALS; s++)); do TARGET_SEEDS+=("${s}"); done
fi
validate_seeds "${TARGET_SEEDS[@]}"
validate_skip_existing

DATASET_TYPE="${DATASET_TYPE:-real}"
REQUESTED_DATASETS=()
if (( $# > 0 )); then
  REQUESTED_DATASETS=("$@")
elif [[ -n "${DATASETS:-}" ]]; then
  read -r -a REQUESTED_DATASETS <<< "${DATASETS}"
fi
resolve_batch_datasets generation "${METHOD}" "${DATASET_TYPE}" "${REQUESTED_DATASETS[@]}"
TASK_OUTPUTS_ROOT="$(workspace_path "${OUTPUTS:-${WORKSPACE_ROOT}/outputs/generation}")"
GEN_OUT="${TASK_OUTPUTS_ROOT}/${METHOD}"
BATCH_PATH_ARGS=(--workspace "${WORKSPACE_ROOT}" --outputs "${TASK_OUTPUTS_ROOT}" --checkpoints "${CHECKPOINTS}")

echo ">>> generation ${METHOD}: ${#DATASETS[@]} datasets; seeds ${TARGET_SEEDS[*]}; skip_existing=${SKIP_EXISTING}"
assert_supports generation "${METHOD}"
detect_gpus
prewarm_data_cache "${METHOD}" "${DATASETS[@]}"
for d in "${DATASETS[@]}"; do
  for s in "${TARGET_SEEDS[@]}"; do
    cmd=("${PYTHON}" -m otbench.run "${BATCH_PATH_ARGS[@]}" --task generation --method "${METHOD}"
         --dataset "${d}" --seed "${s}" --train-seed "${s}" --eval --metric "${METRIC}"
         --store "${STORE}")
    if [[ "${SKIP_EXISTING}" == "1" ]]; then cmd+=(--skip-existing); fi
    if [[ -n "${EXTRA:-}" ]]; then cmd+=(${EXTRA}); fi
    add_job "gen ${METHOD}/${d}/seed${s}" "${GEN_OUT}/logs/gen_${d}_seed${s}.log" "${cmd[@]}"
  done
done
run_jobs
