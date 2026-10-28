#!/usr/bin/env bash
# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
# Prediction over all real datasets, multi-GPU by default.
#
# One checkpoint per dataset (train_seed 0) is trained once, then used to
# predict the test target once per dataset (single trial). Two phases:
#   1) train the shared checkpoint per dataset (train-only; no-op for foundation);
#   2) predict + evaluate once per dataset, reusing the ckpt.
#
# Usage:
#   ./scripts/run_prediction_all.sh <method>
#   ./scripts/run_prediction_all.sh tabpfn wine vehicle
#   OTBENCH_GPUS="0 1 2 3" ./scripts/run_prediction_all.sh tabdiff
#
# Env vars:
#   OTBENCH_GPUS / DATASETS / DATASET_TYPE / STORE / EXTRA (see run_generation_all.sh)
#   TRAIN_SEED       shared checkpoint seed (default: 0)
#   SKIP_EXISTING=1  reuse only provenance- and artifact-validated results
#   DRY_RUN=1        print commands only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
PARENT_ROOT="$(dirname "${PROJECT_ROOT}")"

if [[ $# -lt 1 ]]; then
  echo "usage: [env vars] ./scripts/run_prediction_all.sh <method> [dataset ...]" >&2
  exit 2
fi
METHOD="$1"; shift

source "${SCRIPT_DIR}/_batch_common.sh"

STORE_ROOT="${STORE}"
TRAIN_SEED="${TRAIN_SEED:-0}"

DATASET_TYPE="${DATASET_TYPE:-real}"
REQUESTED_DATASETS=()
if [[ $# -gt 0 ]]; then
  REQUESTED_DATASETS=("$@")
elif [[ -n "${DATASETS:-}" ]]; then
  read -r -a REQUESTED_DATASETS <<< "${DATASETS}"
fi
resolve_batch_datasets prediction "${METHOD}" "${DATASET_TYPE}" "${REQUESTED_DATASETS[@]}"

TASK_OUTPUTS_ROOT="$(workspace_path "${OUTPUTS:-${WORKSPACE_ROOT}/outputs/prediction}")"
PRED_OUT="${TASK_OUTPUTS_ROOT}/${METHOD}"
BATCH_PATH_ARGS=(--workspace "${WORKSPACE_ROOT}" --outputs "${TASK_OUTPUTS_ROOT}" --checkpoints "${CHECKPOINTS}")

echo ">>> task      : prediction"
echo ">>> method    : ${METHOD}"
echo ">>> datasets  : ${#DATASETS[@]} (${DATASETS[*]})"
echo ">>> train seed: ${TRAIN_SEED}"

assert_supports prediction "${METHOD}"
detect_gpus
prewarm_data_cache "${METHOD}" "${DATASETS[@]}"

# --- phase 1: one shared checkpoint per dataset (train-only) ---
echo
echo "=== phase 1: train shared checkpoints (train_seed=${TRAIN_SEED}) ==="
for d in "${DATASETS[@]}"; do
  cmd=("${PYTHON}" -m otbench.run "${BATCH_PATH_ARGS[@]}" --task prediction --method "${METHOD}"
       --dataset "${d}" --seed "${TRAIN_SEED}" --train-seed "${TRAIN_SEED}" --train-only)
  if [[ -n "${STORE}" ]]; then cmd+=(--store "${STORE}"); fi
  if [[ -n "${METRIC:-}" ]]; then cmd+=(--metric "${METRIC}"); fi
  if [[ -n "${EXTRA:-}" ]]; then cmd+=(${EXTRA}); fi
  add_job "train ${METHOD}/${d}" "${PRED_OUT}/logs/train_${d}_train_seed${TRAIN_SEED}.log" "${cmd[@]}"
done
phase1_rc=0
run_jobs || phase1_rc=$?
declare -A TRAIN_UNAVAILABLE=()
for i in "${!DATASETS[@]}"; do
  train_status="$(cat "${JOB_DIR}/status_${i}" 2>/dev/null)"
  if [[ "${train_status}" != "OK" ]]; then
    TRAIN_UNAVAILABLE["${DATASETS[$i]}"]="${train_status:-FAIL}"
  fi
done
if (( phase1_rc != 0 )); then
  echo ">>> WARNING: some checkpoints failed to train; dependent predictions will not be retried."
fi

# --- phase 2: predict + eval once per dataset ---
echo
echo "=== phase 2: predict + eval (single trial per dataset) ==="
reset_jobs
for d in "${DATASETS[@]}"; do
  if [[ -n "${TRAIN_UNAVAILABLE[$d]:-}" ]]; then
    echo "[skip] ${d}: training status=${TRAIN_UNAVAILABLE[$d]} (prediction suppressed)"
    continue
  fi
  cmd=("${PYTHON}" -m otbench.run "${BATCH_PATH_ARGS[@]}" --task prediction --method "${METHOD}"
       --dataset "${d}" --seed "${TRAIN_SEED}" --train-seed "${TRAIN_SEED}" --trials 1)
  if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then cmd+=(--skip-existing); fi
  if [[ -n "${STORE}" ]]; then cmd+=(--store "${STORE}"); fi
  if [[ -n "${METRIC:-}" ]]; then cmd+=(--metric "${METRIC}"); fi
  if [[ -n "${EXTRA:-}" ]]; then cmd+=(${EXTRA}); fi
  add_job "pred ${METHOD}/${d}" "${PRED_OUT}/logs/predict_${d}_seed${TRAIN_SEED}.log" "${cmd[@]}"
done

phase2_rc=0
run_jobs || phase2_rc=$?
if (( phase1_rc != 0 || phase2_rc != 0 )); then
  exit 1
fi
