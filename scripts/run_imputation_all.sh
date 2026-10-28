#!/usr/bin/env bash
# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
# Train one checkpoint per dataset/seed, then complete its missingness settings.
# Usage: [SEEDS="0 1 2"] bash scripts/run_imputation_all.sh METHOD [DATASET ...]
# SCENARIOS and RATES define the grid. TRIALS is the number of completion draws.
# Native conditional methods share fitted state across each dataset/seed grid.
# Workspace, GPU, metric and cache options match run_generation_all.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
if (( $# < 1 )); then
  echo "usage: bash scripts/run_imputation_all.sh METHOD [DATASET ...]" >&2
  exit 2
fi
METHOD="$1"; shift
source "${SCRIPT_DIR}/_batch_common.sh"

SKIP_EXISTING="${SKIP_EXISTING:-1}"
read -r -a TARGET_SEEDS <<< "${SEEDS:-0 1 2}"
read -r -a SCENARIOS <<< "${SCENARIOS:-MCAR MAR MNAR MIXED_MCAR_MAR_MNAR}"
read -r -a RATES <<< "${RATES:-0.1 0.3 0.5}"
validate_seeds "${TARGET_SEEDS[@]}"
validate_skip_existing

DATASET_TYPE="${DATASET_TYPE:-real}"
REQUESTED_DATASETS=()
if (( $# > 0 )); then
  REQUESTED_DATASETS=("$@")
elif [[ -n "${DATASETS:-}" ]]; then
  read -r -a REQUESTED_DATASETS <<< "${DATASETS}"
fi
resolve_batch_datasets imputation "${METHOD}" "${DATASET_TYPE}" "${REQUESTED_DATASETS[@]}"
TASK_OUTPUTS_ROOT="$(workspace_path "${OUTPUTS:-${WORKSPACE_ROOT}/outputs/imputation}")"
IMP_OUT="${TASK_OUTPUTS_ROOT}/${METHOD}"
BATCH_PATH_ARGS=(--workspace "${WORKSPACE_ROOT}" --outputs "${TASK_OUTPUTS_ROOT}" --checkpoints "${CHECKPOINTS}")
assert_supports imputation "${METHOD}"
HAS_DRAWS="$("${PYTHON}" -c 'import sys; from otbench.methods.common.registry import get_spec; print(get_spec(sys.argv[1]).capabilities.can_emit("draws", "imputation"))' "${METHOD}")"
if [[ "${HAS_DRAWS}" == "True" ]]; then TRIALS="${TRIALS:-3}"; else TRIALS="${TRIALS:-1}"; fi
if ! [[ "${TRIALS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: TRIALS must be a positive integer." >&2
  exit 2
fi
N_SETTINGS=$(( ${#SCENARIOS[@]} * ${#RATES[@]} ))
echo ">>> imputation ${METHOD}: ${#DATASETS[@]} datasets; seeds ${TARGET_SEEDS[*]}; ${N_SETTINGS} settings; K=${TRIALS}"
detect_gpus
prewarm_data_cache "${METHOD}" "${DATASETS[@]}"

TRAIN_JOB_KEYS=()
for d in "${DATASETS[@]}"; do
  for s in "${TARGET_SEEDS[@]}"; do
    cmd=("${PYTHON}" -m otbench.run "${BATCH_PATH_ARGS[@]}" --task imputation --method "${METHOD}"
         --dataset "${d}" --seed "${s}" --train-seed "${s}" --train-only --store "${STORE}")
    if [[ -n "${METRIC:-}" ]]; then cmd+=(--metric "${METRIC}"); fi
    if [[ -n "${EXTRA:-}" ]]; then cmd+=(${EXTRA}); fi
    add_job "train ${METHOD}/${d}/seed${s}" "${IMP_OUT}/logs/train_${d}_train_seed${s}.log" "${cmd[@]}"
    TRAIN_JOB_KEYS+=("${d}"$'\x1f'"${s}")
  done
done
phase1_rc=0
run_jobs || phase1_rc=$?
declare -A TRAIN_UNAVAILABLE=()
for job_i in "${!TRAIN_JOB_KEYS[@]}"; do
  train_status="$(cat "${JOB_DIR}/status_${job_i}" 2>/dev/null)"
  if [[ "${train_status}" != "OK" ]]; then
    TRAIN_UNAVAILABLE["${TRAIN_JOB_KEYS[$job_i]}"]="${train_status:-FAIL}"
  fi
done

reset_jobs
for d in "${DATASETS[@]}"; do
  for s in "${TARGET_SEEDS[@]}"; do
    train_key="${d}"$'\x1f'"${s}"
    if [[ -n "${TRAIN_UNAVAILABLE[$train_key]:-}" ]]; then
      echo "[skip] ${d}/seed${s}: training status=${TRAIN_UNAVAILABLE[$train_key]}"
      continue
    fi
    if [[ "${METHOD}" == "tabpfn_unsupervised" || "${METHOD}" == "tabiclv2" || "${METHOD}" == "limix2" ]]; then
      cmd=("${PYTHON}" -m otbench.imputation.grid_cli "${BATCH_PATH_ARGS[@]}"
           --method "${METHOD}" --dataset "${d}" --scenarios "${SCENARIOS[@]}" --rates "${RATES[@]}"
           --seed "${s}" --train-seed "${s}" --trials "${TRIALS}" --store "${STORE}"
           --scientific-model auto --stability-protocol auto)
      if [[ "${SKIP_EXISTING}" == "1" ]]; then cmd+=(--skip-existing); fi
      if [[ -n "${METRIC:-}" ]]; then cmd+=(--metric "${METRIC}"); fi
      if [[ -n "${EXTRA:-}" ]]; then cmd+=(${EXTRA}); fi
      add_job "imp-grid ${METHOD}/${d}/seed${s}" "${IMP_OUT}/logs/impute_${d}_seed${s}_setting_grid.log" "${cmd[@]}"
      continue
    fi
    for sc in "${SCENARIOS[@]}"; do
      for r in "${RATES[@]}"; do
        cmd=("${PYTHON}" -m otbench.run "${BATCH_PATH_ARGS[@]}" --task imputation --method "${METHOD}"
             --dataset "${d}" --scenario "${sc}" --rate "${r}" --seed "${s}" --train-seed "${s}"
             --selection-seed "${s}" --mask-seed "${s}" --impute-seed "${s}"
             --trials "${TRIALS}" --scientific-model auto --stability-protocol auto --store "${STORE}")
        if [[ "${SKIP_EXISTING}" == "1" ]]; then cmd+=(--skip-existing); fi
        if [[ -n "${METRIC:-}" ]]; then cmd+=(--metric "${METRIC}"); fi
        if [[ -n "${EXTRA:-}" ]]; then cmd+=(${EXTRA}); fi
        add_job "imp ${METHOD}/${d}/seed${s}/${sc}/rate${r}" "${IMP_OUT}/logs/impute_${d}_${sc}_rate${r/./p}_seed${s}.log" "${cmd[@]}"
      done
    done
  done
done
phase2_rc=0
run_jobs || phase2_rc=$?
if (( phase1_rc != 0 || phase2_rc != 0 )); then exit 1; fi
