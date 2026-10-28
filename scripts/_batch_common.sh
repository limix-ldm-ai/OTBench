#!/usr/bin/env bash
# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
# Shared helpers for the per-task batch scripts
# (run_generation_all.sh / run_imputation_all.sh / run_prediction_all.sh).
#
# Sourced, NOT executed. The caller must set PROJECT_ROOT first, and the target
# Python environment (e.g. `conda activate benchmark`) must already be active —
# jobs use PYTHON (default: python), with no conda wrapper.
#
# Job queue API:
#   add_job "<label>" "<logfile>" python -m otbench.run ...   # enqueue one run
#   run_jobs                    # execute the queue, static-sharded across GPUs
#   reset_jobs                  # clear the queue (between train/inference phases)
# GPU detection:
#   detect_gpus                 # populate GPUS[] from OTBENCH_GPUS / CUDA_VISIBLE_DEVICES / nvidia-smi
# Misc:
#   assert_supports <task> <method>                # fail fast if unsupported
#   resolve_batch_datasets <task> <method> <type> [names...]  # populate DATASETS[]
#   prewarm_data_cache <method> <datasets...>      # serial vendor-data export for diffusion methods
#   dataset_type <name>                            # echo scm|real (uses STORE_ROOT)

DIFFUSION_METHODS="tabddpm tabsyn tabdiff"
SCRIPT_DIR="${SCRIPT_DIR:-${PROJECT_ROOT}/scripts}"
source "${SCRIPT_DIR}/_workspace.sh"
STORE="$(workspace_path "${STORE:-${WORKSPACE_ROOT}/tabdata/store}")"
STORE_ROOT="${STORE}"
CHECKPOINTS="$(workspace_path "${CHECKPOINTS:-${WORKSPACE_ROOT}/checkpoints}")"
JOB_TMP_ROOT="$(workspace_path "${TMPDIR:-${WORKSPACE_ROOT}/outputs/.tmp}")"
mkdir -p "${JOB_TMP_ROOT}"

JOB_DIR="$(mktemp -d "${JOB_TMP_ROOT}/OTBench_jobs_XXXXXX")"
PACKAGE_SHIM_DIR="$(mktemp -d "${JOB_TMP_ROOT}/OTBench_package_XXXXXX")"
ln -s "${PROJECT_ROOT}" "${PACKAGE_SHIM_DIR}/otbench"
# Import this checkout even when its directory has a suffix and a stale sibling
# named otbench exists. Isolate imports from unrelated PYTHONPATH entries.
PARENT_ROOT="${PACKAGE_SHIM_DIR}"
export PYTHONPATH="${PACKAGE_SHIM_DIR}"
export SCIPY_ARRAY_API=1
# Bound nested numerical-library threads when preprocessing runs in parallel.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
JOB_LABELS=()
JOB_LOGS=()
JOB_N=0

_cleanup_jobs() {
  rm -rf "${JOB_DIR}" "${PACKAGE_SHIM_DIR}" 2>/dev/null || true
}
trap _cleanup_jobs EXIT

validate_seeds() {
  if (( $# == 0 )); then
    echo "ERROR: at least one seed is required." >&2
    exit 2
  fi
  local seed
  local -A seen=()
  for seed in "$@"; do
    if ! [[ "${seed}" =~ ^(0|[1-9][0-9]*)$ ]]; then
      echo "ERROR: seeds must be non-negative integers; got '${seed}'." >&2
      exit 2
    fi
    if [[ -n "${seen[$seed]+set}" ]]; then
      echo "ERROR: duplicate seed '${seed}'." >&2
      exit 2
    fi
    seen["${seed}"]=1
  done
}

validate_skip_existing() {
  if [[ "${SKIP_EXISTING}" != "0" && "${SKIP_EXISTING}" != "1" ]]; then
    echo "ERROR: SKIP_EXISTING must be 0 or 1." >&2
    exit 2
  fi
}

detect_gpus() {
  local src="${OTBENCH_GPUS:-}"
  if [[ -z "${src}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    case "${CUDA_VISIBLE_DEVICES}" in
      -1|NoDevFiles|"") ;;
      *) src="${CUDA_VISIBLE_DEVICES}" ;;
    esac
  fi
  if [[ -z "${src}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    src="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd' ' -)"
  fi
  src="${src:-0}"
  src="${src//,/ }"
  read -r -a GPUS <<< "${src}"
  if [[ "${#GPUS[@]}" -eq 0 ]]; then
    echo "ERROR: no GPU ids resolved; set OTBENCH_GPUS=\"0 1 ...\"." >&2
    exit 2
  fi
}

assert_supports() {
  # Also acts as an environment preflight: if this fails, every job would fail
  # too, so surface the REAL underlying error (conda env missing, package not
  # importable, ...) instead of silently reporting "unsupported".
  local task="$1" method="$2" out rc
  # NOTE: capture rc via `|| rc=$?`; a bare `out="$(failing cmd)"` under
  # `set -e` aborts the whole script at the assignment (silent exit, no logs).
  rc=0
  out="$(cd "${PARENT_ROOT}" && "${PYTHON}" -c '
import sys
from otbench.methods.common.registry import method_tasks
print("SUPPORT=" + ("yes" if sys.argv[1] in {t.value for t in method_tasks(sys.argv[2])} else "no"))
' "${task}" "${method}" 2>&1)" || rc=$?
  if [[ ${rc} -ne 0 || "${out}" != *"SUPPORT="* ]]; then
    echo "ERROR: preflight failed (running 'python' from '${PARENT_ROOT}')." >&2
    echo "       Activate the environment first (e.g. 'conda activate benchmark')" >&2
    echo "       and make sure 'import otbench' works there." >&2
    echo "------------------------- underlying error -------------------------" >&2
    echo "${out}" >&2
    echo "--------------------------------------------------------------------" >&2
    exit 2
  fi
  if [[ "${out##*SUPPORT=}" != "yes" ]]; then
    echo "ERROR: method '${method}' does not support task '${task}'." >&2
    exit 2
  fi
}

resolve_batch_datasets() {
  # Registry + schema are the protocol source of truth. With explicit names,
  # reject one invalid combination as a caller error; without names, discover
  # the materialized store and keep only datasets the method can run on.
  local task="$1" method="$2" dtype="$3"; shift 3
  local cmd=("${PYTHON}" "${SCRIPT_DIR}/protocol_catalog.py" datasets
             --task "${task}" --method "${method}"
             --store-root "${STORE_ROOT}" --dataset-type "${dtype}")
  if [[ $# -gt 0 ]]; then cmd+=(-- "$@"); fi
  local output rc=0
  output="$("${cmd[@]}" 2>&1)" || rc=$?
  if [[ ${rc} -ne 0 ]]; then
    echo "${output}" >&2
    exit 2
  fi
  mapfile -t DATASETS <<< "${output}"
}

dataset_type() {
  local name="$1"
  if [[ -d "${STORE_ROOT}/scm/${name}" ]]; then
    printf 'scm'
  elif [[ -d "${STORE_ROOT}/real/${name}" ]]; then
    printf 'real'
  else
    echo "ERROR: dataset '${name}' is absent from '${STORE_ROOT}'." >&2
    return 2
  fi
}

prewarm_data_cache() {
  # Diffusion methods export a shared vendor-data cache under
  # checkpoints/_data_cache/<dataset>/. If several parallel jobs of the same
  # dataset first-touch it at once they race on the write, so build it serially
  # up front; parallel jobs then only read it. No-op for non-diffusion methods.
  local method="$1"; shift
  case " ${DIFFUSION_METHODS} " in *" ${method} "*) ;; *) return 0 ;; esac
  echo ">>> prewarming vendor-data cache for '${method}' on $# dataset(s) ..."
  local d
  for d in "$@"; do
    if (cd "${PARENT_ROOT}" && "${PYTHON}" -c '
import sys
from otbench.tabdata import load_dataset
from otbench.methods.common.vendor_data import ensure_vendor_data
ensure_vendor_data(load_dataset(sys.argv[1], sys.argv[2]), root=sys.argv[3])
' "${d}" "${STORE_ROOT}" "${CHECKPOINTS}" >/dev/null 2>&1); then
      echo "    cached ${d}"
    else
      echo "    [warn] prewarm failed for ${d} (job will retry inline)"
    fi
  done
}

add_job() {
  local label="$1" log="$2"; shift 2
  JOB_LABELS+=("${label}")
  JOB_LOGS+=("${log}")
  printf '%s\n' "$@" > "${JOB_DIR}/job_${JOB_N}.args"
  JOB_N=$((JOB_N + 1))
}

reset_jobs() {
  JOB_LABELS=()
  JOB_LOGS=()
  JOB_N=0
  rm -f "${JOB_DIR}"/job_*.args "${JOB_DIR}"/status_* 2>/dev/null || true
}

run_one() {
  local i="$1" gpu="$2"
  local log="${JOB_LOGS[$i]}" label="${JOB_LABELS[$i]}"
  local cmd
  mkdir -p "$(dirname "${log}")"
  mapfile -t cmd < "${JOB_DIR}/job_${i}.args"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry] gpu=${gpu} ${label} :: CUDA_VISIBLE_DEVICES=${gpu} ${cmd[*]}"
    echo "OK" > "${JOB_DIR}/status_${i}"
    return 0
  fi
  echo "[run] gpu=${gpu} ${label} -> ${log}"
  local rc=0
  (cd "${PARENT_ROOT}" && CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}") > "${log}" 2>&1 || rc=$?
  if [[ ${rc} -eq 0 ]]; then
    echo "RUN_OK" >> "${log}"
    echo "OK" > "${JOB_DIR}/status_${i}"
    echo "[done] ${label}"
  elif [[ ${rc} -eq 3 ]]; then
    # otbench.run reserves exit 3 for a scientifically unsupported
    # method/dataset combination. Keep it visible without calling it a crash.
    echo "RUN_SKIPPED" >> "${log}"
    echo "SKIP" > "${JOB_DIR}/status_${i}"
    echo "[skip] ${label} (see ${log})"
  else
    echo "FAIL" > "${JOB_DIR}/status_${i}"
    echo "[FAIL] ${label} (see ${log})"
  fi
  return 0
}

_gpu_worker() {
  local slot="$1"
  local gpu="${GPUS[$slot]}"
  local i
  for ((i = slot; i < JOB_N; i += N_GPUS)); do
    run_one "${i}" "${gpu}"
  done
}

run_jobs() {
  N_GPUS="${#GPUS[@]}"
  if [[ "${JOB_N}" -eq 0 ]]; then
    echo ">>> no jobs to run (nothing pending / all skipped)."
    return 0
  fi
  echo ">>> dispatching ${JOB_N} job(s) across ${N_GPUS} GPU(s): ${GPUS[*]}"
  local pids=() g
  for ((g = 0; g < N_GPUS; g++)); do
    _gpu_worker "${g}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}"; done

  local failed=() i status skipped=0
  for ((i = 0; i < JOB_N; i++)); do
    status="$(cat "${JOB_DIR}/status_${i}" 2>/dev/null)"
    if [[ "${status}" == "SKIP" ]]; then
      skipped=$((skipped + 1))
    elif [[ "${status}" != "OK" ]]; then
      failed+=("${JOB_LABELS[$i]}")
    fi
  done
  if [[ "${#failed[@]}" -gt 0 ]]; then
    echo ">>> FAILURES (${#failed[@]}/${JOB_N}):"
    printf '      %s\n' "${failed[@]}"
    return 1
  fi
  echo ">>> completed ${JOB_N} job(s): ok=$((JOB_N - skipped)) skipped=${skipped}."
  return 0
}
